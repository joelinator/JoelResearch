#!/usr/bin/env python3
"""
Benchmark script for the Nine-Species dataset (InstaDeepAI/ms_ninespecies_benchmark).

Evaluates the pre-trained DFM model across the nine individual species (.ipc files)
and/or the official held-out test split (.parquet).

Features:
- Direct Hugging Face Hub download & caching per species
- Zero-copy PyArrow memory-mapping (minimal RAM footprint)
- Automatic sequence normalization (strips flanking dots, canonicalizes PTM annotations)
- Precursor mass correction (deriving neutral mass from m/z and charge)
- High-throughput batched inference on CUDA (default batch size 4000)
- Detailed per-species and aggregated metrics table (Exact, I/L, Mass, Length, AA F1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.constants import M_H
from data.data import (
    SpectrumDataSet,
    build_vocabulary,
    parse_peptide,
    spectrum_collate,
)
from eval.evaluate import evaluate_generative
from flow_matching.scheduler import cosine_scheduler
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

REPO_ID = "InstaDeepAI/ms_ninespecies_benchmark"

SPECIES_MAP = {
    "yeast": "saccharomyces_cerevisiae.ipc",
    "saccharomyces_cerevisiae": "saccharomyces_cerevisiae.ipc",
    "mouse": "mus_musculus.ipc",
    "mus_musculus": "mus_musculus.ipc",
    "human": "h_sapiens.ipc",
    "h_sapiens": "h_sapiens.ipc",
    "candidatus": "candidatus_endoloripes.ipc",
    "candidatus_endoloripes": "candidatus_endoloripes.ipc",
    "tomato": "solanum_lycopersicum.ipc",
    "solanum_lycopersicum": "solanum_lycopersicum.ipc",
    "vigna_mungo": "vigna_mungo.ipc",
    "mungo": "vigna_mungo.ipc",
    "honeybee": "apis_mellifera.ipc",
    "apis_mellifera": "apis_mellifera.ipc",
    "methanosarcina": "methanosarcina_mazei.ipc",
    "methanosarcina_mazei": "methanosarcina_mazei.ipc",
    "bacillus": "bacillus_subtilis.ipc",
    "bacillus_subtilis": "bacillus_subtilis.ipc",
}

ALL_SPECIES = [
    "saccharomyces_cerevisiae",
    "mus_musculus",
    "h_sapiens",
    "candidatus_endoloripes",
    "solanum_lycopersicum",
    "vigna_mungo",
    "apis_mellifera",
    "methanosarcina_mazei",
    "bacillus_subtilis",
]

SPECIES_DISPLAY_NAMES = {
    "yeast": "S. cerevisiae (Yeast)",
    "saccharomyces_cerevisiae": "S. cerevisiae (Yeast)",
    "tomato": "S. lycopersicum (Tomato)",
    "solanum_lycopersicum": "S. lycopersicum (Tomato)",
    "human": "H. sapiens (Human)",
    "h_sapiens": "H. sapiens (Human)",
    "mouse": "M. musculus (Mouse)",
    "mus_musculus": "M. musculus (Mouse)",
    "bacillus": "B. subtilis",
    "bacillus_subtilis": "B. subtilis",
    "mungo": "V. mungo (Black gram)",
    "vigna_mungo": "V. mungo (Black gram)",
    "methanosarcina": "M. mazei (Archaea)",
    "methanosarcina_mazei": "M. mazei (Archaea)",
    "honeybee": "A. mellifera (Honeybee)",
    "apis_mellifera": "A. mellifera (Honeybee)",
    "candidatus": "C. endoloripes",
    "candidatus_endoloripes": "C. endoloripes",
}


def load_species_table(
    species_name: str,
    cache_dir: str = "data/cache",
    token: str | None = None,
    max_length: int = 30,
    max_samples: int | None = None,
) -> pa.Table:
    """Download and memory-map species IPC file, filtering to peptide length <= max_length."""
    ipc_filename = SPECIES_MAP.get(species_name.lower(), f"{species_name}.ipc")
    remote_path = f"data/ninespecies_updated/{ipc_filename}"

    print(f"Downloading/Locating {remote_path} from {REPO_ID}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=remote_path,
        repo_type="dataset",
        cache_dir=cache_dir,
        token=token or os.environ.get("HF_TOKEN"),
    )

    mmap = pa.memory_map(local_path, "r")
    reader = ipc.open_file(mmap)
    batch = reader.get_batch(0)

    # Filter to supported peptide length
    target_count = f"up to {max_samples}" if max_samples else "all"
    print(f"Filtering {species_name} table ({batch.num_rows} spectra) to length <= {max_length} ({target_count})...")
    valid_indices = []
    seq_col = batch["modified_sequence"] if "modified_sequence" in batch.schema.names else batch["sequence"]
    fallback_col = batch["sequence"]

    for i in range(batch.num_rows):
        raw = seq_col[i].as_py() or fallback_col[i].as_py()
        toks = parse_peptide(raw)
        if 1 <= len(toks) <= max_length:
            valid_indices.append(i)
            if max_samples is not None and len(valid_indices) >= max_samples:
                break

    filtered = batch.take(pa.array(valid_indices))
    print(f"Retained {filtered.num_rows} spectra.")
    return filtered


def run_species_evaluation(
    species_name: str,
    table: pa.Table,
    models: tuple,
    vocabulary: dict[str, int],
    device: torch.device,
    batch_size: int = 4000,
    max_batches: int | None = None,
    num_steps: int = 25,
    guidance_scale: float = 1.8,
    top_k_lengths: int = 5,
    alpha: float = 0.5,
    use_knapsack_filter: bool = True,
    knapsack_tol_da: float = 1.0,
    scheduler: str = "linear",
    num_workers: int = 4,
) -> dict[str, Any]:
    """Run generative evaluation on a single species table."""
    spectrum_encoder, length_predictor, decoder, guidance = models

    ds = SpectrumDataSet(table, vocab=vocabulary, is_train=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: spectrum_collate(b, vocabulary),
    )

    total_spectra = min(len(ds), (max_batches or 999999) * batch_size)
    print(f"\n--- Evaluating {species_name.upper()} ({total_spectra} spectra) [Scheduler: {scheduler}] ---")
    start_time = time.time()

    metrics = evaluate_generative(
        loader=loader,
        vocabulary=vocabulary,
        spectrum_encoder=spectrum_encoder,
        length_predictor=length_predictor,
        decoder=decoder,
        guidance=guidance,
        scheduler=scheduler,
        device=device,
        max_batches=max_batches,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        top_k_lengths=top_k_lengths,
        alpha=alpha,
        use_knapsack_filter=use_knapsack_filter,
        knapsack_tol_da=knapsack_tol_da,
        return_details=False,
    )

    elapsed = time.time() - start_time
    print(
        f"[{species_name}] Done in {elapsed:.1f}s | "
        f"Strict: {metrics.exact_peptide_accuracy:.2%} | "
        f"I/L: {metrics.exact_peptide_accuracy_il:.2%} | "
        f"Mass: {metrics.mass_peptide_accuracy:.2%} | "
        f"Length: {metrics.length_accuracy:.2%} | "
        f"AA F1: {metrics.aa_f1:.2%}"
    )

    res = metrics.to_dict()
    res["species"] = species_name
    res["display_name"] = SPECIES_DISPLAY_NAMES.get(species_name, species_name)
    res["eval_time_seconds"] = round(elapsed, 2)
    return res


def plot_ninespecies_barplots(
    results: list[dict[str, Any]],
    output_path: str = "artifacts/ninespecies_full_benchmark_barplots.png",
):
    """Plot multi-panel barplot comparing all 9 species and macro average."""
    import matplotlib.pyplot as plt

    if not results:
        return

    names = [SPECIES_DISPLAY_NAMES.get(r.get("species", ""), r.get("species", "")) for r in results]
    exact_il = [r.get("exact_peptide_accuracy_il", 0) * 100 for r in results]
    mass = [r.get("mass_peptide_accuracy", 0) * 100 for r in results]
    aa_f1 = [r.get("aa_f1", 0) * 100 for r in results]
    len_acc = [r.get("length_accuracy", 0) * 100 for r in results]
    strict = [r.get("exact_peptide_accuracy", 0) * 100 for r in results]

    # Compute macro averages
    avg_il = float(np.mean(exact_il))
    avg_mass = float(np.mean(mass))
    avg_f1 = float(np.mean(aa_f1))
    avg_len = float(np.mean(len_acc))
    avg_strict = float(np.mean(strict))

    # Append Macro Average column
    names_with_avg = names + ["Macro Average"]
    exact_il.append(avg_il)
    mass.append(avg_mass)
    aa_f1.append(avg_f1)
    len_acc.append(avg_len)
    strict.append(avg_strict)

    x = np.arange(len(names_with_avg))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), dpi=300, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    # --- Top Panel: 4-Metric De Novo Sequencing Performance ---
    r1 = ax1.bar(
        x - 1.5 * width, exact_il, width, label="I/L Exact Match (%)", color="#1f77b4", edgecolor="black", linewidth=0.5
    )
    r2 = ax1.bar(
        x - 0.5 * width, mass, width, label="Mass Match (%)", color="#2ca02c", edgecolor="black", linewidth=0.5
    )
    r3 = ax1.bar(
        x + 0.5 * width, aa_f1, width, label="Amino Acid F1 (%)", color="#ff7f0e", edgecolor="black", linewidth=0.5
    )
    r4 = ax1.bar(
        x + 1.5 * width, len_acc, width, label="Length Accuracy (%)", color="#9467bd", edgecolor="black", linewidth=0.5
    )

    ax1.set_ylabel("Accuracy / F1 (%)", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Discrete Flow Matching (DFM) One-Shot Cross-Species Benchmark: Performance Across 9 Species",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_with_avg, rotation=25, ha="right", fontsize=11, fontweight="semibold")
    ax1.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 95)

    # Highlight Macro Average bar group
    ax1.axvline(x[-1] - 0.5, color="gray", linestyle="-.", linewidth=1.5, alpha=0.7)

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax1.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    # --- Bottom Panel: Strict Exact Match Comparison ---
    r_strict = ax2.bar(
        x, strict, width * 2, label="Strict Exact Match (%)", color="#d62728", edgecolor="black", linewidth=0.5
    )
    ax2.set_ylabel("Strict Match (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Strict Exact Match Accuracy Across Species", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_with_avg, rotation=25, ha="right", fontsize=10, fontweight="semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, max(strict) * 1.25)
    ax2.axvline(x[-1] - 0.5, color="gray", linestyle="-.", linewidth=1.5, alpha=0.7)

    for rect in r_strict:
        h = rect.get_height()
        if h > 0:
            ax2.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved benchmark barplots to {out}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DFM on Nine-Species dataset.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/dfm_pl_run_20260915_000148/checkpoints/best-gen-exact-epoch=07-exact=0.3514.ckpt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["all"],
        help="List of species to benchmark, or 'all' for all 9 species.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4000, help="Batch size for inference (default: 4000 on H100)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Max spectra per species to evaluate (default: 50000, evaluates all if <50000).",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per species.")
    parser.add_argument("--num-steps", type=int, default=25, help="Diffusion inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=1.8, help="Classifier-free guidance scale.")
    parser.add_argument("--top-k-lengths", type=int, default=5, help="Number of length candidates in beam search.")
    parser.add_argument("--length-beam-alpha", type=float, default=0.5, help="Precursor mass discrepancy penalty.")
    parser.add_argument(
        "--use-knapsack-filter",
        dest="use_knapsack_filter",
        action="store_true",
        default=True,
        help="Enable vectorized knapsack mass filtering.",
    )
    parser.add_argument(
        "--knapsack-tol-da",
        type=float,
        default=1.0,
        help="Tolerance in Da for knapsack filtering.",
    )
    parser.add_argument("--max-peptide-length", type=int, default=30, help="Maximum peptide length to evaluate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--cache-dir", default="data/cache", help="Hugging Face cache directory.")
    parser.add_argument(
        "--scheduler",
        default="linear",
        choices=["linear", "cosine", "power1_5", "power2", "sigmoid"],
        help="Noise scheduler for flow matching reverse sampling (default: linear).",
    )
    parser.add_argument(
        "--output", default="artifacts/ninespecies_full_benchmark_results.json", help="Output JSON path."
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/ninespecies_full_benchmark_summary.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--plot-output",
        default="artifacts/ninespecies_full_benchmark_barplots.png",
        help="Path to save multi-panel barplot PNG.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================")
    print("  DFM Nine-Species Cross-Species Deep Benchmark")
    print("==========================================================")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Max Samples/Sp:   {args.max_samples}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Inference Steps:  {args.num_steps}")
    print(f"Guidance Scale:   {args.guidance_scale}")
    print(f"Top-K Lengths:    {args.top_k_lengths}")
    print(f"Alpha:            {args.length_beam_alpha}")
    print(f"Knapsack Filter:  {args.use_knapsack_filter} (tol={args.knapsack_tol_da} Da)")
    print(f"Device:           {device}")
    print("==========================================================")

    # 1. Load Checkpoint and Vocabulary
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    vocabulary = ckpt.get("vocabulary")
    if vocabulary is None:
        vocabulary = build_vocabulary(include_ptms=True)

    # 2. Build Models
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)
    load_models_from_checkpoint(ckpt, spectrum_encoder, length_predictor, decoder, guidance)
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()
    models = (spectrum_encoder, length_predictor, decoder, guidance)

    # 3. Determine Species List
    if "all" in [s.lower() for s in args.species]:
        species_list = ALL_SPECIES
    else:
        species_list = []
        for s in args.species:
            s_clean = s.lower().strip()
            if s_clean in SPECIES_MAP:
                species_list.append(s_clean)
            elif s_clean in ALL_SPECIES:
                species_list.append(s_clean)
            else:
                print(f"Warning: Unknown species '{s}', skipping. Valid options: {list(SPECIES_MAP.keys())}")

    print(f"Benchmark plan: {len(species_list)} species -> {species_list}\n")
    all_results = []

    for idx, sp in enumerate(species_list, 1):
        print(f"\n>>> [{idx}/{len(species_list)}] Preparing {sp}...")
        try:
            table = load_species_table(
                species_name=sp,
                cache_dir=args.cache_dir,
                max_length=args.max_peptide_length,
                max_samples=args.max_samples,
            )
            res = run_species_evaluation(
                species_name=sp,
                table=table,
                models=models,
                vocabulary=vocabulary,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                top_k_lengths=args.top_k_lengths,
                alpha=args.length_beam_alpha,
                use_knapsack_filter=args.use_knapsack_filter,
                knapsack_tol_da=args.knapsack_tol_da,
                scheduler=args.scheduler,
                num_workers=args.num_workers,
            )
            all_results.append(res)
        except Exception as e:
            print(f"Error benchmarking {sp}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Save Raw Results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed benchmark results to {out_path}")

    # 5. Compute Macro Averages and Save Summary
    species_summary = []
    for r in all_results:
        species_summary.append({
            "species": r.get("display_name", r.get("species")),
            "samples": r.get("num_samples", 0),
            "exact": round(r.get("exact_peptide_accuracy", 0), 4),
            "exact_il": round(r.get("exact_peptide_accuracy_il", 0), 4),
            "mass": round(r.get("mass_peptide_accuracy", 0), 4),
            "len_acc": round(r.get("length_accuracy", 0), 4),
            "aa_f1": round(r.get("aa_f1", 0), 4),
        })

    # Sort descending by I/L exact match
    species_summary.sort(key=lambda x: x["exact_il"], reverse=True)

    macro_avg = {
        "strict_exact": float(np.mean([x["exact"] for x in species_summary])),
        "exact_il": float(np.mean([x["exact_il"] for x in species_summary])),
        "mass_match": float(np.mean([x["mass"] for x in species_summary])),
        "length_accuracy": float(np.mean([x["len_acc"] for x in species_summary])),
        "aa_f1": float(np.mean([x["aa_f1"] for x in species_summary])),
    }

    summary_data = {
        "species_results": species_summary,
        "macro_average": macro_avg,
    }
    summary_path = Path(args.summary_output)
    with summary_path.open("w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved benchmark summary to {summary_path}")

    # 6. Generate Multi-Panel Barplots
    plot_ninespecies_barplots(all_results, output_path=args.plot_output)

    # 7. Print Formatted Table
    print("\n" + "=" * 105)
    print(f"{'Species':<30} | {'Samples':<8} | {'Strict':<8} | {'I/L Exact':<10} | {'Mass Match':<10} | {'Len Acc':<8} | {'AA F1':<8}")
    print("-" * 105)
    for r in species_summary:
        sp = r.get("species", "Unknown")
        n = r.get("samples", 0)
        st = f"{r.get('exact', 0):.2%}"
        il = f"{r.get('exact_il', 0):.2%}"
        mm = f"{r.get('mass', 0):.2%}"
        la = f"{r.get('len_acc', 0):.2%}"
        f1 = f"{r.get('aa_f1', 0):.2%}"
        print(f"{sp:<30} | {n:<8} | {st:<8} | {il:<10} | {mm:<10} | {la:<8} | {f1:<8}")
    print("-" * 105)
    print(
        f"{'MACRO AVERAGE':<30} | {'-':<8} | "
        f"{macro_avg['strict_exact']:.2%}   | {macro_avg['exact_il']:.2%}     | "
        f"{macro_avg['mass_match']:.2%}     | {macro_avg['length_accuracy']:.2%}   | "
        f"{macro_avg['aa_f1']:.2%}"
    )
    print("=" * 105)


if __name__ == "__main__":
    main()
