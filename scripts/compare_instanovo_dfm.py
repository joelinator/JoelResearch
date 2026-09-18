#!/usr/bin/env python3
"""
Direct benchmark comparison between InstaNovo (autoregressive transformer + beam search)
and DFM (Discrete Flow Matching with knapsack filtering) on 3 species of the Nine Species dataset.

Species:
  1. Yeast (Saccharomyces cerevisiae) - Eukaryote / Fungi
  2. Human (Homo sapiens) - Mammal / Primates
  3. Bacteria (Bacillus subtilis) - Prokaryote / Bacteria

Metrics evaluated identically on both models:
  - Strict Exact Match (%)
  - I/L-Tolerant Exact Match (%)
  - Mass-Based Match (%)
  - Length Accuracy (%)
  - Amino Acid Precision (%)
  - Amino Acid Recall (%)
  - Amino Acid F1 (%)
  - Inference Throughput (spectra/sec) & Total Runtime
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from huggingface_hub import hf_hub_download

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.data import build_vocabulary, parse_peptide
from eval.metrics import compute_denovo_metrics
from flow_matching.scheduler import cosine_scheduler
from scripts.benchmark_ninespecies import (
    REPO_ID,
    SPECIES_DISPLAY_NAMES,
    SPECIES_MAP,
    load_species_table,
    run_species_evaluation,
)
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

BENCHMARK_SPECIES = ["yeast", "human", "bacillus"]
SPECIES_MAP["bacteria"] = "bacillus_subtilis.ipc"
SPECIES_DISPLAY_NAMES["bacteria"] = "B. subtilis (Bacteria)"
SPECIES_DISPLAY_NAMES["bacillus"] = "B. subtilis (Bacteria)"


def parse_args():
    parser = argparse.ArgumentParser(description="InstaNovo vs DFM Head-to-Head Benchmark.")
    parser.add_argument(
        "--species",
        nargs="+",
        default=BENCHMARK_SPECIES,
        help="Species to benchmark (default: yeast, human, bacteria).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum spectra to evaluate per species (default: 5000).",
    )
    parser.add_argument(
        "--dfm-checkpoint",
        default="artifacts/dfm_pl_run_20260915_000148/checkpoints/best-gen-exact-epoch=07-exact=0.3514.ckpt",
        help="Path to DFM checkpoint.",
    )
    parser.add_argument(
        "--instanovo-model",
        default="instanovo-v1.2.0",
        help="InstaNovo pretrained model ID or checkpoint.",
    )
    parser.add_argument(
        "--dfm-batch-size",
        type=int,
        default=4000,
        help="DFM batch size (default: 4000).",
    )
    parser.add_argument(
        "--instanovo-batch-size",
        type=int,
        default=1024,
        help="InstaNovo batch size (default: 1024).",
    )
    parser.add_argument(
        "--dfm-steps",
        type=int,
        default=25,
        help="DFM flow matching steps (default: 25).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.8,
        help="DFM classifier-free guidance scale (default: 1.8).",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/instanovo_vs_dfm",
        help="Directory for intermediate parquet/predictions files.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/instanovo_vs_dfm_comparison.json",
        help="Path to save output comparison JSON.",
    )
    parser.add_argument(
        "--output-plot",
        default="artifacts/instanovo_vs_dfm_comparison.png",
        help="Path to save comparison barplots PNG.",
    )
    return parser.parse_args()


def export_table_to_parquet(table: pa.Table, parquet_path: Path) -> Path:
    """Exports PyArrow Table to Parquet format expected by InstaNovo."""
    df = pl.from_arrow(table)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return parquet_path


def run_instanovo(
    parquet_path: Path,
    predictions_csv: Path,
    model_id: str,
    batch_size: int = 1024,
) -> float:
    """Runs InstaNovo CLI prediction and measures wall-clock execution time."""
    instanovo_bin = Path(sys.executable).parent / "instanovo"
    if not instanovo_bin.exists():
        instanovo_bin = "instanovo"

    cmd = [
        str(instanovo_bin),
        "transformer",
        "predict",
        "--data-path",
        str(parquet_path),
        "--output-path",
        str(predictions_csv),
        "--instanovo-model",
        model_id,
        "--denovo",
        f"batch_size={batch_size}",
    ]
    print(f"\n[InstaNovo] Running: {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"[InstaNovo] Finished in {elapsed:.2f}s.")
    return elapsed


def evaluate_instanovo_predictions(
    predictions_csv: Path,
    ground_truth_targets: list[str],
) -> dict[str, float]:
    """Computes standardized metrics on InstaNovo predictions."""
    pred_df = pd.read_csv(predictions_csv)
    raw_preds = pred_df["predictions"].fillna("").astype(str).tolist()

    clean_preds = ["".join(parse_peptide(p)) for p in raw_preds]
    clean_targets = ["".join(parse_peptide(t)) for t in ground_truth_targets]

    metrics = compute_denovo_metrics(clean_preds, clean_targets)
    d = metrics.to_dict()

    length_acc = float(
        np.mean([len(p) == len(t) for p, t in zip(clean_preds, clean_targets)])
    )

    return {
        "exact_match": d["exact_peptide_accuracy"],
        "exact_match_il": d["exact_peptide_accuracy_il"],
        "mass_match": d["mass_peptide_accuracy"],
        "length_accuracy": length_acc,
        "aa_precision": d["aa_precision"],
        "aa_recall": d["aa_recall"],
        "aa_f1": d["aa_f1"],
    }


def plot_comparison(
    results: dict[str, Any],
    output_png: str | Path,
):
    """Generates a multi-panel comparison figure."""
    species_keys = list(results["species"].keys())
    species_labels = [
        results["species"][sp]["display_name"] for sp in species_keys
    ] + ["Macro Average"]

    dfm_strict = [results["species"][sp]["dfm"]["exact_match"] * 100 for sp in species_keys] + [
        results["macro_average"]["dfm"]["exact_match"] * 100
    ]
    in_strict = [results["species"][sp]["instanovo"]["exact_match"] * 100 for sp in species_keys] + [
        results["macro_average"]["instanovo"]["exact_match"] * 100
    ]

    dfm_il = [results["species"][sp]["dfm"]["exact_match_il"] * 100 for sp in species_keys] + [
        results["macro_average"]["dfm"]["exact_match_il"] * 100
    ]
    in_il = [results["species"][sp]["instanovo"]["exact_match_il"] * 100 for sp in species_keys] + [
        results["macro_average"]["instanovo"]["exact_match_il"] * 100
    ]

    dfm_mass = [results["species"][sp]["dfm"]["mass_match"] * 100 for sp in species_keys] + [
        results["macro_average"]["dfm"]["mass_match"] * 100
    ]
    in_mass = [results["species"][sp]["instanovo"]["mass_match"] * 100 for sp in species_keys] + [
        results["macro_average"]["instanovo"]["mass_match"] * 100
    ]

    dfm_f1 = [results["species"][sp]["dfm"]["aa_f1"] * 100 for sp in species_keys] + [
        results["macro_average"]["dfm"]["aa_f1"] * 100
    ]
    in_f1 = [results["species"][sp]["instanovo"]["aa_f1"] * 100 for sp in species_keys] + [
        results["macro_average"]["instanovo"]["aa_f1"] * 100
    ]

    dfm_speed = [results["species"][sp]["dfm"]["throughput_sps"] for sp in species_keys] + [
        results["macro_average"]["dfm"]["throughput_sps"]
    ]
    in_speed = [results["species"][sp]["instanovo"]["throughput_sps"] for sp in species_keys] + [
        results["macro_average"]["instanovo"]["throughput_sps"]
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "InstaNovo vs DFM Head-to-Head Benchmark on Nine Species",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(species_labels))
    width = 0.35

    color_dfm = "#1D70B8"  # Teal / Blue
    color_in = "#E66101"   # Orange / Coral

    panels = [
        (axes[0, 0], dfm_strict, in_strict, "Strict Exact Match (%)", "%", 85),
        (axes[0, 1], dfm_il, in_il, "I/L-Tolerant Exact Match (%)", "%", 90),
        (axes[0, 2], dfm_mass, in_mass, "Mass-Based Match (%)", "%", 95),
        (axes[1, 0], dfm_f1, in_f1, "Amino Acid F1 (%)", "%", 95),
        (axes[1, 1], dfm_speed, in_speed, "Throughput (Spectra / Sec)", " sps", 165),
    ]

    for ax, d_vals, in_vals, title, unit, ylim in panels:
        bars1 = ax.bar(x - width / 2, d_vals, width, label="DFM (Ours)", color=color_dfm, alpha=0.9)
        bars2 = ax.bar(x + width / 2, in_vals, width, label="InstaNovo (Baseline)", color=color_in, alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(species_labels, rotation=15, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim:
            ax.set_ylim(0, ylim)
        ax.legend(loc="upper left", frameon=True)

        for bar in bars1:
            h = bar.get_height()
            fmt = f"{h:.1f}{unit}" if unit == "%" else f"{int(h)}"
            ax.annotate(fmt, (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            fmt = f"{h:.1f}{unit}" if unit == "%" else f"{int(h)}"
            ax.annotate(fmt, (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 6: Summary Metrics Table
    ax_table = axes[1, 2]
    ax_table.axis("off")

    table_data = [
        ["Metric", "DFM (Ours)", "InstaNovo", "Ratio / Δ"],
        ["Macro Exact Match", f"{results['macro_average']['dfm']['exact_match'] * 100:.2f}%", f"{results['macro_average']['instanovo']['exact_match'] * 100:.2f}%", f"{results['macro_average']['dfm']['exact_match'] / max(1e-6, results['macro_average']['instanovo']['exact_match']):.2f}x"],
        ["Macro I/L Match", f"{results['macro_average']['dfm']['exact_match_il'] * 100:.2f}%", f"{results['macro_average']['instanovo']['exact_match_il'] * 100:.2f}%", f"{results['macro_average']['dfm']['exact_match_il'] / max(1e-6, results['macro_average']['instanovo']['exact_match_il']):.2f}x"],
        ["Macro Mass Match", f"{results['macro_average']['dfm']['mass_match'] * 100:.2f}%", f"{results['macro_average']['instanovo']['mass_match'] * 100:.2f}%", f"{results['macro_average']['dfm']['mass_match'] / max(1e-6, results['macro_average']['instanovo']['mass_match']):.2f}x"],
        ["Macro AA F1", f"{results['macro_average']['dfm']['aa_f1'] * 100:.2f}%", f"{results['macro_average']['instanovo']['aa_f1'] * 100:.2f}%", f"{results['macro_average']['dfm']['aa_f1'] / max(1e-6, results['macro_average']['instanovo']['aa_f1']):.2f}x"],
        ["Avg Throughput", f"{results['macro_average']['dfm']['throughput_sps']:.1f} sps", f"{results['macro_average']['instanovo']['throughput_sps']:.1f} sps", f"{results['macro_average']['dfm']['throughput_sps'] / max(1e-6, results['macro_average']['instanovo']['throughput_sps']):.1f}x Faster"],
    ]

    table = ax_table.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
        bbox=[0.02, 0.12, 0.96, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for i in range(4):
        table[(0, i)].set_facecolor("#D9D9D9")
        table[(0, i)].get_text().set_fontweight("bold")
    for (row, col), cell in table.get_celld().items():
        cell.set_height(0.12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\n✅ Plot saved to {output_png}")

    brain_dir = Path("/home/joelgedeon_aims_ac_za/.gemini/antigravity-cli/brain/ee9cf031-b0c5-4740-b963-40c81c13ea78")
    if brain_dir.exists():
        import shutil
        shutil.copy(output_png, brain_dir / Path(output_png).name)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("  Head-to-Head Benchmark: InstaNovo vs DFM")
    print("==========================================================")
    print(f"Species:              {args.species}")
    print(f"Max Samples/Species:  {args.max_samples}")
    print(f"DFM Checkpoint:       {args.dfm_checkpoint}")
    print(f"InstaNovo Model:      {args.instanovo_model}")
    print(f"DFM Batch Size:       {args.dfm_batch_size}")
    print(f"InstaNovo Batch Size: {args.instanovo_batch_size}")
    print(f"Device:               {device}")
    print("==========================================================")

    # 1. Load DFM Checkpoint & Models
    print("\nLoading DFM model...")
    ckpt = load_checkpoint(args.dfm_checkpoint, map_location=device)
    vocabulary = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)
    load_models_from_checkpoint(ckpt, spectrum_encoder, length_predictor, decoder, guidance)
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()
    dfm_models = (spectrum_encoder, length_predictor, decoder, guidance)

    results: dict[str, Any] = {
        "species": {},
        "macro_average": {},
        "config": vars(args),
    }

    # 2. Iterate Species
    for sp in args.species:
        display_name = SPECIES_DISPLAY_NAMES.get(sp, sp)
        print(f"\n{'='*60}")
        print(f"  BENCHMARKING SPECIES: {display_name} ({sp})")
        print(f"{'='*60}")

        # Load & filter exact table
        table = load_species_table(
            species_name=sp,
            cache_dir=args.cache_dir,
            max_length=30,
            max_samples=args.max_samples,
        )
        n_samples = table.num_rows

        # Save to parquet for InstaNovo
        parquet_file = work_dir / f"{sp}_{n_samples}.parquet"
        pred_csv = work_dir / f"{sp}_{n_samples}_instanovo_preds.csv"
        export_table_to_parquet(table, parquet_file)

        # -------------------- Run InstaNovo --------------------
        in_time = run_instanovo(
            parquet_path=parquet_file,
            predictions_csv=pred_csv,
            model_id=args.instanovo_model,
            batch_size=args.instanovo_batch_size,
        )
        in_sps = n_samples / max(1e-6, in_time)

        # Extract GT sequences
        seq_col = table["modified_sequence"] if "modified_sequence" in table.schema.names else table["sequence"]
        ground_truth = seq_col.to_pylist()
        in_metrics = evaluate_instanovo_predictions(pred_csv, ground_truth)
        in_metrics["time_seconds"] = round(in_time, 2)
        in_metrics["throughput_sps"] = round(in_sps, 2)

        print(f"\n[InstaNovo Results for {display_name}]:")
        print(f"  Strict Exact: {in_metrics['exact_match']:.2%} | I/L: {in_metrics['exact_match_il']:.2%} | Mass: {in_metrics['mass_match']:.2%} | Length: {in_metrics['length_accuracy']:.2%} | AA F1: {in_metrics['aa_f1']:.2%}")
        print(f"  Time: {in_time:.1f}s ({in_sps:.1f} spectra/sec)")

        # -------------------- Run DFM --------------------
        print(f"\n[DFM] Running inference on {n_samples} spectra (batch_size={args.dfm_batch_size})...")
        dfm_start = time.time()
        dfm_res = run_species_evaluation(
            species_name=sp,
            table=table,
            models=dfm_models,
            vocabulary=vocabulary,
            device=device,
            batch_size=args.dfm_batch_size,
            num_steps=args.dfm_steps,
            guidance_scale=args.guidance_scale,
            top_k_lengths=5,
            alpha=0.5,
            use_knapsack_filter=True,
            num_workers=0,
        )
        dfm_time = time.time() - dfm_start
        dfm_sps = n_samples / max(1e-6, dfm_time)

        dfm_metrics = {
            "exact_match": dfm_res["exact_peptide_accuracy"],
            "exact_match_il": dfm_res["exact_peptide_accuracy_il"],
            "mass_match": dfm_res["mass_peptide_accuracy"],
            "length_accuracy": dfm_res["length_accuracy"],
            "aa_precision": dfm_res["aa_precision"],
            "aa_recall": dfm_res["aa_recall"],
            "aa_f1": dfm_res["aa_f1"],
            "time_seconds": round(dfm_time, 2),
            "throughput_sps": round(dfm_sps, 2),
        }

        print(f"\n[DFM Results for {display_name}]:")
        print(f"  Strict Exact: {dfm_metrics['exact_match']:.2%} | I/L: {dfm_metrics['exact_match_il']:.2%} | Mass: {dfm_metrics['mass_match']:.2%} | Length: {dfm_metrics['length_accuracy']:.2%} | AA F1: {dfm_metrics['aa_f1']:.2%}")
        print(f"  Time: {dfm_time:.1f}s ({dfm_sps:.1f} spectra/sec)")

        speedup = dfm_sps / max(1e-6, in_sps)
        print(f"\n⚡ Speedup: DFM is {speedup:.1f}x faster than InstaNovo!")

        results["species"][sp] = {
            "display_name": display_name,
            "num_samples": n_samples,
            "instanovo": in_metrics,
            "dfm": dfm_metrics,
            "speedup": round(speedup, 2),
        }

    # 3. Macro Average
    macro_keys = ["exact_match", "exact_match_il", "mass_match", "length_accuracy", "aa_precision", "aa_recall", "aa_f1", "throughput_sps"]
    results["macro_average"] = {
        "instanovo": {k: float(np.mean([results["species"][sp]["instanovo"][k] for sp in args.species])) for k in macro_keys},
        "dfm": {k: float(np.mean([results["species"][sp]["dfm"][k] for sp in args.species])) for k in macro_keys},
    }

    print("\n" + "="*80)
    print("  OVERALL 3-SPECIES MACRO AVERAGE COMPARISON")
    print("="*80)
    print(f"{'Metric':<25} | {'DFM (Ours)':<15} | {'InstaNovo':<15} | {'Advantage / Ratio':<20}")
    print("-" * 80)
    for k in macro_keys:
        d_val = results["macro_average"]["dfm"][k]
        in_val = results["macro_average"]["instanovo"][k]
        if k == "throughput_sps":
            ratio_str = f"{d_val / max(1e-6, in_val):.1f}x faster"
            print(f"{'Throughput (spectra/s)':<25} | {d_val:<15.1f} | {in_val:<15.1f} | {ratio_str:<20}")
        else:
            ratio_str = f"DFM {(d_val - in_val)*100:+.2f}%" if k != "throughput_sps" else ""
            print(f"{k:<25} | {d_val:<15.2%} | {in_val:<15.2%} | {ratio_str:<20}")
    print("="*80)

    # Save JSON & Plot
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison JSON to {args.output_json}")

    plot_comparison(results, args.output_plot)


if __name__ == "__main__":
    main()
