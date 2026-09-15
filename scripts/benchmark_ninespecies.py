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


def load_species_table(
    species_name: str,
    cache_dir: str = "data/cache",
    token: str | None = None,
    max_length: int = 30,
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
    print(f"Filtering {species_name} table ({batch.num_rows} spectra) to length <= {max_length}...")
    valid_indices = []
    seq_col = batch["modified_sequence"] if "modified_sequence" in batch.schema.names else batch["sequence"]
    fallback_col = batch["sequence"]

    for i in range(batch.num_rows):
        raw = seq_col[i].as_py() or fallback_col[i].as_py()
        toks = parse_peptide(raw)
        if 1 <= len(toks) <= max_length:
            valid_indices.append(i)

    filtered = batch.take(pa.array(valid_indices))
    print(f"Retained {filtered.num_rows} / {batch.num_rows} spectra ({filtered.num_rows / batch.num_rows:.1%}).")
    return filtered


def run_species_evaluation(
    species_name: str,
    table: pa.Table,
    models: tuple,
    vocabulary: dict[str, int],
    device: torch.device,
    batch_size: int = 4000,
    max_batches: int | None = None,
    num_steps: int = 20,
    guidance_scale: float = 1.5,
    top_k_lengths: int = 5,
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
    print(f"\n--- Evaluating {species_name.upper()} ({total_spectra} spectra) ---")
    start_time = time.time()

    metrics = evaluate_generative(
        loader=loader,
        vocabulary=vocabulary,
        spectrum_encoder=spectrum_encoder,
        length_predictor=length_predictor,
        decoder=decoder,
        guidance=guidance,
        scheduler=cosine_scheduler,
        device=device,
        max_batches=max_batches,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        top_k_lengths=top_k_lengths,
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
    res["eval_time_seconds"] = round(elapsed, 2)
    return res


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
        default=["yeast", "mouse"],
        help="List of species to benchmark, or 'all' for all 9 species.",
    )
    parser.add_argument("--batch-size", type=int, default=4000, help="Batch size for inference (default: 4000 on H100).")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per species (None for full split).")
    parser.add_argument("--num-steps", type=int, default=20, help="Diffusion inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="Classifier-free guidance scale.")
    parser.add_argument("--top-k-lengths", type=int, default=5, help="Number of length candidates in beam search.")
    parser.add_argument("--max-peptide-length", type=int, default=30, help="Maximum peptide length to evaluate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--cache-dir", default="data/cache", help="Hugging Face cache directory.")
    parser.add_argument("--output", default="artifacts/ninespecies_benchmark_results.json", help="Output JSON path.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Checkpoint: {args.checkpoint}")

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

    print(f"Benchmark plan: {len(species_list)} species -> {species_list}")
    all_results = []

    for sp in species_list:
        try:
            table = load_species_table(
                species_name=sp,
                cache_dir=args.cache_dir,
                max_length=args.max_peptide_length,
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
                num_workers=args.num_workers,
            )
            all_results.append(res)
        except Exception as e:
            print(f"Error benchmarking {sp}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Save and Print Summary Table
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")

    # Summary Table
    print("\n" + "=" * 95)
    print(f"{'Species':<25} | {'Samples':<8} | {'Strict':<8} | {'I/L Exact':<10} | {'Mass Match':<10} | {'Len Acc':<8} | {'AA F1':<8}")
    print("-" * 95)
    for r in all_results:
        sp = r.get("species", "Unknown")
        n = r.get("num_samples", 0)
        st = f"{r.get('exact_peptide_accuracy', 0):.2%}"
        il = f"{r.get('exact_peptide_accuracy_il', 0):.2%}"
        mm = f"{r.get('mass_peptide_accuracy', 0):.2%}"
        la = f"{r.get('length_accuracy', 0):.2%}"
        f1 = f"{r.get('aa_f1', 0):.2%}"
        print(f"{sp:<25} | {n:<8} | {st:<8} | {il:<10} | {mm:<10} | {la:<8} | {f1:<8}")
    print("=" * 95)


if __name__ == "__main__":
    main()
