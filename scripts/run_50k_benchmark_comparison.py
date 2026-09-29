#!/usr/bin/env python3
"""
50k Spectra Head-to-Head Benchmark:
Compares Baseline (K=3, S=1) vs. Algorithmic Refined (K=3, S=2, eta=0.1)
on identical 50,000 test spectra from:
1. Nine-Species Benchmark (InstaDeepAI/ms_ninespecies_benchmark)
2. Human Core ProteomeTools (InstaDeepAI/ms_proteometools)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_canonical_ckpt = PROJECT_ROOT / "models" / "frozen_production_model.ckpt"
_train_ckpt = PROJECT_ROOT / "artifacts" / "dfm_joint_balanced_30ep" / "checkpoints" / "dfm_balanced_best.ckpt"
DEFAULT_CKPT = _canonical_ckpt if _canonical_ckpt.exists() else _train_ckpt


def load_model(ckpt_path: Path):
    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()
    return enc, lp, dec, guid, vocab


def evaluate_run(
    enc, lp, dec, guid, vocab,
    dataset_name: str,
    max_samples: int = 50000,
    batch_size: int = 128,
    top_k_lengths: int = 3,
    num_samples_per_length: int = 1,
    eta: float = 0.0,
    temperature: float = 0.0,
    desc: str = "",
) -> dict:
    print(f"\n=======================================================")
    print(f"[{desc}] Evaluating on {dataset_name} ({max_samples:,} spectra)")
    print(f"Params: K={top_k_lengths}, S={num_samples_per_length}, eta={eta}, temp={temperature}")
    print(f"=======================================================")

    ds = get_dataset(dataset_name, split="test", cache_dir="data/cache")
    loader = build_dataloader(
        ds,
        vocab,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    t0 = time.time()
    metrics, details = evaluate_generative(
        loader,
        vocab,
        enc,
        lp,
        dec,
        guid,
        scheduler="cosine",
        device=DEVICE,
        max_samples=max_samples,
        num_steps=20,
        top_k_lengths=top_k_lengths,
        num_samples_per_length=num_samples_per_length,
        eta=eta,
        temperature=temperature,
        use_exact_dp_knapsack=True,
        use_composite_ladders=True,
        return_details=True,
    )
    dt = time.time() - t0
    throughput = len(details["predictions"]) / dt

    res = {
        "description": desc,
        "dataset": dataset_name,
        "num_samples": len(details["predictions"]),
        "exact_match_strict": float(metrics.exact_peptide_accuracy),
        "exact_match_il": float(metrics.exact_peptide_accuracy_il),
        "aa_precision": float(metrics.aa_precision),
        "aa_recall": float(metrics.aa_recall),
        "aa_f1": float(metrics.aa_f1),
        "length_accuracy": float(metrics.length_accuracy),
        "elapsed_seconds": round(dt, 2),
        "throughput_spec_per_s": round(throughput, 1),
    }

    print(f"[{desc}] Done in {dt:.1f}s ({throughput:.1f} spec/s)")
    print(f"  Strict Exact Match: {res['exact_match_strict']:.2%}")
    print(f"  I/L Exact Match:    {res['exact_match_il']:.2%}")
    print(f"  Residue F1:         {res['aa_f1']:.2%}")
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head 50,000 spectrum benchmark comparing baseline vs refined decoding on Nine-Species and ProteomeTools."
    )
    parser.add_argument(
        "--model-path",
        "--checkpoint",
        dest="checkpoint",
        default=os.environ.get("MODEL_PATH", os.environ.get("CHECKPOINT", str(DEFAULT_CKPT))),
        help=f"Path to model checkpoint file (default: {DEFAULT_CKPT}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("MAX_SAMPLES", "50000")),
        help="Number of test spectra to evaluate per benchmark (default: 50000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "128")),
        help="Batch size for benchmark evaluation (default: 128).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "artifacts/benchmark_50k_comparison"),
        help="Directory to save comparison JSON results (default: artifacts/benchmark_50k_comparison).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Error: Model checkpoint file not found at '{args.checkpoint}'.", file=sys.stderr)
        print("Please verify the path or download the production checkpoint (see models/README.md).", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model checkpoint from:", args.checkpoint)
    try:
        enc, lp, dec, guid, vocab = load_model(ckpt_path)
    except Exception as exc:
        print(f"Error loading model from '{ckpt_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    results = {}

    # 1. Nine-Species Benchmark
    print("\n>>> STARTING NINE-SPECIES BENCHMARK (50,000 SPECTRA) <<<")
    results["ninespecies_baseline"] = evaluate_run(
        enc, lp, dec, guid, vocab,
        "InstaDeepAI/ms_ninespecies_benchmark",
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        top_k_lengths=3,
        num_samples_per_length=1,
        desc="Nine-Species Baseline",
    )

    results["ninespecies_refined"] = evaluate_run(
        enc, lp, dec, guid, vocab,
        "InstaDeepAI/ms_ninespecies_benchmark",
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        top_k_lengths=3,
        num_samples_per_length=2,
        eta=0.1,
        desc="Nine-Species Refined",
    )

    # 2. Human Core ProteomeTools Benchmark
    print("\n>>> STARTING HC-PT BENCHMARK (50,000 SPECTRA) <<<")
    results["hcpt_baseline"] = evaluate_run(
        enc, lp, dec, guid, vocab,
        "InstaDeepAI/ms_proteometools",
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        top_k_lengths=3,
        num_samples_per_length=1,
        desc="HC-PT Baseline",
    )

    results["hcpt_refined"] = evaluate_run(
        enc, lp, dec, guid, vocab,
        "InstaDeepAI/ms_proteometools",
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        top_k_lengths=3,
        num_samples_per_length=2,
        eta=0.1,
        desc="HC-PT Refined",
    )

    # Save results JSON
    json_path = out_dir / "benchmark_50k_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {json_path}")

    # Print summary table
    print("\n" + "="*80)
    print("  HEAD-TO-HEAD 50K SPECTRA BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Benchmark & Configuration':<32} | {'Strict Match':<12} | {'I/L Match':<12} | {'AA F1':<10} | {'Throughput':<12}")
    print("-" * 80)
    for key, r in results.items():
        print(f"{r['description']:<32} | {r['exact_match_strict']:<11.2%} | {r['exact_match_il']:<11.2%} | {r['aa_f1']:<9.2%} | {r['throughput_spec_per_s']:>6.1f} spec/s")
    print("="*80)


if __name__ == "__main__":
    main()
