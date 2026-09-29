#!/usr/bin/env python3
"""
Evaluates the K=8 Candidate Reranker (K=2 lengths x S=4 trajectories = 8 candidates)
with Composite Fragment Ladder Scoring on 50,000 spectra of:
1. Nine-Species Benchmark (InstaDeepAI/ms_ninespecies_benchmark)
2. Human Core ProteomeTools (InstaDeepAI/ms_proteometools)
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = PROJECT_ROOT / "artifacts" / "dfm_joint_balanced_30ep" / "checkpoints" / "dfm_balanced_best.ckpt"


def load_model():
    ckpt = load_checkpoint(str(CKPT_PATH), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()
    return enc, lp, dec, guid, vocab


def evaluate_k8(enc, lp, dec, guid, vocab, dataset_name: str, max_samples: int = 50000, batch_size: int = 128):
    print(f"\n=======================================================")
    print(f"Evaluating K=8 Candidate Reranker on {dataset_name} ({max_samples:,} spectra)")
    print(f"Config: K=2 lengths x S=4 trajectories (8 candidates), eta=0.1, composite fragment ladder scoring")
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
        top_k_lengths=2,
        num_samples_per_length=4,
        eta=0.1,
        temperature=0.0,
        beta=0.5,
        use_exact_dp_knapsack=True,
        use_composite_ladders=True,
        return_details=True,
    )
    dt = time.time() - t0
    n = len(details["predictions"])
    throughput = n / dt

    res = {
        "dataset": dataset_name,
        "num_samples": n,
        "exact_match_strict": float(metrics.exact_peptide_accuracy),
        "exact_match_il": float(metrics.exact_peptide_accuracy_il),
        "aa_precision": float(metrics.aa_precision),
        "aa_recall": float(metrics.aa_recall),
        "aa_f1": float(metrics.aa_f1),
        "length_accuracy": float(metrics.length_accuracy),
        "elapsed_seconds": round(dt, 2),
        "throughput_spec_per_s": round(throughput, 1),
    }

    print(f"Done in {dt:.1f}s ({throughput:.1f} spec/s)")
    print(f"  Strict Exact Match: {res['exact_match_strict']:.2%}")
    print(f"  I/L Exact Match:    {res['exact_match_il']:.2%}")
    print(f"  Residue F1:         {res['aa_f1']:.2%}")
    return res


def main():
    out_dir = PROJECT_ROOT / "artifacts" / "benchmark_50k_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model from:", CKPT_PATH)
    enc, lp, dec, guid, vocab = load_model()

    # 1. Nine-Species K=8 Reranker
    r_ninespecies_k8 = evaluate_k8(
        enc, lp, dec, guid, vocab,
        "InstaDeepAI/ms_ninespecies_benchmark",
        max_samples=50000,
        batch_size=128,
    )

    # 2. HC-PT K=8 Reranker
    r_hcpt_k8 = evaluate_k8(
        enc, lp, dec, guid, vocab,
        "InstaDeepAI/ms_proteometools",
        max_samples=50000,
        batch_size=128,
    )

    results = {
        "baseline_50k": {
            "ninespecies": {
                "exact_match_strict": 0.6777,
                "exact_match_il": 0.6794,
                "aa_f1": 0.8280,
                "throughput_spec_per_s": 200.5,
                "elapsed_seconds": 249.4,
            },
            "hcpt": {
                "exact_match_strict": 0.3487,
                "exact_match_il": 0.5623,
                "aa_f1": 0.6919,
                "throughput_spec_per_s": 227.8,
                "elapsed_seconds": 219.5,
            },
        },
        "k8_reranker_50k": {
            "ninespecies": r_ninespecies_k8,
            "hcpt": r_hcpt_k8,
        },
    }

    out_file = out_dir / "k8_reranker_50k_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
