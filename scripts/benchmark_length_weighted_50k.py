#!/usr/bin/env python3
"""
Evaluates the length-weighted fine-tuned model on:
1. 50,000 HC-PT test spectra
2. 50,000 Nine-Species test spectra
3. Full length-stratified breakdown across [7-10], [11-14], [15-18], [19-22], [23-30].
4. Compares against the baseline model.
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
from eval.metrics import compute_denovo_metrics
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = list(checkpoint_dir.glob("best-joint-gen-exact-epoch=*.ckpt"))
    if candidates:
        candidates.sort(key=lambda p: float(p.stem.split("exact=")[-1]) if "exact=" in p.stem else 0.0, reverse=True)
        return candidates[0]
    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def evaluate_dataset(ckpt_path: Path, dataset_name: str, max_samples: int = 50000, batch_size: int = 256):
    print(f"\n=======================================================")
    print(f"Evaluating: {ckpt_path.name}")
    print(f"Dataset:    {dataset_name} ({max_samples:,} spectra)")
    print(f"Config:     K=3, S=1, T=20 (Fast SOTA inference)")
    print(f"=======================================================")

    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval(); lp.eval(); dec.eval(); guid.eval()

    ds = get_dataset(dataset_name, split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
        top_k_lengths=3,
        num_samples_per_length=1,
        use_exact_dp_knapsack=True,
        return_details=True,
    )
    dt = time.time() - t0

    preds = details["predictions"]
    targets = details["targets"]
    lengths = [len(t) for t in targets]
    n = len(preds)
    throughput = n / dt

    # Stratified by length
    bins = [(7, 10), (11, 14), (15, 18), (19, 22), (23, 30)]
    stratified = {}
    print(f"\n=== Length Stratification ({n:,} spectra) ===")
    for l_min, l_max in bins:
        sub_p = [p for p, t, l in zip(preds, targets, lengths) if l_min <= l <= l_max]
        sub_t = [t for p, t, l in zip(preds, targets, lengths) if l_min <= l <= l_max]
        if not sub_p:
            continue
        sub_m = compute_denovo_metrics(sub_p, sub_t)
        stratified[f"L[{l_min}-{l_max}]"] = {
            "count": len(sub_p),
            "exact_strict": round(float(sub_m.exact_peptide_accuracy), 4),
            "exact_il": round(float(sub_m.exact_peptide_accuracy_il), 4),
            "aa_f1": round(float(sub_m.aa_f1), 4),
        }
        print(f"Length [{l_min:2d}-{l_max:2d}] (N={len(sub_p):5d}): Strict={sub_m.exact_peptide_accuracy:.1%}, I/L={sub_m.exact_peptide_accuracy_il:.1%}, AA-F1={sub_m.aa_f1:.1%}")

    res = {
        "dataset": dataset_name,
        "checkpoint": ckpt_path.name,
        "num_samples": n,
        "exact_match_strict": round(float(metrics.exact_peptide_accuracy), 4),
        "exact_match_il": round(float(metrics.exact_peptide_accuracy_il), 4),
        "aa_precision": round(float(metrics.aa_precision), 4),
        "aa_recall": round(float(metrics.aa_recall), 4),
        "aa_f1": round(float(metrics.aa_f1), 4),
        "length_accuracy": round(float(metrics.length_accuracy), 4),
        "elapsed_seconds": round(dt, 2),
        "throughput_spec_per_s": round(throughput, 1),
        "stratified_by_length": stratified,
    }

    print(f"\nOverall Summary for {dataset_name}:")
    print(f"  Strict Exact Match: {res['exact_match_strict']:.2%}")
    print(f"  I/L Exact Match:    {res['exact_match_il']:.2%}")
    print(f"  Residue F1:         {res['aa_f1']:.2%}")
    print(f"  Throughput:         {res['throughput_spec_per_s']:.1f} spec/s")
    return res


def main():
    ckpt_dir = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints"
    ckpt_path = find_best_checkpoint(ckpt_dir)
    print(f"Found best fine-tuned checkpoint: {ckpt_path}")

    # 1. 50k HC-PT
    results_hcpt = evaluate_dataset(ckpt_path, "InstaDeepAI/ms_proteometools", max_samples=50000, batch_size=256)
    out_hcpt = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "hcpt_50k_evaluation.json"
    with open(out_hcpt, "w") as f:
        json.dump(results_hcpt, f, indent=2)

    # 2. 50k Nine-Species
    results_ns = evaluate_dataset(ckpt_path, "InstaDeepAI/ms_ninespecies_benchmark", max_samples=50000, batch_size=256)
    out_ns = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "ninespecies_50k_evaluation.json"
    with open(out_ns, "w") as f:
        json.dump(results_ns, f, indent=2)

    print(f"\nAll benchmark evaluations complete!")
    print(f"HC-PT results:        {out_hcpt}")
    print(f"Nine-Species results: {out_ns}")


if __name__ == "__main__":
    main()
