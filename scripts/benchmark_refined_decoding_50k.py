#!/usr/bin/env python3
"""
Evaluates the length-weighted fine-tuned model with Refined Decoding:
1. Sequential Mass-Budget Resolution (exact one-by-one residual token knapsack updates)
2. Peak-Evidence Unmasking Schedule (boost confidence of peak-supported cleavage positions)
3. Detailed Balance Error Correction (eta=0.15)

Evaluates on:
- 50,000 HC-PT test spectra
- 50,000 Nine-Species test spectra
Stratified by length bins: [7-10], [11-14], [15-18], [19-22], [23-30].
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
        candidates.sort(
            key=lambda p: float(p.stem.split("exact=")[-1]) if "exact=" in p.stem else 0.0,
            reverse=True,
        )
        return candidates[0]
    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def evaluate_dataset(
    ckpt_path: Path,
    dataset_name: str,
    max_samples: int = 50000,
    batch_size: int = 256,
    eta: float = 0.15,
):
    print(f"\n=======================================================")
    print(f"Evaluating Refined Decoding on: {dataset_name}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Samples:    {max_samples:,} spectra")
    print(f"Config:     K=3, S=1, T=20, eta={eta}")
    print(f"Refinements: Sequential Knapsack=True, Peak Evidence=True, Detailed Balance=True")
    print(f"=======================================================")

    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()

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
        temperature=0.0,
        use_exact_dp_knapsack=True,
        use_sequential_knapsack=True,
        use_peak_evidence=True,
        eta=eta,
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
    print(f"\n=== Refined Decoding Length Stratification ({n:,} spectra) ===")
    print(f"{'Bin':<10} | {'Count':<7} | {'Strict Exact':<12} | {'I/L Exact':<12} | {'Residue F1':<10}")
    print("-" * 62)
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
        print(
            f"[{l_min:2d}-{l_max:2d}]     | {len(sub_p):<7} | {sub_m.exact_peptide_accuracy:<12.2%} | {sub_m.exact_peptide_accuracy_il:<12.2%} | {sub_m.aa_f1:<10.2%}"
        )

    print("-" * 62)
    print(
        f"{'Overall':<10} | {n:<7} | {metrics.exact_peptide_accuracy:<12.2%} | {metrics.exact_peptide_accuracy_il:<12.2%} | {metrics.aa_f1:<10.2%}"
    )
    print(f"Throughput: {throughput:.1f} spectra/sec (Elapsed: {dt/60:.2f} min)")

    res = {
        "dataset": dataset_name,
        "checkpoint": ckpt_path.name,
        "samples": n,
        "throughput_spectra_per_sec": round(throughput, 1),
        "elapsed_seconds": round(dt, 2),
        "overall": {
            "exact_strict": round(float(metrics.exact_peptide_accuracy), 4),
            "exact_il": round(float(metrics.exact_peptide_accuracy_il), 4),
            "aa_f1": round(float(metrics.aa_f1), 4),
            "aa_precision": round(float(metrics.aa_precision), 4),
            "aa_recall": round(float(metrics.aa_recall), 4),
        },
        "stratified": stratified,
    }
    return res


def main():
    ckpt_dir = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints"
    ckpt_path = find_best_checkpoint(ckpt_dir)
    out_dir = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. HC-PT 50k
    hcpt_res = evaluate_dataset(ckpt_path, "InstaDeepAI/ms_proteometools", max_samples=50000, batch_size=256, eta=0.15)
    with open(out_dir / "hcpt_50k_refined_decoding.json", "w") as f:
        json.dump(hcpt_res, f, indent=2)
    print(f"[Saved] {out_dir / 'hcpt_50k_refined_decoding.json'}")

    # 2. Nine-Species 50k
    ns_res = evaluate_dataset(ckpt_path, "InstaDeepAI/ms_ninespecies_benchmark", max_samples=50000, batch_size=256, eta=0.15)
    with open(out_dir / "ninespecies_50k_refined_decoding.json", "w") as f:
        json.dump(ns_res, f, indent=2)
    print(f"[Saved] {out_dir / 'ninespecies_50k_refined_decoding.json'}")

    print("\n================== BENCHMARK COMPLETED SUCCESSFULLY ==================")


if __name__ == "__main__":
    main()
