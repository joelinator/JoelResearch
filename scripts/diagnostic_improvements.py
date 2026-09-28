#!/usr/bin/env python3
"""
Diagnostic test script to explore potential improvements:
1. Candidate sampling diversity: K=3, S=1 vs K=3, S=2 (temperature=0.7) with composite fragment ladder reranking.
2. Human amino acid background frequency prior for I/L disambiguation.
3. Evaluates on 2,000 HC-PT spectra to measure exact match and latency deltas.
"""

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

def run_diagnostic():
    ckpt_path = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints" / "best-joint-gen-exact-epoch=01-exact=0.4746.ckpt"
    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval(); lp.eval(); dec.eval(); guid.eval()

    ds = get_dataset("InstaDeepAI/ms_proteometools", split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    max_samples = 3000

    print("--- 1. Baseline greedy: K=3, S=1, temp=0.0 ---")
    t0 = time.time()
    m1, d1 = evaluate_generative(
        loader, vocab, enc, lp, dec, guid,
        scheduler="cosine", device=DEVICE,
        max_samples=max_samples, num_steps=20,
        top_k_lengths=3, num_samples_per_length=1,
        temperature=0.0, use_exact_dp_knapsack=True,
        return_details=True,
    )
    dt1 = time.time() - t0
    p1, t1 = d1["predictions"], d1["targets"]
    print(f"K=3, S=1: Strict={m1.exact_peptide_accuracy:.2%}, I/L={m1.exact_peptide_accuracy_il:.2%}, AA-F1={m1.aa_f1:.2%}, Throughput={len(p1)/dt1:.1f} spec/s")

    print("\n--- 2. Stochastic diversity: K=3, S=2, temp=0.7 with composite ladders ---")
    t0 = time.time()
    m2, d2 = evaluate_generative(
        loader, vocab, enc, lp, dec, guid,
        scheduler="cosine", device=DEVICE,
        max_samples=max_samples, num_steps=20,
        top_k_lengths=3, num_samples_per_length=2,
        temperature=0.7, use_exact_dp_knapsack=True,
        use_composite_ladders=True,
        return_details=True,
    )
    dt2 = time.time() - t0
    p2, t2 = d2["predictions"], d2["targets"]
    print(f"K=3, S=2: Strict={m2.exact_peptide_accuracy:.2%}, I/L={m2.exact_peptide_accuracy_il:.2%}, AA-F1={m2.aa_f1:.2%}, Throughput={len(p2)/dt2:.1f} spec/s")

    # 3. Simulate Leucine preference for I/L disambiguation
    # In human tryptic peptides, Leucine is ~1.8x more frequent than Isoleucine (9.9% vs 5.5%)
    # Let's see what happens to strict exact match if we map ambiguous I/L to L
    print("\n--- 3. Testing Proteome Frequency Prior (Leucine default for I/L) ---")
    p1_L_prior = [p.replace("I", "L") for p in p1]
    t1_L_truth = [t.replace("I", "L") for t in t1]  # this is just I/L conflated
    # But what if ground truth has true I and L?
    # How often does p1 match t1 when p1 has an I/L swap?
    correct_strict = sum(1 for p, t in zip(p1, t1) if p == t)
    correct_il = sum(1 for p, t in zip(p1, t1) if p.replace("I", "L") == t.replace("I", "L"))
    print(f"Of {len(p1)} spectra, strict matches: {correct_strict} ({correct_strict/len(p1):.2%}), I/L matches: {correct_il} ({correct_il/len(p1):.2%})")
    print(f"Isobaric gap: {correct_il - correct_strict} sequences are exact up to I/L distinction.")

if __name__ == "__main__":
    run_diagnostic()
