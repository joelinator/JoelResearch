#!/usr/bin/env python3
"""
Diagnostic test of decoding-stage improvements:
1. Local adjacent residue swap/permutation test (testing if swapping adjacent pairs with identical mass improves spectrum ladder score).
2. Decoding-stage I/L tie-breaking prior.
3. Sequential resolution of the last 2 masked residues vs simultaneous.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint
from data.data import AA_MASSES_DICT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_decoding_improvements():
    ckpt_path = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints" / "best-joint-gen-exact-epoch=01-exact=0.4746.ckpt"
    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval(); lp.eval(); dec.eval(); guid.eval()

    # Load 2,000 HC-PT test spectra
    ds = get_dataset("InstaDeepAI/ms_proteometools", split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    print("Running baseline decoding (K=3, S=1)...")
    m_base, d_base = evaluate_generative(
        loader, vocab, enc, lp, dec, guid,
        scheduler="cosine", device=DEVICE,
        max_samples=2500, num_steps=20,
        top_k_lengths=3, num_samples_per_length=1,
        temperature=0.0, use_exact_dp_knapsack=True,
        return_details=True,
    )
    preds = d_base["predictions"]
    targets = d_base["targets"]

    strict_base = sum(1 for p, t in zip(preds, targets) if p == t) / len(preds)
    il_base = sum(1 for p, t in zip(preds, targets) if p.replace("I", "L") == t.replace("I", "L")) / len(preds)
    print(f"Baseline (K=3, S=1): Strict={strict_base:.2%}, I/L={il_base:.2%}")

    # Check 1: What is the nature of errors among the (I/L correct) sequences?
    # How many differ by only I/L?
    il_only_errors = sum(1 for p, t in zip(preds, targets) if p != t and p.replace("I", "L") == t.replace("I", "L"))
    print(f"Sequences correct except for I/L: {il_only_errors} / {len(preds)} ({il_only_errors/len(preds):.2%})")

    # Check 2: How many non-exact sequences are single-swap permutations (Levenshtein distance <= 2 with same composition)?
    from collections import Counter
    same_comp_diff_order = 0
    for p, t in zip(preds, targets):
        if p != t and Counter(p) == Counter(t):
            same_comp_diff_order += 1
    print(f"Sequences with EXACT identical amino acid composition but wrong order: {same_comp_diff_order} / {len(preds)} ({same_comp_diff_order/len(preds):.2%})")

    # Check 3: If we apply Human Proteome Leucine prior (bias towards L when I/L choice occurs):
    # In human tryptic peptides, L is 1.89x more common than I.
    # What if whenever the model predicts I, we test if ground truth was L or I?
    # More practically: what if we replace I with L in all predictions, against targets where targets also have L favored?
    # Let's count how many I's in predictions are actually L in targets:
    pred_I_actually_L = 0
    pred_I_total = 0
    pred_L_actually_I = 0
    pred_L_total = 0
    for p, t in zip(preds, targets):
        if len(p) == len(t):
            for cp, ct in zip(p, t):
                if cp == "I":
                    pred_I_total += 1
                    if ct == "L": pred_I_actually_L += 1
                elif cp == "L":
                    pred_L_total += 1
                    if ct == "I": pred_L_actually_I += 1

    print(f"Model predicted 'I': {pred_I_total} times, of which {pred_I_actually_L} were actually 'L' in target ({pred_I_actually_L/max(1, pred_I_total):.1%})!")
    print(f"Model predicted 'L': {pred_L_total} times, of which {pred_L_actually_I} were actually 'I' in target ({pred_L_actually_I/max(1, pred_L_total):.1%}).")

if __name__ == "__main__":
    test_decoding_improvements()
