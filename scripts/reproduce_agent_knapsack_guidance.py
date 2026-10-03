#!/usr/bin/env python3
"""
Scientific Reproduction Harness: Knapsack Guidance vs Hard Filter (from logs_agents.txt).

In logs_agents.txt, an agent squad claimed:
"Baseline (Hard Filter): 51.21% -> Dynamic Guidance (Strength 5.0): 53.83% (+2.62% exact sequence match)"
However, forensics revealed that the tool actually crashed with ModuleNotFoundError: No module named 'torch',
and the agent hallucinated the entire benchmark run, fake git commit 4a1b3c2, and fake repo push.

This harness empirically tests the mathematical concept:
Does penalizing unreachable tokens with a soft penalty (-guidance_strength) instead of hard masking (-1e9)
actually improve exact sequence matching on real spectra?
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
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 85)
    print("SCIENTIFIC REPRODUCTION EXPERIMENT: KNAPSACK GUIDANCE VS HARD FILTER")
    print("Agent Squad Claim from logs_agents.txt:")
    print("  'Baseline (Hard Filter) 51.21% -> Dynamic Guidance (Strength 5.0) 53.83% (+2.62%)'")
    print("=" * 85)

    ckpt_path = "artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()

    print("Building Nine-Species test dataloader...")
    ds = get_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    N_EVAL = 2000
    print(f"Evaluating {N_EVAL} test spectra per condition on {torch.cuda.get_device_name(0)}...")

    conditions = [
        ("Hard Knapsack Filter (Baseline -1e9)", True, True, None),
        ("Soft Guidance (Strength=5.0 - Agent Claim)", True, True, 5.0),
        ("Soft Guidance (Strength=10.0)", True, True, 10.0),
        ("Soft Guidance (Strength=2.0)", True, True, 2.0),
        ("No Knapsack Guidance (Strength=0.0)", False, False, None),
    ]

    results = {}
    for name, use_filter, use_seq, penalty in conditions:
        print(f"\n---> Running: {name}...")
        t0 = time.time()
        metrics, details = evaluate_generative(
            loader=loader,
            vocabulary=vocab,
            spectrum_encoder=enc,
            length_predictor=lp,
            decoder=dec,
            guidance=guid,
            scheduler="cosine",
            device=DEVICE,
            max_samples=N_EVAL,
            num_steps=20,
            top_k_lengths=3,
            num_samples_per_length=1,
            temperature=0.0,
            use_knapsack_filter=use_filter,
            use_exact_dp_knapsack=use_filter,
            use_sequential_knapsack=use_seq,
            knapsack_guidance_penalty=penalty,
            return_details=True,
        )
        dt = time.time() - t0
        speed = N_EVAL / dt

        res = {
            "strict_exact": float(metrics.exact_peptide_accuracy) * 100.0,
            "il_exact": float(metrics.exact_peptide_accuracy_il) * 100.0,
            "aa_precision": float(metrics.aa_precision) * 100.0,
            "aa_recall": float(metrics.aa_recall) * 100.0,
            "aa_f1": float(metrics.aa_f1) * 100.0,
            "speed": speed,
            "time": dt,
        }
        results[name] = res
        print(f"     Strict Exact: {res['strict_exact']:.2f}% | I/L Exact: {res['il_exact']:.2f}% | AA F1: {res['aa_f1']:.2f}% | Speed: {res['speed']:.1f} spec/s")

    print("\n" + "=" * 90)
    print("REPRODUCTION SUMMARY TABLE (N = 2,000 Nine-Species Test Spectra)")
    print("=" * 90)
    print(f"{'Condition':<44} | {'Exact (%)':<10} | {'I/L Exact (%)':<14} | {'AA F1 (%)':<10} | {'Throughput':<12}")
    print("-" * 90)
    baseline_exact = results["Hard Knapsack Filter (Baseline -1e9)"]["strict_exact"]
    for name, res in results.items():
        delta = res["strict_exact"] - baseline_exact
        sign = "+" if delta >= 0 else ""
        delta_str = f"({sign}{delta:.2f}%)" if name != "Hard Knapsack Filter (Baseline -1e9)" else "(baseline)"
        print(f"{name:<44} | {res['strict_exact']:>6.2f}% {delta_str:<9} | {res['il_exact']:>6.2f}%        | {res['aa_f1']:>6.2f}%   | {res['speed']:>5.1f} spec/s")
    print("=" * 90)


if __name__ == "__main__":
    main()
