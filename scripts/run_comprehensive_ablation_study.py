#!/usr/bin/env python3
"""
Comprehensive Ablation Study for DFlowNovo.

Ablates:
1. Decoding Components:
   - C0: Raw Flow Baseline (K=1, no knapsack, no priors, no ladders)
   - C1: + Exact DP Knapsack Filtering
   - C2: + Multi-Length Beam (K=3)
   - C3: + Trypsin Cleavage Prior
   - C4: + Composite Fragment Ladder Scoring
   - C5: + Peak Evidence & Detailed Balance (Full Refined DFlowNovo)
2. Flow Dynamics & Scheduler:
   - Steps T in {5, 10, 20}
   - Scheduler: Cosine vs. Linear
   - Strategy: Confidence vs. Random unmasking
3. Training & Loss Formulation:
   - Scratch 30ep vs. Pre-trained + Length-Weighted Fine-Tuned
"""

import argparse
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


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = load_checkpoint(str(ckpt_path), map_location=device)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, device)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()
    return enc, lp, dec, guid, vocab


def run_ablation(max_samples: int = 2000, batch_size: int = 256, output_file: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ablation Study on {device} with max_samples={max_samples}")

    ckpt_best = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints" / "best-joint-gen-exact-epoch=01-exact=0.4746.ckpt"
    ckpt_scratch_30ep = PROJECT_ROOT / "artifacts" / "dfm_ninespecies_pure_30ep" / "dfm_ninespecies_pure_30ep" / "checkpoints" / "best-gen-exact-epoch=19-exact=0.1379.ckpt"

    print(f"Loading Main Model: {ckpt_best.name}")
    enc, lp, dec, guid, vocab = load_model(ckpt_best, device)

    ds = get_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    results = {
        "metadata": {
            "max_samples": max_samples,
            "dataset": "InstaDeepAI/ms_ninespecies_benchmark",
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "component_ablation": {},
        "flow_dynamics_ablation": {},
        "training_formulation_ablation": {},
    }

    # =========================================================================
    # 1. COMPONENT ABLATION
    # =========================================================================
    configs_components = [
        (
            "C0_Raw_Flow_Baseline",
            "Raw discrete flow matching reverse CTMC (K=1, no knapsack, no ladders, no priors)",
            dict(
                top_k_lengths=1,
                use_knapsack_filter=False,
                use_exact_dp_knapsack=False,
                trypsin_prior=False,
                use_composite_ladders=False,
                use_peak_evidence=False,
                use_sequential_knapsack=False,
                eta=0.0,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "C1_Plus_Exact_DP_Knapsack",
            "+ Vectorized Exact DP Knapsack reachability filtering",
            dict(
                top_k_lengths=1,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=False,
                use_composite_ladders=False,
                use_peak_evidence=False,
                use_sequential_knapsack=False,
                eta=0.0,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "C2_Plus_Multi_Length_Beam",
            "+ Top-3 Multi-length beam candidate generation",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=False,
                use_composite_ladders=False,
                use_peak_evidence=False,
                use_sequential_knapsack=False,
                eta=0.0,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "C3_Plus_Trypsin_Prior",
            "+ Enzymatic C-terminal cleavage prior bonus (K/R)",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=False,
                use_peak_evidence=False,
                use_sequential_knapsack=False,
                eta=0.0,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "C4_Plus_Composite_Ladders",
            "+ Multi-feature composite b/y fragment ladder rescoring",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=False,
                use_sequential_knapsack=False,
                eta=0.0,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "C5_Full_Refined_DFlowNovo",
            "+ Peak evidence schedule, detailed balance (eta=0.15) & sequential knapsack",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=True,
                use_sequential_knapsack=True,
                eta=0.15,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
    ]

    print("\n--- Running Part 1: Component Ablation ---")
    for name, desc, kwargs in configs_components:
        t0 = time.time()
        metrics = evaluate_generative(
            loader, vocab, enc, lp, dec, guid,
            device=device, max_samples=max_samples,
            **kwargs
        )
        elapsed = time.time() - t0
        speed = max_samples / elapsed
        entry = {
            "description": desc,
            "exact_strict": round(float(metrics.exact_peptide_accuracy) * 100, 2),
            "exact_il": round(float(metrics.exact_peptide_accuracy_il) * 100, 2),
            "aa_f1": round(float(metrics.aa_f1) * 100, 2),
            "length_accuracy": round(float(metrics.length_accuracy) * 100, 2),
            "throughput_spectra_per_sec": round(speed, 1),
            "elapsed_seconds": round(elapsed, 2),
        }
        results["component_ablation"][name] = entry
        print(f"[{name}] Strict={entry['exact_strict']}%, I/L={entry['exact_il']}%, AAF1={entry['aa_f1']}%, LenAcc={entry['length_accuracy']}%, Speed={entry['throughput_spectra_per_sec']} sp/s")

    # =========================================================================
    # 2. FLOW DYNAMICS & SCHEDULING ABLATION
    # =========================================================================
    configs_dynamics = [
        (
            "Steps_T5",
            "5 Flow Matching Steps (Fast inference)",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=True,
                use_sequential_knapsack=True,
                eta=0.15,
                num_steps=5,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "Steps_T10",
            "10 Flow Matching Steps",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=True,
                use_sequential_knapsack=True,
                eta=0.15,
                num_steps=10,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "Steps_T20_Default",
            "20 Flow Matching Steps (Default)",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=True,
                use_sequential_knapsack=True,
                eta=0.15,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="confidence",
            ),
        ),
        (
            "Scheduler_Linear",
            "Linear schedule instead of Cosine schedule",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=True,
                use_sequential_knapsack=True,
                eta=0.15,
                num_steps=20,
                scheduler="linear",
                decoding_strategy="confidence",
            ),
        ),
        (
            "Strategy_Random",
            "Random unmasking instead of Confidence-based unmasking",
            dict(
                top_k_lengths=3,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                trypsin_prior=True,
                use_composite_ladders=True,
                use_peak_evidence=True,
                use_sequential_knapsack=True,
                eta=0.15,
                num_steps=20,
                scheduler="cosine",
                decoding_strategy="random",
            ),
        ),
    ]

    print("\n--- Running Part 2: Flow Dynamics Ablation ---")
    for name, desc, kwargs in configs_dynamics:
        t0 = time.time()
        metrics = evaluate_generative(
            loader, vocab, enc, lp, dec, guid,
            device=device, max_samples=max_samples,
            **kwargs
        )
        elapsed = time.time() - t0
        speed = max_samples / elapsed
        entry = {
            "description": desc,
            "exact_strict": round(float(metrics.exact_peptide_accuracy) * 100, 2),
            "exact_il": round(float(metrics.exact_peptide_accuracy_il) * 100, 2),
            "aa_f1": round(float(metrics.aa_f1) * 100, 2),
            "length_accuracy": round(float(metrics.length_accuracy) * 100, 2),
            "throughput_spectra_per_sec": round(speed, 1),
            "elapsed_seconds": round(elapsed, 2),
        }
        results["flow_dynamics_ablation"][name] = entry
        print(f"[{name}] Strict={entry['exact_strict']}%, I/L={entry['exact_il']}%, AAF1={entry['aa_f1']}%, LenAcc={entry['length_accuracy']}%, Speed={entry['throughput_spectra_per_sec']} sp/s")

    # =========================================================================
    # 3. TRAINING FORMULATION & MODEL CAPACITY ABLATION
    # =========================================================================
    print("\n--- Running Part 3: Training Formulation Ablation ---")
    results["training_formulation_ablation"]["DFlowNovo_Length_Weighted_FineTuned"] = results["component_ablation"]["C5_Full_Refined_DFlowNovo"]

    if ckpt_scratch_30ep.exists():
        print(f"Loading Scratch 30ep Model: {ckpt_scratch_30ep.name}")
        enc_s, lp_s, dec_s, guid_s, vocab_s = load_model(ckpt_scratch_30ep, device)
        t0 = time.time()
        metrics_s = evaluate_generative(
            loader, vocab_s, enc_s, lp_s, dec_s, guid_s,
            device=device, max_samples=max_samples,
            top_k_lengths=3,
            use_knapsack_filter=True,
            use_exact_dp_knapsack=True,
            trypsin_prior=True,
            use_composite_ladders=True,
            use_peak_evidence=True,
            use_sequential_knapsack=True,
            eta=0.15,
            num_steps=20,
            scheduler="cosine",
            decoding_strategy="confidence",
        )
        elapsed_s = time.time() - t0
        results["training_formulation_ablation"]["DFlowNovo_Scratch_30ep_Pure"] = {
            "description": "30-Epoch scratch training without pre-training on ProteomeTools",
            "exact_strict": round(float(metrics_s.exact_peptide_accuracy) * 100, 2),
            "exact_il": round(float(metrics_s.exact_peptide_accuracy_il) * 100, 2),
            "aa_f1": round(float(metrics_s.aa_f1) * 100, 2),
            "length_accuracy": round(float(metrics_s.length_accuracy) * 100, 2),
            "throughput_spectra_per_sec": round(max_samples / elapsed_s, 1),
            "elapsed_seconds": round(elapsed_s, 2),
        }
        print(f"[DFlowNovo_Scratch_30ep_Pure] Strict={results['training_formulation_ablation']['DFlowNovo_Scratch_30ep_Pure']['exact_strict']}%, I/L={results['training_formulation_ablation']['DFlowNovo_Scratch_30ep_Pure']['exact_il']}%, AAF1={results['training_formulation_ablation']['DFlowNovo_Scratch_30ep_Pure']['aa_f1']}%")

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Ablation Results Saved] -> {out_p}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-file", type=str, default="artifacts/ablation_study_results.json")
    args = parser.parse_args()

    run_ablation(max_samples=args.max_samples, batch_size=args.batch_size, output_file=args.output_file)
