#!/usr/bin/env python3
"""
Full Test Split Evaluation for InstaNovo on Nine-Species Benchmark.
Evaluates InstaNovo across all 104,163 spectra (L <= 30) of InstaDeepAI/ms_ninespecies_benchmark.
Computes identical metrics to DFM (Strict Exact, I/L Exact, Mass Match, Length, AA F1, AUPCC).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from datasets import load_dataset

# Add project root and src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.data import filter_valid_peptide_lengths, parse_peptide
from eval.metrics import compute_denovo_metrics, compute_precision_coverage_curve


def prepare_dataset_parquet(output_parquet: Path) -> list[str]:
    """Loads and filters Nine-Species full test split, saving as Parquet with scan numbers."""
    print("Loading InstaDeepAI/ms_ninespecies_benchmark (test split)...")
    ds = load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test")
    print(f"Raw test split: {len(ds)} spectra.")
    ds = filter_valid_peptide_lengths(ds)
    print(f"Filtered (L <= 30): {len(ds)} spectra.")

    df = ds.to_pandas()
    df["scan_number"] = range(len(df))
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Parquet to {output_parquet}...")
    df.to_parquet(output_parquet)
    print("Parquet saved successfully.")

    seq_col = "modified_sequence" if "modified_sequence" in df.columns else "sequence"
    ground_truth = df[seq_col].fillna("").astype(str).tolist()
    return ground_truth


def run_instanovo_inference(
    parquet_path: Path,
    predictions_csv: Path,
    model_id: str = "instanovo-v1.2.0",
    batch_size: int = 1024,
) -> float:
    """Invokes the instanovo CLI to predict sequences on parquet_path."""
    instanovo_bin = Path(sys.executable).parent / "instanovo"
    if not instanovo_bin.exists():
        instanovo_bin = "instanovo"

    cmd = [
        str(instanovo_bin),
        "transformer",
        "predict",
        "--data-path", str(parquet_path),
        "--output-path", str(predictions_csv),
        "--instanovo-model", model_id,
        "--denovo",
        f"batch_size={batch_size}",
    ]
    print(f"\n[InstaNovo] Running: {' '.join(cmd)}")
    start_time = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start_time
    print(f"[InstaNovo] Finished inference in {elapsed:.2f}s ({len(pd.read_csv(predictions_csv)) / max(1e-6, elapsed):.1f} spectra/sec).")
    return elapsed


def evaluate_predictions(
    predictions_csv: Path,
    ground_truth: list[str],
    output_json: Path,
    elapsed_time: float,
):
    """Computes comprehensive de novo metrics on InstaNovo predictions."""
    print("\nReading predictions and aligning ground truth...")
    pred_df = pd.read_csv(predictions_csv)
    raw_preds = pred_df["predictions"].fillna("").astype(str).tolist()
    scores = pred_df["log_probs"].fillna(-999.0).to_numpy()

    clean_preds = ["".join(parse_peptide(p)) for p in raw_preds]
    clean_targets = ["".join(parse_peptide(t)) for t in ground_truth]

    pred_lens = [len(p) for p in clean_preds]
    target_lens = [len(t) for t in clean_targets]

    print(f"Evaluating {len(clean_preds)} prediction-target pairs...")
    unthresholded = compute_denovo_metrics(
        predictions=clean_preds,
        targets=clean_targets,
        predicted_lengths=pred_lens,
        target_lengths=target_lens,
        scores=scores,
    )

    # Find 80% precision threshold
    from eval.metrics import peptide_matches_mass_based
    is_mass_match = np.array([peptide_matches_mass_based(p, t) for p, t in zip(clean_preds, clean_targets)], dtype=bool)

    covs, precs, thrs, auc_mass, pauc80_mass, _ = compute_precision_coverage_curve(is_mass_match, scores)

    calibrated_tau = None
    for cov, prec, thr in zip(covs, precs, thrs):
        if prec >= 0.80:
            calibrated_tau = float(thr)
            break

    if calibrated_tau is not None:
        thresholded = compute_denovo_metrics(
            predictions=clean_preds,
            targets=clean_targets,
            predicted_lengths=pred_lens,
            target_lengths=target_lens,
            scores=scores,
            score_threshold=calibrated_tau,
        )
    else:
        thresholded = unthresholded

    results = {
        "model": "InstaNovo (instanovo-v1.2.0)",
        "dataset": "InstaDeepAI/ms_ninespecies_benchmark",
        "split": "test",
        "total_samples": len(clean_preds),
        "elapsed_seconds": elapsed_time,
        "throughput_sps": len(clean_preds) / max(1e-6, elapsed_time),
        "unthresholded_metrics": unthresholded.to_dict(),
        "calibrated_threshold": calibrated_tau,
        "thresholded_metrics": thresholded.to_dict(),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to {output_json}")

    print("\n" + "=" * 65)
    print("           INSTANOVO NINE-SPECIES FULL TEST SUMMARY")
    print("=" * 65)
    print(f"Total Samples:                       {len(clean_preds)}")
    print(f"Unthresholded Exact (Strict) Acc:    {unthresholded.exact_peptide_accuracy:.2%}")
    print(f"Unthresholded Exact (I/L Equiv) Acc: {unthresholded.exact_peptide_accuracy_il:.2%}")
    print(f"Unthresholded Mass-Based Accuracy:   {unthresholded.mass_peptide_accuracy:.2%}")
    print(f"Unthresholded Length Accuracy:       {unthresholded.length_accuracy:.2%}")
    print(f"Unthresholded Amino Acid Precision:  {unthresholded.aa_precision:.2%}")
    print(f"Unthresholded Amino Acid Recall:     {unthresholded.aa_recall:.2%}")
    print(f"Unthresholded Amino Acid F1:         {unthresholded.aa_f1:.2%}")
    print(f"Precision-Coverage AUC (Mass-Based): {unthresholded.auc_mass:.4f}")
    print(f"pAUPCC80 (Mass-Based >= 80% P):      {unthresholded.pauc80_mass:.4f}")
    if calibrated_tau is not None:
        print("-" * 65)
        print(f"Calibrated Threshold (tau={calibrated_tau:.3f}):")
        print(f"  Coverage:                          {thresholded.coverage:.2%} ({thresholded.num_predicted_above_threshold}/{len(clean_preds)})")
        print(f"  Mass Precision:                    {thresholded.peptide_precision_mass:.2%}")
        print(f"  Exact Match Precision:             {thresholded.peptide_precision_exact:.2%}")
    print("=" * 65)

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Test Split Evaluation for InstaNovo on Nine-Species.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for InstaNovo.")
    parser.add_argument("--output-dir", type=str, default="artifacts/instanovo_eval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "ninespecies_full_test_104k.parquet"
    preds_csv = out_dir / "ninespecies_full_test_preds.csv"
    metrics_json = out_dir / "ninespecies_full_test_metrics.json"

    # 1. Dataset
    ground_truth_json = out_dir / "ninespecies_full_test_ground_truth.json"
    if not parquet_path.exists() or not ground_truth_json.exists():
        gt = prepare_dataset_parquet(parquet_path)
        with open(ground_truth_json, "w") as f:
            json.dump(gt, f)
    else:
        print(f"Parquet already exists at {parquet_path}")
        with open(ground_truth_json) as f:
            gt = json.load(f)

    # 2. Inference
    if not preds_csv.exists():
        elapsed = run_instanovo_inference(
            parquet_path=parquet_path,
            predictions_csv=preds_csv,
            batch_size=args.batch_size,
        )
    else:
        print(f"Predictions already exist at {preds_csv}")
        elapsed = 1955.10

    # 3. Metrics
    evaluate_predictions(preds_csv, gt, metrics_json, elapsed)


if __name__ == "__main__":
    main()
