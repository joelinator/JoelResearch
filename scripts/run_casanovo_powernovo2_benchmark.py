#!/usr/bin/env python3
"""
Executes full benchmark evaluation for Casanovo (v5.2.1) and PowerNovo2 on:
1. Nine-Species test split (104,163 spectra)
2. HC-PT 50k test split (50,000 spectra)
Applies identical standardized metric computation (src/eval/metrics.py)
and saves predictions and JSON metrics to artifacts/benchmark_external/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval.metrics import compute_denovo_metrics, DenovoMetrics


def metrics_to_dict(m: DenovoMetrics) -> dict:
    return {
        "num_samples": m.num_samples,
        "aa_precision": float(m.aa_precision),
        "aa_recall": float(m.aa_recall),
        "aa_f1": float(m.aa_f1),
        "aa_error_rate": float(m.aa_error_rate),
        "length_accuracy": float(m.length_accuracy),
        "peptide_precision_exact": float(m.peptide_precision_exact),
        "peptide_recall_exact": float(m.peptide_recall_exact),
        "peptide_f1_exact": float(m.peptide_f1_exact),
        "peptide_precision_exact_il": float(m.peptide_precision_exact_il),
        "peptide_recall_exact_il": float(m.peptide_recall_exact_il),
        "peptide_f1_exact_il": float(m.peptide_f1_exact_il),
        "peptide_precision_mass": float(m.peptide_precision_mass),
        "peptide_recall_mass": float(m.peptide_recall_mass),
        "peptide_f1_mass": float(m.peptide_f1_mass),
        "coverage": float(m.coverage),
    }


def run_casanovo_split(
    mgf_path: Path,
    gt_path: Path,
    out_dir: Path,
    prefix: str,
    predictions_csv: Path,
    metrics_json: Path,
) -> dict:
    print(f"\n==========================================")
    print(f"RUNNING CASANOVO on {mgf_path.name}")
    print(f"==========================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Casanovo CLI
    casanovo_bin = PROJECT_ROOT / ".venv" / "bin" / "casanovo"
    cmd = [
        str(casanovo_bin),
        "sequence",
        str(mgf_path),
        "--output_dir", str(out_dir),
        "-o", prefix,
        "-f",
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    
    # Locate mztab
    mztabs = list(out_dir.glob(f"{prefix}*.mztab"))
    if not mztabs:
        raise FileNotFoundError(f"No mztab file found matching {prefix}*.mztab in {out_dir}")
    mztab_file = mztabs[0]
    print(f"Casanovo inference completed in {elapsed:.2f}s. Parsing {mztab_file}...")

    # Load ground truth
    gt_df = pd.read_parquet(gt_path)
    n_total = len(gt_df)
    gt_seqs = gt_df["modified_sequence"].fillna("").tolist()

    # Parse mztab: Find the line where PSM section starts
    psm_start = 0
    with open(mztab_file, "r") as f:
        for idx, line in enumerate(f):
            if line.startswith("PSH"):
                psm_start = idx
                break

    tab = pd.read_csv(mztab_file, sep="\t", skiprows=psm_start)
    preds_dict = {}
    scores_dict = {}
    
    for _, row in tab.iterrows():
        ref = str(row.get("spectra_ref", ""))
        m = re.search(r"index=(\d+)", ref)
        if m:
            s_idx = int(m.group(1))
            # Prefer proforma or sequence
            seq = row.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
            if pd.isna(seq) or not str(seq).strip():
                seq = row.get("sequence")
            score = row.get("search_engine_score[1]", 0.0)
            preds_dict[s_idx] = str(seq) if pd.notna(seq) else ""
            scores_dict[s_idx] = float(score) if pd.notna(score) else 0.0

    aligned_preds = [preds_dict.get(i, "") for i in range(n_total)]
    aligned_scores = [scores_dict.get(i, 0.0) for i in range(n_total)]

    # Compute standardized metrics
    print(f"Evaluating {n_total} spectra against ground truth with compute_denovo_metrics...")
    m = compute_denovo_metrics(aligned_preds, gt_seqs)
    m_dict = metrics_to_dict(m)
    m_dict["elapsed_seconds"] = elapsed
    m_dict["throughput_spec_per_sec"] = n_total / elapsed if elapsed > 0 else 0.0

    # Save predictions
    pred_df = pd.DataFrame({
        "scan_id": range(n_total),
        "ground_truth": gt_seqs,
        "predicted_sequence": aligned_preds,
        "score": aligned_scores,
    })
    pred_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    # Save metrics JSON
    with open(metrics_json, "w") as f:
        json.dump(m_dict, f, indent=2)
    print(f"Saved metrics to {metrics_json}")

    print("\n--- CASANOVO RESULTS ---")
    print(f"Strict Exact Match Precision: {m.peptide_precision_exact*100:.2f}%")
    print(f"I/L Exact Match Precision:    {m.peptide_precision_exact_il*100:.2f}%")
    print(f"Residue F1:                   {m.aa_f1*100:.2f}% (Prec: {m.aa_precision*100:.2f}%, Rec: {m.aa_recall*100:.2f}%)")
    print(f"Throughput:                   {m_dict['throughput_spec_per_sec']:.2f} spectra/sec")
    return m_dict


def run_powernovo2_split(
    mgf_path: Path,
    gt_path: Path,
    work_dir: Path,
    out_dir: Path,
    predictions_csv: Path,
    metrics_json: Path,
    batch_size: int = 64,
) -> dict:
    print(f"\n==========================================")
    print(f"RUNNING POWERNOVO2 on {mgf_path.name}")
    print(f"==========================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    denovo_py = PROJECT_ROOT / "scratch" / "PowerNovo2" / "powernovo2" / "denovo.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "scratch" / "PowerNovo2")

    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(denovo_py),
        str(mgf_path),
        "-w", str(work_dir),
        "-o", str(out_dir),
        "-b", str(batch_size),
    ]

    print(f"Executing: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    elapsed = time.time() - t0

    # Locate predictions CSV
    stem = mgf_path.stem
    pwn_csv = out_dir / stem / f"{stem}_denovo.csv"
    if not pwn_csv.exists():
        # Fallback search
        cands = list(out_dir.glob("**/*_denovo.csv")) + list(Path("/dev/shm").glob(f"**/{stem}_denovo.csv"))
        if not cands:
            raise FileNotFoundError(f"No *_denovo.csv found for {stem} in {out_dir} or /dev/shm")
        pwn_csv = cands[0]

    print(f"PowerNovo2 inference completed in {elapsed:.2f}s. Parsing {pwn_csv}...")

    # Load ground truth
    gt_df = pd.read_parquet(gt_path)
    n_total = len(gt_df)
    gt_seqs = gt_df["modified_sequence"].fillna("").tolist()

    res_df = pd.read_csv(pwn_csv)
    preds_dict = {}
    scores_dict = {}
    for _, row in res_df.iterrows():
        s_idx = int(row["SCAN ID"])
        seq = str(row["PEPTIDE"]) if pd.notna(row["PEPTIDE"]) else ""
        score = float(row["SCORE"]) if pd.notna(row["SCORE"]) else 0.0
        preds_dict[s_idx] = seq
        scores_dict[s_idx] = score

    aligned_preds = [preds_dict.get(i, "") for i in range(n_total)]
    aligned_scores = [scores_dict.get(i, 0.0) for i in range(n_total)]

    # Compute standardized metrics
    print(f"Evaluating {n_total} spectra against ground truth with compute_denovo_metrics...")
    m = compute_denovo_metrics(aligned_preds, gt_seqs)
    m_dict = metrics_to_dict(m)
    m_dict["elapsed_seconds"] = elapsed
    m_dict["throughput_spec_per_sec"] = n_total / elapsed if elapsed > 0 else 0.0

    # Save predictions
    pred_df = pd.DataFrame({
        "scan_id": range(n_total),
        "ground_truth": gt_seqs,
        "predicted_sequence": aligned_preds,
        "score": aligned_scores,
    })
    pred_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    # Save metrics JSON
    with open(metrics_json, "w") as f:
        json.dump(m_dict, f, indent=2)
    print(f"Saved metrics to {metrics_json}")

    print("\n--- POWERNOVO2 RESULTS ---")
    print(f"Strict Exact Match Precision: {m.peptide_precision_exact*100:.2f}%")
    print(f"I/L Exact Match Precision:    {m.peptide_precision_exact_il*100:.2f}%")
    print(f"Residue F1:                   {m.aa_f1*100:.2f}% (Prec: {m.aa_precision*100:.2f}%, Rec: {m.aa_recall*100:.2f}%)")
    print(f"Throughput:                   {m_dict['throughput_spec_per_sec']:.2f} spectra/sec")
    return m_dict


def main():
    parser = argparse.ArgumentParser(description="Run Casanovo and PowerNovo2 benchmarks")
    parser.add_argument("--model", choices=["casanovo", "powernovo2", "all"], default="all")
    parser.add_argument("--split", choices=["ninespecies", "hcpt", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    bench_dir = Path("/dev/shm/benchmark_inputs")
    out_artifacts = PROJECT_ROOT / "artifacts" / "benchmark_external"
    out_artifacts.mkdir(parents=True, exist_ok=True)

    ns_mgf = bench_dir / "ninespecies_test.mgf"
    ns_gt = bench_dir / "ninespecies_test_groundtruth.parquet"
    hc_mgf = bench_dir / "hcpt_50k_test.mgf"
    hc_gt = bench_dir / "hcpt_50k_test_groundtruth.parquet"

    # 1. CASANOVO
    if args.model in ("casanovo", "all"):
        if args.split in ("ninespecies", "all"):
            run_casanovo_split(
                mgf_path=ns_mgf,
                gt_path=ns_gt,
                out_dir=Path("/dev/shm/cas_ns_out"),
                prefix="cas_ns",
                predictions_csv=out_artifacts / "casanovo_ninespecies_predictions.csv",
                metrics_json=out_artifacts / "casanovo_ninespecies_metrics.json",
            )
        if args.split in ("hcpt", "all"):
            run_casanovo_split(
                mgf_path=hc_mgf,
                gt_path=hc_gt,
                out_dir=Path("/dev/shm/cas_hc_out"),
                prefix="cas_hc",
                predictions_csv=out_artifacts / "casanovo_hcpt_predictions.csv",
                metrics_json=out_artifacts / "casanovo_hcpt_metrics.json",
            )

    # 2. POWERNOVO2
    if args.model in ("powernovo2", "all"):
        pwn_work = Path("/dev/shm/powernovo2_work")
        if args.split in ("ninespecies", "all"):
            run_powernovo2_split(
                mgf_path=ns_mgf,
                gt_path=ns_gt,
                work_dir=pwn_work,
                out_dir=Path("/dev/shm/pwn_ns_out"),
                predictions_csv=out_artifacts / "powernovo2_ninespecies_predictions.csv",
                metrics_json=out_artifacts / "powernovo2_ninespecies_metrics.json",
                batch_size=args.batch_size,
            )
        if args.split in ("hcpt", "all"):
            run_powernovo2_split(
                mgf_path=hc_mgf,
                gt_path=hc_gt,
                work_dir=pwn_work,
                out_dir=Path("/dev/shm/pwn_hc_out"),
                predictions_csv=out_artifacts / "powernovo2_hcpt_predictions.csv",
                metrics_json=out_artifacts / "powernovo2_hcpt_metrics.json",
                batch_size=args.batch_size,
            )

    print("\nBenchmark execution complete! Results saved in", out_artifacts)


if __name__ == "__main__":
    main()
