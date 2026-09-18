#!/usr/bin/env python3
"""
Benchmark Runner for Balanced Joint Multi-Domain Model Evaluation.

Evaluates the joint model (trained 1:1 on Nine-Species + HC-PT) on:
1. Full Nine-Species Test Split (104,163 spectra)
2. Full HC-PT Test Split (265,369 spectra)

Uses Strategy A Multi-Step Knapsack Flow Matching Guidance + Bayesian Rescoring.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

JOINT_CKPT = "artifacts/dfm_joint_balanced_8ep/checkpoints/best-joint-gen-exact-epoch=04-exact=0.4695.ckpt"

EXPERIMENTS = [
    {
        "name": "Joint Balanced Model -> Nine-Species Full Test",
        "ckpt": JOINT_CKPT,
        "dataset": "InstaDeepAI/ms_ninespecies_benchmark",
        "split": "test",
        "output_json": "artifacts/eval_joint_balanced/joint_ninespecies_full_test_metrics.json",
        "output_csv": "artifacts/eval_joint_balanced/joint_ninespecies_full_test_preds.csv",
        "output_plot": "artifacts/eval_joint_balanced/joint_ninespecies_full_test_plot.png",
    },
    {
        "name": "Joint Balanced Model -> HC-PT Full Test",
        "ckpt": JOINT_CKPT,
        "dataset": "InstaDeepAI/ms_proteometools",
        "split": "test",
        "output_json": "artifacts/eval_joint_balanced/joint_hcpt_full_test_metrics.json",
        "output_csv": "artifacts/eval_joint_balanced/joint_hcpt_full_test_preds.csv",
        "output_plot": "artifacts/eval_joint_balanced/joint_hcpt_full_test_plot.png",
    },
]

def run_experiment(exp, idx, total):
    print(f"\n{'='*70}")
    print(f"[{idx}/{total}] STARTING: {exp['name']}")
    print(f"Checkpoint: {exp['ckpt']}")
    print(f"Dataset:    {exp['dataset']} ({exp['split']})")
    print(f"Output:     {exp['output_json']}")
    print(f"{'='*70}\n", flush=True)

    Path(exp["output_json"]).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--checkpoint", exp["ckpt"],
        "--dataset-name", exp["dataset"],
        "--split", exp["split"],
        "--batch-size", "2048",
        "--num-workers", "8",
        "--output-json", exp["output_json"],
        "--save-predictions", exp["output_csv"],
        "--save-plot", exp["output_plot"],
    ]

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"[-] ERROR in {exp['name']}! Exited with code {res.returncode}", flush=True)
        sys.exit(res.returncode)

    print(f"[+] COMPLETED: {exp['name']} in {elapsed/60:.2f} minutes.\n", flush=True)

def main():
    total = len(EXPERIMENTS)
    start_all = time.time()

    print("======================================================================")
    print("STARTING FULL TEST BENCHMARKS FOR BALANCED JOINT MODEL")
    print(f"Target Checkpoint: {JOINT_CKPT}")
    print(f"Total Runs:        {total}")
    print("======================================================================")

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        run_experiment(exp, idx, total)

    total_elapsed = time.time() - start_all
    print("======================================================================")
    print(f"ALL JOINT BENCHMARKS COMPLETED SUCCESSFULLY IN {total_elapsed/60:.2f} MINUTES!")
    print("======================================================================")

if __name__ == "__main__":
    main()
