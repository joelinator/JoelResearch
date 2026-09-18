#!/usr/bin/env python3
"""
Master Benchmark Runner for Strategy A Evaluation.

Evaluates 4 configurations sequentially on full test splits:
1. Base Model (HCPT trained only) on Nine Species Full Test Split
2. Base Model (HCPT trained only) on HC-PT Full Test Split
3. Finetuned Model (Nine Species Phase 2) on Nine Species Full Test Split
4. Finetuned Model (Nine Species Phase 2) on HC-PT Full Test Split

Saves metrics JSON, predictions CSV, and publication plots.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

BASE_CKPT = "artifacts/dfm_pl_run_20260915_000148/checkpoints/best-gen-exact-epoch=07-exact=0.3514.ckpt"
FINETUNED_CKPT = "artifacts/dfm_pl_ninespecies_finetune_phase2_10ep/checkpoints/best-gen-exact-epoch=02-exact=0.6270.ckpt"

EXPERIMENTS = [
    {
        "name": "Base Model (HCPT-only) -> Nine Species Full Test",
        "ckpt": BASE_CKPT,
        "dataset": "InstaDeepAI/ms_ninespecies_benchmark",
        "split": "test",
        "output_json": "artifacts/eval_strategy_a/base_ninespecies_full_test_metrics.json",
        "output_csv": "artifacts/eval_strategy_a/base_ninespecies_full_test_preds.csv",
        "output_plot": "artifacts/eval_strategy_a/base_ninespecies_full_test_plot.png",
    },
    {
        "name": "Base Model (HCPT-only) -> HC-PT Full Test",
        "ckpt": BASE_CKPT,
        "dataset": "InstaDeepAI/ms_proteometools",
        "split": "test",
        "output_json": "artifacts/eval_strategy_a/base_hcpt_full_test_metrics.json",
        "output_csv": "artifacts/eval_strategy_a/base_hcpt_full_test_preds.csv",
        "output_plot": "artifacts/eval_strategy_a/base_hcpt_full_test_plot.png",
    },
    {
        "name": "Finetuned Model (Nine-Species) -> Nine Species Full Test",
        "ckpt": FINETUNED_CKPT,
        "dataset": "InstaDeepAI/ms_ninespecies_benchmark",
        "split": "test",
        "output_json": "artifacts/eval_strategy_a/finetuned_ninespecies_full_test_metrics.json",
        "output_csv": "artifacts/eval_strategy_a/finetuned_ninespecies_full_test_preds.csv",
        "output_plot": "artifacts/eval_strategy_a/finetuned_ninespecies_full_test_plot.png",
    },
    {
        "name": "Finetuned Model (Nine-Species) -> HC-PT Full Test",
        "ckpt": FINETUNED_CKPT,
        "dataset": "InstaDeepAI/ms_proteometools",
        "split": "test",
        "output_json": "artifacts/eval_strategy_a/finetuned_hcpt_full_test_metrics.json",
        "output_csv": "artifacts/eval_strategy_a/finetuned_hcpt_full_test_preds.csv",
        "output_plot": "artifacts/eval_strategy_a/finetuned_hcpt_full_test_plot.png",
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
    print(f"Starting {total} Strategy A Full Benchmark Evaluations on H100 GPU...")

    for i, exp in enumerate(EXPERIMENTS, 1):
        run_experiment(exp, i, total)

    total_time = time.time() - start_all
    print(f"\n{'='*70}")
    print(f"ALL {total} BENCHMARKS COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
