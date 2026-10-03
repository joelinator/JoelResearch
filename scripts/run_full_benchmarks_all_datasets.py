#!/usr/bin/env python3
"""
Full-scale benchmark runner across Nine-Species (104k) and HC-PT (265k) datasets
using Soft Knapsack Guidance (penalty = 2.0) and optimized batch size 8192
targeting ~80-85% VRAM on NVIDIA H100 80GB GPU.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

CHECKPOINT = "artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt"
BATCH_SIZE = 8192
GUIDANCE_PENALTY = 2.0
NUM_WORKERS = 8
CACHE_DIR = "data/cache"

ENV = os.environ.copy()
ENV["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{ENV.get('PYTHONPATH', '')}"
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_benchmark(dataset_name, dataset_alias, output_json, output_csv):
    print("\n" + "=" * 80)
    print(f"  LAUNCHING FULL BENCHMARK: {dataset_alias.upper()}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Batch Size: {BATCH_SIZE} (~85% VRAM)")
    print(f"  Guidance:   Soft Guidance (penalty = {GUIDANCE_PENALTY})")
    print("=" * 80)

    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--checkpoint", CHECKPOINT,
        "--dataset-name", dataset_name,
        "--split", "test",
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--cache-dir", CACHE_DIR,
        "--num-steps", "25",
        "--top-k-lengths", "3",
        "--guidance-scale", "1.8",
        "--use-knapsack-filter",
        "--use-exact-dp-knapsack",
        "--knapsack-guidance-penalty", str(GUIDANCE_PENALTY),
        "--enzyme", "trypsin",
        "--use-composite-ladders",
        "--output-json", output_json,
        "--save-predictions", output_csv,
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=ENV,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark for {dataset_alias} failed with return code {proc.returncode}")

    print(f"\n[Completed {dataset_alias}] Time elapsed: {dt:.1f}s ({dt/60:.2f} min)")
    return dt


def main():
    print(f"Starting Full Datasets Benchmark with Soft Knapsack Guidance (penalty={GUIDANCE_PENALTY})")
    total_start = time.time()

    # 1. Nine-Species Benchmark (104,163 spectra)
    ninespecies_json = "artifacts/eval_dfm_soft_guidance_ninespecies_full.json"
    ninespecies_csv = "artifacts/eval_dfm_soft_guidance_ninespecies_full_preds.csv"
    dt_ns = run_benchmark(
        dataset_name="InstaDeepAI/ms_ninespecies_benchmark",
        dataset_alias="Nine-Species Full (104k)",
        output_json=ninespecies_json,
        output_csv=ninespecies_csv,
    )

    # 2. Human ProteomeTools (HC-PT, ~265,000 spectra)
    hcpt_json = "artifacts/eval_dfm_soft_guidance_hcpt_full.json"
    hcpt_csv = "artifacts/eval_dfm_soft_guidance_hcpt_full_preds.csv"
    dt_hcpt = run_benchmark(
        dataset_name="InstaDeepAI/ms_proteometools",
        dataset_alias="HC-PT Full (265k)",
        output_json=hcpt_json,
        output_csv=hcpt_csv,
    )

    total_time = time.time() - total_start
    print("\n" + "#" * 80)
    print("  ALL BENCHMARKS SUCCESSFULLY COMPLETED!")
    print(f"  Nine-Species Time: {dt_ns:.1f}s ({dt_ns/60:.2f} min)")
    print(f"  HC-PT Time:        {dt_hcpt:.1f}s ({dt_hcpt/60:.2f} min)")
    print(f"  Total Duration:    {total_time:.1f}s ({total_time/60:.2f} min)")
    print("#" * 80)


if __name__ == "__main__":
    main()
