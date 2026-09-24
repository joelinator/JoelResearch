#!/usr/bin/env python3
"""
Prepares standardized MGF files and ground truth tables in /dev/shm for
Casanovo and PowerNovo2 benchmarking across:
1. Nine-Species test split (up to 20,000 per species -> all 104,163 spectra, L <= 30)
2. HC-PT (ProteomeTools) test split (50,000 spectra sample)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.data import filter_valid_peptide_lengths


def export_to_mgf(df: pd.DataFrame, mgf_path: Path, title_prefix: str = "scan") -> None:
    """Exports a dataframe with precursor_mz, precursor_charge, mz_array, intensity_array to MGF."""
    print(f"Writing {len(df)} spectra to {mgf_path}...")
    with open(mgf_path, "w") as f:
        for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"Exporting {mgf_path.name}")):
            mz_val = getattr(row, "precursor_mz")
            ch_val = getattr(row, "precursor_charge")
            f.write(f"BEGIN IONS\nTITLE={title_prefix}_{idx}\nPEPMASS={mz_val:.6f}\nCHARGE={ch_val}+\n")
            mzs = getattr(row, "mz_array")
            ints = getattr(row, "intensity_array")
            for mz, it in zip(mzs, ints):
                f.write(f"{mz:.4f} {it:.4f}\n")
            f.write("END IONS\n")
    print(f"Finished writing {mgf_path} (size: {mgf_path.stat().st_size / 1e6:.1f} MB).")


def prepare_ninespecies(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mgf_path = output_dir / "ninespecies_test.mgf"
    gt_path = output_dir / "ninespecies_test_groundtruth.parquet"

    if mgf_path.exists() and gt_path.exists():
        print(f"Nine-Species files already exist in {output_dir}")
        return mgf_path, gt_path

    print("Loading InstaDeepAI/ms_ninespecies_benchmark (test split)...")
    ds = load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test")
    print(f"Raw test spectra: {len(ds)}")
    ds = filter_valid_peptide_lengths(ds)
    print(f"Filtered (L <= 30) test spectra: {len(ds)}")

    df = ds.to_pandas()
    df["scan_id"] = range(len(df))
    
    # Save ground truth table
    df[["scan_id", "modified_sequence", "sequence", "precursor_mz", "precursor_charge"]].to_parquet(gt_path)
    print(f"Saved ground truth to {gt_path}")

    # Export MGF
    export_to_mgf(df, mgf_path, title_prefix="scan")
    return mgf_path, gt_path


def prepare_hcpt(output_dir: Path, num_samples: int = 50000, seed: int = 42) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mgf_path = output_dir / f"hcpt_{num_samples // 1000}k_test.mgf"
    gt_path = output_dir / f"hcpt_{num_samples // 1000}k_test_groundtruth.parquet"

    if mgf_path.exists() and gt_path.exists():
        print(f"HC-PT files already exist in {output_dir}")
        return mgf_path, gt_path

    # Locate cached test parquet or load via datasets
    cache_hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    hub_dir = cache_hub / "datasets--InstaDeepAI--ms_proteometools/snapshots"
    candidates = list(hub_dir.glob("*/data/test-*.parquet")) if hub_dir.exists() else []
    if candidates:
        src_parquet = candidates[0]
        print(f"Loading HC-PT test split from {src_parquet}...")
        table = pq.read_table(src_parquet, columns=["precursor_mz", "precursor_charge", "mz_array", "intensity_array", "modified_sequence", "sequence"])
    else:
        print("Loading HC-PT test split via load_dataset...")
        ds_hc = load_dataset("InstaDeepAI/ms_proteometools", split="test")
        table = ds_hc.data.table

    total_len = len(table)
    print(f"Total HC-PT test spectra: {total_len}")

    rng = np.random.default_rng(seed)
    if num_samples < total_len:
        sampled_indices = rng.choice(total_len, size=num_samples, replace=False)
        sampled_indices.sort()
        table_sampled = table.take(sampled_indices)
    else:
        table_sampled = table

    df = table_sampled.to_pandas()
    df["scan_id"] = range(len(df))

    # Save ground truth table
    df[["scan_id", "modified_sequence", "sequence", "precursor_mz", "precursor_charge"]].to_parquet(gt_path)
    print(f"Saved ground truth to {gt_path}")

    # Export MGF
    export_to_mgf(df, mgf_path, title_prefix="scan")
    return mgf_path, gt_path


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare standardized MGF benchmark files.")
    parser.add_argument("--output-dir", type=Path, default=Path("/dev/shm/benchmark_inputs"), help="Directory for output files.")
    parser.add_argument("--hcpt-samples", type=int, default=50000, help="Number of HC-PT test spectra to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Preparing benchmark data in {args.output_dir}...")
    ns_mgf, ns_gt = prepare_ninespecies(args.output_dir)
    hc_mgf, hc_gt = prepare_hcpt(args.output_dir, num_samples=args.hcpt_samples, seed=args.seed)
    print("All benchmark data prepared successfully!")
