#!/usr/bin/env python3
"""
Script to evaluate the pre-trained InstaNovo model on a subset of the Nine Species dataset.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run InstaNovo baseline on Nine Species.")
    parser.add_argument("--subset-size", type=int, default=1000, help="Number of spectra to evaluate.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, validation, test).")
    parser.add_argument("--output-dir", type=str, default="artifacts/instanovo_baseline")
    parser.add_argument("--model-id", type=str, default="instanovo-v1.2.0")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is missing. Please install it (pip install datasets).")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"ninespecies_{args.split}_subset_{args.subset_size}.parquet"
    predictions_path = output_dir / f"instanovo_predictions_{args.subset_size}.csv"

    # 1. Download and subset the dataset
    if not parquet_path.exists():
        print(f"Loading InstaDeepAI/ms_ninespecies_benchmark ({args.split} split)...")
        ds = load_dataset("InstaDeepAI/ms_ninespecies_benchmark", split=args.split)
        
        # Take subset
        if args.subset_size > 0 and args.subset_size < len(ds):
            print(f"Selecting {args.subset_size} samples...")
            ds = ds.select(range(args.subset_size))
        
        print(f"Saving subset to {parquet_path}...")
        ds.to_parquet(parquet_path)
    else:
        print(f"Dataset subset already exists at {parquet_path}")

    # 2. Run InstaNovo Inference
    instanovo_repo = Path(os.path.expanduser("~/DiffusionResearchProject/InstaNovo"))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{instanovo_repo}:{env.get('PYTHONPATH', '')}"

    print(f"\nRunning InstaNovo ({args.model_id}) on {parquet_path}...")
    cmd = [
        sys.executable, "-m", "instanovo.cli", "predict",
        "--data-path", str(parquet_path),
        "--instanovo-model", args.model_id,
        "--output-path", str(predictions_path),
        "--denovo",
        "--no-refinement" # Disables InstaNovo+ (diffusion) to just test the base model
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print(f"\n✅ Success! Predictions saved to {predictions_path}")
        print("Next steps:")
        print("1. Run your DFM model on the exact same parquet file.")
        print("2. Feed both CSV outputs into your metrics script to compare.")
    else:
        print("\n❌ InstaNovo inference failed. Please check the logs above.")

if __name__ == "__main__":
    main()
