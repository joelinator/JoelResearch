#!/usr/bin/env python3
"""Download the ProteomeTools dataset from HuggingFace."""

from __future__ import annotations

import argparse
import os

from bootstrap import setup_src_path

setup_src_path()

from data.data import DEFAULT_DATASET, get_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download InstaDeepAI/ms_proteometools from HuggingFace."
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO", DEFAULT_DATASET),
        help="HuggingFace dataset repository id.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_DATASETS_CACHE", "data/cache"),
        help="Local cache directory for HuggingFace datasets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=os.environ.get("HF_SPLITS", "train validation test").split(),
        help="Dataset splits to download (e.g. train validation test).",
    )
    parser.add_argument(
        "--subset",
        default=os.environ.get("HF_SUBSET", ""),
        help="Optional HuggingFace split slice, e.g. '[:1%%]' for a 1%% subset.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="HuggingFace token. Prefer HF_TOKEN or HUGGINGFACE_TOKEN env vars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.token:
        print("Using HuggingFace token from CLI argument or environment.")
    else:
        print(
            "No HuggingFace token provided. Public datasets still download, "
            "but setting HF_TOKEN can improve rate limits and speed."
        )

    for split_name in args.splits:
        split = f"{split_name}{args.subset}" if args.subset else split_name
        print(f"Downloading {args.repo_id} split={split!r} -> cache={args.cache_dir}")
        dataset = get_dataset(
            repo_id=args.repo_id,
            split=split,
            cache_dir=args.cache_dir,
            token=args.token,
        )
        print(f"  rows={len(dataset):,} columns={dataset.column_names}")

    print("Download complete.")


if __name__ == "__main__":
    main()
