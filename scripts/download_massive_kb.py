#!/usr/bin/env python3
"""
High-throughput multi-threaded downloader for MassIVE-KB dataset shards.
Directs all downloads to /dev/shm/massive_kb to protect root disk quota.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["HF_HOME"] = "/dev/shm/.cache/huggingface"

HF_REPO_ID = "chrisagrams/massive_kb_v1_shuffled"
TARGET_DIR = "/dev/shm/massive_kb"
MAX_WORKERS = 12
# NOTE: /dev/shm is pure RAM. It makes parquet reads blisteringly fast during training,
# but remember to clean up if the node is shared or before shutting down.
# TODO: add optional local SSD fallback when /dev/shm capacity is below 20GB.


def download_single_shard(split: str, shard_idx: int) -> tuple[int, float, float, bool]:
    split_dir = Path(TARGET_DIR) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    dest_path = split_dir / f"{split}_{shard_idx:03d}.parquet"
    filename = f"{split}/{split}_{shard_idx:03d}.parquet"

    if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:
        # Already downloaded and valid (>10MB)
        return shard_idx, dest_path.stat().st_size / 1024 / 1024, 0.0, True

    t0 = time.time()
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=TARGET_DIR,
            cache_dir="/dev/shm/.cache/huggingface",
        )
        dur = time.time() - t0
        sz_mb = dest_path.stat().st_size / 1024 / 1024
        return shard_idx, sz_mb, dur, False
    except Exception as e:
        print(f"[Error] Failed downloading {filename}: {e}", file=sys.stderr)
        return shard_idx, 0.0, 0.0, False


def main():
    parser = argparse.ArgumentParser(description="Download MassIVE-KB dataset shards.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--start-shard", type=int, default=0, help="First shard index to download")
    parser.add_argument("--num-shards", type=int, default=50, help="Total number of shards to ensure present (0 to num-shards)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel download workers")
    args = parser.parse_args()

    split = args.split
    start_idx = args.start_shard
    end_idx = args.num_shards
    total_requested = end_idx - start_idx

    print(f"Starting parallel download of {split} shards {start_idx} to {end_idx-1} into {TARGET_DIR}/{split} ...")
    t_start = time.time()
    total_downloaded_mb = 0.0
    downloaded_count = 0
    cached_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_shard, split, i): i for i in range(start_idx, end_idx)}
        for future in as_completed(futures):
            idx, sz_mb, dur, was_cached = future.result()
            if was_cached:
                cached_count += 1
            else:
                downloaded_count += 1
                total_downloaded_mb += sz_mb
                speed = sz_mb / max(dur, 0.01)
                if downloaded_count % 8 == 0 or downloaded_count == (total_requested - cached_count):
                    elapsed = time.time() - t_start
                    overall_speed = total_downloaded_mb / max(elapsed, 0.01)
                    print(
                        f"Progress: [{cached_count + downloaded_count}/{total_requested}] shards | "
                        f"Downloaded: {total_downloaded_mb/1024:.2f} GB | "
                        f"Overall: {overall_speed:.1f} MB/s | Elapsed: {elapsed:.1f}s"
                    )

    # Clean up any cache remnants in /dev/shm/.cache to maximize free tmpfs space
    cache_dir = Path("/dev/shm/.cache")
    if cache_dir.exists():
        os.system("rm -rf /dev/shm/.cache/huggingface/hub/datasets--chrisagrams--massive_kb_v1_shuffled")

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Completed! {end_idx} {split} shards ready in {TARGET_DIR}/{split}.")
    print(f"Total time: {total_time:.1f}s ({total_downloaded_mb/1024:.2f} GB at {total_downloaded_mb/max(total_time, 0.01):.1f} MB/s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
