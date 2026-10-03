#!/usr/bin/env python3
"""
Calibrate optimal batch size on NVIDIA H100 80GB to target ~80% VRAM usage and maximize throughput.
"""

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_batch_size(batch_size, loader_dataset, vocab, enc, lp, dec, guid):
    loader = build_dataloader(loader_dataset, vocab, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    t0 = time.time()
    try:
        metrics, details = evaluate_generative(
            loader=loader,
            vocabulary=vocab,
            spectrum_encoder=enc,
            length_predictor=lp,
            decoder=dec,
            guidance=guid,
            scheduler="cosine",
            device=DEVICE,
            max_samples=batch_size * 2,  # 2 full batches
            num_steps=20,
            top_k_lengths=3,
            num_samples_per_length=1,
            temperature=0.0,
            use_knapsack_filter=True,
            use_exact_dp_knapsack=True,
            use_sequential_knapsack=True,
            knapsack_guidance_penalty=2.0,
            return_details=True,
        )
        dt = time.time() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
        throughput = (batch_size * 2) / dt
        total_vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024 ** 3)
        pct = (peak_vram_gb / total_vram_gb) * 100
        print(f"Batch Size {batch_size:4d}: Peak VRAM = {peak_vram_gb:6.2f} GB ({pct:5.1f}% of {total_vram_gb:.1f} GB) | Throughput = {throughput:6.1f} spec/s")
        return peak_vram_gb, throughput, True
    except torch.cuda.OutOfMemoryError:
        print(f"Batch Size {batch_size:4d}: OOM (Out Of Memory)")
        torch.cuda.empty_cache()
        return 0, 0, False


def main():
    ckpt_path = "artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt"
    ckpt = load_checkpoint(ckpt_path, map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()

    ds = get_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="test", cache_dir="data/cache")

    print(f"Calibrating batch size on {torch.cuda.get_device_name(0)} (Total VRAM: {torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3):.1f} GB)...")
    for bs in [256, 512, 1024, 1536, 2048, 3072, 4096]:
        peak, th, ok = test_batch_size(bs, ds, vocab, enc, lp, dec, guid)
        if not ok:
            break


if __name__ == "__main__":
    main()
