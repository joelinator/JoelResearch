#!/usr/bin/env python3
"""
Empirical Throughput and Latency Benchmark: InstaNovo v1.2 vs InstaNovo+

Evaluates and compares:
  1. InstaNovo v1.2 (Autoregressive, 94.8M params) with Beam Search B=5 and Greedy B=1
  2. InstaNovo+ (Discrete Multinomial Diffusion, 172.6M params, 20 steps + 4 eval steps) with B=1 and B=5
Across batch sizes (512, 128) on NVIDIA H100 80GB HBM3 GPU.

Records:
  - Total elapsed wall-clock time with torch.cuda.synchronize()
  - Throughput (spectra/second)
  - Latency per spectrum (ms/spectrum)
  - Peak GPU VRAM allocated (GB)
  - Theoretical forward-pass / computational complexity breakdown
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from instanovo.transformer.model import InstaNovo
from instanovo.transformer.data import TransformerDataProcessor
from instanovo.inference.beam_search import BeamSearchDecoder

from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo.diffusion.data import DiffusionDataProcessor
from instanovo.inference.diffusion import DiffusionDecoder


class FastSpectraDataset(Dataset):
    """Direct index wrapper over Hugging Face Arrow table for zero-overhead batching."""

    def __init__(self, ds: HFDataset, processor):
        self.ds = ds
        self.processor = processor

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        res = self.processor.process_row(row)
        res["precursor_mz"] = row["precursor_mz"]
        res["precursor_charge"] = row["precursor_charge"]
        return res


def benchmark_model(
    model_name: str,
    decode_fn,
    loader: DataLoader,
    warmup_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Runs a warmup batch and then benchmarks inference across all batches in loader."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    print(f"  [{model_name}] Warming up GPU...")
    with torch.no_grad():
        for batch in warmup_loader:
            decode_fn(batch)
            break
    torch.cuda.synchronize(device)

    # Timed run
    print(f"  [{model_name}] Measuring timed inference across {len(loader.dataset)} spectra...")
    torch.cuda.reset_peak_memory_stats(device)
    total_spectra = len(loader.dataset)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            decode_fn(batch)

    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    throughput = total_spectra / elapsed
    ms_per_spec = (elapsed / total_spectra) * 1000.0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    result = {
        "model_config": model_name,
        "num_spectra": total_spectra,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_spec_per_s": round(throughput, 2),
        "latency_ms_per_spec": round(ms_per_spec, 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    print(f"    -> Elapsed: {elapsed:.3f}s | Throughput: {throughput:.1f} spec/s | Peak VRAM: {peak_vram_gb:.2f} GB")
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark InstaNovo v1.2 vs InstaNovo+ throughput.")
    parser.add_argument(
        "--arrow-path",
        default="data/cache/InstaDeepAI___ms_ninespecies_benchmark/default/0.0.0/95d03bc005e512bd20c6abb3f8650e327272973a/ms_ninespecies_benchmark-test.arrow",
        help="Path to Nine-Species test arrow dataset.",
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=2048,
        help="Number of test spectra to evaluate (default: 2048).",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/instanovo_vs_plus_empirical_throughput.json",
        help="Path to output json artifact.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 80)
    print(f"  BENCHMARK: INSTANOVO v1.2 vs INSTANOVO+ ON {gpu_name}")
    print(f"  Evaluating {args.num_test_samples} spectra from Nine-Species test split")
    print("=" * 80)

    # Load test split slice
    full_ds = HFDataset.from_file(args.arrow_path)
    warmup_ds = full_ds.select(range(256))
    eval_ds = full_ds.select(range(256, 256 + args.num_test_samples))

    results = {
        "device": gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "num_test_samples": args.num_test_samples,
        "benchmarks": [],
    }

    # -------------------------------------------------------------------------
    # 1. LOAD INSTANOVO v1.2
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Loading InstaNovo v1.2 (Autoregressive)...")
    ckpt_ar = "/home/joelgedeon_aims_ac_za/.cache/instanovo/instanovo-v1.2.0.ckpt"
    model_ar, config_ar = InstaNovo.load(ckpt_ar)
    model_ar = model_ar.to(device).eval()
    proc_ar = TransformerDataProcessor(
        model_ar.residue_set,
        n_peaks=config_ar.get("n_peaks", 200),
        use_spectrum_utils=False,
        annotated=False,
    )
    decoder_ar = BeamSearchDecoder(model=model_ar)
    params_ar = sum(p.numel() for p in model_ar.parameters())
    print(f"  InstaNovo v1.2 Parameters: {params_ar:,} ({params_ar/1e6:.2f}M)")

    warmup_ds_ar = FastSpectraDataset(warmup_ds, proc_ar)
    eval_ds_ar = FastSpectraDataset(eval_ds, proc_ar)

    # 1.1 InstaNovo v1.2 BS=512, B=5
    print("\n--- Running InstaNovo v1.2 (BS=512, Beam=5) ---")
    loader_w_512 = DataLoader(warmup_ds_ar, batch_size=256, shuffle=False, collate_fn=proc_ar.collate_fn)
    loader_e_512 = DataLoader(eval_ds_ar, batch_size=512, shuffle=False, collate_fn=proc_ar.collate_fn)
    def decode_ar_b5(batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return decoder_ar.decode(
                spectra=batch["spectra"].to(device),
                precursors=batch["precursors"].to(device),
                beam_size=5,
                max_length=30,
            )
    res_ar_b5 = benchmark_model("InstaNovo v1.2 (Autoregressive, BS=512, Beam=5)", decode_ar_b5, loader_e_512, loader_w_512, device)
    res_ar_b5["parameters_m"] = round(params_ar / 1e6, 2)
    results["benchmarks"].append(res_ar_b5)

    # 1.2 InstaNovo v1.2 BS=512, B=1 (Greedy)
    print("\n--- Running InstaNovo v1.2 (BS=512, Beam=1 / Greedy) ---")
    def decode_ar_b1(batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return decoder_ar.decode(
                spectra=batch["spectra"].to(device),
                precursors=batch["precursors"].to(device),
                beam_size=1,
                max_length=30,
            )
    res_ar_b1 = benchmark_model("InstaNovo v1.2 (Autoregressive, BS=512, Beam=1 Greedy)", decode_ar_b1, loader_e_512, loader_w_512, device)
    res_ar_b1["parameters_m"] = round(params_ar / 1e6, 2)
    results["benchmarks"].append(res_ar_b1)

    # 1.3 InstaNovo v1.2 BS=128, B=5
    print("\n--- Running InstaNovo v1.2 (BS=128, Beam=5) ---")
    loader_w_128 = DataLoader(warmup_ds_ar, batch_size=128, shuffle=False, collate_fn=proc_ar.collate_fn)
    loader_e_128 = DataLoader(eval_ds_ar, batch_size=128, shuffle=False, collate_fn=proc_ar.collate_fn)
    res_ar_b5_bs128 = benchmark_model("InstaNovo v1.2 (Autoregressive, BS=128, Beam=5)", decode_ar_b5, loader_e_128, loader_w_128, device)
    res_ar_b5_bs128["parameters_m"] = round(params_ar / 1e6, 2)
    results["benchmarks"].append(res_ar_b5_bs128)

    # Free AR model from GPU
    del model_ar, decoder_ar, loader_w_512, loader_e_512, loader_w_128, loader_e_128
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 2. LOAD INSTANOVO+
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Loading InstaNovo+ (Discrete Multinomial Diffusion)...")
    ckpt_diff = "instanovoplus-v1.1.0"
    model_diff, config_diff = InstaNovoPlus.from_pretrained(ckpt_diff)
    model_diff = model_diff.to(device).eval()
    proc_diff = DiffusionDataProcessor(
        model_diff.residue_set,
        n_peaks=config_diff.get("n_peaks", 200),
        use_spectrum_utils=False,
        annotated=False,
    )
    decoder_diff = DiffusionDecoder(model=model_diff)
    params_diff = sum(p.numel() for p in model_diff.parameters())
    print(f"  InstaNovo+ Parameters: {params_diff:,} ({params_diff/1e6:.2f}M)")

    warmup_ds_diff = FastSpectraDataset(warmup_ds, proc_diff)
    eval_ds_diff = FastSpectraDataset(eval_ds, proc_diff)

    # 2.1 InstaNovo+ BS=512, B=1 (Greedy / Single Candidate)
    print("\n--- Running InstaNovo+ (BS=512, Beam=1, 20 Steps) ---")
    loader_diff_w_512 = DataLoader(warmup_ds_diff, batch_size=256, shuffle=False, collate_fn=proc_diff.collate_fn)
    loader_diff_e_512 = DataLoader(eval_ds_diff, batch_size=512, shuffle=False, collate_fn=proc_diff.collate_fn)
    def decode_diff_b1(batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return decoder_diff.decode(
                spectra=batch["spectra"].to(device),
                precursors=batch["precursors"].to(device),
                spectra_padding_mask=batch["spectra_mask"].to(device),
                initial_sequence=None,
                beam_size=1,
            )
    res_diff_b1 = benchmark_model("InstaNovo+ (Diffusion, BS=512, Beam=1, 20 Steps)", decode_diff_b1, loader_diff_e_512, loader_diff_w_512, device)
    res_diff_b1["parameters_m"] = round(params_diff / 1e6, 2)
    results["benchmarks"].append(res_diff_b1)

    # 2.2 InstaNovo+ BS=128, B=1
    print("\n--- Running InstaNovo+ (BS=128, Beam=1, 20 Steps) ---")
    loader_diff_w_128 = DataLoader(warmup_ds_diff, batch_size=128, shuffle=False, collate_fn=proc_diff.collate_fn)
    loader_diff_e_128 = DataLoader(eval_ds_diff, batch_size=128, shuffle=False, collate_fn=proc_diff.collate_fn)
    res_diff_b1_bs128 = benchmark_model("InstaNovo+ (Diffusion, BS=128, Beam=1, 20 Steps)", decode_diff_b1, loader_diff_e_128, loader_diff_w_128, device)
    res_diff_b1_bs128["parameters_m"] = round(params_diff / 1e6, 2)
    results["benchmarks"].append(res_diff_b1_bs128)

    # 2.3 InstaNovo+ BS=128, B=5
    print("\n--- Running InstaNovo+ (BS=128, Beam=5, 20 Steps) ---")
    def decode_diff_b5(batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return decoder_diff.decode(
                spectra=batch["spectra"].to(device),
                precursors=batch["precursors"].to(device),
                spectra_padding_mask=batch["spectra_mask"].to(device),
                initial_sequence=None,
                beam_size=5,
            )
    res_diff_b5_bs128 = benchmark_model("InstaNovo+ (Diffusion, BS=128, Beam=5, 20 Steps)", decode_diff_b5, loader_diff_e_128, loader_diff_w_128, device)
    res_diff_b5_bs128["parameters_m"] = round(params_diff / 1e6, 2)
    results["benchmarks"].append(res_diff_b5_bs128)

    # 2.4 InstaNovo+ BS=512, B=5 (Test if 512*5 fits in VRAM)
    print("\n--- Running InstaNovo+ (BS=512, Beam=5, 20 Steps) ---")
    try:
        res_diff_b5_bs512 = benchmark_model("InstaNovo+ (Diffusion, BS=512, Beam=5, 20 Steps)", decode_diff_b5, loader_diff_e_512, loader_diff_w_512, device)
        res_diff_b5_bs512["parameters_m"] = round(params_diff / 1e6, 2)
        results["benchmarks"].append(res_diff_b5_bs512)
    except Exception as e:
        print(f"  [InstaNovo+ BS=512 Beam=5] Failed or OOM: {e}")
        results["benchmarks"].append({
            "model_config": "InstaNovo+ (Diffusion, BS=512, Beam=5, 20 Steps)",
            "error": str(e),
        })

    # Save output JSON
    out_path = PROJECT_ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical throughput benchmark results to {out_path}")

    # Summary table
    print("\n" + "=" * 95)
    print(f"{'Model Configuration':<52} | {'BS':<4} | {'Beam':<4} | {'Throughput':<12} | {'VRAM':<8}")
    print("=" * 95)
    for b in results["benchmarks"]:
        if "error" in b:
            print(f"{b['model_config']:<52} | ERR  | ERR  | OOM/FAILED   | --")
        else:
            cfg = b["model_config"]
            bs = "512" if "BS=512" in cfg else "128"
            bm = "5" if "Beam=5" in cfg else "1"
            tp = f"{b['throughput_spec_per_s']:.1f} spec/s"
            vram = f"{b['peak_vram_gb']:.2f} GB"
            print(f"{cfg:<52} | {bs:<4} | {bm:<4} | {tp:<12} | {vram:<8}")
    print("=" * 95)


if __name__ == "__main__":
    main()
