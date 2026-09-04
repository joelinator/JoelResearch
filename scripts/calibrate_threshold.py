#!/usr/bin/env bash
"""Calibrate confidence score thresholds for de novo peptide sequencing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from pathlib import Path

scripts_dir = str(Path(__file__).resolve().parent)
while scripts_dir in sys.path:
    sys.path.remove(scripts_dir)

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import torch

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from eval.metrics import calibrate_score_threshold, compute_denovo_metrics
from eval.plots import plot_pauc_curve
from flow_matching.scheduler import cosine_scheduler
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate confidence score threshold on a validation subset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/dfm_pl_run_20260903_092757/checkpoints/last.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="Number of batches to use for fast calibration on validation subset (e.g. 5 batches ~10k spectra)",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Num DataLoader workers")
    parser.add_argument("--num-steps", type=int, default=20, help="Flow Euler integration steps")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="CFG guidance scale")
    parser.add_argument("--top-k-lengths", type=int, default=3, help="Top-K lengths for beam decoding")
    parser.add_argument("--length-beam-alpha", type=float, default=0.01, help="Precursor mass error penalty")
    parser.add_argument("--noising-scheme", type=str, default="mask", choices=["mask", "uniform"], help="Noising scheme (mask or uniform)")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Hugging Face cache directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default="artifacts/calibrated_threshold.json")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save PAUC plot on calibration subset")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("==========================================================")
    print("  DFM Confidence Score Threshold Calibration")
    print("=========================================================="
    f"Checkpoint:       {args.checkpoint}\n"
    f"Split:            {args.split} (max {args.max_batches} batches)\n"
    f"Batch Size:       {args.batch_size}\n"
    f"Guidance Scale:   {args.guidance_scale}\n"
    f"Steps:            {args.num_steps}\n"
    f"Device:           {device}\n"
    "==========================================================")

    ckpt_obj = load_checkpoint(args.checkpoint, map_location=device)
    vocabulary = ckpt_obj.get("vocabulary")
    if vocabulary is None:
        ckpt_p = Path(args.checkpoint).resolve()
        for cand in [ckpt_p.parent / "vocabulary.json", ckpt_p.parent.parent / "vocabulary.json"]:
            if cand.exists():
                with cand.open() as f:
                    vocabulary = json.load(f)
                break

    ds = get_dataset(split=args.split, cache_dir=args.cache_dir)
    if vocabulary is None:
        vocabulary = build_vocabulary(ds)

    loader = build_dataloader(
        ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader.dataset.top_k = DEFAULTS.data.top_k_peaks

    enc, lp, dec, guid = build_models(vocabulary, device)
    load_models_from_checkpoint(ckpt_obj, enc, lp, dec, guid)

    print(f"Running generative inference on validation subset ({args.max_batches} batches)...")
    raw_metrics, details = evaluate_generative(
        loader,
        vocabulary,
        enc,
        lp,
        dec,
        guid,
        cosine_scheduler,
        device,
        max_batches=args.max_batches,
        num_steps=args.num_steps,
        noising_scheme=args.noising_scheme,
        guidance_scale=args.guidance_scale,
        top_k_lengths=args.top_k_lengths,
        alpha=args.length_beam_alpha,
        return_details=True,
    )

    scores = np.asarray(details["scores"], dtype=np.float64)
    exact_matches = np.asarray(details["exact_matches"], dtype=bool)
    mass_matches = np.asarray(details["mass_matches"], dtype=bool)

    # 1. Calibrate Exact Match Thresholds
    calib_exact_80 = calibrate_score_threshold(scores, exact_matches, target_precision=0.80)
    calib_exact_90 = calibrate_score_threshold(scores, exact_matches, target_precision=0.90)
    calib_exact_f1 = calibrate_score_threshold(scores, exact_matches, strategy="max_f1")

    # 2. Calibrate Mass Match Thresholds
    calib_mass_80 = calibrate_score_threshold(scores, mass_matches, target_precision=0.80)
    calib_mass_90 = calibrate_score_threshold(scores, mass_matches, target_precision=0.90)
    calib_mass_f1 = calibrate_score_threshold(scores, mass_matches, strategy="max_f1")

    print("\n---------------- Calibration Results ----------------")
    print(f"Total Subset Samples: {len(scores)}")
    print(f"Unthresholded Exact Accuracy: {raw_metrics.exact_peptide_accuracy:.4f}")
    print(f"Unthresholded Mass Accuracy:  {raw_metrics.mass_peptide_accuracy:.4f}\n")

    print(f"{'Target':<22} | {'Threshold':<10} | {'Coverage':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 75)
    for name, c in [
        ("Exact @ 80% Precision", calib_exact_80),
        ("Exact @ 90% Precision", calib_exact_90),
        ("Exact @ Max F1", calib_exact_f1),
        ("Mass @ 80% Precision", calib_mass_80),
        ("Mass @ 90% Precision", calib_mass_90),
        ("Mass @ Max F1", calib_mass_f1),
    ]:
        print(f"{name:<22} | {c['threshold']:<10.3f} | {c['coverage']:<10.2%} | {c['precision']:<10.2%} | {c['recall']:<10.2%} | {c['f1']:<10.4f}")

    results = {
        "num_calibration_samples": len(scores),
        "unthresholded_exact_accuracy": raw_metrics.exact_peptide_accuracy,
        "unthresholded_mass_accuracy": raw_metrics.mass_peptide_accuracy,
        "auc_exact": raw_metrics.auc_exact,
        "pauc80_exact": raw_metrics.pauc80_exact,
        "auc_mass": raw_metrics.auc_mass,
        "pauc80_mass": raw_metrics.pauc80_mass,
        "calibrations": {
            "exact_prec80": calib_exact_80,
            "exact_prec90": calib_exact_90,
            "exact_max_f1": calib_exact_f1,
            "mass_prec80": calib_mass_80,
            "mass_prec90": calib_mass_90,
            "mass_max_f1": calib_mass_f1,
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved calibration results to {out_json}")

    if args.save_plot:
        plot_pauc_curve(
            scores=scores,
            exact_matches=exact_matches,
            mass_matches=mass_matches,
            output_path=args.save_plot,
            title=f"Validation Subset PAUC & Score Calibration (N={len(scores)})",
            calibrated_threshold=calib_mass_80["threshold"],
            calibrated_coverage=calib_mass_80["coverage"],
            calibrated_precision=calib_mass_80["precision"],
            calibrated_recall=calib_mass_80["recall"],
        )


if __name__ == "__main__":
    main()
