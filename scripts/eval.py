#!/usr/bin/env python3
"""Run de novo evaluation on a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

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
from eval.metrics import calibrate_score_threshold, compute_denovo_metrics, format_metrics
from eval.plots import plot_pauc_curve
from flow_matching.scheduler import cosine_scheduler
from train.io import load_checkpoint, load_models_from_checkpoint
from train.factory import build_models


def parse_args():
    data_cfg = DEFAULTS.data
    eval_cfg = DEFAULTS.eval

    parser = argparse.ArgumentParser(description="Evaluate DFM de novo peptide sequencing.")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--split", default=os.environ.get("EVAL_SPLIT", data_cfg.test_split))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", eval_cfg.batch_size)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", eval_cfg.num_workers)))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("INFERENCE_STEPS", eval_cfg.inference_steps)))
    parser.add_argument(
        "--noising-scheme",
        choices=["uniform", "mask"],
        default=os.environ.get("NOISING_SCHEME", eval_cfg.noising_scheme),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(os.environ.get("GUIDANCE_SCALE", eval_cfg.guidance_scale)),
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=int(os.environ.get("TOP_K_LENGTHS", eval_cfg.top_k_lengths)),
        help="Number of top length candidates to decode in parallel (default: 3).",
    )
    parser.add_argument(
        "--length-beam-alpha",
        type=float,
        default=float(os.environ.get("LENGTH_BEAM_ALPHA", eval_cfg.length_beam_alpha)),
        help="Weight for mass mismatch penalty in length beam score (default: 0.01).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=int(os.environ["MAX_BATCHES"]) if os.environ.get("MAX_BATCHES") else eval_cfg.max_batches,
    )
    parser.add_argument(
        "--calibrate-subset-batches",
        type=int,
        default=int(os.environ["CALIBRATE_SUBSET_BATCHES"]) if os.environ.get("CALIBRATE_SUBSET_BATCHES") else None,
        help="Calibrate score threshold on the first N batches of validation split before evaluating full set",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=float(os.environ.get("TARGET_PRECISION", 0.80)),
        help="Target precision level for confidence threshold calibration (default: 0.80)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=float(os.environ["SCORE_THRESHOLD"]) if os.environ.get("SCORE_THRESHOLD") else None,
        help="Explicit score threshold cutoff",
    )
    parser.add_argument("--output-json", default=os.environ.get("EVAL_OUTPUT_JSON"))
    parser.add_argument("--save-plot", default=os.environ.get("SAVE_PLOT"))
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True, help="Disable automatic mixed precision (FP16/BF16)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("Provide --checkpoint or set CHECKPOINT environment variable.")
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    
    # Resolve vocabulary: from checkpoint, adjacent vocabulary.json, or dataset
    vocabulary = checkpoint.get("vocabulary")
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
    elif build_vocabulary(ds) != vocabulary:
        print("Warning: dataset vocabulary differs from checkpoint vocabulary; using checkpoint vocab.")

    loader = build_dataloader(
        ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader.dataset.top_k = DEFAULTS.data.top_k_peaks

    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)
    load_models_from_checkpoint(
        checkpoint,
        spectrum_encoder,
        length_predictor,
        decoder,
        guidance,
    )

    print("==========================================================")
    print("  DFM De Novo Sequencing Generative Evaluation")
    print("==========================================================")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Split:            {args.split} (total batches: {len(loader)})")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Workers:          {args.num_workers}")
    print(f"Guidance Scale:   {args.guidance_scale}")
    print(f"Inference Steps:  {args.num_steps}")
    print(f"Top-K Lengths:    {args.top_k_lengths}")
    print(f"Alpha:            {args.length_beam_alpha}")
    print(f"Device:           {device}")
    print("==========================================================")

    # 1. Run full evaluation collecting all outputs & scores
    raw_metrics, details = evaluate_generative(
        loader,
        vocabulary,
        spectrum_encoder,
        length_predictor,
        decoder,
        guidance,
        cosine_scheduler,
        device,
        max_batches=args.max_batches,
        num_steps=args.num_steps,
        noising_scheme=args.noising_scheme,
        guidance_scale=args.guidance_scale,
        top_k_lengths=args.top_k_lengths,
        alpha=args.length_beam_alpha,
        aa_mass_tolerance=DEFAULTS.eval.aa_mass_tolerance,
        prefix_mass_tolerance=DEFAULTS.eval.prefix_mass_tolerance,
        amp=args.amp,
        return_details=True,
    )

    scores = np.asarray(details["scores"], dtype=np.float64)
    exact_matches = np.asarray(details["exact_matches"], dtype=bool)
    mass_matches = np.asarray(details["mass_matches"], dtype=bool)

    # 2. Calibrate thresholds on validation subset if requested
    calib_exact = None
    calib_mass = None

    if args.calibrate_subset_batches is not None:
        subset_n = min(len(scores), args.calibrate_subset_batches * args.batch_size)
        sub_scores = scores[:subset_n]
        sub_exact = exact_matches[:subset_n]
        sub_mass = mass_matches[:subset_n]
        print(f"\n[Calibration on subset of {subset_n} spectra]")
        calib_exact = calibrate_score_threshold(sub_scores, sub_exact, target_precision=args.target_precision)
        calib_mass = calibrate_score_threshold(sub_scores, sub_mass, target_precision=args.target_precision)
        print(f"  Calibrated Mass Match Threshold (Target Prec >= {args.target_precision:.0%}): {calib_mass['threshold']:.3f} (Coverage: {calib_mass['coverage']:.2%})")
        print(f"  Calibrated Exact Match Threshold: {calib_exact['threshold']:.3f} (Coverage: {calib_exact['coverage']:.2%}, Prec: {calib_exact['precision']:.2%})")
    else:
        calib_exact = calibrate_score_threshold(scores, exact_matches, target_precision=args.target_precision)
        calib_mass = calibrate_score_threshold(scores, mass_matches, target_precision=args.target_precision)

    # Threshold to use for primary thresholded reporting
    if args.score_threshold is not None:
        primary_threshold = args.score_threshold
    else:
        primary_threshold = calib_mass["threshold"]

    # 3. Compute metrics: Unthresholded, Mass-Calibrated, Exact-Calibrated
    metrics_unthresh = compute_denovo_metrics(
        details["predictions"],
        details["targets"],
        predicted_lengths=details["predicted_lengths"],
        target_lengths=details["target_lengths"],
        scores=scores,
        score_threshold=None,
        aa_mass_tolerance=DEFAULTS.eval.aa_mass_tolerance,
        prefix_mass_tolerance=DEFAULTS.eval.prefix_mass_tolerance,
    )

    metrics_primary = compute_denovo_metrics(
        details["predictions"],
        details["targets"],
        predicted_lengths=details["predicted_lengths"],
        target_lengths=details["target_lengths"],
        scores=scores,
        score_threshold=primary_threshold,
        aa_mass_tolerance=DEFAULTS.eval.aa_mass_tolerance,
        prefix_mass_tolerance=DEFAULTS.eval.prefix_mass_tolerance,
    )

    metrics_exact_calib = compute_denovo_metrics(
        details["predictions"],
        details["targets"],
        predicted_lengths=details["predicted_lengths"],
        target_lengths=details["target_lengths"],
        scores=scores,
        score_threshold=calib_exact["threshold"],
        aa_mass_tolerance=DEFAULTS.eval.aa_mass_tolerance,
        prefix_mass_tolerance=DEFAULTS.eval.prefix_mass_tolerance,
    )

    # 4. Print results
    print("\n" + "=" * 65)
    print("                EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total Samples: {len(scores)}")
    print(f"Unthresholded Exact Peptide Accuracy: {metrics_unthresh.exact_peptide_accuracy * 100:.2f}%")
    print(f"Unthresholded Mass-Based Accuracy:    {metrics_unthresh.mass_peptide_accuracy * 100:.2f}%")
    print(f"Unthresholded Length Accuracy:        {metrics_unthresh.length_accuracy * 100:.2f}%")
    print(f"Unthresholded Amino Acid F1:          {metrics_unthresh.aa_f1 * 100:.2f}%")
    print(f"Peptide PR-AUC (Mass-Based Match):    {metrics_unthresh.pr_auc_mass:.4f}")
    print(f"Peptide pPR-AUC80 (Mass Match >= 80%):{metrics_unthresh.p_pr_auc80_mass:.4f}")
    print(f"Peptide PR-AUC (Exact Match):         {metrics_unthresh.pr_auc_exact:.4f}")
    print(f"Precision-Coverage AUC (Mass-Based):  {metrics_unthresh.auc_mass:.4f}")
    print(f"pAUPCC80 (Mass-Based >= 80% P):       {metrics_unthresh.pauc80_mass:.4f}")
    print(f"Precision-Coverage AUC (Exact Match): {metrics_unthresh.auc_exact:.4f}")
    print("-" * 65)
    print(f"Mass-Calibrated Threshold (tau={primary_threshold:.3f}):")
    print(f"  Coverage:                           {metrics_primary.coverage * 100:.2f}% ({metrics_primary.num_predicted_above_threshold}/{len(scores)})")
    print(f"  Mass-Based Precision:               {metrics_primary.peptide_precision_mass * 100:.2f}%")
    print(f"  Mass-Based Recall:                  {metrics_primary.peptide_recall_mass * 100:.2f}%")
    print(f"  Mass-Based F1:                      {metrics_primary.peptide_f1_mass * 100:.2f}%")
    print(f"  Exact Match Precision:              {metrics_primary.peptide_precision_exact * 100:.2f}%")
    print(f"  Exact Match Recall:                 {metrics_primary.peptide_recall_exact * 100:.2f}%")
    print(f"  Exact Match F1:                     {metrics_primary.peptide_f1_exact * 100:.2f}%")
    print("=" * 65)

    # 5. Save output JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "metrics": metrics_primary.to_dict(),
            "unthresholded_metrics": metrics_unthresh.to_dict(),
            "exact_calibrated_metrics": metrics_exact_calib.to_dict(),
            "subset_calibration_exact": calib_exact,
            "subset_calibration_mass": calib_mass,
            "config": {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "batch_size": args.batch_size,
                "guidance_scale": args.guidance_scale,
                "inference_steps": args.num_steps,
                "top_k_lengths": args.top_k_lengths,
                "length_beam_alpha": args.length_beam_alpha,
                "target_precision": args.target_precision,
                "score_threshold": primary_threshold,
            },
        }
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved metrics to {output_path}")

    # 6. Save PAUC Plot
    if args.save_plot:
        plot_pauc_curve(
            scores=scores,
            exact_matches=exact_matches,
            mass_matches=mass_matches,
            output_path=args.save_plot,
            title=f"Full Validation Split Precision-Recall & Precision-Coverage Analysis (N={len(scores):,})",
            calibrated_threshold=primary_threshold,
            calibrated_coverage=metrics_primary.coverage,
            calibrated_precision=metrics_primary.peptide_precision_mass,
            calibrated_recall=metrics_primary.peptide_recall_mass,
        )


if __name__ == "__main__":
    main()
