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
from inference.knapsack_dp import ExactReachabilityDP
from train.io import infer_model_config_from_checkpoint, load_checkpoint, load_models_from_checkpoint
from train.factory import build_models


def parse_args():
    data_cfg = DEFAULTS.data
    eval_cfg = DEFAULTS.eval

    parser = argparse.ArgumentParser(description="Evaluate DFM de novo peptide sequencing.")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument(
        "--dataset-name",
        "--dataset",
        dest="dataset_name",
        default=os.environ.get("DATASET_NAME", data_cfg.dataset_repo),
        help="Hugging Face dataset repository (e.g. InstaDeepAI/ms_ninespecies_benchmark or InstaDeepAI/ms_proteometools).",
    )
    parser.add_argument("--split", default=os.environ.get("EVAL_SPLIT", data_cfg.test_split))
    parser.add_argument(
        "--scheduler",
        default=os.environ.get("SCHEDULER", "cosine"),
        choices=["linear", "cosine", "power1_5", "power2", "sigmoid"],
        help="Noise scheduler for flow matching (default: cosine).",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", eval_cfg.batch_size)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", eval_cfg.num_workers)))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("INFERENCE_STEPS", 20)))
    parser.add_argument(
        "--noising-scheme",
        choices=["uniform", "mask"],
        default=os.environ.get("NOISING_SCHEME", eval_cfg.noising_scheme),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(os.environ.get("GUIDANCE_SCALE", 1.0)),
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=int(os.environ.get("TOP_K_LENGTHS", 5)),
        help="Number of top length candidates to decode in parallel (default: 5).",
    )
    parser.add_argument(
        "--length-beam-alpha",
        type=float,
        default=float(os.environ.get("LENGTH_BEAM_ALPHA", 12.0)),
        help="Weight for mass mismatch penalty in length beam score (default: 12.0).",
    )
    parser.add_argument(
        "--use-knapsack-filter",
        dest="use_knapsack_filter",
        action="store_true",
        default=eval_cfg.use_knapsack_filter,
        help="Enable vectorized dynamic knapsack precursor mass filter during flow matching (default: True).",
    )
    parser.add_argument(
        "--no-knapsack-filter",
        dest="use_knapsack_filter",
        action="store_false",
        help="Disable knapsack precursor mass filter.",
    )
    parser.add_argument(
        "--knapsack-tol-da",
        type=float,
        default=float(os.environ.get("KNAPSACK_TOL_DA", 0.5)),
        help="Mass tolerance in Daltons for single remaining residue knapsack filter (default: 0.5 Da).",
    )
    parser.add_argument(
        "--knapsack-guidance-penalty",
        type=float,
        default=float(os.environ["KNAPSACK_GUIDANCE_PENALTY"]) if os.environ.get("KNAPSACK_GUIDANCE_PENALTY") else None,
        help="Soft knapsack penalty subtracted from unreachable token logits instead of -1e9 hard mask (e.g. 2.0 or 5.0).",
    )
    parser.add_argument(
        "--num-samples-per-length",
        type=int,
        default=int(os.environ.get("NUM_SAMPLES_PER_LENGTH", 2)),
        help="Number of stochastic trajectories to sample per length candidate for test-time reranking (default: 2).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=float(os.environ.get("ETA", 0.0)),
        help="Detailed Balance stochasticity parameter (Campbell et al., arXiv:2402.04997). Default 0.0.",
    )
    parser.add_argument(
        "--fragment-matching-weight",
        type=float,
        default=float(os.environ.get("FRAGMENT_MATCHING_WEIGHT", 2.5)),
        help="Weight beta for theoretical b/y fragment ion matching in beam scoring (default: 2.5).",
    )
    parser.add_argument(
        "--decoding-strategy",
        type=str,
        default=os.environ.get("DECODING_STRATEGY", eval_cfg.decoding_strategy),
        choices=["confidence", "random"],
        help="Discrete flow matching unmasking strategy during inference (default: confidence).",
    )
    parser.add_argument(
        "--decoding-temperature",
        type=float,
        default=float(os.environ.get("DECODING_TEMPERATURE", 0.5)),
        help="Sampling temperature during inference (default: 0.5).",
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=float(os.environ.get("LENGTH_WEIGHT", 0.10)),
        help="Weight for length log probability in candidate scoring (default: 0.10).",
    )
    parser.add_argument(
        "--terminal-weight",
        type=float,
        default=float(os.environ.get("TERMINAL_WEIGHT", 8.0)),
        help="Weight for enzymatic terminal prior bonus in candidate scoring (default: 8.0).",
    )
    parser.add_argument(
        "--hybrid-beam",
        dest="hybrid_beam",
        action="store_true",
        default=True,
        help="Enable hybrid beam decoding (default: True).",
    )
    parser.add_argument(
        "--no-hybrid-beam",
        dest="hybrid_beam",
        action="store_false",
        help="Disable hybrid beam decoding.",
    )
    parser.add_argument(
        "--trypsin-prior",
        dest="trypsin_prior",
        action="store_true",
        default=bool(int(os.environ.get("TRYPSIN_PRIOR", 1 if eval_cfg.trypsin_prior else 0))),
        help="Apply enzymatic C-terminal cleavage prior bonus (K/R) in candidate scoring (default: True).",
    )
    parser.add_argument(
        "--no-trypsin-prior",
        dest="trypsin_prior",
        action="store_false",
        help="Disable enzymatic C-terminal cleavage prior bonus.",
    )
    parser.add_argument(
        "--mask-self-attention",
        dest="mask_self_attention",
        action="store_true",
        default=bool(int(os.environ.get("MASK_SELF_ATTENTION", "1"))),
        help="Apply sequence padding mask to decoder self-attention (default: True).",
    )
    parser.add_argument(
        "--no-mask-self-attention",
        dest="mask_self_attention",
        action="store_false",
        help="Disable sequence padding mask in decoder self-attention.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=int(os.environ["MAX_BATCHES"]) if os.environ.get("MAX_BATCHES") else eval_cfg.max_batches,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ["MAX_SAMPLES"]) if os.environ.get("MAX_SAMPLES") else None,
        help="Maximum number of spectra to evaluate (e.g. 50000).",
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
    parser.add_argument("--save-predictions", default=os.environ.get("SAVE_PREDICTIONS"), help="Path to save predictions CSV")
    parser.add_argument("--save-plot", default=os.environ.get("SAVE_PLOT"))
    parser.add_argument(
        "--use-exact-dp-knapsack",
        dest="use_exact_dp_knapsack",
        action="store_true",
        default=True,
        help="Use exact reachability dynamic programming knapsack table (default: True).",
    )
    parser.add_argument(
        "--no-exact-dp-knapsack",
        dest="use_exact_dp_knapsack",
        action="store_false",
        help="Disable exact DP knapsack table and fall back to interval bounds.",
    )
    parser.add_argument(
        "--enzyme",
        default="trypsin",
        help="Digestion enzyme for evidence-conditioned cleavage prior ('trypsin' or 'none', default: trypsin).",
    )
    parser.add_argument(
        "--use-composite-ladders",
        dest="use_composite_ladders",
        action="store_true",
        default=True,
        help="Use multi-feature composite fragment ladder scoring (default: True).",
    )
    parser.add_argument(
        "--no-composite-ladders",
        dest="use_composite_ladders",
        action="store_false",
        help="Use simple explained intensity fragment scoring.",
    )
    parser.add_argument(
        "--implicit-length",
        action="store_true",
        default=bool(int(os.environ.get("IMPLICIT_LENGTH", 0))),
        help="Use implicit length prediction via <pad> tokens in decoder (freeze length classifier, set L=30).",
    )
    parser.add_argument(
        "--esm-mode",
        choices=["none", "in_loop", "late_stage", "rerank"],
        default=os.environ.get("ESM_MODE", "none"),
        help="ESM-2 guidance/rerank mode: none, in_loop, late_stage, rerank (default: none).",
    )
    parser.add_argument(
        "--esm-model-name",
        type=str,
        default=os.environ.get("ESM_MODEL_NAME", "facebook/esm2_t6_8M_UR50D"),
        help="HuggingFace model identifier for ESM-2 (default: facebook/esm2_t6_8M_UR50D).",
    )
    parser.add_argument(
        "--esm-guidance-scale",
        type=float,
        default=float(os.environ.get("ESM_GUIDANCE_SCALE", 0.2)),
        help="Weight/scale for ESM-2 guidance or candidate reranking (default: 0.2).",
    )
    parser.add_argument(
        "--esm-late-stage-t",
        type=float,
        default=float(os.environ.get("ESM_LATE_STAGE_T", 0.75)),
        help="Flow time threshold for late-stage annealed ESM guidance (default: 0.75).",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True, help="Disable automatic mixed precision (FP16/BF16)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("Provide --checkpoint or set CHECKPOINT environment variable.")
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0.10"))
        try:
            torch.cuda.set_per_process_memory_fraction(mem_frac, device.index if device.index is not None else 0)
        except Exception:
            pass

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

    ds = get_dataset(repo_id=args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    if vocabulary is None:
        vocabulary = build_vocabulary(include_ptms=True)

    model_cfg = infer_model_config_from_checkpoint(checkpoint)
    implicit_length = args.implicit_length or bool(model_cfg.get("implicit_length", False))
    model_cfg["implicit_length"] = implicit_length
    if implicit_length:
        args.top_k_lengths = 1
        args.use_knapsack_filter = False

    loader = build_dataloader(
        ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        max_peptide_length=DEFAULTS.model.max_length if implicit_length else None,
    )
    loader.dataset.top_k = DEFAULTS.data.top_k_peaks

    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device, model_cfg=model_cfg)
    load_models_from_checkpoint(
        checkpoint,
        spectrum_encoder,
        length_predictor,
        decoder,
        guidance,
    )
    total_params = sum(p.numel() for p in spectrum_encoder.parameters()) + sum(p.numel() for p in length_predictor.parameters()) + sum(p.numel() for p in decoder.parameters()) + sum(p.numel() for p in guidance.parameters())

    esm_guidance_helper = None
    if args.esm_mode != "none":
        print(f"Loading ESM-2 guidance helper ({args.esm_model_name}, mode={args.esm_mode}, scale={args.esm_guidance_scale})...")
        from model.esm_guidance import ESMGuidanceHelper
        esm_guidance_helper = ESMGuidanceHelper(
            model_name=args.esm_model_name,
            device=device,
            vocabulary=vocabulary,
            fp16=True,
        )

    print("==========================================================")
    print("  DFM De Novo Sequencing Generative Evaluation")
    print("==========================================================")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Model Parameters: {total_params/1e6:.2f}M (Enc: {model_cfg.get('encoder_layers', 6)}L, Dec: {model_cfg.get('decoder_blocks', 6)}L, Dim: {model_cfg.get('model_dim', 512)})")
    print(f"Implicit Length:  {implicit_length}")
    print(f"ESM Mode:         {args.esm_mode} (scale={args.esm_guidance_scale}, late_t={args.esm_late_stage_t})")
    print(f"Split:            {args.split} (total batches: {len(loader)})")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Workers:          {args.num_workers}")
    print(f"Guidance Scale:   {args.guidance_scale}")
    print(f"Inference Steps:  {args.num_steps}")
    print(f"Top-K Lengths:    {args.top_k_lengths}")
    print(f"Alpha:            {args.length_beam_alpha}")
    print(f"Knapsack Filter:  {args.use_knapsack_filter} (tol={args.knapsack_tol_da} Da)")
    print(f"Samples/Length:   {args.num_samples_per_length}")
    print(f"Device:           {device}")
    print("==========================================================")

    # NOTE: full Nine-Species evaluation (111k spectra) takes ~8-10 mins on an 80GB H100.
    # KnapsackDP filtering removes impossible prefix sums early and saves significant compute.
    # TODO: dump per-species subset TSVs automatically if species column is available in metadata.

    # 1. Run full evaluation collecting all outputs & scores
    reachability_dp = None
    if args.use_exact_dp_knapsack:
        reachability_dp = ExactReachabilityDP.get_default(device=device, tol_da=args.knapsack_tol_da)

    raw_metrics, details = evaluate_generative(
        loader,
        vocabulary,
        spectrum_encoder,
        length_predictor,
        decoder,
        guidance,
        args.scheduler,
        device,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        num_steps=args.num_steps,
        noising_scheme=args.noising_scheme,
        guidance_scale=args.guidance_scale,
        top_k_lengths=args.top_k_lengths,
        alpha=args.length_beam_alpha,
        beta=args.fragment_matching_weight,
        decoding_strategy=args.decoding_strategy,
        temperature=args.decoding_temperature,
        trypsin_prior=args.trypsin_prior,
        mask_self_attention=args.mask_self_attention,
        aa_mass_tolerance=DEFAULTS.eval.aa_mass_tolerance,
        prefix_mass_tolerance=DEFAULTS.eval.prefix_mass_tolerance,
        amp=args.amp,
        return_details=True,
        use_knapsack_filter=args.use_knapsack_filter,
        use_exact_dp_knapsack=args.use_exact_dp_knapsack,
        knapsack_tol_da=args.knapsack_tol_da,
        reachability_dp=reachability_dp,
        knapsack_guidance_penalty=args.knapsack_guidance_penalty,
        num_samples_per_length=args.num_samples_per_length,
        eta=args.eta,
        enzyme=args.enzyme,
        use_composite_ladders=args.use_composite_ladders,
        implicit_length=implicit_length,
        esm_guidance_helper=esm_guidance_helper,
        esm_mode=args.esm_mode,
        esm_guidance_scale=args.esm_guidance_scale,
        esm_late_stage_t=args.esm_late_stage_t,
        length_weight=args.length_weight,
        terminal_weight=args.terminal_weight,
        hybrid_beam=args.hybrid_beam,
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
    if "throughput" in details:
        print(f"Throughput:                           {details['throughput']:.1f} spectra/sec ({details['elapsed_time']:.1f}s total)")
    print(f"Unthresholded Exact (Strict) Accuracy: {metrics_unthresh.exact_peptide_accuracy * 100:.2f}%")
    print(f"Unthresholded Exact (I/L Equiv) Acc:   {metrics_unthresh.exact_peptide_accuracy_il * 100:.2f}%")
    print(f"Unthresholded Mass-Based Accuracy:     {metrics_unthresh.mass_peptide_accuracy * 100:.2f}%")
    print(f"Unthresholded Length Accuracy:        {metrics_unthresh.length_accuracy * 100:.2f}%")
    print(f"Unthresholded Amino Acid F1:          {metrics_unthresh.aa_f1 * 100:.2f}%")
    print(f"Peptide PR-AUC (Mass-Based Match):    {metrics_unthresh.pr_auc_mass:.4f}")
    print(f"Peptide pPR-AUC80 (Mass Match >= 80%):{metrics_unthresh.p_pr_auc80_mass:.4f}")
    print(f"Peptide PR-AUC (Exact Match):         {metrics_unthresh.pr_auc_exact:.4f}")
    print(f"Precision-Coverage AUC (Mass-Based):  {metrics_unthresh.auc_mass:.4f}")
    print(f"pAUPCC80 (Mass-Based >= 80% P):       {metrics_unthresh.pauc80_mass:.4f}")
    print(f"Precision-Coverage AUC (Exact Match): {metrics_unthresh.auc_exact:.4f}")
    if metrics_unthresh.coverage_at_80 > 0:
        print(f"Coverage @ 80% Precision:             {metrics_unthresh.coverage_at_80 * 100:.2f}%")
        print(f"Coverage @ 90% Precision:             {metrics_unthresh.coverage_at_90 * 100:.2f}%")
        print(f"Coverage @ 95% Precision:             {metrics_unthresh.coverage_at_95 * 100:.2f}%")
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

    # Stratified breakdown display
    if "stratified_breakdown" in details:
        print("\n=== Stratified Performance Breakdown ===")
        for group, vals in details["stratified_breakdown"].items():
            print(f"  {group:18s} | N={vals['count']:5d} | Exact(I=L): {vals['exact_acc']:.2%} | AA F1: {vals['aa_f1']:.4f}")

    # 5. Save output JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "metrics": metrics_primary.to_dict(),
            "unthresholded_metrics": metrics_unthresh.to_dict(),
            "exact_calibrated_metrics": metrics_exact_calib.to_dict(),
            "stratified_breakdown": details.get("stratified_breakdown"),
            "elapsed_time": details.get("elapsed_time"),
            "throughput": details.get("throughput"),
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
                "use_exact_dp_knapsack": args.use_exact_dp_knapsack,
                "knapsack_guidance_penalty": args.knapsack_guidance_penalty,
                "enzyme": args.enzyme,
                "use_composite_ladders": args.use_composite_ladders,
                "esm_mode": args.esm_mode,
                "esm_model_name": args.esm_model_name if args.esm_mode != "none" else None,
                "esm_guidance_scale": args.esm_guidance_scale if args.esm_mode != "none" else None,
                "esm_late_stage_t": args.esm_late_stage_t if args.esm_mode == "late_stage" else None,
            },
        }
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved metrics to {output_path}")

    # 6. Save Predictions CSV
    if args.save_predictions:
        import pandas as pd
        exact_matches_il = details.get("exact_matches_il")
        if exact_matches_il is None:
            exact_matches_il = [
                p.replace("I", "L") == t.replace("I", "L")
                for p, t in zip(details["predictions"], details["targets"])
            ]
        pred_df = pd.DataFrame({
            "target": details["targets"],
            "prediction": details["predictions"],
            "score": scores,
            "exact_match": exact_matches,
            "exact_match_il": exact_matches_il,
            "mass_match": mass_matches,
            "target_length": details["target_lengths"],
            "pred_length": details["predicted_lengths"],
        })
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

    # 7. Save PAUC Plot
    if args.save_plot:
        plot_pauc_curve(
            scores=scores,
            exact_matches=exact_matches,
            mass_matches=mass_matches,
            output_path=args.save_plot,
            title=f"{args.dataset_name.split('/')[-1]} ({args.split}) PR & Precision-Coverage Analysis (N={len(scores):,})",
            calibrated_threshold=primary_threshold,
            calibrated_coverage=metrics_primary.coverage,
            calibrated_precision=metrics_primary.peptide_precision_mass,
            calibrated_recall=metrics_primary.peptide_recall_mass,
        )


if __name__ == "__main__":
    main()
