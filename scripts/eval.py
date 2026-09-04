#!/usr/bin/env python3
"""Evaluate a checkpoint on a dataset split with InstaNovo-style metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bootstrap import setup_src_path

setup_src_path()

import torch

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from eval.metrics import format_metrics
from flow_matching.scheduler import cosine_scheduler
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint


def parse_args():
    data_cfg = DEFAULTS.data
    eval_cfg = DEFAULTS.eval
    parser = argparse.ArgumentParser(description="Evaluate DFM de novo peptide sequencing models.")
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
        help="Weight for mass mismatch penalty in length beam score (default: 0.1).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=int(os.environ["MAX_BATCHES"]) if os.environ.get("MAX_BATCHES") else eval_cfg.max_batches,
    )
    parser.add_argument("--output-json", default=os.environ.get("EVAL_OUTPUT_JSON"))
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
        # Look in same directory or parent directory (e.g. runs/<run_name>/checkpoints/../vocabulary.json)
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

    loader = build_dataloader(ds, vocabulary, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    loader.dataset.top_k = DEFAULTS.data.top_k_peaks

    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)
    load_models_from_checkpoint(
        checkpoint,
        spectrum_encoder,
        length_predictor,
        decoder,
        guidance,
    )

    metrics = evaluate_generative(
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
    )

    print(format_metrics(metrics))
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
