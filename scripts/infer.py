#!/usr/bin/env python3
"""Entry point for de novo peptide inference."""

from __future__ import annotations

import argparse
import os

from bootstrap import move_batch_to_device, setup_src_path

setup_src_path()

import torch

from data.data import build_dataloader, build_vocabulary, get_dataset
from flow_matching.scheduler import cosine_scheduler
from inference.predict import predict_peptide
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run DFM de novo peptide inference.")
    parser.add_argument("--split", default=os.environ.get("INFER_SPLIT", "test[:1%]"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NUM_STEPS", "20")))
    parser.add_argument(
        "--noising-scheme",
        choices=["uniform", "mask"],
        default=os.environ.get("NOISING_SCHEME", "uniform"),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(os.environ.get("GUIDANCE_SCALE", "1.0")),
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=int(os.environ.get("TOP_K_LENGTHS", "3")),
        help="Number of top length candidates to decode in parallel (default: 3).",
    )
    parser.add_argument(
        "--length-beam-alpha",
        type=float,
        default=float(os.environ.get("LENGTH_BEAM_ALPHA", "0.1")),
        help="Weight for mass mismatch penalty in length beam score (default: 0.1).",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max-batches", type=int, default=int(os.environ.get("MAX_BATCHES", "1")))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT"),
        help="Path to a training checkpoint (latest.pt or best_valid.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    vocabulary = None
    if checkpoint is not None:
        vocabulary = checkpoint.get("vocabulary")
        if vocabulary is None and args.checkpoint:
            ckpt_p = Path(args.checkpoint).resolve()
            for cand in [ckpt_p.parent / "vocabulary.json", ckpt_p.parent.parent / "vocabulary.json"]:
                if cand.exists():
                    import json
                    with cand.open() as f:
                        vocabulary = json.load(f)
                    break

    ds = get_dataset(split=args.split, cache_dir=args.cache_dir)
    if vocabulary is None:
        vocabulary = build_vocabulary(ds)
    loader = build_dataloader(ds, vocabulary, batch_size=args.batch_size, shuffle=False)

    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)

    if checkpoint is not None:
        load_models_from_checkpoint(
            checkpoint,
            spectrum_encoder,
            length_predictor,
            decoder,
            guidance,
        )
        print(f"Loaded checkpoint: {args.checkpoint}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        (
            mz_array,
            intensity_array,
            precursor_mass,
            precursor_charge,
            _sequence,
            mz_complementary,
            _length,
            _padded_mask,
            spectrum_mask,
        ) = move_batch_to_device(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            token_ids, predicted_lengths, sequences = predict_peptide(
                mz_array=mz_array,
                intensity_array=intensity_array,
                precursor_mass=precursor_mass,
                precursor_charge=precursor_charge,
                mz_complementary=mz_complementary,
                spectrum_mask=spectrum_mask,
                vocabulary=vocabulary,
                spectrum_encoder=spectrum_encoder,
                length_predictor=length_predictor,
                decoder=decoder,
                guidance=guidance,
                scheduler=cosine_scheduler,
                num_steps=args.num_steps,
                noising_scheme=args.noising_scheme,
                guidance_scale=args.guidance_scale,
                top_k_lengths=args.top_k_lengths,
                alpha=args.length_beam_alpha,
            )

        for idx, sequence in enumerate(sequences):
            print(
                f"sample={batch_idx * args.batch_size + idx} "
                f"predicted_length={int(predicted_lengths[idx])} "
                f"sequence={sequence}"
            )


if __name__ == "__main__":
    main()

