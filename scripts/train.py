#!/usr/bin/env python3
"""Entry point for training discrete flow matching peptide models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bootstrap import setup_src_path

setup_src_path()

import torch
from torch.optim import AdamW

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset, get_output_aa_masses
from flow_matching.scheduler import cosine_scheduler
from train.factory import build_models
from train.io import (
    TrainingRunLogger,
    build_checkpoint_payload,
    load_checkpoint,
    load_models_from_checkpoint,
)
from train.train import training_loop


def parse_args():
    data_cfg = DEFAULTS.data
    train_cfg = DEFAULTS.train
    parser = argparse.ArgumentParser(description="Train DFM de novo peptide sequencing models.")
    parser.add_argument("--train-split", default=os.environ.get("TRAIN_SPLIT", data_cfg.train_split))
    parser.add_argument("--valid-split", default=os.environ.get("VALID_SPLIT", data_cfg.valid_split))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", train_cfg.batch_size)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", train_cfg.epochs)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LEARNING_RATE", train_cfg.learning_rate)))
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(os.environ.get("WEIGHT_DECAY", train_cfg.weight_decay)),
    )
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", train_cfg.num_workers)))
    parser.add_argument("--top-k-peaks", type=int, default=int(os.environ.get("TOP_K_PEAKS", data_cfg.top_k_peaks)))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", train_cfg.device if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", train_cfg.output_dir))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))
    parser.add_argument("--resume-from", default=os.environ.get("RESUME_FROM"))
    parser.add_argument("--eval-every", type=int, default=int(os.environ.get("EVAL_EVERY", train_cfg.eval_every)))
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=int(os.environ.get("EVAL_MAX_BATCHES", train_cfg.eval_max_batches)),
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=int(os.environ.get("INFERENCE_STEPS", train_cfg.inference_steps)),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = None
    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, map_location=device)

    run_name = args.run_name
    if run_name is None and checkpoint is not None:
        run_name = checkpoint.get("args", {}).get("run_name")
    if run_name is None:
        run_name = f"train_{torch.randint(0, 1_000_000, ()).item():06d}"

    run_logger = TrainingRunLogger(Path(args.output_dir), run_name)
    run_logger.log_run_start({"run_name": run_name, "args": vars(args), "defaults": DEFAULTS.to_dict()})

    train_ds = get_dataset(split=args.train_split, cache_dir=args.cache_dir)
    valid_ds = get_dataset(split=args.valid_split, cache_dir=args.cache_dir)

    vocabulary = build_vocabulary(train_ds)
    if checkpoint is not None and checkpoint["vocabulary"] != vocabulary:
        raise ValueError("Checkpoint vocabulary does not match the dataset vocabulary.")

    aa_masses = get_output_aa_masses(vocabulary).to(device)

    train_loader = build_dataloader(
        train_ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_dataloader(
        valid_ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # Apply top-k via dataset attribute after build (collate uses dataset).
    train_loader.dataset.top_k = args.top_k_peaks
    valid_loader.dataset.top_k = args.top_k_peaks

    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)

    optimizer = AdamW(
        list(spectrum_encoder.parameters())
        + list(length_predictor.parameters())
        + list(decoder.parameters())
        + list(guidance.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    initial_history = None
    if checkpoint is not None:
        load_models_from_checkpoint(
            checkpoint,
            spectrum_encoder,
            length_predictor,
            decoder,
            guidance,
            optimizer=optimizer,
        )
        run_logger.resume_from_checkpoint(checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        initial_history = checkpoint.get("history")
        print(
            f"Resumed from {args.resume_from} at epoch {start_epoch} "
            f"(best valid loss={run_logger.best_valid_loss:.4f})"
        )

    total_epochs = args.epochs
    if start_epoch >= total_epochs:
        print(
            f"Nothing to train: start_epoch={start_epoch} >= total_epochs={total_epochs}. "
            "Increase --epochs to continue."
        )
        return

    def on_epoch_end(
        *,
        epoch,
        total_epochs,
        history,
        train_metrics,
        valid_metrics,
        weights,
        generative_metrics=None,
    ):
        checkpoint_payload = build_checkpoint_payload(
            epoch=epoch,
            total_epochs=total_epochs,
            history=history,
            vocabulary=vocabulary,
            args={**vars(args), "run_name": run_name},
            optimizer=optimizer,
            spectrum_encoder=spectrum_encoder,
            length_predictor=length_predictor,
            decoder=decoder,
            guidance=guidance,
            best_valid_loss=run_logger.best_valid_loss,
        )
        run_logger.log_epoch(
            epoch=epoch,
            total_epochs=total_epochs,
            history=history,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            weights=weights,
            checkpoint_payload=checkpoint_payload,
            generative_metrics=generative_metrics,
        )

    history = training_loop(
        optimizer=optimizer,
        epochs=range(start_epoch, total_epochs),
        vocabulary=vocabulary,
        guidance=guidance,
        spectrum_encoder=spectrum_encoder,
        length_predictor=length_predictor,
        decoder=decoder,
        aa_masses=aa_masses,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scheduler=cosine_scheduler,
        device=device,
        total_epochs=total_epochs,
        initial_history=initial_history,
        epoch_end_callback=on_epoch_end,
        eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches,
        inference_steps=args.inference_steps,
        noising_scheme=DEFAULTS.train.noising_scheme,
        guidance_scale=DEFAULTS.train.guidance_scale,
    )

    print("Training finished.")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final valid loss: {history['valid_loss'][-1]:.4f}")
    if history.get("valid_peptide_recall"):
        print(f"Final valid peptide recall: {history['valid_peptide_recall'][-1]:.4f}")
    print(f"Best valid loss: {run_logger.best_valid_loss:.4f}")
    print(f"Best peptide recall: {run_logger.best_peptide_recall:.4f}")
    print(f"Latest checkpoint: {run_logger.latest_ckpt}")
    print(f"Best checkpoint (loss): {run_logger.best_ckpt}")
    print(f"Best checkpoint (peptide recall): {run_logger.best_metric_ckpt}")
    print(f"Metrics log: {run_logger.metrics_jsonl}")
    print(f"Artifacts saved to: {run_logger.run_dir}")


if __name__ == "__main__":
    main()
