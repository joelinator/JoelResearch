#!/usr/bin/env python3
"""
Fine-tuning DFlowNovo on Human Core ProteomeTools (HC-PT) using Length-Stratified
Batch Sampling (>= 20% long peptides L in [19, 30] per batch) and Length-Weighted Loss.

Warm-starts from the best length-weighted checkpoint:
  artifacts/dfm_length_weighted_10ep/checkpoints/best-joint-gen-exact-epoch=01-exact=0.4746.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from config.defaults import DEFAULTS
from data.data import (
    SpectrumDataSet,
    build_vocabulary,
    get_dataset,
    parse_peptide,
    spectrum_collate,
)
from data.samplers import LengthStratifiedBatchSampler
from train.callbacks import EMACallback
from train.lightning import DFMLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DFlowNovo on HC-PT with Length-Stratified Batch Sampling (>=20% long)."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="artifacts/dfm_length_weighted_10ep/checkpoints/best-joint-gen-exact-epoch=01-exact=0.4746.ckpt",
        help="Path to best length-weighted checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/dfm_hcpt_length_stratified",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="InstaDeepAI/ms_proteometools",
        help="HuggingFace dataset repository ID.",
    )
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=500_000,
        help="Number of training samples drawn per epoch (default: 500k).",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=25_000,
        help="Number of validation samples evaluated per epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k-peaks", type=int, default=200)
    parser.add_argument("--peak-dropout", type=float, default=0.1)

    # Length-Stratification Parameters
    parser.add_argument("--p-short", type=float, default=0.40, help="Fraction of short peptides [7-12].")
    parser.add_argument("--p-medium", type=float, default=0.40, help="Fraction of medium peptides [13-18].")
    parser.add_argument("--p-long", type=float, default=0.20, help="Fraction of long peptides [19-30] (>=20%).")

    # Loss and Regularization
    parser.add_argument("--use-length-weighted-loss", action="store_true", default=True)
    parser.add_argument("--length-weight-alpha", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)

    # Architectural Specs (matching 59.5M SOTA)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-ff-dim", type=int, default=1536)
    parser.add_argument("--decoder-blocks", type=int, default=6)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1536)

    # Decoding Validation Proxies
    parser.add_argument("--guidance-scale", type=float, default=1.8)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--top-k-lengths", type=int, default=3)
    parser.add_argument("--length-beam-alpha", type=float, default=0.5)
    parser.add_argument("--use-knapsack-filter", action="store_true", default=True)
    parser.add_argument("--use-exact-dp-knapsack", action="store_true", default=True)
    parser.add_argument("--knapsack-guidance-penalty", type=float, default=2.0)
    parser.add_argument("--gen-eval-proxy-batches", type=int, default=4)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument(
        "--vram-target-pct",
        type=float,
        default=72.0,
        help="Target minimum VRAM utilization percentage (default: 72.0%%).",
    )

    return parser.parse_args()


class SamplerEpochCallback(pl.Callback):
    """Updates epoch seed on the batch sampler each epoch."""
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        dl = trainer.train_dataloader
        sampler = getattr(dl, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)


def extract_lengths_fast(dataset) -> np.ndarray:
    """Extract integer peptide lengths from sequence column in milliseconds via PyArrow."""
    try:
        import pyarrow.compute as pc
        table = getattr(dataset.data, "table", dataset.data)
        col_name = "sequence" if "sequence" in dataset.column_names else "modified_sequence"
        lengths = pc.utf8_length(table.column(col_name)).to_numpy()
        return lengths.astype(np.int32)
    except Exception as e:
        print(f"Fallback to parse_peptide extraction: {e}")
        col = "modified_sequence" if "modified_sequence" in dataset.column_names else "sequence"
        seqs = dataset[col]
        lengths = np.zeros(len(seqs), dtype=np.int32)
        for i, s in enumerate(seqs):
            lengths[i] = len(parse_peptide(s)) if s else 12
        return lengths


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"dfm_hcpt_stratified_{timestamp}"

    print("=" * 80)
    print("  DFM Fine-Tuning: HC-PT Domain Adaptation with Length-Stratified Sampler")
    print("=" * 80)
    print(f"Run Name:            {args.run_name}")
    print(f"Output Directory:    {output_dir}")
    print(f"Warm-Start Checkpoint: {args.resume_from}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Epochs:              {args.epochs}")
    print(f"Samples per Epoch:   {args.samples_per_epoch:,}")
    print(f"Learning Rate:       {args.lr}")
    print(f"Strata Target:       {args.p_short*100:.0f}% Short [7-12], {args.p_medium*100:.0f}% Med [13-18], {args.p_long*100:.0f}% Long [19-30]")
    print(f"Guidance Penalty:    {args.knapsack_guidance_penalty}")
    print("=" * 80)

    # 1. Vocabulary
    vocabulary = build_vocabulary(include_ptms=True)
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f, indent=2)

    # 2. Datasets
    print("\nLoading HC-PT dataset from cache...")
    raw_train = get_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)
    raw_val = get_dataset(args.dataset_name, split="validation", cache_dir=args.cache_dir)
    print(f"  HC-PT: train={len(raw_train):,}, val={len(raw_val):,}")

    # Extract lengths for stratified sampling
    print("Extracting peptide lengths for stratified batching...")
    train_lengths = extract_lengths_fast(raw_train)

    spec_train = SpectrumDataSet(
        raw_train,
        vocabulary,
        top_k=args.top_k_peaks,
        peak_dropout_prob=args.peak_dropout,
        is_train=True,
    )
    spec_val = SpectrumDataSet(
        raw_val,
        vocabulary,
        top_k=args.top_k_peaks,
        peak_dropout_prob=0.0,
        is_train=False,
    )

    # 3. Stratified Batch Sampler
    num_batches = args.samples_per_epoch // args.batch_size
    train_batch_sampler = LengthStratifiedBatchSampler(
        lengths=train_lengths,
        batch_size=args.batch_size,
        short_range=(7, 12),
        medium_range=(13, 18),
        long_range=(19, 30),
        p_short=args.p_short,
        p_medium=args.p_medium,
        p_long=args.p_long,
        shuffle=True,
        seed=args.seed,
        num_batches_per_epoch=num_batches,
    )

    pin = torch.cuda.is_available()
    collate = partial(spectrum_collate, vocab=vocabulary)

    train_loader = DataLoader(
        spec_train,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Subsample validation loader for fast epoch checkpoints
    val_indices = np.random.default_rng(args.seed).choice(
        len(spec_val), size=min(args.val_samples, len(spec_val)), replace=False
    )
    val_subset = torch.utils.data.Subset(spec_val, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=min(args.batch_size, 512),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # 4. Lightning Module & Warm Start
    model_cfg = {
        "model_dim": args.model_dim,
        "encoder_layers": args.encoder_layers,
        "encoder_heads": args.encoder_heads,
        "encoder_ff_dim": args.encoder_ff_dim,
        "decoder_blocks": args.decoder_blocks,
        "decoder_heads": args.decoder_heads,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "max_charge": DEFAULTS.model.max_charge,
        "max_length": DEFAULTS.model.max_length,
        "min_length": DEFAULTS.model.min_length,
        "dropout": DEFAULTS.model.dropout,
    }
    args_dict = vars(args)
    args_dict["model_cfg"] = model_cfg
    args_dict["learning_rate"] = args.lr
    args_dict["eval_chunk_size"] = 256

    model = DFMLightningModule(vocabulary, args_dict)

    if args.resume_from and Path(args.resume_from).exists():
        print(f"\nLoading warm-start model weights from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        model_sd = model.state_dict()
        matching_sd = {
            k: v for k, v in state_dict.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        missing, unexpected = model.load_state_dict(matching_sd, strict=False)
        print(f"  Warm-start weights transferred: {len(matching_sd)} keys matched with identical shapes.")

    # 5. Callbacks and Trainer
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-hcpt-stratified-epoch={epoch:02d}-exact={valid_gen_exact_match:.4f}",
        monitor="valid_gen_exact_match",
        mode="max",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    sampler_cb = SamplerEpochCallback()

    callbacks = [checkpoint_cb, lr_monitor, sampler_cb]
    if args.use_ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    loggers = [
        TensorBoardLogger(save_dir=output_dir, name="tb_logs"),
        CSVLogger(save_dir=output_dir, name="csv_logs"),
    ]

    trainer_kwargs = dict(
        default_root_dir=output_dir,
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )
    if args.limit_train_batches > 0:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches > 0:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)

    print("\nStarting Fine-Tuning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
