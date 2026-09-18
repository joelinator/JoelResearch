#!/usr/bin/env python3
"""
Joint Balanced Training Script for DFM De Novo Peptide Sequencing.
Trains on a 1:1 balanced mixture of:
  - Nine-Species Benchmark (Train: 487,312 spectra) - Biological cell lysates
  - HC-PT ProteomeTools (Train: 2,132,847 spectra) - Synthetic human tryptic peptides

Each epoch samples 500,000 spectra from each domain (1,000,000 spectra/epoch total).
Evaluates on a joint validation set containing both biological and synthetic peptides.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from config.defaults import DEFAULTS
from data.data import (
    SpectrumDataSet,
    build_vocabulary,
    get_dataset,
    spectrum_collate,
)
from train.callbacks import EMACallback
from train.lightning import DFMLightningModule


class BalancedJointDataset(Dataset):
    """
    A 1:1 balanced joint dataset combining two underlying datasets.
    Resamples indices each epoch to cycle through larger datasets without bias.
    """

    def __init__(
        self,
        ds_a: Dataset,
        ds_b: Dataset,
        samples_per_epoch: int = 1_000_000,
        seed: int = 42,
    ):
        super().__init__()
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.len_a = len(ds_a)
        self.len_b = len(ds_b)
        self.samples_per_epoch = samples_per_epoch
        self.half = samples_per_epoch // 2
        self.seed = seed
        self.epoch = 0
        self.rng = np.random.default_rng(seed)
        self._resample_indices()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.rng = np.random.default_rng(self.seed + epoch * 10007)
        self._resample_indices()

    def _resample_indices(self):
        replace_a = self.half > self.len_a
        self.indices_a = self.rng.choice(self.len_a, size=self.half, replace=replace_a)
        replace_b = self.half > self.len_b
        self.indices_b = self.rng.choice(self.len_b, size=self.half, replace=replace_b)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        if idx % 2 == 0:
            return self.ds_a[int(self.indices_a[idx // 2])]
        else:
            return self.ds_b[int(self.indices_b[idx // 2])]


class JointResampleCallback(pl.Callback):
    """Callback to trigger index resampling on each training epoch start."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        dl = trainer.train_dataloader
        ds = getattr(dl, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(trainer.current_epoch)
            print(f"\n[JointResampleCallback] Resampled 1:1 indices for epoch {trainer.current_epoch} (total samples={len(ds):,})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFM on balanced Nine-Species + HC-PT dataset.")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument("--output-dir", default="artifacts/dfm_joint_balanced_8ep")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume-from",
        default="artifacts/dfm_pl_ninespecies_finetune_phase2_10ep/checkpoints/best-gen-exact-epoch=02-exact=0.6270.ckpt",
        help="Checkpoint to initialize model weights from.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=1_000_000, help="Total train samples per epoch (50%% each domain)")
    parser.add_argument("--val-samples", type=int, default=20_000, help="Total validation samples (50%% each domain)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k-peaks", type=int, default=200)
    parser.add_argument("--peak-dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument(
        "--gen-eval-proxy-batches",
        type=int,
        default=5,
        help="Number of validation batches (micro-batches of 256) decoded for generative proxy evaluation.",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.8)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--top-k-lengths", type=int, default=3)
    parser.add_argument("--length-beam-alpha", type=float, default=0.5)
    parser.add_argument("--use-knapsack-filter", action="store_true", default=True)
    parser.add_argument("--knapsack-tol-da", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"dfm_joint_balanced_{timestamp}"

    print("==================================================================")
    print("      DFM Joint Balanced Training (Nine-Species + HC-PT)          ")
    print("==================================================================")
    print(f"Run Name:            {args.run_name}")
    print(f"Output Directory:    {output_dir}")
    print(f"Warm-Start Checkpoint: {args.resume_from}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Epochs:              {args.epochs}")
    print(f"Samples per Epoch:   {args.samples_per_epoch:,} (50% NS, 50% HC-PT)")
    print(f"Learning Rate:       {args.lr}")
    print(f"Device:              {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("==================================================================")

    # 1. Build Vocabulary
    vocabulary = build_vocabulary(include_ptms=True)
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f, indent=2)
    print(f"Loaded vocabulary ({len(vocabulary)} tokens)")

    # 2. Load Datasets from Cache
    print("Loading Nine-Species dataset from cache...")
    raw_ns_train = get_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="train", cache_dir=args.cache_dir)
    raw_ns_val = get_dataset("InstaDeepAI/ms_ninespecies_benchmark", split="validation", cache_dir=args.cache_dir)
    print(f"  Nine-Species: train={len(raw_ns_train):,}, val={len(raw_ns_val):,}")

    print("Loading HC-PT dataset from cache...")
    raw_hc_train = get_dataset("InstaDeepAI/ms_proteometools", split="train", cache_dir=args.cache_dir)
    raw_hc_val = get_dataset("InstaDeepAI/ms_proteometools", split="validation", cache_dir=args.cache_dir)
    print(f"  HC-PT:        train={len(raw_hc_train):,}, val={len(raw_hc_val):,}")

    # 3. Wrap in SpectrumDataSets
    spec_ns_train = SpectrumDataSet(raw_ns_train, vocabulary, top_k=args.top_k_peaks, peak_dropout_prob=args.peak_dropout, is_train=True)
    spec_hc_train = SpectrumDataSet(raw_hc_train, vocabulary, top_k=args.top_k_peaks, peak_dropout_prob=args.peak_dropout, is_train=True)

    spec_ns_val = SpectrumDataSet(raw_ns_val, vocabulary, top_k=args.top_k_peaks, peak_dropout_prob=0.0, is_train=False)
    spec_hc_val = SpectrumDataSet(raw_hc_val, vocabulary, top_k=args.top_k_peaks, peak_dropout_prob=0.0, is_train=False)

    # 4. Construct Balanced Joint Datasets
    joint_train_ds = BalancedJointDataset(
        spec_ns_train,
        spec_hc_train,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )

    joint_val_ds = BalancedJointDataset(
        spec_ns_val,
        spec_hc_val,
        samples_per_epoch=args.val_samples,
        seed=args.seed + 999,
    )

    pin = torch.cuda.is_available()
    collate = partial(spectrum_collate, vocab=vocabulary)

    train_loader = DataLoader(
        joint_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        joint_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print(f"Train Dataloader: {len(train_loader)} batches of {args.batch_size} ({len(joint_train_ds):,} samples/epoch)")
    print(f"Val Dataloader:   {len(val_loader)} batches of {args.batch_size} ({len(joint_val_ds):,} samples)")

    # 5. Initialize Lightning Module with warm-started weights
    args_dict = vars(args)
    args_dict["learning_rate"] = args.lr
    args_dict["eval_chunk_size"] = 256

    model = DFMLightningModule(vocabulary, args_dict)

    if args.resume_from and Path(args.resume_from).exists():
        print(f"\nLoading warm-start model weights from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Warm-start weights loaded (missing keys: {len(missing)}, unexpected keys: {len(unexpected)}).")
        print("  Optimizer & Scheduler reset to start fresh at epoch 0 with cosine warmup schedule.")

    # 6. Callbacks & Loggers
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_gen_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-joint-gen-exact-epoch={epoch:02d}-exact={valid_gen_exact_match:.4f}",
        monitor="valid_gen_exact_match",
        mode="max",
        save_top_k=2,
        auto_insert_metric_name=False,
    )

    last_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        save_on_train_epoch_end=True,
    )

    callbacks = [
        best_gen_cb,
        last_cb,
        JointResampleCallback(),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.use_ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    csv_logger = CSVLogger(save_dir=args.output_dir, name="logs_csv")
    tb_logger = TensorBoardLogger(save_dir=args.output_dir, name="logs_tb")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
        precision="bf16-mixed" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "16-mixed",
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )

    print("\nStarting Lightning Trainer...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\n==================================================================")
    print("      JOINT BALANCED TRAINING COMPLETE                           ")
    print("==================================================================")
    print(f"Best checkpoint: {best_gen_cb.best_model_path} (exact={best_gen_cb.best_model_score})")
    print(f"Latest checkpoint: {last_cb.last_model_path}")


if __name__ == "__main__":
    main()
