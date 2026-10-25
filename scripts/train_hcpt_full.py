#!/usr/bin/env python3
"""
Full-scale training / fine-tuning of DFlowNovo on the complete ProteomeTools (HC-PT) dataset.

Dataset: InstaDeepAI/ms_proteometools (train split: 2,132,847 spectra across 7 shards).
Warm-starts from: models/dfm_massive_kb.ckpt (pre-trained on 28M MassIVE-KB spectra).
Hardware: Optimized for NVIDIA H100 80GB (Batch Size: 1024, BF16-Mixed, ~3,300 spec/s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if p != str(Path(__file__).resolve().parent)]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from train.callbacks import EMACallback
from train.lightning import DFMLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DFlowNovo on full HC-PT (ProteomeTools) dataset."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="InstaDeepAI/ms_proteometools",
        help="HuggingFace dataset repository name.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Training split name.",
    )
    parser.add_argument(
        "--valid-split",
        type=str,
        default="validation",
        help="Validation split name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Path to cached datasets.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="models/dfm_massive_kb.ckpt",
        help="Checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size per step (1024 fits ~39GB on H100).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Total training epochs (5 epochs = ~10.66M spectrum encounters, ~55 min).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="wsd",
        choices=["wsd", "cosine", "constant"],
        help="Learning rate schedule.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup steps fraction.",
    )
    parser.add_argument(
        "--stable-ratio",
        type=float,
        default=0.65,
        help="Stable phase fraction for WSD schedule.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Minimum LR ratio floor at end of decay.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--top-k-peaks",
        type=int,
        default=200,
        help="Top-k peaks per spectrum (standard 200).",
    )
    parser.add_argument(
        "--peak-dropout",
        type=float,
        default=0.15,
        help="Peak dropout regularization probability.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/dfm_hcpt_full_finetune",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="dfm_hcpt_full_5ep",
        help="Run identifier name.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--gen-eval-proxy-batches",
        type=int,
        default=5,
        help="Validation batches for generative de novo evaluation proxy.",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=256,
        help="Micro-batch chunk size during generative evaluation.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Maintain EMA shadow weights.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay coefficient.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction or number of train batches.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction or number of val batches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"================================================================")
    print(f"  DFlowNovo Full HC-PT Training Pipeline")
    print(f"  Run Name:       {args.run_name}")
    print(f"  Dataset:        {args.dataset_name} ({args.train_split} -> {args.valid_split})")
    print(f"  Warm-Start:     {args.resume_from}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning Rate:  {args.lr} ({args.lr_schedule})")
    print(f"  Output Dir:     {output_dir}")
    print(f"================================================================")

    pl.seed_everything(args.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    train_ds = get_dataset(repo_id=args.dataset_name, split=args.train_split, cache_dir=args.cache_dir)
    valid_ds = get_dataset(repo_id=args.dataset_name, split=args.valid_split, cache_dir=args.cache_dir)
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Valid samples: {len(valid_ds):,}")

    vocabulary = build_vocabulary(include_ptms=True)
    print(f"  Vocabulary size: {len(vocabulary)} tokens")

    # Save vocabulary for downstream inference and rerankers
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f, indent=2)

    pin = torch.cuda.is_available()
    train_loader = build_dataloader(
        train_ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        top_k=args.top_k_peaks,
        peak_dropout_prob=args.peak_dropout,
        is_train=True,
    )
    valid_loader = build_dataloader(
        valid_ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        top_k=args.top_k_peaks,
        peak_dropout_prob=0.0,
        is_train=False,
    )

    # 2. Build Model Configuration
    print("\n[2/5] Building model configuration...")
    model_cfg = {
        "model_dim": DEFAULTS.model.model_dim,
        "encoder_layers": DEFAULTS.model.encoder_layers,
        "encoder_heads": DEFAULTS.model.encoder_heads,
        "encoder_ff_dim": DEFAULTS.model.encoder_ff_dim,
        "decoder_blocks": DEFAULTS.model.decoder_blocks,
        "decoder_heads": DEFAULTS.model.decoder_heads,
        "mlp_hidden_dim": DEFAULTS.model.mlp_hidden_dim,
        "max_charge": DEFAULTS.model.max_charge,
        "max_length": DEFAULTS.model.max_length,
        "min_length": DEFAULTS.model.min_length,
        "dropout": DEFAULTS.model.dropout,
    }

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps:     {total_steps:,}")

    args_dict = vars(args)
    args_dict["model_cfg"] = model_cfg
    args_dict["learning_rate"] = args.lr
    args_dict["total_steps"] = total_steps
    args_dict["use_length_weighted_loss"] = True
    args_dict["length_weight_alpha"] = 0.5
    args_dict["inference_steps"] = 20
    args_dict["top_k_lengths"] = 3
    args_dict["length_beam_alpha"] = 0.01
    args_dict["guidance_scale"] = 1.5
    args_dict["decoding_strategy"] = "confidence"
    args_dict["decoding_temperature"] = 0.0
    args_dict["fragment_matching_weight"] = 0.5
    args_dict["trypsin_prior"] = True
    args_dict["mask_self_attention"] = True
    args_dict["use_knapsack_filter"] = True

    model = DFMLightningModule(vocabulary, args_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model initialized: {total_params / 1e6:.2f}M parameters")

    # 3. Warm-Start from Checkpoint
    if args.resume_from and Path(args.resume_from).is_file():
        print(f"\n[3/5] Warm-starting model weights from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Weights loaded successfully: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys sample: {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys sample: {unexpected[:5]}")
    else:
        print(f"\n[3/5] No valid resume checkpoint found; training from scratch.")

    # 4. Configure Checkpointing and Logging
    print("\n[4/5] Setting up callbacks and loggers...")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_exact_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-gen-exact-{epoch:02d}-{valid_gen_exact_match_il:.4f}",
        monitor="valid_gen_exact_match_il",
        mode="max",
        save_top_k=2,
        save_weights_only=True,
    )

    best_loss_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-val-loss-{epoch:02d}-{valid_loss:.4f}",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )

    last_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        save_weights_only=True,
        save_on_train_epoch_end=True,
    )

    callbacks = [
        best_exact_cb,
        best_loss_cb,
        last_cb,
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.use_ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    csv_logger = CSVLogger(save_dir=args.output_dir, name=args.run_name)
    tb_logger = TensorBoardLogger(save_dir=args.output_dir, name=args.run_name)

    limit_train = int(args.limit_train_batches) if args.limit_train_batches > 1.0 else args.limit_train_batches
    limit_val = int(args.limit_val_batches) if args.limit_val_batches > 1.0 else args.limit_val_batches

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.eval_every,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    # 5. Execute Training
    print("\n[5/5] Launching PyTorch Lightning Trainer...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    print("\n================================================================")
    print("  Training Complete!")
    print(f"  Best Generative Exact I/L Checkpoint: {best_exact_cb.best_model_path}")
    print(f"  Best Validation Loss Checkpoint:     {best_loss_cb.best_model_path}")
    print(f"  Last Checkpoint:                     {last_cb.last_model_path}")
    print("================================================================")


if __name__ == "__main__":
    main()
