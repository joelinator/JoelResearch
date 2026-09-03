#!/usr/bin/env python3
"""
Multi-GPU training script using PyTorch Lightning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from train.lightning import DFMLightningModule


def parse_args():
    data_cfg = DEFAULTS.data
    train_cfg = DEFAULTS.train
    parser = argparse.ArgumentParser(description="Train DFM de novo peptide sequencing models with PyTorch Lightning.")
    parser.add_argument("--train-split", default=os.environ.get("TRAIN_SPLIT", data_cfg.train_split))
    parser.add_argument("--valid-split", default=os.environ.get("VALID_SPLIT", data_cfg.valid_split))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", train_cfg.batch_size)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", train_cfg.epochs)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LEARNING_RATE", train_cfg.learning_rate)))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", train_cfg.weight_decay)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", train_cfg.num_workers)))
    parser.add_argument("--top-k-peaks", type=int, default=int(os.environ.get("TOP_K_PEAKS", data_cfg.top_k_peaks)))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", train_cfg.output_dir))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))
    parser.add_argument("--resume-from", default=os.environ.get("RESUME_FROM"))
    parser.add_argument("--noising-scheme", default=train_cfg.noising_scheme)
    parser.add_argument("--compile", action="store_true", default=train_cfg.compile, help="Compile models")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=train_cfg.amp)
    parser.add_argument("--eval-every", type=int, default=int(os.environ.get("EVAL_EVERY", 1)))
    parser.add_argument("--limit-val-batches", type=int, default=int(os.environ.get("EVAL_MAX_BATCHES", 0)))
    parser.add_argument(
        "--gen-eval-proxy-batches",
        type=int,
        default=int(os.environ.get("GEN_EVAL_PROXY_BATCHES", 5)),
        help="Number of validation batches to run generative de novo evaluation proxy on (default: 5).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(os.environ.get("GUIDANCE_SCALE", 1.5)),
        help="Classifier-free guidance scale for generative proxy evaluation (default: 1.5).",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=int(os.environ.get("INFERENCE_STEPS", 20)),
        help="Flow matching integration steps for generative proxy evaluation (default: 20).",
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=int(os.environ.get("TOP_K_LENGTHS", 3)),
        help="Top-K length candidates for generative proxy evaluation (default: 3).",
    )
    parser.add_argument(
        "--length-beam-alpha",
        type=float,
        default=float(os.environ.get("LENGTH_BEAM_ALPHA", 0.01)),
        help="Mass mismatch penalty weight for generative proxy evaluation (default: 0.01).",
    )
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=int(os.environ.get("CHECKPOINT_EVERY_N_EPOCHS", 10)),
        help="Save unconditional periodic checkpoint every N epochs to capture grokking (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"dfm_pl_run_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PyTorch Lightning Training: {args.run_name} ===")
    print(f"Effective Batch Size: {args.batch_size * args.accumulate_grad_batches} (Batch: {args.batch_size}, Accumulate: {args.accumulate_grad_batches})")
    pl.seed_everything(args.seed, workers=True)

    train_ds = get_dataset(split=args.train_split, cache_dir=args.cache_dir)
    valid_ds = get_dataset(split=args.valid_split, cache_dir=args.cache_dir)

    vocabulary = build_vocabulary(train_ds)
    print(f"Vocabulary size: {len(vocabulary)}")

    pin = torch.cuda.is_available()
    train_loader = build_dataloader(
        train_ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    valid_loader = build_dataloader(
        valid_ds,
        vocabulary,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    train_loader.dataset.top_k = args.top_k_peaks
    valid_loader.dataset.top_k = args.top_k_peaks

    args_dict = vars(args)
    args_dict["learning_rate"] = args.lr
    
    vocab_path = output_dir / args.run_name / "vocabulary.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f)

    model = DFMLightningModule(vocabulary, args_dict)

    csv_logger = CSVLogger(save_dir=args.output_dir, name=args.run_name)
    tb_logger = TensorBoardLogger(save_dir=args.output_dir, name=args.run_name)
    loggers = [csv_logger, tb_logger]
    
    checkpoint_dir = output_dir / args.run_name / "checkpoints"

    # Tier 1: Checkpoint best models based on true Generative Exact Peptide Match
    best_gen_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-gen-exact-epoch={epoch:02d}-exact={valid_gen_exact_match:.4f}",
        monitor="valid_gen_exact_match",
        mode="max",
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    # Tier 2: Unconditional periodic checkpoints every N epochs (e.g. 10 epochs to capture grokking)
    periodic_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="periodic-epoch={epoch:02d}",
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    # Tier 3: Monitored validation loss (continuous loss minimum)
    best_loss_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-loss-epoch={epoch:02d}-loss={valid_loss:.4f}",
        monitor="valid_loss",
        mode="min",
        save_top_k=2,
        auto_insert_metric_name=False,
    )

    # Tier 4: Guaranteed latest snapshot at the end of every training epoch
    last_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
        save_on_train_epoch_end=True,
    )

    checkpoint_callbacks = [best_gen_cb, periodic_cb, best_loss_cb, last_cb]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.eval_every,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        precision="bf16-mixed" if (args.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) 
                  else ("16-mixed" if args.amp else "32-true"),
        logger=loggers,
        callbacks=[*checkpoint_callbacks, LearningRateMonitor(logging_interval="step")],
        default_root_dir=args.output_dir,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=args.resume_from)
    print(f"=== Training complete ===")
    print(f"Best generative model: {best_gen_cb.best_model_path} (score={best_gen_cb.best_model_score})")
    print(f"Best loss model:       {best_loss_cb.best_model_path} (score={best_loss_cb.best_model_score})")


if __name__ == "__main__":
    main()
