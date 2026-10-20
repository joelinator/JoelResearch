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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if p != str(Path(__file__).resolve().parent)]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from train.callbacks import EMACallback
from train.lightning import DFMLightningModule


def get_recommended_batch_size(target_vram_pct: float = 0.85, encoder_layers: int = 6, decoder_blocks: int = 6) -> int:
    """Calculates recommended batch size to reach >=80% VRAM based on available GPU capacity and model depth."""
    if not torch.cuda.is_available():
        return 64
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb >= 70:  # 80GB H100 / A100: 1152 for deep 9L, 1792 for 6L achieves ~80-85% VRAM
        if encoder_layers >= 8 or decoder_blocks >= 8:
            return 1152 if target_vram_pct >= 0.80 else 896
        return 1792 if target_vram_pct >= 0.80 else 1024
    elif total_gb >= 35:  # 40GB A100
        return 512 if (encoder_layers >= 8 or decoder_blocks >= 8) else 896
    elif total_gb >= 20:  # 24GB RTX 3090 / 4090 / A10G
        return 256 if (encoder_layers >= 8 or decoder_blocks >= 8) else 448
    return 128


def parse_args():
    data_cfg = DEFAULTS.data
    train_cfg = DEFAULTS.train
    parser = argparse.ArgumentParser(description="Train DFM de novo peptide sequencing models with PyTorch Lightning.")
    parser.add_argument("--dataset-name", default=os.environ.get("DATASET_NAME", "InstaDeepAI/ms_ninespecies_benchmark"))
    parser.add_argument("--train-split", default=os.environ.get("TRAIN_SPLIT", data_cfg.train_split))
    parser.add_argument("--valid-split", default=os.environ.get("VALID_SPLIT", data_cfg.valid_split))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ["BATCH_SIZE"]) if os.environ.get("BATCH_SIZE") else None,
        help="Batch size. If None and --optimize-vram is set, auto-tunes to saturate >=80%% VRAM.",
    )
    parser.add_argument(
        "--optimize-vram",
        action="store_true",
        default=bool(int(os.environ.get("OPTIMIZE_VRAM", 1))),
        help="Auto-tune batch size to saturate >=80%% VRAM on detected GPU (default: True).",
    )
    parser.add_argument(
        "--target-vram-pct",
        type=float,
        default=float(os.environ.get("TARGET_VRAM_PCT", 0.85)),
        help="Target GPU VRAM utilization ratio (default: 0.85 for >=80%% VRAM saturation).",
    )
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", train_cfg.epochs)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LEARNING_RATE", train_cfg.learning_rate)))
    parser.add_argument("--lr-schedule", type=str, default=os.environ.get("LR_SCHEDULE", "wsd"), choices=["wsd", "cosine", "constant", "flat"], help="Learning rate schedule (default: wsd).")
    parser.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("WARMUP_RATIO", 0.05)), help="Fraction of total steps for linear warmup (default: 0.05).")
    parser.add_argument("--stable-ratio", type=float, default=float(os.environ.get("STABLE_RATIO", 0.70)), help="Fraction of total steps for stable phase (default: 0.70).")
    parser.add_argument("--min-lr-ratio", type=float, default=float(os.environ.get("MIN_LR_RATIO", 0.01)), help="Minimum LR ratio at end of decay (default: 0.01).")
    parser.add_argument(
        "--use-length-weighted-loss",
        dest="use_length_weighted_loss",
        action="store_true",
        default=bool(int(os.environ.get("USE_LENGTH_WEIGHTED_LOSS", 1))),
        help="Enable length-weighted peptide loss to balance short vs long peptides (default: True).",
    )
    parser.add_argument(
        "--no-length-weighted-loss",
        dest="use_length_weighted_loss",
        action="store_false",
        help="Disable length-weighted loss.",
    )
    parser.add_argument(
        "--length-weight-alpha",
        type=float,
        default=float(os.environ.get("LENGTH_WEIGHT_ALPHA", 0.5)),
        help="Alpha power for length-weighted loss (default: 0.5).",
    )
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
    parser.add_argument("--limit-train-batches", type=int, default=int(os.environ.get("TRAIN_MAX_BATCHES", 0)))
    parser.add_argument("--limit-val-batches", type=int, default=int(os.environ.get("EVAL_MAX_BATCHES", 0)))
    parser.add_argument(
        "--gen-eval-proxy-batches",
        type=int,
        default=int(os.environ.get("GEN_EVAL_PROXY_BATCHES", 5)),
        help="Number of validation batches to run generative de novo evaluation proxy on (default: 5).",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=int(os.environ.get("EVAL_CHUNK_SIZE", 256)),
        help="Micro-batch chunk size streamed to GPU during generative evaluation to prevent OOM (default: 256).",
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
        "--reset-lr-on-resume",
        dest="reset_lr_on_resume",
        action="store_true",
        default=bool(int(os.environ.get("RESET_LR_ON_RESUME", 1))),
        help="Reset optimizer state to use learning rate from args when resuming (default: True).",
    )
    parser.add_argument(
        "--no-reset-lr-on-resume",
        dest="reset_lr_on_resume",
        action="store_false",
        help="Preserve saved optimizer state and decayed learning rate when resuming from checkpoint.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        default=bool(int(os.environ.get("WEIGHTS_ONLY", 0))),
        help="Load model weights only, ignoring saved optimizer/scheduler states and epoch counter.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(os.environ.get("PATIENCE", 0)),
        help="Early stopping patience in epochs based on valid_gen_exact_match (default: 0 = disabled).",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        default=bool(int(os.environ.get("USE_EMA", 1))),
        help="Maintain Exponential Moving Average (EMA) shadow weights for evaluation and inference (default: True).",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA shadow weights.",
    )
    parser.add_argument(
        "--mask-self-attention",
        dest="mask_self_attention",
        action="store_true",
        default=bool(int(os.environ.get("MASK_SELF_ATTENTION", 1))),
        help="Apply sequence padding mask to decoder self-attention (default: True).",
    )
    parser.add_argument(
        "--no-mask-self-attention",
        dest="mask_self_attention",
        action="store_false",
        help="Disable sequence padding mask in decoder self-attention.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=float(os.environ.get("EMA_DECAY", 0.999)),
        help="EMA decay coefficient (default: 0.999).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=float(os.environ.get("LABEL_SMOOTHING", 0.05)),
        help="Label smoothing factor for peptide cross-entropy loss (default: 0.05).",
    )
    parser.add_argument(
        "--peak-dropout",
        type=float,
        default=float(os.environ.get("PEAK_DROPOUT", 0.15)),
        help="Fraction of spectral peaks to randomly drop during training (default: 0.15).",
    )
    parser.add_argument(
        "--decoding-strategy",
        type=str,
        default=os.environ.get("DECODING_STRATEGY", "confidence"),
        choices=["confidence", "random"],
        help="Discrete flow matching unmasking strategy during inference (default: confidence).",
    )
    parser.add_argument(
        "--decoding-temperature",
        type=float,
        default=float(os.environ.get("DECODING_TEMPERATURE", 0.0)),
        help="Sampling temperature during inference (default: 0.0 for greedy argmax).",
    )
    parser.add_argument(
        "--fragment-matching-weight",
        type=float,
        default=float(os.environ.get("FRAGMENT_MATCHING_WEIGHT", 0.5)),
        help="Weight beta for theoretical b/y fragment ion matching in beam scoring (default: 0.5).",
    )
    parser.add_argument(
        "--trypsin-prior",
        dest="trypsin_prior",
        action="store_true",
        default=bool(int(os.environ.get("TRYPSIN_PRIOR", 1))),
        help="Apply enzymatic C-terminal cleavage prior bonus (K/R) in candidate scoring (default: True).",
    )
    parser.add_argument(
        "--no-trypsin-prior",
        dest="trypsin_prior",
        action="store_false",
        help="Disable enzymatic C-terminal cleavage prior bonus.",
    )
    parser.add_argument(
        "--use-ptm",
        dest="use_ptm",
        action="store_true",
        default=bool(int(os.environ.get("USE_PTM", 1))),
        help="Include post-translational modifications (PTMs) in vocabulary (default: True).",
    )
    parser.add_argument(
        "--no-ptm",
        dest="use_ptm",
        action="store_false",
        help="Exclude PTMs and use standard 20 amino acids vocabulary.",
    )
    parser.add_argument(
        "--no-use-ptm",
        dest="use_ptm",
        action="store_false",
        help="Exclude PTMs and use standard 20 amino acids vocabulary (alias for --no-ptm).",
    )
    parser.add_argument("--model-dim", type=int, default=512, help="Hidden embedding dimension (default: 512)")
    parser.add_argument("--encoder-layers", type=int, default=6, help="Number of SpectrumEncoder layers (default: 6)")
    parser.add_argument("--encoder-heads", type=int, default=8, help="Number of encoder attention heads (default: 8)")
    parser.add_argument("--encoder-ff-dim", type=int, default=1536, help="Encoder FFN dimension (default: 1536)")
    parser.add_argument("--decoder-blocks", type=int, default=6, help="Number of AdaLN decoder blocks (default: 6)")
    parser.add_argument("--decoder-heads", type=int, default=8, help="Number of decoder attention heads (default: 8)")
    parser.add_argument("--mlp-hidden-dim", type=int, default=1536, help="Decoder SwiGLU MLP dimension (default: 1536)")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size is None:
        if args.optimize_vram:
            args.batch_size = get_recommended_batch_size(
                args.target_vram_pct,
                encoder_layers=args.encoder_layers,
                decoder_blocks=args.decoder_blocks,
            )
            print(f"[VRAM Optimization] Auto-tuned batch_size={args.batch_size} targeting >={int(args.target_vram_pct*100)}% VRAM on device.")
        else:
            args.batch_size = DEFAULTS.train.batch_size

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"dfm_pl_run_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PyTorch Lightning Training: {args.run_name} ===")
    print(f"Effective Batch Size: {args.batch_size * args.accumulate_grad_batches} (Batch: {args.batch_size}, Accumulate: {args.accumulate_grad_batches})")
    pl.seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_ds = get_dataset(repo_id=args.dataset_name, split=args.train_split, cache_dir=args.cache_dir)
    valid_ds = get_dataset(repo_id=args.dataset_name, split=args.valid_split, cache_dir=args.cache_dir)

    vocabulary = build_vocabulary(include_ptms=args.use_ptm)
    print(f"Vocabulary size: {len(vocabulary)} (PTM enabled: {args.use_ptm})")

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
    
    vocab_path = output_dir / args.run_name / "vocabulary.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f, indent=2)
    # Also save at root of output_dir for easy discovery by downstream stages
    with open(output_dir / "vocabulary.json", "w") as f:
        json.dump(vocabulary, f, indent=2)

    model = DFMLightningModule(vocabulary, args_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized: {total_params / 1e6:.2f}M parameters ({args.encoder_layers}L Encoder, {args.decoder_blocks}L Decoder, Dim: {args.model_dim})")

    resume_ckpt_path = None
    if args.resume_from and str(args.resume_from).strip().lower() not in ("none", "false", "", "null"):
        if not Path(args.resume_from).is_file():
            print(f"[Warning] Specified resume_from checkpoint '{args.resume_from}' does not exist on disk. Starting training from scratch.")
        else:
            resume_ckpt_path = str(args.resume_from)
            ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            source_vocab_size = state_dict.get("decoder.peptide_embedding.weight", torch.zeros(0)).shape[0]
            target_vocab_size = max(vocabulary.values()) + 1

            if source_vocab_size > 0 and source_vocab_size != target_vocab_size:
                print(f"Vocabulary mismatch detected (source: {source_vocab_size}, target: {target_vocab_size}).")
                print("Applying automatic checkpoint surgery to transplant weights...")
                from data.constants import STANDARD_AMINO_ACIDS
                from train.surgery import transplant_state_dict
                
                # Find old vocabulary
                cand_paths = [
                    Path(args.resume_from).parent.parent / "vocabulary.json",
                    Path(args.resume_from).parent / "vocabulary.json",
                ]
                old_vocab = None
                for p in cand_paths:
                    if p.exists():
                        with open(p) as f:
                            old_vocab = json.load(f)
                        break
                if old_vocab is None:
                    old_vocab = {aa: idx for idx, aa in enumerate(STANDARD_AMINO_ACIDS)}
                    old_vocab["<pad>"] = 20
                    old_vocab["<mask_token>"] = 21
                    old_vocab["<mask" + ">"] = 21

                state_dict = transplant_state_dict(state_dict, old_vocab, vocabulary)
                if "ema_shadow_params" in ckpt and ckpt["ema_shadow_params"]:
                    ckpt["ema_shadow_params"] = transplant_state_dict(
                        ckpt["ema_shadow_params"], old_vocab, vocabulary
                    )
                ckpt["state_dict"] = state_dict
                ckpt["optimizer_states"] = []
                ckpt["lr_schedulers"] = []
                ckpt["epoch"] = 0
                ckpt["global_step"] = 0

                transplanted_path = output_dir / args.run_name / "transplanted_init.ckpt"
                torch.save(ckpt, transplanted_path)
                print(f"Transplanted checkpoint saved to: {transplanted_path}")
                resume_ckpt_path = str(transplanted_path)

            if args.weights_only:
                print(f"Loading weights only from checkpoint: {resume_ckpt_path or args.resume_from}")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights (missing: {len(missing)}, unexpected: {len(unexpected)}). Starting training at epoch 0.")
                resume_ckpt_path = None
            else:
                mode_desc = f"resetting optimizer with default LR ({args.lr})" if args.reset_lr_on_resume else "preserving saved optimizer state"
                print(f"Resuming training from checkpoint: {resume_ckpt_path} ({mode_desc})")

    csv_logger = CSVLogger(save_dir=args.output_dir, name=args.run_name)
    tb_logger = TensorBoardLogger(save_dir=args.output_dir, name=args.run_name)
    loggers = [csv_logger, tb_logger]
    
    checkpoint_dir = output_dir / args.run_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Best model based strictly on Generative Exact Peptide Match (keep top 1)
    best_gen_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-gen-exact-epoch={epoch:02d}-exact={valid_gen_exact_match:.4f}",
        monitor="valid_gen_exact_match",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    # 2. Latest model snapshot saved at the end of every training epoch
    last_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        save_weights_only=True,
        save_on_train_epoch_end=True,
    )

    checkpoint_callbacks = [best_gen_cb, last_cb]
    callbacks = [*checkpoint_callbacks, LearningRateMonitor(logging_interval="step")]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="valid_gen_exact_match",
                mode="max",
                patience=args.patience,
                min_delta=0.001,
                verbose=True,
            )
        )
    if args.use_ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.eval_every,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        precision="bf16-mixed" if (args.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) 
                  else ("16-mixed" if args.amp else "32-true"),
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )


    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_ckpt_path,
    )
    print(f"=== Training complete ===")
    print(f"Best generative model: {best_gen_cb.best_model_path} (score={best_gen_cb.best_model_score})")
    print(f"Last model snapshot:   {last_cb.last_model_path}")


if __name__ == "__main__":
    main()
