"""Checkpointing and metric logging helpers for training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return str(value)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2) + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_builtin(payload)) + "\n")


def empty_history() -> dict[str, list]:
    return {
        "train_loss": [],
        "train_decoder_loss": [],
        "train_length_loss": [],
        "train_mass_loss": [],
        "valid_loss": [],
        "valid_decoder_loss": [],
        "valid_length_loss": [],
        "valid_mass_loss": [],
        "lambda": [],
        "gamma": [],
    }


def copy_history(history: dict | None) -> dict[str, list]:
    if history is None:
        return empty_history()
    return {key: list(history.get(key, [])) for key in empty_history()}


def build_checkpoint_payload(
    *,
    epoch: int,
    total_epochs: int,
    history: dict,
    vocabulary: dict,
    args: dict,
    optimizer,
    spectrum_encoder,
    length_predictor,
    decoder,
    guidance,
    best_valid_loss: float,
) -> dict:
    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "history": history,
        "vocabulary": vocabulary,
        "args": _to_builtin(args),
        "best_valid_loss": best_valid_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "spectrum_encoder_state_dict": spectrum_encoder.state_dict(),
        "length_predictor_state_dict": length_predictor.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "guidance_state_dict": guidance.state_dict(),
    }


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor], prefix: str = "") -> dict[str, torch.Tensor]:
    result = {}
    prefix_dot = f"{prefix}." if prefix else ""
    for k, v in state_dict.items():
        if prefix_dot and not k.startswith(prefix_dot):
            continue
        sub_k = k[len(prefix_dot):] if prefix_dot else k
        # Handle torch.compile '_orig_mod.' prefix if present
        sub_k = sub_k.replace("_orig_mod.", "")
        result[sub_k] = v
    return result


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load checkpoint dictionary from file."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_models_from_checkpoint(
    checkpoint: dict,
    spectrum_encoder,
    length_predictor,
    decoder,
    guidance,
    optimizer=None,
) -> dict:
    """Load model (and optionally optimizer) weights from a checkpoint dict (supports both custom and PyTorch Lightning checkpoints)."""
    if "state_dict" in checkpoint:
        # PyTorch Lightning checkpoint format
        pl_sd = checkpoint["state_dict"]
        spectrum_encoder.load_state_dict(_clean_state_dict_keys(pl_sd, "spectrum_encoder"))
        length_predictor.load_state_dict(_clean_state_dict_keys(pl_sd, "length_predictor"))
        decoder.load_state_dict(_clean_state_dict_keys(pl_sd, "decoder"))
        guidance.load_state_dict(_clean_state_dict_keys(pl_sd, "guidance"))
    else:
        # Custom training loop checkpoint format
        spectrum_encoder.load_state_dict(_clean_state_dict_keys(checkpoint["spectrum_encoder_state_dict"]))
        length_predictor.load_state_dict(_clean_state_dict_keys(checkpoint["length_predictor_state_dict"]))
        decoder.load_state_dict(_clean_state_dict_keys(checkpoint["decoder_state_dict"]))
        guidance.load_state_dict(_clean_state_dict_keys(checkpoint["guidance_state_dict"]))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


@dataclass
class TrainingRunLogger:
    output_dir: Path
    run_name: str

    def __post_init__(self) -> None:
        self.run_dir = self.output_dir / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_jsonl = self.run_dir / "metrics.jsonl"
        self.history_json = self.run_dir / "history.json"
        self.summary_json = self.run_dir / "summary.json"
        self.config_json = self.run_dir / "config.json"
        self.latest_ckpt = self.checkpoint_dir / "latest.pt"
        self.best_ckpt = self.checkpoint_dir / "best_valid.pt"
        self.best_metric_ckpt = self.checkpoint_dir / "best_peptide_recall.pt"
        self.best_valid_loss = float("inf")
        self.best_peptide_recall = -1.0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_run_start(self, config: dict) -> None:
        save_json(self.config_json, config)

    def resume_from_checkpoint(self, checkpoint: dict) -> None:
        if "best_valid_loss" in checkpoint:
            self.best_valid_loss = float(checkpoint["best_valid_loss"])
        elif checkpoint.get("history", {}).get("valid_loss"):
            self.best_valid_loss = float(checkpoint["history"]["valid_loss"][-1])
        if "best_peptide_recall" in checkpoint:
            self.best_peptide_recall = float(checkpoint["best_peptide_recall"])
        elif checkpoint.get("history", {}).get("valid_peptide_recall"):
            self.best_peptide_recall = float(max(checkpoint["history"]["valid_peptide_recall"]))

    def log_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        history: dict,
        train_metrics: dict,
        valid_metrics: dict,
        weights: dict,
        checkpoint_payload: dict,
        generative_metrics: dict | None = None,
    ) -> None:
        valid_loss = float(valid_metrics["loss"])
        record = {
            "epoch": epoch,
            "epoch_1_indexed": epoch + 1,
            "total_epochs": total_epochs,
            "weights": weights,
            "train": train_metrics,
            "valid": valid_metrics,
            "generative": generative_metrics,
            "best_valid_loss_so_far": min(self.best_valid_loss, valid_loss),
        }
        append_jsonl(self.metrics_jsonl, record)
        save_json(self.history_json, history)

        torch.save(checkpoint_payload, self.latest_ckpt)
        if valid_loss <= self.best_valid_loss:
            self.best_valid_loss = valid_loss
            best_payload = {
                **checkpoint_payload,
                "best_valid_loss": self.best_valid_loss,
                "best_peptide_recall": self.best_peptide_recall,
            }
            torch.save(best_payload, self.best_ckpt)

        peptide_recall = None
        if generative_metrics is not None:
            peptide_recall = float(generative_metrics.get("peptide_recall", -1.0))
            if peptide_recall > self.best_peptide_recall:
                self.best_peptide_recall = peptide_recall
                metric_payload = {
                    **checkpoint_payload,
                    "best_valid_loss": self.best_valid_loss,
                    "best_peptide_recall": self.best_peptide_recall,
                }
                torch.save(metric_payload, self.best_metric_ckpt)

        save_json(
            self.summary_json,
            {
                "run_name": self.run_name,
                "epoch": epoch,
                "epoch_1_indexed": epoch + 1,
                "total_epochs": total_epochs,
                "latest_checkpoint": str(self.latest_ckpt),
                "best_checkpoint": str(self.best_ckpt),
                "best_peptide_recall_checkpoint": str(self.best_metric_ckpt),
                "best_valid_loss": self.best_valid_loss,
                "best_peptide_recall": self.best_peptide_recall,
                "last_peptide_recall": peptide_recall,
                "last_train_loss": train_metrics["loss"],
                "last_valid_loss": valid_loss,
                "metrics_jsonl": str(self.metrics_jsonl),
                "history_json": str(self.history_json),
            },
        )
