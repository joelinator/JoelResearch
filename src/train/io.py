"""Checkpointing and metric logging helpers for training runs."""

from __future__ import annotations

import json
import os
import urllib.request
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

OFFICIAL_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "dfm_massive_kb.ckpt": {
        "tag": "v0.3.0",
        "url": "https://github.com/joelinator/JoelResearch/releases/download/v0.3.0/dfm_massive_kb.ckpt",
        "size": 238063513,
        "description": "MassIVE-KB model (max length 50, 59.5M parameters, trained on 28.2M spectra)",
        "aliases": [
            "massive-kb",
            "massive_kb",
            "massive-kb-59m",
            "massive_kb_59m",
            "dfm_massive_kb",
            "dfm_massive_kb.ckpt",
            "MassIVE-KB",
            "MassIVE-KB Model",
            "MassIVE-KB 59M",
            "MassIVE-KB 59.5M",
            "MassIVE-KB 59M Foundation",
            "dfm_massive_kb_continual.ckpt",
            "massive-kb-28m",
            "massive_kb_28m",
            "dfm_massive_kb_28m",
            "dfm_massive_kb_28m.ckpt",
            "MassIVE-KB 28M",
            "MassIVE-KB 28M Foundation",
        ],
        "local_relpaths": [
            "checkpoints/dfm_massive_kb.ckpt",
            "checkpoints/dfm_massive_kb_28m.ckpt",
            "artifacts/continual_hcpt_massivekb_run/stage3_massivekb/checkpoints/best-exact-epoch=02-exact=0.5389.ckpt",
            "artifacts/massive_kb_run/dfm_massive_kb_full_28m_length50_h100/checkpoints/best-exact-epoch=02-exact=0.5199.ckpt",
        ],
    },
    "dfm_balanced_best.ckpt": {
        "tag": "v0.2.0",
        "url": "https://github.com/joelinator/JoelResearch/releases/download/v0.2.0/dfm_balanced_best.ckpt",
        "size": 475945737,
        "description": "Nine-Species fine-tuned model (max length 30, 59.5M parameters)",
        "aliases": [
            "nine-species",
            "nine_species",
            "Nine-Species",
            "Nine-Species Fine-Tuned",
            "dfm_balanced_best",
            "balanced",
            "frozen_production_model",
            "nine-species-59m",
            "Nine-Species 59M",
        ],
        "local_relpaths": [
            "checkpoints/dfm_balanced_best.ckpt",
            "checkpoints/frozen_production_model.ckpt",
            "artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt",
        ],
    },
}


def download_file_with_progress(
    url: str,
    dest_path: Path,
    expected_size: int | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with user-friendly progress reporting and atomic rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f"{dest_path.name}.download.tmp")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DFlowNovoDownloader/1.0; +https://github.com/joelinator/JoelResearch)"},
    )

    print(f"Downloading checkpoint from GitHub release:\n  URL:  {url}")
    with urllib.request.urlopen(req) as resp:
        content_len = resp.headers.get("Content-Length")
        total_size = int(content_len) if content_len else (expected_size or 0)

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path.name)
        except Exception:
            pbar = None

        downloaded = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if pbar:
                    pbar.update(len(chunk))
                elif total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (20 * 1024 * 1024) < chunk_size:
                        print(f"  Progress: {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.1f}%)")

        if pbar:
            pbar.close()

    temp_path.replace(dest_path)
    print(f"Checkpoint downloaded successfully: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return dest_path


def ensure_checkpoint(
    model_name_or_path: str | Path,
    cache_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Resolve a checkpoint path locally, or download it from official GitHub releases.

    Args:
        model_name_or_path: File path or model alias (e.g. 'MassIVE-KB',
            'Nine-Species Fine-Tuned', 'dfm_massive_kb.ckpt', 'dfm_balanced_best.ckpt').
        cache_dir: Optional explicit directory to store/check checkpoints.
        repo_root: Optional repository root path.
        force_download: If True, re-download even if local file exists.

    Returns:
        Resolved Path to valid checkpoint file.
    """
    p = Path(model_name_or_path)
    if not force_download and p.is_file():
        return p.resolve()

    name_key = str(model_name_or_path).strip()
    normalized = name_key.lower().replace("-", "_").replace(" ", "_").replace(".ckpt", "").strip()

    matched_info = None
    target_filename = None
    for filename, info in OFFICIAL_CHECKPOINTS.items():
        fn_norm = filename.lower().replace("-", "_").replace(" ", "_").replace(".ckpt", "").strip()
        if normalized == fn_norm or name_key == filename:
            matched_info = info
            target_filename = filename
            break
        for alias in info.get("aliases", []):
            al_norm = alias.lower().replace("-", "_").replace(" ", "_").replace(".ckpt", "").strip()
            if normalized == al_norm or name_key.lower() == alias.lower():
                matched_info = info
                target_filename = filename
                break
        if matched_info:
            break

    if not matched_info:
        if p.exists():
            return p.resolve()
        valid_names = list(OFFICIAL_CHECKPOINTS.keys()) + [a for info in OFFICIAL_CHECKPOINTS.values() for a in info["aliases"]]
        raise FileNotFoundError(
            f"Checkpoint '{model_name_or_path}' was not found locally and is not a recognized official release name.\n"
            f"Available official models: {list(OFFICIAL_CHECKPOINTS.keys())}"
        )

    # Infer repo_root if not given
    if repo_root is None:
        curr = Path.cwd().resolve()
        for cand in [curr] + list(curr.parents):
            if (cand / "pyproject.toml").exists() or (cand / ".git").exists():
                repo_root = cand
                break
        if repo_root is None:
            repo_root = curr
    else:
        repo_root = Path(repo_root).resolve()

    # Search local paths
    if not force_download:
        if cache_dir:
            c_path = Path(cache_dir).expanduser().resolve() / target_filename
            if c_path.is_file():
                return c_path

        for rel in matched_info.get("local_relpaths", []):
            cand = repo_root / rel
            if cand.is_file():
                return cand.resolve()

        cand_cwd = Path.cwd().resolve() / "checkpoints" / target_filename
        if cand_cwd.is_file():
            return cand_cwd

        user_cache = Path.home() / ".cache" / "dflow_novo" / "checkpoints" / target_filename
        if user_cache.is_file():
            return user_cache

    # Determine destination directory
    if cache_dir:
        dest_dir = Path(cache_dir).expanduser().resolve()
    else:
        local_checkpoints = repo_root / "checkpoints"
        try:
            local_checkpoints.mkdir(parents=True, exist_ok=True)
            test_file = local_checkpoints / ".write_test"
            test_file.touch()
            test_file.unlink()
            dest_dir = local_checkpoints
        except Exception:
            dest_dir = Path.home() / ".cache" / "dflow_novo" / "checkpoints"

    dest_path = dest_dir / target_filename
    return download_file_with_progress(
        url=matched_info["url"],
        dest_path=dest_path,
        expected_size=matched_info.get("size"),
    )


def load_checkpoint(
    path_or_model: str | Path,
    map_location: str | torch.device = "cpu",
    auto_download: bool = True,
) -> dict:
    """
    Load checkpoint dictionary from file, local path, or official model name.
    
    If path_or_model is not an existing file on disk and auto_download is True,
    it will automatically resolve and download the model via ensure_checkpoint.
    """
    path = Path(path_or_model)
    if not path.is_file() and auto_download:
        path = ensure_checkpoint(path_or_model)
    return torch.load(path, map_location=map_location, weights_only=False)


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


def infer_model_config_from_checkpoint(checkpoint: dict | str | Path) -> dict:
    """Automatically infer model architectural dimensions from checkpoint tensor keys and shapes."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_cfg" in checkpoint and isinstance(checkpoint["model_cfg"], dict):
        return dict(checkpoint["model_cfg"])

    hparams = checkpoint.get("hyper_parameters", {})
    if isinstance(hparams, dict):
        if "model_cfg" in hparams and isinstance(hparams["model_cfg"], dict):
            cfg = dict(hparams["model_cfg"])
            if "implicit_length" in hparams.get("args", {}):
                cfg["implicit_length"] = hparams["args"]["implicit_length"]
            return cfg
        if "args" in hparams and isinstance(hparams["args"], dict):
            args = hparams["args"]
            if "model_cfg" in args and isinstance(args["model_cfg"], dict):
                cfg = dict(args["model_cfg"])
                if "implicit_length" in args:
                    cfg["implicit_length"] = args["implicit_length"]
                return cfg

    sd = checkpoint.get("state_dict", checkpoint.get("spectrum_encoder_state_dict", checkpoint))
    cfg = {}
    
    # 1. Count encoder layers
    enc_layer_indices = []
    for k in sd.keys():
        clean_k = k.replace("_orig_mod.", "")
        if "transformer_encoder.layers." in clean_k:
            parts = clean_k.split("transformer_encoder.layers.")[1].split(".")
            if parts[0].isdigit():
                enc_layer_indices.append(int(parts[0]))
    if enc_layer_indices:
        cfg["encoder_layers"] = max(enc_layer_indices) + 1
        
    # 2. Count decoder blocks
    dec_block_indices = []
    for k in sd.keys():
        clean_k = k.replace("_orig_mod.", "")
        if "decoder_blocks." in clean_k:
            parts = clean_k.split("decoder_blocks.")[1].split(".")
            if parts[0].isdigit():
                dec_block_indices.append(int(parts[0]))
    if dec_block_indices:
        cfg["decoder_blocks"] = max(dec_block_indices) + 1

    # 3. Model dimension
    for k in sd.keys():
        if "cls_token" in k:
            cfg["model_dim"] = sd[k].squeeze().shape[-1]
            break

    # 4. Encoder feedforward dimension
    for k in sd.keys():
        if "transformer_encoder.layers.0.linear1.weight" in k:
            cfg["encoder_ff_dim"] = sd[k].shape[0]
            break

    # 5. Decoder MLP hidden dimension
    for k in sd.keys():
        if "decoder_blocks.0.mlp.0.weight" in k or "decoder_blocks.0.mlp.linear1.weight" in k:
            cfg["mlp_hidden_dim"] = sd[k].shape[0]
            break

    # 6. Check output layer size for implicit length
    for k in sd.keys():
        clean_k = k.replace("_orig_mod.", "")
        if "output_layer.weight" in clean_k:
            out_dim = sd[k].shape[0]
            if out_dim == 31:  # 32 tokens minus mask = 31 (pad included)
                cfg["implicit_length"] = True
            break

    # 7. Max length inference from length_embedding or length_predictor
    for k, v in sd.items():
        if "length_embedding.weight" in k:
            cfg["max_length"] = v.shape[0] - 1
            cfg["min_length"] = 1
            break
        if "length_predictor" in k and ("mlp.7.weight" in k or "mlp.7.bias" in k):
            cfg["max_length"] = v.shape[0]
            cfg["min_length"] = 1
            break
    if "max_length" not in cfg:
        dec_sd = checkpoint.get("decoder_state_dict", {})
        lp_sd = checkpoint.get("length_predictor_state_dict", {})
        for k, v in dec_sd.items():
            if "length_embedding.weight" in k:
                cfg["max_length"] = v.shape[0] - 1
                cfg["min_length"] = 1
                break
        if "max_length" not in cfg:
            for k, v in lp_sd.items():
                if "mlp.7.weight" in k or "mlp.7.bias" in k:
                    cfg["max_length"] = v.shape[0]
                    cfg["min_length"] = 1
                    break

    return cfg


def load_models_from_checkpoint(
    checkpoint: dict,
    spectrum_encoder,
    length_predictor,
    decoder,
    guidance,
    optimizer=None,
    use_ema: bool = True,
    strict: bool = True,
) -> dict:
    """Load model (and optionally optimizer) weights from a checkpoint dict (supports both custom and PyTorch Lightning checkpoints)."""
    if "state_dict" in checkpoint:
        # PyTorch Lightning checkpoint format
        pl_sd = dict(checkpoint["state_dict"])
        if use_ema and checkpoint.get("ema_shadow_params"):
            pl_sd.update(checkpoint["ema_shadow_params"])
        enc_sd = _clean_state_dict_keys(pl_sd, "spectrum_encoder")
        lp_sd = _clean_state_dict_keys(pl_sd, "length_predictor")
        dec_sd = _clean_state_dict_keys(pl_sd, "decoder")
        guid_sd = _clean_state_dict_keys(pl_sd, "guidance")
    else:
        # Custom training loop checkpoint format
        enc_sd = _clean_state_dict_keys(checkpoint["spectrum_encoder_state_dict"])
        lp_sd = _clean_state_dict_keys(checkpoint["length_predictor_state_dict"])
        dec_sd = _clean_state_dict_keys(checkpoint["decoder_state_dict"])
        guid_sd = _clean_state_dict_keys(checkpoint["guidance_state_dict"])

    if not strict:
        # Filter to keys with exact matching shapes for safe architecture scaling
        enc_sd = {k: v for k, v in enc_sd.items() if k in spectrum_encoder.state_dict() and spectrum_encoder.state_dict()[k].shape == v.shape}
        lp_sd = {k: v for k, v in lp_sd.items() if k in length_predictor.state_dict() and length_predictor.state_dict()[k].shape == v.shape}
        dec_sd = {k: v for k, v in dec_sd.items() if k in decoder.state_dict() and decoder.state_dict()[k].shape == v.shape}
        guid_sd = {k: v for k, v in guid_sd.items() if k in guidance.state_dict() and guidance.state_dict()[k].shape == v.shape}

    # Dynamic self-healing: adapt length_predictor output layer if checkpoint has different max_length
    if "mlp.7.weight" in lp_sd and hasattr(length_predictor, "mlp") and len(length_predictor.mlp) > 7:
        ckpt_classes, in_feat = lp_sd["mlp.7.weight"].shape
        cur_layer = length_predictor.mlp[7]
        if cur_layer.weight.shape != lp_sd["mlp.7.weight"].shape:
            dev = cur_layer.weight.device
            dtype = cur_layer.weight.dtype
            length_predictor.mlp[7] = torch.nn.Linear(in_feat, ckpt_classes, device=dev, dtype=dtype)
            length_predictor.n_classes = ckpt_classes
            min_l = getattr(length_predictor, "min_length", 1)
            length_predictor.max_length = min_l + ckpt_classes - 1

    # Dynamic self-healing: adapt decoder length embedding if checkpoint has different max_length
    if "length_embedding.weight" in dec_sd and hasattr(decoder, "length_embedding"):
        ckpt_num_emb, ckpt_emb_dim = dec_sd["length_embedding.weight"].shape
        if decoder.length_embedding.weight.shape != dec_sd["length_embedding.weight"].shape:
            dev = decoder.length_embedding.weight.device
            decoder.length_embedding = torch.nn.Embedding(ckpt_num_emb, ckpt_emb_dim, device=dev)
            decoder.max_length = ckpt_num_emb - 1

    # Dynamic self-healing: adapt decoder output layer if output dimension differs
    if "output_layer.weight" in dec_sd and hasattr(decoder, "output_layer"):
        ckpt_vocab_out, in_f = dec_sd["output_layer.weight"].shape
        if decoder.output_layer.weight.shape != dec_sd["output_layer.weight"].shape:
            dev = decoder.output_layer.weight.device
            dtype = decoder.output_layer.weight.dtype
            decoder.output_layer = torch.nn.Linear(in_f, ckpt_vocab_out, device=dev, dtype=dtype)

    spectrum_encoder.load_state_dict(enc_sd, strict=strict)
    length_predictor.load_state_dict(lp_sd, strict=strict)
    decoder.load_state_dict(dec_sd, strict=strict)
    guidance.load_state_dict(guid_sd, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"Notice: optimizer state could not be loaded ({e}); continuing with fresh optimizer.")
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

