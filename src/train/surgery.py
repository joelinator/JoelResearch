"""
Weight surgery module for transferring trained model weights to expanded vocabularies (e.g. PTM support).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from data.constants import PTM_PARENT_MAP
from data.data import get_output_aa_masses


def transplant_state_dict(
    state_dict: dict[str, torch.Tensor],
    old_vocab: dict[str, int],
    new_vocab: dict[str, int],
    parent_map: Mapping[str, str] = PTM_PARENT_MAP,
    perturbation_std: float = 0.005,
    bias_prior_offset: float = -0.5,
) -> dict[str, torch.Tensor]:
    """
    Transplant and expand model weights to match a new vocabulary.

    1. Embeddings:
       Existing tokens copied directly.
       PTM tokens warm-started from parent amino acid embedding + small noise.
       Special tokens (<pad>, <mask_token>) moved to their new positions.

    2. Output Head:
       Existing classes copied directly.
       PTM classes warm-started from parent amino acid head row.
       PTM bias initialized with slight negative offset to prevent initial over-prediction.

    3. aa_masses buffer:
       Updated with new output residue masses.
    """
    new_state_dict = {}
    old_out_size = max(old_vocab.values()) - 1
    new_out_size = max(new_vocab.values()) - 1

    def get_old_idx(tok: str) -> int | None:
        if tok in old_vocab:
            return old_vocab[tok]
        if tok in ("<mask_token>", "<mask" + ">"):
            return old_vocab.get("<mask_token>", old_vocab.get("<mask" + ">"))
        return None

    for key, tensor in state_dict.items():
        if key == "aa_masses":
            new_state_dict[key] = get_output_aa_masses(new_vocab).to(tensor.device)

        elif key.endswith("peptide_embedding.weight"):
            emb_dim = tensor.shape[1]
            vocab_size = max(new_vocab.values()) + 1
            new_emb = torch.zeros(vocab_size, emb_dim, dtype=tensor.dtype, device=tensor.device)
            assigned_indices = set()

            for tok, new_idx in new_vocab.items():
                if new_idx in assigned_indices:
                    continue
                old_idx = get_old_idx(tok)
                if old_idx is not None:
                    new_emb[new_idx] = tensor[old_idx]
                    assigned_indices.add(new_idx)
                elif tok in parent_map and parent_map[tok] in old_vocab:
                    parent_tok = parent_map[tok]
                    parent_old_idx = old_vocab[parent_tok]
                    noise = torch.randn(emb_dim, dtype=tensor.dtype, device=tensor.device) * perturbation_std
                    new_emb[new_idx] = tensor[parent_old_idx] + noise
                    assigned_indices.add(new_idx)
                else:
                    new_emb[new_idx] = torch.randn(emb_dim, dtype=tensor.dtype, device=tensor.device) * 0.02
                    assigned_indices.add(new_idx)

            new_state_dict[key] = new_emb

        elif key.endswith("head.weight"):
            emb_dim = tensor.shape[1]
            new_head_w = torch.zeros(new_out_size, emb_dim, dtype=tensor.dtype, device=tensor.device)
            assigned_indices = set()

            for tok, new_idx in new_vocab.items():
                if new_idx >= new_out_size or new_idx in assigned_indices:
                    continue  # Special tokens (<pad>, <mask_token>) not in output head
                old_idx = get_old_idx(tok)
                if old_idx is not None and old_idx < old_out_size:
                    new_head_w[new_idx] = tensor[old_idx]
                    assigned_indices.add(new_idx)
                elif (
                    tok in parent_map
                    and parent_map[tok] in old_vocab
                    and old_vocab[parent_map[tok]] < old_out_size
                ):
                    parent_tok = parent_map[tok]
                    parent_old_idx = old_vocab[parent_tok]
                    noise = torch.randn(emb_dim, dtype=tensor.dtype, device=tensor.device) * perturbation_std
                    new_head_w[new_idx] = tensor[parent_old_idx] + noise
                    assigned_indices.add(new_idx)
                else:
                    new_head_w[new_idx] = torch.randn(emb_dim, dtype=tensor.dtype, device=tensor.device) * 0.02
                    assigned_indices.add(new_idx)

            new_state_dict[key] = new_head_w

        elif key.endswith("head.bias"):
            new_head_b = torch.zeros(new_out_size, dtype=tensor.dtype, device=tensor.device)
            assigned_indices = set()

            for tok, new_idx in new_vocab.items():
                if new_idx >= new_out_size or new_idx in assigned_indices:
                    continue
                old_idx = get_old_idx(tok)
                if old_idx is not None and old_idx < old_out_size:
                    new_head_b[new_idx] = tensor[old_idx]
                    assigned_indices.add(new_idx)
                elif (
                    tok in parent_map
                    and parent_map[tok] in old_vocab
                    and old_vocab[parent_map[tok]] < old_out_size
                ):
                    parent_tok = parent_map[tok]
                    parent_old_idx = old_vocab[parent_tok]
                    new_head_b[new_idx] = tensor[parent_old_idx] + bias_prior_offset
                    assigned_indices.add(new_idx)
                else:
                    new_head_b[new_idx] = bias_prior_offset
                    assigned_indices.add(new_idx)

            new_state_dict[key] = new_head_b

        else:
            # 100% untouched transfer for Encoder, LengthPredictor, Guidance, and DecoderBlocks
            new_state_dict[key] = tensor.clone()

    return new_state_dict


def transplant_checkpoint(
    source_ckpt_path: str | Path,
    target_ckpt_path: str | Path,
    new_vocab: dict[str, int] | None = None,
    old_vocab: dict[str, int] | None = None,
    parent_map: Mapping[str, str] = PTM_PARENT_MAP,
    perturbation_std: float = 0.005,
    bias_prior_offset: float = -0.5,
    reset_optimizer: bool = True,
) -> dict:
    """
    Transplant an existing checkpoint to a new vocabulary and save the new checkpoint.
    """
    source_path = Path(source_ckpt_path)
    target_path = Path(target_ckpt_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading source checkpoint from: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)

    if old_vocab is None:
        # 1. Check if vocabulary was saved inside checkpoint
        if isinstance(checkpoint, dict) and "vocabulary" in checkpoint and checkpoint["vocabulary"]:
            old_vocab = checkpoint["vocabulary"]
            print(f"Loaded old vocabulary from checkpoint payload: {len(old_vocab)} tokens")
        else:
            # 2. Check if vocabulary.json exists next to checkpoint directory
            candidate_vocab_paths = [
                source_path.parent.parent / "vocabulary.json",
                source_path.parent / "vocabulary.json",
            ]
            for p in candidate_vocab_paths:
                if p.exists():
                    with open(p) as f:
                        old_vocab = json.load(f)
                    print(f"Loaded old vocabulary from: {p}")
                    break

    if old_vocab is None:
        from data.constants import STANDARD_AMINO_ACIDS
        old_vocab = {aa: idx for idx, aa in enumerate(STANDARD_AMINO_ACIDS)}
        old_vocab["<pad>"] = 20
        old_vocab["<mask_token>"] = 21
        print(f"Using default standard 20 AA vocabulary ({len(old_vocab)} tokens)")

    if new_vocab is None:
        from data.data import build_vocabulary
        new_vocab = build_vocabulary(include_ptms=True)
        print(f"Using standard target PTM vocabulary ({len(new_vocab)} tokens)")

    print(f"Old vocabulary size: {len(old_vocab)} (max id: {max(old_vocab.values())})")
    print(f"New vocabulary size: {len(new_vocab)} (max id: {max(new_vocab.values())})")

    # 1. Surgery on state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = transplant_state_dict(
        state_dict,
        old_vocab,
        new_vocab,
        parent_map=parent_map,
        perturbation_std=perturbation_std,
        bias_prior_offset=bias_prior_offset,
    )
    checkpoint["state_dict"] = new_state_dict

    # 2. Surgery on EMA shadow params if present
    if "ema_shadow_params" in checkpoint and checkpoint["ema_shadow_params"]:
        print("Transplanting EMA shadow parameters...")
        checkpoint["ema_shadow_params"] = transplant_state_dict(
            checkpoint["ema_shadow_params"],
            old_vocab,
            new_vocab,
            parent_map=parent_map,
            perturbation_std=perturbation_std,
            bias_prior_offset=bias_prior_offset,
        )

    # 3. Clean up optimizer state if resetting
    if reset_optimizer:
        print("Resetting optimizer states and step counters for fresh warm-start fine-tuning.")
        checkpoint["optimizer_states"] = []
        checkpoint["lr_schedulers"] = []
        checkpoint["epoch"] = 0
    checkpoint["vocabulary"] = new_vocab
    print(f"Saving transplanted checkpoint to: {target_path}")
    torch.save(checkpoint, target_path)

    # Also save the new vocabulary.json alongside the target checkpoint
    vocab_save_path = target_path.parent / "vocabulary.json"
    with open(vocab_save_path, "w") as f:
        json.dump(new_vocab, f, indent=2)
    print(f"Saved new vocabulary to: {vocab_save_path}")

    return checkpoint


def transplant_length_state_dict(
    state_dict: dict[str, torch.Tensor],
    old_max_length: int = 30,
    new_max_length: int = 50,
    min_length: int = 1,
    perturbation_std: float = 0.005,
    weight_noise_std: float = 0.01,
    bias_decay_slope: float = 0.03,
) -> dict[str, torch.Tensor]:
    """
    Surgically expand peptide length capacity from old_max_length to new_max_length.
    Preserves 100% bitwise verbatim weights for all positions/classes <= old_max_length.

    1. decoder.length_embedding.weight:
       Shape [old_max_length + 1, emb_dim] -> [new_max_length + 1, emb_dim]
       Rows 0..old_max_length copied verbatim.
       Rows (old_max_length+1)..new_max_length initialized from row old_max_length + N(0, perturbation_std^2).

    2. length_predictor.mlp.7.weight:
       Shape [old_n_classes, 512] -> [new_n_classes, 512]
       Rows 0..(old_n_classes-1) copied verbatim.
       Rows old_n_classes..(new_n_classes-1) warm-started from row (old_n_classes-1) + N(0, weight_noise_std^2).

    3. length_predictor.mlp.7.bias:
       Shape [old_n_classes] -> [new_n_classes]
       Indices 0..(old_n_classes-1) copied verbatim.
       Indices i in old_n_classes..(new_n_classes-1) initialized with empirical log-decay:
       bias[i] = bias[old_n_classes-1] - bias_decay_slope * (i - old_n_classes + 1).

    4. All other parameter tensors copied verbatim.
    """
    new_state_dict = {}
    old_n_classes = old_max_length - min_length + 1
    new_n_classes = new_max_length - min_length + 1

    for key, tensor in state_dict.items():
        if key.endswith("decoder.length_embedding.weight") or key == "decoder.length_embedding.weight":
            emb_dim = tensor.shape[1]
            new_len_emb = torch.zeros(
                new_max_length + 1, emb_dim, dtype=tensor.dtype, device=tensor.device
            )
            # Copy existing length embeddings verbatim (0..old_max_length)
            new_len_emb[: old_max_length + 1] = tensor[: old_max_length + 1]
            # Warm-start lengths old_max_length+1 .. new_max_length from old_max_length + small perturbation
            last_emb = tensor[old_max_length]
            for i in range(old_max_length + 1, new_max_length + 1):
                noise = torch.randn(emb_dim, dtype=tensor.dtype, device=tensor.device) * perturbation_std
                new_len_emb[i] = last_emb + noise
            new_state_dict[key] = new_len_emb

        elif key.endswith("length_predictor.mlp.7.weight") or key == "length_predictor.mlp.7.weight":
            in_dim = tensor.shape[1]
            new_w = torch.zeros(
                new_n_classes, in_dim, dtype=tensor.dtype, device=tensor.device
            )
            # Copy existing length classifier weights verbatim (0..old_n_classes-1)
            new_w[:old_n_classes] = tensor[:old_n_classes]
            # Warm-start newly added length classes from last class + small perturbation
            last_row = tensor[old_n_classes - 1]
            for i in range(old_n_classes, new_n_classes):
                noise = torch.randn(in_dim, dtype=tensor.dtype, device=tensor.device) * weight_noise_std
                new_w[i] = last_row + noise
            new_state_dict[key] = new_w

        elif key.endswith("length_predictor.mlp.7.bias") or key == "length_predictor.mlp.7.bias":
            new_b = torch.zeros(
                new_n_classes, dtype=tensor.dtype, device=tensor.device
            )
            # Copy existing length classifier biases verbatim (0..old_n_classes-1)
            new_b[:old_n_classes] = tensor[:old_n_classes]
            # Set empirical log-frequency decay prior for longer peptides
            last_bias = tensor[old_n_classes - 1]
            for i in range(old_n_classes, new_n_classes):
                offset = bias_decay_slope * float(i - old_n_classes + 1)
                new_b[i] = last_bias - offset
            new_state_dict[key] = new_b

        else:
            new_state_dict[key] = tensor.clone()

    return new_state_dict


def transplant_length_checkpoint(
    source_ckpt_path: str | Path,
    target_ckpt_path: str | Path,
    new_max_length: int = 50,
    min_length: int = 1,
    perturbation_std: float = 0.005,
    weight_noise_std: float = 0.01,
    bias_decay_slope: float = 0.03,
    reset_optimizer: bool = True,
) -> dict:
    """
    Transplant an existing checkpoint to a larger max sequence length (e.g. 30 -> 50 AA)
    and save the new checkpoint ready for immediate fine-tuning or evaluation.
    """
    source_path = Path(source_ckpt_path)
    target_path = Path(target_ckpt_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Length Surgery] Loading source checkpoint from: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)

    old_max_length = 30
    if "model_cfg" in checkpoint and "max_length" in checkpoint["model_cfg"]:
        old_max_length = int(checkpoint["model_cfg"]["max_length"])
    elif "hyper_parameters" in checkpoint and "args" in checkpoint["hyper_parameters"]:
        args = checkpoint["hyper_parameters"]["args"]
        if isinstance(args, dict) and "model_cfg" in args and "max_length" in args["model_cfg"]:
            old_max_length = int(args["model_cfg"]["max_length"])

    print(f"[Length Surgery] Expanding max sequence length: {old_max_length} -> {new_max_length} AA")

    # 1. Surgery on state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = transplant_length_state_dict(
        state_dict,
        old_max_length=old_max_length,
        new_max_length=new_max_length,
        min_length=min_length,
        perturbation_std=perturbation_std,
        weight_noise_std=weight_noise_std,
        bias_decay_slope=bias_decay_slope,
    )
    checkpoint["state_dict"] = new_state_dict

    # 2. Surgery on EMA shadow params if present
    if "ema_shadow_params" in checkpoint and checkpoint["ema_shadow_params"]:
        print("[Length Surgery] Transplanting EMA shadow parameters...")
        checkpoint["ema_shadow_params"] = transplant_length_state_dict(
            checkpoint["ema_shadow_params"],
            old_max_length=old_max_length,
            new_max_length=new_max_length,
            min_length=min_length,
            perturbation_std=perturbation_std,
            weight_noise_std=weight_noise_std,
            bias_decay_slope=bias_decay_slope,
        )

    # 3. Update configuration metadata
    if "model_cfg" in checkpoint and isinstance(checkpoint["model_cfg"], dict):
        checkpoint["model_cfg"]["max_length"] = new_max_length
    if "hyper_parameters" in checkpoint and isinstance(checkpoint["hyper_parameters"], dict):
        args = checkpoint["hyper_parameters"].get("args", {})
        if isinstance(args, dict):
            args["max_length"] = new_max_length
            if "model_cfg" in args and isinstance(args["model_cfg"], dict):
                args["model_cfg"]["max_length"] = new_max_length

    # 4. Clean up optimizer state if resetting
    if reset_optimizer:
        print("[Length Surgery] Resetting optimizer states and step counters for fresh warm-start fine-tuning.")
        checkpoint["optimizer_states"] = []
        checkpoint["lr_schedulers"] = []
        checkpoint["epoch"] = 0
        checkpoint["global_step"] = 0

    print(f"[Length Surgery] Saving transplanted checkpoint to: {target_path}")
    torch.save(checkpoint, target_path)
    print(f"[Length Surgery] Successfully saved transplanted checkpoint!")
    return checkpoint


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weight Surgery: Expand vocabulary to support PTMs.")
    parser.add_argument("--source-ckpt", type=str, required=True, help="Path to input checkpoint (standard 20 AA).")
    parser.add_argument("--target-ckpt", type=str, required=True, help="Path to output transplanted checkpoint.")
    parser.add_argument("--perturbation-std", type=float, default=0.005, help="Noise std for PTM embedding initialization.")
    parser.add_argument("--bias-prior-offset", type=float, default=-0.5, help="Bias prior offset for PTM head initialization.")
    parser.add_argument("--old-vocab", type=str, default=None, help="Path to old vocabulary.json (optional).")
    parser.add_argument("--new-vocab", type=str, default=None, help="Path to new vocabulary.json (optional).")
    parser.add_argument("--keep-optimizer", action="store_true", help="Keep optimizer states instead of resetting.")

    args = parser.parse_args()

    old_vocab = None
    if args.old_vocab:
        with open(args.old_vocab) as f:
            old_vocab = json.load(f)

    new_vocab = None
    if args.new_vocab:
        with open(args.new_vocab) as f:
            new_vocab = json.load(f)

    transplant_checkpoint(
        source_ckpt_path=args.source_ckpt,
        target_ckpt_path=args.target_ckpt,
        new_vocab=new_vocab,
        old_vocab=old_vocab,
        perturbation_std=args.perturbation_std,
        bias_prior_offset=args.bias_prior_offset,
        reset_optimizer=not args.keep_optimizer,
    )
    print("Weight surgery completed successfully.")


if __name__ == "__main__":
    main()
