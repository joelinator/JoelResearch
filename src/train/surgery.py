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
