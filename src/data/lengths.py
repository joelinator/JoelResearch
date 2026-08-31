"""Peptide length constants and class-index mapping."""

from __future__ import annotations

import torch

MIN_PEPTIDE_LENGTH = 1
MAX_PEPTIDE_LENGTH = 30
NUM_LENGTH_CLASSES = MAX_PEPTIDE_LENGTH - MIN_PEPTIDE_LENGTH + 1


def validate_peptide_length(length: int) -> None:
    if not MIN_PEPTIDE_LENGTH <= length <= MAX_PEPTIDE_LENGTH:
        raise ValueError(
            f"Peptide length {length} is outside supported range "
            f"[{MIN_PEPTIDE_LENGTH}, {MAX_PEPTIDE_LENGTH}]."
        )


def length_to_class(length: torch.Tensor) -> torch.Tensor:
    """Map peptide length L ∈ [1, 30] to class index L - 1 ∈ [0, 29]."""
    return length - MIN_PEPTIDE_LENGTH


def class_to_length(class_id: torch.Tensor) -> torch.Tensor:
    """Map class index ∈ [0, 29] back to peptide length ∈ [1, 30]."""
    return class_id + MIN_PEPTIDE_LENGTH


def clamp_length(length: torch.Tensor) -> torch.Tensor:
    return length.clamp(min=MIN_PEPTIDE_LENGTH, max=MAX_PEPTIDE_LENGTH)


def length_to_active_mask(length: torch.Tensor, seq_len: int) -> torch.Tensor:
    """True for positions 0 .. length-1 (peptide residues), False for padding tail."""
    positions = torch.arange(seq_len, device=length.device)
    return positions.unsqueeze(0) < length.unsqueeze(1)


def apply_length_padding(
    token_ids: torch.Tensor,
    length: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Force `<pad>` at all positions j >= length (structural padding, not sampled)."""
    active_mask = length_to_active_mask(length, token_ids.shape[1])
    return token_ids.masked_fill(~active_mask, pad_id)

