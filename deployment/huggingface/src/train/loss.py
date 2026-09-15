"""
Loss-weight schedules for multi-task training.

Total loss:
  L = L_decoder + λ(epoch) · L_length + γ(epoch) · L_mass

Design rationale
----------------
* L_decoder (flow-matching CE) is always the primary objective (implicit weight 1).
* λ (length): ramp up early-mid training. Length conditions the decoder at inference,
  but should not compete with the decoder while it is still learning denoising.
* γ (mass): start later and ramp slowly. Mass consistency is a physics prior that
  only helps once residue distributions are meaningful; applying it too early adds
  noisy gradients on corrupted x_t states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.lengths import length_to_class

mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

# Default terminal weights (tune on validation if needed).
LAMBDA_FINAL = 0.15
GAMMA_FINAL = 0.08


def _ramp_factor(
    epoch: int,
    total_epochs: int,
    start_frac: float,
    end_frac: float,
) -> float:
    """
    Piecewise-linear factor in [0, 1].

    Returns 0 before `start_frac · total_epochs`, 1 after `end_frac · total_epochs`,
    and linearly interpolates in between.
    """
    if total_epochs <= 0:
        return 1.0

    start_epoch = int(start_frac * total_epochs)
    end_epoch = max(start_epoch + 1, int(end_frac * total_epochs))

    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    return (epoch - start_epoch) / (end_epoch - start_epoch)


def lambda_schedule(
    epoch: int,
    total_epochs: int,
    final: float = LAMBDA_FINAL,
    warmup_end_frac: float = 0.15,
) -> float:
    """
    Length-loss weight λ: linear warmup from 0 → `final`, then hold.

    With default settings and 100 epochs, λ reaches full value around epoch 15.
    """
    return final * _ramp_factor(epoch, total_epochs, 0.0, warmup_end_frac)


def gamma_schedule(
    epoch: int,
    total_epochs: int,
    final: float = GAMMA_FINAL,
    start_frac: float = 0.20,
    ramp_end_frac: float = 0.60,
) -> float:
    """
    Mass-loss weight γ: zero early, linear ramp, then hold.

    With default settings and 100 epochs, γ stays 0 until ~epoch 20 and reaches
    `final` around epoch 60.
    """
    return final * _ramp_factor(epoch, total_epochs, start_frac, ramp_end_frac)


def loss_weights(epoch: int, total_epochs: int) -> dict[str, float]:
    """Return all scalar loss weights for logging."""
    return {
        "lambda": lambda_schedule(epoch, total_epochs),
        "gamma": gamma_schedule(epoch, total_epochs),
    }


def mass_loss_hubert(
    logits,
    aa_masses,
    precursor_mass,
    active_mask=None,
    temperature=0.5,
    threshold=1e-2,
):
    seq_probs = F.softmax(logits / temperature, dim=-1)
    if active_mask is not None:
        weights = active_mask.float().unsqueeze(-1)
        seq_probs = seq_probs * weights
    seq_probs = seq_probs.sum(dim=1)
        
    # Precursor neutral mass = sum(residue masses) + M_H2O (18.010565 Da)
    average_mass = torch.sum(seq_probs * aa_masses.unsqueeze(0), dim=-1)
    target_residue_mass = (precursor_mass - 18.010565).clamp(min=1.0)
    rel_error = torch.abs(target_residue_mass - average_mass) / target_residue_mass
    mask = rel_error < threshold
    loss = torch.where(mask, 0.5 * (rel_error**2) / threshold, rel_error - 0.5 * threshold)
    return loss.mean()

def mass_loss_hubert_cum(
    logits,
    aa_masses,
    true_sequence,
    active_mask=None,
    temperature=0.5,
    threshold=1e-2,
    vocab_aa_masses=None,
):
    """
    Cumulative prefix mass Huber loss.

    Penalizes relative errors between the predicted expected cumulative prefix masses
    and the true cumulative prefix masses at each active residue position.

    Args:
        logits: Tensor of shape (B, L, num_classes) where num_classes is typically 20 (standard amino acids).
        aa_masses: Tensor of shape (num_classes,) containing residue masses for the decoder output classes.
        true_sequence: Tensor of shape (B, L) with vocabulary token IDs (which include <pad> and <mask_token>).
        active_mask: Optional boolean tensor of shape (B, L) where True denotes active (non-padded) residues.
        temperature: Temperature for softening logits probabilities.
        threshold: Huber loss transition threshold.
        vocab_aa_masses: Optional full vocabulary mass tensor of shape (vocab_size,), where <pad> is at index 20 (mass 0.0).
            If omitted, aa_masses is automatically zero-padded for special tokens >= num_classes.
    """
    seq_probs = F.softmax(logits / temperature, dim=-1)
    num_classes = logits.size(-1)
    # Output amino acid masses aligned with decoder logits (e.g. 20 amino acids)
    output_masses = aa_masses[:num_classes]
    expected_residue_mass = torch.sum(seq_probs * output_masses, dim=-1)

    # Resolve residue masses for true_sequence:
    # 1. If explicit vocab_aa_masses is provided (e.g. size 22), use it directly.
    # 2. Else if aa_masses is already large enough to cover all token IDs in true_sequence, use it.
    # 3. Else, zero-pad aa_masses for special tokens (<pad>=20, <mask_token>=21 with 0.0 mass).
    if vocab_aa_masses is not None:
        token_masses = vocab_aa_masses
    elif aa_masses.size(0) >= (int(true_sequence.max().item()) + 1):
        token_masses = aa_masses
    else:
        pad_size = (int(true_sequence.max().item()) + 1) - aa_masses.size(0)
        token_masses = F.pad(aa_masses, (0, pad_size), value=0.0)

    true_masses = token_masses[true_sequence]

    if active_mask is not None:
        mask_f = active_mask.float()
        expected_residue_mass = expected_residue_mass * mask_f
        true_masses = true_masses * mask_f

    cum_average_mass = expected_residue_mass.cumsum(dim=1)
    true_cum_masses = true_masses.cumsum(dim=1)

    rel_error = torch.abs(true_cum_masses - cum_average_mass) / true_cum_masses.clamp(min=1.0)
    huber_mask = rel_error < threshold
    loss = torch.where(huber_mask, 0.5 * (rel_error**2) / threshold, rel_error - 0.5 * threshold)

    if active_mask is not None:
        loss = loss * active_mask.float()
        return loss.sum() / active_mask.sum().clamp(min=1.0)
    return loss.mean()


def length_loss(logits, length, loss_fn=ce_loss):
    """
    Cross-entropy over length classes.

    Peptide lengths are 1..30; class indices are 0..29 via `length_to_class`.
    """
    return loss_fn(logits, length_to_class(length))


def peptide_loss(logits, peptide, pad_id: int, label_smoothing: float = 0.0, loss_fn=None):
    """
    Cross-entropy on decoder logits (amino acids only).

    Padding positions in `peptide` are ignored via `ignore_index=pad_id`.
    When `label_smoothing > 0.0`, regularizes the model against overconfidence
    on noisy or ambiguous fragment peaks.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    return loss_fn(
        logits.reshape(-1, logits.shape[-1]),
        peptide.reshape(-1),
    )

