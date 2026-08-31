"""
Forward and reverse sampling for discrete flow matching.

Padding policy
--------------
* `<pad>` is structural batch padding, not a generative token.
* Active positions (0 .. length-1): sample / noise amino acids only.
* Inactive positions (j >= length): stay `<pad>`, never noised, never updated.
* `<mask>` may appear in the input embedding during noising but is not in the
  decoder output head.
"""

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .scheduler import clean_weight_denominator

SPECIAL_TOKENS = ("<pad>", "<mask>")


def special_token_ids(vocab: dict[str, int]) -> tuple[int, int]:
    """Return (pad_id, mask_id) from the vocabulary."""
    return vocab["<pad>"], vocab["<mask>"]


def amino_acid_token_indices(
    vocab: dict[str, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sorted amino-acid token ids (excludes <pad> and <mask>)."""
    pad_id, mask_id = special_token_ids(vocab)
    special = {pad_id, mask_id}
    indices = sorted(token_id for token_id in vocab.values() if token_id not in special)
    if not indices:
        raise ValueError("Vocabulary has no amino-acid tokens for uniform sampling.")
    return torch.tensor(indices, dtype=torch.long, device=device)


def sample_uniform_noise(
    vocab: dict[str, int],
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Sample random amino-acid token ids, excluding <pad> and <mask>."""
    aa_indices = amino_acid_token_indices(vocab, device=device)
    picks = torch.randint(0, aa_indices.numel(), shape, device=device)
    return aa_indices[picks]


def sample_noising_step_mask(kt, x1, vocab, padding_mask=None):
    """Forward mask corruption on active positions only."""
    xt = x1.clone()
    noise_mask = torch.rand(xt.shape, device=xt.device) > kt.unsqueeze(-1)
    if padding_mask is not None:
        noise_mask = noise_mask & ~padding_mask
    xt[noise_mask] = vocab["<mask>"]
    return xt


def sample_noising_step_uniform(kt, x1, vocab, padding_mask=None):
    """Forward uniform corruption on active positions only."""
    xt = x1.clone()
    noise_mask = torch.rand(xt.shape, device=xt.device) > kt.unsqueeze(-1)
    if padding_mask is not None:
        noise_mask = noise_mask & ~padding_mask
    if noise_mask.any():
        xt[noise_mask] = sample_uniform_noise(
            vocab,
            xt[noise_mask].shape,
            device=xt.device,
        )
    return xt


def _apply_active_only(x_t, samples, active_mask):
    if active_mask is None:
        return samples
    return torch.where(active_mask, samples, x_t)


def inference_sample_mask(
    kt, kt_derivative, x_t, logits, vocab, delta_t, active_mask=None
):
    """Reverse mask step; only active positions may change."""
    probs = F.softmax(logits, dim=-1)
    x_1 = Categorical(probs).sample()
    denom = clean_weight_denominator(kt).unsqueeze(-1)
    will_unmask = torch.rand_like(x_t, dtype=torch.float32) < (
        kt_derivative.unsqueeze(-1) * delta_t / denom
    )
    will_unmask = will_unmask & (x_t == vocab["<mask>"])
    if active_mask is not None:
        will_unmask = will_unmask & active_mask
    updated = x_t.clone()
    updated[will_unmask] = x_1[will_unmask]
    return updated


def inference_sample_uniform(
    kt, kt_derivatives, x_t, logits, vocab, delta_t, active_mask=None
):
    """Reverse uniform step; only active positions may change."""
    probs = F.softmax(logits, dim=-1)
    denom = clean_weight_denominator(kt).unsqueeze(-1)
    step_probs = (probs * kt_derivatives.unsqueeze(-1) * delta_t / denom).clamp(max=1.0)
    step_probs = step_probs.scatter(-1, x_t.unsqueeze(-1), 0.0)
    remaining = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    step_probs = step_probs.scatter(-1, x_t.unsqueeze(-1), remaining)
    samples = Categorical(step_probs.clamp(min=0.0)).sample()
    return _apply_active_only(x_t, samples, active_mask)
