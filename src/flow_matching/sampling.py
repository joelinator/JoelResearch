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
    mask_id = vocab.get("<mask_token>", vocab.get("<mask" + ">"))
    return vocab["<pad>"], mask_id


def amino_acid_token_indices(
    vocab: dict[str, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sorted amino-acid token ids (excludes <pad> and <mask_token>)."""
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
    """Sample random amino-acid token ids, excluding <pad> and <mask_token>."""
    aa_indices = amino_acid_token_indices(vocab, device=device)
    picks = torch.randint(0, aa_indices.numel(), shape, device=device)
    return aa_indices[picks]


def sample_noising_step_mask(kt, x1, vocab, padding_mask=None):
    """Forward mask corruption on active positions only."""
    xt = x1.clone()
    noise_mask = torch.rand(xt.shape, device=xt.device) > kt.unsqueeze(-1)
    if padding_mask is not None:
        noise_mask = noise_mask & ~padding_mask
    mask_id = vocab.get("<mask_token>", vocab.get("<mask" + ">"))
    xt[noise_mask] = mask_id
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
    kt,
    kt_derivative,
    x_t,
    logits,
    vocab,
    delta_t,
    active_mask=None,
    temperature: float = 0.0,
    strategy: str = "confidence",
    kt_next: torch.Tensor | None = None,
    eta: float = 0.0,
    is_final_step: bool = False,
):
    """
    Reverse mask step for discrete flow matching. Only active positions may change.

    Args:
        kt: Scheduler state at time t, shape (B,).
        kt_derivative: Time derivative of scheduler, shape (B,).
        x_t: Current noisy sequence of token IDs, shape (B, L).
        logits: Unnormalized decoder logits over vocabulary, shape (B, L, num_classes).
        vocab: Token vocabulary mapping string tokens to token IDs.
        delta_t: Discretization step size dt.
        active_mask: Boolean tensor (B, L) where True indicates active (non-padded) residues.
        temperature: Sampling temperature. If <= 0.05, uses greedy MAP unmasking (argmax).
        strategy: Unmasking strategy:
            - 'confidence': Top-confidence unmasking (MaskGIT / MDLM style) where positions
                            with highest prediction certainty are unmasked first.
            - 'random': Stochastic uniform unmasking according to rate dt * k'(t) / (1 - k(t)).
        kt_next: Optional scheduler state at time t + dt, shape (B,). When provided,
                 uses cumulative unmasking matching the scheduler curve κ(t).
        eta: Detailed Balance stochasticity parameter (Campbell et al., arXiv:2402.04997).
             When eta > 0, introduces reversible transitions (re-masking and boosted unmasking)
             satisfying detailed balance, enabling error correction and diverse stochastic paths.
        is_final_step: If True, disables re-masking on the final integration step so sequences
                       terminate clean.
    """
    if isinstance(temperature, torch.Tensor):
        t_clamped = temperature.view(-1, 1, 1).clamp(min=1e-3)
        probs = F.softmax(logits / t_clamped, dim=-1)
        greedy = logits.argmax(dim=-1)
        stochastic = Categorical(probs).sample()
        is_greedy = (temperature.view(-1, 1) <= 0.05).expand(-1, logits.shape[1])
        x_1 = torch.where(is_greedy, greedy, stochastic)
    elif temperature <= 0.05:
        probs = F.softmax(logits, dim=-1)
        x_1 = logits.argmax(dim=-1)
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        x_1 = Categorical(probs).sample()

    mask_id = vocab.get("<mask_token>", vocab.get("<mask" + ">"))
    is_masked = (x_t == mask_id)
    if active_mask is not None:
        is_masked = is_masked & active_mask

    denom = clean_weight_denominator(kt).view(-1, 1)
    if eta > 0.0:
        # Boosted unmasking rate under Detailed Balance: (1 + eta * kt) / (1 - kt) * k'(t) * dt
        boost = 1.0 + eta * kt.view(-1, 1)
        prob_unmask = (kt_derivative.view(-1, 1) * delta_t * boost / denom).clamp(0.0, 1.0)
    else:
        prob_unmask = (kt_derivative.view(-1, 1) * delta_t / denom).clamp(0.0, 1.0)

    if strategy == "confidence":
        # Confidence is maximum predicted probability across classes
        confidence = probs.max(dim=-1).values
        # Exclude already unmasked or inactive positions
        confidence = confidence.masked_fill(~is_masked, -1e9)

        if active_mask is not None:
            num_active = active_mask.sum(dim=-1, keepdim=True).float()
        else:
            num_active = torch.full(
                (x_t.shape[0], 1), x_t.shape[1], device=x_t.device, dtype=torch.float32
            )

        if kt_next is not None:
            # Cumulative unmasking with optional Detailed Balance boost
            if eta > 0.0:
                target_clean = torch.round(num_active * (kt_next.view(-1, 1) + eta * kt.view(-1, 1) * delta_t)).long()
            else:
                target_clean = torch.round(num_active * kt_next.view(-1, 1)).long()
            act = active_mask if active_mask is not None else torch.ones_like(is_masked)
            current_clean = (~is_masked & act).sum(dim=-1, keepdim=True)
            num_to_unmask = torch.clamp(target_clean - current_clean, min=0)
        else:
            num_to_unmask = torch.clamp(torch.ceil(num_active * prob_unmask).long(), min=1)

        # Vectorized top-k selection via double argsort (rank 0 is highest confidence)
        if isinstance(temperature, torch.Tensor):
            stoch_mask = (temperature.view(-1, 1) > 0.05).float()
            conf_noise = torch.rand_like(confidence) * stoch_mask * 0.05
            ranks = torch.argsort(torch.argsort(confidence + conf_noise, dim=-1, descending=True), dim=-1)
        else:
            ranks = torch.argsort(torch.argsort(confidence, dim=-1, descending=True), dim=-1)
        will_unmask = (ranks < num_to_unmask) & is_masked
    else:
        will_unmask = (torch.rand_like(x_t, dtype=torch.float32) < prob_unmask) & is_masked

    updated = x_t.clone()
    updated[will_unmask] = x_1[will_unmask]

    # Detailed Balance Re-Masking (stochastic jump back to mask state, Campbell et al. 2024)
    if eta > 0.0 and not is_final_step:
        prob_remask = min(1.0, float(delta_t * eta))
        act = active_mask if active_mask is not None else torch.ones_like(is_masked)
        is_unmasked = (~is_masked) & act
        will_remask = (torch.rand_like(x_t, dtype=torch.float32) < prob_remask) & is_unmasked
        updated[will_remask] = mask_id

    return updated


def inference_sample_uniform(
    kt, kt_derivatives, x_t, logits, vocab, delta_t, active_mask=None, temperature: float = 1.0
):
    """Reverse uniform step; only active positions may change."""
    if temperature > 0.0 and temperature != 1.0:
        probs = F.softmax(logits / temperature, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)
    denom = clean_weight_denominator(kt).view(-1, 1, 1)
    step_probs = (probs * kt_derivatives.view(-1, 1, 1) * delta_t / denom).clamp(max=1.0)
    x_t_clamped = x_t.clamp(0, probs.shape[-1] - 1)
    step_probs = step_probs.scatter(-1, x_t_clamped.unsqueeze(-1), 0.0)
    remaining = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    step_probs = step_probs.scatter(-1, x_t_clamped.unsqueeze(-1), remaining)
    samples = Categorical(step_probs.clamp(min=0.0)).sample()
    return _apply_active_only(x_t, samples, active_mask)

