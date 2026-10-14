# Forward and reverse sampling for discrete flow matching.
# Reverse Euler solver with continuous-time detailed balance re-masking (eta).
#
# NOTE: Active positions (0 .. length-1) get unmasked/remasked; inactive positions stay <pad>.
# NOTE: temperature <= 0.05 is treated as greedy MAP argmax to avoid 0-division bugs.
# TODO: experiment with adaptive temperature decay over the integration steps.

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
    confidence_bonus: torch.Tensor | None = None,
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
        confidence_bonus: Optional additive boost tensor (B, L) based on physical peak evidence.
    """
    if isinstance(temperature, torch.Tensor):
        t_tensor = temperature.view(-1, 1, 1)
        t_safe = torch.where(t_tensor <= 0.05, torch.tensor(1.0, device=logits.device, dtype=logits.dtype), t_tensor.clamp(min=1e-3))
        probs = F.softmax(logits / t_safe, dim=-1)
        greedy = logits.argmax(dim=-1)
        stochastic = Categorical(probs).sample()
        is_greedy = (t_tensor.squeeze(-1) <= 0.05).expand(-1, logits.shape[1])
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
        if confidence_bonus is not None:
            confidence = confidence + confidence_bonus
        # Exclude already unmasked or inactive positions
        confidence = confidence.masked_fill(~is_masked, -1e9)

        if active_mask is not None:
            num_active = active_mask.sum(dim=-1, keepdim=True).float()
        else:
            num_active = torch.full(
                (x_t.shape[0], 1), x_t.shape[1], device=x_t.device, dtype=torch.float32
            )

        if kt_next is not None:
            # Target clean token count at time t + dt according to the flow matching schedule
            target_clean = torch.round(num_active * kt_next.view(-1, 1).clamp(0.0, 1.0)).long()
            act = active_mask if active_mask is not None else torch.ones_like(is_masked)
            current_clean = (~is_masked & act).sum(dim=-1, keepdim=True)
            num_masked = (is_masked & act).sum(dim=-1, keepdim=True)

            if eta > 0.0 and not is_final_step:
                # Campbell et al. (2024) Detailed Balance:
                # current_clean tokens will be re-masked with probability eta * delta_t.
                # In expectation: E[C_{t+dt}] = current_clean + num_unmask - current_clean * eta * delta_t = target_clean
                # => num_unmask = target_clean - current_clean + current_clean * eta * delta_t
                remask_offset = current_clean.float() * (eta * delta_t)
                num_to_unmask = torch.round(target_clean.float() - current_clean.float() + remask_offset).long()
            else:
                num_to_unmask = target_clean - current_clean

            num_to_unmask = torch.minimum(torch.clamp(num_to_unmask, min=0), num_masked)
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
    kt: torch.Tensor,
    kt_derivatives: torch.Tensor,
    x_t: torch.Tensor,
    logits: torch.Tensor,
    vocab: dict[str, int],
    delta_t: float,
    active_mask: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 0.0,
    eta: float = 0.0,
    is_final_step: bool = False,
) -> torch.Tensor:
    """
    Reverse uniform step under continuous-time discrete flow matching (Campbell et al. 2024).

    Implements the conditional continuous-time Markov chain (CTMC) jump rate matrix on the
    uniform probability path with Detailed Balance exploration and knapsack-constrained transitions.

    Rate matrix (Campbell et al. 2024, arXiv:2402.04997):
        For j != x_t (j in valid amino acids):
            R_t(x_t, j) = p_theta(j | x_t) * [k'(t) / (1 - k(t)) + eta]
        Discretized transition probability over step dt:
            P(x_{t+dt} = j | x_t) = dt * R_t(x_t, j) = c_t * p_theta(j | x_t)    (j != x_t)
            P(x_{t+dt} = x_t | x_t) = (1 - c_t) + c_t * p_theta(x_t | x_t)
        where c_t = clamp(dt * [k'(t) / (1 - k(t)) + eta], 0.0, 1.0).

    Args:
        kt: Scheduler state at time t, shape (B,).
        kt_derivatives: Time derivative of scheduler, shape (B,).
        x_t: Current noisy sequence of token IDs, shape (B, L).
        logits: Unnormalized decoder logits over vocabulary, shape (B, L, num_classes).
        vocab: Token vocabulary mapping string tokens to token IDs.
        delta_t: Discretization step size dt.
        active_mask: Boolean tensor (B, L) where True indicates active (non-padded) residues.
        temperature: Sampling temperature. If <= 0.05, uses greedy MAP on final step.
        eta: Detailed Balance stochasticity parameter (Campbell et al. 2024).
             Introduces reversible symmetric transitions enabling error correction.
        is_final_step: If True, disables stochastic noise to produce clean terminal predictions.
    """
    # Restrict active predictions strictly to valid amino acids (mask <pad> and <mask_token>)
    pad_id, mask_id = special_token_ids(vocab)
    special = {pad_id, mask_id}
    num_classes = logits.shape[-1]
    is_valid_aa = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
    for s_id in special:
        if s_id is not None and 0 <= s_id < num_classes:
            is_valid_aa[s_id] = False

    clean_logits = logits.masked_fill(~is_valid_aa.view(1, 1, -1), -1e9)

    # Compute predicted posterior p_theta(x_1 | x_t)
    if isinstance(temperature, torch.Tensor):
        t_clamped = temperature.view(-1, 1, 1).clamp(min=1e-3)
        probs = F.softmax(clean_logits / t_clamped, dim=-1)
        is_greedy = (temperature.view(-1, 1) <= 0.05).expand(-1, logits.shape[1])
    elif temperature <= 0.05:
        probs = F.softmax(clean_logits, dim=-1)
        is_greedy = True
    else:
        probs = F.softmax(clean_logits / temperature, dim=-1)
        is_greedy = False

    # On final step with greedy temperature, take deterministic MAP prediction
    if is_final_step:
        greedy_samples = clean_logits.argmax(dim=-1)
        if isinstance(is_greedy, torch.Tensor):
            stoch_samples = Categorical(probs.clamp(min=1e-8)).sample()
            final_samples = torch.where(is_greedy, greedy_samples, stoch_samples)
        elif is_greedy:
            final_samples = greedy_samples
        else:
            final_samples = Categorical(probs.clamp(min=1e-8)).sample()
        return _apply_active_only(x_t, final_samples, active_mask)

    # Campbell CTMC jump rate matrix (Campbell et al. 2024, Eq. 325):
    # For j != x_t:
    #   R_t(x_t, j) = [(k'(t) + eta*(1 + (S-1)*k(t))) / (1 - k(t))] * p_theta(j|x_t) + eta * p_theta(x_t|x_t)
    # Transition probability over delta_t:
    #   P(j | x_t) = delta_t * R_t(x_t, j)  for j != x_t
    #   P(x_t | x_t) = 1 - sum_{j != x_t} P(j | x_t)
    S = is_valid_aa.sum().float()
    denom = clean_weight_denominator(kt).view(-1, 1, 1)
    drift_coeff = (kt_derivatives.view(-1, 1, 1) + eta * (1.0 + (S - 1.0) * kt.view(-1, 1, 1))) / denom

    x_t_clamped = x_t.clamp(0, num_classes - 1)
    curr_p = probs.gather(-1, x_t_clamped.unsqueeze(-1))  # (B, L, 1)

    # Compute off-diagonal step probabilities
    step_probs = delta_t * (drift_coeff * probs + eta * curr_p)

    # Zero out diagonal, mask non-amino-acid tokens
    step_probs = step_probs.masked_fill(~is_valid_aa.view(1, 1, -1), 0.0)
    step_probs.scatter_(-1, x_t_clamped.unsqueeze(-1), 0.0)

    # Compute diagonal (stay probability) ensuring sum = 1
    off_diag_sum = step_probs.sum(dim=-1, keepdim=True)
    stay_prob = (1.0 - off_diag_sum).clamp(min=0.0)
    step_probs.scatter_(-1, x_t_clamped.unsqueeze(-1), stay_prob)

    # Numerical normalization over valid amino acid simplex
    step_probs = step_probs.clamp(min=0.0)
    step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    samples = Categorical(step_probs).sample()
    return _apply_active_only(x_t, samples, active_mask)

