import random
import torch
from data.lengths import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH


def length_noiser(
    batch_lengths: torch.Tensor,
    noising_prob: float = 0.1,
    max_: int = MAX_PEPTIDE_LENGTH,
    min_: int = MIN_PEPTIDE_LENGTH,
) -> tuple[torch.Tensor, bool]:
    """
    Perturb batch peptide lengths by +/-1 with probability `noising_prob`.

    Args:
        batch_lengths: Tensor of peptide lengths (B,).
        noising_prob: Probability of perturbing lengths.
        max_: Maximum allowed peptide length.
        min_: Minimum allowed peptide length.

    Returns:
        (noised_lengths, is_noised)
    """
    r = random.random()
    if r <= noising_prob / 2.0:
        noised = (batch_lengths - 1).clamp(min=min_, max=max_)
        is_noised = bool((noised != batch_lengths).any().item())
        return noised, is_noised

    elif r <= noising_prob:
        noised = (batch_lengths + 1).clamp(min=min_, max=max_)
        is_noised = bool((noised != batch_lengths).any().item())
        return noised, is_noised

    return batch_lengths, False


def get_recommended_batch_size(target_vram_pct: float = 0.85) -> int:
    """Recommend batch size based on available GPU VRAM."""
    if not torch.cuda.is_available():
        return 64
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_mem_gb >= 70:
        return 1792
    elif total_mem_gb >= 35:
        return 896
    elif total_mem_gb >= 20:
        return 448
    else:
        return 128

