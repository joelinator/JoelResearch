"""
Noise schedulers for discrete flow matching.

Convention (shared by forward noising and reverse sampling)
---------------------------------------------------------
κ(t)  := P(a position still equals the clean token x₁ at time t)
κ'(t) := dκ/dt

Requirements:
  - κ(0) = 0  (fully corrupted / noisy at the source)
  - κ(1) = 1  (fully clean at the target)
  - κ is strictly increasing on [0, 1]

Forward noising (training) uses:
  corrupt position when Uniform(0, 1) > κ(t)

Reverse sampling (inference) integrates t : 0 → 1 with:
  P(transition) ∝ κ'(t) · Δt / (1 - κ(t))
and clamps (1 - κ) from below by SCHEDULER_EPS to avoid singularities near κ = 1.
"""

from __future__ import annotations

from typing import Callable

import torch

SCHEDULER_EPS = 1e-5


def linear_scheduler(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """κ(t) = t."""
    return time, torch.ones_like(time)


def cosine_scheduler(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Smooth cosine ramp: κ(t) = sin²(θ(t)),  θ(t) = (t + s) / (1 + s) · π/2.

  The small offset s > 0 keeps κ(0) ≈ 0 and κ'(0) > 0 numerically stable.
    """
    s = 0.008
    theta = (time + s) / (1 + s) * (torch.pi / 2)
    kt = torch.sin(theta).pow(2)
    kt_derivative = (torch.sin(2 * theta) * (torch.pi / 2) / (1 + s)).clamp(min=0.0)
    return kt, kt_derivative


def power1_5_scheduler(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convex power law: κ(t) = t^1.5. Delayed initial unmasking, rapid convergence."""
    kt = time.clamp(min=0.0).pow(1.5)
    kt_derivative = (1.5 * time.clamp(min=1e-5).pow(0.5)).clamp(min=0.0)
    return kt, kt_derivative


def power2_scheduler(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadratic schedule: κ(t) = t². Slow start, fast crystallization."""
    kt = time.clamp(min=0.0).pow(2)
    kt_derivative = (2.0 * time.clamp(min=0.0)).clamp(min=0.0)
    return kt, kt_derivative


def sigmoid_scheduler(time: torch.Tensor, k: float = 6.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth S-curve sigmoid schedule: slow start, fast middle, smooth landing."""
    s_t = torch.sigmoid(k * (time - 0.5))
    s_0 = torch.sigmoid(torch.tensor(-0.5 * k, device=time.device, dtype=time.dtype))
    s_1 = torch.sigmoid(torch.tensor(0.5 * k, device=time.device, dtype=time.dtype))
    norm = (s_1 - s_0).clamp(min=1e-6)
    kt = (s_t - s_0) / norm
    kt_derivative = (k * s_t * (1.0 - s_t)) / norm
    return kt.clamp(0.0, 1.0), kt_derivative.clamp(min=0.0)


def min_snr_scheduler(time: torch.Tensor, gamma: float = 5.0, logsnr_min: float = -10.0, logsnr_max: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Variance-preserving schedule from diffusion models, adapted for flow matching.
    Ref: https://arxiv.org/abs/2303.09556 (Common Diffusion Noise Schedules)
    κ(t) = σ(-logsnr(t))
    """
    t_min = torch.exp(-gamma * torch.tensor(logsnr_min, device=time.device, dtype=time.dtype))
    t_max = torch.exp(-gamma * torch.tensor(logsnr_max, device=time.device, dtype=time.dtype))

    # Interpolate in the exponentiated space
    s_t = t_min + time * (t_max - t_min)
    logsnr_t = -torch.log(s_t) / gamma

    # κ(t) = σ(-logsnr(t))
    kt = torch.sigmoid(logsnr_t)

    # Derivative κ'(t) via chain rule
    # dκ/d(logsnr) = σ(logsnr)(1-σ(logsnr)) = κ(1-κ)
    # d(logsnr)/dt = -(t_max - t_min) / (gamma * s_t)
    kt_derivative = kt * (1 - kt) * (-(t_max - t_min) / (gamma * s_t))

    return kt.clamp(0.0, 1.0), kt_derivative.clamp(min=0.0)


SCHEDULER_REGISTRY: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
    "linear": linear_scheduler,
    "cosine": cosine_scheduler,
    "power1_5": power1_5_scheduler,
    "power2": power2_scheduler,
    "sigmoid": sigmoid_scheduler,
    "min_snr": min_snr_scheduler,
}


def get_scheduler(
    scheduler: str | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Resolve a scheduler name or return the callable directly."""
    if callable(scheduler):
        return scheduler
    name = scheduler.lower().strip()
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler '{scheduler}'. Available: {list(SCHEDULER_REGISTRY.keys())}")
    return SCHEDULER_REGISTRY[name]



def clean_weight_denominator(kt: torch.Tensor, eps: float = SCHEDULER_EPS) -> torch.Tensor:
    """Clamp 1 - κ(t) for reverse-step denominators."""
    return (1.0 - kt).clamp(min=eps)


def verify_scheduler(
    scheduler: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    name: str = "scheduler",
    atol: float = 1e-3,
) -> dict[str, float | bool]:
    """
    Check κ(0) ≈ 0, κ(1) ≈ 1, monotonicity, and κ' ≥ 0 on a dense grid.

    Returns a summary dict; raises ValueError if checks fail.
    """
    grid = torch.linspace(0.0, 1.0, 1001)
    kt, kt_derivative = scheduler(grid)

    k0 = float(kt[0])
    k1 = float(kt[-1])
    min_derivative = float(kt_derivative.min())
    monotone = bool((kt[1:] >= kt[:-1] - 1e-7).all())

    summary = {
        "name": name,
        "k(0)": k0,
        "k(1)": k1,
        "k_min_derivative": min_derivative,
        "monotone": monotone,
    }

    if not torch.allclose(torch.tensor(k0), torch.tensor(0.0), atol=atol):
        raise ValueError(f"{name} failed check: κ(0)={k0} is not close to 0.")
    if not torch.allclose(torch.tensor(k1), torch.tensor(1.0), atol=atol):
        raise ValueError(f"{name} failed check: κ(1)={k1} is not close to 1.")
    if not monotone:
        raise ValueError(f"{name} failed check: κ(t) is not monotone increasing.")
    if min_derivative < -atol:
        raise ValueError(f"{name} failed check: κ'(t) has negative values ({min_derivative}).")

    return summary
