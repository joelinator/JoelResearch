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


SCHEDULER_REGISTRY: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
    "linear": linear_scheduler,
    "cosine": cosine_scheduler,
    "power1_5": power1_5_scheduler,
    "power2": power2_scheduler,
    "sigmoid": sigmoid_scheduler,
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
        "kappa_at_0": k0,
        "kappa_at_1": k1,
        "min_kappa_derivative": min_derivative,
        "monotone_increasing": monotone,
    }

    errors = []
    if k0 > atol:
        errors.append(f"κ(0)={k0:.6f}, expected ≈ 0")
    if abs(k1 - 1.0) > atol:
        errors.append(f"κ(1)={k1:.6f}, expected ≈ 1")
    if min_derivative < -1e-7:
        errors.append(f"κ' becomes negative (min={min_derivative:.6f})")
    if not monotone:
        errors.append("κ(t) is not monotone increasing on [0, 1]")

    if errors:
        raise ValueError(f"{name} failed verification: " + "; ".join(errors))

    return summary
