#!/usr/bin/env python3
"""
Generate publication-quality CTMC flow dynamics, biological fidelity, and efficiency figures.

Figures generated:
1. ctmc_flow_dynamics.png: Token unmasking trajectory heatmap, Shannon entropy decay, and jump flux.
2. proteomics_biological_fidelity.png: AA frequency parity (y=x), precursor mass residual KDE, and fragment ion coverage.
3. sampling_dynamics_and_latency.png: Accuracy vs NFE steps, and latency vs sequence length O(K) vs O(L).
4. precision_coverage_benchmark.png: Residue and peptide precision vs coverage curves across models.
5. length_dependent_accuracy.png: Multi-model accuracy breakdown across peptide length intervals.
6. scheduler_and_knapsack_ablation.png: Probability interpolant and knapsack tolerance ablation.

ALL data loaded 100% dynamically from ground truth evaluation artifact files in artifacts/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "docs" / "figures"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
THESIS_DIR = PROJECT_ROOT / "thesis" / "images"
THESIS_DIR.mkdir(parents=True, exist_ok=True)
PRES_DIR = PROJECT_ROOT / "presentation" / "images"
PRES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

_output_env = os.environ.get("OUTPUT_FIGURES_DIR")
OUTPUT_DIR = Path(_output_env) if _output_env else None

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.0,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})

# Curated Okabe-Ito Colorblind-Safe Palette
COLOR_BLUE = "#0072B2"
COLOR_VERMILLION = "#D55E00"
COLOR_TEAL = "#009E73"
COLOR_PURPLE = "#CC79A7"
COLOR_ORANGE = "#E69F00"
COLOR_SKY_BLUE = "#56B4E9"
COLOR_DARK = "#1A252F"


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save figure to docs/figures, thesis/images, and presentation/images at 300 DPI."""
    targets = [DOCS_DIR / filename, THESIS_DIR / filename, PRES_DIR / filename]
    if OUTPUT_DIR and OUTPUT_DIR.exists():
        targets.append(OUTPUT_DIR / filename)

    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Saved: {p}")


def load_json(filename: str) -> dict:
    """Load JSON file from artifacts directory."""
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with open(path) as f:
        return json.load(f)


# =========================================================================
# Figure 1: CTMC Flow Dynamics & Jump Trajectory
# =========================================================================
def generate_figure1_ctmc_flow_dynamics() -> None:
    print("Generating Figure 1: CTMC Flow Dynamics & Jump Trajectory from artifacts/ctmc_flow_trajectory_empirical.json...")
    traj_path = ARTIFACTS_DIR / "ctmc_flow_trajectory_empirical.json"
    
    if not traj_path.exists():
        print(f"Artifact {traj_path} not found. Running empirical extraction...")
        from scripts.extract_ctmc_trajectory import extract_trajectory
        extract_trajectory()

    data = load_json("ctmc_flow_trajectory_empirical.json")
    meta = data["metadata"]
    traj_info = data["flow_trajectory"]
    flux_info = data["flux_schedule"]

    target_seq = meta["target_sequence"]
    seq_len = meta["length"]
    trajectory_tokens = traj_info["trajectory_tokens"]
    conf_history = traj_info["confidence_matrix"]
    entropy_history = traj_info["entropy_matrix"]
    num_steps = meta["num_steps"]

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel 1A: Discrete Token Unmasking Heatmap
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    steps_arr = np.arange(len(trajectory_tokens))
    heat_matrix = np.zeros((seq_len, len(steps_arr)))

    for step_idx in range(len(steps_arr)):
        for pos in range(seq_len):
            tok = trajectory_tokens[step_idx][pos]
            if "mask" in tok.lower() or tok == "[MASK]":
                heat_matrix[pos, step_idx] = 0.05
            else:
                heat_matrix[pos, step_idx] = conf_history[step_idx][pos]

    cmap = plt.cm.Blues
    im = ax1.imshow(heat_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")

    # Overlay token characters in cells
    for step_idx in range(len(steps_arr)):
        for pos in range(seq_len):
            tok = trajectory_tokens[step_idx][pos]
            is_masked = "mask" in tok.lower() or tok == "[MASK]"
            display_char = "•" if is_masked else tok
            color = "#777777" if is_masked else "white" if heat_matrix[pos, step_idx] > 0.6 else "black"
            fontweight = "normal" if is_masked else "bold"
            fontsize = 9 if not is_masked else 12
            ax1.text(step_idx, pos, display_char, ha="center", va="center",
                     color=color, fontsize=fontsize, fontweight=fontweight)

    ax1.set_title(f"A. Empirical Discrete Token State Trajectory across Normalized Flow Time (Peptide: {target_seq}, L={seq_len})",
                  fontweight="bold", fontsize=13, pad=12)
    ax1.set_xlabel(f"Flow Step $k$ ($t = k / {num_steps}$, from Noisy Prior $t=0$ to Decoded Clean Sequence $t=1$)", fontweight="bold")
    ax1.set_ylabel("Sequence Position ($1 \\dots L$)", fontweight="bold")
    ax1.set_xticks(range(0, len(steps_arr), 2))
    ax1.set_xticklabels([f"{s}\n(t={s/num_steps:.2f})" for s in range(0, len(steps_arr), 2)])
    ax1.set_yticks(range(seq_len))
    ax1.set_yticklabels([f"Pos {i+1} ({target_seq[i]})" for i in range(seq_len)])

    cbar = fig.colorbar(im, ax=ax1, orientation="vertical", pad=0.02, shrink=0.9)
    cbar.set_label("Token Probability / Confidence", fontweight="bold")

    # -------------------------------------------------------------
    # Panel 1B: Positional Shannon Entropy Decay
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ent_arr = np.array(entropy_history)
    time_pts = np.linspace(0.0, 1.0, len(ent_arr))

    # Plot informative positions: C-term, N-term, internal positions, and mean
    ax2.plot(time_pts, ent_arr[:, -1], label=f"C-term (Pos {seq_len}: {target_seq[-1]}) [Tryptic Anchor]", color=COLOR_VERMILLION, linewidth=2.5)
    ax2.plot(time_pts, ent_arr[:, 0], label=f"N-term (Pos 1: {target_seq[0]}) [N-terminal Ladder]", color=COLOR_BLUE, linewidth=2.2)
    mid_pos = seq_len // 2
    ax2.plot(time_pts, ent_arr[:, mid_pos], label=f"Internal (Pos {mid_pos+1}: {target_seq[mid_pos]})", color=COLOR_TEAL, linewidth=2.0)
    ax2.plot(time_pts, ent_arr[:, seq_len - 2], label=f"Internal (Pos {seq_len-1}: {target_seq[seq_len-2]})", color=COLOR_ORANGE, linewidth=2.0, linestyle="--")
    ax2.plot(time_pts, ent_arr.mean(axis=1), label="Mean Sequence Entropy", color=COLOR_DARK, linewidth=2.5, linestyle=":")

    ax2.set_title("B. Empirical Positional Categorical Shannon Entropy Decay Across Flow Time", fontweight="bold", fontsize=12.5)
    ax2.set_xlabel("Flow Time $t \\in [0, 1]$", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Shannon Entropy $H(p_t)$ (Bits)", fontweight="bold", fontsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, max(0.6, float(ent_arr.max()) * 1.15))
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=9.5)

    # -------------------------------------------------------------
    # Panel 1C: Jump Flux / Velocity Profile Across Flow Time
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    t_vals = np.array(flux_info["t_continuous"])
    kappa_t = np.array(flux_info["kappa_t"])
    kappa_deriv = np.array(flux_info["kappa_deriv"])
    unmasking_flux = np.array(flux_info["unmasking_flux"])

    ax3.plot(t_vals, kappa_t, label="Cosine State Interpolant $\\kappa(t)$", color=COLOR_TEAL, linewidth=2.2)
    ax3.plot(t_vals, kappa_deriv, label="Velocity Field Intensity $\\kappa'(t)$", color=COLOR_BLUE, linewidth=2.2)
    ax3.plot(t_vals, unmasking_flux / 5.0, label="Normalized Jump Flux $\\frac{\\kappa'(t)}{1 - \\kappa(t)}$",
             color=COLOR_PURPLE, linewidth=2.2, linestyle="--")

    ax3.set_title("C. Continuous-Time Jump Rate Schedule and Transition Flux", fontweight="bold", fontsize=12.5)
    ax3.set_xlabel("Flow Time $t \\in [0, 1]$", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Rate Magnitude / Transition Probability", fontweight="bold", fontsize=11)
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 2.5)
    ax3.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=9.5)

    save_figure(fig, "ctmc_flow_dynamics.png")
    plt.close(fig)


def main():
    print("Generating complete publication and thesis CTMC visualization suite (100% Artifact-Driven)...")
    generate_figure1_ctmc_flow_dynamics()

    # Delegate to unified high-contrast, colorblind-safe publication generators
    from scripts.generate_publication_figures import (
        generate_proteomics_biological_fidelity,
        generate_sampling_dynamics_and_latency,
        generate_precision_coverage_benchmark,
        generate_length_dependent_accuracy,
        generate_scheduler_and_knapsack_ablation,
    )
    generate_proteomics_biological_fidelity()
    generate_sampling_dynamics_and_latency()
    generate_precision_coverage_benchmark()
    generate_length_dependent_accuracy()
    generate_scheduler_and_knapsack_ablation()
    print("All 6 publication/thesis figures successfully generated from artifacts!")


if __name__ == "__main__":
    main()
