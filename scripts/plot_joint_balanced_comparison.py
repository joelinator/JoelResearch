#!/usr/bin/env python3
"""
Generate publication-quality 4-panel comparison figure comparing:
1. Base Model (HC-PT only)
2. Finetuned Model (Nine-Species Phase 2 only)
3. InstaNovo Benchmark Baseline
4. Balanced Joint Model (Universal Multi-Domain)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})

FIG_PATH = Path("/home/joelgedeon_aims_ac_za/.gemini/antigravity-cli/brain/ee9cf031-b0c5-4740-b963-40c81c13ea78/joint_balanced_multi_domain_comparison.png")

# Data definitions
models = ["Base (HC-PT)", "Finetuned (9-Sp)", "InstaNovo", "Joint Balanced (Ours)"]
colors = ["#4A90E2", "#E25959", "#F5A623", "#2ECC71"]

# 1. Nine-Species Full Test Split (N=104,163)
ns_exact_il = [50.49, 65.63, 62.06, 65.07]
ns_aa_f1 = [71.68, 82.29, 76.88, 81.97]
ns_len_acc = [75.86, 83.58, 80.65, 83.27]

# 2. HC-PT Full Test Split (N=265,369)
hc_exact_il = [56.24, 48.57, 50.85, 55.95]  # InstaNovo test on HC-PT was 50.85%
hc_exact_strict = [36.41, 12.85, 23.40, 34.93]
hc_aa_f1 = [69.87, 69.55, 68.20, 70.04]

# 3. Cross-Domain Retention Delta (HC-PT recovery vs Catastrophic Forgetting)
# Model balance: (NS score + HCPT score) / 2
joint_harmonic_mean = [
    2 * (50.49 * 56.24) / (50.49 + 56.24),
    2 * (65.63 * 48.57) / (65.63 + 48.57),
    2 * (62.06 * 50.85) / (62.06 + 50.85),
    2 * (65.07 * 55.95) / (65.07 + 55.95),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Universal Multi-Domain Peptide Sequencing: Balanced Joint Model vs Baselines", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Nine-Species Full Test (N=104,163)
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, ns_exact_il, width, label="I/L Exact Match (%)", color="#2ECC71", alpha=0.85)
bars2 = ax1.bar(x, ns_aa_f1, width, label="Amino Acid F1 (%)", color="#3498DB", alpha=0.85)
bars3 = ax1.bar(x + width, ns_len_acc, width, label="Length Accuracy (%)", color="#9B59B6", alpha=0.85)

ax1.set_title("A. Nine-Species Benchmark (Full Test Split, N=104,163)", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight="semibold")
ax1.set_ylabel("Metric Score (%)")
ax1.set_ylim(40, 95)
ax1.legend(loc="upper left", frameon=True)

for bar in ax1.patches:
    h = bar.get_height()
    if h > 0:
        ax1.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 2: HC-PT ProteomeTools Full Test (N=265,369)
ax2 = axes[0, 1]
bars4 = ax2.bar(x - width, hc_exact_il, width, label="I/L Exact Match (%)", color="#E67E22", alpha=0.85)
bars5 = ax2.bar(x, hc_exact_strict, width, label="Strict Exact Match (%)", color="#E74C3C", alpha=0.85)
bars6 = ax2.bar(x + width, hc_aa_f1, width, label="Amino Acid F1 (%)", color="#1ABC9C", alpha=0.85)

ax2.set_title("B. HC-PT ProteomeTools (Full Test Split, N=265,369)", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight="semibold")
ax2.set_ylabel("Metric Score (%)")
ax2.set_ylim(0, 80)
ax2.legend(loc="upper left", frameon=True)

for bar in ax2.patches:
    h = bar.get_height()
    if h > 0:
        ax2.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 3: Domain Generalization & Catastrophic Forgetting Tradeoff (Harmonic Mean Score)
ax3 = axes[1, 0]
bars7 = ax3.bar(models, joint_harmonic_mean, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
ax3.set_title("C. Cross-Domain Generalization (Harmonic Mean of Both Benchmarks)", fontweight="bold")
ax3.set_ylabel("Harmonic Mean I/L Exact (%)")
ax3.set_ylim(40, 68)

for bar, val in zip(bars7, joint_harmonic_mean):
    ax3.annotate(f"{val:.2f}%",
                 xy=(bar.get_x() + bar.get_width() / 2, val),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

ax3.axhline(55.8, color="gray", linestyle="--", alpha=0.7, label="InstaNovo Level (55.8%)")
ax3.legend(loc="upper left")

# Panel 4: 2D Domain Landscape (Nine-Species vs HC-PT Accuracy)
ax4 = axes[1, 1]
ax4.set_title("D. 2D Domain Generalization Landscape", fontweight="bold")
ax4.set_xlabel("Nine-Species Full Test I/L Exact Match (%)", fontweight="semibold")
ax4.set_ylabel("HC-PT Full Test I/L Exact Match (%)", fontweight="semibold")

markers = ["o", "s", "^", "*"]
sizes = [140, 140, 160, 260]

for i, (m, c, mark, sz) in enumerate(zip(models, colors, markers, sizes)):
    ax4.scatter(ns_exact_il[i], hc_exact_il[i], color=c, s=sz, marker=mark, label=m, edgecolors="black", linewidth=1.5, zorder=5)
    offset_y = 1.0 if m != "Base (HC-PT)" else -1.8
    offset_x = 0.0 if m != "Joint Balanced (Ours)" else -1.5
    ax4.annotate(m, (ns_exact_il[i] + offset_x, hc_exact_il[i] + offset_y), fontsize=10, fontweight="bold", ha="center")

# Draw Pareto frontier / reference box
ax4.plot([50.49, 65.07], [56.24, 55.95], linestyle=":", color="#2ECC71", alpha=0.6, linewidth=1.5)
ax4.plot([65.63, 65.07], [48.57, 55.95], linestyle=":", color="#2ECC71", alpha=0.6, linewidth=1.5)

ax4.set_xlim(48, 68)
ax4.set_ylim(45, 60)
ax4.legend(loc="lower left", frameon=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(FIG_PATH, dpi=300)
print(f"Publication figure saved to: {FIG_PATH}")
