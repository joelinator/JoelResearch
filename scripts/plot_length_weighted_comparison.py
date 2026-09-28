#!/usr/bin/env python3
"""
Generates publication-quality comparison figures for:
1. Baseline vs Length-Weighted Fine-Tuned Model Across Peptide Length Bins (Nine-Species & HC-PT)
2. Saves plots to artifacts/figures/ and brain artifact directory.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Data
bins = ["7-10", "11-14", "15-18", "19-22", "23-30"]
x = np.arange(len(bins))
width = 0.35

# Nine-Species Strict Exact Match
ns_baseline_strict = [89.2, 81.7, 67.3, 50.1, 25.9]
ns_lw_strict = [90.3, 82.5, 70.7, 50.4, 33.6]

# HC-PT Strict Exact Match
hcpt_baseline_strict = [40.4, 35.5, 24.0, 14.4, 5.2]
hcpt_lw_strict = [46.2, 38.5, 26.3, 19.7, 9.6]

# HC-PT I/L Exact Match
hcpt_baseline_il = [71.8, 60.4, 41.8, 28.7, 5.8]
hcpt_lw_il = [72.8, 60.1, 44.0, 32.0, 16.3]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

# 1. Nine-Species Strict
ax = axes[0]
rects1 = ax.bar(x - width/2, ns_baseline_strict, width, label="Baseline DFM (30 ep)", color="#4575b4", alpha=0.9)
rects2 = ax.bar(x + width/2, ns_lw_strict, width, label="Length-Weighted DFM", color="#d73027", alpha=0.9)
ax.set_title("Nine-Species Benchmark: Strict Exact Match", fontsize=12, fontweight="bold")
ax.set_xlabel("Peptide Length Bins (Residues)", fontsize=11)
ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.set_ylim(0, 100)
ax.legend(frameon=True)
for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8)
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# 2. HC-PT Strict
ax = axes[1]
rects1 = ax.bar(x - width/2, hcpt_baseline_strict, width, label="Baseline DFM (30 ep)", color="#4575b4", alpha=0.9)
rects2 = ax.bar(x + width/2, hcpt_lw_strict, width, label="Length-Weighted DFM", color="#d73027", alpha=0.9)
ax.set_title("HC-PT Benchmark: Strict Exact Match", fontsize=12, fontweight="bold")
ax.set_xlabel("Peptide Length Bins (Residues)", fontsize=11)
ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.set_ylim(0, 60)
ax.legend(frameon=True)
for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8)
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# 3. HC-PT I/L Conflated
ax = axes[2]
rects1 = ax.bar(x - width/2, hcpt_baseline_il, width, label="Baseline DFM (30 ep)", color="#4575b4", alpha=0.9)
rects2 = ax.bar(x + width/2, hcpt_lw_il, width, label="Length-Weighted DFM", color="#d73027", alpha=0.9)
ax.set_title("HC-PT Benchmark: I/L-Conflated Exact Match", fontsize=12, fontweight="bold")
ax.set_xlabel("Peptide Length Bins (Residues)", fontsize=11)
ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.set_ylim(0, 90)
ax.legend(frameon=True)
for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8)
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
out_fig = FIG_DIR / "length_weighted_stratification_comparison.png"
plt.savefig(out_fig, dpi=300)
print(f"Plot saved to: {out_fig}")
