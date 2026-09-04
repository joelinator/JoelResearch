#!/usr/bin/env python3
"""Generates all presentation figures in presentation/figures/."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorboard.backend.event_processing.event_accumulator as ea

OUTPUT_DIR = Path("presentation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def plot_loss_curves():
    acc = ea.EventAccumulator("artifacts/dfm_pl_run_20260903_092757/version_1")
    acc.Reload()

    # Extract validation scalars (one per epoch, total 20 epochs)
    v_loss = [e.value for e in acc.Scalars("valid/loss")]
    v_dec = [e.value for e in acc.Scalars("valid/decoder_loss")]
    v_len = [e.value for e in acc.Scalars("valid/length_loss")]
    v_tok_acc = [e.value * 100 for e in acc.Scalars("valid/token_accuracy")]
    v_len_acc = [e.value * 100 for e in acc.Scalars("valid/length_accuracy")]
    epochs = list(range(len(v_loss)))

    # Extract training scalars and aggregate per epoch
    t_loss_raw = [e.value for e in acc.Scalars("train/loss")]
    t_dec_raw = [e.value for e in acc.Scalars("train/decoder_loss")]
    t_len_raw = [e.value for e in acc.Scalars("train/length_loss")]
    
    # Subsample or group train loss to match epoch scale
    chunk_size = len(t_loss_raw) // len(epochs)
    t_loss = [float(np.mean(t_loss_raw[i*chunk_size : (i+1)*chunk_size])) for i in range(len(epochs))]
    t_dec = [float(np.mean(t_dec_raw[i*chunk_size : (i+1)*chunk_size])) for i in range(len(epochs))]
    t_len = [float(np.mean(t_len_raw[i*chunk_size : (i+1)*chunk_size])) for i in range(len(epochs))]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    # 1. Total Loss
    ax1.plot(epochs, t_loss, "o-", color="#1f77b4", lw=2.2, label="Train Total Loss")
    ax1.plot(epochs, v_loss, "s--", color="#d62728", lw=2.2, label="Valid Total Loss")
    ax1.axvline(4, color="gray", linestyle=":", alpha=0.7, label="Initial Sanity Gate (Epoch 4)")
    ax1.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Total Composite Loss Progression", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5)

    # 2. Decoder Flow Matching Loss
    ax2.plot(epochs, t_dec, "o-", color="#2ca02c", lw=2.2, label="Train Flow Loss")
    ax2.plot(epochs, v_dec, "s--", color="#9467bd", lw=2.2, label="Valid Flow Loss")
    ax2.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax2.set_title("Discrete Flow Matching Velocity Loss ($L_{FM}$)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9.5)

    # 3. Length Predictor Loss
    ax3.plot(epochs, t_len, "o-", color="#ff7f0e", lw=2.2, label="Train Length Loss")
    ax3.plot(epochs, v_len, "s--", color="#8c564b", lw=2.2, label="Valid Length Loss")
    ax3.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax3.set_title("Peptide Length Predictor Loss ($L_{len}$)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9.5)

    # 4. Token & Length Accuracy
    ax4.plot(epochs, v_tok_acc, "^-", color="#17becf", lw=2.4, label="Residue Token Acc (%)")
    ax4.plot(epochs, v_len_acc, "D-", color="#e377c2", lw=2.4, label="Peptide Length Acc (%)")
    ax4.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Validation Accuracy Convergence (Token & Length)", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="lower right", fontsize=9.5)

    plt.suptitle("DFM Training & Validation Dynamics Across 20 Epochs", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "loss_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_generative_performance():
    categories = [
        "Exact Peptide\nAccuracy",
        "Mass-Based\nAccuracy",
        "Peptide Length\nAccuracy",
        "Residue Amino\nAcid F1",
    ]

    ep10_vals = [24.54, 40.98, 60.78, 51.09]
    ep19_vals = [25.86, 42.02, 61.17, 51.98]
    ep19_sota_vals = [30.52, 49.31, 78.84, 58.69]

    x = np.arange(len(categories))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    rects1 = ax.bar(x - width, ep10_vals, width, label="Epoch 10 (Greedy, CFG 1.0)", color="#aec7e8", edgecolor="#1f77b4", lw=1.2)
    rects2 = ax.bar(x, ep19_vals, width, label="Epoch 19 (Greedy, CFG 1.0)", color="#1f77b4", edgecolor="#084594", lw=1.2)
    rects3 = ax.bar(x + width, ep19_sota_vals, width, label="Epoch 19 + Top-3 Beam + CFG 1.5", color="#2ca02c", edgecolor="#006d2c", lw=1.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.annotate("+4.66% Exact\n(+18.0% Rel)",
                xy=(x[0] + width, ep19_sota_vals[0]),
                xytext=(x[0] + width + 0.05, ep19_sota_vals[0] + 6.0),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight="bold", color="#006d2c", ha="center")

    ax.annotate("+17.67% Length\n(+28.9% Rel)",
                xy=(x[2] + width, ep19_sota_vals[2]),
                xytext=(x[2] + width + 0.05, ep19_sota_vals[2] + 6.0),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight="bold", color="#006d2c", ha="center")

    ax.set_ylabel("Accuracy / F1 (%) on Full Validation Split (257k Spectra)", fontsize=11, fontweight="bold")
    ax.set_title("De Novo Generative Sequencing Performance Progression", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "generative_performance.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_cfg_tuning():
    cfgs = [1.0, 1.2, 1.35, 1.5, 1.8]
    exact_accs = [28.32, 29.11, 29.46, 30.03, 29.20]
    mass_accs = [50.45, 51.72, 52.61, 53.00, 51.80]
    aa_f1s = [58.82, 59.95, 60.53, 61.19, 60.10]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax1.plot(cfgs, exact_accs, "o-", color="#2ca02c", lw=2.5, label="Exact Peptide Match (%)")
    ax1.plot(cfgs, mass_accs, "s-", color="#1f77b4", lw=2.5, label="Mass-Based Match (%)")
    ax1.plot(cfgs, aa_f1s, "^-", color="#ff7f0e", lw=2.5, label="Amino Acid F1 (%)")

    ax1.axvline(1.5, color="red", linestyle="--", alpha=0.8, lw=1.8, label="Optimal Guidance Scale (s* = 1.5)")
    ax1.scatter([1.5], [30.03], color="red", s=100, zorder=5)

    ax1.set_xlabel("Classifier-Free Guidance Scale ($s$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Metrics (%) on 10k Subset", fontsize=11, fontweight="bold")
    ax1.set_title("Hyperparameter Sensitivity: Classifier-Free Guidance (CFG)", fontsize=12.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "cfg_tuning.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.axis("off")

    c_spectrum = "#e8f4f8"
    c_encoder = "#d1e7dd"
    c_length = "#fff3cd"
    c_decoder = "#cfe2ff"
    c_beam = "#f8d7da"

    def draw_box(x, y, w, h, title, subtitle, color, edgecolor="#333333"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=color, edgecolor=edgecolor, lw=1.8
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.68, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w/2, y + h*0.32, subtitle, ha="center", va="center", fontsize=8.5, color="#444444")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#222222")
        )
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Boxes
    draw_box(0.02, 0.58, 0.16, 0.32, "MS/MS Spectrum", "m/z peaks + Intensities\nPrecursor m/z + Charge", c_spectrum)
    draw_box(0.02, 0.12, 0.16, 0.32, "Peptide Sequence Noise", "Discrete Categorical Prior\nx_1 ~ Dirichlet / Uniform", c_spectrum)
    draw_box(0.24, 0.58, 0.20, 0.32, "Spectrum Encoder", "Sinusoidal m/z Embedding\nTransformer Encoder (d=512)\nNested Tensor Attention", c_encoder)
    draw_box(0.50, 0.65, 0.20, 0.25, "Length Predictor", "Cross-Attention Pooling\n+ Precursor Mass MLP\nCategorical Dist P(L|S)", c_length)
    draw_box(0.48, 0.15, 0.24, 0.38, "Discrete Flow Decoder", "Continuous Time t in [0, 1]\nLinear Probability Path\nSpectrum Cross-Attention\nVelocity Field v_theta(x_t, t, c)", c_decoder)
    draw_box(0.78, 0.28, 0.20, 0.44, "Bayesian Beam Search", "1. Top-k Lengths (k=3)\n2. DFM Integration (20 steps)\n3. Joint Posterior Scoring:\nS = log P(L) + log P(y|S)\n   - alpha * |mass(y) - m|", c_beam)

    # Connections
    draw_arrow(0.18, 0.74, 0.24, 0.74, "Peaks")
    draw_arrow(0.44, 0.74, 0.50, 0.74, "Encoder Mem")
    draw_arrow(0.34, 0.58, 0.52, 0.45, "Conditioning c")
    draw_arrow(0.18, 0.28, 0.48, 0.28, "Noisy State x_t")
    draw_arrow(0.60, 0.65, 0.60, 0.53, "Length Prior")
    draw_arrow(0.70, 0.77, 0.78, 0.55, "Top-3 Lengths")
    draw_arrow(0.72, 0.34, 0.78, 0.38, "Denoised Tokens")

    ax.set_title("Discrete Flow Matching Architecture for De Novo Peptide Sequencing", fontsize=14, fontweight="bold", y=0.97)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "architecture_diagram.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_generative_performance()
    plot_cfg_tuning()
    plot_architecture_diagram()
    print("All presentation figures successfully generated.")
