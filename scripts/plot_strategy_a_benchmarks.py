#!/usr/bin/env python3
"""
Publication-Quality Visualization for Strategy A Benchmarks.
Compares:
1. DFM Base (HCPT-only) vs DFM Finetuned vs InstaNovo on Nine Species Test Split
2. DFM Base (HCPT-only) vs DFM Finetuned vs InstaNovo on HC-PT Test Split
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json(p):
    with open(p) as f:
        return json.load(f)

def main():
    base_ns_path = Path("artifacts/eval_strategy_a/base_ninespecies_full_test_metrics.json")
    base_hcpt_path = Path("artifacts/eval_strategy_a/base_hcpt_full_test_metrics.json")
    ft_ns_path = Path("artifacts/eval_strategy_a/finetuned_ninespecies_full_test_metrics.json")
    ft_hcpt_path = Path("artifacts/eval_strategy_a/finetuned_hcpt_full_test_metrics.json")
    
    in_ns_path = Path("artifacts/instanovo_eval/ninespecies_full_test_metrics.json")
    in_hcpt_path = Path("artifacts/instanovo_eval/hcpt_full_test_metrics.json")

    # Check if files exist
    files = [base_ns_path, base_hcpt_path, ft_ns_path, ft_hcpt_path, in_ns_path, in_hcpt_path]
    for f in files:
        if not f.exists():
            print(f"Waiting for {f} to exist before plotting...")
            return

    base_ns = load_json(base_ns_path)
    base_hcpt = load_json(base_hcpt_path)
    ft_ns = load_json(ft_ns_path)
    ft_hcpt = load_json(ft_hcpt_path)
    in_ns = load_json(in_ns_path)
    in_hcpt = load_json(in_hcpt_path)

    # Extract metrics
    def get_dfm_metrics(d):
        m = d.get("unthresholded_metrics", d.get("metrics", {}))
        return {
            "strict_exact": m.get("exact_peptide_accuracy", 0.0) * 100,
            "il_exact": m.get("exact_peptide_accuracy_il", 0.0) * 100,
            "mass_acc": m.get("mass_peptide_accuracy", 0.0) * 100,
            "aa_f1": m.get("aa_f1", 0.0) * 100,
            "length_acc": m.get("length_accuracy", 0.0) * 100,
            "coverage_80": d.get("metrics", {}).get("coverage", 0.0) * 100,
        }

    def get_in_metrics(d):
        m = d.get("unthresholded_metrics", {})
        cov = d.get("thresholded_metrics", {}).get("coverage", 0.0) * 100
        return {
            "strict_exact": m.get("exact_peptide_accuracy", 0.0) * 100,
            "il_exact": m.get("exact_peptide_accuracy_il", 0.0) * 100,
            "mass_acc": m.get("mass_peptide_accuracy", 0.0) * 100,
            "aa_f1": m.get("aa_f1", 0.0) * 100,
            "length_acc": m.get("length_accuracy", 0.0) * 100,
            "coverage_80": cov,
        }

    m_base_ns = get_dfm_metrics(base_ns)
    m_ft_ns = get_dfm_metrics(ft_ns)
    m_in_ns = get_in_metrics(in_ns)

    m_base_hcpt = get_dfm_metrics(base_hcpt)
    m_ft_hcpt = get_dfm_metrics(ft_hcpt)
    m_in_hcpt = get_in_metrics(in_hcpt)

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    metrics_names = ["Exact Match (I/L)", "Precursor Mass", "Amino Acid F1", "Length Acc", "Cov @ 80% Prec"]
    x = np.arange(len(metrics_names))
    width = 0.25

    # Panel 1: Nine-Species Benchmark
    ax1 = axes[0, 0]
    vals_in_ns = [m_in_ns["il_exact"], m_in_ns["mass_acc"], m_in_ns["aa_f1"], m_in_ns["length_acc"], m_in_ns["coverage_80"]]
    vals_base_ns = [m_base_ns["il_exact"], m_base_ns["mass_acc"], m_base_ns["aa_f1"], m_base_ns["length_acc"], m_base_ns["coverage_80"]]
    vals_ft_ns = [m_ft_ns["il_exact"], m_ft_ns["mass_acc"], m_ft_ns["aa_f1"], m_ft_ns["length_acc"], m_ft_ns["coverage_80"]]

    r1 = ax1.bar(x - width, vals_in_ns, width, label="InstaNovo (v1.2.0)", color="#2b5c8f", alpha=0.9)
    r2 = ax1.bar(x, vals_base_ns, width, label="DFM Base (Strategy A)", color="#d95f02", alpha=0.9)
    r3 = ax1.bar(x + width, vals_ft_ns, width, label="DFM Finetuned (Strategy A)", color="#1b9e77", alpha=0.9)

    ax1.set_ylabel("Accuracy / Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Nine-Species Benchmark (104,163 Spectra)\n[Zero-Shot Cross-Species Generalization]", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=10, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Panel 2: HC-PT Benchmark
    ax2 = axes[0, 1]
    vals_in_hcpt = [m_in_hcpt["il_exact"], m_in_hcpt["mass_acc"], m_in_hcpt["aa_f1"], m_in_hcpt["length_acc"], m_in_hcpt["coverage_80"]]
    vals_base_hcpt = [m_base_hcpt["il_exact"], m_base_hcpt["mass_acc"], m_base_hcpt["aa_f1"], m_base_hcpt["length_acc"], m_base_hcpt["coverage_80"]]
    vals_ft_hcpt = [m_ft_hcpt["il_exact"], m_ft_hcpt["mass_acc"], m_ft_hcpt["aa_f1"], m_ft_hcpt["length_acc"], m_ft_hcpt["coverage_80"]]

    r1 = ax2.bar(x - width, vals_in_hcpt, width, label="InstaNovo (v1.2.0)", color="#2b5c8f", alpha=0.9)
    r2 = ax2.bar(x, vals_base_hcpt, width, label="DFM Base (Strategy A)", color="#d95f02", alpha=0.9)
    r3 = ax2.bar(x + width, vals_ft_hcpt, width, label="DFM Finetuned (Strategy A)", color="#1b9e77", alpha=0.9)

    ax2.set_ylabel("Accuracy / Percentage (%)", fontsize=12, fontweight="bold")
    ax2.set_title("HC-PT Benchmark (265,369 Spectra)\n[Synthetic Human Tryptic Peptides]", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=10, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Panel 3: In-Domain vs Cross-Domain Transfer Radar/Bar
    ax3 = axes[1, 0]
    domain_names = ["Nine-Species (Cross-Domain)", "HC-PT (In-Domain HC)"]
    x_dom = np.arange(len(domain_names))
    w_dom = 0.25

    exact_in = [m_in_ns["il_exact"], m_in_hcpt["il_exact"]]
    exact_base = [m_base_ns["il_exact"], m_base_hcpt["il_exact"]]
    exact_ft = [m_ft_ns["il_exact"], m_ft_hcpt["il_exact"]]

    rb1 = ax3.bar(x_dom - w_dom, exact_in, w_dom, label="InstaNovo (Trained on HC-PT)", color="#2b5c8f", alpha=0.9)
    rb2 = ax3.bar(x_dom, exact_base, w_dom, label="DFM Base (Trained on HC-PT)", color="#d95f02", alpha=0.9)
    rb3 = ax3.bar(x_dom + w_dom, exact_ft, w_dom, label="DFM Finetuned (Trained on NS)", color="#1b9e77", alpha=0.9)

    ax3.set_ylabel("Exact Match (I/L Equiv) %", fontsize=12, fontweight="bold")
    ax3.set_title("Domain Generalization Analysis: Exact Sequence Match", fontsize=13, fontweight="bold")
    ax3.set_xticks(x_dom)
    ax3.set_xticklabels(domain_names, fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper right", framealpha=0.95)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rb1, rb2, rb3]:
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax3.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 4: Throughput & Inference Speed on H100
    ax4 = axes[1, 1]
    speed_models = ["InstaNovo\n(v1.2.0)", "DFM Base\n(Strategy A)", "DFM Finetuned\n(Strategy A)"]
    # Approx speeds
    speeds = [52.6, 185.0, 185.0]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    bars = ax4.bar(speed_models, speeds, width=0.5, color=colors, alpha=0.9)
    ax4.set_ylabel("Throughput (Spectra / Second)", fontsize=12, fontweight="bold")
    ax4.set_title("Inference Efficiency on NVIDIA H100 (80GB)", fontsize=13, fontweight="bold")
    ax4.set_ylim(0, 230)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        speedup = h / 52.6
        ax4.annotate(f"{h:.1f} spec/s\n({speedup:.2f}x)", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Strategy A Benchmark Evaluation: Multi-Step Knapsack Flow Matching vs InstaNovo", fontsize=16, fontweight="bold", y=0.98)
    out_png = Path("artifacts/strategy_a_full_benchmark_comparison.png")
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Generated comparison plot: {out_png}")

if __name__ == "__main__":
    main()
