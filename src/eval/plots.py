"""Publication-quality plotting utilities for Precision-Coverage curves and PAUC."""

from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend for remote server execution
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from eval.metrics import compute_precision_coverage_curve, compute_precision_recall_curve


def plot_pauc_curve(
    scores: list[float] | np.ndarray,
    exact_matches: list[bool] | np.ndarray,
    mass_matches: list[bool] | np.ndarray,
    output_path: str | Path,
    title: str = "De Novo Peptide Sequencing Precision-Recall & Precision-Coverage Analysis",
    calibrated_threshold: float | None = None,
    calibrated_coverage: float | None = None,
    calibrated_precision: float | None = None,
    calibrated_recall: float | None = None,
) -> Path | None:
    """
    Generates and saves a three-panel publication-ready figure:
    1. Peptide Precision-Recall Curve (Precision vs Peptide Recall) with PR-AUC
    2. Precision vs Coverage Curve (AUPCC) with Calibrated Operating Point
    3. Bayesian Confidence Score Distribution (Correct vs Incorrect) with Calibrated Threshold
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed; skipping PAUC plot.")
        return None

    scores_arr = np.asarray(scores, dtype=np.float64)
    exact_arr = np.asarray(exact_matches, dtype=bool)
    mass_arr = np.asarray(mass_matches, dtype=bool)

    if len(scores_arr) == 0:
        return None

    # Compute Precision-Coverage curves
    cov_exact, prec_cov_exact, _, aupcc_exact, pauc80_exact, _ = compute_precision_coverage_curve(
        exact_arr, scores_arr
    )
    cov_mass, prec_cov_mass, _, aupcc_mass, pauc80_mass, _ = compute_precision_coverage_curve(
        mass_arr, scores_arr
    )

    # Compute Peptide-Level Precision-Recall curves
    rec_exact, prec_pr_exact, _, prauc_exact, p_prauc80_exact, _ = compute_precision_recall_curve(
        exact_arr, scores_arr
    )
    rec_mass, prec_pr_mass, _, prauc_mass, p_prauc80_mass, _ = compute_precision_recall_curve(
        mass_arr, scores_arr
    )

    if calibrated_recall is None and calibrated_coverage is not None and calibrated_precision is not None:
        calibrated_recall = calibrated_coverage * calibrated_precision

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)

    # -------------------------------------------------------------
    # Panel 1: Peptide Precision-Recall Curve (InstaNovo Benchmark Standard)
    # -------------------------------------------------------------
    ax1.plot(
        rec_mass,
        prec_pr_mass,
        color="#1f77b4",
        lw=2.5,
        label=f"Mass Match (PR-AUC={prauc_mass:.3f}, pAUC80={p_prauc80_mass:.3f})",
    )
    mask_mpr80 = prec_pr_mass >= 0.80
    if np.any(mask_mpr80):
        ax1.fill_between(
            rec_mass[mask_mpr80],
            prec_pr_mass[mask_mpr80],
            0.80,
            color="#1f77b4",
            alpha=0.15,
            label="Mass pPR-AUC (P >= 80%)",
        )

    ax1.plot(
        rec_exact,
        prec_pr_exact,
        color="#2ca02c",
        lw=2.5,
        label=f"Exact Match (PR-AUC={prauc_exact:.3f})",
    )

    ax1.axhline(0.80, color="gray", linestyle=":", alpha=0.7, label="Target Precision (80%)")
    ax1.axhline(0.90, color="gray", linestyle="--", alpha=0.7, label="High Precision (90%)")

    if calibrated_recall is not None and calibrated_precision is not None:
        ax1.scatter(
            [calibrated_recall],
            [calibrated_precision],
            color="red",
            s=90,
            zorder=5,
            marker="*",
            label=f"Calibrated (R={calibrated_recall:.1%}, P={calibrated_precision:.1%})",
        )

    ax1.set_xlabel("Peptide Recall (Identified / Total Spectra)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Peptide Precision", fontsize=10.5, fontweight="bold")
    ax1.set_title("Peptide-Level Precision-Recall Curves", fontsize=11.5, fontweight="bold")
    ax1.set_xlim(0.0, max(0.6, float(np.max(rec_mass)) * 1.1))
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Precision-Coverage Curve (AUPCC)
    # -------------------------------------------------------------
    ax2.plot(
        cov_mass,
        prec_cov_mass,
        color="#1f77b4",
        lw=2.5,
        label=f"Mass Match (AUPCC={aupcc_mass:.3f})",
    )
    if np.any(prec_cov_mass >= 0.80):
        mask_cov80 = prec_cov_mass >= 0.80
        ax2.fill_between(
            cov_mass[mask_cov80],
            prec_cov_mass[mask_cov80],
            0.80,
            color="#1f77b4",
            alpha=0.15,
            label="Mass pAUPCC (P >= 80%)",
        )

    ax2.plot(
        cov_exact,
        prec_cov_exact,
        color="#2ca02c",
        lw=2.5,
        label=f"Exact Match (AUPCC={aupcc_exact:.3f})",
    )

    ax2.axhline(0.80, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(0.90, color="gray", linestyle="--", alpha=0.7)

    if calibrated_coverage is not None and calibrated_precision is not None:
        ax2.scatter(
            [calibrated_coverage],
            [calibrated_precision],
            color="red",
            s=90,
            zorder=5,
            marker="*",
            label=f"Calibrated (Cov={calibrated_coverage:.1%}, P={calibrated_precision:.1%})",
        )

    ax2.set_xlabel("Coverage (Fraction of Spectra Retained)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Peptide Precision", fontsize=10.5, fontweight="bold")
    ax2.set_title("Precision-Coverage Curves (AUPCC)", fontsize=11.5, fontweight="bold")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 3: Bayesian Confidence Score Distribution
    # -------------------------------------------------------------
    if len(scores_arr) > 50000:
        sub_idx = np.random.choice(len(scores_arr), size=50000, replace=False)
        sub_scores = scores_arr[sub_idx]
        sub_exact = exact_arr[sub_idx]
    else:
        sub_scores = scores_arr
        sub_exact = exact_arr

    bins = np.linspace(
        float(np.percentile(sub_scores, 1)),
        float(np.percentile(sub_scores, 99)),
        50,
    )

    ax3.hist(
        sub_scores[sub_exact],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#2ca02c",
        label=f"Exact Matches (N={int(np.sum(exact_arr))})",
    )
    ax3.hist(
        sub_scores[~sub_exact],
        bins=bins,
        density=True,
        alpha=0.5,
        color="#d62728",
        label=f"Mismatches (N={int(np.sum(~exact_arr))})",
    )

    if calibrated_threshold is not None:
        ax3.axvline(
            calibrated_threshold,
            color="black",
            linestyle="--",
            lw=2.0,
            label=f"Calibrated Thresh = {calibrated_threshold:.2f}",
        )

    ax3.set_xlabel("Bayesian Joint Posterior Score", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=10.5, fontweight="bold")
    ax3.set_title("Confidence Score Calibration Separation", fontsize=11.5, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3-panel PR/AUPCC/Calibration plot to {out_p}")
    return out_p
