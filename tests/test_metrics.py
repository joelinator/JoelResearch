"""Tests for de novo evaluation metrics."""

import numpy as np
from eval.metrics import (
    calibrate_score_threshold,
    compute_denovo_metrics,
    compute_precision_coverage_curve,
    compute_precision_recall_curve,
    peptide_matches_mass_based,
)


def test_exact_peptide_match():
    metrics = compute_denovo_metrics(["PEPTIDE", "ACE"], ["PEPTIDE", "ADE"])
    assert metrics.exact_peptide_accuracy == 0.5
    assert metrics.num_samples == 2


def test_mass_based_peptide_match():
    assert peptide_matches_mass_based("PEPTIDE", "PEPTIDE")
    # I/L are isobaric in monoisotopic mass.
    assert peptide_matches_mass_based("IL", "LI")


def test_aa_precision_recall_perfect():
    metrics = compute_denovo_metrics(["PEPTIDE"], ["PEPTIDE"])
    assert metrics.aa_precision == 1.0
    assert metrics.aa_recall == 1.0
    assert metrics.peptide_precision == 1.0
    assert metrics.peptide_recall == 1.0


def test_exact_vs_mass_differentiation():
    # 'IL' vs 'LI' is a mass match but NOT an exact string match.
    # 'PEPTIDE' vs 'PEPTIDE' is both mass and exact match.
    # 'AAA' vs 'KKK' is neither.
    preds = ["PEPTIDE", "IL", "AAA"]
    targets = ["PEPTIDE", "LI", "KKK"]
    scores = [2.0, 1.0, -1.0]

    metrics = compute_denovo_metrics(preds, targets, scores=scores)
    assert abs(metrics.exact_peptide_accuracy - 1.0 / 3.0) < 1e-4
    assert abs(metrics.mass_peptide_accuracy - 2.0 / 3.0) < 1e-4
    assert metrics.auc_mass > metrics.auc_exact


def test_thresholded_precision_recall():
    preds = ["PEPTIDE", "IL", "AAA", "CCC"]
    targets = ["PEPTIDE", "LI", "KKK", "DDD"]
    scores = [10.0, 5.0, 1.0, 0.0]

    # Threshold at 5.0 accepts top 2: 'PEPTIDE' (exact & mass) and 'IL' (mass only)
    metrics = compute_denovo_metrics(preds, targets, scores=scores, score_threshold=5.0)
    assert metrics.num_predicted_above_threshold == 2
    assert metrics.coverage == 0.5

    # Mass matches: 2 correct out of 2 accepted -> precision = 1.0, recall = 2/4 = 0.5
    assert metrics.peptide_precision_mass == 1.0
    assert metrics.peptide_recall_mass == 0.5

    # Exact matches: 1 correct out of 2 accepted -> precision = 0.5, recall = 1/4 = 0.25
    assert metrics.peptide_precision_exact == 0.5
    assert metrics.peptide_recall_exact == 0.25


def test_calibration_and_pauc():
    scores = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0, -1.0])
    correct = np.array([True, True, True, True, False, False, False, False])

    calib = calibrate_score_threshold(scores, correct, target_precision=1.0)
    assert calib["threshold"] == 4.0
    assert calib["precision"] == 1.0
    assert calib["coverage"] == 0.5

    covs, precs, threshs, auc, pauc80, pauc50 = compute_precision_coverage_curve(correct, scores)
    assert auc > 0.5
    assert pauc80 > 0.0


def test_precision_recall_curve():
    scores = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0, -1.0])
    correct = np.array([True, True, True, True, False, False, False, False])

    recs, precs, threshs, pr_auc, p_pr_auc80, _ = compute_precision_recall_curve(correct, scores)
    assert pr_auc > 0.0
    assert p_pr_auc80 > 0.0
    assert len(recs) == len(scores)
    # Max recall should be 4/8 = 0.5
    assert abs(recs[-1] - 0.5) < 1e-6


if __name__ == "__main__":
    test_exact_peptide_match()
    test_mass_based_peptide_match()
    test_aa_precision_recall_perfect()
    test_exact_vs_mass_differentiation()
    test_thresholded_precision_recall()
    test_calibration_and_pauc()
    test_precision_recall_curve()
    print("All metrics checks passed.")
