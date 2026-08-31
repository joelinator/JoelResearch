"""Tests for de novo evaluation metrics."""

from eval.metrics import compute_denovo_metrics, peptide_matches_mass_based


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


if __name__ == "__main__":
    test_exact_peptide_match()
    test_mass_based_peptide_match()
    test_aa_precision_recall_perfect()
    print("All metrics checks passed.")
