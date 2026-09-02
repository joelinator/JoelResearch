"""De novo sequencing evaluation metrics (InstaNovo-compatible definitions)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.constants import AA_MASSES_DICT


def _residue_mass(amino_acid: str) -> float:
    return AA_MASSES_DICT[amino_acid]


def _prefix_mass(sequence: str, end: int) -> float:
    return sum(_residue_mass(amino_acid) for amino_acid in sequence[:end])


def amino_acid_pair_matches(
    pred_aa: str,
    true_aa: str,
    pred_prefix_mass: float,
    true_prefix_mass: float,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
) -> bool:
    """InstaNovo AA match: residue mass and prefix mass within tolerance."""
    if abs(_residue_mass(pred_aa) - _residue_mass(true_aa)) >= aa_mass_tolerance:
        return False
    return abs(pred_prefix_mass - true_prefix_mass) < prefix_mass_tolerance


def count_matching_amino_acids(
    predicted: str,
    target: str,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
) -> int:
    """Maximum number of mass-consistent amino-acid matches via DP alignment."""
    n, m = len(predicted), len(target)
    if n == 0 or m == 0:
        return 0

    # Fast path: identical strings — every position matches.
    if predicted == target:
        return n

    # Pre-compute cumulative prefix masses as numpy arrays (avoids O(N²) Python calls).
    pred_masses = np.array([_residue_mass(aa) for aa in predicted], dtype=np.float64)
    true_masses = np.array([_residue_mass(aa) for aa in target], dtype=np.float64)
    pred_prefixes = np.cumsum(pred_masses)
    true_prefixes = np.cumsum(true_masses)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1, j], dp[i, j - 1])
            if (
                abs(pred_masses[i - 1] - true_masses[j - 1]) < aa_mass_tolerance
                and abs(pred_prefixes[i - 1] - true_prefixes[j - 1]) < prefix_mass_tolerance
            ):
                best = max(best, dp[i - 1, j - 1] + 1)
            dp[i, j] = best
    return int(dp[n, m])


def peptide_matches_mass_based(
    predicted: str,
    target: str,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
) -> bool:
    """Peptide match: equal length and every aligned residue matches by mass."""
    if len(predicted) != len(target):
        return False
    for index in range(len(predicted)):
        if not amino_acid_pair_matches(
            predicted[index],
            target[index],
            _prefix_mass(predicted, index + 1),
            _prefix_mass(target, index + 1),
            aa_mass_tolerance,
            prefix_mass_tolerance,
        ):
            return False
    return True


@dataclass
class DenovoMetrics:
    num_samples: int
    aa_precision: float
    aa_recall: float
    aa_f1: float
    aa_error_rate: float
    peptide_precision: float
    peptide_recall: float
    peptide_f1: float
    exact_peptide_accuracy: float
    length_accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_samples": self.num_samples,
            "aa_precision": self.aa_precision,
            "aa_recall": self.aa_recall,
            "aa_f1": self.aa_f1,
            "aa_error_rate": self.aa_error_rate,
            "peptide_precision": self.peptide_precision,
            "peptide_recall": self.peptide_recall,
            "peptide_f1": self.peptide_f1,
            "exact_peptide_accuracy": self.exact_peptide_accuracy,
            "length_accuracy": self.length_accuracy,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_denovo_metrics(
    predictions: list[str],
    targets: list[str],
    predicted_lengths: list[int] | None = None,
    target_lengths: list[int] | None = None,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
) -> DenovoMetrics:
    """
    Corpus-level metrics following InstaNovo definitions.

    AA precision/recall use mass-based optimal alignment counts.
    Peptide precision/recall use mass-based full-peptide matches.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    num_samples = len(predictions)
    aa_matches = 0
    aa_pred_total = 0
    aa_gold_total = 0
    peptide_matches = 0
    exact_matches = 0
    length_correct = 0

    for index, (predicted, target) in enumerate(zip(predictions, targets)):
        aa_matches += count_matching_amino_acids(
            predicted,
            target,
            aa_mass_tolerance,
            prefix_mass_tolerance,
        )
        aa_pred_total += len(predicted)
        aa_gold_total += len(target)

        if peptide_matches_mass_based(
            predicted,
            target,
            aa_mass_tolerance,
            prefix_mass_tolerance,
        ):
            peptide_matches += 1
        if predicted == target:
            exact_matches += 1
        if predicted_lengths is not None and target_lengths is not None:
            if predicted_lengths[index] == target_lengths[index]:
                length_correct += 1

    aa_precision = _safe_div(aa_matches, aa_pred_total)
    aa_recall = _safe_div(aa_matches, aa_gold_total)
    aa_f1 = _safe_div(2 * aa_precision * aa_recall, aa_precision + aa_recall)
    aa_error_rate = 1.0 - aa_recall

    peptide_precision = _safe_div(peptide_matches, num_samples)
    peptide_recall = _safe_div(peptide_matches, num_samples)
    peptide_f1 = _safe_div(2 * peptide_precision * peptide_recall, peptide_precision + peptide_recall)

    length_accuracy = _safe_div(length_correct, num_samples) if predicted_lengths else 0.0

    return DenovoMetrics(
        num_samples=num_samples,
        aa_precision=aa_precision,
        aa_recall=aa_recall,
        aa_f1=aa_f1,
        aa_error_rate=aa_error_rate,
        peptide_precision=peptide_precision,
        peptide_recall=peptide_recall,
        peptide_f1=peptide_f1,
        exact_peptide_accuracy=_safe_div(exact_matches, num_samples),
        length_accuracy=length_accuracy,
    )


def format_metrics(metrics: DenovoMetrics) -> str:
    values = metrics.to_dict()
    return (
        f"n={values['num_samples']} "
        f"aa_P={values['aa_precision']:.4f} aa_R={values['aa_recall']:.4f} "
        f"pep_P={values['peptide_precision']:.4f} pep_R={values['peptide_recall']:.4f} "
        f"exact={values['exact_peptide_accuracy']:.4f} len_acc={values['length_accuracy']:.4f}"
    )
