"""De novo sequencing evaluation metrics (InstaNovo & Proteomics-compatible definitions).

Provides exact sequence match, mass-based match (with precursor & prefix mass tolerance),
confidence-score thresholded Precision/Recall/F1, and Area Under the Precision-Coverage Curve (AUC / PAUC).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data.constants import AA_MASSES_DICT


def _residue_mass(amino_acid: str) -> float:
    return AA_MASSES_DICT.get(amino_acid, 0.0)


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
    if predicted == target:
        return True
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


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Computes trapezoidal integral compatible across NumPy 1.x and 2.x."""
    if len(x) < 2:
        return 0.0
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        return float(trapz_fn(y, x))
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def compute_precision_coverage_curve(
    is_correct: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Computes Precision-Coverage Curve, Full AUC, and Partial AUCs.

    Args:
        is_correct: 1D boolean array indicating correctness (length N)
        scores: 1D float array of confidence scores (length N)

    Returns:
        coverages: array of coverage values in [0, 1]
        precisions: array of precision values (C_above / N_above)
        thresholds: sorted score cutoffs
        auc: Full Area Under Precision-Coverage Curve
        pauc80: Partial Area where Precision >= 0.80
        pauc50: Partial Area where Precision >= 0.50
    """
    if len(scores) == 0:
        return (
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            0.0,
            0.0,
            0.0,
        )

    # Sort descending by score
    order = np.argsort(-scores)
    sorted_correct = is_correct[order]
    thresholds = scores[order]

    cum_correct = np.cumsum(sorted_correct)
    cum_preds = np.arange(1, len(scores) + 1)

    precisions = cum_correct / cum_preds
    coverages = cum_preds / len(scores)

    # Ensure curve starts at coverage 0 with precision at top cutoff
    cov_points = np.concatenate(([0.0], coverages))
    prec_points = np.concatenate(([precisions[0]], precisions))

    auc = _integrate_trapz(prec_points, cov_points)

    # Partial AUC @ Precision >= 0.80
    mask80 = prec_points >= 0.80
    if np.any(mask80):
        cov_80 = cov_points[mask80]
        prec_80 = prec_points[mask80]
        pauc80 = _integrate_trapz(prec_80, cov_80)
    else:
        pauc80 = 0.0

    # Partial AUC @ Precision >= 0.50
    mask50 = prec_points >= 0.50
    if np.any(mask50):
        cov_50 = cov_points[mask50]
        prec_50 = prec_points[mask50]
        pauc50 = _integrate_trapz(prec_50, cov_50)
    return coverages, precisions, thresholds, auc, pauc80, pauc50


def compute_precision_recall_curve(
    is_correct: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Computes Peptide-level Precision-Recall Curve (Precision vs. Peptide Recall),
    Area Under PR Curve (PR-AUC), and Partial PR-AUCs.

    Args:
        is_correct: 1D boolean array indicating correctness (length N)
        scores: 1D float array of confidence scores (length N)

    Returns:
        recalls: array of peptide recall values (cum_correct / N_total)
        precisions: array of precision values (cum_correct / cum_preds)
        thresholds: sorted score cutoffs
        pr_auc: Full Area Under Precision-Recall Curve
        p_pr_auc80: Partial Area where Precision >= 0.80
        p_pr_auc50: Partial Area where Precision >= 0.50
    """
    if len(scores) == 0:
        return (
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            0.0,
            0.0,
            0.0,
        )

    # Sort descending by score
    order = np.argsort(-scores)
    sorted_correct = is_correct[order]
    thresholds = scores[order]

    cum_correct = np.cumsum(sorted_correct)
    cum_preds = np.arange(1, len(scores) + 1)

    precisions = cum_correct / cum_preds
    recalls = cum_correct / len(scores)

    # In PR curves, the curve starts at recall 0 with precision at top cutoff
    rec_points = np.concatenate(([0.0], recalls))
    prec_points = np.concatenate(([precisions[0]], precisions))

    pr_auc = _integrate_trapz(prec_points, rec_points)

    # Partial PR-AUC @ Precision >= 0.80
    mask80 = prec_points >= 0.80
    if np.any(mask80):
        rec_80 = rec_points[mask80]
        prec_80 = prec_points[mask80]
        p_pr_auc80 = _integrate_trapz(prec_80, rec_80)
    else:
        p_pr_auc80 = 0.0

    # Partial PR-AUC @ Precision >= 0.50
    mask50 = prec_points >= 0.50
    if np.any(mask50):
        rec_50 = rec_points[mask50]
        prec_50 = prec_points[mask50]
        p_pr_auc50 = _integrate_trapz(prec_50, rec_50)
    else:
        p_pr_auc50 = 0.0

    return recalls, precisions, thresholds, pr_auc, p_pr_auc80, p_pr_auc50


def calibrate_score_threshold(
    scores: np.ndarray | list[float],
    is_correct: np.ndarray | list[bool],
    target_precision: float = 0.80,
    strategy: str = "target_precision",
    min_samples: int = 10,
) -> dict[str, float]:
    """
    Calibrates confidence score threshold on a validation subset.

    Args:
        scores: array of confidence scores
        is_correct: array of correctness booleans
        target_precision: target precision level (e.g. 0.80 or 0.90)
        strategy: 'target_precision' (highest coverage achieving target precision)
                  or 'max_f1' (threshold maximizing F1 score)
        min_samples: minimum number of accepted predictions required

    Returns:
        dict with keys: threshold, precision, recall, f1, coverage, num_accepted
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    correct_arr = np.asarray(is_correct, dtype=bool)
    n_total = len(scores_arr)

    if n_total == 0:
        return {
            "threshold": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
            "num_accepted": 0,
        }

    order = np.argsort(-scores_arr)
    sorted_correct = correct_arr[order]
    sorted_scores = scores_arr[order]

    cum_correct = np.cumsum(sorted_correct)
    cum_preds = np.arange(1, n_total + 1)

    precisions = cum_correct / cum_preds
    recalls = cum_correct / n_total
    f1s = np.zeros_like(precisions)
    denom = precisions + recalls
    valid_f1 = denom > 0
    f1s[valid_f1] = 2.0 * precisions[valid_f1] * recalls[valid_f1] / denom[valid_f1]

    if strategy == "target_precision":
        # Find lowest threshold (highest coverage) where precision >= target_precision
        valid_indices = np.where((precisions >= target_precision) & (cum_preds >= min_samples))[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[-1]  # largest coverage satisfying target precision
        else:
            # Fall back to max F1 if target precision cannot be met
            best_idx = int(np.argmax(f1s))
    else:  # max_f1
        best_idx = int(np.argmax(f1s))

    return {
        "threshold": float(sorted_scores[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
        "coverage": float(cum_preds[best_idx] / n_total),
        "num_accepted": int(cum_preds[best_idx]),
    }


@dataclass
class DenovoMetrics:
    num_samples: int
    # Amino acid level
    aa_precision: float
    aa_recall: float
    aa_f1: float
    aa_error_rate: float
    length_accuracy: float

    # Exact Match Level (Strict String Equality)
    exact_peptide_accuracy: float
    peptide_precision_exact: float
    peptide_recall_exact: float
    peptide_f1_exact: float

    # Mass-Based Match Level (Residue + Prefix Mass Alignment)
    mass_peptide_accuracy: float
    peptide_precision_mass: float
    peptide_recall_mass: float
    peptide_f1_mass: float

    # Thresholding & Coverage
    score_threshold: float | None = None
    num_predicted_above_threshold: int | None = None
    coverage: float = 1.0

    # Area Under Precision-Coverage Curve (AUPCC)
    auc_exact: float = 0.0
    pauc80_exact: float = 0.0
    pauc50_exact: float = 0.0
    auc_mass: float = 0.0
    pauc80_mass: float = 0.0
    pauc50_mass: float = 0.0

    # Area Under Precision-Recall Curve (PR-AUC)
    pr_auc_exact: float = 0.0
    p_pr_auc80_exact: float = 0.0
    pr_auc_mass: float = 0.0
    p_pr_auc80_mass: float = 0.0

    # Backward compatibility properties
    @property
    def peptide_precision(self) -> float:
        return self.peptide_precision_mass

    @property
    def peptide_recall(self) -> float:
        return self.peptide_recall_mass

    @property
    def peptide_f1(self) -> float:
        return self.peptide_f1_mass

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "num_samples": self.num_samples,
            "score_threshold": self.score_threshold,
            "num_predicted_above_threshold": self.num_predicted_above_threshold,
            "coverage": self.coverage,
            # Amino acid metrics
            "aa_precision": self.aa_precision,
            "aa_recall": self.aa_recall,
            "aa_f1": self.aa_f1,
            "aa_error_rate": self.aa_error_rate,
            "length_accuracy": self.length_accuracy,
            # Exact sequence metrics
            "exact_peptide_accuracy": self.exact_peptide_accuracy,
            "peptide_precision_exact": self.peptide_precision_exact,
            "peptide_recall_exact": self.peptide_recall_exact,
            "peptide_f1_exact": self.peptide_f1_exact,
            "auc_exact": self.auc_exact,
            "pauc80_exact": self.pauc80_exact,
            "pauc50_exact": self.pauc50_exact,
            # Mass-based metrics
            "mass_peptide_accuracy": self.mass_peptide_accuracy,
            "peptide_precision_mass": self.peptide_precision_mass,
            "peptide_recall_mass": self.peptide_recall_mass,
            "peptide_f1_mass": self.peptide_f1_mass,
            # Precision-Coverage (AUPCC)
            "auc_exact": self.auc_exact,
            "pauc80_exact": self.pauc80_exact,
            "pauc50_exact": self.pauc50_exact,
            "auc_mass": self.auc_mass,
            "pauc80_mass": self.pauc80_mass,
            "pauc50_mass": self.pauc50_mass,
            # Precision-Recall (PR-AUC)
            "pr_auc_exact": self.pr_auc_exact,
            "p_pr_auc80_exact": self.p_pr_auc80_exact,
            "pr_auc_mass": self.pr_auc_mass,
            "p_pr_auc80_mass": self.p_pr_auc80_mass,
            # Legacy / compatibility keys
            "peptide_precision": self.peptide_precision_mass,
            "peptide_recall": self.peptide_recall_mass,
            "peptide_f1": self.peptide_f1_mass,
            # User formula explicitly mapped:
            "user_peptide_recall_exact": self.peptide_precision_exact,
            "user_peptide_precision_exact": self.exact_peptide_accuracy,
            "user_peptide_recall_mass": self.peptide_precision_mass,
            "user_peptide_precision_mass": self.mass_peptide_accuracy,
        }


def compute_denovo_metrics(
    predictions: list[str],
    targets: list[str],
    predicted_lengths: list[int] | None = None,
    target_lengths: list[int] | None = None,
    scores: list[float] | np.ndarray | None = None,
    score_threshold: float | None = None,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
) -> DenovoMetrics:
    """
    Corpus-level metrics following InstaNovo & Proteomics benchmarks.

    Distinguishes:
    1. Exact Match: pred_seq == target_seq
    2. Mass-Based Match: length matches and all amino acids align within mass tolerance
    3. Score Thresholding: filters predictions where score >= score_threshold
    4. Precision-Coverage Curves: Full AUC and PAUC (Precision >= 0.80)
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    num_samples = len(predictions)
    if num_samples == 0:
        return DenovoMetrics(
            num_samples=0,
            aa_precision=0.0,
            aa_recall=0.0,
            aa_f1=0.0,
            aa_error_rate=1.0,
            length_accuracy=0.0,
            exact_peptide_accuracy=0.0,
            peptide_precision_exact=0.0,
            peptide_recall_exact=0.0,
            peptide_f1_exact=0.0,
            mass_peptide_accuracy=0.0,
            peptide_precision_mass=0.0,
            peptide_recall_mass=0.0,
            peptide_f1_mass=0.0,
        )

    aa_matches = 0
    aa_pred_total = 0
    aa_gold_total = 0
    length_correct = 0

    is_exact_list: list[bool] = []
    is_mass_list: list[bool] = []

    for index, (predicted, target) in enumerate(zip(predictions, targets)):
        aa_matches += count_matching_amino_acids(
            predicted,
            target,
            aa_mass_tolerance,
            prefix_mass_tolerance,
        )
        aa_pred_total += len(predicted)
        aa_gold_total += len(target)

        exact_match = (predicted == target)
        mass_match = peptide_matches_mass_based(
            predicted,
            target,
            aa_mass_tolerance,
            prefix_mass_tolerance,
        )

        is_exact_list.append(exact_match)
        is_mass_list.append(mass_match)

        if predicted_lengths is not None and target_lengths is not None:
            if predicted_lengths[index] == target_lengths[index]:
                length_correct += 1

    is_exact = np.array(is_exact_list, dtype=bool)
    is_mass = np.array(is_mass_list, dtype=bool)

    aa_precision = _safe_div(aa_matches, aa_pred_total)
    aa_recall = _safe_div(aa_matches, aa_gold_total)
    aa_f1 = _safe_div(2 * aa_precision * aa_recall, aa_precision + aa_recall)
    aa_error_rate = 1.0 - aa_recall
    length_accuracy = _safe_div(length_correct, num_samples) if predicted_lengths else 0.0

    # Unthresholded accuracies
    exact_acc = float(np.mean(is_exact))
    mass_acc = float(np.mean(is_mass))

    # Curve analysis if scores are available
    auc_exact = 0.0
    pauc80_exact = 0.0
    pauc50_exact = 0.0
    auc_mass = 0.0
    pauc80_mass = 0.0
    pauc50_mass = 0.0

    pr_auc_exact = 0.0
    p_pr_auc80_exact = 0.0
    pr_auc_mass = 0.0
    p_pr_auc80_mass = 0.0

    if scores is not None:
        scores_arr = np.asarray(scores, dtype=np.float64)
        # Precision-Coverage Curves
        _, _, _, auc_exact, pauc80_exact, pauc50_exact = compute_precision_coverage_curve(
            is_exact, scores_arr
        )
        _, _, _, auc_mass, pauc80_mass, pauc50_mass = compute_precision_coverage_curve(
            is_mass, scores_arr
        )
        # Precision-Recall Curves (Peptide Recall on x-axis, Precision on y-axis)
        _, _, _, pr_auc_exact, p_pr_auc80_exact, _ = compute_precision_recall_curve(
            is_exact, scores_arr
        )
        _, _, _, pr_auc_mass, p_pr_auc80_mass, _ = compute_precision_recall_curve(
            is_mass, scores_arr
        )

    # Thresholded evaluation
    if scores is not None and score_threshold is not None:
        scores_arr = np.asarray(scores, dtype=np.float64)
        mask = scores_arr >= score_threshold
        num_above = int(np.sum(mask))
        coverage = _safe_div(num_above, num_samples)

        exact_above = int(np.sum(is_exact[mask]))
        mass_above = int(np.sum(is_mass[mask]))

        # Precision = correct above / predictions above
        pep_prec_exact = _safe_div(exact_above, num_above)
        pep_prec_mass = _safe_div(mass_above, num_above)

        # Recall = correct above / total ground truth targets
        pep_rec_exact = _safe_div(exact_above, num_samples)
        pep_rec_mass = _safe_div(mass_above, num_samples)

        pep_f1_exact = _safe_div(2 * pep_prec_exact * pep_rec_exact, pep_prec_exact + pep_rec_exact)
        pep_f1_mass = _safe_div(2 * pep_prec_mass * pep_rec_mass, pep_prec_mass + pep_rec_mass)
    else:
        # Default unthresholded (coverage = 100%)
        num_above = num_samples
        coverage = 1.0
        pep_prec_exact = exact_acc
        pep_rec_exact = exact_acc
        pep_f1_exact = exact_acc

        pep_prec_mass = mass_acc
        pep_rec_mass = mass_acc
        pep_f1_mass = mass_acc

    return DenovoMetrics(
        num_samples=num_samples,
        aa_precision=aa_precision,
        aa_recall=aa_recall,
        aa_f1=aa_f1,
        aa_error_rate=aa_error_rate,
        length_accuracy=length_accuracy,
        exact_peptide_accuracy=exact_acc,
        peptide_precision_exact=pep_prec_exact,
        peptide_recall_exact=pep_rec_exact,
        peptide_f1_exact=pep_f1_exact,
        mass_peptide_accuracy=mass_acc,
        peptide_precision_mass=pep_prec_mass,
        peptide_recall_mass=pep_rec_mass,
        peptide_f1_mass=pep_f1_mass,
        score_threshold=score_threshold,
        num_predicted_above_threshold=num_above,
        coverage=coverage,
        auc_exact=auc_exact,
        pauc80_exact=pauc80_exact,
        pauc50_exact=pauc50_exact,
        auc_mass=auc_mass,
        pauc80_mass=pauc80_mass,
        pauc50_mass=pauc50_mass,
        pr_auc_exact=pr_auc_exact,
        p_pr_auc80_exact=p_pr_auc80_exact,
        pr_auc_mass=pr_auc_mass,
        p_pr_auc80_mass=p_pr_auc80_mass,
    )


def format_metrics(metrics: DenovoMetrics) -> str:
    vals = metrics.to_dict()
    thresh_str = f"thresh={vals['score_threshold']:.3f} cov={vals['coverage']:.3f} " if vals['score_threshold'] is not None else ""
    pauc_str = f"auc_exact={vals['auc_exact']:.4f} pauc80_exact={vals['pauc80_exact']:.4f} " if vals['auc_exact'] > 0 else ""
    return (
        f"n={vals['num_samples']} {thresh_str}"
        f"Exact[acc={vals['exact_peptide_accuracy']:.4f} P={vals['peptide_precision_exact']:.4f} R={vals['peptide_recall_exact']:.4f}] "
        f"Mass[acc={vals['mass_peptide_accuracy']:.4f} P={vals['peptide_precision_mass']:.4f} R={vals['peptide_recall_mass']:.4f}] "
        f"AA[P={vals['aa_precision']:.4f} R={vals['aa_recall']:.4f} F1={vals['aa_f1']:.4f}] "
        f"{pauc_str}"
        f"len_acc={vals['length_accuracy']:.4f}"
    )
