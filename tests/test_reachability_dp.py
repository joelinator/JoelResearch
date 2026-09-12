"""
Unit tests for reachability and evaluation utilities:
- Exact Reachability Dynamic Programming (ExactReachabilityDP)
- Multi-feature Fragment Ladder Scoring (compute_fragment_matching_scores)
- Evidence-Conditioned Enzymatic Cleavage Prior (compute_terminal_prior)
- McNemar's Paired Significance Test (mcnemar_paired_test)
- Stratified Performance Breakdown (stratified_performance_breakdown)
- VRAM Optimization Auto-Tuning
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from data.data import build_vocabulary
from eval.metrics import compute_denovo_metrics, mcnemar_paired_test, stratified_performance_breakdown
from inference.knapsack_dp import ExactReachabilityDP
from inference.predict import (
    compute_fragment_matching_scores,
    compute_terminal_prior,
    knapsack_filter_logits_vectorized,
)
from scripts.train_joint_balanced import get_recommended_batch_size


class TestExactReachabilityDP:
    def test_dp_table_initialization(self):
        dp = ExactReachabilityDP(max_len=15, max_mass=2000.0, bin_size=0.05, tol_da=0.5)
        assert dp.dp_table.shape[0] == 16
        assert dp.dp_table.ndim == 2
        assert dp.dp_table.dtype == torch.bool

    def test_singleton_cache(self):
        dp1 = ExactReachabilityDP.get_default(device="cpu", tol_da=1.0)
        dp2 = ExactReachabilityDP.get_default(device="cpu", tol_da=1.0)
        assert dp1 is dp2

    def test_reachability_queries(self):
        dp = ExactReachabilityDP.get_default(device="cpu", tol_da=1.0)
        # 0 residues must have ~0 remaining mass
        zero_k = torch.tensor([0])
        zero_mass = torch.tensor([0.0])
        bad_mass = torch.tensor([150.0])
        assert dp.is_reachable(zero_k, zero_mass).item() is True
        assert dp.is_reachable(zero_k, bad_mass).item() is False

        # 1 residue of Alanine mass (71.037 Da)
        one_k = torch.tensor([1])
        ala_mass = torch.tensor([71.037])
        assert dp.is_reachable(one_k, ala_mass).item() is True

        # Impossible single residue mass (e.g. 25.0 Da)
        impossible_mass = torch.tensor([25.0])
        assert dp.is_reachable(one_k, impossible_mass).item() is False

    def test_knapsack_filtering_with_exact_dp(self):
        device = torch.device("cpu")
        vocab = build_vocabulary(include_ptms=True)
        V = len(vocab)
        L = 6
        B = 2

        logits = torch.zeros(B, L, V, device=device)
        x_t = torch.tensor([[vocab["<mask_token>"]] * L, [vocab["<mask_token>"]] * L], device=device)
        active_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        mask_id = vocab["<mask_token>"]

        mass_table = torch.zeros(V, device=device)
        for tok, idx in vocab.items():
            if tok in {"A", "G", "K", "L", "E", "S"}:
                mass_table[idx] = 100.0

        target_residue_mass = torch.tensor([600.0, 600.0], device=device)

        filtered_logits = knapsack_filter_logits_vectorized(
            logits=logits,
            x_t=x_t,
            target_residue_mass=target_residue_mass,
            active_mask=active_mask,
            mask_id=mask_id,
            mass_table=mass_table,
            full_mass_table=mass_table,
            tol=1.0,
            use_exact_dp=True,
        )
        assert filtered_logits.shape == (B, L, V)
        assert not torch.isnan(filtered_logits).any()


class TestMultiFeatureFragmentScoring:
    def test_composite_ladders(self):
        device = torch.device("cpu")
        vocab = build_vocabulary(include_ptms=True)
        B = 2
        L = 8
        S = 50

        x_t = torch.tensor([
            [vocab["A"], vocab["C"], vocab["D"], vocab["E"], vocab["F"], vocab["G"], vocab["H"], vocab["K"]],
            [vocab["K"], vocab["H"], vocab["G"], vocab["F"], vocab["E"], vocab["D"], vocab["C"], vocab["A"]],
        ], device=device)
        active_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        mz_array = torch.linspace(100.0, 1000.0, S, device=device).unsqueeze(0).expand(B, S)
        intensity_array = torch.ones(B, S, device=device)
        spectrum_mask = torch.zeros(B, S, dtype=torch.bool, device=device)

        # Basic explained intensity
        simple_score = compute_fragment_matching_scores(
            x_t=x_t,
            active_mask=active_mask,
            mz_array_exp=mz_array,
            intensity_array_exp=intensity_array,
            spectrum_mask_exp=spectrum_mask,
            vocabulary=vocab,
            use_composite_ladders=False,
        )
        assert simple_score.shape == (B,)
        assert (simple_score >= 0.0).all() and (simple_score <= 1.0).all()

        # Composite multi-feature ladders
        composite_score = compute_fragment_matching_scores(
            x_t=x_t,
            active_mask=active_mask,
            mz_array_exp=mz_array,
            intensity_array_exp=intensity_array,
            spectrum_mask_exp=spectrum_mask,
            vocabulary=vocab,
            use_composite_ladders=True,
        )
        assert composite_score.shape == (B,)
        assert (composite_score >= 0.0).all() and (composite_score <= 1.0).all()


class TestTerminalPrior:
    def test_trypsin_prior_with_y1_evidence(self):
        device = torch.device("cpu")
        vocab = build_vocabulary(include_ptms=True)
        B = 2
        L = 5
        # Candidate 0 ends in K; Candidate 1 ends in A
        x_t = torch.tensor([
            [vocab["A"], vocab["A"], vocab["A"], vocab["A"], vocab["K"]],
            [vocab["A"], vocab["A"], vocab["A"], vocab["A"], vocab["A"]],
        ], device=device)
        cand_lengths = torch.tensor([5, 5], device=device)

        # Spectrum with y1(K) peak at 147.11 Da
        mz_array = torch.tensor([
            [147.1128, 300.0, 500.0],
            [147.1128, 300.0, 500.0],
        ], device=device)
        spectrum_mask = torch.zeros_like(mz_array, dtype=torch.bool)

        priors = compute_terminal_prior(
            x_t=x_t,
            cand_lengths=cand_lengths,
            vocabulary=vocab,
            enzyme="trypsin",
            mz_array_exp=mz_array,
            spectrum_mask_exp=spectrum_mask,
        )
        assert priors.shape == (B,)
        # Candidate 0 has physical evidence bonus (0.25)
        assert priors[0].item() == pytest.approx(0.25, abs=1e-3)
        # Candidate 1 is non-tryptic (0.0)
        assert priors[1].item() == pytest.approx(0.0, abs=1e-3)

    def test_non_tryptic_enzyme(self):
        device = torch.device("cpu")
        vocab = build_vocabulary(include_ptms=True)
        x_t = torch.tensor([[vocab["A"], vocab["K"]]], device=device)
        cand_lengths = torch.tensor([2], device=device)
        priors = compute_terminal_prior(x_t, cand_lengths, vocab, enzyme="none")
        assert priors[0].item() == 0.0


class TestEvaluationEnhancements:
    def test_coverage_at_precision(self):
        preds = ["PEPTIDEK", "ALANINEK", "GLYCINEK", "VALINEK", "SERINEK"]
        targets = ["PEPTIDEK", "ALANINEK", "GLYCINEK", "WRONGK", "WRONGK"]
        scores = [0.95, 0.90, 0.85, 0.40, 0.20]

        metrics = compute_denovo_metrics(preds, targets, scores=scores)
        assert metrics.coverage_at_80 >= 0.0
        assert metrics.average_precision >= 0.0

    def test_mcnemar_paired_test(self):
        # Model A correct on 80, Model B correct on 60
        # Discordant: A right B wrong = 25; A wrong B right = 5
        correct_a = [True] * 25 + [False] * 5 + [True] * 55 + [False] * 15
        correct_b = [False] * 25 + [True] * 5 + [True] * 55 + [False] * 15

        res = mcnemar_paired_test(correct_a, correct_b)
        assert res["discordant_a_not_b"] == 25
        assert res["discordant_b_not_a"] == 5
        assert res["chi2_stat"] > 0
        assert res["p_value"] < 0.01
        assert res["significant_at_01"] is True

    def test_stratified_performance_breakdown(self):
        preds = ["PEPTIDE", "PEPTIDEALANINE", "PEPTIDEALANINEGLYCINEVALINE"]
        targets = ["PEPTIDE", "PEPTIDEALANINE", "PEPTIDEALANINEGLYCINEVALINE"]
        lengths = [7, 14, 27]
        charges = [2, 3, 4]

        res = stratified_performance_breakdown(preds, targets, lengths, charges)
        assert "length_short_<=10" in res
        assert "length_medium_11-16" in res
        assert "length_long_>16" in res
        assert "charge_2" in res
        assert "charge_3" in res
        assert "charge_4+" in res
        assert res["length_short_<=10"]["exact_acc"] == 1.0


class TestVRAMAutoTuning:
    def test_recommended_batch_size(self):
        # Verify recommended batch sizes return expected positive values
        bs = get_recommended_batch_size(target_vram_pct=0.85)
        assert bs in (1792, 896, 448, 128, 64)
