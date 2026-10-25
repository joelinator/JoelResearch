# candidate reranker for de novo peptide sequencing.
# scores candidate beams using physical MS/MS features:
# - precursor mass precision (ppm error, knapsack bounds)
# - theoretical b/y ion series continuity (ladder score)
# - explained spectral intensity & coverage
# - length prior from spectrum encoder
#
# NOTE: we score based on isobaric I/L peptide accuracy (standard in proteomics).
# TODO: consider training an MLP reranker once we collect more candidate triplets.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_NAMES = [
    "cand_length_log_prob",
    "mean_log_prob",
    "total_log_prob",
    "frag_score",
    "explained_intensity",
    "ladder_score",
    "theo_coverage",
    "ion_balance",
    "unmatched_penalty",
    "term_bonus",
    "ppm_error",
    "mass_gate_20ppm",
    "mass_gate_10ppm",
    "mass_gaussian",
    "len_diff_expected",
    "cand_length",
]


def extract_candidate_features(
    candidate_records: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract structured feature matrix and ground truth match matrix.

    Returns:
        features: Array of shape (N, M, D)
        match_il: Boolean array of shape (N, M) indicating peptide match (I/L equivalence)
    """
    N = len(candidate_records)
    M = len(candidate_records[0]["candidates"])
    D = len(FEATURE_NAMES)

    features = np.zeros((N, M, D), dtype=np.float32)
    match_il = np.zeros((N, M), dtype=bool)

    for i, r in enumerate(candidate_records):
        tgt = r["target"]
        tgt_il = tgt.replace("I", "L")
        p_m = float(r.get("precursor_mass", 1000.0))
        # rough average amino acid mass is ~111.1 Da; water is ~18.015 Da
        expected_len = round((p_m - 18.015) / 111.1)

        c_seqs = r["candidates"]
        c_lens = r["cand_lengths"]
        c_len_lp = r.get("cand_length_log_prob", [0.0] * M)
        c_mean_lp = r.get("mean_log_prob", [0.0] * M)
        c_frag = r.get("frag_score", [0.0] * M)
        c_expl = r.get("explained_intensity", [0.0] * M)
        c_ladd = r.get("ladder_score", [0.0] * M)
        c_cov = r.get("theo_coverage", [0.0] * M)
        c_bal = r.get("ion_balance", [0.0] * M)
        c_unm = r.get("unmatched_penalty", [0.0] * M)
        c_term = r.get("term_bonus", [0.0] * M)
        c_ppm = r.get("ppm_error", [0.0] * M)

        for j in range(min(M, len(c_seqs))):
            seq = c_seqs[j]
            seq_il = seq.replace("I", "L")
            match_il[i, j] = (seq_il == tgt_il)

            l_j = float(c_lens[j])
            mean_lp = float(c_mean_lp[j])
            total_lp = mean_lp * l_j
            ppm = float(c_ppm[j])

            features[i, j, 0] = float(c_len_lp[j])
            features[i, j, 1] = mean_lp
            features[i, j, 2] = total_lp
            features[i, j, 3] = float(c_frag[j])
            features[i, j, 4] = float(c_expl[j])
            features[i, j, 5] = float(c_ladd[j])
            features[i, j, 6] = float(c_cov[j])
            features[i, j, 7] = float(c_bal[j])
            features[i, j, 8] = float(c_unm[j])
            features[i, j, 9] = float(c_term[j])
            features[i, j, 10] = ppm
            features[i, j, 11] = 1.0 if ppm <= 20.0 else 0.0
            features[i, j, 12] = 1.0 if ppm <= 10.0 else 0.0
            features[i, j, 13] = float(np.exp(-0.5 * (ppm / 10.0) ** 2))
            features[i, j, 14] = abs(l_j - expected_len)
            features[i, j, 15] = l_j

    return features, match_il


class CandidateReranker(nn.Module):
    """
    Lightweight, physics-informed candidate reranker for de novo sequencing.
    Can operate in linear mode (fully interpretable weights) or non-linear MLP mode.
    """
    """
    Lightweight, physics-informed candidate reranker for de novo sequencing.
    Can operate in linear mode (fully interpretable weights) or non-linear MLP mode.
    """

    def __init__(self, mode: str = "linear", hidden_dim: int = 32):
        super().__init__()
        self.mode = mode
        in_dim = len(FEATURE_NAMES)
        if mode == "linear":
            self.net = nn.Linear(in_dim, 1, bias=False)
            # Initialize with sensible mass-spectrometry physical priors
            with torch.no_grad():
                w = torch.zeros(1, in_dim)
                w[0, FEATURE_NAMES.index("cand_length_log_prob")] = 0.1
                w[0, FEATURE_NAMES.index("mean_log_prob")] = 1.0
                w[0, FEATURE_NAMES.index("frag_score")] = 4.0
                w[0, FEATURE_NAMES.index("ladder_score")] = 2.0
                w[0, FEATURE_NAMES.index("theo_coverage")] = 1.5
                w[0, FEATURE_NAMES.index("term_bonus")] = 1.0
                w[0, FEATURE_NAMES.index("mass_gate_20ppm")] = 2.0
                w[0, FEATURE_NAMES.index("mass_gaussian")] = 1.0
                w[0, FEATURE_NAMES.index("ppm_error")] = -0.05
                self.net.weight.copy_(w)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) -> (...,)
        """
        return self.net(x).squeeze(-1)

    def score_numpy(self, features: np.ndarray) -> np.ndarray:
        """
        Score a numpy feature matrix of shape (N, M, D) -> (N, M)
        """
        self.eval()
        with torch.no_grad():
            t_feat = torch.from_numpy(features).float()
            scores = self.forward(t_feat).numpy()
        return scores

    def rerank(self, features: np.ndarray) -> np.ndarray:
        """
        Returns the index of the top-ranked candidate for each spectrum: (N,)
        """
        scores = self.score_numpy(features)
        return np.argmax(scores, axis=-1)

    def fit_listnet(
        self,
        features: np.ndarray,
        match_il: np.ndarray,
        val_features: np.ndarray | None = None,
        val_match_il: np.ndarray | None = None,
        epochs: int = 150,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Fit reranker parameters using ListNet (Softmax Cross-Entropy on candidates).
        Only spectra with at least one matching candidate provide ranking gradient.
        """
        self.to(device)
        self.train()

        # Identify informative training spectra (where oracle has at least one positive)
        has_pos = match_il.any(axis=-1)
        if not has_pos.any():
            return {"train_exact_il": 0.0, "val_exact_il": 0.0}

        t_feat = torch.from_numpy(features).float().to(device)  # (N, M, D)
        t_labels = torch.from_numpy(match_il.astype(np.float32)).to(device)  # (N, M)

        # Normalize ground truth labels to probability distribution over matching candidates
        pos_mask = has_pos
        t_feat_train = t_feat[pos_mask]
        t_labels_train = t_labels[pos_mask]
        t_target_probs = t_labels_train / t_labels_train.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_exact = -1.0
        best_state = None
        v_feat = torch.from_numpy(val_features).float().to(device) if val_features is not None else None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            pred_scores = self.forward(t_feat_train)  # (N_train, M)
            log_pred_probs = F.log_softmax(pred_scores, dim=-1)

            # Cross entropy between predicted candidate distribution and target distribution
            loss = -(t_target_probs * log_pred_probs).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()

            # Evaluate training accuracy on full set
            self.eval()
            with torch.no_grad():
                train_preds = self.forward(t_feat).argmax(dim=-1).cpu().numpy()
                train_exact = np.mean(match_il[np.arange(len(match_il)), train_preds])

                val_exact = train_exact
                if v_feat is not None and val_match_il is not None:
                    v_preds = self.forward(v_feat).argmax(dim=-1).cpu().numpy()
                    val_exact = np.mean(val_match_il[np.arange(len(val_match_il)), v_preds])

                if val_exact > best_val_exact:
                    best_val_exact = val_exact
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        self.to("cpu")
        final_train_preds = self.rerank(features)
        final_train_exact = float(np.mean(match_il[np.arange(len(match_il)), final_train_preds]))

        final_val_exact = final_train_exact
        if val_features is not None and val_match_il is not None:
            final_val_preds = self.rerank(val_features)
            final_val_exact = float(np.mean(val_match_il[np.arange(len(val_match_il)), final_val_preds]))

        return {
            "train_exact_il": final_train_exact,
            "val_exact_il": final_val_exact,
        }

    def get_feature_weights(self) -> Dict[str, float]:
        """
        Return dictionary of feature weights if model is linear.
        """
        if self.mode != "linear":
            return {}
        w = self.net.weight.detach().cpu().squeeze().numpy()
        return {name: float(w[i]) for i, name in enumerate(FEATURE_NAMES)}

    def save(self, path: str | Path):
        """Save reranker weights to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "state_dict": self.state_dict(),
                "feature_names": FEATURE_NAMES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> CandidateReranker:
        """Load reranker weights from file."""
        data = torch.load(path, map_location="cpu")
        model = cls(mode=data["mode"])
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model
