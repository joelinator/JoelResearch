"""
Exact Dynamic Programming Reachable-Mass Table for Mass-Constrained De Novo Peptide Sequencing.

Replaces heuristic interval bounds with an exact reachable-mass dynamic programming
grid. For any remaining sequence length k in [0, max_length] and any candidate residue a,
it determines in O(1) whether the remaining mass budget (R - m_a) can be exactly
partitioned into k - 1 valid amino acid residues within instrument tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from data.constants import AA_MASSES_DICT, STANDARD_AMINO_ACIDS
from data.lengths import MAX_PEPTIDE_LENGTH


class ExactReachabilityDP:
    """
    Exact reachable-mass table computed via dynamic programming.

    State:
        dp[k, b] = True iff exactly k residues can sum to mass bin b.

    With tolerance pooling:
        reachable[k, b] = True iff |sum_{i=1}^k m(a_i) - mass(b)| <= tol_da.
    """

    _cached_instance: ExactReachabilityDP | None = None

    def __init__(
        self,
        aa_masses: list[float] | torch.Tensor | None = None,
        max_len: int = MAX_PEPTIDE_LENGTH,
        max_mass: float = 5000.0,
        bin_size: float = 0.02,
        tol_da: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        self.max_len = max_len
        self.max_mass = max_mass
        self.bin_size = bin_size
        self.tol_da = tol_da
        self.device = torch.device(device)
        self.num_bins = int(round(max_mass / bin_size)) + 1

        if aa_masses is None:
            # Standard 20 amino acids + primary biological PTMs
            raw_masses = [
                AA_MASSES_DICT[aa]
                for aa in STANDARD_AMINO_ACIDS
                if aa in AA_MASSES_DICT and AA_MASSES_DICT[aa] > 0
            ]
            # Add target PTMs
            ptm_keys = ["M(ox)", "C(cam)", "N(deam)", "Q(deam)", "S(ph)", "T(ph)", "Y(ph)"]
            for k in ptm_keys:
                if k in AA_MASSES_DICT:
                    raw_masses.append(AA_MASSES_DICT[k])
            masses = sorted(list(set(raw_masses)))
        elif isinstance(aa_masses, torch.Tensor):
            masses = [float(m) for m in aa_masses.view(-1).tolist() if float(m) > 10.0]
        else:
            masses = [float(m) for m in aa_masses if float(m) > 10.0]

        self.masses = masses
        self.dp_table = self._build_dp_table()
        self.pooled_table = self._build_pooled_table(tol_da)

    def _build_dp_table(self) -> torch.Tensor:
        """Construct reachable mass table for all k in [0, max_len]."""
        table = torch.zeros((self.max_len + 1, self.num_bins), dtype=torch.bool, device=self.device)
        table[0, 0] = True

        mass_bins = sorted(
            list(
                set(
                    int(round(m / self.bin_size))
                    for m in self.masses
                    if 0 < int(round(m / self.bin_size)) < self.num_bins
                )
            )
        )

        for k in range(1, self.max_len + 1):
            prev = table[k - 1]
            curr = torch.zeros(self.num_bins, dtype=torch.bool, device=self.device)
            for mb in mass_bins:
                curr[mb:] |= prev[:-mb]
            table[k] = curr

        return table

    def _build_pooled_table(self, tol_da: float) -> torch.Tensor:
        """Apply max-pooling across tolerance window for O(1) reachability lookup."""
        w = max(1, int(round(tol_da / self.bin_size)))
        kernel_size = 2 * w + 1
        # 1D max pooling over bin dimension
        float_dp = self.dp_table.unsqueeze(0).float()  # (1, K+1, num_bins)
        pooled = F.max_pool1d(float_dp, kernel_size=kernel_size, stride=1, padding=w).squeeze(0)
        return pooled > 0.5

    def to(self, device: torch.device | str) -> ExactReachabilityDP:
        dev = torch.device(device)
        if self.device != dev:
            self.device = dev
            self.dp_table = self.dp_table.to(dev)
            self.pooled_table = self.pooled_table.to(dev)
        return self

    def is_reachable(self, k: int | torch.Tensor, remaining_mass: float | torch.Tensor) -> torch.Tensor:
        """
        Query reachability: can k residues sum to remaining_mass within tol_da?

        Args:
            k: Tensor of remaining lengths (integer >= 0)
            remaining_mass: Tensor of target remaining masses in Daltons
        """
        if not isinstance(remaining_mass, torch.Tensor):
            remaining_mass = torch.tensor(remaining_mass, device=self.device, dtype=torch.float32)
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=self.device, dtype=torch.long)

        k = k.to(device=self.device, dtype=torch.long)
        remaining_mass = remaining_mass.to(device=self.device, dtype=torch.float32)

        # k == 0 case: remaining mass must be ~0
        k0_valid = (k == 0) & (remaining_mass.abs() <= self.tol_da)

        # k in [1, max_len] case:
        in_range = (k >= 1) & (k <= self.max_len) & (remaining_mass >= -self.tol_da) & (remaining_mass <= self.max_mass)
        clamped_mass = remaining_mass.clamp(0.0, self.max_mass)
        bins = (clamped_mass / self.bin_size).round().long().clamp(0, self.num_bins - 1)
        clamped_k = k.clamp(0, self.max_len)

        dp_valid = self.pooled_table[clamped_k, bins] & in_range

        return torch.where(k == 0, k0_valid, dp_valid)

    @classmethod
    def get_default(cls, device: torch.device | str = "cpu", tol_da: float = 0.5) -> ExactReachabilityDP:
        """Get or initialize default singleton instance."""
        dev = torch.device(device)
        if cls._cached_instance is None:
            cls._cached_instance = cls(device=dev, tol_da=tol_da)
        elif cls._cached_instance.device != dev or abs(cls._cached_instance.tol_da - tol_da) > 1e-4:
            cls._cached_instance = cls(device=dev, tol_da=tol_da)
        return cls._cached_instance
