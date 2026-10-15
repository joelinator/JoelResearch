"""
Length-Stratified and Length-Balanced Samplers for De Novo Peptide Sequencing.

Enforces that each mini-batch contains a targeted minimum representation of long
peptides (e.g. >= 20% peptides with length >= 19 residues), combating the severe
class imbalance in natural tryptic digest datasets where peptides >= 20 AA represent
less than 6% of spectra.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler


class LengthStratifiedBatchSampler(Sampler[list[int]]):
    """
    BatchSampler that enforces targeted length stratification within each mini-batch.

    Length strata:
      - Short:  L in [min_len, short_max]   (default: [7, 12],  target: 40%)
      - Medium: L in [short_max+1, med_max] (default: [13, 18], target: 40%)
      - Long:   L in [med_max+1, max_len]   (default: [19, 30], target: 20% minimum)

    Guarantees every batch has:
      - n_short = round(batch_size * p_short)
      - n_medium = round(batch_size * p_medium)
      - n_long = batch_size - n_short - n_medium  (>= 20%)

    Indices within each stratum are shuffled every epoch. When smaller strata
    (such as long peptides) run out of samples, they wrap around with fresh reshuffling,
    ensuring continuous coverage throughout the training epoch.
    Indices within each yielded batch are shuffled to ensure position invariance.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        batch_size: int = 256,
        short_range: tuple[int, int] = (7, 12),
        medium_range: tuple[int, int] = (13, 18),
        long_range: tuple[int, int] = (19, 30),
        p_short: float = 0.40,
        p_medium: float = 0.40,
        p_long: float = 0.20,
        shuffle: bool = True,
        seed: int = 42,
        num_batches_per_epoch: int | None = None,
    ):
        super().__init__()
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Validate ranges
        self.short_range = short_range
        self.medium_range = medium_range
        self.long_range = long_range

        # Partition indices into strata
        self.short_indices = np.where(
            (self.lengths >= short_range[0]) & (self.lengths <= short_range[1])
        )[0]
        self.medium_indices = np.where(
            (self.lengths >= medium_range[0]) & (self.lengths <= medium_range[1])
        )[0]
        self.long_indices = np.where(
            (self.lengths >= long_range[0]) & (self.lengths <= long_range[1])
        )[0]

        # Allocate batch counts
        # Normalize probabilities if sum != 1.0
        total_p = p_short + p_medium + p_long
        p_short /= total_p
        p_medium /= total_p
        p_long /= total_p

        self.n_short = int(round(batch_size * p_short))
        self.n_medium = int(round(batch_size * p_medium))
        self.n_long = batch_size - self.n_short - self.n_medium

        if len(self.long_indices) == 0:
            raise ValueError(f"No samples found in long range {long_range}")
        if len(self.short_indices) == 0:
            raise ValueError(f"No samples found in short range {short_range}")
        if len(self.medium_indices) == 0:
            raise ValueError(f"No samples found in medium range {medium_range}")

        # Total batches per epoch
        if num_batches_per_epoch is not None:
            self.total_batches = num_batches_per_epoch
        else:
            self.total_batches = max(1, len(self.lengths) // batch_size)

        print(
            f"[LengthStratifiedBatchSampler] Initialized with {len(self.lengths):,} samples:\n"
            f"  - Short  [{short_range[0]}-{short_range[1]}]:  {len(self.short_indices):,} samples ({len(self.short_indices)/len(self.lengths):.1%}) -> {self.n_short}/batch ({self.n_short/batch_size:.1%})\n"
            f"  - Medium [{medium_range[0]}-{medium_range[1]}]: {len(self.medium_indices):,} samples ({len(self.medium_indices)/len(self.lengths):.1%}) -> {self.n_medium}/batch ({self.n_medium/batch_size:.1%})\n"
            f"  - Long   [{long_range[0]}-{long_range[1]}]:   {len(self.long_indices):,} samples ({len(self.long_indices)/len(self.lengths):.1%}) -> {self.n_long}/batch ({self.n_long/batch_size:.1%})\n"
            f"  - Total Batches/Epoch: {self.total_batches:,} (Batch Size={batch_size})"
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 10007)

        # Helper to produce endless shuffled generator from a pool
        def get_pool_generator(indices: np.ndarray, n_needed_per_step: int):
            pool = indices.copy()
            if self.shuffle:
                rng.shuffle(pool)
            ptr = 0
            while True:
                if ptr + n_needed_per_step > len(pool):
                    # Reshuffle and wrap around
                    if self.shuffle:
                        rng.shuffle(pool)
                    ptr = 0
                batch_slice = pool[ptr : ptr + n_needed_per_step]
                ptr += n_needed_per_step
                yield batch_slice

        gen_short = get_pool_generator(self.short_indices, self.n_short)
        gen_medium = get_pool_generator(self.medium_indices, self.n_medium)
        gen_long = get_pool_generator(self.long_indices, self.n_long)

        for _ in range(self.total_batches):
            s_idx = next(gen_short)
            m_idx = next(gen_medium)
            l_idx = next(gen_long)

            batch = np.concatenate([s_idx, m_idx, l_idx])
            if self.shuffle:
                rng.shuffle(batch)
            yield batch.tolist()
