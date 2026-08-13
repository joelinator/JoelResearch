"""
Unit tests for LengthStratifiedBatchSampler.
Verifies that every generated mini-batch conforms to the prescribed length proportions:
  - Short:  L in [7, 12]  (~40%)
  - Medium: L in [13, 18] (~40%)
  - Long:   L in [19, 30] (>= 20%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from data.samplers import LengthStratifiedBatchSampler


def test_length_stratified_batch_sampler_proportions():
    # Simulate a realistic right-skewed peptide length distribution:
    # 70% short [7-12], 25% medium [13-18], only 5% long [19-30]
    rng = np.random.default_rng(42)
    n_samples = 10_000
    lengths = []
    for _ in range(n_samples):
        r = rng.random()
        if r < 0.70:
            lengths.append(rng.integers(7, 13))
        elif r < 0.95:
            lengths.append(rng.integers(13, 19))
        else:
            lengths.append(rng.integers(19, 31))

    lengths = np.array(lengths)

    batch_size = 100
    sampler = LengthStratifiedBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        p_short=0.40,
        p_medium=0.40,
        p_long=0.20,
        shuffle=True,
        seed=123,
    )

    batches = list(sampler)
    assert len(batches) == len(lengths) // batch_size

    # Check every batch
    for b_idx, batch in enumerate(batches):
        assert len(batch) == batch_size
        batch_lens = lengths[batch]
        n_short = np.sum((batch_lens >= 7) & (batch_lens <= 12))
        n_med = np.sum((batch_lens >= 13) & (batch_lens <= 18))
        n_long = np.sum((batch_lens >= 19) & (batch_lens <= 30))

        assert n_short == 40, f"Batch {b_idx} short count: {n_short}"
        assert n_med == 40, f"Batch {b_idx} med count: {n_med}"
        assert n_long == 20, f"Batch {b_idx} long count: {n_long}"

    print(f"\n[Test Passed] Verified {len(batches)} batches: Exactly 40% short, 40% med, 20% long in every batch!")


if __name__ == "__main__":
    test_length_stratified_batch_sampler_proportions()
