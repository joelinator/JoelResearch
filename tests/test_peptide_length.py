"""Peptide length class mapping tests (no torch required for core logic)."""

import torch

from data.lengths import (
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    NUM_LENGTH_CLASSES,
    class_to_length,
    length_to_class,
    validate_peptide_length,
)
from model.model import PeptideLengthClassifier


def test_length_class_roundtrip():
    for length in range(MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH + 1):
        class_id = int(length_to_class(torch.tensor(length)))
        assert class_id == length - 1
        assert int(class_to_length(torch.tensor(class_id))) == length


def test_length_classifier_output_dim():
    classifier = PeptideLengthClassifier()
    assert classifier.n_classes == NUM_LENGTH_CLASSES == 30


def test_validate_bounds():
  for length in (MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH):
    validate_peptide_length(length)

  try:
    validate_peptide_length(0)
    assert False, "expected ValueError for length 0"
  except ValueError:
    pass

  try:
    validate_peptide_length(31)
    assert False, "expected ValueError for length 31"
  except ValueError:
    pass


if __name__ == "__main__":
    test_length_class_roundtrip()
    test_length_classifier_output_dim()
    test_validate_bounds()
    print("All peptide length checks passed.")
