"""Peptide length class mapping, noising, and embedding tests."""

import torch

from data.lengths import (
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    NUM_LENGTH_CLASSES,
    class_to_length,
    length_to_class,
    validate_peptide_length,
)
from model.model import DFMPeptideDecoder, PeptideLengthClassifier
from train.utils import length_noiser


def test_length_class_roundtrip():
    for length in range(MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH + 1):
        class_id = int(length_to_class(torch.tensor(length)))
        assert class_id == length - 1
        assert int(class_to_length(torch.tensor(class_id))) == length


def test_length_classifier_output_dim():
    classifier = PeptideLengthClassifier()
    assert classifier.n_classes == NUM_LENGTH_CLASSES


def test_validate_bounds():
    for length in (MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH):
        validate_peptide_length(length)

    try:
        validate_peptide_length(0)
        assert False, "expected ValueError for length 0"
    except ValueError:
        pass

    try:
        validate_peptide_length(MAX_PEPTIDE_LENGTH + 1)
        assert False, f"expected ValueError for length {MAX_PEPTIDE_LENGTH + 1}"
    except ValueError:
        pass


def test_length_noiser():
    lengths = torch.tensor([5, 10, 15, 20, 30])
    
    # When noising_prob=0.0, it should never noise
    noised, is_noised = length_noiser(lengths, noising_prob=0.0)
    assert not is_noised
    assert torch.equal(noised, lengths)

    # When noising_prob=1.0, it should always noise by +/- 1 and stay in [MIN, MAX]
    noised, is_noised = length_noiser(lengths, noising_prob=1.0)
    assert is_noised
    assert (noised >= MIN_PEPTIDE_LENGTH).all()
    assert (noised <= MAX_PEPTIDE_LENGTH).all()
    assert ((noised - lengths).abs() <= 1).all()


def test_decoder_length_embedding():
    vocab = {chr(65 + i): i for i in range(20)}
    vocab["<pad>"] = 20
    vocab["<mask>"] = 21

    decoder = DFMPeptideDecoder.from_vocabulary(
        vocab,
        spec_dim=64,
        emb_dim=64,
        mlp_hidden_dim=64,
        n_decoder_blocks=1,
        num_heads=2,
        min_length=MIN_PEPTIDE_LENGTH,
        max_length=MAX_PEPTIDE_LENGTH,
    )
    # Check length_embedding table size is max_length + 1
    assert decoder.length_embedding.num_embeddings == MAX_PEPTIDE_LENGTH + 1
    assert decoder.length_embedding.embedding_dim == 64


if __name__ == "__main__":
    test_length_class_roundtrip()
    test_length_classifier_output_dim()
    test_validate_bounds()
    test_length_noiser()
    test_decoder_length_embedding()
    print("All peptide length, noising, and embedding checks passed.")
