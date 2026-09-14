"""
Unit tests for Post-Translational Modifications (PTM) tokenization, vocabulary,
model surgery, and generative evaluation metrics.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from data.constants import (
    AA_MASSES_DICT,
    CANONICAL_TO_UNIMOD_MAP,
    N_TERM_MODIFICATIONS,
    PTM_ALIAS_MAP,
    PTM_PARENT_MAP,
    STANDARD_AMINO_ACIDS,
    TARGET_PTM_TOKENS,
)
from data.data import (
    build_vocabulary,
    canonicalize_peptide,
    decoder_output_size,
    get_aa_masses,
    get_output_aa_masses,
    parse_peptide,
    to_unimod_sequence,
)
from eval.metrics import (
    compute_denovo_metrics,
    count_matching_amino_acids,
    peptide_matches_mass_based,
)
from model.model import DFMPeptideDecoder
from train.lightning import DFMLightningModule
from train.surgery import transplant_checkpoint, transplant_state_dict


def test_parse_peptide_standard():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    tokens = parse_peptide(seq)
    assert len(tokens) == 20
    assert tokens == list(seq)


def test_parse_peptide_ptms_and_aliases():
    seq = "AENM(ox)IQLQFR"
    tokens = parse_peptide(seq)
    assert tokens == ["A", "E", "N", "M(ox)", "I", "Q", "L", "Q", "F", "R"]

    seq2 = "S(ph)EEEPKPK"
    tokens2 = parse_peptide(seq2)
    assert tokens2 == ["S(ph)", "E", "E", "E", "P", "K", "P", "K"]

    seq3 = "N(deam)Q(deam)C(cam)"
    tokens3 = parse_peptide(seq3)
    assert tokens3 == ["N(deam)", "Q(deam)", "C(cam)"]

    # Alias normalization
    seq4 = "M(+15.99)C(+57.02)T(+79.97)Y(+79.97)"
    tokens4 = parse_peptide(seq4)
    assert tokens4 == ["M(ox)", "C(cam)", "T(ph)", "Y(ph)"]


def test_vocabulary_structure():
    vocab = build_vocabulary(include_ptms=True)
    # Check standard amino acids are at 0..19
    for idx, aa in enumerate(STANDARD_AMINO_ACIDS):
        assert vocab[aa] == idx

    # Check target PTMs are present
    for ptm in TARGET_PTM_TOKENS:
        assert ptm in vocab
        assert vocab[ptm] >= 20

    # Check special tokens are at the end
    assert "<pad>" in vocab
    assert "<mask_token>" in vocab
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask_token>"]
    assert pad_id == max(vocab.values()) - 1
    assert mask_id == max(vocab.values())

    out_size = decoder_output_size(vocab)
    assert out_size == len(STANDARD_AMINO_ACIDS) + len(TARGET_PTM_TOKENS)

    masses = get_output_aa_masses(vocab)
    assert masses.shape[0] == out_size
    assert (masses != 0).all()  # Non-zero (ammonia loss has negative mass delta -17.03 Da)


def test_decoder_initialization_with_ptm():
    vocab = build_vocabulary(include_ptms=True)
    decoder = DFMPeptideDecoder.from_vocabulary(vocab)
    assert decoder.peptide_embedding.weight.shape[0] == max(vocab.values()) + 1
    assert decoder.head.weight.shape[0] == decoder.output_vocab_size
    assert decoder.head.bias.shape[0] == decoder.output_vocab_size


def test_mass_based_matching_with_ptms():
    # Exact PTM match
    assert peptide_matches_mass_based("AENM(ox)R", "AENM(ox)R")

    # M vs M(ox) should NOT match because mass differs by 16 Da
    assert not peptide_matches_mass_based("AENMR", "AENM(ox)R")

    # I and L should match due to identical mass (113.084 Da)
    assert peptide_matches_mass_based("AENIR", "AENLR")

    # Count matching residues
    matches = count_matching_amino_acids("AENM(ox)R", "AENM(ox)R")
    assert matches == 5


def test_weight_surgery_preserves_and_expands():
    # Construct synthetic old vocabulary (20 AA + pad + mask = 22)
    old_vocab = {aa: idx for idx, aa in enumerate(STANDARD_AMINO_ACIDS)}
    old_vocab["<pad>"] = 20
    old_vocab["<mask_token>"] = 21

    new_vocab = build_vocabulary(include_ptms=True)

    emb_dim = 64
    old_out_size = 20
    new_out_size = len(STANDARD_AMINO_ACIDS) + len(TARGET_PTM_TOKENS)

    old_state_dict = {
        "decoder.peptide_embedding.weight": torch.randn(len(old_vocab), emb_dim),
        "decoder.head.weight": torch.randn(old_out_size, emb_dim),
        "decoder.head.bias": torch.randn(old_out_size),
        "decoder.spectrum_proj.weight": torch.randn(emb_dim, emb_dim),
        "aa_masses": torch.randn(old_out_size),
    }

    new_state_dict = transplant_state_dict(
        old_state_dict,
        old_vocab=old_vocab,
        new_vocab=new_vocab,
        parent_map=PTM_PARENT_MAP,
        perturbation_std=0.001,
        bias_prior_offset=-0.5,
    )

    # 1. Untouched layers must be bitwise identical
    assert torch.equal(
        new_state_dict["decoder.spectrum_proj.weight"],
        old_state_dict["decoder.spectrum_proj.weight"],
    )

    # 2. Existing standard amino acids embeddings must be preserved
    for aa in STANDARD_AMINO_ACIDS:
        old_idx = old_vocab[aa]
        new_idx = new_vocab[aa]
        assert torch.equal(
            new_state_dict["decoder.peptide_embedding.weight"][new_idx],
            old_state_dict["decoder.peptide_embedding.weight"][old_idx],
        )

    # 3. PTM tokens must be close to parent amino acid
    for ptm, parent in PTM_PARENT_MAP.items():
        parent_idx = old_vocab[parent]
        ptm_idx = new_vocab[ptm]
        diff = (
            new_state_dict["decoder.peptide_embedding.weight"][ptm_idx]
            - old_state_dict["decoder.peptide_embedding.weight"][parent_idx]
        )
        assert diff.abs().mean() < 0.05

    # 4. Special tokens moved correctly
    assert torch.equal(
        new_state_dict["decoder.peptide_embedding.weight"][new_vocab["<pad>"]],
        old_state_dict["decoder.peptide_embedding.weight"][old_vocab["<pad>"]],
    )
    assert torch.equal(
        new_state_dict["decoder.peptide_embedding.weight"][new_vocab["<mask_token>"]],
        old_state_dict["decoder.peptide_embedding.weight"][old_vocab["<mask_token>"]],
    )


def test_lightning_module_training_step():
    vocab = build_vocabulary(include_ptms=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DFMLightningModule(vocab, {"learning_rate": 1e-4}).to(device)

    # Create dummy micro-batch
    B, S, L = 4, 30, 8
    mz_array = torch.rand(B, S, device=device) * 1000.0
    intensity_array = torch.rand(B, S, device=device)
    precursor_mass = torch.tensor([800.0, 950.0, 1100.0, 750.0], device=device)
    precursor_charge = torch.tensor([2, 2, 3, 2], device=device)
    # Include both standard AA and PTM tokens (e.g. index 20 is M(ox))
    sequence = torch.tensor(
        [
            [0, 1, 20, 3, 27, 27, 27, 27],
            [4, 21, 6, 7, 8, 27, 27, 27],
            [22, 10, 11, 12, 13, 14, 27, 27],
            [15, 24, 17, 18, 19, 2, 3, 27],
        ],
        device=device,
    )
    mz_complementary = 1000.0 - mz_array
    length = torch.tensor([4, 5, 6, 7], device=device)
    padded_mask = sequence == vocab["<pad>"]
    spectrum_mask = torch.zeros(B, S, dtype=torch.bool, device=device)

    batch = (
        mz_array,
        intensity_array,
        precursor_mass,
        precursor_charge,
        sequence,
        mz_complementary,
        length,
        padded_mask,
        spectrum_mask,
    )

    loss = model.training_step(batch, 0)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_parse_peptide_unimod_and_nterm():
    # UNIMOD format parsing
    seq1 = "PEPM[UNIMOD:35]TIDEC[UNIMOD:4]K"
    toks1 = parse_peptide(seq1)
    assert toks1 == ["P", "E", "P", "M(ox)", "T", "I", "D", "E", "C(cam)", "K"]

    seq2 = "S[UNIMOD:21]T[UNIMOD:21]Y[UNIMOD:21]"
    toks2 = parse_peptide(seq2)
    assert toks2 == ["S(ph)", "T(ph)", "Y(ph)"]

    seq3 = "N[UNIMOD:7]Q[UNIMOD:7]S(p)T(p)Y(p)"
    toks3 = parse_peptide(seq3)
    assert toks3 == ["N(deam)", "Q(deam)", "S(ph)", "T(ph)", "Y(ph)"]

    # N-terminal modification parsing
    seq4 = "(+42.01)PEPTIDEK"
    toks4 = parse_peptide(seq4)
    assert toks4 == ["(+42.01)", "P", "E", "P", "T", "I", "D", "E", "K"]

    seq5 = "[UNIMOD:1]PEPTIDEK"
    toks5 = parse_peptide(seq5)
    assert toks5 == ["(+42.01)", "P", "E", "P", "T", "I", "D", "E", "K"]

    seq6 = "(+43.01)PEPTIDEK"
    toks6 = parse_peptide(seq6)
    assert toks6 == ["(+43.01)", "P", "E", "P", "T", "I", "D", "E", "K"]

    seq7 = "(-17.03)QPEPTIDEK"
    toks7 = parse_peptide(seq7)
    assert toks7 == ["(-17.03)", "Q", "P", "E", "P", "T", "I", "D", "E", "K"]

    seq8 = "[UNIMOD:385]QPEPTIDEK"
    toks8 = parse_peptide(seq8)
    assert toks8 == ["(-17.03)", "Q", "P", "E", "P", "T", "I", "D", "E", "K"]


def test_to_unimod_sequence_and_canonicalize():
    # Shorthand to UNIMOD
    assert to_unimod_sequence("M(ox)ATIK") == "M[UNIMOD:35]ATIK"
    assert to_unimod_sequence("(+42.01)M(ox)ATIK") == "[UNIMOD:1]M[UNIMOD:35]ATIK"
    assert to_unimod_sequence("(+43.01)C(cam)S(ph)K") == "[UNIMOD:5]C[UNIMOD:4]S[UNIMOD:21]K"
    assert to_unimod_sequence("(-17.03)N(deam)Q(deam)K") == "[UNIMOD:385]N[UNIMOD:7]Q[UNIMOD:7]K"

    # Canonicalize
    assert canonicalize_peptide("[UNIMOD:1]M[UNIMOD:35]ATIK") == "(+42.01)M(ox)ATIK"
    assert canonicalize_peptide("M(+15.99)ATIK") == "M(ox)ATIK"
    assert canonicalize_peptide("S(p)T(p)Y(p)") == "S(ph)T(ph)Y(ph)"


def test_exact_and_mass_match_across_unimod_and_deltas():
    # Both formats evaluate as exact matches
    assert canonicalize_peptide("M(ox)ATIK") == canonicalize_peptide("M[UNIMOD:35]ATIK")
    assert canonicalize_peptide("(+42.01)PEPTIDEK") == canonicalize_peptide("[UNIMOD:1]PEPTIDEK")

    # Mass based match
    assert peptide_matches_mass_based("M(ox)ATIK", "M[UNIMOD:35]ATIK")
    assert peptide_matches_mass_based("(+42.01)PEPTIDEK", "[UNIMOD:1]PEPTIDEK")
    assert peptide_matches_mass_based("(+43.01)PEPTIDEK", "[UNIMOD:5]PEPTIDEK")
    assert peptide_matches_mass_based("(-17.03)QPEPTIDEK", "[UNIMOD:385]QPEPTIDEK")

    # Metrics exact peptide accuracy
    metrics = compute_denovo_metrics(
        predictions=["M(ox)ATIK", "(+42.01)PEPTIDEK"],
        targets=["M[UNIMOD:35]ATIK", "[UNIMOD:1]PEPTIDEK"],
    )
    assert metrics.exact_peptide_accuracy == 1.0
    assert metrics.mass_peptide_accuracy == 1.0


if __name__ == "__main__":
    test_parse_peptide_standard()
    print("✓ test_parse_peptide_standard passed")
    test_parse_peptide_ptms_and_aliases()
    print("✓ test_parse_peptide_ptms_and_aliases passed")
    test_parse_peptide_unimod_and_nterm()
    print("✓ test_parse_peptide_unimod_and_nterm passed")
    test_to_unimod_sequence_and_canonicalize()
    print("✓ test_to_unimod_sequence_and_canonicalize passed")
    test_exact_and_mass_match_across_unimod_and_deltas()
    print("✓ test_exact_and_mass_match_across_unimod_and_deltas passed")
    test_vocabulary_structure()
    print("✓ test_vocabulary_structure passed")
    test_decoder_initialization_with_ptm()
    print("✓ test_decoder_initialization_with_ptm passed")
    test_mass_based_matching_with_ptms()
    print("✓ test_mass_based_matching_with_ptms passed")
    test_weight_surgery_preserves_and_expands()
    print("✓ test_weight_surgery_preserves_and_expands passed")
    test_lightning_module_training_step()
    print("✓ test_lightning_module_training_step passed")
    print("🎉 ALL PTM TESTS PASSED SUCCESSFULLY!")
