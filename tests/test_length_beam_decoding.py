"""Tests for Top-k Length Beam Decoding at inference."""

import torch

from data.constants import AA_MASSES_DICT, STANDARD_AMINO_ACIDS
from data.lengths import MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH
from flow_matching.scheduler import cosine_scheduler
from inference.predict import decode_tokens, predict_peptide
from model.guidance import ClfGuidance
from model.model import DFMPeptideDecoder, PeptideLengthClassifier, SpectrumEncoder


def _build_mock_models():
    vocab = {aa: i for i, aa in enumerate(sorted(STANDARD_AMINO_ACIDS))}
    vocab["<pad>"] = len(vocab)
    vocab["<mask_token>"] = len(vocab)
    vocab["<mask_token>"]  # check

    # Alias <mask_token> and <mask> for robustness
    vocab["<mask_token>"] = vocab["<mask_token>"]
    vocab["<mask>"] = vocab["<mask_token>"]

    spec_enc = SpectrumEncoder(model_dim=32, num_layers=1, nhead=2, dim_feedforward=64)
    len_pred = PeptideLengthClassifier(
        n_classes=MAX_PEPTIDE_LENGTH - MIN_PEPTIDE_LENGTH + 1,
        in_dim=32,
        hidden_dim=16,
        emb_dim=16,
    )
    decoder = DFMPeptideDecoder.from_vocabulary(
        vocab,
        spec_dim=32,
        emb_dim=32,
        mlp_hidden_dim=32,
        n_decoder_blocks=1,
        num_heads=2,
    )
    guidance = ClfGuidance(cond_dim=32)

    return vocab, spec_enc, len_pred, decoder, guidance


def test_beam_decoding_shapes_and_selection():
    vocab, spec_enc, len_pred, decoder, guidance = _build_mock_models()
    batch_size = 2
    n_peaks = 10

    mz_array = torch.rand(batch_size, n_peaks) * 1000.0
    intensity_array = torch.rand(batch_size, n_peaks)
    precursor_mass = torch.tensor([800.0, 1200.0])
    precursor_charge = torch.tensor([2, 3])
    mz_complementary = torch.rand(batch_size, n_peaks) * 1000.0
    spectrum_mask = torch.zeros(batch_size, n_peaks, dtype=torch.bool)

    # Test K=1 (argmax mode)
    tokens_k1, lengths_k1, seqs_k1 = predict_peptide(
        mz_array=mz_array,
        intensity_array=intensity_array,
        precursor_mass=precursor_mass,
        precursor_charge=precursor_charge,
        mz_complementary=mz_complementary,
        spectrum_mask=spectrum_mask,
        vocabulary=vocab,
        spectrum_encoder=spec_enc,
        length_predictor=len_pred,
        decoder=decoder,
        guidance=guidance,
        scheduler=cosine_scheduler,
        num_steps=2,
        top_k_lengths=1,
    )
    assert tokens_k1.shape[0] == batch_size
    assert lengths_k1.shape[0] == batch_size
    assert len(seqs_k1) == batch_size

    # Test K=3 (Top-3 beam decoding)
    tokens_k3, lengths_k3, seqs_k3 = predict_peptide(
        mz_array=mz_array,
        intensity_array=intensity_array,
        precursor_mass=precursor_mass,
        precursor_charge=precursor_charge,
        mz_complementary=mz_complementary,
        spectrum_mask=spectrum_mask,
        vocabulary=vocab,
        spectrum_encoder=spec_enc,
        length_predictor=len_pred,
        decoder=decoder,
        guidance=guidance,
        scheduler=cosine_scheduler,
        num_steps=2,
        top_k_lengths=3,
        alpha=0.1,
    )
    assert tokens_k3.shape[0] == batch_size
    assert lengths_k3.shape[0] == batch_size
    assert len(seqs_k3) == batch_size
    for seq in seqs_k3:
        assert isinstance(seq, str)


if __name__ == "__main__":
    test_beam_decoding_shapes_and_selection()
    print("All Top-k Length Beam Decoding checks passed.")
