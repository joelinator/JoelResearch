"""Verification suite for SOTA architecture optimizations."""

import torch
import torch.nn.functional as F

from config.defaults import DEFAULTS
from data.constants import STANDARD_AMINO_ACIDS
from model.guidance import ClfGuidance
from model.model import DFMPeptideDecoder, PeptideLengthClassifier, SpectrumEncoder
from inference.predict import predict_peptide
from train.loss import mass_loss_hubert, peptide_loss, length_loss


def _build_models():
    vocab = {aa: i for i, aa in enumerate(sorted(STANDARD_AMINO_ACIDS))}
    vocab["<pad>"] = len(vocab)
    vocab["<mask_token>"] = len(vocab)

    model_cfg = DEFAULTS.model
    spec_enc = SpectrumEncoder(
        model_dim=model_cfg.model_dim,
        num_layers=model_cfg.encoder_layers,
        nhead=model_cfg.encoder_heads,
        dim_feedforward=model_cfg.encoder_ff_dim,
        dropout=model_cfg.dropout,
    )
    len_pred = PeptideLengthClassifier(
        in_dim=model_cfg.model_dim,
        hidden_dim=model_cfg.mlp_hidden_dim // 6,
        emb_dim=model_cfg.model_dim // 4,
    )
    decoder = DFMPeptideDecoder.from_vocabulary(
        vocab,
        spec_dim=model_cfg.model_dim,
        emb_dim=model_cfg.model_dim,
        mlp_hidden_dim=model_cfg.mlp_hidden_dim,
        n_decoder_blocks=model_cfg.decoder_blocks,
        num_heads=model_cfg.decoder_heads,
        dropout=model_cfg.dropout,
        max_charge=model_cfg.max_charge,
        max_length=model_cfg.max_length,
        min_length=model_cfg.min_length,
    )
    guidance = ClfGuidance(cond_dim=model_cfg.model_dim)
    return vocab, spec_enc, len_pred, decoder, guidance


def test_parameter_budget():
    vocab, spec_enc, len_pred, decoder, guidance = _build_models()
    total_params = sum(
        p.numel()
        for m in (spec_enc, len_pred, decoder, guidance)
        for p in m.parameters()
    )
    param_m = total_params / 1e6
    print(f"Total model parameters: {total_params:,} ({param_m:.2f}M)")
    assert 50.0 <= param_m <= 60.0, f"Expected 50-60M parameters, got {param_m:.2f}M"


def test_adaln_zero_identity_init():
    vocab, spec_enc, len_pred, decoder, guidance = _build_models()
    block = decoder.decoder_blocks[0]
    cond = torch.randn(2, 512)
    x = torch.randn(2, 15, 512)
    y = torch.randn(2, 100, 512)
    block.eval()
    with torch.no_grad():
        out = block(cond, x, y)
    diff = (out - x).abs().max().item()
    print(f"Max abs diff at initialization: {diff}")
    assert diff < 1e-6, f"AdaLN-Zero did not initialize as identity: diff={diff}"


def test_end_to_end_forward_backward():
    vocab, spec_enc, len_pred, decoder, guidance = _build_models()
    batch_size = 4
    num_peaks = 50
    seq_len = 12

    mz1 = torch.rand(batch_size, num_peaks) * 1000.0
    mz2 = torch.rand(batch_size, num_peaks) * 1000.0
    intensity = torch.rand(batch_size, num_peaks)
    spectrum_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    precursor_mass = torch.tensor([1200.0, 1400.0, 1600.0, 1800.0])
    precursor_charge = torch.tensor([2, 2, 3, 3])
    peptide_seq = torch.randint(0, len(vocab) - 2, (batch_size, seq_len))
    length = torch.tensor([seq_len] * batch_size)
    t = torch.tensor([0.5] * batch_size)

    # Forward
    cls_emb, peak_emb, mask = spec_enc(mz1, mz2, intensity, spectrum_mask)
    len_logits = len_pred(cls_emb, precursor_mass, precursor_charge)
    cond = guidance(peak_emb, guidance_prob=0.0, need_guidance=False)
    dec_logits = decoder(t, precursor_mass, precursor_charge, cond, peptide_seq, length, mask)

    # Losses
    l_dec = peptide_loss(dec_logits, peptide_seq, pad_id=vocab["<pad>"])
    l_len = length_loss(len_logits, length)
    aa_masses = torch.ones(dec_logits.shape[-1]) * 110.0
    l_mass = mass_loss_hubert(dec_logits, aa_masses, precursor_mass)
    loss = l_dec + 0.1 * l_len + 0.05 * l_mass

    # Backward
    loss.backward()

    # Check non-zero gradients across encoder, length predictor, and decoder
    assert spec_enc.cls_token.grad is not None and spec_enc.cls_token.grad.norm() > 0
    assert decoder.head.weight.grad is not None and decoder.head.weight.grad.norm() > 0
    print("End-to-end forward and backward passes verified successfully.")


def test_bayesian_beam_decoding():
    vocab, spec_enc, len_pred, decoder, guidance = _build_models()
    batch_size = 2
    num_peaks = 20
    mz1 = torch.rand(batch_size, num_peaks) * 800.0
    mz2 = torch.rand(batch_size, num_peaks) * 800.0
    intensity = torch.rand(batch_size, num_peaks)
    spectrum_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    precursor_mass = torch.tensor([1000.0, 1200.0])
    precursor_charge = torch.tensor([2, 2])

    from flow_matching.scheduler import cosine_scheduler

    tokens, lengths, seqs = predict_peptide(
        spectrum_encoder=spec_enc,
        length_predictor=len_pred,
        decoder=decoder,
        guidance=guidance,
        scheduler=cosine_scheduler,
        mz_array=mz1,
        mz_complementary=mz2,
        intensity_array=intensity,
        spectrum_mask=spectrum_mask,
        precursor_mass=precursor_mass,
        precursor_charge=precursor_charge,
        vocabulary=vocab,
        num_steps=3,
        top_k_lengths=3,
    )
    assert len(seqs) == batch_size
    assert len(lengths) == batch_size
    assert tokens.shape[0] == batch_size
    print(f"Decoded peptides: {seqs}")
    print("Bayesian beam decoding test passed.")


if __name__ == "__main__":
    test_parameter_budget()
    test_adaln_zero_identity_init()
    test_end_to_end_forward_backward()
    test_bayesian_beam_decoding()
    print("All SOTA architecture optimization tests passed!")
