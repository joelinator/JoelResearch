import pytest
import torch
import torch.nn as nn

from data.data import build_vocabulary, decoder_output_size, decoder_output_token_ids, get_output_aa_masses
from eval.evaluate import evaluate_teacher_forced
from inference.predict import predict_peptide
from model.model import DFMPeptideDecoder, PeptideLengthClassifier, SpectrumEncoder
from train.lightning import DFMLightningModule


def test_implicit_length_vocabulary_and_output_size():
    vocab = build_vocabulary(include_ptms=True)
    vocab_size = len(vocab)
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask_token>"]

    # When implicit_length=True, pad_id IS included in the output targets (only mask is special)
    out_ids = decoder_output_token_ids(vocab, include_pad=True)
    assert pad_id in out_ids
    assert mask_id not in out_ids
    assert len(out_ids) == vocab_size - 1
    assert decoder_output_size(vocab, include_pad=True) == vocab_size - 1

    # Masses should have length vocab_size - 1, and pad_id mass should be 0.0
    masses = get_output_aa_masses(vocab, include_pad=True)
    assert masses.shape[0] == vocab_size - 1
    assert masses[pad_id].item() == 0.0


def test_implicit_length_lightning_module_and_freezing():
    vocab = build_vocabulary(include_ptms=True)
    args = {
        "implicit_length": True,
        "model_dim": 64,
        "encoder_layers": 2,
        "encoder_heads": 4,
        "encoder_ff_dim": 128,
        "decoder_blocks": 2,
        "decoder_heads": 4,
        "mlp_hidden_dim": 128,
        "learning_rate": 1e-3,
    }
    model = DFMLightningModule(vocab, args)

    # Check length_predictor is frozen
    for p in model.length_predictor.parameters():
        assert not p.requires_grad

    # Decoder and encoder should have trainable parameters
    assert any(p.requires_grad for p in model.decoder.parameters())
    assert any(p.requires_grad for p in model.spectrum_encoder.parameters())

    # Create dummy batch with max_peptide_length=30
    batch_size = 4
    seq_len = 30
    mz_array = torch.rand(batch_size, 50) * 1000.0
    intensity_array = torch.rand(batch_size, 50)
    precursor_mass = torch.full((batch_size,), 1200.0)
    precursor_charge = torch.full((batch_size,), 2, dtype=torch.long)
    sequence = torch.full((batch_size, seq_len), vocab["<pad>"], dtype=torch.long)
    # Fill first 10 positions with standard amino acids
    sequence[:, :10] = torch.randint(0, 20, (batch_size, 10))
    mz_complementary = torch.rand(batch_size, 50) * 1000.0
    length = torch.full((batch_size,), 10, dtype=torch.long)
    padded_mask = sequence == vocab["<pad>"]
    spectrum_mask = torch.zeros(batch_size, 50, dtype=torch.bool)

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
    assert loss is not None
    assert not torch.isnan(loss)
    loss.backward()

    # Check that decoder got gradients but length_predictor didn't
    for p in model.length_predictor.parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_implicit_length_inference():
    vocab = build_vocabulary(include_ptms=True)
    args = {
        "implicit_length": True,
        "model_dim": 64,
        "encoder_layers": 2,
        "encoder_heads": 4,
        "encoder_ff_dim": 128,
        "decoder_blocks": 2,
        "decoder_heads": 4,
        "mlp_hidden_dim": 128,
    }
    model = DFMLightningModule(vocab, args)
    model.eval()

    batch_size = 2
    mz_array = torch.rand(batch_size, 50) * 1000.0
    intensity_array = torch.rand(batch_size, 50)
    precursor_mass = torch.full((batch_size,), 1200.0)
    precursor_charge = torch.full((batch_size,), 2, dtype=torch.long)
    mz_complementary = torch.rand(batch_size, 50) * 1000.0
    spectrum_mask = torch.zeros(batch_size, 50, dtype=torch.bool)

    with torch.no_grad():
        token_ids, pred_lengths, sequences = predict_peptide(
            mz_array=mz_array,
            mz_complementary=mz_complementary,
            intensity_array=intensity_array,
            spectrum_mask=spectrum_mask,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            spectrum_encoder=model.spectrum_encoder,
            length_predictor=model.length_predictor,
            decoder=model.decoder,
            guidance=model.guidance,
            scheduler="cosine",
            vocabulary=vocab,
            num_steps=3,
            implicit_length=True,
            max_peptide_length=30,
        )

    assert token_ids.shape == (batch_size, 30)
    assert len(sequences) == batch_size
    assert len(pred_lengths) == batch_size
    for i, seq in enumerate(sequences):
        assert pred_lengths[i].item() == len(seq)
