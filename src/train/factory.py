"""Model construction helpers."""

from __future__ import annotations

import torch

from config.defaults import DEFAULTS
from model.guidance import ClfGuidance
from model.model import DFMPeptideDecoder, PeptideLengthClassifier, SpectrumEncoder


def build_models(
    vocabulary: dict[str, int],
    device: torch.device,
    compile_models: bool = False,
):
    model_cfg = DEFAULTS.model
    spectrum_encoder = SpectrumEncoder(
        model_dim=model_cfg.model_dim,
        num_layers=model_cfg.encoder_layers,
        nhead=model_cfg.encoder_heads,
        dim_feedforward=model_cfg.encoder_ff_dim,
        dropout=model_cfg.dropout,
    ).to(device)
    length_predictor = PeptideLengthClassifier(
        in_dim=model_cfg.model_dim,
        hidden_dim=model_cfg.mlp_hidden_dim // 2,
        emb_dim=model_cfg.mlp_hidden_dim // 4,
    ).to(device)
    decoder = DFMPeptideDecoder.from_vocabulary(
        vocabulary,
        spec_dim=model_cfg.model_dim,
        emb_dim=model_cfg.model_dim,
        mlp_hidden_dim=model_cfg.mlp_hidden_dim,
        n_decoder_blocks=model_cfg.decoder_blocks,
        num_heads=model_cfg.decoder_heads,
        dropout=model_cfg.dropout,
        max_charge=model_cfg.max_charge,
    ).to(device)
    guidance = ClfGuidance(cond_dim=model_cfg.model_dim).to(device)

    if compile_models and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            # The encoder runs once per batch – default mode (max optimisations).
            spectrum_encoder = torch.compile(spectrum_encoder)
            # The decoder runs num_steps times per batch in a fixed-shape loop;
            # reduce-overhead mode minimises Python dispatch cost in that loop.
            decoder = torch.compile(decoder, mode="reduce-overhead")
            print("torch.compile applied: spectrum_encoder (default), decoder (reduce-overhead)")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing with uncompiled models.")

    return spectrum_encoder, length_predictor, decoder, guidance
