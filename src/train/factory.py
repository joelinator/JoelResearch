"""Model construction helpers."""

from __future__ import annotations

import torch

from config.defaults import DEFAULTS, MODEL_CONFIG
from model.guidance import ClfGuidance
from model.model import DFMPeptideDecoder, PeptideLengthClassifier, SpectrumEncoder


def build_models(
    vocabulary: dict[str, int],
    device: torch.device,
    compile_models: bool = False,
    model_cfg: dict | None = None,
):
    if model_cfg is None:
        cfg = DEFAULTS.model
    elif isinstance(model_cfg, dict):
        cfg = model_cfg
    else:
        cfg = model_cfg

    def _get(key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    model_dim = int(_get("model_dim", DEFAULTS.model.model_dim))
    encoder_layers = int(_get("encoder_layers", DEFAULTS.model.encoder_layers))
    encoder_heads = int(_get("encoder_heads", DEFAULTS.model.encoder_heads))
    encoder_ff_dim = int(_get("encoder_ff_dim", DEFAULTS.model.encoder_ff_dim))
    decoder_blocks = int(_get("decoder_blocks", DEFAULTS.model.decoder_blocks))
    decoder_heads = int(_get("decoder_heads", DEFAULTS.model.decoder_heads))
    mlp_hidden_dim = int(_get("mlp_hidden_dim", DEFAULTS.model.mlp_hidden_dim))
    dropout = float(_get("dropout", DEFAULTS.model.dropout))
    max_charge = int(_get("max_charge", DEFAULTS.model.max_charge))
    max_length = int(_get("max_length", DEFAULTS.model.max_length))
    min_length = int(_get("min_length", DEFAULTS.model.min_length))

    spectrum_encoder = SpectrumEncoder(
        model_dim=model_dim,
        num_layers=encoder_layers,
        nhead=encoder_heads,
        dim_feedforward=encoder_ff_dim,
        dropout=dropout,
    ).to(device)
    length_predictor = PeptideLengthClassifier(
        in_dim=model_dim,
        hidden_dim=256,
        emb_dim=128,
    ).to(device)
    decoder = DFMPeptideDecoder.from_vocabulary(
        vocabulary,
        spec_dim=model_dim,
        emb_dim=model_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        n_decoder_blocks=decoder_blocks,
        num_heads=decoder_heads,
        dropout=dropout,
        max_charge=max_charge,
        max_length=max_length,
        min_length=min_length,
    ).to(device)
    guidance = ClfGuidance(cond_dim=model_dim).to(device)

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
