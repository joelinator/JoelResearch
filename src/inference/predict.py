from __future__ import annotations

import torch

from data.data import invert_vocabulary
from data.lengths import apply_length_padding, class_to_length, clamp_length, length_to_active_mask
from flow_matching.sampling import (
    inference_sample_mask,
    inference_sample_uniform,
    sample_uniform_noise,
)


def decode_tokens(token_ids: torch.Tensor, vocab: dict[str, int]) -> list[str]:
    """Convert token indices to amino-acid strings (strips <pad> and <mask>)."""
    index_to_token = invert_vocabulary(vocab)
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask>"]
    sequences = []

    for row in token_ids.tolist():
        residues = []
        for token_id in row:
            if token_id in (pad_id, mask_id):
                continue
            residues.append(index_to_token[token_id])
        sequences.append("".join(residues))
    return sequences


def _initialize_noisy_sequence(
    batch_size: int,
    seq_len: int,
    vocab: dict[str, int],
    device: torch.device,
    scheme: str,
    length: torch.Tensor,
) -> torch.Tensor:
    """
    Initialize x_t at t=0.

    Active positions (0 .. length-1) start noisy; the padding tail is `<pad>`.
    """
    pad_id = vocab["<pad>"]
    active_mask = length_to_active_mask(length, seq_len)

    if scheme == "mask":
        noisy = torch.full(
            (batch_size, seq_len),
            vocab["<mask>"],
            dtype=torch.long,
            device=device,
        )
    else:
        noisy = sample_uniform_noise(vocab, (batch_size, seq_len), device=device)

    x_t = torch.where(active_mask, noisy, torch.tensor(pad_id, device=device))
    return x_t


@torch.no_grad()
def predict_peptide(
    mz_array: torch.Tensor,
    intensity_array: torch.Tensor,
    precursor_mass: torch.Tensor,
    precursor_charge: torch.Tensor,
    mz_complementary: torch.Tensor,
    spectrum_mask: torch.Tensor,
    vocabulary: dict[str, int],
    spectrum_encoder,
    length_predictor,
    decoder,
    guidance,
    scheduler,
    num_steps: int = 20,
    noising_scheme: str = "uniform",
    guidance_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Run de novo inference: predict peptide length, then decode the sequence.

    Length governs which positions are decoded (amino acids only); positions
    j >= predicted_length are kept as structural `<pad>` throughout.
    """
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()

    device = mz_array.device
    batch_size = mz_array.shape[0]
    pad_id = vocabulary["<pad>"]


    print("----->>>>Debug, mz_array", mz_array[:2])

    print("----->>>>Debug, int_array", intensity_array[:2])

    print("----->>>>Debug, mz_compelementary", mz_complementary[:2])
    
    print("----->>>>Debug, spectrum_mask", spectrum_mask[:2])
    

    spectrum_emb_cls, spectrum_emb_peaks, full_mask = spectrum_encoder(
        mz_array,
        mz_complementary,
        intensity_array,
        spectrum_mask,
    )

    print("----->>>>Debug, emb_cls", spectrum_emb_cls[:2])

    print("----->>>>Debug, emb_peaks", spectrum_emb_peaks[:2])

    print("----->>>>Debug, full_mask", full_mask[:2])

    length_logits = length_predictor(
        spectrum_emb_cls,
        precursor_mass,
        precursor_charge,
    )
    predicted_lengths = clamp_length(class_to_length(length_logits.argmax(dim=-1)))
    max_len = int(predicted_lengths.max().item())
    active_mask = length_to_active_mask(predicted_lengths, max_len)

    print("----->>>>Debug, length", length_logits[:2])

    x_t = _initialize_noisy_sequence(
        batch_size,
        max_len,
        vocabulary,
        device,
        noising_scheme,
        predicted_lengths,
    )

    time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    sample_step = (
        inference_sample_mask if noising_scheme == "mask" else inference_sample_uniform
    )

    for step_idx in range(num_steps):
        t = time_grid[step_idx].expand(batch_size)
        delta_t = float(time_grid[step_idx + 1] - time_grid[step_idx])
        kt, kt_derivative = scheduler(t)

        if guidance_scale != 1.0:
            conditioned = guidance(spectrum_emb_peaks, guidance_prob=0.0, need_guidance=False)
            unconditioned = guidance.unconditional.expand_as(spectrum_emb_peaks)
            conditioner = unconditioned + guidance_scale * (conditioned - unconditioned)
        else:
            conditioner = guidance(
                spectrum_emb_peaks,
                guidance_prob=0.0,
                need_guidance=False,
            )
        
        print("----->>>>Debug, conditioner", conditioner[:2])

        logits = decoder(
            t,
            precursor_mass,
            precursor_charge,
            conditioner,
            x_t,
            predicted_lengths.float(),
            full_mask,
        )
        print("----->>>>Debug, logits", logits[:2])

        # Only score / update active peptide positions (amino-acid logits only).
        logits = logits.masked_fill(~active_mask.unsqueeze(-1), float("-inf"))

        x_t = sample_step(
            kt, kt_derivative, x_t, logits, vocabulary, delta_t, active_mask=active_mask
        )
        x_t = apply_length_padding(x_t, predicted_lengths, pad_id)

    sequences = decode_tokens(x_t, vocabulary)
    return x_t, predicted_lengths, sequences
