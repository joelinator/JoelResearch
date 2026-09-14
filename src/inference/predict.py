from __future__ import annotations

import torch
import torch.nn.functional as F

from data.constants import AA_MASSES_DICT, M_H, M_H2O
from data.data import invert_vocabulary
from data.lengths import apply_length_padding, class_to_length, clamp_length, length_to_active_mask
from flow_matching.sampling import (
    inference_sample_mask,
    inference_sample_uniform,
    sample_uniform_noise,
)


def decode_tokens(token_ids: torch.Tensor, vocab: dict[str, int]) -> list[str]:
    """Convert token indices to amino-acid strings (strips <pad> and <mask_token>)."""
    index_to_token = invert_vocabulary(vocab)
    pad_id = vocab["<pad>"]
    mask_id = vocab.get("<mask_token>", vocab.get("<mask>"))
    sequences = []

    for row in token_ids.tolist():
        residues = []
        for token_id in row:
            if token_id in (pad_id, mask_id):
                continue
            residues.append(index_to_token.get(token_id, "?"))
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
    mask_id = vocab.get("<mask_token>", vocab.get("<mask" + ">"))
    active_mask = length_to_active_mask(length, seq_len)

    if scheme == "mask":
        noisy = torch.full(
            (batch_size, seq_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )
    else:
        noisy = sample_uniform_noise(vocab, (batch_size, seq_len), device=device)

    x_t = torch.where(active_mask, noisy, torch.tensor(pad_id, device=device))
    return x_t


def compute_fragment_matching_scores(
    x_t: torch.Tensor,
    active_mask: torch.Tensor,
    mz_array_exp: torch.Tensor,
    intensity_array_exp: torch.Tensor,
    spectrum_mask_exp: torch.Tensor,
    vocabulary: dict[str, int],
    tolerance_ppm: float = 20.0,
    tolerance_da: float = 0.05,
) -> torch.Tensor:
    """
    Compute the explained spectral intensity fraction for each candidate sequence
    by matching theoretical b- and y-ions against experimental spectrum peaks.

    Args:
        x_t: Candidate token sequences of shape (B_total, L).
        active_mask: Boolean tensor of shape (B_total, L) indicating active residues.
        mz_array_exp: Experimental peak m/z values of shape (B_total, S).
        intensity_array_exp: Peak intensities of shape (B_total, S).
        spectrum_mask_exp: True for padded/missing spectrum peaks (B_total, S).
        vocabulary: Token dictionary.
        tolerance_ppm: Mass tolerance in parts-per-million (default: 20.0).
        tolerance_da: Absolute mass tolerance in Daltons (default: 0.05).

    Returns:
        explained_intensity: Tensor of shape (B_total,) in [0, 1].
    """
    device = x_t.device
    mass_table = torch.zeros(max(vocabulary.values()) + 1, device=device, dtype=torch.float32)
    for tok, idx in vocabulary.items():
        if tok in AA_MASSES_DICT:
            mass_table[idx] = AA_MASSES_DICT[tok]

    # Token masses: (B_total, L)
    token_mass = mass_table[x_t] * active_mask.float()
    lengths = active_mask.sum(dim=-1).clamp(min=1)

    # b-ions: cumulative prefix sum + M_H (1.007276 Da)
    b_ions = token_mass.cumsum(dim=1) + M_H  # (B_total, L)

    # y-ions: total peptide mass - b_ions + 2*M_H + M_H2O
    total_mass = token_mass.sum(dim=-1, keepdim=True)
    y_ions = total_mass - (b_ions - M_H) + M_H2O + M_H  # (B_total, L)

    # Valid fragment ions exclude index L-1 (full peptide intact mass)
    positions = torch.arange(x_t.size(1), device=device).unsqueeze(0)
    valid_fragment = positions < (lengths.unsqueeze(1) - 1)  # (B_total, L)

    # Collect theoretical ions: (B_total, 2*L)
    theo_ions = torch.cat([b_ions, y_ions], dim=1)
    theo_valid = torch.cat([valid_fragment, valid_fragment], dim=1)  # (B_total, 2*L)
    theo_ions = theo_ions.masked_fill(~theo_valid, -1e9)

    # Mask padded spectrum peaks
    valid_peaks = ~spectrum_mask_exp  # (B_total, S)
    exp_intensity = intensity_array_exp * valid_peaks.float()
    total_spec_intensity = exp_intensity.sum(dim=-1).clamp(min=1e-7)

    # Pairwise matching between experimental peaks and theoretical ions
    exp_mz = mz_array_exp.unsqueeze(-1)  # (B_total, S, 1)
    theo_mz = theo_ions.unsqueeze(1)     # (B_total, 1, 2*L)

    diff = (exp_mz - theo_mz).abs()      # (B_total, S, 2*L)
    tol = torch.maximum(theo_mz * (tolerance_ppm * 1e-6), torch.tensor(tolerance_da, device=device))
    matched = (diff <= tol) & theo_valid.unsqueeze(1) & valid_peaks.unsqueeze(-1)  # (B_total, S, 2*L)

    peak_is_matched = matched.any(dim=-1)  # (B_total, S)
    matched_intensity = (exp_intensity * peak_is_matched.float()).sum(dim=-1)

    return (matched_intensity / total_spec_intensity).clamp(0.0, 1.0)


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
    noising_scheme: str = "mask",
    guidance_scale: float = 1.0,
    top_k_lengths: int = 3,
    alpha: float = 0.1,
    beta: float = 0.5,
    decoding_strategy: str = "confidence",
    temperature: float = 0.0,
    trypsin_prior: bool = False,
    tolerance_ppm: float = 20.0,
    tolerance_da: float = 0.05,
    return_scores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]] | tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """
    Run de novo inference with Top-k Length Beam Decoding and SOTA scoring.

    Predicts the top-k most likely peptide lengths from the length classifier,
    runs the peptide decoder on all candidate lengths in parallel across the batch,
    and scores candidate sequences based on:
        Score(Y) = log P(L|S) + mean log P(Y_i|S,L) - alpha * MassError + beta * FragMatch + Prior
    and selects the optimal sequence per spectrum.
    """
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()

    device = mz_array.device
    batch_size = mz_array.shape[0]
    pad_id = vocabulary["<pad>"]

    # 1. Encode spectra (computed once per spectrum)
    spectrum_emb_cls, spectrum_emb_peaks, peak_mask = spectrum_encoder(
        mz_array,
        mz_complementary,
        intensity_array,
        spectrum_mask,
    )

    # 2. Predict candidate lengths
    length_logits = length_predictor(
        spectrum_emb_cls,
        precursor_mass,
        precursor_charge,
    )
    k_cand = max(1, min(top_k_lengths, length_logits.shape[-1]))

    if k_cand == 1:
        length_classes = length_logits.argmax(dim=-1)
        cand_lengths_flat = clamp_length(class_to_length(length_classes))
        K = 1
        length_log_probs = F.log_softmax(length_logits, dim=-1)
        cand_length_log_probs = length_log_probs.gather(-1, length_classes.unsqueeze(-1)).squeeze(-1)
        precursor_mass_exp = precursor_mass
        precursor_charge_exp = precursor_charge
        spectrum_emb_peaks_exp = spectrum_emb_peaks
        peak_mask_exp = peak_mask
        mz_array_exp = mz_array
        intensity_array_exp = intensity_array
        spectrum_mask_exp = spectrum_mask
    else:
        topk = torch.topk(length_logits, k=k_cand, dim=-1)
        topk_classes = topk.indices  # (B, K)
        cand_lengths = clamp_length(class_to_length(topk_classes))  # (B, K)
        cand_lengths_flat = cand_lengths.reshape(-1)  # (B * K,)
        K = k_cand
        length_log_probs = F.log_softmax(length_logits, dim=-1)
        cand_length_log_probs = length_log_probs.gather(-1, topk_classes).reshape(-1)  # (B * K,)

        # Parallelize across candidates via batch interleave
        precursor_mass_exp = precursor_mass.repeat_interleave(K, dim=0)
        precursor_charge_exp = precursor_charge.repeat_interleave(K, dim=0)
        spectrum_emb_peaks_exp = spectrum_emb_peaks.repeat_interleave(K, dim=0)
        peak_mask_exp = peak_mask.repeat_interleave(K, dim=0)
        mz_array_exp = mz_array.repeat_interleave(K, dim=0)
        intensity_array_exp = intensity_array.repeat_interleave(K, dim=0)
        spectrum_mask_exp = spectrum_mask.repeat_interleave(K, dim=0)

    total_samples = batch_size * K
    max_len = int(cand_lengths_flat.max().item())
    active_mask = length_to_active_mask(cand_lengths_flat, max_len)

    # 3. Initialize noisy candidate sequences
    x_t = _initialize_noisy_sequence(
        total_samples,
        max_len,
        vocabulary,
        device,
        noising_scheme,
        cand_lengths_flat,
    )

    time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    sample_step = (
        inference_sample_mask if noising_scheme == "mask" else inference_sample_uniform
    )

    # Pre-cache conditioners outside integration loop
    if guidance_scale != 1.0:
        cond_conditioner = guidance(
            spectrum_emb_peaks_exp, guidance_prob=0.0, need_guidance=False
        )
        uncond_conditioner = guidance.unconditional.view(1, 1, -1).expand_as(
            spectrum_emb_peaks_exp
        )
    else:
        cond_conditioner = guidance(
            spectrum_emb_peaks_exp, guidance_prob=0.0, need_guidance=False
        )
        uncond_conditioner = None

    # 4. Parallel generative decoding loop
    last_logits = None
    for step_idx in range(num_steps):
        t = time_grid[step_idx].expand(total_samples)
        delta_t = float(time_grid[step_idx + 1] - time_grid[step_idx])
        kt, kt_derivative = scheduler(t)

        if guidance_scale != 1.0:
            cond_logits = decoder(
                t,
                precursor_mass_exp,
                precursor_charge_exp,
                cond_conditioner,
                x_t,
                cand_lengths_flat,
                peak_mask_exp,
                ~active_mask,
            )
            uncond_logits = decoder(
                t,
                precursor_mass_exp,
                precursor_charge_exp,
                uncond_conditioner,
                x_t,
                cand_lengths_flat,
                peak_mask_exp,
                ~active_mask,
            )
            logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        else:
            logits = decoder(
                t,
                precursor_mass_exp,
                precursor_charge_exp,
                cond_conditioner,
                x_t,
                cand_lengths_flat,
                peak_mask_exp,
                ~active_mask,
            )

        last_logits = logits
        if noising_scheme == "mask":
            x_t = sample_step(
                kt,
                kt_derivative,
                x_t,
                logits,
                vocabulary,
                delta_t,
                active_mask=active_mask,
                temperature=temperature,
                strategy=decoding_strategy,
            )
        else:
            x_t = sample_step(
                kt,
                kt_derivative,
                x_t,
                logits,
                vocabulary,
                delta_t,
                active_mask=active_mask,
                temperature=temperature if temperature > 0.0 else 1.0,
            )
        x_t = apply_length_padding(x_t, cand_lengths_flat, pad_id)

    # Final unmasking safety: if any active position is still masked, unmask via argmax logits
    mask_token_id = vocabulary.get("<mask_token>", vocabulary.get("<mask" + ">"))
    if mask_token_id is not None and last_logits is not None:
        rem_mask = (x_t == mask_token_id) & active_mask
        if rem_mask.any():
            x_t[rem_mask] = last_logits.argmax(dim=-1)[rem_mask]

    # 5. Candidate scoring & beam selection
    if last_logits is not None:
        # Build mass lookup table on device
        mass_table = torch.zeros(
            max(vocabulary.values()) + 1, device=device, dtype=torch.float32
        )
        for token, idx in vocabulary.items():
            if token in AA_MASSES_DICT:
                mass_table[idx] = AA_MASSES_DICT[token]

        # Mass error: |sum m(aa) - (M_prec - M_H2O)|
        seq_masses = (mass_table[x_t] * active_mask.float()).sum(dim=-1)
        target_residue_mass = (precursor_mass_exp - M_H2O).clamp(min=0.0)
        delta_m = (seq_masses - target_residue_mass).abs()

        # Relative PPM mass error (in 100 ppm units)
        relative_mass_error = (delta_m / target_residue_mass.clamp(min=1.0)) * 1e4

        # Spectral log-likelihood: mean_{i} log P(Y_i | Spectrum)
        log_probs = F.log_softmax(last_logits, dim=-1)
        token_indices = x_t.clamp(min=0, max=last_logits.shape[-1] - 1)
        token_log_probs = log_probs.gather(
            dim=-1, index=token_indices.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * active_mask.float()
        mean_log_prob = token_log_probs.sum(dim=-1) / cand_lengths_flat.float().clamp(min=1.0)

        # Explained spectral intensity from theoretical b/y fragment matching
        if beta > 0.0:
            frag_score = compute_fragment_matching_scores(
                x_t=x_t,
                active_mask=active_mask,
                mz_array_exp=mz_array_exp,
                intensity_array_exp=intensity_array_exp,
                spectrum_mask_exp=spectrum_mask_exp,
                vocabulary=vocabulary,
                tolerance_ppm=tolerance_ppm,
                tolerance_da=tolerance_da,
            )
        else:
            frag_score = torch.zeros(total_samples, device=device)

        # Optional enzymatic C-terminal cleavage prior (Trypsin: C-terminal K or R)
        if trypsin_prior:
            last_pos = (cand_lengths_flat - 1).clamp(min=0)
            last_tokens = x_t.gather(dim=-1, index=last_pos.unsqueeze(-1)).squeeze(-1)
            k_id = vocabulary.get("K", -1)
            r_id = vocabulary.get("R", -1)
            trypsin_bonus = ((last_tokens == k_id) | (last_tokens == r_id)).float() * 0.2
        else:
            trypsin_bonus = 0.0

        # Bayesian Joint Posterior Score:
        # log P(L | S) + mean_i log P(Y_i | S, L) - alpha * mass_error + beta * frag_score + prior
        score = (
            cand_length_log_probs
            + mean_log_prob
            - alpha * relative_mass_error
            + beta * frag_score
            + trypsin_bonus
        )
    else:
        score = torch.zeros(total_samples, device=device)

    if K > 1:
        score_2d = score.view(batch_size, K)
        best_k = score_2d.argmax(dim=-1)  # (B,)

        batch_idx = torch.arange(batch_size, device=device)
        best_indices = batch_idx * K + best_k

        selected_x_t = x_t[best_indices]
        selected_lengths = cand_lengths_flat[best_indices]
        selected_scores = score_2d.gather(dim=-1, index=best_k.unsqueeze(-1)).squeeze(-1)
    else:
        selected_x_t = x_t
        selected_lengths = cand_lengths_flat
        selected_scores = score

    sequences = decode_tokens(selected_x_t, vocabulary)
    if return_scores:
        return selected_x_t, selected_lengths, sequences, selected_scores
    return selected_x_t, selected_lengths, sequences
