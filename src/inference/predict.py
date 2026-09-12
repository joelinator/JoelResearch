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
from flow_matching.scheduler import get_scheduler
from inference.knapsack_dp import ExactReachabilityDP


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
    use_composite_ladders: bool = False,
) -> torch.Tensor:
    """
    Compute the explained spectral intensity and ion-ladder continuity fraction
    for each candidate sequence by matching theoretical b- and y-ions against
    experimental spectrum peaks.

    Args:
        x_t: Candidate token sequences of shape (B_total, L).
        active_mask: Boolean tensor of shape (B_total, L) indicating active residues.
        mz_array_exp: Experimental peak m/z values of shape (B_total, S).
        intensity_array_exp: Peak intensities of shape (B_total, S).
        spectrum_mask_exp: True for padded/missing spectrum peaks (B_total, S).
        vocabulary: Token dictionary.
        tolerance_ppm: Mass tolerance in parts-per-million (default: 20.0).
        tolerance_da: Absolute mass tolerance in Daltons (default: 0.05).
        use_composite_ladders: If True, combines explained intensity, consecutive b/y ladders,
            theoretical coverage, ion balance, and high-intensity unexplained peak penalty.

    Returns:
        score: Tensor of shape (B_total,) in [0, 1].
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
    explained_intensity = (matched_intensity / total_spec_intensity).clamp(0.0, 1.0)

    if not use_composite_ladders:
        return explained_intensity

    # Multi-feature fragment scoring: consecutive ion ladders, theoretical coverage, ion balance
    theo_is_matched = matched.any(dim=1)  # (B_total, 2*L)
    L_seq = x_t.size(1)
    b_matched = theo_is_matched[:, :L_seq] & valid_fragment  # (B_total, L)
    y_matched = theo_is_matched[:, L_seq:] & valid_fragment  # (B_total, L)

    # 1. Consecutive ion ladder series (contiguous sequence evidence)
    max_pairs = (lengths - 2).clamp(min=1).float()
    b_ladder_ratio = (b_matched[:, :-1] & b_matched[:, 1:]).float().sum(dim=-1) / max_pairs
    y_ladder_ratio = (y_matched[:, :-1] & y_matched[:, 1:]).float().sum(dim=-1) / max_pairs
    ladder_score = ((b_ladder_ratio + y_ladder_ratio) / 2.0).clamp(0.0, 1.0)

    # 2. Theoretical ion coverage (fraction of theoretical b and y observed)
    total_theo = (2 * (lengths - 1)).float().clamp(min=1.0)
    theo_coverage = ((b_matched.sum(dim=-1) + y_matched.sum(dim=-1)).float() / total_theo).clamp(0.0, 1.0)

    # 3. Ion type balance (observing both b and y series)
    n_b = b_matched.sum(dim=-1).float()
    n_y = y_matched.sum(dim=-1).float()
    ion_balance = (1.0 - (n_b - n_y).abs() / (n_b + n_y + 1e-6)).clamp(0.0, 1.0)

    # 4. Unmatched high-intensity peaks penalty
    top_k = min(5, exp_intensity.size(-1))
    unmatched_peaks_int = exp_intensity * (~peak_is_matched).float()
    top_unmatched = unmatched_peaks_int.topk(k=top_k, dim=-1).values.sum(dim=-1) / total_spec_intensity
    unmatched_penalty = top_unmatched.clamp(0.0, 1.0)

    composite_score = (
        0.45 * explained_intensity
        + 0.20 * theo_coverage
        + 0.20 * ladder_score
        + 0.15 * ion_balance
        - 0.10 * unmatched_penalty
    ).clamp(0.0, 1.0)

    return composite_score


def compute_terminal_prior(
    x_t: torch.Tensor,
    cand_lengths: torch.Tensor,
    vocabulary: dict[str, int],
    enzyme: str | None = "trypsin",
    mz_array_exp: torch.Tensor | None = None,
    spectrum_mask_exp: torch.Tensor | None = None,
    tolerance_da: float = 0.05,
) -> torch.Tensor:
    """
    Compute dataset- and evidence-conditioned C-terminal cleavage prior.

    If enzyme == 'trypsin':
      - C-terminal residue must be Lysine (K) or Arginine (R).
      - If experimental y1 peak is observed in mz_array (y1(K) ~ 147.11 Da, y1(R) ~ 175.12 Da),
        awards an evidence bonus of 0.25.
      - If y1 peak is not observed, awards a baseline prior of 0.08.
    If enzyme is None or non-tryptic:
      - Returns 0.0 (no bias towards K/R).
    """
    total_samples = x_t.shape[0]
    device = x_t.device
    if enzyme is None or enzyme.lower() in ("none", "unspecific", "nonspecific"):
        return torch.zeros(total_samples, device=device)

    last_pos = (cand_lengths - 1).clamp(min=0)
    last_tokens = x_t.gather(dim=-1, index=last_pos.unsqueeze(-1)).squeeze(-1)
    k_id = vocabulary.get("K", -1)
    r_id = vocabulary.get("R", -1)
    is_tryptic_terminus = (last_tokens == k_id) | (last_tokens == r_id)

    if enzyme.lower() == "trypsin":
        if mz_array_exp is not None and spectrum_mask_exp is not None:
            # Physical y1 theoretical masses (Da)
            y1_k = 147.1128
            y1_r = 175.1189
            # Check for y1 peak in unmasked peaks
            valid_peaks = ~spectrum_mask_exp
            mz = mz_array_exp
            has_y1_k = (((mz - y1_k).abs() <= tolerance_da) & valid_peaks).any(dim=-1)
            has_y1_r = (((mz - y1_r).abs() <= tolerance_da) & valid_peaks).any(dim=-1)
            has_y1_evidence = ((last_tokens == k_id) & has_y1_k) | ((last_tokens == r_id) & has_y1_r)
            bonus = torch.where(has_y1_evidence, 0.25, torch.where(is_tryptic_terminus, 0.08, 0.0))
            return bonus.float()
        else:
            return is_tryptic_terminus.float() * 0.15

    return torch.zeros(total_samples, device=device)


def knapsack_filter_logits_vectorized(
    logits: torch.Tensor,
    x_t: torch.Tensor,
    target_residue_mass: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int,
    mass_table: torch.Tensor,
    full_mass_table: torch.Tensor,
    tol: float = 1.0,
    valid_aa_masses: torch.Tensor | None = None,
    pairs_2: torch.Tensor | None = None,
    min_aa_mass: float = 57.021464,
    max_aa_mass: float = 186.079313,
    reachability_dp: ExactReachabilityDP | None = None,
    use_exact_dp: bool = True,
) -> torch.Tensor:
    """
    Multi-step dynamic precursor mass-budget (knapsack) constraint for discrete flow matching.

    Supports exact dynamic programming reachability table (replaces coarse interval bounds
    with exact reachability across all remaining lengths K >= 1), or fallback 4-stage checks.
    """
    is_masked = (x_t == mask_id) & active_mask
    num_masked = is_masked.sum(dim=-1, keepdim=True)  # (B_total, 1)

    current_mass = (
        full_mass_table[x_t.clamp(0, len(full_mass_table) - 1)]
        * (~is_masked & active_mask).float()
    ).sum(dim=-1, keepdim=True)  # (B_total, 1)

    rem_mass = target_residue_mass.unsqueeze(-1) - current_mass  # (B_total, 1)

    # Exact Reachable-Mass Dynamic Programming Table
    if use_exact_dp:
        if reachability_dp is None:
            reachability_dp = ExactReachabilityDP.get_default(device=logits.device, tol_da=tol)
        else:
            reachability_dp = reachability_dp.to(logits.device)

        k_next = (num_masked - 1).clamp(min=0)
        R_next = rem_mass.unsqueeze(-1) - mass_table.view(1, 1, -1)
        valid_tokens = reachability_dp.is_reachable(k_next.unsqueeze(-1), R_next) & (mass_table.view(1, 1, -1) > 0)
        has_valid = valid_tokens.any(dim=-1, keepdim=True)

        dist_to_budget = (R_next - (k_next * 110.0).unsqueeze(-1)).abs()
        modified_logits = torch.where(
            has_valid,
            torch.where(valid_tokens, logits, torch.full_like(logits, -1e9)),
            logits - dist_to_budget * 2.0,
        )
        return torch.where(is_masked.unsqueeze(-1), modified_logits, logits)

    # Fallback 4-stage interval bounds
    single_rem_mask = (num_masked == 1) & is_masked  # (B_total, L)
    if single_rem_mask.any():
        target_m = rem_mass.unsqueeze(-1)  # (B_total, 1, 1)
        mass_diffs = (mass_table.view(1, 1, -1) - target_m).abs()  # (B_total, 1, V)
        valid_aa = (mass_diffs <= tol) & (mass_table.view(1, 1, -1) > 0)  # (B_total, 1, V)
        has_valid = valid_aa.any(dim=-1, keepdim=True)  # (B_total, 1, 1)

        modified_logits = torch.where(
            has_valid,
            torch.where(valid_aa, logits, torch.full_like(logits, -1e9)),
            logits - mass_diffs * 5.0,
        )
        logits = torch.where(single_rem_mask.unsqueeze(-1), modified_logits, logits)

    two_rem_mask = (num_masked == 2) & is_masked  # (B_total, L)
    if two_rem_mask.any() and valid_aa_masses is not None:
        R = rem_mass.unsqueeze(-1) - mass_table.view(1, 1, -1)  # (B_total, 1, V)
        dist_to_single = (R.unsqueeze(-1) - valid_aa_masses.view(1, 1, 1, -1)).abs().min(dim=-1).values
        valid_2 = (dist_to_single <= tol) & (mass_table.view(1, 1, -1) > 0)
        has_valid_2 = valid_2.any(dim=-1, keepdim=True)
        mod_2 = torch.where(
            has_valid_2,
            torch.where(valid_2, logits, torch.full_like(logits, -1e9)),
            logits - dist_to_single * 5.0,
        )
        logits = torch.where(two_rem_mask.unsqueeze(-1), mod_2, logits)

    three_rem_mask = (num_masked == 3) & is_masked  # (B_total, L)
    if three_rem_mask.any() and pairs_2 is not None:
        R = rem_mass.unsqueeze(-1) - mass_table.view(1, 1, -1)  # (B_total, 1, V)
        dist_to_pair = (R.unsqueeze(-1) - pairs_2.view(1, 1, 1, -1)).abs().min(dim=-1).values
        valid_3 = (dist_to_pair <= tol) & (mass_table.view(1, 1, -1) > 0)
        has_valid_3 = valid_3.any(dim=-1, keepdim=True)
        mod_3 = torch.where(
            has_valid_3,
            torch.where(valid_3, logits, torch.full_like(logits, -1e9)),
            logits - dist_to_pair * 5.0,
        )
        logits = torch.where(three_rem_mask.unsqueeze(-1), mod_3, logits)

    multi_rem_mask = (num_masked >= 4) & is_masked  # (B_total, L)
    if multi_rem_mask.any():
        rem_count = (num_masked - 1).float()
        m_lower = rem_mass - rem_count * max_aa_mass - tol
        m_upper = rem_mass - rem_count * min_aa_mass + tol
        in_interval = (
            (mass_table.view(1, 1, -1) >= m_lower.unsqueeze(-1))
            & (mass_table.view(1, 1, -1) <= m_upper.unsqueeze(-1))
            & (mass_table.view(1, 1, -1) > 0)
        )
        has_interval = in_interval.any(dim=-1, keepdim=True)
        interval_penalty = (
            torch.clamp(m_lower.unsqueeze(-1) - mass_table.view(1, 1, -1), min=0.0)
            + torch.clamp(mass_table.view(1, 1, -1) - m_upper.unsqueeze(-1), min=0.0)
        )
        mod_multi = torch.where(
            has_interval,
            torch.where(in_interval, logits, torch.full_like(logits, -1e9)),
            logits - interval_penalty * 5.0,
        )
        logits = torch.where(multi_rem_mask.unsqueeze(-1), mod_multi, logits)

    return logits


def resolve_sequential_knapsack(
    x_t: torch.Tensor,
    logits: torch.Tensor,
    active_mask: torch.Tensor,
    target_residue_mass: torch.Tensor,
    mask_id: int,
    mass_table: torch.Tensor,
    full_mass_table: torch.Tensor,
    tol: float = 1.0,
    valid_aa_masses: torch.Tensor | None = None,
    pairs_2: torch.Tensor | None = None,
    min_aa_mass: float = 57.021464,
    max_aa_mass: float = 186.079313,
    reachability_dp: ExactReachabilityDP | None = None,
    use_exact_dp: bool = True,
) -> torch.Tensor:
    """
    Sequentially resolves residual masked positions one-by-one with exact DP reachability updates.
    Prevents joint knapsack collisions when two or more positions are unmasked in the final step.
    """
    x_res = x_t.clone()
    batch_size = x_res.shape[0]
    batch_indices = torch.arange(batch_size, device=x_res.device)

    # Loop at most max_len times (typically 1-3 iterations)
    for _ in range(x_res.shape[1]):
        is_masked = (x_res == mask_id) & active_mask
        if not is_masked.any():
            break

        filtered_logits = knapsack_filter_logits_vectorized(
            logits=logits.clone(),
            x_t=x_res,
            target_residue_mass=target_residue_mass,
            active_mask=active_mask,
            mask_id=mask_id,
            mass_table=mass_table,
            full_mass_table=full_mass_table,
            tol=tol,
            valid_aa_masses=valid_aa_masses,
            pairs_2=pairs_2,
            min_aa_mass=min_aa_mass,
            max_aa_mass=max_aa_mass,
            reachability_dp=reachability_dp,
            use_exact_dp=use_exact_dp,
        )

        conf = filtered_logits.max(dim=-1).values
        conf_masked = torch.where(is_masked, conf, torch.full_like(conf, -1e9))

        best_pos = conf_masked.argmax(dim=-1)
        has_masked = is_masked.any(dim=-1)

        active_b = batch_indices[has_masked]
        active_pos = best_pos[has_masked]

        chosen_tokens = filtered_logits[active_b, active_pos].argmax(dim=-1)
        x_res[active_b, active_pos] = chosen_tokens

    return x_res


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
    num_steps: int = 25,
    noising_scheme: str = "mask",
    guidance_scale: float = 1.8,
    top_k_lengths: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    decoding_strategy: str = "confidence",
    temperature: float = 0.0,
    trypsin_prior: bool = True,
    tolerance_ppm: float = 20.0,
    tolerance_da: float = 0.05,
    mask_self_attention: bool = True,
    return_scores: bool = False,
    use_knapsack_filter: bool = True,
    use_exact_dp_knapsack: bool = True,
    knapsack_tol_da: float = 1.0,
    reachability_dp: ExactReachabilityDP | None = None,
    num_samples_per_length: int = 1,
    eta: float = 0.0,
    enzyme: str | None = "trypsin",
    use_composite_ladders: bool = True,
    use_sequential_knapsack: bool = True,
    use_peak_evidence: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[str]] | tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """
    Run de novo inference with Top-k Length Beam Decoding, Dynamic Knapsack Filtering, and fragment ladder scoring.

    Predicts the top-k most likely peptide lengths from the length classifier,
    runs discrete flow matching integration with dynamic mass budget logit masking,
    and scores candidate sequences based on:
        Score(Y) = log P(L|S) + mean log P(Y_i|S,L) - alpha * MassError + beta * FragMatch + Prior
    and selects the optimal sequence per spectrum.
    """
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()
    scheduler = get_scheduler(scheduler)

    device = mz_array.device
    batch_size = mz_array.shape[0]
    pad_id = vocabulary["<pad>"]
    mask_token_id = vocabulary.get("<mask_token>", vocabulary.get("<mask" + ">"))

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

    S = max(1, int(num_samples_per_length))
    if S > 1:
        cand_lengths_flat = cand_lengths_flat.repeat_interleave(S, dim=0)
        cand_length_log_probs = cand_length_log_probs.repeat_interleave(S, dim=0)
        precursor_mass_exp = precursor_mass_exp.repeat_interleave(S, dim=0)
        precursor_charge_exp = precursor_charge_exp.repeat_interleave(S, dim=0)
        spectrum_emb_peaks_exp = spectrum_emb_peaks_exp.repeat_interleave(S, dim=0)
        peak_mask_exp = peak_mask_exp.repeat_interleave(S, dim=0)
        mz_array_exp = mz_array_exp.repeat_interleave(S, dim=0)
        intensity_array_exp = intensity_array_exp.repeat_interleave(S, dim=0)
        spectrum_mask_exp = spectrum_mask_exp.repeat_interleave(S, dim=0)
        K_total = K * S
    else:
        K_total = K

    total_samples = batch_size * K_total
    max_len = int(cand_lengths_flat.max().item())
    active_mask = length_to_active_mask(cand_lengths_flat, max_len)
    target_residue_mass_exp = (precursor_mass_exp - M_H2O).clamp(min=0.0)

    # Build mass lookup tables
    num_classes = decoder.head.weight.shape[0] if hasattr(decoder, "head") else 30
    mass_table = torch.zeros(num_classes, device=device, dtype=torch.float32)
    for token, idx in vocabulary.items():
        if token in AA_MASSES_DICT and idx < num_classes:
            mass_table[idx] = AA_MASSES_DICT[token]

    full_mass_table = torch.zeros(
        max(vocabulary.values()) + 1, device=device, dtype=torch.float32
    )
    for token, idx in vocabulary.items():
        if token in AA_MASSES_DICT:
            full_mass_table[idx] = AA_MASSES_DICT[token]

    # Precompute valid single and pairwise AA masses for multi-step knapsack guidance
    valid_masses_list = sorted(set([v for k, v in AA_MASSES_DICT.items() if len(k) == 1 and v > 0]))
    valid_aa_masses = torch.tensor(valid_masses_list, dtype=torch.float32, device=device)
    min_aa_mass = float(valid_aa_masses.min().item())
    max_aa_mass = float(valid_aa_masses.max().item())
    pairs_2 = (valid_aa_masses.unsqueeze(0) + valid_aa_masses.unsqueeze(1)).view(-1).unique()

    # Multi-candidate sampling temperatures: candidate 0 is greedy MAP, 1..S-1 are stochastic
    if S > 1:
        sample_idx = torch.arange(total_samples, device=device) % S
        sample_temperature = torch.where(
            sample_idx == 0,
            torch.tensor(temperature, device=device),
            torch.tensor(0.7 if temperature <= 0.05 else temperature, device=device),
        ).view(-1, 1)
    else:
        sample_temperature = temperature

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

    seq_padding_mask = ~active_mask if mask_self_attention else None

    # 4. Discrete flow matching reverse integration
    last_logits = None
    for step in range(num_steps):
        t_scalar = step / num_steps
        delta_t = 1.0 / num_steps
        t = torch.full((total_samples,), t_scalar, device=device)
        kt, kt_derivative = scheduler(t)
        t_next = torch.full((total_samples,), min(1.0, (step + 1) / num_steps), device=device)
        kt_next, _ = scheduler(t_next)

        if guidance_scale > 1.0 and uncond_conditioner is not None:
            cond_logits = decoder(
                t,
                precursor_mass_exp,
                precursor_charge_exp,
                cond_conditioner,
                x_t,
                cand_lengths_flat,
                peak_mask_exp,
                seq_padding_mask,
            )
            uncond_logits = decoder(
                t,
                precursor_mass_exp,
                precursor_charge_exp,
                uncond_conditioner,
                x_t,
                cand_lengths_flat,
                peak_mask_exp,
                seq_padding_mask,
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
                seq_padding_mask,
            )

        last_logits = logits
        if use_knapsack_filter:
            logits = knapsack_filter_logits_vectorized(
                logits=logits,
                x_t=x_t,
                target_residue_mass=target_residue_mass_exp,
                active_mask=active_mask,
                mask_id=mask_token_id,
                mass_table=mass_table,
                full_mass_table=full_mass_table,
                tol=knapsack_tol_da,
                valid_aa_masses=valid_aa_masses,
                pairs_2=pairs_2,
                min_aa_mass=min_aa_mass,
                max_aa_mass=max_aa_mass,
                reachability_dp=reachability_dp,
                use_exact_dp=use_exact_dp_knapsack,
            )

        confidence_bonus = None
        if use_peak_evidence and mz_array_exp is not None and noising_scheme == "mask":
            x_1_cand = logits.argmax(dim=-1)
            cand_masses = full_mass_table[x_1_cand.clamp(0, len(full_mass_table) - 1)] * active_mask.float()
            b_masses = cand_masses.cumsum(dim=-1) + M_H
            valid_peaks = ~spectrum_mask_exp

            sub_len = min(120, mz_array_exp.shape[1])
            mz_sub = mz_array_exp[:, :sub_len]
            valid_sub = valid_peaks[:, :sub_len]

            b_diff = (mz_sub.unsqueeze(1) - b_masses.unsqueeze(-1)).abs()
            b_match = (b_diff <= tolerance_da) & valid_sub.unsqueeze(1)
            has_b = b_match.any(dim=-1)

            y_masses = target_residue_mass_exp.unsqueeze(-1) - (b_masses - M_H) + M_H2O + M_H
            y_diff = (mz_sub.unsqueeze(1) - y_masses.unsqueeze(-1)).abs()
            y_match = (y_diff <= tolerance_da) & valid_sub.unsqueeze(1)
            has_y = y_match.any(dim=-1)

            confidence_bonus = torch.where((has_b | has_y) & active_mask, 0.25, 0.0)

        if noising_scheme == "mask":
            x_t = inference_sample_mask(
                kt,
                kt_derivative,
                x_t,
                logits,
                vocabulary,
                delta_t,
                active_mask=active_mask,
                temperature=sample_temperature,
                strategy=decoding_strategy,
                kt_next=kt_next,
                eta=eta,
                is_final_step=(step == num_steps - 1),
                confidence_bonus=confidence_bonus,
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

    # Final unmasking safety: if any active position is still masked, unmask via sequential knapsack
    mask_token_id = vocabulary.get("<mask_token>", vocabulary.get("<mask" + ">"))
    if mask_token_id is not None and last_logits is not None:
        rem_mask = (x_t == mask_token_id) & active_mask
        if rem_mask.any():
            if use_sequential_knapsack:
                x_t = resolve_sequential_knapsack(
                    x_t=x_t,
                    logits=last_logits,
                    active_mask=active_mask,
                    target_residue_mass=target_residue_mass_exp,
                    mask_id=mask_token_id,
                    mass_table=mass_table,
                    full_mass_table=full_mass_table,
                    tol=knapsack_tol_da,
                    valid_aa_masses=valid_aa_masses,
                    pairs_2=pairs_2,
                    min_aa_mass=min_aa_mass,
                    max_aa_mass=max_aa_mass,
                    reachability_dp=reachability_dp,
                    use_exact_dp=use_exact_dp_knapsack,
                )
            elif use_knapsack_filter:
                final_logits = knapsack_filter_logits_vectorized(
                    logits=last_logits.clone(),
                    x_t=x_t,
                    target_residue_mass=target_residue_mass_exp,
                    active_mask=active_mask,
                    mask_id=mask_token_id,
                    mass_table=mass_table,
                    full_mass_table=full_mass_table,
                    tol=knapsack_tol_da,
                    valid_aa_masses=valid_aa_masses,
                    pairs_2=pairs_2,
                    min_aa_mass=min_aa_mass,
                    max_aa_mass=max_aa_mass,
                    reachability_dp=reachability_dp,
                    use_exact_dp=use_exact_dp_knapsack,
                )
                x_t[rem_mask] = final_logits.argmax(dim=-1)[rem_mask]
            else:
                x_t[rem_mask] = last_logits.argmax(dim=-1)[rem_mask]

    # 5. Candidate scoring & beam selection
    if last_logits is not None:
        # Mass error: |sum m(aa) - (M_prec - M_H2O)|
        seq_masses = (
            full_mass_table[x_t.clamp(0, len(full_mass_table) - 1)] * active_mask.float()
        ).sum(dim=-1)
        target_residue_mass = target_residue_mass_exp
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
                use_composite_ladders=use_composite_ladders,
            )
        else:
            frag_score = torch.zeros(total_samples, device=device)

        # Optional enzymatic C-terminal cleavage prior (Evidence-conditioned)
        terminal_bonus = compute_terminal_prior(
            x_t=x_t,
            cand_lengths=cand_lengths_flat,
            vocabulary=vocabulary,
            enzyme=enzyme if enzyme is not None else ("trypsin" if trypsin_prior else None),
            mz_array_exp=mz_array_exp,
            spectrum_mask_exp=spectrum_mask_exp,
            tolerance_da=tolerance_da,
        )

        # Bayesian Joint Posterior Score:
        # log P(L | S) + mean_i log P(Y_i | S, L) - alpha * mass_error + beta * frag_score + prior
        score = (
            cand_length_log_probs
            + mean_log_prob
            - alpha * relative_mass_error
            + beta * frag_score
            + terminal_bonus
        )
    else:
        score = torch.zeros(total_samples, device=device)

    if K_total > 1:
        score_2d = score.view(batch_size, K_total)
        best_k = score_2d.argmax(dim=-1)  # (B,)

        batch_idx = torch.arange(batch_size, device=device)
        best_indices = batch_idx * K_total + best_k

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
