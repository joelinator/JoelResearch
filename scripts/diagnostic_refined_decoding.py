#!/usr/bin/env python3
"""
Diagnostic benchmark comparing:
1. Baseline Decoding (simultaneous unmasking, eta=0.0, raw confidence)
2. Refined Decoding:
   - Sequential Mass Budget Resolution (exact one-by-one reachability updates)
   - Peak-Evidence Unmasking Schedule (boost confidence of peak-supported positions)
   - Detailed Balance Error Correction (eta=0.15)
Evaluates on 2,000 HC-PT test spectra.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from data.data import build_dataloader, build_vocabulary, get_dataset
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint
from inference.knapsack_dp import ExactReachabilityDP
from inference.predict import (
    knapsack_filter_logits_vectorized,
    apply_length_padding,
    compute_terminal_prior,
    compute_fragment_matching_scores,
    M_H, M_H2O, AA_MASSES_DICT,
)
from flow_matching.sampling import inference_sample_mask
from flow_matching.scheduler import get_scheduler
from eval.metrics import compute_denovo_metrics, DenovoMetrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_sequential_knapsack(
    x_t: torch.Tensor,
    logits: torch.Tensor,
    active_mask: torch.Tensor,
    target_residue_mass: torch.Tensor,
    mask_id: int,
    mass_table: torch.Tensor,
    full_mass_table: torch.Tensor,
    tol: float = 1.0,
    reachability_dp: ExactReachabilityDP | None = None,
    use_exact_dp: bool = True,
    valid_aa_masses: torch.Tensor | None = None,
    pairs_2: torch.Tensor | None = None,
    min_aa_mass: float = 57.021464,
    max_aa_mass: float = 186.079313,
) -> torch.Tensor:
    x_res = x_t.clone()
    batch_size = x_res.shape[0]
    batch_indices = torch.arange(batch_size, device=x_res.device)

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

def compute_peak_evidence_boost(
    x_1: torch.Tensor,
    active_mask: torch.Tensor,
    target_residue_mass_exp: torch.Tensor,
    mz_array_exp: torch.Tensor,
    spectrum_mask_exp: torch.Tensor,
    mass_table: torch.Tensor,
    tolerance_da: float = 0.05,
    boost_weight: float = 0.35,
) -> torch.Tensor:
    token_mass = mass_table[x_1] * active_mask.float()
    b_ions = token_mass.cumsum(dim=-1) + M_H  # (B, L)
    total_mass = target_residue_mass_exp.unsqueeze(-1)
    y_ions = total_mass - (b_ions - M_H) + M_H2O + M_H  # (B, L)

    # Check against experimental peaks (B, S)
    valid_peaks = ~spectrum_mask_exp
    # Downsample peak count if needed for speed: take top 100 peaks
    mz_sub = mz_array_exp[:, :120]
    valid_sub = valid_peaks[:, :120]

    b_diff = (mz_sub.unsqueeze(1) - b_ions.unsqueeze(-1)).abs() <= tolerance_da
    b_match = (b_diff & valid_sub.unsqueeze(1)).any(dim=-1)

    y_diff = (mz_sub.unsqueeze(1) - y_ions.unsqueeze(-1)).abs() <= tolerance_da
    y_match = (y_diff & valid_sub.unsqueeze(1)).any(dim=-1)

    has_peak = (b_match | y_match) & active_mask
    boost = torch.where(has_peak, boost_weight, 0.0)
    return boost

@torch.no_grad()
def evaluate_custom(loader, vocab, enc, lp, dec, guid, device, use_refinements=False, max_samples=2000, eta=0.0):
    enc.eval(); lp.eval(); dec.eval(); guid.eval()
    scheduler = get_scheduler("cosine")
    reachability_dp = ExactReachabilityDP.get_default(device=device, tol_da=1.0)

    pad_id = vocab["<pad>"]
    mask_token_id = vocab.get("<mask_token>", vocab.get("<mask" + ">"))
    num_classes = dec.head.weight.shape[0] if hasattr(dec, "head") else 30
    mass_table = torch.zeros(num_classes, device=device, dtype=torch.float32)
    for tok, idx in vocab.items():
        if tok in AA_MASSES_DICT and idx < num_classes:
            mass_table[idx] = AA_MASSES_DICT[tok]
    full_mass_table = torch.zeros(max(vocab.values()) + 1, device=device, dtype=torch.float32)
    for tok, idx in vocab.items():
        if tok in AA_MASSES_DICT:
            full_mass_table[idx] = AA_MASSES_DICT[tok]

    preds = []
    targets = []
    scores = []

    count = 0
    t0 = time.time()
    for batch in loader:
        (mz_array, intensity_array, precursor_mass, precursor_charge, sequence,
         mz_complementary, length, padded_mask, spectrum_mask) = tuple(
            tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in batch
        )
        batch_size = mz_array.shape[0]

        spectrum_emb_cls, spectrum_emb_peaks, peak_mask = enc(
            mz_array, mz_complementary, intensity_array, spectrum_mask
        )
        length_logits = lp(spectrum_emb_cls, precursor_mass, precursor_charge)
        topk = torch.topk(length_logits, k=3, dim=-1)
        length_classes = topk.indices
        cand_lengths = length_classes + 7
        cand_lengths_flat = cand_lengths.reshape(-1)

        K = 3
        cand_length_log_probs = F.log_softmax(length_logits, dim=-1).gather(-1, length_classes).reshape(-1)
        precursor_mass_exp = precursor_mass.repeat_interleave(K, dim=0)
        precursor_charge_exp = precursor_charge.repeat_interleave(K, dim=0)
        spectrum_emb_peaks_exp = spectrum_emb_peaks.repeat_interleave(K, dim=0)
        peak_mask_exp = peak_mask.repeat_interleave(K, dim=0)
        mz_array_exp = mz_array.repeat_interleave(K, dim=0)
        intensity_array_exp = intensity_array.repeat_interleave(K, dim=0)
        spectrum_mask_exp = spectrum_mask.repeat_interleave(K, dim=0)

        total_samples = precursor_mass_exp.shape[0]
        max_len = 30
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(total_samples, -1)
        active_mask = positions < cand_lengths_flat.unsqueeze(1)
        target_residue_mass_exp = precursor_mass_exp - M_H2O

        x_t = torch.where(active_mask, torch.tensor(mask_token_id, device=device), torch.tensor(pad_id, device=device))
        cond_conditioner = guid(spectrum_emb_peaks_exp, guidance_prob=0.0, need_guidance=False)
        seq_padding_mask = ~active_mask

        num_steps = 20
        last_logits = None
        for step in range(num_steps):
            t_scalar = step / num_steps
            delta_t = 1.0 / num_steps
            t = torch.full((total_samples,), t_scalar, device=device)
            kt, kt_derivative = scheduler(t)
            t_next = torch.full((total_samples,), min(1.0, (step + 1) / num_steps), device=device)
            kt_next, _ = scheduler(t_next)

            logits = dec(
                t, precursor_mass_exp, precursor_charge_exp,
                cond_conditioner, x_t, cand_lengths_flat, peak_mask_exp, seq_padding_mask
            )
            last_logits = logits
            logits = knapsack_filter_logits_vectorized(
                logits=logits, x_t=x_t, target_residue_mass=target_residue_mass_exp,
                active_mask=active_mask, mask_id=mask_token_id, mass_table=mass_table,
                full_mass_table=full_mass_table, tol=1.0, reachability_dp=reachability_dp,
                use_exact_dp=True
            )

            # Peak evidence schedule boost
            if use_refinements:
                probs = F.softmax(logits, dim=-1)
                x_1_cand = logits.argmax(dim=-1)
                boost = compute_peak_evidence_boost(
                    x_1=x_1_cand, active_mask=active_mask,
                    target_residue_mass_exp=target_residue_mass_exp,
                    mz_array_exp=mz_array_exp, spectrum_mask_exp=spectrum_mask_exp,
                    mass_table=mass_table,
                )
            else:
                boost = None

            x_t = inference_sample_mask(
                kt, kt_derivative, x_t, logits, vocab, delta_t,
                active_mask=active_mask, temperature=0.0, strategy="confidence",
                kt_next=kt_next, eta=eta, is_final_step=(step == num_steps - 1)
            )
            x_t = apply_length_padding(x_t, cand_lengths_flat, pad_id)

        # Final unmasking safety
        if use_refinements:
            x_t = resolve_sequential_knapsack(
                x_t=x_t, logits=last_logits, active_mask=active_mask,
                target_residue_mass=target_residue_mass_exp, mask_id=mask_token_id,
                mass_table=mass_table, full_mass_table=full_mass_table, tol=1.0,
                reachability_dp=reachability_dp, use_exact_dp=True
            )
        else:
            rem_mask = (x_t == mask_token_id) & active_mask
            if rem_mask.any():
                final_logits = knapsack_filter_logits_vectorized(
                    logits=last_logits.clone(), x_t=x_t, target_residue_mass=target_residue_mass_exp,
                    active_mask=active_mask, mask_id=mask_token_id, mass_table=mass_table,
                    full_mass_table=full_mass_table, tol=1.0, reachability_dp=reachability_dp,
                    use_exact_dp=True
                )
                x_t[rem_mask] = final_logits.argmax(dim=-1)[rem_mask]

        # Candidate scoring
        log_probs = F.log_softmax(last_logits, dim=-1)
        token_indices = x_t.clamp(min=0, max=last_logits.shape[-1] - 1)
        token_log_probs = log_probs.gather(dim=-1, index=token_indices.unsqueeze(-1)).squeeze(-1) * active_mask.float()
        mean_log_prob = token_log_probs.sum(dim=-1) / cand_lengths_flat.float().clamp(min=1.0)
        seq_masses = (full_mass_table[x_t.clamp(0, len(full_mass_table) - 1)] * active_mask.float()).sum(dim=-1)
        delta_m = (seq_masses - target_residue_mass_exp).abs()
        relative_mass_error = (delta_m / target_residue_mass_exp.clamp(min=1.0)) * 1e4

        frag_score = compute_fragment_matching_scores(
            x_t=x_t, active_mask=active_mask, mz_array_exp=mz_array_exp,
            intensity_array_exp=intensity_array_exp, spectrum_mask_exp=spectrum_mask_exp,
            vocabulary=vocab, tolerance_ppm=20.0, tolerance_da=0.05, use_composite_ladders=True
        )
        terminal_bonus = compute_terminal_prior(
            x_t=x_t, cand_lengths=cand_lengths_flat, vocabulary=vocab,
            enzyme="trypsin", mz_array_exp=mz_array_exp, spectrum_mask_exp=spectrum_mask_exp,
            tolerance_da=0.05
        )
        score = cand_length_log_probs + mean_log_prob - 0.5 * relative_mass_error + 0.5 * frag_score + terminal_bonus

        score_matrix = score.view(batch_size, K)
        best_cand_idx = score_matrix.argmax(dim=-1)
        best_flat_idx = torch.arange(batch_size, device=device) * K + best_cand_idx

        best_tokens = x_t[best_flat_idx]
        best_lengths = cand_lengths_flat[best_flat_idx]
        best_scores = score[best_flat_idx]

        idx_to_aa = {v: k for k, v in vocab.items()}
        for i in range(batch_size):
            L_i = int(best_lengths[i].item())
            toks = best_tokens[i, :L_i].cpu().tolist()
            seq_str = "".join([idx_to_aa.get(t, "") for t in toks if t in idx_to_aa and idx_to_aa[t] not in ("<pad>", "<mask_token>", "</s>")])
            preds.append(seq_str)
            scores.append(float(best_scores[i].item()))
        targets.extend(sequence)

        count += batch_size
        if count >= max_samples:
            break

    dt = time.time() - t0
    m = compute_denovo_metrics(preds[:max_samples], targets[:max_samples], scores[:max_samples])
    throughput = len(preds[:max_samples]) / dt
    return m, throughput, preds[:max_samples], targets[:max_samples]

if __name__ == "__main__":
    ckpt_path = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints" / "best-joint-gen-exact-epoch=01-exact=0.4746.ckpt"
    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)

    ds = get_dataset("InstaDeepAI/ms_proteometools", split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    print("--- 1. Baseline (Standard simultaneous unmasking, eta=0.0) ---")
    m1, tp1, p1, t1 = evaluate_custom(loader, vocab, enc, lp, dec, guid, DEVICE, use_refinements=False, max_samples=1500, eta=0.0)
    print(f"Baseline: Strict = {m1.exact_peptide_accuracy:.2%}, I/L = {m1.exact_peptide_accuracy_il:.2%}, AA-F1 = {m1.aa_f1:.2%}, Throughput = {tp1:.1f} spec/s")

    print("\n--- 2. Refined (Sequential mass budget + Peak evidence + Detailed balance eta=0.15) ---")
    m2, tp2, p2, t2 = evaluate_custom(loader, vocab, enc, lp, dec, guid, DEVICE, use_refinements=True, max_samples=1500, eta=0.15)
    print(f"Refined: Strict = {m2.exact_peptide_accuracy:.2%}, I/L = {m2.exact_peptide_accuracy_il:.2%}, AA-F1 = {m2.aa_f1:.2%}, Throughput = {tp2:.1f} spec/s")
