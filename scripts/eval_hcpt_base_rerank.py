#!/usr/bin/env python3
"""
Multi-Candidate Beam Reranking Evaluation for Un-Finetuned HC-PT Pretrained Model.

Evaluates the pure HC-PT pre-trained checkpoint (artifacts/canonical_reproduction/stage3_stratified/checkpoints/best-hcpt-stratified-epoch=04-exact=0.3799.ckpt)
across:
  1. The full HC-PT test split (InstaDeepAI/ms_proteometools, split='test', N=265,369)
  2. The full Nine-Species test split (InstaDeepAI/ms_ninespecies_benchmark, split='test', N=111,029)

Decoding Configuration:
  - Beam: Top-K lengths K=5, S=2 candidate trajectories per length (Total 10 candidates per spectrum)
  - Flow: Continuous-time discrete flow matching, 20 cosine steps, knapsack tolerance 0.5 Da
  - Candidates: Candidate 0 is greedy MAP (T=0.0), Candidates 1..9 explore stochastic trajectories (T=0.5)

Reranking Formulations Evaluated:
  1. Raw Greedy MAP (Candidate 0, No Rerank)
  2. Baseline Top-1 (Default Linear Combination)
  3. Pure Fragment Ladder Match (Highest FragScore)
  4. Pure Sequence Likelihood (Highest Mean LogProb)
  5. Composite Ladder (Beta=2.0, LengthGamma=0.2)
  6. Composite Ladder (Beta=4.0, LengthGamma=0.1)
  7. Strict Mass Gated (PPM <= 20) + Frag Dominant (Beta=3.0)
  8. Strict Mass Gated (PPM <= 20) + Frag Dominant (Beta=5.0)
  9. Frontier Calibrated Setup (gamma=0.85, w_len=0.5, b=3.5, a=16, term=8)
 10. Learned Linear ListNet Reranker (Permutation Softmax Cross-Entropy)
 11. Learned Deep MLP Reranker (Non-Linear Feature Interaction)
 12. Oracle Candidate Ceiling (Theoretical Upper Bound)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from config.defaults import DEFAULTS
from data.constants import AA_MASSES_DICT, M_H, M_H2O
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.metrics import compute_denovo_metrics
from inference.knapsack_dp import ExactReachabilityDP
from inference.predict import decode_tokens, predict_peptide
from inference.reranker import CandidateReranker, extract_candidate_features
from train.factory import build_models
from train.io import infer_model_config_from_checkpoint, load_checkpoint, load_models_from_checkpoint


def il_match(pred: str, target: str) -> bool:
    """Check isobaric I/L equality."""
    return pred.replace("L", "I") == target.replace("L", "I")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Candidate Beam Reranking Evaluation for HC-PT Model.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/canonical_reproduction/stage3_stratified/checkpoints/best-hcpt-stratified-epoch=04-exact=0.3799.ckpt",
        help="Path to pure HC-PT model checkpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        default="InstaDeepAI/ms_proteometools",
        help="Dataset repository name (InstaDeepAI/ms_proteometools or InstaDeepAI/ms_ninespecies_benchmark).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Cache directory for datasets.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of test spectra to evaluate (-1 for full split).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=5,
        help="Number of candidate lengths from length predictor (default: 5).",
    )
    parser.add_argument(
        "--num-samples-per-length",
        type=int,
        default=2,
        help="Number of stochastic trajectories per length (default: 2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for stochastic paths (default: 0.5).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Flow matching reverse integration steps (default: 20).",
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        help="Noise scheduler for flow matching (default: cosine).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0).",
    )
    parser.add_argument(
        "--knapsack-tol-da",
        type=float,
        default=0.5,
        help="Dynamic knapsack DP mass tolerance in Da (default: 0.5).",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/eval_beam_rerank_hcpt_base/results.json",
        help="Output path for evaluation JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print("\n" + "=" * 90, flush=True)
    print("MULTI-CANDIDATE BEAM RERANKING EVALUATION (HC-PT PRETRAINED UN-FINETUNED)", flush=True)
    print("=" * 90, flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Dataset: {args.dataset_name} ({args.split} split)", flush=True)
    print(f"Beam: K={args.top_k_lengths} lengths x S={args.num_samples_per_length} samples = {args.top_k_lengths * args.num_samples_per_length} candidates/spectrum", flush=True)
    print(f"Flow: Steps={args.num_steps}, Scheduler={args.scheduler}, Knapsack Tol={args.knapsack_tol_da} Da, Temp={args.temperature}", flush=True)
    print(f"Evaluation Target: {'FULL SPLIT' if args.num_samples <= 0 else f'{args.num_samples:,}'} spectra (batch_size={args.batch_size})", flush=True)

    # 1. Load Checkpoint & Architecture
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"\n[1/4] Loading checkpoint from {ckpt_path}...", flush=True)
    ckpt = load_checkpoint(str(ckpt_path), map_location=device)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    m_cfg = infer_model_config_from_checkpoint(ckpt)
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocab, device, model_cfg=m_cfg)
    load_models_from_checkpoint(ckpt, spectrum_encoder, length_predictor, decoder, guidance, use_ema=True)

    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()

    # Precompute Exact Reachability DP table
    reachability_dp = ExactReachabilityDP.get_default(device=device, tol_da=args.knapsack_tol_da)

    # 2. Build Dataset & DataLoader
    print(f"\n[2/4] Loading dataset: {args.dataset_name} ({args.split} split)...", flush=True)
    ds = get_dataset(repo_id=args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    total_available = len(ds)
    eval_count = total_available if args.num_samples <= 0 else min(total_available, args.num_samples)
    print(f"  Dataset contains {total_available:,} spectra; evaluating {eval_count:,} spectra.", flush=True)

    loader = build_dataloader(
        ds,
        vocab,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader.dataset.top_k = DEFAULTS.data.top_k_peaks

    # 3. Inference Loop
    print(f"\n[3/4] Running multi-candidate inference across {eval_count:,} spectra...", flush=True)
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    all_targets: list[str] = []
    all_target_lengths: list[int] = []
    all_precursor_masses: list[float] = []

    candidate_records: list[dict] = []

    t0 = time.time()
    processed_count = 0
    total_candidates_per_spectrum = args.top_k_lengths * args.num_samples_per_length
    running_oracle_matches = 0
    running_base_matches = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if processed_count >= eval_count:
                break

            (
                mz_arr,
                int_arr,
                p_mass,
                p_chg,
                seq,
                mz_comp,
                length,
                _pad_mask,
                spec_mask,
            ) = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)

            b_size = mz_arr.shape[0]
            if processed_count + b_size > eval_count:
                rem = eval_count - processed_count
                mz_arr = mz_arr[:rem]
                int_arr = int_arr[:rem]
                p_mass = p_mass[:rem]
                p_chg = p_chg[:rem]
                seq = seq[:rem]
                mz_comp = mz_comp[:rem]
                length = length[:rem]
                spec_mask = spec_mask[:rem]
                b_size = rem

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                (
                    _sel_xt,
                    _sel_lens,
                    sel_seqs,
                    sel_scs,
                    cand_data,
                ) = predict_peptide(
                    mz_array=mz_arr,
                    intensity_array=int_arr,
                    precursor_mass=p_mass,
                    precursor_charge=p_chg,
                    mz_complementary=mz_comp,
                    spectrum_mask=spec_mask,
                    vocabulary=vocab,
                    spectrum_encoder=spectrum_encoder,
                    length_predictor=length_predictor,
                    decoder=decoder,
                    guidance=guidance,
                    scheduler=args.scheduler,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    top_k_lengths=args.top_k_lengths,
                    num_samples_per_length=args.num_samples_per_length,
                    temperature=args.temperature,
                    hybrid_beam=True,
                    knapsack_tol_da=args.knapsack_tol_da,
                    reachability_dp=reachability_dp,
                    use_knapsack_filter=True,
                    use_exact_dp_knapsack=True,
                    alpha=16.0,
                    beta=3.5,
                    length_weight=0.50,
                    terminal_weight=8.0,
                    return_scores=True,
                    return_all_candidates=True,
                )

            batch_targets = decode_tokens(seq, vocab)
            all_targets.extend(batch_targets)
            all_target_lengths.extend(int(v) for v in length.tolist())
            all_precursor_masses.extend(float(v) for v in p_mass.tolist())

            batch_cand_seqs = cand_data["sequences"]
            batch_cand_lengths = cand_data["cand_lengths"]
            batch_cand_length_lp = cand_data["cand_length_log_probs"]
            batch_mean_log_prob = cand_data["mean_log_prob"]
            batch_delta_m = cand_data["delta_m"]
            batch_frag_score = cand_data["frag_score"]
            batch_expl = cand_data.get("explained_intensity", torch.zeros_like(batch_frag_score))
            batch_cov = cand_data.get("theo_coverage", torch.zeros_like(batch_frag_score))
            batch_ladd = cand_data.get("ladder_score", torch.zeros_like(batch_frag_score))
            batch_bal = cand_data.get("ion_balance", torch.zeros_like(batch_frag_score))
            batch_unm = cand_data.get("unmatched_penalty", torch.zeros_like(batch_frag_score))
            batch_term_bonus = cand_data["terminal_bonus"]
            batch_default_scores = cand_data["scores"]

            for i in range(b_size):
                tgt = batch_targets[i]
                c_seqs = batch_cand_seqs[i]
                c_lens = [int(v) for v in batch_cand_lengths[i].tolist()]
                c_len_lp = [float(v) for v in batch_cand_length_lp[i].tolist()]
                c_mean_lp = [float(v) for v in batch_mean_log_prob[i].tolist()]
                c_dm = [float(v) for v in batch_delta_m[i].tolist()]
                c_frag = [float(v) for v in batch_frag_score[i].tolist()]
                c_expl = [float(v) for v in batch_expl[i].tolist()]
                c_cov = [float(v) for v in batch_cov[i].tolist()]
                c_ladd = [float(v) for v in batch_ladd[i].tolist()]
                c_bal = [float(v) for v in batch_bal[i].tolist()]
                c_unm = [float(v) for v in batch_unm[i].tolist()]
                c_term = [float(v) for v in batch_term_bonus[i].tolist()]
                c_def_sc = [float(v) for v in batch_default_scores[i].tolist()]

                p_m = float(p_mass[i].item())
                tgt_residue_mass = max(1.0, p_m - M_H2O)
                c_ppm = [(abs(dm) / tgt_residue_mass) * 1e6 for dm in c_dm]

                oracle_il = any(il_match(cs, tgt) for cs in c_seqs)
                base_match = il_match(sel_seqs[i], tgt)

                if oracle_il:
                    running_oracle_matches += 1
                if base_match:
                    running_base_matches += 1

                candidate_records.append({
                    "target": tgt,
                    "target_length": len(tgt),
                    "precursor_mass": p_m,
                    "candidates": c_seqs,
                    "cand_lengths": c_lens,
                    "cand_length_log_prob": c_len_lp,
                    "mean_log_prob": c_mean_lp,
                    "delta_m": c_dm,
                    "ppm_error": c_ppm,
                    "frag_score": c_frag,
                    "explained_intensity": c_expl,
                    "theo_coverage": c_cov,
                    "ladder_score": c_ladd,
                    "ion_balance": c_bal,
                    "unmatched_penalty": c_unm,
                    "term_bonus": c_term,
                    "default_score": c_def_sc,
                    "frontier_prediction": sel_seqs[i],
                    "oracle_il": oracle_il,
                })

            processed_count += b_size
            elapsed = time.time() - t0
            rate = processed_count / max(elapsed, 1e-4)
            if (b_idx + 1) % 25 == 0 or processed_count >= eval_count:
                cur_oracle = (running_oracle_matches / processed_count) * 100.0
                cur_base = (running_base_matches / processed_count) * 100.0
                print(
                    f"   [{processed_count:,}/{eval_count:,}] ({rate:.1f} spec/s) | "
                    f"Frontier Exact(I/L): {cur_base:.2f}% | Oracle Ceiling: {cur_oracle:.2f}%",
                    flush=True,
                )

    total_time = time.time() - t0
    n_total = len(candidate_records)
    overall_oracle_il = (running_oracle_matches / n_total) * 100.0
    print("\n" + "=" * 90, flush=True)
    print(f"CANDIDATE GENERATION COMPLETED ({n_total:,} spectra, {total_time:.1f}s, {n_total/total_time:.1f} spec/s)", flush=True)
    print(f"  >>> ORACLE EXACT MATCH (I/L) CEILING: {overall_oracle_il:.2f}% <<<", flush=True)
    print("=" * 90, flush=True)

    # 4. Feature Extraction & ListNet Training
    print("\n[4/4] Fitting ListNet Reranker & Evaluating All Strategies...", flush=True)
    features, match_il = extract_candidate_features(candidate_records)

    # Train / Val split (80/20) for learned rerankers
    split_idx = int(n_total * 0.8)
    tr_feat, val_feat = features[:split_idx], features[split_idx:]
    tr_il, val_il = match_il[:split_idx], match_il[split_idx:]

    print(f"  Train set: {split_idx:,} spectra | Val set: {n_total - split_idx:,} spectra", flush=True)
    print("  Fitting Linear ListNet model...", flush=True)
    lin_reranker = CandidateReranker(mode="linear")
    lin_stats = lin_reranker.fit_listnet(
        features=tr_feat,
        match_il=tr_il,
        val_features=val_feat,
        val_match_il=val_il,
        epochs=150,
        lr=0.02,
        device=str(device),
    )

    print("\nLearned Linear Feature Weights:", flush=True)
    for fname, fweight in lin_reranker.get_feature_weights().items():
        print(f"   {fname:<25}: {fweight:+.4f}", flush=True)

    print("\n  Fitting Deep MLP ListNet model (hidden_dim=32)...", flush=True)
    mlp_reranker = CandidateReranker(mode="mlp", hidden_dim=32)
    mlp_stats = mlp_reranker.fit_listnet(
        features=tr_feat,
        match_il=tr_il,
        val_features=val_feat,
        val_match_il=val_il,
        epochs=150,
        lr=0.005,
        device=str(device),
    )

    trained_rerankers = {
        "Learned Linear ListNet Reranker": lin_reranker,
        "Learned Deep MLP Reranker": mlp_reranker,
    }

    # Evaluate all candidate reranking configurations
    rerank_configs = [
        {
            "name": "Raw Greedy MAP (Candidate 0, No Rerank)",
            "fn": lambda r, idx: 0,
        },
        {
            "name": "Baseline Top-1 (Default Weights)",
            "fn": lambda r, idx: np.argmax(r["default_score"]),
        },
        {
            "name": "Pure Fragment Ladder Match (Highest FragScore)",
            "fn": lambda r, idx: np.argmax(r["frag_score"]),
        },
        {
            "name": "Pure Sequence Likelihood (Highest Mean LogProb)",
            "fn": lambda r, idx: np.argmax(r["mean_log_prob"]),
        },
        {
            "name": "Composite Ladder (Beta=2.0, LengthGamma=0.2)",
            "fn": lambda r, idx: np.argmax(
                0.2 * np.array(r["cand_length_log_prob"])
                + 1.0 * np.array(r["mean_log_prob"])
                + 2.0 * np.array(r["frag_score"])
                + 1.0 * np.array(r["term_bonus"])
                - 0.05 * np.array(r["ppm_error"])
            ),
        },
        {
            "name": "Composite Ladder (Beta=4.0, LengthGamma=0.1)",
            "fn": lambda r, idx: np.argmax(
                0.1 * np.array(r["cand_length_log_prob"])
                + 1.0 * np.array(r["mean_log_prob"])
                + 4.0 * np.array(r["frag_score"])
                + 1.0 * np.array(r["term_bonus"])
                - 0.05 * np.array(r["ppm_error"])
            ),
        },
        {
            "name": "Strict Mass Gated (PPM <= 20) + Frag Dominant (Beta=3.0)",
            "fn": lambda r, idx: np.argmax(
                np.where(np.array(r["ppm_error"]) <= 20.0, 0.0, -100.0)
                + 0.15 * np.array(r["cand_length_log_prob"])
                + 1.0 * np.array(r["mean_log_prob"])
                + 3.0 * np.array(r["frag_score"])
                + 1.0 * np.array(r["term_bonus"])
            ),
        },
        {
            "name": "Strict Mass Gated (PPM <= 20) + Frag Dominant (Beta=5.0)",
            "fn": lambda r, idx: np.argmax(
                np.where(np.array(r["ppm_error"]) <= 20.0, 0.0, -100.0)
                + 0.10 * np.array(r["cand_length_log_prob"])
                + 0.8 * np.array(r["mean_log_prob"])
                + 5.0 * np.array(r["frag_score"])
                + 1.0 * np.array(r["term_bonus"])
            ),
        },
        {
            "name": "Frontier Calibrated Setup (gamma=0.85, w_len=0.5, b=3.5, a=16, term=8)",
            "fn": lambda r, idx: np.argmax(
                0.50 * np.array(r["cand_length_log_prob"])
                + np.array(r["mean_log_prob"]) / (np.maximum(1, np.array(r["cand_lengths"])) ** 0.85)
                - 16.0 * (np.array(r["delta_m"]) / max(1.0, r["precursor_mass"]))
                + 3.5 * np.array(r["frag_score"])
                + 8.0 * np.array(r["term_bonus"])
                + np.where(np.array(r["ppm_error"]) <= 20.0, 0.0, -100.0)
            ),
        },
    ]

    for r_name, r_model in trained_rerankers.items():
        r_model.eval()
        sc_mat = r_model.score_numpy(features)
        rerank_configs.append({
            "name": r_name,
            "fn": (lambda sm: lambda r, idx: int(np.argmax(sm[idx])))(sc_mat),
            "model_obj": r_model,
        })

    # Add Oracle Ceiling
    rerank_configs.append({
        "name": "Oracle Candidate Ceiling",
        "fn": lambda r, idx: (
            next((j for j, cs in enumerate(r["candidates"]) if il_match(cs, r["target"])), 0)
        ),
    })

    evaluation_summary = {
        "checkpoint": str(ckpt_path),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_spectra": n_total,
        "oracle_exact_il": overall_oracle_il,
        "linear_weights": lin_reranker.get_feature_weights(),
        "rerank_results": [],
    }

    print(f"\n{'Reranking Formulation':<62} | {'Peptide Acc (I/L)':<18} | {'Strict Exact':<13} | {'Mass Match':<10} | {'AA Prec':<8} | {'AA Rec':<8} | {'AA F1':<8} | {'Len Acc':<8} | {'Val Acc (I/L)':<13}", flush=True)
    print("-" * 170, flush=True)

    best_exact_il = -1.0
    best_config_name = ""

    for cfg in rerank_configs:
        preds = []
        p_lens = []
        for idx, r in enumerate(candidate_records):
            best_idx = cfg["fn"](r, idx)
            pred_seq = r["candidates"][best_idx]
            preds.append(pred_seq)
            p_lens.append(len(pred_seq))

        metrics = compute_denovo_metrics(
            predictions=preds,
            targets=all_targets,
            predicted_lengths=p_lens,
            target_lengths=all_target_lengths,
        )

        val_exact_il = None
        if split_idx < n_total:
            val_preds = preds[split_idx:]
            val_targets = all_targets[split_idx:]
            val_p_lens = p_lens[split_idx:]
            val_t_lens = all_target_lengths[split_idx:]
            val_metrics = compute_denovo_metrics(
                predictions=val_preds,
                targets=val_targets,
                predicted_lengths=val_p_lens,
                target_lengths=val_t_lens,
            )
            val_exact_il = val_metrics.exact_peptide_accuracy_il

        res_entry = {
            "name": cfg["name"],
            "peptide_accuracy_il": metrics.exact_peptide_accuracy_il,
            "strict_exact": metrics.exact_peptide_accuracy,
            "mass_match": metrics.mass_peptide_accuracy,
            "aa_precision": metrics.aa_precision,
            "aa_recall": metrics.aa_recall,
            "aa_f1": metrics.aa_f1,
            "aa_error_rate": metrics.aa_error_rate,
            "length_accuracy": metrics.length_accuracy,
            "val_peptide_accuracy_il": val_exact_il,
        }
        evaluation_summary["rerank_results"].append(res_entry)

        val_str = f"{val_exact_il*100:>12.2f}%" if val_exact_il is not None else "N/A"
        print(
            f"{cfg['name']:<62} | "
            f"{metrics.exact_peptide_accuracy_il*100:>17.2f}% | "
            f"{metrics.exact_peptide_accuracy*100:>12.2f}% | "
            f"{metrics.mass_peptide_accuracy*100:>9.2f}% | "
            f"{metrics.aa_precision*100:>7.2f}% | "
            f"{metrics.aa_recall*100:>7.2f}% | "
            f"{metrics.aa_f1*100:>7.2f}% | "
            f"{metrics.length_accuracy*100:>7.2f}% | "
            f"{val_str}",
            flush=True,
        )

        if cfg["name"] != "Oracle Candidate Ceiling" and metrics.exact_peptide_accuracy_il > best_exact_il:
            best_exact_il = metrics.exact_peptide_accuracy_il
            best_config_name = cfg["name"]

    print("-" * 170, flush=True)
    print(f"Top Performing Deployable Strategy: '{best_config_name}' ({best_exact_il*100:.2f}% Exact I/L)", flush=True)
    print(f"Oracle Ceiling: {overall_oracle_il:.2f}% Exact I/L", flush=True)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"\nSaved full calibration results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
