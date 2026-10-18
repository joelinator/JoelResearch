#!/usr/bin/env python3
# Candidate beam decoding and ListNet reranking evaluation for DFlowNovo.
# Generates top-K length candidates with hybrid deterministic (T=0) + stochastic (T=0.7) paths.
# Scores candidates using mass spectrometry features:
# - Precursor mass error (ppm)
# - Fragment ion series continuity (b/y ladders)
# - Spectral intensity coverage
# - Decoder sequence log-likelihood
#
# NOTE: evaluates peptide accuracy with standard isobaric I/L equivalence.
# TODO: add GPU-batched feature extraction to speed up large evaluations.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

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
from inference.reranker import (
    CandidateReranker,
    extract_candidate_features,
)
from train.factory import build_models
from train.io import infer_model_config_from_checkpoint, load_checkpoint, load_models_from_checkpoint


def il_match(pred: str, target: str) -> bool:
    """Check isobaric I/L equality."""
    return pred.replace("L", "I") == target.replace("L", "I")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Candidate Beam Reranking Evaluation.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/continual_hcpt_massivekb_run/stage3_massivekb/checkpoints/best-exact-epoch=02-exact=0.5389.ckpt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        default="InstaDeepAI/ms_ninespecies_benchmark",
        help="Dataset repository name.",
    )
    parser.add_argument(
        "--species",
        default="",
        help="Optional individual species name from Nine-Species benchmark (e.g. mus_musculus, h_sapiens, candidatus_endoloripes, saccharomyces_cerevisiae, vigna_mungo, solanum_lycopersicum, apis_mellifera, methanosarcina_mazei, bacillus_subtilis).",
    )
    parser.add_argument(
        "--ipc-path",
        default="",
        help="Optional direct path to an Arrow/IPC dataset file.",
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
        default=2000,
        help="Number of test spectra to evaluate (-1 for full split).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
        default=3,
        help="Number of stochastic trajectories per length (default: 3). Total candidates = K * S.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for stochastic paths 1..S-1 (default: 0.7).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=15,
        help="Flow matching reverse integration steps (default: 15).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.8,
        help="Classifier-free guidance scale (default: 1.8).",
    )
    parser.add_argument(
        "--knapsack-tol-da",
        type=float,
        default=0.002,
        help="Dynamic knapsack DP mass tolerance in Da (default: 0.002).",
    )
    parser.add_argument(
        "--hybrid-beam",
        action="store_true",
        default=False,
        help="Use hybrid 6-candidate beam (Cand 0: Greedy T=0.0 L1, Cands 1-5: Stochastic T=0.7 L1, L2, L3).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.15,
        help="Detailed balance rate matrix parameter (default: 0.15).",
    )
    parser.add_argument(
        "--output-json",
        default="/dev/shm/beam_rerank_calibration.json",
        help="Output path for evaluation JSON.",
    )
    parser.add_argument(
        "--dump-records",
        default="",
        help="Optional path to dump candidate records (.pt).",
    )
    parser.add_argument(
        "--save-reranker",
        default="",
        help="Optional path to save trained reranker weights (.pt).",
    )
    parser.add_argument(
        "--trained-reranker",
        default="",
        help="Optional path to load pre-trained reranker weights (.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print("\n" + "=" * 90)
    print("MULTI-CANDIDATE BEAM RERANKING CALIBRATION")
    print("=" * 90)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Candidate Configuration: K={args.top_k_lengths} lengths x S={args.num_samples_per_length} samples = {args.top_k_lengths * args.num_samples_per_length} candidates/spectrum")
    print(f"Stochastic Temperature: {args.temperature} | Steps: {args.num_steps} | Knapsack Tol: {args.knapsack_tol_da} Da")
    print(f"Evaluation Size: {args.num_samples:,} spectra")

    # 1. Load Checkpoint & Architecture
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"\n[1/4] Loading checkpoint from {ckpt_path}...")
    ckpt = load_checkpoint(str(ckpt_path), map_location=device)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    m_cfg = infer_model_config_from_checkpoint(ckpt)
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocab, device, model_cfg=m_cfg)
    load_models_from_checkpoint(ckpt, spectrum_encoder, length_predictor, decoder, guidance, use_ema=True)

    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()

    # Reachability DP table
    reachability_dp = ExactReachabilityDP.get_default(device=device, tol_da=args.knapsack_tol_da)

    # 2. Build Dataset & DataLoader
    if args.species:
        species_name = args.species.lower().strip().replace(" ", "_").replace("-", "_")
        if not species_name.endswith(".ipc"):
            ipc_filename = f"{species_name}.ipc"
        else:
            ipc_filename = species_name
            species_name = species_name[:-4]

        ipc_local_dir = Path("/dev/shm/temp_species_cache")
        ipc_local_dir.mkdir(parents=True, exist_ok=True)
        local_ipc = ipc_local_dir / ipc_filename

        if not local_ipc.exists():
            # Check if downloaded in hf_tmp or nested
            alt_path = ipc_local_dir / "data" / "ninespecies_updated" / ipc_filename
            hf_tmp_path = Path("/dev/shm/hf_tmp/data/ninespecies_updated") / ipc_filename
            if alt_path.exists():
                local_ipc = alt_path
            elif hf_tmp_path.exists():
                local_ipc = hf_tmp_path
            else:
                print(f"\nDownloading species data for '{species_name}' from Hugging Face...")
                from huggingface_hub import hf_hub_download
                dl_path = hf_hub_download(
                    repo_id=args.dataset_name,
                    filename=f"data/ninespecies_updated/{ipc_filename}",
                    repo_type="dataset",
                    local_dir=str(ipc_local_dir),
                )
                local_ipc = Path(dl_path)

        print(f"\n[2/4] Loading species dataset: {species_name} from {local_ipc}...")
        from datasets import load_dataset as hf_load_dataset
        from data.data import filter_valid_peptide_lengths
        ds = hf_load_dataset("arrow", data_files={"test": str(local_ipc)}, split="test")
        ds = filter_valid_peptide_lengths(ds)
    elif args.ipc_path:
        print(f"\n[2/4] Loading dataset from IPC file: {args.ipc_path}...")
        from datasets import load_dataset as hf_load_dataset
        from data.data import filter_valid_peptide_lengths
        ds = hf_load_dataset("arrow", data_files={"test": args.ipc_path}, split="test")
        ds = filter_valid_peptide_lengths(ds)
    else:
        print(f"\n[2/4] Loading dataset: {args.dataset_name} ({args.split} split)...")
        ds = get_dataset(repo_id=args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    total_available = len(ds)
    eval_count = total_available if args.num_samples <= 0 else min(total_available, args.num_samples)
    print(f"  Dataset contains {total_available:,} spectra; evaluating first {eval_count:,} spectra.")

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
    print(f"\n[3/4] Running multi-candidate inference across {eval_count:,} spectra...")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    all_targets: list[str] = []
    all_target_lengths: list[int] = []
    all_precursor_masses: list[float] = []

    # Store full candidate pool info
    candidate_records: list[dict] = []

    t0 = time.time()
    processed_count = 0
    total_candidates_per_spectrum = args.top_k_lengths * args.num_samples_per_length

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

            # Limit last batch if needed
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
                    scheduler="linear",
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    top_k_lengths=args.top_k_lengths,
                    num_samples_per_length=args.num_samples_per_length,
                    temperature=args.temperature,
                    eta=args.eta,
                    hybrid_beam=args.hybrid_beam,
                    knapsack_tol_da=args.knapsack_tol_da,
                    reachability_dp=reachability_dp,
                    use_knapsack_filter=True,
                    use_exact_dp_knapsack=True,
                    use_composite_ladders=True,
                    return_scores=True,
                    return_all_candidates=True,
                )

            batch_targets = decode_tokens(seq, vocab)
            all_targets.extend(batch_targets)
            all_target_lengths.extend(int(v) for v in length.tolist())
            all_precursor_masses.extend(float(v) for v in p_mass.tolist())

            # Unpack candidate data
            batch_cand_seqs = cand_data["sequences"]  # List[List[str]] (B, M)
            batch_cand_lengths = cand_data["cand_lengths"]  # Tensor (B, M)
            batch_cand_length_lp = cand_data["cand_length_log_probs"]  # Tensor (B, M)
            batch_mean_log_prob = cand_data["mean_log_prob"]  # Tensor (B, M)
            batch_delta_m = cand_data["delta_m"]  # Tensor (B, M)
            batch_rel_mass_err = cand_data["relative_mass_error"]  # Tensor (B, M)
            batch_frag_score = cand_data["frag_score"]  # Tensor (B, M)
            batch_expl = cand_data.get("explained_intensity", torch.zeros_like(batch_frag_score))
            batch_cov = cand_data.get("theo_coverage", torch.zeros_like(batch_frag_score))
            batch_ladd = cand_data.get("ladder_score", torch.zeros_like(batch_frag_score))
            batch_bal = cand_data.get("ion_balance", torch.zeros_like(batch_frag_score))
            batch_unm = cand_data.get("unmatched_penalty", torch.zeros_like(batch_frag_score))
            batch_term_bonus = cand_data["terminal_bonus"]  # Tensor (B, M)
            batch_default_scores = cand_data["scores"]  # Tensor (B, M)

            for i in range(b_size):
                tgt = batch_targets[i]
                c_seqs = batch_cand_seqs[i]
                c_lens = [int(v) for v in batch_cand_lengths[i].tolist()]
                c_len_lp = [float(v) for v in batch_cand_length_lp[i].tolist()]
                c_mean_lp = [float(v) for v in batch_mean_log_prob[i].tolist()]
                c_dm = [float(v) for v in batch_delta_m[i].tolist()]
                c_rel_m = [float(v) for v in batch_rel_mass_err[i].tolist()]
                c_frag = [float(v) for v in batch_frag_score[i].tolist()]
                c_expl = [float(v) for v in batch_expl[i].tolist()]
                c_cov = [float(v) for v in batch_cov[i].tolist()]
                c_ladd = [float(v) for v in batch_ladd[i].tolist()]
                c_bal = [float(v) for v in batch_bal[i].tolist()]
                c_unm = [float(v) for v in batch_unm[i].tolist()]
                c_term = [float(v) for v in batch_term_bonus[i].tolist()]
                c_def_sc = [float(v) for v in batch_default_scores[i].tolist()]

                # Calculate mass error in ppm for each candidate
                p_m = float(p_mass[i].item())
                tgt_residue_mass = max(1.0, p_m - M_H2O)
                c_ppm = [(abs(dm) / tgt_residue_mass) * 1e6 for dm in c_dm]

                # Oracle check: is target in candidate pool?
                oracle_il = any(il_match(cs, tgt) for cs in c_seqs)

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
                    "baseline_prediction": sel_seqs[i],
                    "oracle_il": oracle_il,
                })

            processed_count += b_size
            elapsed = time.time() - t0
            rate = processed_count / max(elapsed, 1e-4)
            if (b_idx + 1) % 20 == 0 or processed_count >= eval_count:
                cur_oracle = sum(r["oracle_il"] for r in candidate_records) / len(candidate_records) * 100.0
                cur_base = sum(il_match(r["baseline_prediction"], r["target"]) for r in candidate_records) / len(candidate_records) * 100.0
                print(
                    f"   [{processed_count:,}/{eval_count:,}] ({rate:.1f} spec/s) | "
                    f"Baseline Exact(I/L): {cur_base:.2f}% | Oracle Ceiling: {cur_oracle:.2f}%"
                )

    total_time = time.time() - t0
    n_total = len(candidate_records)
    overall_oracle_il = sum(r["oracle_il"] for r in candidate_records) / n_total * 100.0
    print("\n" + "=" * 90)
    print(f"CANDIDATE GENERATION COMPLETED ({n_total:,} spectra, {total_time:.1f}s, {n_total/total_time:.1f} spec/s)")
    print(f"  >>> ORACLE EXACT MATCH (I/L) CEILING: {overall_oracle_il:.2f}% <<<")
    print("=" * 90)

    # 4. Reranking Calibration & Grid Search
    print("\n[4/4] Evaluating Reranking Formulations on Candidate Pool...")

    if args.dump_records:
        dump_p = Path(args.dump_records)
        dump_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(candidate_records, dump_p)
        print(f"Saved {len(candidate_records)} candidate records to {dump_p}")

    # Extract numerical features for machine learning rerankers
    features, match_il = extract_candidate_features(candidate_records)

    trained_rerankers = {}
    if args.trained_reranker and Path(args.trained_reranker).is_file():
        print(f"Loading pre-trained reranker from {args.trained_reranker}...")
        loaded_model = CandidateReranker.load(args.trained_reranker)
        trained_rerankers["Pre-trained Reranker"] = loaded_model

    # Train linear and MLP rerankers with train/val split (80/20)
    n_samples = len(candidate_records)
    split_idx = int(n_samples * 0.8)
    tr_feat, val_feat = features[:split_idx], features[split_idx:]
    tr_il, val_il = match_il[:split_idx], match_il[split_idx:]

    # 1. Linear Reranker
    lin_reranker = CandidateReranker(mode="linear")
    lin_stats = lin_reranker.fit_listnet(
        features=tr_feat,
        match_il=tr_il,
        val_features=val_feat,
        val_match_il=val_il,
        epochs=150,
        lr=0.02,
    )
    trained_rerankers["Learned Linear ListNet Reranker"] = lin_reranker
    print("\nLearned Linear Feature Weights:")
    for fname, fweight in lin_reranker.get_feature_weights().items():
        print(f"   {fname:<25}: {fweight:+.4f}")

    # 2. MLP Reranker
    mlp_reranker = CandidateReranker(mode="mlp", hidden_dim=32)
    mlp_stats = mlp_reranker.fit_listnet(
        features=tr_feat,
        match_il=tr_il,
        val_features=val_feat,
        val_match_il=val_il,
        epochs=150,
        lr=0.005,
    )
    trained_rerankers["Learned Deep MLP Reranker"] = mlp_reranker

    rerank_configs = [
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
            "fn": lambda r, idx: (
                np.argmax(
                    np.where(np.array(r["ppm_error"]) <= 20.0, 0.0, -100.0)
                    + 0.15 * np.array(r["cand_length_log_prob"])
                    + 1.0 * np.array(r["mean_log_prob"])
                    + 3.0 * np.array(r["frag_score"])
                    + 1.0 * np.array(r["term_bonus"])
                )
            ),
        },
        {
            "name": "Strict Mass Gated (PPM <= 20) + Frag Dominant (Beta=5.0)",
            "fn": lambda r, idx: (
                np.argmax(
                    np.where(np.array(r["ppm_error"]) <= 20.0, 0.0, -100.0)
                    + 0.10 * np.array(r["cand_length_log_prob"])
                    + 0.8 * np.array(r["mean_log_prob"])
                    + 5.0 * np.array(r["frag_score"])
                    + 1.0 * np.array(r["term_bonus"])
                )
            ),
        },
        {
            "name": "Mass Gated (PPM <= 10) + Frag Dominant (Beta=6.0)",
            "fn": lambda r, idx: (
                np.argmax(
                    np.where(np.array(r["ppm_error"]) <= 10.0, 0.0, -100.0)
                    + 0.05 * np.array(r["cand_length_log_prob"])
                    + 0.5 * np.array(r["mean_log_prob"])
                    + 6.0 * np.array(r["frag_score"])
                    + 1.0 * np.array(r["term_bonus"])
                )
            ),
        },
    ]

    # Add learned models to evaluation configs
    for r_name, r_model in trained_rerankers.items():
        r_model.eval()
        sc_mat = r_model.score_numpy(features)
        rerank_configs.append({
            "name": r_name,
            "fn": (lambda sm: lambda r, idx: int(np.argmax(sm[idx])))(sc_mat),
            "model_obj": r_model,
        })

    # Grid search over fine-grained weights to find optimal formulation
    best_exact_il = -1.0
    best_config_name = ""
    evaluation_summary = {
        "checkpoint": str(ckpt_path),
        "num_spectra": n_total,
        "oracle_exact_il": overall_oracle_il,
        "rerank_results": [],
    }

    print(f"\n{'Reranking Formulation':<56} | {'Peptide Acc (I/L)':<18} | {'Mass Match':<10} | {'AA Prec':<8} | {'AA Rec':<8} | {'AA F1':<8} | {'Len Acc':<8}")
    print("-" * 130)

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
            "peptide_accuracy": metrics.exact_peptide_accuracy_il,
            "mass_match": metrics.mass_peptide_accuracy,
            "aa_precision": metrics.aa_precision,
            "aa_recall": metrics.aa_recall,
            "aa_f1": metrics.aa_f1,
            "aa_error_rate": metrics.aa_error_rate,
            "length_accuracy": metrics.length_accuracy,
            "peptide_precision_il": metrics.peptide_precision_exact_il,
            "peptide_recall_il": metrics.peptide_recall_exact_il,
            "peptide_f1_il": metrics.peptide_f1_exact_il,
            "val_peptide_accuracy": val_exact_il,
        }
        evaluation_summary["rerank_results"].append(res_entry)

        val_str = f" | Val: {val_exact_il*100:>5.2f}%" if val_exact_il is not None else ""
        print(
            f"{cfg['name']:<56} | "
            f"{metrics.exact_peptide_accuracy_il*100:>17.2f}% | "
            f"{metrics.mass_peptide_accuracy*100:>9.2f}% | "
            f"{metrics.aa_precision*100:>7.2f}% | "
            f"{metrics.aa_recall*100:>7.2f}% | "
            f"{metrics.aa_f1*100:>7.2f}% | "
            f"{metrics.length_accuracy*100:>7.2f}%"
            f"{val_str}"
        )

        if metrics.exact_peptide_accuracy_il > best_exact_il:
            best_exact_il = metrics.exact_peptide_accuracy_il
            best_config_name = cfg["name"]
            if "model_obj" in cfg and args.save_reranker:
                cfg["model_obj"].save(args.save_reranker)
                print(f"   [Checkpoint] Saved best reranker model to {args.save_reranker}")

    print("-" * 100)
    print(f"Top Performing Formulation: '{best_config_name}' ({best_exact_il*100:.2f}% Exact I/L)")
    print(f"Oracle Ceiling: {overall_oracle_il:.2f}% Exact I/L")

    # Save results
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"\nSaved full calibration results to {out_path}")


if __name__ == "__main__":
    main()
