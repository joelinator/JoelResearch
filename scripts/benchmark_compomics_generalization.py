#!/usr/bin/env python3
"""
Generalization Benchmark: Evaluating SOTA DFlowNovo on InstaDeepAI/CompOmics_PRIDE test split.

Cross-project generalization evaluation across 18 PRIDE projects completely
unseen and independent of HCPT (ProteomeTools) and Nine-Species.

Evaluates 1,000 real spectra uniformly sampled across all 18 PRIDE project shards.
Filters:
- Peptide length: 7 to 30 residues
- Precursor charge: 2 to 4
- Full token validation against model vocabulary (including M(ox), C(cam), etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from data.constants import AA_MASSES_DICT, M_H, M_H2O
from data.data import build_vocabulary, parse_peptide
from eval.metrics import compute_denovo_metrics
from inference.knapsack_dp import ExactReachabilityDP
from inference.predict import predict_peptide
from train.factory import build_models
from train.io import infer_model_config_from_checkpoint, load_checkpoint, load_models_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DFlowNovo on CompOmics_PRIDE generalization test split.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/dfm_length_weighted_10ep/checkpoints/best-joint-gen-exact-epoch=01-exact=0.4746.ckpt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Total number of spectra to evaluate across all shards (default: 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32, keeps VRAM < 1GB).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of flow matching integration steps (default: 20).",
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=3,
        help="Top-k length candidates to evaluate (default: 3).",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/eval_compomics_pride_generalization.json",
        help="Output path for evaluation JSON.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    return parser.parse_args()


class CompOmicsPRIDEDataset(Dataset):
    """In-memory dataset for the collected 1,000 CompOmics_PRIDE spectra."""

    def __init__(self, samples: list[dict], vocab: dict[str, int], top_k: int = 200):
        self.samples = samples
        self.vocab = vocab
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]
        mz_array = torch.tensor(row["mz_array"], dtype=torch.float32)
        intensity_array = torch.tensor(row["intensity_array"], dtype=torch.float32)
        precursor_charge = int(row.get("precursor_charge") or row.get("charge") or 1)

        if "precursor_mass" in row and row["precursor_mass"] is not None:
            precursor_mass = float(row["precursor_mass"])
            if "precursor_mz" in row and row["precursor_mz"] is not None:
                pmz = float(row["precursor_mz"])
                if abs(precursor_mass - pmz) < 1.0 and precursor_charge > 1:
                    precursor_mass = pmz * precursor_charge - precursor_charge * M_H
        elif "precursor_mz" in row and row["precursor_mz"] is not None:
            precursor_mass = float(row["precursor_mz"]) * precursor_charge - precursor_charge * M_H
        else:
            raise KeyError("Row missing both precursor_mass and precursor_mz")

        # Remove precursor peak if present
        precursor_mz = float(row.get("precursor_mz") or ((precursor_mass + precursor_charge * M_H) / max(precursor_charge, 1)))
        keep = (mz_array - precursor_mz).abs() > 1.5
        if keep.any():
            mz_array = mz_array[keep]
            intensity_array = intensity_array[keep]

        # Guarantee at least one peak
        if mz_array.numel() == 0:
            mz_array = torch.tensor([precursor_mz], dtype=torch.float32)
            intensity_array = torch.tensor([1.0], dtype=torch.float32)

        # Select top-k peaks
        if self.top_k is not None:
            k = min(self.top_k, intensity_array.shape[0])
            if k > 0:
                intensity_array, indices = torch.topk(intensity_array, k)
                mz_array = mz_array[indices]

        raw_sequence = row.get("modified_sequence") or row["sequence"]
        tokens = parse_peptide(raw_sequence)
        sequence = torch.tensor([self.vocab[t] for t in tokens], dtype=torch.long)
        length = len(tokens)
        canonical_seq = "".join(tokens)
        mz_complementary = precursor_mass + 2 * M_H - mz_array

        return {
            "mz_array": mz_array,
            "intensity_array": intensity_array,
            "precursor_mass": precursor_mass,
            "precursor_charge": precursor_charge,
            "sequence": sequence,
            "mz_complementary": mz_complementary,
            "length": length,
            "canonical_seq": canonical_seq,
            "pxd": row.get("PXD_identifier", "Unknown"),
            "raw_sequence": raw_sequence,
        }


def compomics_collate_fn(batch_items: list[dict], vocab: dict[str, int]) -> dict:
    mz_list = [item["mz_array"] for item in batch_items]
    int_list = [item["intensity_array"] for item in batch_items]
    mz_comp_list = [item["mz_complementary"] for item in batch_items]
    seq_list = [item["sequence"] for item in batch_items]

    spectrum_mask = pad_sequence(
        [torch.zeros(len(x), dtype=torch.bool) for x in int_list],
        batch_first=True,
        padding_value=True,
    )
    intensity_array = pad_sequence(int_list, batch_first=True, padding_value=0.0, padding_side="right")
    max_intensity = intensity_array.amax(dim=-1, keepdim=True).clamp(min=1e-7)
    intensity_array = (intensity_array / max_intensity).sqrt()

    mz_array = pad_sequence(mz_list, batch_first=True, padding_value=0.0, padding_side="right")
    mz_complementary = pad_sequence(mz_comp_list, batch_first=True, padding_value=0.0, padding_side="right")

    precursor_mass = torch.tensor([item["precursor_mass"] for item in batch_items], dtype=torch.float32)
    precursor_charge = torch.tensor([item["precursor_charge"] for item in batch_items], dtype=torch.long)
    lengths = torch.tensor([item["length"] for item in batch_items], dtype=torch.long)

    canonical_seqs = [item["canonical_seq"] for item in batch_items]
    pxds = [item["pxd"] for item in batch_items]

    return {
        "mz_array": mz_array,
        "intensity_array": intensity_array,
        "precursor_mass": precursor_mass,
        "precursor_charge": precursor_charge,
        "mz_complementary": mz_complementary,
        "spectrum_mask": spectrum_mask,
        "lengths": lengths,
        "canonical_seqs": canonical_seqs,
        "pxds": pxds,
    }


def stream_compomics_samples(num_samples: int, vocab: dict[str, int]) -> list[dict]:
    """Streams and collects valid spectra uniformly across all 18 PRIDE project shards."""
    print(f"\n[1/3] Streaming test split from InstaDeepAI/CompOmics_PRIDE across 18 shards...")
    ds = load_dataset("InstaDeepAI/CompOmics_PRIDE", split="test", streaming=True)
    num_shards = ds.n_shards
    print(f"Dataset has {num_shards} distinct project shards.")

    # Distribute target quota across shards
    base_per_shard = num_samples // num_shards
    remainder = num_samples % num_shards
    target_per_shard = [base_per_shard + (1 if i < remainder else 0) for i in range(num_shards)]

    collected_samples = []
    pxd_counts = defaultdict(int)

    t_start = time.time()
    for shard_idx in range(num_shards):
        quota = target_per_shard[shard_idx]
        if quota <= 0:
            continue
        shard_ds = ds.shard(num_shards=num_shards, index=shard_idx)
        shard_collected = 0
        pxd_name = None

        for sample in shard_ds:
            charge = int(sample.get("precursor_charge") or sample.get("charge") or 0)
            if not (2 <= charge <= 4):
                continue

            raw_seq = sample.get("modified_sequence") or sample.get("sequence")
            if not raw_seq:
                continue

            tokens = parse_peptide(raw_seq)
            if not (7 <= len(tokens) <= 30):
                continue

            # Ensure all parsed tokens are representable in model vocabulary
            if not all(t in vocab for t in tokens):
                continue

            mz_arr = sample.get("mz_array")
            if not mz_arr or len(mz_arr) == 0:
                continue

            pxd = sample.get("PXD_identifier", f"Shard_{shard_idx}")
            pxd_name = pxd
            collected_samples.append(sample)
            pxd_counts[pxd] += 1
            shard_collected += 1

            if shard_collected >= quota:
                break

        print(f"  - Shard {shard_idx:2d} ({pxd_name}): collected {shard_collected} valid spectra")

    elapsed = time.time() - t_start
    print(f"Successfully collected {len(collected_samples)} spectra across {len(pxd_counts)} PRIDE projects in {elapsed:.1f}s.")
    return collected_samples


def compute_peptide_mass(tokens_or_seq: str | list[str]) -> float:
    """Calculates neutral monoisotopic mass of a peptide."""
    if isinstance(tokens_or_seq, str):
        tokens = parse_peptide(tokens_or_seq)
    else:
        tokens = tokens_or_seq
    return sum(AA_MASSES_DICT.get(t, 0.0) for t in tokens) + M_H2O


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Strictly limit GPU memory to prevent interfering with background training
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.04, device.index if device.index is not None else 0)
        torch.set_float32_matmul_precision("high")

    print("================================================================================")
    print("  DFlowNovo: Generalization Benchmark on InstaDeepAI/CompOmics_PRIDE Test Split ")
    print("================================================================================")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Device:           {device}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Flow Steps:       {args.num_steps}")
    print(f"Top-K Lengths:    {args.top_k_lengths}")
    print(f"Target Spectra:   {args.num_samples}")
    print(f"Output File:      {args.output_path}")
    print("================================================================================")

    # 1. Load model checkpoint
    print("\n[2/3] Loading checkpoint and initializing neural models...")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    vocab = ckpt.get("vocabulary")
    if vocab is None:
        vocab = build_vocabulary(include_ptms=True)

    model_cfg = infer_model_config_from_checkpoint(ckpt)
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocab, device, model_cfg=model_cfg)
    load_models_from_checkpoint(ckpt, spectrum_encoder, length_predictor, decoder, guidance, use_ema=True)

    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()
    del ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Pre-build Reachability DP table
    reachability_dp = ExactReachabilityDP.get_default(device=device, tol_da=0.5)

    # 2. Collect dataset samples
    samples = stream_compomics_samples(args.num_samples, vocab)
    dataset = CompOmicsPRIDEDataset(samples, vocab, top_k=200)
    collate = partial(compomics_collate_fn, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # 3. Run Inference
    print(f"\n[3/3] Running de novo sequencing inference on {len(dataset)} spectra (Batch Size: {args.batch_size})...")
    all_predictions: list[str] = []
    all_targets: list[str] = []
    all_pred_lengths: list[int] = []
    all_target_lengths: list[int] = []
    all_scores: list[float] = []
    all_pxds: list[str] = []
    all_precursor_masses: list[float] = []
    all_precursor_charges: list[int] = []

    t_infer_start = time.time()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mz_array = batch["mz_array"].to(device)
            intensity_array = batch["intensity_array"].to(device)
            precursor_mass = batch["precursor_mass"].to(device)
            precursor_charge = batch["precursor_charge"].to(device)
            mz_complementary = batch["mz_complementary"].to(device)
            spectrum_mask = batch["spectrum_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                token_ids, pred_lengths, pred_seqs, pred_scores = predict_peptide(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    precursor_mass=precursor_mass,
                    precursor_charge=precursor_charge,
                    mz_complementary=mz_complementary,
                    spectrum_mask=spectrum_mask,
                    vocabulary=vocab,
                    spectrum_encoder=spectrum_encoder,
                    length_predictor=length_predictor,
                    decoder=decoder,
                    guidance=guidance,
                    scheduler="cosine",
                    num_steps=args.num_steps,
                    top_k_lengths=args.top_k_lengths,
                    temperature=0.0,
                    use_knapsack_filter=True,
                    use_exact_dp_knapsack=True,
                    knapsack_tol_da=0.5,
                    reachability_dp=reachability_dp,
                    use_composite_ladders=True,
                    use_sequential_knapsack=True,
                    use_peak_evidence=True,
                    return_scores=True,
                )

            all_predictions.extend(pred_seqs)
            all_targets.extend(batch["canonical_seqs"])
            all_pred_lengths.extend(pred_lengths.cpu().tolist())
            all_target_lengths.extend(batch["lengths"].tolist())
            all_scores.extend(pred_scores.cpu().tolist())
            all_pxds.extend(batch["pxds"])
            all_precursor_masses.extend(batch["precursor_mass"].tolist())
            all_precursor_charges.extend(batch["precursor_charge"].tolist())

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                elapsed_b = time.time() - t_infer_start
                rate_b = len(all_predictions) / elapsed_b
                print(f"  Processed {len(all_predictions):4d}/{len(dataset)} spectra | {rate_b:.1f} spec/s", flush=True)

    total_infer_time = time.time() - t_infer_start
    throughput = len(all_predictions) / total_infer_time

    # 4. Compute Metrics
    print("\nComputing comprehensive de novo sequencing metrics...")
    metrics = compute_denovo_metrics(
        predictions=all_predictions,
        targets=all_targets,
        predicted_lengths=all_pred_lengths,
        target_lengths=all_target_lengths,
        scores=all_scores,
    )

    # Compute precursor mass error within 20 ppm
    mass_match_prec_count = 0
    mass_match_target_count = 0
    ppm_errors_prec = []
    ppm_errors_target = []
    length_match_count = 0

    for pred_seq, tgt_seq, prec_m in zip(all_predictions, all_targets, all_precursor_masses):
        pred_m = compute_peptide_mass(pred_seq)
        tgt_m = compute_peptide_mass(tgt_seq)

        # vs experimental precursor mass
        ppm_prec = abs(pred_m - prec_m) / max(prec_m, 1.0) * 1e6
        ppm_errors_prec.append(ppm_prec)
        if ppm_prec <= 20.0:
            mass_match_prec_count += 1

        # vs theoretical target mass
        ppm_tgt = abs(pred_m - tgt_m) / max(tgt_m, 1.0) * 1e6
        ppm_errors_target.append(ppm_tgt)
        if ppm_tgt <= 20.0:
            mass_match_target_count += 1

        if len(parse_peptide(pred_seq)) == len(parse_peptide(tgt_seq)):
            length_match_count += 1

    n_total = len(all_predictions)
    mass_match_prec_rate = mass_match_prec_count / n_total
    mass_match_target_rate = mass_match_target_count / n_total
    computed_length_acc = length_match_count / n_total

    # Stratification by PRIDE Project
    pxd_results = {}
    unique_pxds = sorted(list(set(all_pxds)))
    for pxd in unique_pxds:
        indices = [i for i, p in enumerate(all_pxds) if p == pxd]
        sub_preds = [all_predictions[i] for i in indices]
        sub_tgts = [all_targets[i] for i in indices]
        sub_pred_lens = [all_pred_lengths[i] for i in indices]
        sub_tgt_lens = [all_target_lengths[i] for i in indices]
        sub_prec_m = [all_precursor_masses[i] for i in indices]

        sub_m = compute_denovo_metrics(sub_preds, sub_tgts, predicted_lengths=sub_pred_lens, target_lengths=sub_tgt_lens)
        sub_ppm_match = sum(
            1 for p, pm in zip(sub_preds, sub_prec_m)
            if abs(compute_peptide_mass(p) - pm) / max(pm, 1.0) * 1e6 <= 20.0
        ) / len(indices)
        sub_len_acc = sum(1 for p, t in zip(sub_preds, sub_tgts) if len(parse_peptide(p)) == len(parse_peptide(t))) / len(indices)

        pxd_results[pxd] = {
            "count": len(indices),
            "exact_strict": round(float(sub_m.exact_peptide_accuracy), 4),
            "exact_il": round(float(sub_m.exact_peptide_accuracy_il), 4),
            "aa_precision": round(float(sub_m.aa_precision), 4),
            "aa_recall": round(float(sub_m.aa_recall), 4),
            "aa_f1": round(float(sub_m.aa_f1), 4),
            "mass_match_20ppm": round(float(sub_ppm_match), 4),
            "length_accuracy": round(float(sub_len_acc), 4),
        }

    # Stratification by Peptide Length Bins
    length_bins = [(7, 10), (11, 14), (15, 18), (19, 22), (23, 30)]
    length_results = {}
    for l_min, l_max in length_bins:
        indices = [i for i, l in enumerate(all_target_lengths) if l_min <= l <= l_max]
        if not indices:
            continue
        sub_preds = [all_predictions[i] for i in indices]
        sub_tgts = [all_targets[i] for i in indices]
        sub_m = compute_denovo_metrics(sub_preds, sub_tgts)
        sub_len_acc = sum(1 for p, t in zip(sub_preds, sub_tgts) if len(parse_peptide(p)) == len(parse_peptide(t))) / len(indices)

        length_results[f"L[{l_min:02d}-{l_max:02d}]"] = {
            "count": len(indices),
            "exact_strict": round(float(sub_m.exact_peptide_accuracy), 4),
            "exact_il": round(float(sub_m.exact_peptide_accuracy_il), 4),
            "aa_precision": round(float(sub_m.aa_precision), 4),
            "aa_recall": round(float(sub_m.aa_recall), 4),
            "aa_f1": round(float(sub_m.aa_f1), 4),
            "length_accuracy": round(float(sub_len_acc), 4),
        }

    # Stratification by Precursor Charge
    charge_results = {}
    for c in [2, 3, 4]:
        indices = [i for i, ch in enumerate(all_precursor_charges) if ch == c]
        if not indices:
            continue
        sub_preds = [all_predictions[i] for i in indices]
        sub_tgts = [all_targets[i] for i in indices]
        sub_m = compute_denovo_metrics(sub_preds, sub_tgts)

        charge_results[f"Charge +{c}"] = {
            "count": len(indices),
            "exact_strict": round(float(sub_m.exact_peptide_accuracy), 4),
            "exact_il": round(float(sub_m.exact_peptide_accuracy_il), 4),
            "aa_f1": round(float(sub_m.aa_f1), 4),
        }

    # Samples Preview (First 10)
    samples_preview = []
    for i in range(min(10, n_total)):
        samples_preview.append({
            "pxd": all_pxds[i],
            "charge": all_precursor_charges[i],
            "precursor_mass": round(all_precursor_masses[i], 4),
            "target": all_targets[i],
            "prediction": all_predictions[i],
            "score": round(all_scores[i], 4),
            "exact_strict": all_predictions[i] == all_targets[i],
            "exact_il": all_predictions[i].replace("I", "L") == all_targets[i].replace("I", "L"),
            "ppm_error_prec": round(ppm_errors_prec[i], 2),
        })

    # Prepare complete output dictionary
    output_dict = {
        "dataset": "InstaDeepAI/CompOmics_PRIDE",
        "split": "test",
        "checkpoint": Path(args.checkpoint).name,
        "num_samples": n_total,
        "throughput_spectra_per_sec": round(throughput, 1),
        "elapsed_seconds": round(total_infer_time, 2),
        "decoding_parameters": {
            "num_steps": args.num_steps,
            "top_k_lengths": args.top_k_lengths,
            "scheduler": "cosine",
            "temperature": 0.0,
            "use_knapsack_filter": True,
            "use_exact_dp_knapsack": True,
            "knapsack_tol_da": 0.5,
            "use_composite_ladders": True,
            "use_sequential_knapsack": True,
            "use_peak_evidence": True,
        },
        "overall_metrics": {
            "exact_strict": round(float(metrics.exact_peptide_accuracy), 4),
            "exact_il": round(float(metrics.exact_peptide_accuracy_il), 4),
            "aa_precision": round(float(metrics.aa_precision), 4),
            "aa_recall": round(float(metrics.aa_recall), 4),
            "aa_f1": round(float(metrics.aa_f1), 4),
            "mass_match_20ppm_precursor": round(float(mass_match_prec_rate), 4),
            "mass_match_20ppm_target": round(float(mass_match_target_rate), 4),
            "length_accuracy": round(float(computed_length_acc), 4),
            "auc_exact": round(float(metrics.auc_exact), 4),
            "pauc80_exact": round(float(metrics.pauc80_exact), 4),
            "auc_exact_il": round(float(metrics.auc_exact_il), 4),
            "pauc80_exact_il": round(float(metrics.pauc80_exact_il), 4),
        },
        "stratified_by_length": length_results,
        "stratified_by_charge": charge_results,
        "per_pride_project": pxd_results,
        "samples_preview": samples_preview,
    }

    # Save to disk
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n[Artifact Saved] Successfully written results to {out_file.resolve()}")

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("                     OVERALL GENERALIZATION SUMMARY                       ")
    print("=" * 80)
    print(f"Total Evaluated Spectra:            {n_total:,}")
    print(f"Throughput:                         {throughput:.1f} spectra/sec ({total_infer_time:.1f}s)")
    print(f"Strict Exact Match:                 {metrics.exact_peptide_accuracy * 100:.2f}%")
    print(f"I/L Exact Match:                    {metrics.exact_peptide_accuracy_il * 100:.2f}%")
    print(f"Residue Precision:                  {metrics.aa_precision * 100:.2f}%")
    print(f"Residue Recall:                     {metrics.aa_recall * 100:.2f}%")
    print(f"Residue F1:                         {metrics.aa_f1 * 100:.2f}%")
    print(f"Precursor Mass Match (<= 20 ppm):   {mass_match_prec_rate * 100:.2f}%")
    print(f"Theoretical Mass Match (<= 20 ppm): {mass_match_target_rate * 100:.2f}%")
    print(f"Peptide Length Accuracy:            {computed_length_acc * 100:.2f}%")
    print("=" * 80)

    print("\n" + "-" * 75)
    print(f"{'Length Bin':<12} | {'Count':<7} | {'Strict Exact':<12} | {'I/L Exact':<12} | {'Residue F1':<10} | {'Len Acc':<8}")
    print("-" * 75)
    for bin_name, stats in length_results.items():
        print(
            f"{bin_name:<12} | {stats['count']:<7} | {stats['exact_strict']:<12.2%} | "
            f"{stats['exact_il']:<12.2%} | {stats['aa_f1']:<10.2%} | {stats['length_accuracy']:<8.2%}"
        )
    print("-" * 75)

    print("\n" + "-" * 88)
    print(f"{'PRIDE Project':<14} | {'Count':<6} | {'Strict':<8} | {'I/L':<8} | {'Residue F1':<11} | {'Mass <=20ppm':<13} | {'Len Acc':<8}")
    print("-" * 88)
    for pxd, stats in pxd_results.items():
        print(
            f"{pxd:<14} | {stats['count']:<6} | {stats['exact_strict']:<8.2%} | "
            f"{stats['exact_il']:<8.2%} | {stats['aa_f1']:<11.2%} | {stats['mass_match_20ppm']:<13.2%} | {stats['length_accuracy']:<8.2%}"
        )
    print("-" * 88)


if __name__ == "__main__":
    main()
