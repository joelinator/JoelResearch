#!/usr/bin/env python3
"""
Scientific Reproduction Experiment: Evaluating the Agent Squad's Claimed Improvements.

Tests:
1. Baseline Decoding: Fixed Knapsack DP mass tolerance (tol = 0.5 Da).
2. Agent Proposal A: Charge-Adaptive Knapsack DP tolerance delta(z) = delta_0 * (1 + 0.1*(z-1)).
3. Agent Proposal B: Min-SNR vs Cosine Annealing scheduler.

Evaluates on a deterministic 2,000-spectra sample of Nine-Species test split.
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
from eval.metrics import compute_denovo_metrics
from flow_matching.scheduler import get_scheduler
from inference.knapsack_dp import ExactReachabilityDP, calculate_charge_adaptive_tolerance
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_decoding_experiment(
    checkpoint_path: Path,
    dataset_name: str = "InstaDeepAI/ms_ninespecies_benchmark",
    num_samples: int = 2000,
    batch_size: int = 128,
    use_charge_adaptive: bool = False,
    scheduler_name: str = "cosine",
    alpha_adaptive: float = 0.1,
):
    print(f"\n--- Testing Configuration ---")
    print(f"Dataset:            {dataset_name}")
    print(f"Samples:            {num_samples}")
    print(f"Scheduler:          {scheduler_name}")
    print(f"Charge-Adaptive DP: {use_charge_adaptive} (alpha={alpha_adaptive})")

    ckpt = load_checkpoint(str(checkpoint_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()

    ds = get_dataset(dataset_name, split="test", cache_dir="data/cache")
    loader = build_dataloader(ds, vocab, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize Reachability DP table
    reachability_dp = ExactReachabilityDP.get_default(device=DEVICE, tol_da=0.5)

    all_preds = []
    all_targets = []
    all_scores = []
    total_processed = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if total_processed >= num_samples:
                break
            (
                mz_array,
                intensity_array,
                precursor_mass,
                precursor_charge,
                sequence,
                mz_complementary,
                length,
                _padded_mask,
                spectrum_mask,
            ) = tuple(tensor.to(DEVICE) if torch.is_tensor(tensor) else tensor for tensor in batch)

            B = mz_array.shape[0]
            # Use predict_peptide with or without charge adaptive tolerance
            from inference.predict import predict_peptide

            # If charge adaptive is requested, pass scaled tolerance
            if use_charge_adaptive:
                # Average charge in batch or per-sample tolerance
                # Calculate adaptive tolerance per sample
                base_tol = 0.5
                avg_charge = precursor_charge.float().mean().item()
                effective_tol = base_tol * (1.0 + alpha_adaptive * (max(1.0, avg_charge) - 1.0))
            else:
                effective_tol = 0.5

            token_ids, pred_lengths, pred_seqs, pred_scores = predict_peptide(
                mz_array=mz_array,
                intensity_array=intensity_array,
                precursor_mass=precursor_mass,
                precursor_charge=precursor_charge,
                mz_complementary=mz_complementary,
                spectrum_mask=spectrum_mask,
                vocabulary=vocab,
                spectrum_encoder=enc,
                length_predictor=lp,
                decoder=dec,
                guidance=guid,
                scheduler=scheduler_name,
                num_steps=20,
                top_k_lengths=3,
                temperature=0.0,
                use_knapsack_filter=True,
                use_exact_dp_knapsack=True,
                knapsack_tol_da=effective_tol,
                reachability_dp=reachability_dp,
                return_scores=True,
            )

            from eval.evaluate import sequences_from_batch
            targets = sequences_from_batch(sequence, vocab)

            all_preds.extend(pred_seqs)
            all_targets.extend(targets)
            all_scores.extend([float(s) for s in pred_scores.tolist()])
            total_processed += B

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {len(all_preds)} / {num_samples} spectra...", flush=True)

    elapsed = time.time() - t0
    all_preds = all_preds[:num_samples]
    all_targets = all_targets[:num_samples]
    all_scores = all_scores[:num_samples]

    metrics = compute_denovo_metrics(all_preds, all_targets, scores=all_scores)
    throughput = len(all_preds) / elapsed

    print(f"Results for (Adaptive={use_charge_adaptive}, Scheduler={scheduler_name}):")
    print(f"  Strict Exact Match: {metrics.exact_peptide_accuracy * 100:.2f}%")
    print(f"  I/L Exact Match:    {metrics.exact_peptide_accuracy_il * 100:.2f}%")
    print(f"  Residue F1:         {metrics.aa_f1 * 100:.2f}%")
    print(f"  Throughput:         {throughput:.1f} spec/s")

    return {
        "strict_exact": metrics.exact_peptide_accuracy * 100,
        "il_exact": metrics.exact_peptide_accuracy_il * 100,
        "aa_f1": metrics.aa_f1 * 100,
        "aa_precision": metrics.aa_precision * 100,
        "aa_recall": metrics.aa_recall * 100,
        "throughput": throughput,
    }


def main():
    ckpt_path = PROJECT_ROOT / "artifacts" / "dfm_joint_balanced_30ep" / "checkpoints" / "dfm_balanced_best.ckpt"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found!")
        return

    print("================================================================================")
    print("   SCIENTIFIC REPRODUCTION AUDIT: AGENT SQUAD CLAIMED ALGORITHM IMPROVEMENTS")
    print("================================================================================")

    # 1. Baseline: Cosine scheduler, fixed Knapsack DP tolerance (0.5 Da)
    res_baseline = run_decoding_experiment(
        checkpoint_path=ckpt_path,
        num_samples=2000,
        use_charge_adaptive=False,
        scheduler_name="cosine",
    )

    # 2. Agent Proposal A: Charge-Adaptive Knapsack DP tolerance (alpha = 0.1)
    res_charge_adaptive = run_decoding_experiment(
        checkpoint_path=ckpt_path,
        num_samples=2000,
        use_charge_adaptive=True,
        scheduler_name="cosine",
        alpha_adaptive=0.1,
    )

    # 3. Agent Proposal B: Min-SNR Scheduler vs Cosine
    res_min_snr = run_decoding_experiment(
        checkpoint_path=ckpt_path,
        num_samples=2000,
        use_charge_adaptive=False,
        scheduler_name="min_snr",
    )

    print("\n" + "=" * 80)
    print("                      HEAD-TO-HEAD REPRODUCTION SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Strict Match':<12} | {'I/L Match':<12} | {'AA F1':<10} | {'Throughput'}")
    print("-" * 80)
    print(f"{'1. Baseline (Fixed DP tol=0.5)':<35} | {res_baseline['strict_exact']:>10.2f}% | {res_baseline['il_exact']:>10.2f}% | {res_baseline['aa_f1']:>8.2f}% | {res_baseline['throughput']:>8.1f} sps")
    print(f"{'2. Agent Proposal (Charge-Adaptive)':<35} | {res_charge_adaptive['strict_exact']:>10.2f}% | {res_charge_adaptive['il_exact']:>10.2f}% | {res_charge_adaptive['aa_f1']:>8.2f}% | {res_charge_adaptive['throughput']:>8.1f} sps")
    print(f"{'3. Agent Proposal (Min-SNR Sched)':<35} | {res_min_snr['strict_exact']:>10.2f}% | {res_min_snr['il_exact']:>10.2f}% | {res_min_snr['aa_f1']:>8.2f}% | {res_min_snr['throughput']:>8.1f} sps")
    print("=" * 80)


if __name__ == "__main__":
    main()
