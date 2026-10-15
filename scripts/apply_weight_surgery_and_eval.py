#!/usr/bin/env python3
"""
Weight Surgery & Evaluation Script for DFlowNovo.

Transplants a model trained exclusively on standard 20 amino acids
to an expanded vocabulary including target PTM modifications (M(ox), N/Q(deam), C(cam), S/T/Y(ph)),
and evaluates the zero-shot / post-surgery de novo sequencing performance on test spectra.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from data.constants import STANDARD_AMINO_ACIDS, PTM_PARENT_MAP
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint
from train.surgery import transplant_state_dict


def find_best_standard_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = list(checkpoints_dir.glob("best-gen-exact-epoch=*.ckpt"))
    if candidates:
        candidates.sort(
            key=lambda p: float(p.stem.split("exact=")[-1]) if "exact=" in p.stem else 0.0,
            reverse=True,
        )
        return candidates[0]
    last = checkpoints_dir / "last.ckpt"
    if last.exists():
        return last
    all_ckpts = list(checkpoints_dir.glob("*.ckpt"))
    if all_ckpts:
        return all_ckpts[0]
    raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}")


def apply_surgery(
    ckpt_path: Path,
    out_ckpt_path: Path,
    perturbation_std: float = 0.005,
    bias_prior_offset: float = -0.5,
) -> tuple[dict, dict, dict]:
    print(f"\n[Surgery] Loading standard checkpoint from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # 1. Old standard vocabulary
    old_vocab = ckpt.get("vocabulary")
    if not old_vocab:
        old_vocab = {aa: idx for idx, aa in enumerate(STANDARD_AMINO_ACIDS)}
        old_vocab["<pad>"] = len(old_vocab)
        old_vocab["<mask_token>"] = len(old_vocab)

    print(f"[Surgery] Old vocabulary size: {len(old_vocab)} tokens (Standard 20 AA)")

    # 2. New expanded vocabulary with target PTMs
    new_vocab = build_vocabulary(include_ptms=True)
    print(f"[Surgery] New vocabulary size: {len(new_vocab)} tokens (Including PTMs)")

    # 3. Apply weight surgery
    print(f"[Surgery] Transplanting embeddings, output head, and mass buffers...")
    new_state_dict = transplant_state_dict(
        state_dict,
        old_vocab=old_vocab,
        new_vocab=new_vocab,
        parent_map=PTM_PARENT_MAP,
        perturbation_std=perturbation_std,
        bias_prior_offset=bias_prior_offset,
    )

    new_ckpt = {
        "state_dict": new_state_dict,
        "vocabulary": new_vocab,
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "surgery_source": str(ckpt_path.name),
        "surgery_timestamp": time.time(),
    }

    if "ema_shadow_params" in ckpt and ckpt["ema_shadow_params"]:
        print(f"[Surgery] Transplanting EMA shadow parameters...")
        new_ckpt["ema_shadow_params"] = transplant_state_dict(
            ckpt["ema_shadow_params"],
            old_vocab=old_vocab,
            new_vocab=new_vocab,
            parent_map=PTM_PARENT_MAP,
            perturbation_std=0.0,
            bias_prior_offset=bias_prior_offset,
        )

    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, str(out_ckpt_path))
    print(f"[Surgery] Saved transplanted PTM checkpoint to: {out_ckpt_path} ({os.path.getsize(out_ckpt_path) / 1e6:.1f} MB)")

    return new_ckpt, old_vocab, new_vocab


def main():
    parser = argparse.ArgumentParser(description="Apply weight surgery and evaluate on test spectra.")
    parser.add_argument(
        "--standard-dir",
        type=str,
        default="artifacts/dfm_ninespecies_standard_30ep/dfm_ninespecies_standard_30ep/checkpoints",
        help="Directory containing standard 20 AA checkpoints.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Direct path to standard checkpoint. If None, discovers best checkpoint in --standard-dir.",
    )
    parser.add_argument(
        "--out-ckpt",
        type=str,
        default=None,
        help="Path to save transplanted checkpoint. Defaults to best_transplanted_ptm.ckpt in parent of standard-dir.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="InstaDeepAI/ms_ninespecies_benchmark",
        help="Dataset name for evaluation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Number of spectra to evaluate (default: 10,000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--knapsack-tol-da",
        type=float,
        default=0.1,
        help="Knapsack mass tolerance in Daltons (default: 0.1 Da).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.15,
        help="Detailed balance error correction rate.",
    )
    parser.add_argument(
        "--bias-prior-offset",
        type=float,
        default=-0.5,
        help="Logit prior bias offset for newly added PTM classes.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Cap evaluation process memory to 8% (~6.5 GiB) to guarantee zero interference with active training
        torch.cuda.set_per_process_memory_fraction(0.08, 0)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_path = find_best_standard_checkpoint(Path(args.standard_dir))

    if args.out_ckpt:
        out_ckpt_path = Path(args.out_ckpt)
    else:
        out_ckpt_path = ckpt_path.parent / "best_transplanted_ptm.ckpt"

    # Step 1: Weight surgery
    transplanted_ckpt, old_vocab, new_vocab = apply_surgery(
        ckpt_path=ckpt_path,
        out_ckpt_path=out_ckpt_path,
        bias_prior_offset=args.bias_prior_offset,
    )

    # Step 2: Evaluation on test split with Refined Decoding
    print(f"\n[Evaluation] Loading test split from {args.dataset_name} ({args.split})...")
    ds = get_dataset(args.dataset_name, split=args.split, cache_dir="data/cache")
    loader = build_dataloader(
        ds,
        vocab=new_vocab,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        is_train=False,
    )

    print(f"[Evaluation] Instantiating models with expanded vocabulary ({len(new_vocab)} tokens)...")
    enc, lp, dec, guid = build_models(new_vocab, device)
    load_models_from_checkpoint(transplanted_ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()

    print(f"[Evaluation] Running Refined Generative Decoding on {args.max_samples:,} spectra...")
    print(f"  Config: K=3, S=1, num_steps=20, knapsack_tol={args.knapsack_tol_da} Da, eta={args.eta}")

    t0 = time.time()
    metrics, details = evaluate_generative(
        loader=loader,
        vocabulary=new_vocab,
        spectrum_encoder=enc,
        length_predictor=lp,
        decoder=dec,
        guidance=guid,
        scheduler="cosine",
        device=device,
        max_samples=args.max_samples,
        num_steps=20,
        top_k_lengths=3,
        num_samples_per_length=1,
        temperature=0.0,
        use_exact_dp_knapsack=True,
        use_sequential_knapsack=True,
        use_peak_evidence=True,
        knapsack_tol_da=args.knapsack_tol_da,
        eta=args.eta,
        return_details=True,
    )
    dt = time.time() - t0

    preds = details["predictions"]
    targets = details["targets"]
    n = len(preds)
    m_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else vars(metrics)
    exact_strict = m_dict.get("exact_peptide_accuracy", 0.0)
    exact_il = m_dict.get("exact_peptide_accuracy_il", 0.0)
    aa_f1 = m_dict.get("aa_f1", 0.0)
    aa_prec = m_dict.get("aa_precision", 0.0)
    aa_rec = m_dict.get("aa_recall", 0.0)
    len_acc = m_dict.get("length_accuracy", 0.0)

    print(f"\n=======================================================")
    print(f"📊 Zero-Shot PTM Post-Surgery Evaluation Results")
    print(f"=======================================================")
    print(f"Evaluated Spectra:       {n:,} (in {dt:.1f}s, {n / max(dt, 0.1):.1f} spec/s)")
    print(f"Strict Exact Match:      {exact_strict * 100:.2f}%")
    print(f"I/L-Equivalent Match:    {exact_il * 100:.2f}%")
    print(f"Length Accuracy:         {len_acc * 100:.2f}%")
    print(f"Amino Acid F1:           {aa_f1 * 100:.2f}%")
    print(f"Amino Acid Precision:    {aa_prec * 100:.2f}%")
    print(f"Amino Acid Recall:       {aa_rec * 100:.2f}%")

    # Sample comparison
    print(f"\nSample Predictions (First 5):")
    for i in range(min(5, n)):
        print(f"  [{i+1}] Target: {targets[i]}")
        print(f"      Pred:   {preds[i]} {'✓' if preds[i] == targets[i] else '✗'}")

    # Save results json
    results_path = out_ckpt_path.parent / "weight_surgery_zero_shot_results.json"
    results_payload = {
        "surgery_source_ckpt": str(ckpt_path.name),
        "transplanted_ckpt": str(out_ckpt_path.name),
        "num_samples": n,
        "elapsed_seconds": round(dt, 2),
        "throughput": round(n / max(dt, 0.1), 1),
        "exact_strict": round(exact_strict, 4),
        "exact_il": round(exact_il, 4),
        "length_accuracy": round(len_acc, 4),
        "aa_f1": round(aa_f1, 4),
        "aa_precision": round(aa_prec, 4),
        "aa_recall": round(aa_rec, 4),
        "stratified_breakdown": details.get("stratified_breakdown", {}),
    }
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[Saved] Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
