"""
Benchmark script comparing Baseline Decoding vs New SOTA Decoding
on a representative validation subset using artifacts/dfm_pl_run_20260904_144123/checkpoints/last.ckpt.
"""

import json
import time
from pathlib import Path
import torch
import numpy as np

import sys
sys.path.insert(0, "src")

from data.data import build_dataloader, get_dataset
from eval.evaluate import sequences_from_batch
from eval.metrics import calibrate_score_threshold, compute_denovo_metrics, peptide_matches_mass_based
from flow_matching.scheduler import cosine_scheduler
from inference.predict import predict_peptide
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint


def run_benchmark(
    checkpoint_path: str = "artifacts/dfm_pl_run_20260904_144123/checkpoints/last.ckpt",
    vocab_path: str = "artifacts/dfm_pl_run_20260904_144123/vocabulary.json",
    num_samples: int = 1000,
    batch_size: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    with open(vocab_path) as f:
        vocabulary = json.load(f)

    # 1. Build and load models
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocabulary, device)
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    load_models_from_checkpoint(ckpt, spectrum_encoder, length_predictor, decoder, guidance)
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()

    # 2. Load validation dataset subset
    print(f"Loading validation[:{num_samples}]...")
    ds = get_dataset(split=f"validation[:{num_samples}]", cache_dir="data/cache")
    loader = build_dataloader(ds, vocabulary, batch_size=batch_size, shuffle=False, num_workers=4, is_train=False)

    # 3. Configurations to benchmark
    configs = [
        {
            "name": "1. Baseline (Random Unmasking, Beta=0, No Prior)",
            "decoding_strategy": "random",
            "temperature": 1.0,
            "beta": 0.0,
            "trypsin_prior": False,
        },
        {
            "name": "2. + Confidence-Based MAP Unmasking",
            "decoding_strategy": "confidence",
            "temperature": 0.0,
            "beta": 0.0,
            "trypsin_prior": False,
        },
        {
            "name": "3. + Trypsin Enzymatic Prior (+0.2)",
            "decoding_strategy": "confidence",
            "temperature": 0.0,
            "beta": 0.0,
            "trypsin_prior": True,
        },
        {
            "name": "4. + Theoretical Fragment Ion Matching (Beta=0.5)",
            "decoding_strategy": "confidence",
            "temperature": 0.0,
            "beta": 0.5,
            "trypsin_prior": False,
        },
        {
            "name": "5. Full SOTA Pipeline (Confidence + Beta=0.5 + Trypsin Prior)",
            "decoding_strategy": "confidence",
            "temperature": 0.0,
            "beta": 0.5,
            "trypsin_prior": True,
        },
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n=======================================================")
        print(f"Evaluating: {cfg_name}")
        print(f"=======================================================")

        all_preds = []
        all_targets = []
        all_pred_lens = []
        all_tgt_lens = []
        all_scores = []

        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                (mz, inten, pm, pc, seq, comp, length, _, spec_mask) = [
                    t.to(device) if torch.is_tensor(t) else t for t in batch
                ]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens, pred_lens, pred_seqs, pred_scores = predict_peptide(
                        mz_array=mz,
                        intensity_array=inten,
                        precursor_mass=pm,
                        precursor_charge=pc,
                        mz_complementary=comp,
                        spectrum_mask=spec_mask,
                        vocabulary=vocabulary,
                        spectrum_encoder=spectrum_encoder,
                        length_predictor=length_predictor,
                        decoder=decoder,
                        guidance=guidance,
                        scheduler=cosine_scheduler,
                        num_steps=20,
                        guidance_scale=1.5,
                        top_k_lengths=3,
                        alpha=0.01,
                        beta=cfg["beta"],
                        decoding_strategy=cfg["decoding_strategy"],
                        temperature=cfg["temperature"],
                        trypsin_prior=cfg["trypsin_prior"],
                        mask_self_attention=False,
                        return_scores=True,
                    )
                all_preds.extend(pred_seqs)
                all_targets.extend(sequences_from_batch(seq, vocabulary))
                all_pred_lens.extend(pred_lens.tolist())
                all_tgt_lens.extend(length.tolist())
                all_scores.extend(pred_scores.tolist())

        elapsed = time.time() - t0
        throughput = len(all_targets) / elapsed

        mass_matches = [
            peptide_matches_mass_based(p, t, 0.1, 0.5)
            for p, t in zip(all_preds, all_targets)
        ]
        calib = calibrate_score_threshold(all_scores, mass_matches, target_precision=0.80)
        calibrated_threshold = calib["threshold"]

        metrics = compute_denovo_metrics(
            all_preds,
            all_targets,
            predicted_lengths=all_pred_lens,
            target_lengths=all_tgt_lens,
            scores=all_scores,
            score_threshold=calibrated_threshold,
            aa_mass_tolerance=0.1,
            prefix_mass_tolerance=0.5,
        )

        res = {
            "name": cfg_name,
            "elapsed_s": elapsed,
            "throughput_seq_per_s": throughput,
            "exact_acc": metrics.exact_peptide_accuracy,
            "exact_acc_il": metrics.exact_peptide_accuracy_il,
            "mass_acc": metrics.mass_peptide_accuracy,
            "length_acc": metrics.length_accuracy,
            "aa_precision": metrics.aa_precision,
            "aa_recall": metrics.aa_recall,
            "aa_f1": metrics.aa_f1,
            "auc_exact": metrics.auc_exact,
            "auc_exact_il": metrics.auc_exact_il,
            "auc_mass": metrics.auc_mass,
            "pauc80_mass": metrics.pauc80_mass,
            "prauc_exact": metrics.pr_auc_exact,
            "prauc_exact_il": metrics.pr_auc_exact_il,
            "prauc_mass": metrics.pr_auc_mass,
            "prauc80_mass": metrics.p_pr_auc80_mass,
            "calibrated_threshold": calibrated_threshold,
            "coverage_at_80prec": metrics.coverage,
            "mass_prec_at_80prec": metrics.peptide_precision_mass,
            "mass_rec_at_80prec": metrics.peptide_recall_mass,
        }
        all_results[cfg_name] = res

        print(f"Time: {elapsed:.2f}s ({throughput:.1f} spectra/s)")
        print(f"  Exact Accuracy (Strict):      {res['exact_acc']*100:.2f}%")
        print(f"  Exact Accuracy (I/L Equiv):   {res['exact_acc_il']*100:.2f}%")
        print(f"  Mass-Based Accuracy:          {res['mass_acc']*100:.2f}%")
        print(f"  Peptide Length Accuracy:      {res['length_acc']*100:.2f}%")
        print(f"  Amino Acid Precision:         {res['aa_precision']*100:.2f}%")
        print(f"  Amino Acid Recall:            {res['aa_recall']*100:.2f}%")
        print(f"  Amino Acid F1:                {res['aa_f1']*100:.2f}%")
        print(f"  PR-AUC (I/L Exact):           {res['prauc_exact_il']*100:.2f}%")
        print(f"  PR-AUC (Mass-based):          {res['prauc_mass']*100:.2f}%")
        print(f"  AUPCC Mass (Precision-Cov):   {res['auc_mass']*100:.2f}%")
        print(f"  pAUC80 Mass:                  {res['pauc80_mass']*100:.2f}%")
        print(f"  @ 80% Calibrated Target:")
        print(f"    Threshold:                  {res['calibrated_threshold']:.4f}")
        print(f"    Coverage:                   {res['coverage_at_80prec']*100:.2f}%")
        print(f"    Mass Precision:             {res['mass_prec_at_80prec']*100:.2f}%")
        print(f"    Mass Recall:                {res['mass_rec_at_80prec']*100:.2f}%")

    output_path = Path("artifacts/benchmark_val_subset_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {output_path}")
    return all_results


if __name__ == "__main__":
    run_benchmark(num_samples=1000)
