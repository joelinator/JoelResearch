#!/usr/bin/env python3
"""Systematic hyperparameter tuning for de novo generative evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
import torch

from bootstrap import setup_src_path

setup_src_path()

from config.defaults import DEFAULTS
from data.data import build_dataloader, build_vocabulary, get_dataset
from eval.evaluate import evaluate_generative
from flow_matching.scheduler import cosine_scheduler
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint


def run_eval_on_batches(
    batches,
    vocabulary,
    spectrum_encoder,
    length_predictor,
    decoder,
    guidance,
    device,
    *,
    num_steps: int = 20,
    guidance_scale: float = 1.0,
    top_k_lengths: int = 3,
    alpha: float = 0.01,
):
    from inference.predict import predict_peptide
    from eval.metrics import compute_denovo_metrics
    from data.data import decoder_output_token_ids, invert_vocabulary

    predictions = []
    targets = []
    pred_lengths_all = []
    target_lengths_all = []

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch in batches:
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
        ) = batch

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            token_ids, pred_lens, pred_seqs = predict_peptide(
                mz_array=mz_array,
                intensity_array=intensity_array,
                precursor_mass=precursor_mass,
                precursor_charge=precursor_charge,
                mz_complementary=mz_complementary,
                spectrum_mask=spectrum_mask,
                vocabulary=vocabulary,
                spectrum_encoder=spectrum_encoder,
                length_predictor=length_predictor,
                decoder=decoder,
                guidance=guidance,
                scheduler=cosine_scheduler,
                num_steps=num_steps,
                noising_scheme="mask",
                guidance_scale=guidance_scale,
                top_k_lengths=top_k_lengths,
                alpha=alpha,
            )

        from eval.evaluate import sequences_from_batch
        predictions.extend(pred_seqs)
        targets.extend(sequences_from_batch(sequence, vocabulary))
        pred_lengths_all.extend(int(v) for v in pred_lens.tolist())
        target_lengths_all.extend(int(v) for v in length.tolist())

    return compute_denovo_metrics(
        predictions,
        targets,
        predicted_lengths=pred_lengths_all,
        target_lengths=target_lengths_all,
        aa_mass_tolerance=0.1,
        prefix_mass_tolerance=0.5,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"Loading validation dataset for hyperparameter tuning on {device}...")
    ds = get_dataset(split="validation", cache_dir="data/cache")
    vocab_path = Path("artifacts/dfm_pl_run_20260903_092757/vocabulary.json")
    with open(vocab_path) as f:
        vocabulary = json.load(f)

    # Pre-load 5 validation batches (10,240 spectra) directly into GPU memory
    loader = build_dataloader(ds, vocabulary, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)
    cached_batches = []
    print("Pre-caching 5 batches (10,240 spectra)...")
    for idx, batch in enumerate(loader):
        if idx >= 5:
            break
        cached_batches.append(
            tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
        )
    print(f"Cached {len(cached_batches)} batches ({len(cached_batches) * 2048} spectra).")

    ckpt_base = Path("artifacts/dfm_pl_run_20260903_092757/checkpoints")
    ckpt_last = str(ckpt_base / "last.ckpt")  # Epoch 19
    ckpt_e17 = str(ckpt_base / "best-epoch=17-valid/loss=0.4756.ckpt")

    models_cache = {}

    def get_models(ckpt_path):
        if ckpt_path not in models_cache:
            print(f"Loading checkpoint {ckpt_path}...")
            spec_enc, len_pred, dec, guid = build_models(vocabulary, device)
            ckpt = load_checkpoint(ckpt_path, map_location=device)
            load_models_from_checkpoint(ckpt, spec_enc, len_pred, dec, guid)
            models_cache[ckpt_path] = (spec_enc, len_pred, dec, guid)
        return models_cache[ckpt_path]

    # Hyperparameter trials
    trials = [
        # 1. Baseline: Checkpoint Epoch 19 vs Epoch 17
        {"name": "Epoch 19 (Baseline)", "ckpt": ckpt_last, "guidance": 1.0, "steps": 20, "top_k": 3, "alpha": 0.01},
        {"name": "Epoch 17", "ckpt": ckpt_e17, "guidance": 1.0, "steps": 20, "top_k": 3, "alpha": 0.01},

        # 2. Guidance scale sweep (on Epoch 19)
        {"name": "Epoch 19, CFG=1.2", "ckpt": ckpt_last, "guidance": 1.2, "steps": 20, "top_k": 3, "alpha": 0.01},
        {"name": "Epoch 19, CFG=1.35", "ckpt": ckpt_last, "guidance": 1.35, "steps": 20, "top_k": 3, "alpha": 0.01},
        {"name": "Epoch 19, CFG=1.5", "ckpt": ckpt_last, "guidance": 1.5, "steps": 20, "top_k": 3, "alpha": 0.01},

        # 3. Steps sweep
        {"name": "Epoch 19, Steps=15", "ckpt": ckpt_last, "guidance": 1.0, "steps": 15, "top_k": 3, "alpha": 0.01},
        {"name": "Epoch 19, Steps=25", "ckpt": ckpt_last, "guidance": 1.0, "steps": 25, "top_k": 3, "alpha": 0.01},
        {"name": "Epoch 19, Steps=30", "ckpt": ckpt_last, "guidance": 1.0, "steps": 30, "top_k": 3, "alpha": 0.01},

        # 4. Top-K candidates sweep
        {"name": "Epoch 19, Top-K=1", "ckpt": ckpt_last, "guidance": 1.0, "steps": 20, "top_k": 1, "alpha": 0.01},
        {"name": "Epoch 19, Top-K=4", "ckpt": ckpt_last, "guidance": 1.0, "steps": 20, "top_k": 4, "alpha": 0.01},

        # 5. Mass penalty alpha sweep
        {"name": "Epoch 19, Alpha=0.005", "ckpt": ckpt_last, "guidance": 1.0, "steps": 20, "top_k": 3, "alpha": 0.005},
        {"name": "Epoch 19, Alpha=0.02", "ckpt": ckpt_last, "guidance": 1.0, "steps": 20, "top_k": 3, "alpha": 0.02},
    ]

    results = []

    print("\n" + "=" * 90)
    print(f"{'Trial Name':<25} | {'Exact Match':<11} | {'Pep F1':<8} | {'AA F1':<8} | {'Len Acc':<8} | {'Time (s)':<8}")
    print("=" * 90)

    for trial in trials:
        spec_enc, len_pred, dec, guid = get_models(trial["ckpt"])
        t0 = time.time()
        metrics = run_eval_on_batches(
            cached_batches,
            vocabulary,
            spec_enc,
            len_pred,
            dec,
            guid,
            device,
            num_steps=trial["steps"],
            guidance_scale=trial["guidance"],
            top_k_lengths=trial["top_k"],
            alpha=trial["alpha"],
        )
        elapsed = time.time() - t0

        m_dict = metrics.to_dict()
        exact = m_dict["exact_peptide_accuracy"]
        pep_f1 = m_dict["peptide_f1"]
        aa_f1 = m_dict["aa_f1"]
        len_acc = m_dict["length_accuracy"]

        print(f"{trial['name']:<25} | {exact*100:>10.2f}% | {pep_f1*100:>7.2f}% | {aa_f1*100:>7.2f}% | {len_acc*100:>7.2f}% | {elapsed:>7.1f}s", flush=True)

        results.append({
            "trial": trial,
            "metrics": m_dict,
            "time_seconds": elapsed,
        })

    # Sort results by exact match accuracy
    results.sort(key=lambda r: r["metrics"]["exact_peptide_accuracy"], reverse=True)
    best = results[0]

    out_file = Path("artifacts/dfm_pl_run_20260903_092757/hyperparameter_tuning_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"BEST CONFIGURATION: {best['trial']['name']}")
    print(f"Exact Peptide Accuracy: {best['metrics']['exact_peptide_accuracy']*100:.2f}%")
    print(f"Peptide F1:             {best['metrics']['peptide_f1']*100:.2f}%")
    print(f"Amino Acid F1:          {best['metrics']['aa_f1']*100:.2f}%")
    print(f"Length Accuracy:        {best['metrics']['length_accuracy']*100:.2f}%")
    print(f"Parameters: steps={best['trial']['steps']}, guidance={best['trial']['guidance']}, top_k={best['trial']['top_k']}, alpha={best['trial']['alpha']}")
    print(f"Saved tuning log to {out_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
