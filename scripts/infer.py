#!/usr/bin/env python3
"""
Entry point for de novo peptide inference.

NOTE: for large datasets (>10k spectra), batch-size 256 maximizes H100 throughput.
TODO: support reading raw .mgf / .mzML files directly from the command line.
"""

from __future__ import annotations

import argparse
import os

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(Path(__file__).resolve().parent)]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def move_batch_to_device(batch, device: torch.device):
    return tuple(
        tensor.to(device) if torch.is_tensor(tensor) else tensor
        for tensor in batch
    )

import torch

from functools import partial
import numpy as np
from torch.utils.data import DataLoader, Dataset

from data.constants import M_H
from data.data import build_dataloader, build_vocabulary, get_dataset, spectrum_collate
from flow_matching.scheduler import cosine_scheduler
from inference.predict import predict_peptide
from train.factory import build_models
from train.io import ensure_checkpoint, infer_model_config_from_checkpoint, load_checkpoint, load_models_from_checkpoint


class MGFInferenceDataset(Dataset):
    """Simple dataset for loading experimental MGF spectra for inference."""

    def __init__(self, mgf_path: str | Path, max_spectra: int | None = None, top_k: int = 200):
        import pyteomics.mgf
        self.spectra = []
        path = Path(mgf_path)
        if not path.exists():
            raise FileNotFoundError(f"Input spectra file not found: {mgf_path}")
        with pyteomics.mgf.read(str(path)) as reader:
            for idx, spec in enumerate(reader):
                if max_spectra is not None and idx >= max_spectra:
                    break
                params = spec.get("params", {})
                pepmass = params.get("pepmass", [0.0])
                precursor_mz = float(pepmass[0]) if isinstance(pepmass, (list, tuple)) else float(pepmass)
                charge_val = params.get("charge", [2])
                charge_str = str(charge_val[0] if isinstance(charge_val, (list, tuple)) else charge_val).rstrip("+")
                try:
                    charge = int(charge_str)
                except ValueError:
                    charge = 2

                mz_arr = spec["m/z array"].astype(np.float32)
                int_arr = spec["intensity array"].astype(np.float32)

                keep = np.abs(mz_arr - precursor_mz) > 1.5
                if np.any(keep):
                    mz_arr = mz_arr[keep]
                    int_arr = int_arr[keep]

                if len(mz_arr) == 0:
                    continue

                if len(mz_arr) > top_k:
                    top_idx = np.argsort(int_arr)[-top_k:]
                    mz_arr = mz_arr[top_idx]
                    int_arr = int_arr[top_idx]

                precursor_mass = float(precursor_mz * charge - charge * M_H)
                mz_tensor = torch.from_numpy(mz_arr)
                int_tensor = torch.from_numpy(int_arr)
                mz_comp = torch.from_numpy(precursor_mass + 2 * M_H - mz_arr)
                self.spectra.append((
                    mz_tensor,
                    int_tensor,
                    precursor_mass,
                    charge,
                    torch.tensor([], dtype=torch.long),
                    mz_comp,
                    0,
                ))

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DFM de novo peptide inference.")
    parser.add_argument("--input", default=os.environ.get("INPUT"), help="Optional path to custom MGF file for inference.")
    parser.add_argument("--split", default=os.environ.get("INFER_SPLIT", "test[:1%]"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NUM_STEPS", "20")))
    parser.add_argument(
        "--noising-scheme",
        choices=["uniform", "mask"],
        default=os.environ.get("NOISING_SCHEME", "mask"),
        help="Noising scheme: 'mask' (recommended) or 'uniform'.",
    )
    parser.add_argument(
        "--strategy",
        "--decoding-strategy",
        dest="decoding_strategy",
        choices=["confidence", "random"],
        default=os.environ.get("DECODING_STRATEGY", "confidence"),
        help="Unmasking strategy: 'confidence' (guided) or 'random'.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(os.environ.get("GUIDANCE_SCALE", "1.8")),
        help="Classifier-Free Guidance (CFG) scale (default: 1.8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.0")),
        help="Sampling temperature (default: 0.0 for greedy MAP).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=float(os.environ.get("ETA", "0.0")),
        help="Detailed balance re-masking stochasticity (default: 0.0).",
    )
    parser.add_argument(
        "--knapsack-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable exact dynamic programming knapsack mass pruning.",
    )
    parser.add_argument(
        "--knapsack-tol-da",
        type=float,
        default=float(os.environ.get("KNAPSACK_TOL_DA", "1.0")),
        help="Precursor mass tolerance for knapsack filter in Da (default: 1.0).",
    )
    parser.add_argument(
        "--peak-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable physical b/y-ion peak evidence guidance.",
    )
    parser.add_argument(
        "--top-k-lengths",
        type=int,
        default=int(os.environ.get("TOP_K_LENGTHS", "3")),
        help="Number of top length candidates to decode in parallel (default: 3).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=float(os.environ.get("LENGTH_BEAM_ALPHA", "0.5")),
        help="Weight for mass mismatch penalty in length beam score (default: 0.5).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=float(os.environ.get("LENGTH_BEAM_BETA", "0.5")),
        help="Weight for fragment ladder matching in candidate score (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT"),
        help="Optional path to save predictions to CSV or TSV file.",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max-batches", type=int, default=int(os.environ.get("MAX_BATCHES", "1")))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT"),
        help="Path to a training checkpoint (latest.pt or best_valid.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    vocabulary = None
    if checkpoint is not None:
        vocabulary = checkpoint.get("vocabulary")
        if vocabulary is None and args.checkpoint:
            ckpt_p = Path(args.checkpoint).resolve()
            for cand in [ckpt_p.parent / "vocabulary.json", ckpt_p.parent.parent / "vocabulary.json"]:
                if cand.exists():
                    import json
                    with cand.open() as f:
                        vocabulary = json.load(f)
                    break

    if args.input:
        if vocabulary is None:
            vocabulary = build_vocabulary(ds=None)
        dataset = MGFInferenceDataset(args.input)
        collate = partial(spectrum_collate, vocab=vocabulary, max_peptide_length=None)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    else:
        ds = get_dataset(split=args.split, cache_dir=args.cache_dir)
        if vocabulary is None:
            vocabulary = build_vocabulary(ds)
        loader = build_dataloader(ds, vocabulary, batch_size=args.batch_size, shuffle=False)

    model_cfg = infer_model_config_from_checkpoint(checkpoint) if checkpoint is not None else None
    spectrum_encoder, length_predictor, decoder, guidance = build_models(
        vocabulary, device, model_cfg=model_cfg
    )

    if checkpoint is not None:
        load_models_from_checkpoint(
            checkpoint,
            spectrum_encoder,
            length_predictor,
            decoder,
            guidance,
        )
        print(f"Loaded checkpoint: {args.checkpoint}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        (
            mz_array,
            intensity_array,
            precursor_mass,
            precursor_charge,
            _sequence,
            mz_complementary,
            _length,
            _padded_mask,
            spectrum_mask,
        ) = move_batch_to_device(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            token_ids, predicted_lengths, sequences = predict_peptide(
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
                num_steps=args.num_steps,
                noising_scheme=args.noising_scheme,
                guidance_scale=args.guidance_scale,
                top_k_lengths=args.top_k_lengths,
                alpha=args.alpha,
                beta=args.beta,
                decoding_strategy=args.decoding_strategy,
                temperature=args.temperature,
                eta=args.eta,
                use_knapsack_filter=args.knapsack_filter,
                knapsack_tol_da=args.knapsack_tol_da,
                use_peak_evidence=args.peak_evidence,
            )

        for idx, sequence in enumerate(sequences):
            out_line = (
                f"sample={batch_idx * args.batch_size + idx}\t"
                f"predicted_length={int(predicted_lengths[idx])}\t"
                f"sequence={sequence}"
            )
            print(out_line)
            if args.output:
                with open(args.output, "a") as f_out:
                    f_out.write(out_line + "\n")


if __name__ == "__main__":
    main()
