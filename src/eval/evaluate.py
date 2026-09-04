"""Run generative de novo evaluation on a dataloader."""

from __future__ import annotations

import torch

from data.data import decoder_output_token_ids, invert_vocabulary
from eval.metrics import DenovoMetrics, compute_denovo_metrics, format_metrics
from inference.predict import decode_tokens, predict_peptide


def sequences_from_batch(sequence: torch.Tensor, vocab: dict[str, int]) -> list[str]:
    return decode_tokens(sequence, vocab)


@torch.no_grad()
def evaluate_generative(
    loader,
    vocabulary: dict[str, int],
    spectrum_encoder,
    length_predictor,
    decoder,
    guidance,
    scheduler,
    device: torch.device,
    *,
    max_batches: int | None = None,
    num_steps: int = 20,
    noising_scheme: str = "mask",
    guidance_scale: float = 1.0,
    top_k_lengths: int = 3,
    alpha: float = 0.1,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
    amp: bool = True,
    score_threshold: float | None = None,
    return_details: bool = False,
) -> DenovoMetrics | tuple[DenovoMetrics, dict]:
    """Decode peptides with the full DFM inference loop and score against labels."""
    predictions: list[str] = []
    targets: list[str] = []
    predicted_lengths: list[int] = []
    target_lengths: list[int] = []
    scores: list[float] = []

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
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
        ) = tuple(
            tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in batch
        )

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp and device.type == "cuda"),
        ):
            token_ids, pred_lengths, pred_sequences, pred_scores = predict_peptide(
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
                scheduler=scheduler,
                num_steps=num_steps,
                noising_scheme=noising_scheme,
                guidance_scale=guidance_scale,
                top_k_lengths=top_k_lengths,
                alpha=alpha,
                return_scores=True,
            )

        predictions.extend(pred_sequences)
        targets.extend(sequences_from_batch(sequence, vocabulary))
        predicted_lengths.extend(int(value) for value in pred_lengths.tolist())
        target_lengths.extend(int(value) for value in length.tolist())
        scores.extend(float(s) for s in pred_scores.tolist())

        total_batches = len(loader) if hasattr(loader, "__len__") else None
        if (batch_idx + 1) % 10 == 0 or (total_batches is not None and (batch_idx + 1) == total_batches):
            total_str = f"/{total_batches}" if total_batches else ""
            print(f"[{batch_idx + 1}{total_str}] Processed {len(predictions)} peptides...", flush=True)

        del token_ids

    metrics = compute_denovo_metrics(
        predictions,
        targets,
        predicted_lengths=predicted_lengths,
        target_lengths=target_lengths,
        scores=scores,
        score_threshold=score_threshold,
        aa_mass_tolerance=aa_mass_tolerance,
        prefix_mass_tolerance=prefix_mass_tolerance,
    )

    if return_details:
        from eval.metrics import peptide_matches_mass_based
        exact_matches = [p == t for p, t in zip(predictions, targets)]
        mass_matches = [
            peptide_matches_mass_based(p, t, aa_mass_tolerance, prefix_mass_tolerance)
            for p, t in zip(predictions, targets)
        ]
        details = {
            "predictions": predictions,
            "targets": targets,
            "predicted_lengths": predicted_lengths,
            "target_lengths": target_lengths,
            "scores": scores,
            "exact_matches": exact_matches,
            "mass_matches": mass_matches,
        }
        return metrics, details

    return metrics


@torch.no_grad()
def evaluate_teacher_forced(
    peptide_logits: torch.Tensor,
    sequence: torch.Tensor,
    length_logits: torch.Tensor,
    length: torch.Tensor,
    active_mask: torch.Tensor,
    vocabulary: dict[str, int],
) -> dict[str, float]:
    """Fast proxy metrics from a single forward pass (not full generative decoding)."""
    pad_id = vocabulary["<pad>"]
    output_token_tensor = torch.tensor(
        decoder_output_token_ids(vocabulary), device=sequence.device, dtype=sequence.dtype
    )

    # ── Token accuracy (fully vectorised) ──────────────────────────────────
    pred_class = peptide_logits.argmax(dim=-1)            # (B, L)
    pred_tokens = output_token_tensor[pred_class]          # (B, L) vocab ids
    token_correct = (pred_tokens == sequence) & active_mask
    token_accuracy = token_correct.sum().item() / active_mask.sum().clamp(min=1).item()

    # ── Length accuracy ────────────────────────────────────────────────────
    pred_lengths = length_logits.argmax(dim=-1) + 1
    length_accuracy = (pred_lengths == length).float().mean().item()

    # ── Exact peptide accuracy (vectorised string assembly) ────────────────
    index_to_token = invert_vocabulary(vocabulary)
    predictions: list[str] = []
    targets: list[str] = []

    # Move to CPU once; iterate over rows only (not over tokens).
    pred_tokens_cpu = pred_tokens.cpu()
    sequence_cpu = sequence.cpu()
    active_mask_cpu = active_mask.cpu()

    for row_idx in range(sequence_cpu.shape[0]):
        mask_row = active_mask_cpu[row_idx]           # (L,) bool
        tgt_ids = sequence_cpu[row_idx][mask_row]     # active target ids
        prd_ids = pred_tokens_cpu[row_idx][mask_row]  # active pred ids

        targets.append("".join(
            index_to_token[t.item()] for t in tgt_ids if t.item() != pad_id
        ))
        predictions.append("".join(
            index_to_token.get(p.item(), "?") for p in prd_ids if p.item() != pad_id
        ))

    exact = sum(p == t for p, t in zip(predictions, targets))
    exact_accuracy = exact / len(targets) if targets else 0.0

    return {
        "token_accuracy": float(token_accuracy),
        "length_accuracy": float(length_accuracy),
        "exact_peptide_accuracy": float(exact_accuracy),
    }

