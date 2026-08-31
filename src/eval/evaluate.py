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
    num_steps: int = 50,
    noising_scheme: str = "uniform",
    guidance_scale: float = 1.0,
    aa_mass_tolerance: float = 0.1,
    prefix_mass_tolerance: float = 0.5,
) -> DenovoMetrics:
    """Decode peptides with the full DFM inference loop and score against labels."""
    predictions: list[str] = []
    targets: list[str] = []
    predicted_lengths: list[int] = []
    target_lengths: list[int] = []

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

        token_ids, pred_lengths, pred_sequences = predict_peptide(
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
        )

        predictions.extend(pred_sequences)
        targets.extend(sequences_from_batch(sequence, vocabulary))
        predicted_lengths.extend(int(value) for value in pred_lengths.tolist())
        target_lengths.extend(int(value) for value in length.tolist())

        del token_ids

    return compute_denovo_metrics(
        predictions,
        targets,
        predicted_lengths=predicted_lengths,
        target_lengths=target_lengths,
        aa_mass_tolerance=aa_mass_tolerance,
        prefix_mass_tolerance=prefix_mass_tolerance,
    )


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
        decoder_output_token_ids, device=sequence.device, dtype=sequence.dtype
    )
    pred_class = peptide_logits.argmax(dim=-1)
    pred_tokens = output_token_tensor[pred_class]
    token_correct = (pred_tokens == sequence) & active_mask
    token_accuracy = token_correct.sum().item() / active_mask.sum().clamp(min=1).item()

    pred_lengths = length_logits.argmax(dim=-1) + 1
    length_accuracy = (pred_lengths == length).float().mean().item()

    index_to_token = invert_vocabulary(vocabulary)
    predictions = []
    targets = []
    for row_idx in range(sequence.shape[0]):
        pred_chars = []
        target_chars = []
        for col_idx in range(sequence.shape[1]):
            if not active_mask[row_idx, col_idx]:
                continue
            target_id = int(sequence[row_idx, col_idx].item())
            pred_id = int(pred_tokens[row_idx, col_idx].item())
            if target_id == pad_id:
                continue
            target_chars.append(index_to_token[target_id])
            if pred_id != pad_id:
                pred_chars.append(index_to_token.get(pred_id, "?"))
        predictions.append("".join(pred_chars))
        targets.append("".join(target_chars))

    exact = sum(pred == target for pred, target in zip(predictions, targets))
    exact_accuracy = exact / len(targets) if targets else 0.0

    return {
        "token_accuracy": float(token_accuracy),
        "length_accuracy": float(length_accuracy),
        "exact_peptide_accuracy": float(exact_accuracy),
    }
