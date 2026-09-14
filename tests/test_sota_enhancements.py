"""
Tests for SOTA Enhancements in DFM De Novo Peptide Sequencing:
1. Confidence-based rank-ordered discrete flow unmasking and temperature control
2. Theoretical b/y fragment ion matching & spectral intensity scoring
3. Spectrum peak dropout data augmentation
4. Label smoothing in peptide cross-entropy loss
5. Exponential Moving Average (EMA) callback & parameter swapping
6. I/L equivalence in exact match evaluation metrics
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.constants import AA_MASSES_DICT, M_H, M_H2O
from data.data import SpectrumDataSet
from eval.metrics import compute_denovo_metrics
from flow_matching.sampling import inference_sample_mask
from flow_matching.scheduler import linear_scheduler
from inference.predict import compute_fragment_matching_scores
from train.callbacks import EMACallback
from train.loss import peptide_loss


def _make_dummy_vocab():
    aa_tokens = list(AA_MASSES_DICT.keys())
    vocab = {aa: i for i, aa in enumerate(aa_tokens)}
    vocab["<pad>"] = len(vocab)
    vocab["<mask_token>"] = len(vocab)
    return vocab


def test_confidence_based_unmasking():
    vocab = _make_dummy_vocab()
    mask_token_id = vocab["<mask_token>"]
    pad_id = vocab["<pad>"]

    # 1 sequence of length 5 (indices 0..4 active, index 5 padded)
    B, L, V = 1, 6, len(vocab)
    active_mask = torch.tensor([[True, True, True, True, True, False]])
    x_t = torch.tensor([[mask_token_id, mask_token_id, mask_token_id, mask_token_id, mask_token_id, pad_id]])

    # Design logits such that position 2 has the highest confidence (sharp peak), position 0 is next
    logits = torch.randn(B, L, V) * 0.1
    # Position 2: highest confidence on token 3
    logits[0, 2, 3] = 10.0
    # Position 0: second highest confidence on token 1
    logits[0, 0, 1] = 5.0
    # Position 1, 3, 4: low confidence
    logits[0, 1, :] = 0.0
    logits[0, 3, :] = 0.0
    logits[0, 4, :] = 0.0

    scheduler = linear_scheduler
    t = torch.tensor([0.2])
    kt, kt_derivative = scheduler(t)
    delta_t = 0.2

    # Unmask with confidence strategy
    x_next = inference_sample_mask(
        kt=kt,
        kt_derivative=kt_derivative,
        x_t=x_t.clone(),
        logits=logits,
        vocab=vocab,
        delta_t=delta_t,
        active_mask=active_mask,
        temperature=0.0,
        strategy="confidence",
    )

    # Position 2 had highest confidence, must be unmasked first to token 3
    assert x_next[0, 2].item() == 3, f"Expected pos 2 to be unmasked to 3, got {x_next[0, 2].item()}"
    # Padded position must remain pad
    assert x_next[0, 5].item() == pad_id, "Padded position was modified!"
    print("test_confidence_based_unmasking passed.")


def test_fragment_ion_matching_scores():
    vocab = _make_dummy_vocab()
    # Peptide: 'A' (71.03711) and 'C' (103.00919) -> length 2
    # b1 ion: m(A) + M_H = 71.03711 + 1.007276 = 72.044386
    # y1 ion: m(C) + M_H2O + M_H = 103.00919 + 18.010565 + 1.007276 = 122.027031
    a_id = vocab["A"]
    c_id = vocab["C"]
    pad_id = vocab["<pad>"]

    x_t = torch.tensor([[a_id, c_id, pad_id]])
    active_mask = torch.tensor([[True, True, False]])

    # Synthetic experimental peaks: exactly match b1 and y1 with high intensity, plus 1 unmatched peak
    b1_mz = 71.03711 + M_H
    y1_mz = 103.00919 + M_H2O + M_H
    unmatched_mz = 500.0

    mz_exp = torch.tensor([[b1_mz, y1_mz, unmatched_mz, 0.0]])
    intensity_exp = torch.tensor([[50.0, 50.0, 100.0, 0.0]])
    spectrum_mask = torch.tensor([[False, False, False, True]])  # 4th peak is padded

    score = compute_fragment_matching_scores(
        x_t=x_t,
        active_mask=active_mask,
        mz_array_exp=mz_exp,
        intensity_array_exp=intensity_exp,
        spectrum_mask_exp=spectrum_mask,
        vocabulary=vocab,
        tolerance_ppm=20.0,
        tolerance_da=0.05,
    )

    # Matched intensity = 50 + 50 = 100. Total valid intensity = 200. Expected ratio = 0.5
    assert abs(score[0].item() - 0.5) < 1e-4, f"Expected explained intensity 0.5, got {score[0].item()}"
    print("test_fragment_ion_matching_scores passed.")


def test_peak_dropout_augmentation():
    vocab = _make_dummy_vocab()
    raw_data = [{
        "mz_array": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "intensity_array": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0],  # 100 is base peak
        "precursor_mass": 1000.0,
        "precursor_charge": 2,
        "sequence": "PEPTIDE",
    }]

    # In train mode with peak dropout 0.5:
    train_ds = SpectrumDataSet(raw_data, vocab, top_k=8, remove_precursor_peak=False, peak_dropout_prob=0.5, is_train=True)
    # Sample multiple times to verify dropout occurs
    zero_counts = 0
    for _ in range(20):
        _, int_arr, _, _, _, _, _ = train_ds[0]
        zero_counts += int((int_arr == 0.0).sum().item())
    assert zero_counts > 0, "Expected some peaks to be dropped during training!"

    # In eval mode (is_train=False): NO peaks dropped
    eval_ds = SpectrumDataSet(raw_data, vocab, top_k=8, remove_precursor_peak=False, peak_dropout_prob=0.5, is_train=False)
    for _ in range(5):
        _, int_arr, _, _, _, _, _ = eval_ds[0]
        assert (int_arr == 0.0).sum().item() == 0, "No peaks should be dropped when is_train=False!"
    print("test_peak_dropout_augmentation passed.")


def test_label_smoothing_loss():
    logits = torch.randn(2, 4, 20, requires_grad=True)
    peptide = torch.tensor([[1, 2, 3, 20], [4, 5, 20, 20]])  # 20 is pad_id
    pad_id = 20

    loss_no_smooth = peptide_loss(logits, peptide, pad_id=pad_id, label_smoothing=0.0)
    loss_smooth = peptide_loss(logits, peptide, pad_id=pad_id, label_smoothing=0.1)

    assert loss_no_smooth.item() > 0.0
    assert loss_smooth.item() > 0.0
    # Both losses should be differentiable
    loss_smooth.backward()
    assert logits.grad is not None
    print("test_label_smoothing_loss passed.")


def test_ema_callback_mechanics():
    # Simple toy module
    class ToyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        @property
        def device(self):
            return self.linear.weight.device

    model = ToyModule()
    ema = EMACallback(decay=0.9, update_interval=1)

    # Initialize shadow params
    ema._init_shadow_params(model)
    orig_weight = model.linear.weight.data.clone()

    # Change weights as if a training step occurred
    with torch.no_grad():
        model.linear.weight.data.fill_(10.0)

    # Simulate batch end
    class FakeTrainer:
        global_step = 0

    ema.on_train_batch_end(FakeTrainer(), model, None, None, 0)
    expected_shadow = 0.9 * orig_weight + 0.1 * 10.0
    assert torch.allclose(ema.shadow_params["linear.weight"], expected_shadow, atol=1e-5)

    # Test validation swap
    ema.on_validation_start(FakeTrainer(), model)
    assert torch.allclose(model.linear.weight.data, expected_shadow, atol=1e-5), "EMA weights were not swapped in for validation!"

    # Test restoration after validation
    ema.on_validation_end(FakeTrainer(), model)
    assert torch.allclose(model.linear.weight.data, torch.tensor(10.0), atol=1e-5), "Original training weights were not restored after validation!"
    print("test_ema_callback_mechanics passed.")


def test_il_equivalence_metric():
    # Predictions differ from targets ONLY by Leucine vs Isoleucine
    preds = ["PEPTLDE", "SAMPLER"]
    targets = ["PEPTIDE", "SAMPLER"]

    metrics = compute_denovo_metrics(preds, targets)

    # Strict match: 1 out of 2 is correct = 50%
    assert abs(metrics.exact_peptide_accuracy - 0.5) < 1e-5, f"Expected 0.5 strict accuracy, got {metrics.exact_peptide_accuracy}"
    # I/L conflated match: both are correct = 100%
    assert abs(metrics.exact_peptide_accuracy_il - 1.0) < 1e-5, f"Expected 1.0 I/L accuracy, got {metrics.exact_peptide_accuracy_il}"
    print("test_il_equivalence_metric passed.")


if __name__ == "__main__":
    test_confidence_based_unmasking()
    test_fragment_ion_matching_scores()
    test_peak_dropout_augmentation()
    test_label_smoothing_loss()
    test_ema_callback_mechanics()
    test_il_equivalence_metric()
    print("All SOTA enhancement tests passed successfully!")
