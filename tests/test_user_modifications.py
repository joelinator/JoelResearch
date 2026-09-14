"""
Unit tests for user modifications:
1. Cumulative mass loss (mass_loss_hubert_cum)
2. DecoderBlock and DFMPeptideDecoder self-attention key padding masking
3. Lightning checkpoint resumption with default learning rate
"""

import torch
from data.data import build_vocabulary, get_output_aa_masses
from data.lengths import length_to_active_mask
from model.model import DecoderBlock
from train.loss import mass_loss_hubert_cum
from train.lightning import DFMLightningModule


def test_cumulative_mass_loss_padding_safety():
    vocab = build_vocabulary({"sequence": ["ACDEFGHIKLMNPQRSTVWY"]})
    aa_masses = get_output_aa_masses(vocab)  # shape (20,)
    pad_id = vocab["<pad>"]
    mask_id = vocab.get("<mask>", vocab.get("<mask_token>"))

    # Batch with active tokens and special tokens
    true_seq = torch.tensor([
        [0, 1, 2, pad_id, pad_id],
        [3, 4, pad_id, pad_id, pad_id],
        [5, mask_id, pad_id, pad_id, pad_id],
    ], dtype=torch.long)
    lengths = torch.tensor([3, 2, 2], dtype=torch.long)
    active_mask = length_to_active_mask(lengths, 5)

    logits = torch.randn(3, 5, len(aa_masses), requires_grad=True)

    # Must NOT raise IndexError on pad_id (20) or mask_id (21)
    loss = mass_loss_hubert_cum(
        logits=logits,
        aa_masses=aa_masses,
        true_sequence=true_seq,
        active_mask=active_mask,
    )

    assert torch.isfinite(loss).item(), "Loss must be finite"
    assert loss.item() > 0.0, "Loss must be positive"

    loss.backward()
    assert logits.grad is not None, "Gradients must exist"
    assert not torch.isnan(logits.grad).any().item(), "No NaNs in gradients"

    # Gradients on padded positions must be zero
    padded_positions = ~active_mask
    assert torch.all(logits.grad[padded_positions] == 0.0).item(), (
        "Gradients on padded positions must be exactly 0"
    )
    print("test_cumulative_mass_loss_padding_safety passed.")


def test_cumulative_mass_loss_invariance_to_pad_values():
    """Varying token IDs at padded positions must not affect the computed loss."""
    vocab = build_vocabulary({"sequence": ["ACDEFGHIKLMNPQRSTVWY"]})
    aa_masses = get_output_aa_masses(vocab)
    pad_id = vocab["<pad>"]

    seq1 = torch.tensor([[0, 1, pad_id, pad_id]], dtype=torch.long)
    seq2 = torch.tensor([[0, 1, 5, 12]], dtype=torch.long)  # different tokens in padding
    active_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

    logits = torch.randn(1, 4, len(aa_masses))

    loss1 = mass_loss_hubert_cum(logits, aa_masses, seq1, active_mask=active_mask)
    loss2 = mass_loss_hubert_cum(logits, aa_masses, seq2, active_mask=active_mask)

    assert torch.isclose(loss1, loss2, atol=1e-6), "Loss must be invariant to values at padded positions"
    print("test_cumulative_mass_loss_invariance_to_pad_values passed.")


def test_decoder_self_attention_masking():
    """Masked keys in self-attention must not affect representations of active positions."""
    torch.manual_seed(42)
    block = DecoderBlock(emb_dim=64, num_heads=4, mlp_hidden_dim=128, dropout=0.0)
    block.eval()

    cond = torch.randn(1, 64)
    x = torch.randn(1, 4, 64)
    y = torch.randn(1, 8, 64)

    # Active length is 2, padding tail is 2
    mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)

    # Pass 1: standard x
    out1 = block(cond, x.clone(), y, sequence_padding_mask=mask)

    # Pass 2: perturb the padded positions of x
    x_perturbed = x.clone()
    x_perturbed[:, 2:, :] += 100.0  # huge perturbation on padding
    out2 = block(cond, x_perturbed, y, sequence_padding_mask=mask)

    # The active positions (0, 1) MUST NOT change despite perturbation on padded keys
    diff = torch.abs(out1[:, :2, :] - out2[:, :2, :]).max().item()
    assert diff < 1e-5, f"Self-attention leaked information from masked keys! diff={diff}"
    print("test_decoder_self_attention_masking passed.")


def test_lightning_reset_lr_on_resume():
    """Verify on_load_checkpoint correctly resets optimizer/scheduler states when enabled."""
    vocab = build_vocabulary({"sequence": ["ACDEFGHIKLMNPQRSTVWY"]})
    args = {"learning_rate": 6e-4, "reset_lr_on_resume": True}
    model = DFMLightningModule(vocab, args)

    dummy_checkpoint = {
        "optimizer_states": [{"param_groups": [{"lr": 1e-6}]}],
        "lr_schedulers": [{"_step_count": 50000}],
    }

    model.on_load_checkpoint(dummy_checkpoint)
    assert dummy_checkpoint["optimizer_states"] == [], "Optimizer states must be emptied to reset LR"
    assert dummy_checkpoint["lr_schedulers"] == [], "LR schedulers must be emptied to reset schedule"

    # Test disabled reset
    args_keep = {"learning_rate": 6e-4, "reset_lr_on_resume": False}
    model_keep = DFMLightningModule(vocab, args_keep)
    dummy_checkpoint_keep = {
        "optimizer_states": [{"param_groups": [{"lr": 1e-6}]}],
        "lr_schedulers": [{"_step_count": 50000}],
    }
    model_keep.on_load_checkpoint(dummy_checkpoint_keep)
    assert len(dummy_checkpoint_keep["optimizer_states"]) == 1, "Optimizer states should be preserved"
    assert len(dummy_checkpoint_keep["lr_schedulers"]) == 1, "LR schedulers should be preserved"
    print("test_lightning_reset_lr_on_resume passed.")


def test_pad_token_mass_accuracy():
    """Verify that pad and mask tokens have exactly 0.0 mass at their correct indices."""
    from data.data import get_aa_masses
    vocab = build_vocabulary({"sequence": ["ACDEFGHIKLMNPQRSTVWY"]})
    full_masses = get_aa_masses(vocab)
    pad_id = vocab["<pad>"]
    mask_id = vocab.get("<mask" + ">", vocab.get("<mask_token>"))

    assert full_masses[pad_id].item() == 0.0, "Pad token must have 0.0 mass"
    assert full_masses[mask_id].item() == 0.0, "Mask token must have 0.0 mass"

    output_masses = get_output_aa_masses(vocab)
    logits = torch.randn(2, 6, 20)
    seq = torch.tensor([
        [0, 1, 2, pad_id, pad_id, pad_id],
        [3, 4, 5, 6, pad_id, pad_id],
    ])
    active = torch.tensor([
        [True, True, True, False, False, False],
        [True, True, True, True, False, False],
    ])

    # Test with explicit vocab_aa_masses
    loss_explicit = mass_loss_hubert_cum(logits, output_masses, seq, active_mask=active, vocab_aa_masses=full_masses)
    # Test with automatic padding
    loss_auto = mass_loss_hubert_cum(logits, output_masses, seq, active_mask=active)

    assert torch.isclose(loss_explicit, loss_auto), "Explicit vocab_aa_masses and auto-padded masses must match"
    print("test_pad_token_mass_accuracy passed.")


if __name__ == "__main__":
    test_cumulative_mass_loss_padding_safety()
    test_cumulative_mass_loss_invariance_to_pad_values()
    test_decoder_self_attention_masking()
    test_lightning_reset_lr_on_resume()
    test_pad_token_mass_accuracy()
    print("All user modification verification tests passed successfully!")
