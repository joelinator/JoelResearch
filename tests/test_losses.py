import torch
import pytest
from src.train.loss import complementary_ion_loss, length_weighted_peptide_loss


def test_complementary_ion_loss_perfect_symmetry():
    # 20 standard amino acid masses
    from src.data.constants import AA_MASSES_DICT, STANDARD_AMINO_ACIDS
    aa_masses = torch.tensor([AA_MASSES_DICT[aa] for aa in STANDARD_AMINO_ACIDS], dtype=torch.float32)

    B, L, C = 2, 10, len(STANDARD_AMINO_ACIDS)
    # Perfect logits predicting alanine (mass 71.03711) at all positions
    logits = torch.zeros(B, L, C)
    ala_idx = STANDARD_AMINO_ACIDS.index("A")
    logits[:, :, ala_idx] = 20.0  # very confident

    # 10 alanines: 10 * 71.03711 + 18.010565
    precursor_mass = torch.tensor([10 * 71.03711 + 18.010565] * B, dtype=torch.float32)

    loss = complementary_ion_loss(logits, aa_masses, precursor_mass)
    assert loss.item() >= 0.0
    assert loss.item() < 0.01  # Should be near zero for perfect mass match


def test_complementary_ion_loss_nonzero_on_mismatch():
    from src.data.constants import AA_MASSES_DICT, STANDARD_AMINO_ACIDS
    aa_masses = torch.tensor([AA_MASSES_DICT[aa] for aa in STANDARD_AMINO_ACIDS], dtype=torch.float32)

    B, L, C = 2, 10, len(STANDARD_AMINO_ACIDS)
    logits = torch.zeros(B, L, C)
    # Target precursor mass is completely off by 500 Da
    precursor_mass = torch.tensor([1500.0, 1500.0], dtype=torch.float32)

    loss = complementary_ion_loss(logits, aa_masses, precursor_mass)
    assert loss.item() > 0.05


def test_length_weighted_peptide_loss():
    B, L, C = 4, 15, 20
    logits = torch.randn(B, L, C)
    targets = torch.randint(0, 20, (B, L))
    lengths = torch.tensor([8, 12, 15, 25], dtype=torch.long)
    pad_id = 20

    loss = length_weighted_peptide_loss(logits, targets, lengths, pad_id=pad_id, alpha=0.5)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
