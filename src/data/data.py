import os
from functools import partial

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .constants import AA_MASSES_DICT, M_H, STANDARD_AMINO_ACIDS
from .lengths import validate_peptide_length

DEFAULT_DATASET = "InstaDeepAI/ms_proteometools"
DEFAULT_SPLIT = "train"

# HuggingFace column names -> internal names used by the training code.
HF_COLUMN_MAP = {
    "precursor_mz": "precursor_mass",
    "charge": "precursor_charge",
}


def precursor_mass_from_mz(precursor_mz: float, charge: int) -> float:
    """Convert precursor m/z to neutral monoisotopic mass."""
    return precursor_mz * charge - charge * M_H


def get_dataset(
    repo_id: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    cache_dir: str | None = None,
    token: str | None = None,
):
    """
    Load the ProteomeTools dataset from HuggingFace.

    The raw dataset exposes `precursor_mz` and `charge`; we derive `precursor_mass`
    so the rest of the pipeline can keep using mass-based conditioning.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir, token=token)

    if "precursor_mass" not in ds.column_names and "precursor_mz" in ds.column_names:
        ds = ds.map(
            lambda row: {
                "precursor_mass": precursor_mass_from_mz(
                    row["precursor_mz"], int(row["charge"])
                )
            },
            desc="Deriving precursor_mass from precursor_mz",
        )

    if "precursor_charge" not in ds.column_names and "charge" in ds.column_names:
        ds = ds.rename_column("charge", "precursor_charge")

    keep_columns = [
        "mz_array",
        "intensity_array",
        "precursor_mass",
        "precursor_charge",
        "sequence",
    ]
    existing = [column for column in keep_columns if column in ds.column_names]
    ds = ds.select_columns(existing)
    return ds


def build_vocabulary(ds):
    """Build token vocabulary with stable special tokens and standard amino acids."""
    tokens = set(STANDARD_AMINO_ACIDS)
    for sequence in ds["sequence"]:
        tokens.update(sequence)

    vocab = {token: index for index, token in enumerate(sorted(tokens))}
    vocab["<pad>"] = len(vocab)
    vocab["<mask>"] = len(vocab) + 1
    return vocab


def invert_vocabulary(vocab: dict[str, int]) -> dict[int, str]:
    return {index: token for token, index in vocab.items()}


def decoder_output_token_ids(vocab: dict[str, int]) -> list[int]:
    """Token ids the decoder may predict: amino acids only (no <pad>, no <mask>)."""
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask>"]
    special = {pad_id, mask_id}
    return sorted(token_id for token_id in vocab.values() if token_id not in special)


def decoder_output_size(vocab: dict[str, int]) -> int:
    return len(decoder_output_token_ids(vocab))


def get_aa_masses(vocab):
    masses = []
    for token, _ in sorted(vocab.items(), key=lambda item: item[1]):
        if token not in AA_MASSES_DICT:
            raise KeyError(
                f"Amino acid '{token}' is missing from AA_MASSES_DICT. "
                "Add its monoisotopic mass or filter the vocabulary."
            )
        masses.append(AA_MASSES_DICT[token])
    return torch.tensor(masses, dtype=torch.float32)


def get_output_aa_masses(vocab: dict[str, int]) -> torch.Tensor:
    """Per-residue masses aligned with decoder output logits (amino acids only)."""
    masses = get_aa_masses(vocab)
    return masses[decoder_output_token_ids(vocab)]

class SpectrumDataSet(Dataset):
    def __init__(self, data, vocab, top_k: int = 200, remove_precursor_peak: bool = True):
        self.data = data
        self.top_k = top_k
        self.remove_precursor_peak = remove_precursor_peak
        self.vocab = vocab

    def __getitem__(self, idx):
        row = self.data[idx]
        mz_array = torch.tensor(row["mz_array"], dtype=torch.float32)
        intensity_array = torch.tensor(row["intensity_array"], dtype=torch.float32)
        precursor_mass = float(row["precursor_mass"])
        precursor_charge = int(row["precursor_charge"])

        if self.remove_precursor_peak:
            tol = 1e-3
            keep = (mz_array - precursor_mass).abs() > tol
            mz_array = mz_array[keep]
            intensity_array = intensity_array[keep]

        if self.top_k is not None:
            k = min(self.top_k, intensity_array.shape[0])
            if k > 0:
                intensity_array, indices = torch.topk(intensity_array, k)
                mz_array = mz_array[indices]

        sequence = torch.tensor(
            [self.vocab[residue] for residue in row["sequence"]],
            dtype=torch.long,
        )
        length = len(row["sequence"])
        validate_peptide_length(length)
        mz_complementary = precursor_mass + 2 * M_H - mz_array

        return (
            mz_array,
            intensity_array,
            precursor_mass,
            precursor_charge,
            sequence,
            mz_complementary,
            length,
        )

    def __len__(self):
        return len(self.data)


def spectrum_collate(batch_data, vocab):
    (
        mz_array,
        intensity_array,
        precursor_mass,
        precursor_charge,
        sequence,
        mz_complementary,
        length,
    ) = zip(*batch_data)

    mz_array = pad_sequence(
        mz_array, batch_first=True, padding_value=-1.0, padding_side="right"
    )
    mz_complementary = pad_sequence(
        mz_complementary, batch_first=True, padding_value=-1.0, padding_side="right"
    )
    intensity_array = pad_sequence(
        intensity_array, batch_first=True, padding_value=-1.0, padding_side="right"
    )

    spectrum_mask = intensity_array == -1
    precursor_mass = torch.tensor(precursor_mass, dtype=torch.float32)
    precursor_charge = torch.tensor(precursor_charge, dtype=torch.long)
    sequence = pad_sequence(
        sequence, batch_first=True, padding_value=vocab["<pad>"], padding_side="right"
    )
    padded_mask = sequence == vocab["<pad>"]

    return (
        mz_array,
        intensity_array,
        precursor_mass,
        precursor_charge,
        sequence,
        mz_complementary,
        torch.tensor(length, dtype=torch.long),
        padded_mask,
        spectrum_mask,
    )


def build_dataloader(ds, vocab, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    dataset = SpectrumDataSet(ds, vocab)
    collate_fn = partial(spectrum_collate, vocab=vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
