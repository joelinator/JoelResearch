import os
import re
from functools import partial

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .constants import (
    AA_MASSES_DICT,
    M_H,
    PTM_ALIAS_MAP,
    STANDARD_AMINO_ACIDS,
    TARGET_PTM_TOKENS,
)
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

    charge_col = "precursor_charge" if "precursor_charge" in ds.column_names else "charge"
    if "precursor_mass" not in ds.column_names and "precursor_mz" in ds.column_names:
        ds = ds.map(
            lambda row: {
                "precursor_mass": precursor_mass_from_mz(
                    row["precursor_mz"], int(row[charge_col])
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
        "modified_sequence",
    ]
    existing = [column for column in keep_columns if column in ds.column_names]
    ds = ds.select_columns(existing)
    return ds

TOKEN_REGEX = re.compile(
    r"(\[[^\]]+\]|\([+-]?\d+(?:\.\d+)?\)|\([^\)]+\)|[A-Z](?:\[[^\]]+\]|\([^\)]+\))?)"
)


def normalize_peptide_sequence(sequence: str) -> str:
    """
    Normalize raw peptide strings from various datasets (MaxQuant, Nine-Species, etc.):
    - Strips leading/trailing flanking dots (.PEPTIDE.)
    - Converts residue-attached bracket deltas (e.g. M[15.9949] -> M(ox), C[57.02] -> C(cam))
    - Converts N-terminal brackets (e.g. [42.0106] -> (+42.01), [-17.0265] -> (-17.03))
    """
    if not sequence:
        return ""
    sequence = sequence.strip(".")
    # 1. Acetylation (+42.01)
    sequence = re.sub(r"([A-Z])\[42\.01\d*\]", r"(+42.01)\1", sequence)
    # 2. Carbamylation (+43.01)
    sequence = re.sub(r"([A-Z])\[43\.00\d*\]", r"(+43.01)\1", sequence)
    # 3. Ammonia loss (-17.03)
    sequence = re.sub(r"([A-Z])\[-17\.02\d*\]", r"(-17.03)\1", sequence)
    # 4. Standalone brackets
    sequence = re.sub(r"\[42\.01\d*\]", r"(+42.01)", sequence)
    sequence = re.sub(r"\[43\.00\d*\]", r"(+43.01)", sequence)
    sequence = re.sub(r"\[-17\.02\d*\]", r"(-17.03)", sequence)
    # 5. Common mass deltas
    sequence = re.sub(r"M\[15\.99\d*\]", "M(ox)", sequence)
    sequence = re.sub(r"C\[57\.02\d*\]", "C(cam)", sequence)
    sequence = re.sub(r"C\(\+57\.02\d*\)", "C(cam)", sequence)
    sequence = re.sub(r"N\[0\.98\d*\]", "N(deam)", sequence)
    sequence = re.sub(r"Q\[0\.98\d*\]", "Q(deam)", sequence)
    sequence = re.sub(r"([STY])\[79\.96\d*\]", r"\1(ph)", sequence)
    return sequence


def parse_peptide(sequence: str) -> list[str]:
    """
    Tokenize a peptide string into amino-acid and PTM residue tokens.
    Supports N-terminal modifications, UNIMOD identifiers, bracket and parenthesized deltas.
    Canonicalizes PTM delta aliases (e.g. M[UNIMOD:35] -> M(ox), [UNIMOD:1] -> (+42.01)).
    """
    if not sequence:
        return []
    sequence = normalize_peptide_sequence(sequence)
    tokens = [t for t in TOKEN_REGEX.findall(sequence) if t]
    if "".join(tokens) != sequence:
        # Fallback if unparenthesized unexpected formatting occurs
        tokens = list(sequence)
    return [PTM_ALIAS_MAP.get(tok, tok) for tok in tokens]


def canonicalize_peptide(sequence: str | list[str]) -> str:
    """
    Convert any peptide sequence (with UNIMOD tags, deltas, or aliases) 
    into a canonical string representation for exact match comparisons.
    """
    if isinstance(sequence, str):
        tokens = parse_peptide(sequence)
    else:
        tokens = [PTM_ALIAS_MAP.get(t, t) for t in sequence]
    return "".join(tokens)


def to_unimod_sequence(sequence: str | list[str]) -> str:
    """
    Convert any peptide sequence or token list into standard UNIMOD format
    (e.g., M[UNIMOD:35], C[UNIMOD:4], [UNIMOD:1], etc.).
    """
    from .constants import CANONICAL_TO_UNIMOD_MAP
    if isinstance(sequence, str):
        tokens = parse_peptide(sequence)
    else:
        tokens = [PTM_ALIAS_MAP.get(t, t) for t in sequence]
    return "".join(CANONICAL_TO_UNIMOD_MAP.get(tok, tok) for tok in tokens)


def build_vocabulary(ds=None, include_ptms: bool = True) -> dict[str, int]:
    """
    Build token vocabulary with stable ordering:
    1. Standard 20 amino acids (indices 0..19)
    2. Primary PTM tokens (indices 20..26)
    3. Any additional modified residues observed in data
    4. <pad> and <mask_token> (strictly the last two entries)
    """
    # 1. Standard 20 amino acids in canonical order
    vocab = {token: index for index, token in enumerate(STANDARD_AMINO_ACIDS)}

    # 2. Add target PTM tokens
    if include_ptms:
        for ptm in TARGET_PTM_TOKENS:
            if ptm not in vocab:
                vocab[ptm] = len(vocab)

    # 3. Discover any additional tokens in dataset
    if ds is not None:
        cols = getattr(ds, "column_names", None) or (list(ds.keys()) if hasattr(ds, "keys") else [])
        col = "modified_sequence" if "modified_sequence" in cols else "sequence"
        for seq in ds[col]:
            if seq:
                for tok in parse_peptide(seq):
                    if tok not in vocab:
                        vocab[tok] = len(vocab)

    # 4. Mandatory: <pad> and <mask_token> must be the last two entries
    vocab["<pad>"] = len(vocab)
    mask_idx = len(vocab)
    vocab["<mask_token>"] = mask_idx
    vocab["<mask" + ">"] = mask_idx
    return vocab

def invert_vocabulary(vocab: dict[str, int]) -> dict[int, str]:
    return {index: token for token, index in vocab.items()}

def decoder_output_token_ids(vocab: dict[str, int]) -> list[int]:
    pad_id = vocab["<pad>"]
    mask_id = vocab.get("<mask_token>", vocab.get("<mask" + ">"))
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
    def __init__(
        self,
        data,
        vocab,
        top_k: int = 200,
        remove_precursor_peak: bool = True,
        peak_dropout_prob: float = 0.0,
        is_train: bool = True,
    ):
        self.data = data
        self.top_k = top_k
        self.remove_precursor_peak = remove_precursor_peak
        self.vocab = vocab
        self.peak_dropout_prob = peak_dropout_prob
        self.is_train = is_train

    def __getitem__(self, idx):
        if hasattr(self.data, "column_names") and not hasattr(self.data, "info"):
            # PyArrow Table
            cols = ["mz_array", "intensity_array", "precursor_mass", "precursor_mz", "precursor_charge", "charge", "sequence", "modified_sequence"]
            row = {col: self.data[col][idx].as_py() for col in cols if col in self.data.column_names}
        elif hasattr(self.data, "schema") and not hasattr(self.data, "info"):
            # PyArrow RecordBatch
            cols = ["mz_array", "intensity_array", "precursor_mass", "precursor_mz", "precursor_charge", "charge", "sequence", "modified_sequence"]
            row = {col: self.data[col][idx].as_py() for col in cols if col in self.data.schema.names}
        else:
            row = self.data[idx]
        mz_array = torch.tensor(row["mz_array"], dtype=torch.float32)
        intensity_array = torch.tensor(row["intensity_array"], dtype=torch.float32)
        precursor_charge = int(row.get("precursor_charge") or row.get("charge") or 1)
        if "precursor_mass" in row and row["precursor_mass"] is not None:
            precursor_mass = float(row["precursor_mass"])
            # In Nine-Species IPC files, precursor_mass was saved as precursor_mz
            if "precursor_mz" in row and row["precursor_mz"] is not None:
                pmz = float(row["precursor_mz"])
                if abs(precursor_mass - pmz) < 1.0 and precursor_charge > 1:
                    precursor_mass = pmz * precursor_charge - precursor_charge * M_H
        elif "precursor_mz" in row and row["precursor_mz"] is not None:
            precursor_mass = float(row["precursor_mz"]) * precursor_charge - precursor_charge * M_H
        else:
            raise KeyError("Row missing both precursor_mass and precursor_mz")

        if self.remove_precursor_peak:
            precursor_mz = float(row.get("precursor_mz") or ((precursor_mass + precursor_charge * M_H) / max(precursor_charge, 1)))
            keep = (mz_array - precursor_mz).abs() > 1.5
            mz_array = mz_array[keep]
            intensity_array = intensity_array[keep]

        if self.top_k is not None:
            k = min(self.top_k, intensity_array.shape[0])
            if k > 0:
                intensity_array, indices = torch.topk(intensity_array, k)
                mz_array = mz_array[indices]

        # Spectral peak dropout: randomly zero-out peaks during training
        if self.is_train and self.peak_dropout_prob > 0.0 and intensity_array.numel() > 1:
            dropout_mask = torch.rand_like(intensity_array) > self.peak_dropout_prob
            if not dropout_mask.any():
                dropout_mask[torch.argmax(intensity_array)] = True
            intensity_array = intensity_array * dropout_mask.float()

        raw_sequence = row.get("modified_sequence") or row["sequence"]
        tokens = parse_peptide(raw_sequence)
        if any(token not in self.vocab for token in tokens):
            plain_seq = row.get("sequence", "")
            # If plain_seq has flanking residues matching dot notation (.SEQ.)
            if row.get("modified_sequence", "").startswith(".") and len(plain_seq) >= 2:
                plain_seq = plain_seq[1:-1]
            tokens = [t for t in list(plain_seq) if t in self.vocab]
        sequence = torch.tensor(
            [self.vocab[token] for token in tokens],
            dtype=torch.long,
        )
        length = len(tokens)
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

    spectrum_mask = pad_sequence(
        [torch.zeros(len(x), dtype=torch.bool) for x in intensity_array],
        batch_first=True,
        padding_value=True,
    )
    intensity_array = pad_sequence(
        intensity_array, batch_first=True, padding_value=0.0, padding_side="right"
    )
    max_intensity = intensity_array.amax(dim=-1, keepdim=True).clamp(min=1e-7)
    intensity_array = (intensity_array / max_intensity).sqrt()

    mz_array = pad_sequence(
        mz_array, batch_first=True, padding_value=.0, padding_side="right"
    )
    mz_complementary = pad_sequence(
        mz_complementary, batch_first=True, padding_value=.0, padding_side="right"
    )

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

def build_dataloader(
    ds,
    vocab,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    top_k: int = 200,
    peak_dropout_prob: float = 0.0,
    is_train: bool = True,
):
    dataset = SpectrumDataSet(
        ds,
        vocab,
        top_k=top_k,
        peak_dropout_prob=peak_dropout_prob,
        is_train=is_train,
    )
    collate_fn = partial(spectrum_collate, vocab=vocab)
    # persistent_workers avoids re-forking worker processes each epoch.
    # prefetch_factor keeps the GPU fed with 4 batches in flight at all times.
    # pin_memory allows async DMA transfers to the GPU (zero-copy host memory).
    use_persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
    )

