"""
MassIVE-KB Dataset Loader and Shard Manager for DFlowNovo.

NOTE: /dev/shm is an absolute lifesaver here. Streaming 28M spectra directly from
NVMe drive hammered the I/O bus and starved the H100. RAM disk keeps GPU utilization >95%.

TODO: maybe add multi-process prefetching if we scale to multi-node training later.
"""

from __future__ import annotations

# Life is too short to parse 28M spectra on a single thread...
import glob
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from huggingface_hub import hf_hub_download

from .constants import AA_MASSES_DICT, M_H, M_H2O
from .data import parse_peptide, spectrum_collate
from .lengths import validate_peptide_length

HF_REPO_ID = "chrisagrams/massive_kb_v1_shuffled"


def ensure_shards_available(
    split: str = "train",
    num_shards: int = 5,
    target_dir: str = "/dev/shm/massive_kb",
) -> List[str]:
    """
    Ensure the requested number of parquet shards exist in target_dir.
    Downloads missing shards from Hugging Face if needed.
    """
    split_dir = Path(target_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    available = sorted(glob.glob(str(split_dir / f"{split}_*.parquet")))
    if len(available) >= num_shards:
        return available[:num_shards]

    print(f"[MassIVE-KB] Downloading {num_shards} {split} shards to {split_dir}...")
    for i in range(num_shards):
        filename = f"{split}/{split}_{i:03d}.parquet"
        dest_path = split_dir / f"{split}_{i:03d}.parquet"
        if not dest_path.exists():
            print(f"[MassIVE-KB] Downloading {filename}...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=target_dir,
                cache_dir="/dev/shm/.cache/huggingface",
            )
    return sorted(glob.glob(str(split_dir / f"{split}_*.parquet")))[:num_shards]


class MassIVEKBSpectrumDataset(Dataset):
    """
    Ultra-efficient PyTorch Dataset for MassIVE-KB spectra.
    Maintains pre-tokenized peptide sequences and masses in contiguous memory,
    with zero-copy PyArrow peak array access.
    """

    def __init__(
        self,
        mz_col,
        int_col,
        precursor_masses: List[float],
        precursor_charges: List[int],
        token_tensors: List[torch.Tensor],
        vocab: Dict[str, int],
        top_k: int = 200,
        remove_precursor_peak: bool = True,
        peak_dropout_prob: float = 0.0,
        is_train: bool = True,
        valid_indices: Optional[List[int]] = None,
    ):
        self.mz_col = mz_col
        self.int_col = int_col
        self.precursor_masses = precursor_masses
        self.precursor_charges = precursor_charges
        self.token_tensors = token_tensors
        self.vocab = vocab
        self.top_k = top_k
        self.remove_precursor_peak = remove_precursor_peak
        self.peak_dropout_prob = peak_dropout_prob
        self.is_train = is_train
        self.valid_indices = valid_indices

    def __len__(self) -> int:
        return len(self.precursor_masses)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx] if self.valid_indices is not None else idx
        mz_raw = self.mz_col[real_idx].as_py()
        int_raw = self.int_col[real_idx].as_py()

        mz_array = torch.tensor(mz_raw, dtype=torch.float32)
        intensity_array = torch.tensor(int_raw, dtype=torch.float32)

        precursor_mass = self.precursor_masses[idx]
        precursor_charge = self.precursor_charges[idx]
        sequence = self.token_tensors[idx]
        length = sequence.numel()

        if self.remove_precursor_peak:
            # NOTE: strip out the unfragmented precursor peak so the model actually learns fragmentation ladders
            precursor_mz = (precursor_mass + precursor_charge * M_H) / max(precursor_charge, 1)
            keep = (mz_array - precursor_mz).abs() > 1.5
            if keep.any():
                mz_array = mz_array[keep]
                intensity_array = intensity_array[keep]

        if mz_array.numel() == 0:
            precursor_mz = (precursor_mass + precursor_charge * M_H) / max(precursor_charge, 1)
            mz_array = torch.tensor([precursor_mz], dtype=torch.float32)
            intensity_array = torch.tensor([1.0], dtype=torch.float32)

        if self.top_k is not None:
            k = min(self.top_k, intensity_array.shape[0])
            if k > 0:
                intensity_array, indices = torch.topk(intensity_array, k)
                mz_array = mz_array[indices]

        if self.is_train and self.peak_dropout_prob > 0.0 and intensity_array.numel() > 1:
            dropout_mask = torch.rand_like(intensity_array) > self.peak_dropout_prob
            if not dropout_mask.any():
                dropout_mask[torch.argmax(intensity_array)] = True
            intensity_array = intensity_array * dropout_mask.float()

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


class MassIVEKBStreamingDataset(IterableDataset):
    """
    High-performance streaming PyTorch IterableDataset for full-scale MassIVE-KB.
    Streams spectra shard-by-shard with constant O(1) RAM usage, multi-worker
    shard partitioning, per-epoch shard shuffling, and an intra-shard shuffle buffer.
    """

    def __init__(
        self,
        shard_files: List[str],
        vocab: Dict[str, int],
        is_train: bool = True,
        max_length: int = 50,
        top_k: int = 200,
        remove_precursor_peak: bool = True,
        peak_dropout_prob: float = 0.15,
        shuffle_buffer_size: int = 4096,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_files = sorted(shard_files)
        self.vocab = vocab
        self.is_train = is_train
        self.max_length = max_length
        self.top_k = top_k
        self.remove_precursor_peak = remove_precursor_peak
        self.peak_dropout_prob = peak_dropout_prob if is_train else 0.0
        self.shuffle_buffer_size = shuffle_buffer_size if is_train else 1
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _parse_shard(self, shard_path: str):
        try:
            table = pq.read_table(shard_path, columns=["m/z", "int", "peptide_charge"])
        except Exception as e:
            print(f"[MassIVE-KB Warning] Error reading shard {shard_path}: {e}")
            return

        pc_col = table["peptide_charge"].to_pylist()
        mz_col = table["m/z"]
        int_col = table["int"]

        vocab = self.vocab
        is_train = self.is_train
        top_k = self.top_k
        remove_precursor = self.remove_precursor_peak
        dropout_prob = self.peak_dropout_prob
        max_len = self.max_length

        for idx, pc in enumerate(pc_col):
            try:
                seq, ch_str = pc.rsplit("_", 1)
                tokens = parse_peptide(seq)
                L = len(tokens)
                if not (7 <= L <= max_len and all(t in vocab for t in tokens)):
                    continue

                charge = int(ch_str)
                precursor_mass = sum(AA_MASSES_DICT[t] for t in tokens) + M_H2O
                token_ids = [vocab[t] for t in tokens]
                token_tensor = torch.tensor(token_ids, dtype=torch.long)

                mz_raw = mz_col[idx].as_py()
                int_raw = int_col[idx].as_py()
                mz_array = torch.tensor(mz_raw, dtype=torch.float32)
                intensity_array = torch.tensor(int_raw, dtype=torch.float32)

                if remove_precursor:
                    precursor_mz = (precursor_mass + charge * M_H) / max(charge, 1)
                    keep = (mz_array - precursor_mz).abs() > 1.5
                    if keep.any():
                        mz_array = mz_array[keep]
                        intensity_array = intensity_array[keep]

                if mz_array.numel() == 0:
                    precursor_mz = (precursor_mass + charge * M_H) / max(charge, 1)
                    mz_array = torch.tensor([precursor_mz], dtype=torch.float32)
                    intensity_array = torch.tensor([1.0], dtype=torch.float32)

                if top_k is not None:
                    k = min(top_k, intensity_array.shape[0])
                    if k > 0:
                        intensity_array, indices = torch.topk(intensity_array, k)
                        mz_array = mz_array[indices]

                if is_train and dropout_prob > 0.0 and intensity_array.numel() > 1:
                    dropout_mask = torch.rand_like(intensity_array) > dropout_prob
                    if not dropout_mask.any():
                        dropout_mask[torch.argmax(intensity_array)] = True
                    intensity_array = intensity_array * dropout_mask.float()

                mz_complementary = precursor_mass + 2 * M_H - mz_array

                yield (
                    mz_array,
                    intensity_array,
                    precursor_mass,
                    charge,
                    token_tensor,
                    mz_complementary,
                    L,
                )
            except Exception:
                continue

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shards = list(self.shard_files)

        if self.is_train:
            # Deterministic per-epoch shuffle of shard ordering
            rng = random.Random(self.seed + self.epoch * 10007)
            rng.shuffle(shards)

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = shards[worker_id::num_workers]

        if not self.is_train or self.shuffle_buffer_size <= 1:
            for shard_path in shards:
                yield from self._parse_shard(shard_path)
            return

        # Reservoir / Shuffle Buffer for training
        buffer = []
        rng = random.Random(self.seed + self.epoch * 10007 + (worker_info.id if worker_info else 0))

        for shard_path in shards:
            for item in self._parse_shard(shard_path):
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    rand_idx = rng.randint(0, len(buffer) - 1)
                    yield buffer.pop(rand_idx)

        # Flush remaining buffer
        rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()


def load_and_preprocess_shards(
    parquet_files: List[str],
    vocab: Dict[str, int],
    is_train: bool = True,
    max_length: int = 50,
    top_k: int = 200,
    peak_dropout_prob: float = 0.15,
) -> MassIVEKBSpectrumDataset:
    """
    Reads multiple Parquet files into memory, parses and filters peptides, and returns a fast Dataset.
    """
    t0 = time.time()
    tables = [pq.read_table(f, columns=["m/z", "int", "peptide_charge"]) for f in parquet_files]
    concat_table = pa.concat_tables(tables)
    total_raw = len(concat_table)

    pc_col = concat_table["peptide_charge"].to_pylist()
    mz_col = concat_table["m/z"]
    int_col = concat_table["int"]

    valid_indices = []
    masses: List[float] = []
    charges: List[int] = []
    token_tensors: List[torch.Tensor] = []

    for idx, pc in enumerate(pc_col):
        seq, ch_str = pc.rsplit("_", 1)
        tokens = parse_peptide(seq)
        L = len(tokens)
        if 7 <= L <= max_length and all(t in vocab for t in tokens):
            valid_indices.append(idx)
            masses.append(sum(AA_MASSES_DICT[t] for t in tokens) + M_H2O)
            charges.append(int(ch_str))
            token_ids = [vocab[t] for t in tokens]
            token_tensors.append(torch.tensor(token_ids, dtype=torch.long))

    print(
        f"[MassIVE-KB] Preprocessed {len(valid_indices):,}/{total_raw:,} spectra "
        f"({len(valid_indices)/total_raw*100:.1f}%) in {time.time() - t0:.2f}s."
    )

    return MassIVEKBSpectrumDataset(
        mz_col=mz_col,
        int_col=int_col,
        precursor_masses=masses,
        precursor_charges=charges,
        token_tensors=token_tensors,
        vocab=vocab,
        top_k=top_k,
        remove_precursor_peak=True,
        peak_dropout_prob=peak_dropout_prob if is_train else 0.0,
        is_train=is_train,
        valid_indices=valid_indices if len(valid_indices) < total_raw else None,
    )


def build_massive_kb_dataloaders(
    vocab: Dict[str, int],
    train_shards: int = 256,
    valid_shards: int = 1,
    target_dir: str = "/dev/shm/massive_kb",
    batch_size: int = 1024,
    max_length: int = 50,
    num_workers: int = 8,
    top_k: int = 200,
    peak_dropout: float = 0.15,
    streaming: Optional[bool] = None,
    shuffle_buffer_size: int = 4096,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Builds optimized training and validation DataLoaders for MassIVE-KB.
    Returns (train_loader, valid_loader, estimated_batches_per_epoch).
    """
    train_files = ensure_shards_available("train", train_shards, target_dir)
    valid_files = ensure_shards_available("test", valid_shards, target_dir)

    # Use streaming by default when shard count > 5 to guarantee constant RAM usage
    if streaming is None:
        streaming = len(train_files) > 5

    pin = torch.cuda.is_available()

    if streaming:
        print(f"[MassIVE-KB] Initializing Streaming Dataset over {len(train_files)} shards (shuffle buffer={shuffle_buffer_size}, max_len={max_length})...")
        train_ds = MassIVEKBStreamingDataset(
            shard_files=train_files,
            vocab=vocab,
            is_train=True,
            max_length=max_length,
            top_k=top_k,
            remove_precursor_peak=True,
            peak_dropout_prob=peak_dropout,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,  # Shuffle handled within IterableDataset
            num_workers=num_workers,
            pin_memory=pin,
            collate_fn=lambda b: spectrum_collate(b, vocab=vocab),
            persistent_workers=False,  # Re-seed each epoch for fresh shard order
        )
        # Average valid spectra per MassIVE-KB shard is ~108,500
        estimated_batches_per_epoch = int(len(train_files) * (108500 / batch_size))
    else:
        print(f"[MassIVE-KB] Initializing In-Memory Dataset over {len(train_files)} shards (max_len={max_length})...")
        train_ds = load_and_preprocess_shards(
            train_files, vocab, is_train=True, max_length=max_length, top_k=top_k, peak_dropout_prob=peak_dropout
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            collate_fn=lambda b: spectrum_collate(b, vocab=vocab),
            persistent_workers=(num_workers > 0),
        )
        estimated_batches_per_epoch = len(train_loader)

    # Validation is always kept in-memory for instant, deterministic evaluation
    valid_ds = load_and_preprocess_shards(
        valid_files, vocab, is_train=False, max_length=max_length, top_k=top_k, peak_dropout_prob=0.0
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=pin,
        collate_fn=lambda b: spectrum_collate(b, vocab=vocab),
        persistent_workers=(num_workers > 0),
    )

    return train_loader, valid_loader, estimated_batches_per_epoch
