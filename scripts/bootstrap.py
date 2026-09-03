"""Shared bootstrap helpers for CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def setup_src_path() -> Path:
    """Ensure `src/` is on sys.path so modules can be imported."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    src_str = str(src)
    scripts_str = str(root / "scripts")
    while scripts_str in sys.path:
        sys.path.remove(scripts_str)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)
    if "eval" in sys.modules and not hasattr(sys.modules["eval"], "__path__"):
        del sys.modules["eval"]
    return root


def move_batch_to_device(batch, device: torch.device):
    return tuple(
        tensor.to(device) if torch.is_tensor(tensor) else tensor
        for tensor in batch
    )
