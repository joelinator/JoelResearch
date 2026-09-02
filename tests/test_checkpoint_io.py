"""Checkpoint/logging helper tests (no torch required)."""

from train.io import copy_history, empty_history


def test_copy_history_preserves_existing_metrics():
    initial = empty_history()
    initial["train_loss"] = [1.0, 0.9]
    copied = copy_history(initial)
    copied["train_loss"].append(0.8)
    assert initial["train_loss"] == [1.0, 0.9]
    assert copied["train_loss"] == [1.0, 0.9, 0.8]


if __name__ == "__main__":
    test_copy_history_preserves_existing_metrics()
    print("All checkpoint helper checks passed.")
