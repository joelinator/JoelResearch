"""Checkpoint/logging helper tests (no torch required)."""

from train.io import copy_history, empty_history


def test_copy_history_preserves_existing_metrics():
    initial = empty_history()
    initial["train_loss"] = [1.0, 0.9]
    copied = copy_history(initial)
    copied["train_loss"].append(0.8)
    assert initial["train_loss"] == [1.0, 0.9]
    assert copied["train_loss"] == [1.0, 0.9, 0.8]


def test_load_checkpoint_missing_file_raises_clear_error():
    import pytest
    from train.io import load_checkpoint

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint("non_existent_model_checkpoint_path_12345.ckpt")


if __name__ == "__main__":
    test_copy_history_preserves_existing_metrics()
    test_load_checkpoint_missing_file_raises_clear_error()
    print("All checkpoint helper checks passed.")

