"""Scheduler ↔ noising consistency checks."""

import torch

from flow_matching.scheduler import cosine_scheduler, linear_scheduler, verify_scheduler
from flow_matching.sampling import sample_noising_step_mask, sample_noising_step_uniform


def test_forward_boundaries_mask():
    vocab = {"A": 0, "<pad>": 1, "<mask>": 2}
    x1 = torch.zeros(10_000, 20, dtype=torch.long)  # all token 0

    for scheduler, name in ((linear_scheduler, "linear"), (cosine_scheduler, "cosine")):
        kt0, _ = scheduler(torch.tensor([0.0]))
        kt1, _ = scheduler(torch.tensor([1.0]))

        xt0 = sample_noising_step_mask(kt0, x1, vocab)
        xt1 = sample_noising_step_mask(kt1, x1, vocab)

        mask_rate_t0 = (xt0 == vocab["<mask>"]).float().mean().item()
        mask_rate_t1 = (xt1 == vocab["<mask>"]).float().mean().item()

        assert mask_rate_t0 > 0.95, f"{name}: expected ~all masked at t=0, got {mask_rate_t0}"
        assert mask_rate_t1 < 0.05, f"{name}: expected ~none masked at t=1, got {mask_rate_t1}"


def test_forward_boundaries_uniform():
    vocab = {chr(65 + i): i for i in range(20)}
    vocab["<pad>"] = len(vocab)
    vocab["<mask_token>"] = len(vocab)
    x1 = torch.zeros(10_000, 20, dtype=torch.long)

    for scheduler, name in ((linear_scheduler, "linear"), (cosine_scheduler, "cosine")):
        kt0, _ = scheduler(torch.tensor([0.0]))
        kt1, _ = scheduler(torch.tensor([1.0]))

        xt0 = sample_noising_step_uniform(kt0, x1, vocab)
        xt1 = sample_noising_step_uniform(kt1, x1, vocab)

        clean_rate_t0 = (xt0 == 0).float().mean().item()
        clean_rate_t1 = (xt1 == 0).float().mean().item()

        assert clean_rate_t0 < 0.08, f"{name}: expected ~no clean at t=0, got {clean_rate_t0}"
        assert clean_rate_t1 > 0.95, f"{name}: expected ~all clean at t=1, got {clean_rate_t1}"


def test_scheduler_verification():
    verify_scheduler(linear_scheduler, name="linear")
    verify_scheduler(cosine_scheduler, name="cosine")


if __name__ == "__main__":
    test_scheduler_verification()
    test_forward_boundaries_mask()
    test_forward_boundaries_uniform()
    print("All scheduler/noising consistency checks passed.")
