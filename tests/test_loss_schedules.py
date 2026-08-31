"""Loss weight schedule tests."""

from train.loss import gamma_schedule, lambda_schedule


def test_lambda_warmup_starts_at_zero():
    assert lambda_schedule(0, 100) == 0.0
    assert lambda_schedule(0, 5) == 0.0


def test_lambda_reaches_final():
    assert lambda_schedule(100, 100) == 0.15
    assert lambda_schedule(50, 100) == 0.15


def test_gamma_starts_late():
    assert gamma_schedule(0, 100) == 0.0
    assert gamma_schedule(15, 100) == 0.0


def test_gamma_ramps_then_holds():
    mid = gamma_schedule(40, 100)
    final = gamma_schedule(100, 100)
    assert 0.0 < mid < final
    assert final == 0.08


def test_short_run():
    # 5-epoch smoke test: λ on by epoch 2, γ ramps in later epochs.
    assert lambda_schedule(1, 5) == 0.15
    assert gamma_schedule(0, 5) == 0.0
    assert gamma_schedule(4, 5) == 0.08


if __name__ == "__main__":
    test_lambda_warmup_starts_at_zero()
    test_lambda_reaches_final()
    test_gamma_starts_late()
    test_gamma_ramps_then_holds()
    test_short_run()
    print("All loss schedule checks passed.")
