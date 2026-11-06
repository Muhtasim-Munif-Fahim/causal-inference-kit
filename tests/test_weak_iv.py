import numpy as np
import pytest

from causal_inference.generators import simulate_iv_data
from causal_inference.estimators import two_stage_least_squares
from causal_inference.weak_iv import (
    anderson_rubin_ci,
    anderson_rubin_test,
    stock_yogo_critical_value,
    weak_instrument_flag,
)


def test_ar_rejects_wrong_null_with_strong_instrument():
    instrument, treatment, outcome, _, _ = simulate_iv_data(n=4000, seed=0)
    res = anderson_rubin_test(instrument, treatment, outcome, beta0=-5.0)
    assert res.reject
    assert res.p_value < 0.05


def test_ar_accepts_near_tsls_point():
    instrument, treatment, outcome, _, _ = simulate_iv_data(n=3000, seed=1)
    tsls = two_stage_least_squares(instrument, treatment, outcome)
    res = anderson_rubin_test(instrument, treatment, outcome, beta0=tsls.estimate)
    assert not res.reject
    assert res.p_value > 0.05


def test_ar_ci_covers_tsls():
    instrument, treatment, outcome, _, true_late = simulate_iv_data(n=2500, seed=2)
    tsls = two_stage_least_squares(instrument, treatment, outcome)
    radius = max(5.0, 5 * abs(tsls.estimate) + 1)
    ci = anderson_rubin_ci(
        instrument, treatment, outcome, grid_radius=radius, grid_size=301
    )
    assert ci.lower is not None and ci.upper is not None
    assert ci.lower <= tsls.estimate <= ci.upper
    assert ci.lower <= true_late <= ci.upper


def test_stock_yogo_and_flag():
    assert stock_yogo_critical_value(1) == pytest.approx(16.38)
    assert weak_instrument_flag(5.0, 1) is True
    assert weak_instrument_flag(20.0, 1) is False


def test_rejects_bad_alpha():
    instrument, treatment, outcome, _, _ = simulate_iv_data(n=100, seed=3)
    with pytest.raises(ValueError):
        anderson_rubin_test(instrument, treatment, outcome, alpha=1.5)
