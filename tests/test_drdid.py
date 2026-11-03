import numpy as np
import pytest

from causal_inference import (
    DRDiDResult,
    dr_did,
    ipw_did,
    or_did,
    simulate_conditional_did_data,
)


def _fit(**kw):
    d = simulate_conditional_did_data(**kw)
    return d, dr_did(d["y_pre"], d["y_post"], d["treatment"], d["X"])


def test_result_fields_and_summary():
    _, r = _fit(n=800, seed=1)
    assert isinstance(r, DRDiDResult)
    assert r.ci_low < r.att < r.ci_high
    assert r.n_treated + r.n_control == 800
    assert r.influence.shape == (800,)
    assert np.isclose(r.se, r.influence.std(ddof=1) / np.sqrt(800))
    assert "DR-DiD ATT" in r.summary()


def test_naive_did_is_biased_but_dr_recovers_att():
    _, r = _fit(n=5000, seed=2)
    assert abs(r.att - 1.0) < 0.1
    assert abs(r.naive_att - 1.0) > 0.5


def test_or_and_ipw_helpers_match_result_fields():
    d, r = _fit(n=600, seed=3)
    args = (d["y_pre"], d["y_post"], d["treatment"], d["X"])
    assert np.isclose(or_did(*args), r.or_att)
    assert np.isclose(ipw_did(*args), r.ipw_att)


def test_double_robustness_outcome_model_misspecified():
    ests = [
        _fit(n=2000, nonlinear_trend=True, seed=s)[1] for s in range(30)
    ]
    dr = np.mean([r.att for r in ests])
    orr = np.mean([r.or_att for r in ests])
    assert abs(dr - 1.0) < 0.1
    assert abs(orr - 1.0) > 0.3


def test_double_robustness_propensity_misspecified():
    ests = [
        _fit(n=2000, nonlinear_propensity=True, seed=s)[1] for s in range(30)
    ]
    dr = np.mean([r.att for r in ests])
    ipw = np.mean([r.ipw_att for r in ests])
    assert abs(dr - 1.0) < 0.05
    assert abs(ipw - 1.0) > 0.15


def test_coverage_near_nominal():
    hits = 0
    for s in range(150):
        _, r = _fit(n=600, seed=100 + s)
        hits += r.ci_low <= 1.0 <= r.ci_high
    assert 0.88 <= hits / 150 <= 0.99


def test_no_covariates_reduces_to_plain_did():
    d = simulate_conditional_did_data(n=400, seed=4)
    r = dr_did(d["y_pre"], d["y_post"], d["treatment"])
    assert np.isclose(r.att, r.naive_att)
    assert np.isclose(r.att, r.or_att)
    assert np.isclose(r.att, r.ipw_att)


def test_zero_effect_null_p_value_not_tiny():
    _, r = _fit(n=1000, att=0.0, seed=5)
    assert r.p_value > 0.01


def test_trimming_drops_extreme_controls():
    d = simulate_conditional_did_data(n=1000, seed=6)
    r = dr_did(d["y_pre"], d["y_post"], d["treatment"], d["X"], trim=0.3)
    assert r.n_trimmed > 0
    assert np.all(np.isfinite([r.att, r.se]))


def test_validation():
    with pytest.raises(ValueError):
        dr_did([1, 2], [1, 2, 3], [0, 1])
    with pytest.raises(ValueError):
        dr_did([1, 2, 3], [1, 2, 3], [0, 2, 1])
    with pytest.raises(ValueError):
        dr_did([1, 2, 3], [1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError):
        dr_did([1, 2, 3], [1, 2, 3], [0, 1, 0], alpha=1.5)
    with pytest.raises(ValueError):
        dr_did([1, 2, 3, 4], [1, 2, 3, 4], [0, 1, 0, 1], X=np.ones((3, 1)))
    with pytest.raises(ValueError):
        simulate_conditional_did_data(n=2)
