"""Tests for E-values and Rosenbaum bounds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from causal_inference import (
    EValueResult,
    RosenbaumBoundsResult,
    bias_factor,
    e_value,
    e_value_from_ate,
    rosenbaum_bounds,
    rosenbaum_sensitivity_value,
)


def test_vanderweele_ding_worked_example():
    # Annals of Internal Medicine 2017: RR 3.9 (95% CI 1.8 to 8.7)
    res = e_value(3.9, 1.8, 8.7)
    assert isinstance(res, EValueResult)
    assert res.e_value == pytest.approx(3.9 + math.sqrt(3.9 * 2.9))
    assert res.e_value == pytest.approx(7.26, abs=0.01)
    assert res.e_value_ci == pytest.approx(3.0)


def test_protective_effect_uses_inverse_and_upper_limit():
    res = e_value(0.5, 0.3, 0.8)
    assert res.e_value == pytest.approx(2.0 + math.sqrt(2.0))
    inv = 1.0 / 0.8
    assert res.e_value_ci == pytest.approx(inv + math.sqrt(inv * (inv - 1.0)))


def test_null_and_ci_crossing_null():
    assert e_value(1.0).e_value == 1.0
    res = e_value(1.4, 0.9, 2.1)
    assert res.e_value_ci == 1.0
    assert res.e_value > 1.0
    assert e_value(2.0).e_value_ci is None


def test_odds_ratio_conversion():
    common = e_value(4.0, measure="OR")
    assert common.rr == pytest.approx(2.0)
    rare = e_value(4.0, measure="or", rare=True)
    assert rare.rr == pytest.approx(4.0)
    assert rare.e_value > common.e_value


def test_hazard_ratio_conversion():
    res = e_value(2.0, measure="HR")
    root = math.sqrt(2.0)
    expected_rr = (1 - 0.5**root) / (1 - 0.5 ** (1 / root))
    assert res.rr == pytest.approx(expected_rr)
    assert e_value(2.0, measure="HR", rare=True).rr == pytest.approx(2.0)
    # conversion is symmetric: HR and 1/HR give reciprocal RRs
    assert e_value(0.5, measure="HR").rr == pytest.approx(1.0 / expected_rr)


def test_smd_with_se_and_from_ate():
    res = e_value(0.5, measure="SMD", se=0.1)
    assert res.rr == pytest.approx(math.exp(0.455))
    assert res.rr_lower == pytest.approx(math.exp(0.455 - 0.178))
    assert res.rr_upper == pytest.approx(math.exp(0.455 + 0.178))
    lo = res.rr_lower
    assert res.e_value_ci == pytest.approx(lo + math.sqrt(lo * (lo - 1.0)))
    ate = e_value_from_ate(2.0, outcome_sd=4.0, se=0.4)
    assert ate.estimate == pytest.approx(0.5)
    assert ate.rr == pytest.approx(res.rr)
    assert ate.e_value_ci == pytest.approx(res.e_value_ci)


def test_bias_factor_inverts_e_value():
    for rr in (1.3, 2.0, 5.5):
        e = e_value(rr).e_value
        assert bias_factor(e, e) == pytest.approx(rr)
    assert bias_factor(1.0, 7.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bias_factor(0.5, 2.0)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((-1.0,), {}),
        ((0.0,), {"measure": "OR"}),
        ((2.0,), {"measure": "beta"}),
        ((2.0, 1.5), {}),
        ((2.0, 2.5, 3.0), {}),
        ((2.0,), {"se": 0.1}),
        ((0.3,), {"measure": "SMD", "se": -1.0}),
    ],
)
def test_e_value_errors(args, kwargs):
    with pytest.raises(ValueError):
        e_value(*args, **kwargs)


def _pairs(effect, n=60, seed=0):
    rng = np.random.default_rng(seed)
    return effect + rng.normal(0, 1, n)


def test_gamma_one_matches_wilcoxon_signed_rank():
    stats = pytest.importorskip("scipy.stats")
    d = _pairs(0.4, n=50, seed=1)
    res = rosenbaum_bounds(d, gammas=[1.0])
    ref = stats.wilcoxon(d, alternative="greater", method="approx", correction=True)
    assert res.p_upper[0] == pytest.approx(ref.pvalue, rel=1e-6)
    assert res.p_lower[0] == pytest.approx(res.p_upper[0])
    assert res.statistic == pytest.approx(ref.statistic)


def test_bounds_are_monotone_in_gamma():
    d = _pairs(0.6, n=80, seed=2)
    res = rosenbaum_bounds(d, gammas=[1.0, 1.5, 2.0, 3.0, 5.0])
    assert isinstance(res, RosenbaumBoundsResult)
    assert all(a <= b for a, b in zip(res.p_upper, res.p_upper[1:]))
    assert all(a >= b for a, b in zip(res.p_lower, res.p_lower[1:]))
    assert all(lo <= hi for lo, hi in zip(res.p_lower, res.p_upper))
    assert [row["gamma"] for row in res.rows] == res.gammas


def test_zero_differences_dropped_and_ties():
    d = [0.0, 0.0, 1.0, 1.0, 2.0, -1.0, 3.0, 3.0, 3.0]
    res = rosenbaum_bounds(d, gammas=[1.0])
    assert res.n_zero == 2 and res.n_pairs == 7
    # |d| = 1,1,2,1,3,3,3 -> ranks 2,2,4,2,6,6,6; positives all but the -1
    assert res.statistic == pytest.approx(2 + 2 + 4 + 6 + 6 + 6)


def test_alternative_less_mirrors_greater():
    d = _pairs(-0.5, n=40, seed=3)
    less = rosenbaum_bounds(d, gammas=[1.0, 2.0], alternative="less")
    greater = rosenbaum_bounds(-d, gammas=[1.0, 2.0])
    np.testing.assert_allclose(less.p_upper, greater.p_upper)


def test_sensitivity_value():
    d = _pairs(0.8, n=100, seed=4)
    gamma = rosenbaum_sensitivity_value(d, alpha=0.05)
    assert gamma > 1.5
    at = rosenbaum_bounds(d, gammas=[gamma])
    assert at.p_upper[0] == pytest.approx(0.05, abs=1e-4)
    # a stronger effect is robust to more hidden bias
    assert rosenbaum_sensitivity_value(_pairs(1.5, n=100, seed=4)) > gamma
    # not significant at all -> 1.0
    assert rosenbaum_sensitivity_value(_pairs(-0.5, n=50, seed=5)) == 1.0
    assert rosenbaum_sensitivity_value(np.full(30, 5.0), gamma_max=2.0) == math.inf


def test_rosenbaum_errors():
    with pytest.raises(ValueError):
        rosenbaum_bounds([])
    with pytest.raises(ValueError):
        rosenbaum_bounds([0.0, 0.0])
    with pytest.raises(ValueError):
        rosenbaum_bounds([1.0, 2.0], gammas=[0.5])
    with pytest.raises(ValueError):
        rosenbaum_bounds([1.0, 2.0], alternative="two-sided")
    with pytest.raises(ValueError):
        rosenbaum_sensitivity_value([1.0, 2.0], alpha=1.5)
    with pytest.raises(ValueError):
        rosenbaum_bounds([1.0, float("nan")])
