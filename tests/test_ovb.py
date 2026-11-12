import math

import numpy as np
import pytest

from causal_inference import (
    ovb_adjusted_estimate,
    ovb_bounds,
    ovb_sensitivity,
    partial_r2,
    robustness_value,
)
from causal_inference.ovb import _t_ppf, _t_sf


def _resid(v, M):
    beta = np.linalg.lstsq(M, v, rcond=None)[0]
    return v - M @ beta


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Z = rng.normal(size=n)
    D = 0.6 * X[:, 0] + 0.5 * Z + rng.normal(size=n)
    Y = 1.0 * D + 0.8 * X[:, 0] - 0.3 * X[:, 1] + 0.7 * Z + rng.normal(size=n)
    return X, Z, D, Y


def test_t_distribution_helpers_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 3, 10, 57, 500):
        for t in (-3.1, -0.4, 0.0, 1.2, 4.5):
            assert _t_sf(t, df) == pytest.approx(stats.t.sf(t, df), rel=1e-9, abs=1e-14)
        for p in (0.005, 0.025, 0.3, 0.975):
            assert _t_ppf(p, df) == pytest.approx(stats.t.ppf(p, df), rel=1e-8)


def test_partial_r2_from_t_equals_fwl_partial_r2():
    X, _, D, Y = _data()
    res = ovb_sensitivity(X, D, Y)
    M = np.column_stack([np.ones(D.size), X])
    ry, rd = _resid(Y, M), _resid(D, M)
    r2 = (ry @ rd) ** 2 / ((ry @ ry) * (rd @ rd))
    assert res.partial_r2_yd == pytest.approx(r2, rel=1e-10)
    assert res.estimate == pytest.approx((ry @ rd) / (rd @ rd))


def test_ovb_formula_recovers_estimate_with_actual_confounder():
    X, Z, D, Y = _data(seed=3)
    short = ovb_sensitivity(X, D, Y)
    full = ovb_sensitivity(np.column_stack([X, Z]), D, Y)
    M = np.column_stack([np.ones(D.size), X])
    rz = _resid(Z, M)
    rd = _resid(D, M)
    r2dz = (rz @ rd) ** 2 / ((rz @ rz) * (rd @ rd))
    MD = np.column_stack([M, D])
    rzy = _resid(Z, MD)
    ry = _resid(Y, MD)
    r2yz = (rzy @ ry) ** 2 / ((rzy @ rzy) * (ry @ ry))
    adj, adj_se, lo, hi = ovb_adjusted_estimate(short.estimate, short.se, short.dof, r2dz, r2yz)
    assert adj == pytest.approx(full.estimate, rel=1e-9)
    assert adj_se == pytest.approx(full.se, rel=1e-9)
    assert lo < adj < hi


def test_robustness_value_confounder_exactly_nulls_estimate():
    X, _, D, Y = _data(seed=4)
    res = ovb_sensitivity(X, D, Y, q=1.0)
    rv = res.rv_q
    adj, *_ = ovb_adjusted_estimate(res.estimate, res.se, res.dof, rv, rv)
    assert adj == pytest.approx(0.0, abs=1e-9)
    half = robustness_value(res.t_stat, res.dof, q=0.5)
    adj_half, *_ = ovb_adjusted_estimate(res.estimate, res.se, res.dof, half, half)
    assert adj_half == pytest.approx(0.5 * res.estimate, rel=1e-9)
    assert 0 < res.rv_qa < res.rv_q
    # RV_{q,alpha}: adjusted CI just touches zero
    rva = res.rv_qa
    _, _, lo, hi = ovb_adjusted_estimate(res.estimate, res.se, res.dof, rva, rva)
    assert min(abs(lo), abs(hi)) < 0.02 * abs(res.estimate)


def test_rv_alpha_zero_when_insignificant_and_partial_r2():
    assert robustness_value(1.0, 100, alpha=0.05) == 0.0
    assert partial_r2(2.0, 4.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        robustness_value(2.0, 1)


def test_benchmark_bounds_shape_and_monotonic():
    X, _, D, Y = _data(seed=5)
    res = ovb_sensitivity(X, D, Y, benchmark=0, benchmark_name="x0", kd=(1, 2, 3))
    assert [b.kd for b in res.bounds] == [1.0, 2.0, 3.0]
    adj = [b.adjusted_estimate for b in res.bounds]
    assert adj[0] < res.estimate and adj[0] > adj[1] > adj[2]
    r2dz = [b.r2dz_x for b in res.bounds]
    assert r2dz == sorted(r2dz)
    assert "x0" in res.summary()
    # kd=1 bound with r2dxj: r2dz = r2dxj/(1-r2dxj)
    b = ovb_bounds(0.1, 0.2, 1.0, 0.1, 100, kd=[1.0])[0]
    assert b.r2dz_x == pytest.approx(0.1 / 0.9)
    with pytest.raises(ValueError):
        ovb_bounds(0.6, 0.2, 1.0, 0.1, 100, kd=[2.0])


def test_validation():
    X, _, D, Y = _data(n=50)
    with pytest.raises(ValueError):
        ovb_sensitivity(X, np.ones_like(D), Y)
    with pytest.raises(ValueError):
        ovb_sensitivity(X[:-1], D, Y)
    with pytest.raises(ValueError):
        ovb_sensitivity(X, D, Y, benchmark=5)
    with pytest.raises(ValueError):
        ovb_adjusted_estimate(1.0, 0.1, 100, 1.0, 0.1)
    assert math.isfinite(ovb_sensitivity(None, D, Y).estimate)
