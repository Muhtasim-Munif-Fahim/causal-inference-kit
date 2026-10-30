import numpy as np
import pytest

from causal_inference.its import (
    comparative_interrupted_time_series,
    interrupted_time_series,
    its_placebo_test,
    newey_west_cov,
)


def _ar1(n, rho, sd, rng):
    e = np.empty(n)
    e[0] = rng.normal(0, sd / np.sqrt(1 - rho**2))
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, sd)
    return e


def _series(n=120, t0=70, level=5.0, slope=0.3, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    D = (t >= t0).astype(float)
    y = 20 + 0.1 * t + level * D + slope * (t - t0) * D + _ar1(n, rho, 1.0, rng)
    return y


def test_recovers_level_and_slope_change_and_effects():
    y = _series()
    res = interrupted_time_series(y, 70)
    assert res.level_change == pytest.approx(5.0, abs=1.5)
    assert res.slope_change == pytest.approx(0.3, abs=0.1)
    assert res.pre_slope == pytest.approx(0.1, abs=0.05)
    assert res.post_slope == pytest.approx(res.pre_slope + res.slope_change)
    assert res.n_pre == 70 and res.n_post == 50
    np.testing.assert_allclose(res.effects, res.level_change + res.slope_change * np.arange(50))
    assert res.average_effect == pytest.approx(res.effects.mean())
    assert res.cumulative_effect == pytest.approx(res.effects.sum())
    np.testing.assert_allclose(res.fitted[70:] - res.counterfactual[70:], res.effects)
    np.testing.assert_allclose(res.fitted[:70], res.counterfactual[:70])
    assert res.p_value("slope_change") < 0.001
    lo, hi = res.ci("level_change")
    assert lo < res.level_change < hi
    assert res.residual_autocorrelation > 0.2 and res.durbin_watson < 1.6
    assert "Newey-West" in res.summary()


def test_ols_coefficients_match_lstsq():
    y = _series(seed=3)
    t = np.arange(y.size, dtype=float)
    D = (t >= 70).astype(float)
    X = np.column_stack([np.ones_like(t), t - 70, D, (t - 70) * D])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = interrupted_time_series(y, 70, hac_lags=0)
    np.testing.assert_allclose(list(res.coef.values()), beta)


def test_newey_west_reduces_to_hc1_and_matches_manual_formula():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(50), rng.normal(size=50)])
    u = rng.normal(size=50)
    bread = np.linalg.inv(X.T @ X)
    hc0 = bread @ (X.T * u**2) @ X @ bread
    np.testing.assert_allclose(newey_west_cov(X, u, 0, small_sample=False), hc0)
    np.testing.assert_allclose(newey_west_cov(X, u, 0), hc0 * 50 / 48)
    # lag 1 Bartlett weight 1/2
    S = (X.T * u**2) @ X
    G = sum(u[t] * u[t - 1] * np.outer(X[t], X[t - 1]) for t in range(1, 50))
    S = S + 0.5 * (G + G.T)
    np.testing.assert_allclose(newey_west_cov(X, u, 1, small_sample=False), bread @ S @ bread)
    with pytest.raises(ValueError):
        newey_west_cov(X, u, -1)


def test_hac_coverage_under_autocorrelation_beats_naive():
    hits_hac = hits_iid = 0
    reps = 200
    for s in range(reps):
        y = _series(n=150, t0=80, rho=0.6, seed=100 + s)
        r_hac = interrupted_time_series(y, 80, hac_lags=8)
        r_iid = interrupted_time_series(y, 80, hac_lags=0)
        for r, name in ((r_hac, "hac"), (r_iid, "iid")):
            lo, hi = r.ci("level_change")
            if lo <= 5.0 <= hi:
                if name == "hac":
                    hits_hac += 1
                else:
                    hits_iid += 1
    assert hits_hac / reps > 0.83
    assert hits_hac > hits_iid


def test_seasonality_and_covariates_and_time_axis():
    rng = np.random.default_rng(2)
    n, t0 = 96, 60
    t = np.arange(n, dtype=float)
    season = 3 * np.sin(2 * np.pi * t / 12)
    x = rng.normal(size=n)
    y = 10 + 0.05 * t + season + 2 * x + 4 * (t >= t0) + rng.normal(0, 0.5, n)
    res = interrupted_time_series(y, t0, period=12, covariates=x)
    assert res.level_change == pytest.approx(4.0, abs=0.6)
    assert res.coef["cov0"] == pytest.approx(2.0, abs=0.2)
    assert "sin1" in res.coef and "cos1" in res.coef
    no_season = interrupted_time_series(y, t0, covariates=x)
    assert np.std(no_season.residuals) > 2 * np.std(res.residuals)
    # a dated axis: intervention given on the time scale
    res2 = interrupted_time_series(y, 2000 + t0 / 12, time=2000 + t / 12, period=1.0, covariates=x)
    assert res2.level_change == pytest.approx(res.level_change, rel=1e-6)
    assert res2.slope_change == pytest.approx(res.slope_change * 12, rel=1e-6)


def test_comparative_its_removes_common_shock():
    rng = np.random.default_rng(4)
    n, t0 = 100, 60
    t = np.arange(n, dtype=float)
    common = 6.0 * (t >= t0)  # concurrent shock hitting both series
    yc = 5 + 0.2 * t + common + rng.normal(0, 0.7, n)
    yt = 8 + 0.25 * t + common + 2.0 * (t >= t0) + rng.normal(0, 0.7, n)
    single = interrupted_time_series(yt, t0)
    cits = comparative_interrupted_time_series(yt, yc, t0)
    assert single.level_change > 6.5
    assert cits.level_change == pytest.approx(2.0, abs=0.8)
    assert cits.kind == "cits" and cits.relative_effect is None
    assert "Comparative" in cits.summary()
    with pytest.raises(ValueError):
        comparative_interrupted_time_series(yt, yc[:-1], t0)


def test_placebo_test_flags_misspecified_trend():
    rng = np.random.default_rng(1)
    t = np.arange(120, dtype=float)
    linear = 10 + 0.1 * t + rng.normal(0, 1.0, 120)
    cyclic = 10 + 3 * np.sin(t / 12) + rng.normal(0, 1.0, 120)  # unmodelled cycle
    ok = its_placebo_test(linear, 90, hac_lags=0)
    bad = its_placebo_test(cyclic, 90, hac_lags=0)
    assert ok["times"].size == 90 - 12 + 1
    assert ok["rejection_rate"] < 0.1
    assert bad["rejection_rate"] > 0.5
    assert bad["rejection_rate"] > ok["rejection_rate"]
    assert np.abs(bad["level_change"]).mean() > np.abs(ok["level_change"]).mean()


def test_validation():
    y = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="pre-intervention"):
        interrupted_time_series(y, 1)
    with pytest.raises(ValueError, match="post-intervention"):
        interrupted_time_series(y, 19.5)
    with pytest.raises(ValueError, match="increasing"):
        interrupted_time_series(y, 10, time=np.r_[np.arange(19), 5.0])
    with pytest.raises(ValueError, match="finite"):
        interrupted_time_series(np.r_[y[:-1], np.nan], 10)
    with pytest.raises(ValueError, match="alpha"):
        interrupted_time_series(y, 10, alpha=1.5)
    with pytest.raises(ValueError, match="covariates"):
        interrupted_time_series(y, 10, covariates=np.ones(5))


def test_cli_its(tmp_path, capsys):
    import pandas as pd

    from causal_inference.cli import main

    y = _series(seed=7)
    yc = 0.5 * y + np.random.default_rng(1).normal(0, 1, y.size)
    path = tmp_path / "its.csv"
    pd.DataFrame({"outcome": y, "ctrl": yc}).to_csv(path, index=False)
    assert main(["its", "--data", str(path), "--intervention", "70", "--placebo"]) == 0
    out = capsys.readouterr().out
    assert "Interrupted time series" in out and "level_change" in out and "placebo-in-time" in out
    assert main(["its", "--data", str(path), "--intervention", "70", "--control", "ctrl"]) == 0
    assert "Comparative" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["its", "--data", str(path), "--intervention", "70", "--outcome", "nope"])
