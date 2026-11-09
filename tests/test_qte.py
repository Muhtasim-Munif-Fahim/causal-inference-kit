"""Tests for IPW quantile treatment effects (Firpo 2007)."""

import numpy as np
import pytest

from causal_inference import (
    QuantileTreatmentEffectResult,
    ipw_weighted_quantile,
    quantile_treatment_effects,
)

TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


def _confounded(n=6000, seed=0):
    """Treatment spreads the outcome: QTE negative low, positive high, ATE ~ 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-1.2 * x))
    t = (rng.random(n) < p).astype(int)
    e = rng.normal(size=n)
    y0 = 2.0 * x + e
    y1 = 2.0 * x + 1.0 + 2.5 * e  # wider residual spread
    y = np.where(t == 1, y1, y0)
    return x.reshape(-1, 1), t, y, y0, y1, p


def test_weighted_quantile_matches_numpy_with_equal_weights():
    rng = np.random.default_rng(3)
    v = rng.normal(size=101)
    got = ipw_weighted_quantile(v, np.ones_like(v), TAUS)
    want = np.quantile(v, TAUS, method="inverted_cdf")
    np.testing.assert_allclose(got, want)


def test_weighted_quantile_respects_weights():
    v = np.array([0.0, 1.0, 2.0])
    assert ipw_weighted_quantile(v, [1, 1, 8], [0.5])[0] == 2.0
    assert ipw_weighted_quantile(v, [8, 1, 1], [0.5])[0] == 0.0


def test_recovers_true_qte_under_confounding():
    X, t, y, y0, y1, _ = _confounded()
    truth = np.quantile(y1, TAUS) - np.quantile(y0, TAUS)
    res = quantile_treatment_effects(X, t, y, quantiles=TAUS)
    assert isinstance(res, QuantileTreatmentEffectResult)
    np.testing.assert_allclose(res.qte, truth, atol=0.3)
    naive = np.quantile(y[t == 1], TAUS) - np.quantile(y[t == 0], TAUS)
    assert np.abs(naive - truth).max() > 1.0
    assert res.qte[0] < 0 < res.qte[-1]


def test_known_propensity_and_att_target():
    X, t, y, y0, y1, p = _confounded(seed=1)
    treated = t == 1
    truth = np.quantile(y1[treated], TAUS) - np.quantile(y0[treated], TAUS)
    res = quantile_treatment_effects(X, t, y, quantiles=TAUS, propensity=p, target="att")
    np.testing.assert_allclose(res.qte, truth, atol=0.35)
    np.testing.assert_allclose(res.q_treated, np.quantile(y[treated], TAUS, method="inverted_cdf"))


def test_randomized_equals_raw_quantile_difference():
    rng = np.random.default_rng(4)
    n = 500
    t = rng.integers(0, 2, n)
    y = rng.normal(size=n) + t
    res = quantile_treatment_effects(np.zeros((n, 1)), t, y, quantiles=TAUS, propensity=np.full(n, 0.5))
    want = np.quantile(y[t == 1], TAUS, method="inverted_cdf") - np.quantile(
        y[t == 0], TAUS, method="inverted_cdf"
    )
    np.testing.assert_allclose(res.qte, want)


def test_bootstrap_inference_covers_truth():
    X, t, y, y0, y1, _ = _confounded(n=3000, seed=2)
    truth = np.quantile(y1, TAUS) - np.quantile(y0, TAUS)
    res = quantile_treatment_effects(X, t, y, quantiles=TAUS, n_bootstrap=120, seed=0, trim=0.01)
    assert res.bootstrap_draws.shape == (120, TAUS.size)
    assert np.all(res.se > 0)
    assert np.all(res.ci_lower <= res.qte) and np.all(res.qte <= res.ci_upper)
    covered = (res.ci_lower <= truth) & (truth <= res.ci_upper)
    assert covered.sum() >= 4
    d = res.to_dict()
    assert d["target"] == "ate" and len(d["rows"]) == 5 and "se" in d["rows"][0]
    assert "QTE" in res.summary()


def test_reproducible_bootstrap():
    X, t, y, *_ = _confounded(n=800, seed=5)
    a = quantile_treatment_effects(X, t, y, n_bootstrap=20, seed=7)
    b = quantile_treatment_effects(X, t, y, n_bootstrap=20, seed=7)
    np.testing.assert_allclose(a.bootstrap_draws, b.bootstrap_draws)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantiles": [0.0]},
        {"quantiles": [1.2]},
        {"target": "atc"},
        {"ci": 1.0},
        {"trim": 0.6},
        {"n_bootstrap": -1},
    ],
)
def test_validation(kwargs):
    X, t, y, *_ = _confounded(n=200)
    with pytest.raises(ValueError):
        quantile_treatment_effects(X, t, y, **kwargs)


def test_requires_both_groups():
    with pytest.raises(ValueError):
        quantile_treatment_effects(np.zeros((5, 1)), np.ones(5), np.arange(5.0))


def test_cli_qte(tmp_path, capsys):
    from causal_inference.cli import main

    data = tmp_path / "obs.csv"
    assert main(["simulate", "--out", str(data), "--n", "600", "--seed", "2"]) == 0
    capsys.readouterr()
    assert main(["qte", "--data", str(data), "--boot", "30", "--quantiles", "0.25", "0.5", "0.75"]) == 0
    out = capsys.readouterr().out
    assert "Firpo" in out and "tau=0.50" in out and "CI" in out
