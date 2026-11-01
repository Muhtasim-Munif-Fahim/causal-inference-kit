"""Tests for targeted maximum likelihood estimation (tmle_ate)."""

import numpy as np
import pytest

from causal_inference import simulate_observational_data, tmle_ate
from causal_inference.cli import main
from causal_inference.tmle import _expit, _fluctuate, _logit


class ConstantModel:
    """Deliberately misspecified outcome model: predicts the arm mean."""

    def fit(self, X, y):
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


class OLS:
    def fit(self, X, y):
        Z = np.column_stack([np.ones(len(X)), X])
        self.beta_ = np.linalg.lstsq(Z, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.beta_


class HalfProb:
    """Deliberately wrong propensity model: always 0.5."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def _binary_data(n=4000, seed=1, strength=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    g = _expit(strength * x[:, 0] + 0.5 * x[:, 1])
    t = rng.binomial(1, g).astype(float)
    p0 = _expit(-1.0 + x[:, 0] + 0.8 * x[:, 1])
    p1 = _expit(-0.3 + x[:, 0] + 0.8 * x[:, 1])
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(float)
    return x, t, y, float(np.mean(p1 - p0)), float(p1.mean() / p0.mean())


def _confounded_linear(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    g = _expit(1.2 * x[:, 0])
    t = rng.binomial(1, g).astype(float)
    y = 1.0 + 2.0 * x[:, 0] + x[:, 1] + 1.5 * t + rng.normal(size=n)
    return x, t, y, g


def test_recovers_ate_continuous():
    X, W, T, Y, ate = simulate_observational_data(n=3000, seed=3)
    res = tmle_ate(np.hstack([X, W]), T, Y)
    assert res.outcome_type == "continuous"
    assert res.ci_lower < ate < res.ci_upper
    assert abs(res.estimate - (res.psi1 - res.psi0)) < 1e-10
    assert res.p_value < 1e-6


def test_solves_efficient_influence_function_equation():
    x, t, y, _ = _confounded_linear()
    res = tmle_ate(x, t, y)
    # targeted fit: the IC has mean zero (to Newton tolerance)
    assert abs(np.mean(res.influence)) < 1e-8
    g = res.propensity
    h = np.where(t == 1, 1 / g, -1 / (1 - g))
    qa = np.where(t == 1, res.q1, res.q0)
    assert abs(np.mean(h * (y - qa))) < 1e-8


def test_doubly_robust_wrong_outcome_model():
    x, t, y, _ = _confounded_linear(n=8000, seed=4)
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(naive - 1.5) > 1.0  # strong confounding
    res = tmle_ate(x, t, y, ml_q=ConstantModel())
    assert abs(res.initial_estimate - naive) < 1e-6  # constant Q => plug-in is naive
    assert abs(res.estimate - 1.5) < 0.2
    assert abs(res.epsilon) > 0


def test_doubly_robust_wrong_propensity_model():
    x, t, y, _ = _confounded_linear(n=8000, seed=5)
    res = tmle_ate(x, t, y, ml_q=OLS(), ml_g=HalfProb())
    assert abs(res.estimate - 1.5) < 0.1


def test_known_propensity_and_cross_fitting():
    x, t, y, g = _confounded_linear(n=4000, seed=6)
    a = tmle_ate(x, t, y, propensity=g)
    b = tmle_ate(x, t, y, n_folds=5, random_state=2)
    for res in (a, b):
        assert res.ci_lower < 1.5 < res.ci_upper
    assert b.n_folds == 5
    np.testing.assert_allclose(a.propensity, np.clip(g, 0.01, 0.99))


def test_binary_outcome_rr_or():
    x, t, y, rd, rr = _binary_data(n=6000)
    res = tmle_ate(x, t, y)
    assert res.outcome_type == "binary"
    assert res.ci_lower < rd < res.ci_upper
    assert res.rr_ci[0] < rr < res.rr_ci[1]
    assert 0 < res.psi0 < res.psi1 < 1
    assert res.odds_ratio > res.risk_ratio > 1
    assert "RR" in res.summary()


def test_substitution_estimator_respects_bounds_under_poor_overlap():
    x, t, y, *_ = _binary_data(n=1500, seed=9, strength=4.0)
    res = tmle_ate(x, t, y, truncation=0.0)
    assert 0.0 <= res.psi0 <= 1.0 and 0.0 <= res.psi1 <= 1.0
    assert -1.0 <= res.estimate <= 1.0
    assert np.all((res.q1 >= 0) & (res.q1 <= 1) & (res.q0 >= 0) & (res.q0 <= 1))


def test_truncation_counts_and_bounds_scaling():
    x, t, y, _ = _confounded_linear(n=2000, seed=7)
    loose = tmle_ate(x, t, y, truncation=0.2)
    assert loose.n_truncated > 0
    assert loose.propensity.min() >= 0.2 - 1e-12
    wide = tmle_ate(x, t, y, bounds=(y.min() - 10, y.max() + 10))
    tight = tmle_ate(x, t, y)
    assert wide.bounds[0] < tight.bounds[0]
    assert abs(wide.estimate - tight.estimate) < 0.1


def test_matches_aipw_closely_with_good_overlap():
    x, t, y, _ = _confounded_linear(n=5000, seed=8)
    res = tmle_ate(x, t, y)
    assert abs(res.estimate - res.aipw_estimate) < 0.02


def test_fluctuation_newton_solves_score():
    rng = np.random.default_rng(0)
    ys = rng.uniform(size=200)
    off = rng.normal(size=200)
    h = rng.normal(size=200)
    eps = _fluctuate(ys, off, h)
    assert abs(np.sum(h * (ys - _expit(off + eps * h)))) < 1e-8
    np.testing.assert_allclose(_expit(_logit(np.array([0.2, 0.7]))), [0.2, 0.7])


def test_validation_errors():
    x, t, y, _ = _confounded_linear(n=200)
    with pytest.raises(ValueError, match="binary 0/1"):
        tmle_ate(x, t * 2, y)
    with pytest.raises(ValueError, match="both treatment"):
        tmle_ate(x, np.ones_like(t), y)
    with pytest.raises(ValueError, match="truncation"):
        tmle_ate(x, t, y, truncation=0.5)
    with pytest.raises(ValueError, match="outside the supplied bounds"):
        tmle_ate(x, t, y, bounds=(0, 1))
    with pytest.raises(ValueError, match="binary outcome"):
        tmle_ate(x, t, y, outcome_type="binary")
    with pytest.raises(ValueError, match="propensity"):
        tmle_ate(x, t, y, propensity=np.ones(200))
    with pytest.raises(ValueError, match="outcome_type"):
        tmle_ate(x, t, y, outcome_type="count")


def test_cli_tmle(tmp_path, capsys):
    data = tmp_path / "obs.csv"
    assert main(["simulate", "--out", str(data), "--n", "800", "--seed", "1"]) == 0
    capsys.readouterr()
    assert main(["tmle", "--data", str(data), "--folds", "3"]) == 0
    out = capsys.readouterr().out
    assert "TMLE ATE" in out and "epsilon" in out and "n_truncated" in out
