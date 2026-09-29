"""Tests for T-learner CATE."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    TLearnerResult,
    outcome_regression,
    simulate_observational_data,
    t_learner,
)


def test_exports():
    assert callable(t_learner)
    assert TLearnerResult is not None


def test_mean_cate_recovers_ate_with_low_noise():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=4000,
        d_x=2,
        d_w=1,
        ate=2.0,
        confounding=0.5,
        selection_bias=0.0,
        effect_heterogeneity=0.0,
        seed=1,
    )
    result = t_learner(X, treatment, outcome, W=W)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.25)


def test_cate_equals_outcome_regression_difference():
    X, W, treatment, outcome, _ = simulate_observational_data(n=500, seed=2)
    result = t_learner(X, treatment, outcome, W=W)
    mu1, mu0 = outcome_regression(X, treatment, outcome, W=W)
    assert result.cate == pytest.approx(mu1 - mu0)
    assert result.mean_cate == pytest.approx(float(np.mean(mu1 - mu0)))
    assert result.n == len(treatment)
    assert result.n_treated + result.n_control == result.n


def test_captures_heterogeneity_direction():
    """With effect_heterogeneity > 0, CATE should correlate with X[:, 0]."""
    X, W, treatment, outcome, _ = simulate_observational_data(
        n=3000,
        d_x=2,
        d_w=0,
        ate=1.0,
        confounding=0.3,
        effect_heterogeneity=2.0,
        seed=5,
    )
    result = t_learner(X, treatment, outcome)
    corr = np.corrcoef(X[:, 0], result.cate)[0, 1]
    assert corr > 0.5


def test_without_W():
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=2000, d_w=0, ate=1.5, confounding=0.2, effect_heterogeneity=0.0, seed=3
    )
    result = t_learner(X, treatment, outcome)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.3)


def test_rejects_single_group():
    X = np.random.randn(40, 2)
    with pytest.raises(ValueError, match="both treatment"):
        t_learner(X, np.ones(40), np.random.randn(40))
    with pytest.raises(ValueError, match="both treatment"):
        t_learner(X, np.zeros(40), np.random.randn(40))


def test_cli_t_learner(capsys):
    from causal_inference.cli import main

    assert main(["t-learner", "--n", "400", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "T-learner" in out
    assert "mean_cate" in out
