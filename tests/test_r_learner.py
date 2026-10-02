"""Tests for R-learner CATE."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    RLearnerResult,
    simulate_observational_data,
    s_learner,
    t_learner,
    x_learner,
    r_learner,
)


def test_exports():
    assert callable(r_learner)
    assert RLearnerResult is not None


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
    result = r_learner(X, treatment, outcome, W=W)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.35)


def test_cate_shape_and_counts():
    X, W, treatment, outcome, _ = simulate_observational_data(
        n=800, d_x=2, d_w=1, effect_heterogeneity=0.5, seed=2
    )
    result = r_learner(X, treatment, outcome, W=W)
    assert result.cate.shape == (len(treatment),)
    assert result.n == len(treatment)
    assert result.n_treated + result.n_control == result.n
    assert np.isfinite(result.cate).all()


def test_matches_peers_under_homogeneous_effect():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=3000,
        d_x=2,
        d_w=0,
        ate=1.5,
        confounding=0.3,
        effect_heterogeneity=0.0,
        seed=4,
    )
    r_res = r_learner(X, treatment, outcome)
    t_res = t_learner(X, treatment, outcome)
    s_res = s_learner(X, treatment, outcome)
    x_res = x_learner(X, treatment, outcome)
    assert r_res.mean_cate == pytest.approx(t_res.mean_cate, abs=0.3)
    assert r_res.mean_cate == pytest.approx(s_res.mean_cate, abs=0.3)
    assert r_res.mean_cate == pytest.approx(x_res.mean_cate, abs=0.3)
    assert r_res.mean_cate == pytest.approx(true_ate, abs=0.4)


def test_without_W():
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=2000, d_w=0, ate=1.5, confounding=0.2, effect_heterogeneity=0.0, seed=3
    )
    result = r_learner(X, treatment, outcome)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.4)


def test_rejects_single_group():
    X = np.random.randn(40, 2)
    with pytest.raises(ValueError, match="both treatment"):
        r_learner(X, np.ones(40), np.random.randn(40))
    with pytest.raises(ValueError, match="both treatment"):
        r_learner(X, np.zeros(40), np.random.randn(40))


def test_cli_r_learner(capsys):
    from causal_inference.cli import main

    assert main(["r-learner", "--n", "400", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "R-learner" in out
    assert "mean_cate" in out
