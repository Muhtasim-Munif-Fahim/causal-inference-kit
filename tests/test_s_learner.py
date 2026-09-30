"""Tests for S-learner CATE."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    SLearnerResult,
    simulate_observational_data,
    s_learner,
    t_learner,
)


def test_exports():
    assert callable(s_learner)
    assert SLearnerResult is not None


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
    result = s_learner(X, treatment, outcome, W=W)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.25)


def test_cate_is_constant_under_linear_base_learner():
    """Without treatment–covariate interactions the linear S-learner CATE is flat."""
    X, W, treatment, outcome, _ = simulate_observational_data(
        n=800, d_x=2, d_w=1, effect_heterogeneity=0.0, seed=2
    )
    result = s_learner(X, treatment, outcome, W=W)
    assert result.cate.std() == pytest.approx(0.0, abs=1e-10)
    assert result.n == len(treatment)
    assert result.n_treated + result.n_control == result.n


def test_matches_t_learner_ate_under_homogeneous_effect():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=3000,
        d_x=2,
        d_w=0,
        ate=1.5,
        confounding=0.3,
        effect_heterogeneity=0.0,
        seed=4,
    )
    s_res = s_learner(X, treatment, outcome)
    t_res = t_learner(X, treatment, outcome)
    assert s_res.mean_cate == pytest.approx(t_res.mean_cate, abs=0.15)
    assert s_res.mean_cate == pytest.approx(true_ate, abs=0.3)


def test_without_W():
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=2000, d_w=0, ate=1.5, confounding=0.2, effect_heterogeneity=0.0, seed=3
    )
    result = s_learner(X, treatment, outcome)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.3)


def test_rejects_single_group():
    X = np.random.randn(40, 2)
    with pytest.raises(ValueError, match="both treatment"):
        s_learner(X, np.ones(40), np.random.randn(40))
    with pytest.raises(ValueError, match="both treatment"):
        s_learner(X, np.zeros(40), np.random.randn(40))


def test_cli_s_learner(capsys):
    from causal_inference.cli import main

    assert main(["s-learner", "--n", "400", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "S-learner" in out
    assert "mean_cate" in out
