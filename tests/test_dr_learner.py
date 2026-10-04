"""Tests for DR-learner CATE."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    DRLearnerResult,
    simulate_observational_data,
    s_learner,
    t_learner,
    x_learner,
    r_learner,
    dr_learner,
)


def test_exports():
    assert callable(dr_learner)
    assert DRLearnerResult is not None


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
    result = dr_learner(X, treatment, outcome, W=W)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.35)


def test_cate_shape_and_counts():
    X, W, treatment, outcome, _ = simulate_observational_data(
        n=800, d_x=2, d_w=1, effect_heterogeneity=0.5, seed=2
    )
    result = dr_learner(X, treatment, outcome, W=W)
    assert result.cate.shape == (len(treatment),)
    assert result.n == len(treatment)
    assert result.n_treated + result.n_control == result.n
    assert np.isfinite(result.cate).all()


def test_comparison_sanity_vs_t_learner():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=3000,
        d_x=2,
        d_w=0,
        ate=1.5,
        confounding=0.3,
        effect_heterogeneity=0.0,
        seed=4,
    )
    dr_res = dr_learner(X, treatment, outcome)
    t_res = t_learner(X, treatment, outcome)
    s_res = s_learner(X, treatment, outcome)
    x_res = x_learner(X, treatment, outcome)
    r_res = r_learner(X, treatment, outcome)
    assert dr_res.mean_cate == pytest.approx(t_res.mean_cate, abs=0.35)
    assert dr_res.mean_cate == pytest.approx(s_res.mean_cate, abs=0.35)
    assert dr_res.mean_cate == pytest.approx(x_res.mean_cate, abs=0.35)
    assert dr_res.mean_cate == pytest.approx(r_res.mean_cate, abs=0.35)
    assert dr_res.mean_cate == pytest.approx(true_ate, abs=0.4)


def test_without_W():
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=2000, d_w=0, ate=1.5, confounding=0.2, effect_heterogeneity=0.0, seed=3
    )
    result = dr_learner(X, treatment, outcome)
    assert result.mean_cate == pytest.approx(true_ate, abs=0.4)


def test_rejects_single_group():
    X = np.random.randn(40, 2)
    with pytest.raises(ValueError, match="both treatment"):
        dr_learner(X, np.ones(40), np.random.randn(40))
    with pytest.raises(ValueError, match="both treatment"):
        dr_learner(X, np.zeros(40), np.random.randn(40))


def test_cli_dr_learner(capsys):
    from causal_inference.cli import main

    assert main(["dr-learner", "--n", "400", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "DR-learner" in out
    assert "mean_cate" in out
