"""Tests for linear product-of-coefficients mediation."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    MediationResult,
    linear_mediation,
    simulate_mediation_data,
)


def test_exports():
    assert callable(linear_mediation)
    assert MediationResult is not None


def test_recovers_paths_on_near_noise_free_sem():
    # Tiny mediator noise keeps M from being exactly collinear with binary T
    # (which would make the outcome design rank-deficient).
    treatment, mediator, outcome, covariates, truth = simulate_mediation_data(
        n=2000,
        direct_effect=1.2,
        a_path=0.9,
        b_path=1.5,
        noise_m=0.05,
        noise_y=0.05,
        confounding=0.0,
        seed=1,
    )
    result = linear_mediation(treatment, mediator, outcome)
    assert result.a_path == pytest.approx(truth["a_path"], abs=0.05)
    assert result.b_path == pytest.approx(truth["b_path"], abs=0.05)
    assert result.direct_effect == pytest.approx(truth["direct"], abs=0.05)
    assert result.indirect_effect == pytest.approx(truth["indirect"], abs=0.08)
    assert result.total_effect == pytest.approx(truth["total"], abs=0.05)
    assert result.total_effect == pytest.approx(
        result.direct_effect + result.indirect_effect, abs=0.05
    )


def test_recovers_approximately_with_noise():
    treatment, mediator, outcome, _, truth = simulate_mediation_data(
        n=5000,
        direct_effect=0.5,
        a_path=1.0,
        b_path=0.75,
        noise_m=0.5,
        noise_y=0.5,
        seed=7,
    )
    result = linear_mediation(treatment, mediator, outcome)
    assert result.direct_effect == pytest.approx(truth["direct"], abs=0.15)
    assert result.indirect_effect == pytest.approx(truth["indirect"], abs=0.15)
    assert result.total_effect == pytest.approx(truth["total"], abs=0.15)


def test_with_covariates():
    treatment, mediator, outcome, covariates, truth = simulate_mediation_data(
        n=3000,
        direct_effect=1.0,
        a_path=1.2,
        b_path=0.6,
        n_covariates=2,
        covariate_effect=0.8,
        noise_m=0.3,
        noise_y=0.3,
        seed=3,
    )
    result = linear_mediation(treatment, mediator, outcome, covariates=covariates)
    assert result.n_covariates == 2
    assert result.direct_effect == pytest.approx(truth["direct"], abs=0.15)
    assert result.indirect_effect == pytest.approx(truth["indirect"], abs=0.15)


def test_sobel_se_positive_and_finite():
    treatment, mediator, outcome, _, _ = simulate_mediation_data(n=500, seed=2)
    result = linear_mediation(treatment, mediator, outcome)
    assert result.se_indirect > 0.0
    assert np.isfinite(result.se_indirect)
    assert result.se_a > 0.0
    assert result.se_b > 0.0


def test_rejects_constant_treatment():
    n = 50
    with pytest.raises(ValueError, match="treatment must have variation"):
        linear_mediation(np.ones(n), np.random.randn(n), np.random.randn(n))


def test_rejects_constant_mediator():
    n = 50
    t = np.r_[np.zeros(25), np.ones(25)]
    with pytest.raises(ValueError, match="mediator must have variation"):
        linear_mediation(t, np.ones(n), np.random.randn(n))


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        linear_mediation(np.ones(10), np.ones(9), np.ones(10))


def test_rejects_too_few_observations():
    with pytest.raises(ValueError, match="at least three"):
        linear_mediation([0, 1], [0.1, 0.2], [1.0, 2.0])


def test_generator_true_effects_consistent():
    *_, truth = simulate_mediation_data(
        n=100, direct_effect=2.0, a_path=1.5, b_path=0.5, seed=0
    )
    assert truth["indirect"] == pytest.approx(0.75)
    assert truth["total"] == pytest.approx(2.75)


def test_cli_mediation(capsys):
    from causal_inference.cli import main

    rc = main(["mediation", "--n", "800", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "linear mediation" in out
    assert "indirect" in out
