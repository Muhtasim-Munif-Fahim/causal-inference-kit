"""Tests for the synthetic data generator."""

import numpy as np
import pytest

from causal_inference.generators import simulate_observational_data


def test_returns_expected_shapes():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=500, d_x=3, d_w=2, seed=1
    )
    assert X.shape == (500, 3)
    assert W.shape == (500, 2)
    assert treatment.shape == (500,)
    assert outcome.shape == (500,)
    assert np.isfinite(true_ate)


def test_seed_reproducibility():
    a = simulate_observational_data(seed=7)
    b = simulate_observational_data(seed=7)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_different_seeds_differ():
    a = simulate_observational_data(seed=1)
    b = simulate_observational_data(seed=2)
    assert not np.array_equal(a[0], b[0])
    assert not np.array_equal(a[3], b[3])


def test_treatment_has_both_groups():
    _, _, treatment, _, _ = simulate_observational_data(n=2000, seed=3)
    assert set(np.unique(treatment)) == {0.0, 1.0}
    share = treatment.mean()
    assert 0.2 < share < 0.8


def test_first_covariate_of_each_block_is_binary():
    X, W, _, _, _ = simulate_observational_data(d_x=4, d_w=3, seed=5)
    assert set(np.unique(X[:, 0])) <= {0.0, 1.0}
    assert set(np.unique(W[:, 0])) <= {0.0, 1.0}


def test_true_ate_equals_requested_ate_without_heterogeneity():
    for seed in range(5):
        _, _, _, _, true_ate = simulate_observational_data(
            ate=2.5, effect_heterogeneity=0.0, seed=seed
        )
        assert true_ate == pytest.approx(2.5)


def test_true_ate_uses_first_confounder_when_heterogeneous():
    X, _, _, _, true_ate = simulate_observational_data(
        ate=1.0, effect_heterogeneity=4.0, seed=11
    )
    expected = 1.0 + 4.0 * X[:, 0].mean()
    assert true_ate == pytest.approx(expected)


def test_true_att_differs_from_ate_under_selection_on_effect_modifier():
    X, _, treatment, _, true_ate = simulate_observational_data(
        ate=1.0, effect_heterogeneity=5.0, confounding=2.0, n=5000, seed=13
    )
    true_att = 1.0 + 5.0 * X[treatment == 1, 0].mean()
    assert abs(true_att - true_ate) > 0.1


def test_confounding_creates_imbalance():
    X, _, treatment, _, _ = simulate_observational_data(
        confounding=3.0, n=5000, seed=17
    )
    d = X[treatment == 1].mean(axis=0) - X[treatment == 0].mean(axis=0)
    assert np.abs(d).max() > 0.15


def test_no_confounding_gives_balanced_groups():
    X, _, treatment, _, _ = simulate_observational_data(
        confounding=0.0, n=5000, seed=19
    )
    d = X[treatment == 1].mean(axis=0) - X[treatment == 0].mean(axis=0)
    assert np.abs(d).max() < 0.15


def test_outcome_only_covariates_balanced_across_groups():
    _, W, treatment, _, _ = simulate_observational_data(
        confounding=3.0, n=5000, seed=23
    )
    d = W[treatment == 1].mean(axis=0) - W[treatment == 0].mean(axis=0)
    assert np.abs(d).max() < 0.15


def test_selection_bias_inflates_naive_estimator_error():
    naive_errors = {0.0: [], 3.0: []}
    for seed in range(8):
        for sb in (0.0, 3.0):
            _, _, treatment, outcome, true_ate = simulate_observational_data(
                n=8000,
                confounding=0.0,
                selection_bias=sb,
                effect_heterogeneity=0.0,
                seed=seed,
            )
            naive = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
            naive_errors[sb].append(abs(naive - true_ate))
    assert np.mean(naive_errors[3.0]) > np.mean(naive_errors[0.0])


def test_zero_ate_dataset():
    _, _, treatment, outcome, true_ate = simulate_observational_data(
        ate=0.0, effect_heterogeneity=0.0, seed=29
    )
    assert true_ate == pytest.approx(0.0)
    assert np.isfinite(outcome).all()


def test_small_n():
    X, W, treatment, outcome, true_ate = simulate_observational_data(n=10, seed=31)
    assert X.shape[0] == 10
    assert X.shape[1] == 3
    assert treatment.sum() >= 0
    assert np.isfinite(true_ate)


def test_zero_covariate_dimensions():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        d_x=0, d_w=0, n=100, seed=37
    )
    assert X.shape == (100, 0)
    assert W.shape == (100, 0)
    assert np.isfinite(outcome).all()
    assert np.isfinite(true_ate)


def test_zero_covariate_dimensions_balanced_treatment():
    _, _, treatment, _, _ = simulate_observational_data(
        d_x=0, d_w=0, n=5000, seed=41
    )
    assert 0.4 < treatment.mean() < 0.6


def test_outcome_is_continuous_and_finite():
    for seed in range(4):
        _, _, _, outcome, _ = simulate_observational_data(seed=seed)
        assert np.isfinite(outcome).all()
        assert len(np.unique(outcome)) > 10
