"""Tests for entropy balancing (Hainmueller, 2012)."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    difference_in_means,
    entropy_balancing_ate,
    entropy_balancing_att,
    entropy_balancing_weights,
    simulate_observational_data,
    weighted_standardized_mean_differences,
)
from causal_inference.evaluate import standard_estimators


def _data(n=6000, confounding=1.5, seed=0):
    return simulate_observational_data(n=n, confounding=confounding, seed=seed)


def test_att_weights_match_treated_means_exactly():
    X, _, t, _, _ = _data(n=3000)
    w = entropy_balancing_weights(X, t, estimand="att")
    treated = t == 1
    assert np.allclose(w[treated], 1.0)
    assert w[~treated].sum() == pytest.approx(treated.sum())
    assert np.all(w > 0)
    wc = w[~treated] / w[~treated].sum()
    np.testing.assert_allclose(wc @ X[~treated], X[treated].mean(axis=0), atol=1e-6)
    smd = weighted_standardized_mean_differences(X, t, w)
    assert np.max(np.abs(smd)) < 1e-6


def test_ate_weights_match_full_sample_in_both_groups():
    X, _, t, _, _ = _data(n=3000)
    w = entropy_balancing_weights(X, t, estimand="ate")
    for group in (t == 1, t == 0):
        assert w[group].sum() == pytest.approx(group.sum())
        wg = w[group] / w[group].sum()
        np.testing.assert_allclose(wg @ X[group], X.mean(axis=0), atol=1e-6)


def test_second_moments_are_balanced():
    X, _, t, _, _ = _data(n=3000)
    w = entropy_balancing_weights(X, t, estimand="att", moments=2)
    treated = t == 1
    wc = w[~treated] / w[~treated].sum()
    np.testing.assert_allclose(wc @ X[~treated] ** 2, (X[treated] ** 2).mean(axis=0), atol=1e-5)


def test_already_balanced_groups_get_uniform_weights():
    X = np.array([[0.0], [2.0], [0.0], [2.0], [1.0], [1.0]])
    t = np.array([1, 1, 0, 0, 0, 0])
    w = entropy_balancing_weights(X, t)
    np.testing.assert_allclose(w[t == 0], np.full(4, 0.5))


def test_closed_form_two_point_control():
    # Controls at 0 and 1 (uniform base); treated mean 0.75 -> weights 1/4, 3/4.
    X = np.array([[0.5], [1.0], [0.0], [1.0]])
    t = np.array([1, 1, 0, 0])
    w = entropy_balancing_weights(X, t)
    np.testing.assert_allclose(w[t == 0] / w[t == 0].sum(), [0.25, 0.75], atol=1e-7)


def test_base_weights_are_respected():
    X = np.array([[0.0], [2.0], [0.0], [2.0]])
    t = np.array([1, 1, 0, 0])
    w = entropy_balancing_weights(X, t, base_weights=[1, 1, 1, 3])
    # target already met by uniform control weights; base tilts them, so the
    # solver must pull them back to equal effective mass.
    np.testing.assert_allclose(w[t == 0], [1.0, 1.0], atol=1e-7)


def test_effect_estimates_remove_observed_confounding():
    X, _, t, y, true_ate = _data(n=20000, confounding=1.5)
    naive = difference_in_means(t, y)
    att = entropy_balancing_att(X, t, y)
    ate = entropy_balancing_ate(X, t, y)
    assert abs(ate - true_ate) < abs(naive - true_ate)
    assert abs(ate - true_ate) < 0.35
    assert np.isfinite(att)


def test_infeasible_target_raises():
    X = np.array([[5.0], [6.0], [0.0], [1.0]])
    t = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError, match="convex hull"):
        entropy_balancing_weights(X, t)


def test_validation():
    X, _, t, y, _ = _data(n=200)
    with pytest.raises(ValueError, match="estimand"):
        entropy_balancing_weights(X, t, estimand="atc")
    with pytest.raises(ValueError, match="moments"):
        entropy_balancing_weights(X, t, moments=3)
    with pytest.raises(ValueError, match="base_weights"):
        entropy_balancing_weights(X, t, base_weights=np.zeros(len(t)))
    with pytest.raises(ValueError):
        entropy_balancing_weights(X, np.ones(len(t)))
    with pytest.raises(ValueError):
        entropy_balancing_att(X, t, y[:10])


def test_registered_in_standard_estimators():
    assert "entropy_balancing" in standard_estimators()
