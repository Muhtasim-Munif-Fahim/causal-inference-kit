"""Tests for overlap and matching weights (Li, Morgan & Zaslavsky)."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference.estimators import (
    difference_in_means,
    ipw_ate,
    matching_weights,
    matching_weights_ate,
    overlap_ate,
    overlap_weights,
)
from causal_inference.evaluate import standard_estimators
from causal_inference.generators import simulate_observational_data
from causal_inference.propensity import propensity_scores


def _confounded(n=20000, confounding=1.5, seed=0):
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=n, confounding=confounding, seed=seed
    )
    return X, W, treatment, outcome, true_ate


def test_overlap_weights_formula():
    X, _, treatment, outcome, _ = _confounded(n=800)
    p, _ = propensity_scores(X, treatment)
    w = overlap_weights(X, treatment, propensity=p)
    expected = np.where(treatment == 1, 1.0 - p, p)
    np.testing.assert_allclose(w, expected)
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)


def test_matching_weights_formula():
    X, _, treatment, _, _ = _confounded(n=800)
    p, _ = propensity_scores(X, treatment)
    w = matching_weights(X, treatment, propensity=p)
    m = np.minimum(p, 1.0 - p)
    expected = np.where(treatment == 1, m / p, m / (1.0 - p))
    np.testing.assert_allclose(w, expected)
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0 + 1e-12)


def test_overlap_ate_recovers_without_selection_bias():
    X, _, treatment, outcome, true_ate = _confounded()
    estimate = overlap_ate(X, treatment, outcome)
    assert abs(estimate - true_ate) < 0.35


def test_matching_weights_ate_recovers_without_selection_bias():
    X, _, treatment, outcome, true_ate = _confounded()
    estimate = matching_weights_ate(X, treatment, outcome)
    assert abs(estimate - true_ate) < 0.35


def test_overlap_improves_on_naive():
    X, _, treatment, outcome, true_ate = _confounded()
    ov = overlap_ate(X, treatment, outcome)
    naive = difference_in_means(treatment, outcome)
    assert abs(ov - true_ate) < abs(naive - true_ate)


def test_overlap_close_to_ipw_when_balanced():
    # Low confounding → overlap and IPW should agree closely.
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=15000, confounding=0.2, seed=11
    )
    ov = overlap_ate(X, treatment, outcome)
    ipw = ipw_ate(X, treatment, outcome)
    assert abs(ov - ipw) < 0.25
    assert abs(ov - true_ate) < 0.3


def test_overlap_uses_explicit_propensity():
    X, _, treatment, outcome, _ = _confounded(n=3000)
    p, _ = propensity_scores(X, treatment)
    assert overlap_ate(X, treatment, outcome, propensity=p) == pytest.approx(
        overlap_ate(X, treatment, outcome), abs=1e-8
    )


def test_rejects_bad_inputs():
    X, _, treatment, outcome, _ = _confounded(n=100)
    with pytest.raises(ValueError):
        overlap_weights(X, treatment[:50])
    with pytest.raises(ValueError):
        overlap_ate(X, treatment, outcome[:50])
    with pytest.raises(ValueError):
        matching_weights(X, np.ones(100))
    with pytest.raises(ValueError):
        overlap_weights(X, treatment, propensity=np.zeros(100))


def test_exports_and_standard_estimators():
    from causal_inference import matching_weights_ate as mwa
    from causal_inference import overlap_ate as oa
    from causal_inference import overlap_weights as ow

    assert callable(oa) and callable(ow) and callable(mwa)
    names = standard_estimators()
    assert "overlap" in names
    assert "matching_weights" in names
