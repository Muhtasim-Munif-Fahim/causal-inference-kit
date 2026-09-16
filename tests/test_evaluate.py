"""Tests for estimator evaluation against ground truth."""

import numpy as np
import pytest

from causal_inference.estimators import aipw_ate, difference_in_means, ipw_ate, propensity_matching
from causal_inference.evaluate import EvaluationResult, evaluate, standard_estimators
from causal_inference.generators import simulate_observational_data


def _data(n=15000, confounding=1.0, seed=0):
    return simulate_observational_data(n=n, confounding=confounding, seed=seed)


def _estimators():
    return {
        "naive": lambda X, W, t, y, p: difference_in_means(t, y),
        "ipw": lambda X, W, t, y, p: ipw_ate(X, t, y, propensity=p),
        "aipw": lambda X, W, t, y, p: aipw_ate(X, t, y, propensity=p, W=W),
        "matching": lambda X, W, t, y, p: propensity_matching(
            X, t, y, propensity=p, with_replacement=True
        ),
    }


def test_evaluate_returns_one_result_per_estimator():
    X, W, t, y, true_ate = _data()
    results = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=50)
    assert [r.name for r in results] == ["naive", "ipw", "aipw", "matching"]
    assert all(isinstance(r, EvaluationResult) for r in results)


def test_evaluate_bias_small_for_aipw_when_ignorable():
    X, W, t, y, true_ate = _data()
    results = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=100)
    aipw = next(r for r in results if r.name == "aipw")
    assert abs(aipw.bias) < 0.3


def test_evaluate_bias_small_for_ipw_when_ignorable():
    X, W, t, y, true_ate = _data()
    results = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=100)
    ipw = next(r for r in results if r.name == "ipw")
    assert abs(ipw.bias) < 0.3


def test_evaluate_naive_bias_larger_than_ipw_bias():
    X, W, t, y, true_ate = _data(confounding=1.5, seed=3)
    results = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=100)
    naive = next(r for r in results if r.name == "naive")
    ipw = next(r for r in results if r.name == "ipw")
    assert abs(naive.bias) > abs(ipw.bias)


def test_evaluate_se_positive_and_finite():
    X, W, t, y, true_ate = _data()
    results = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=50)
    for result in results:
        assert np.isfinite(result.se)
        assert result.se >= 0


def test_evaluate_rmse_consistent_with_bias_and_variance():
    X, W, t, y, true_ate = _data()
    results = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=200)
    for result in results:
        assert result.rmse ** 2 == pytest.approx(
            result.bias ** 2 + result.se ** 2, rel=0.02
        )


def test_evaluate_seed_reproducibility():
    X, W, t, y, true_ate = _data()
    a = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=60, seed=11)
    b = evaluate(_estimators(), X, W, t, y, true_ate, n_boot=60, seed=11)
    for left, right in zip(a, b):
        assert left.estimate == right.estimate
        assert left.se == right.se
        assert left.bias == right.bias


def test_evaluate_empty_estimators():
    X, W, t, y, true_ate = _data()
    assert evaluate([], X, W, t, y, true_ate) == []


def test_evaluate_list_of_pairs():
    X, W, t, y, true_ate = _data()
    pairs = [("ipw", lambda X, W, t, y, p: ipw_ate(X, t, y, propensity=p))]
    results = evaluate(pairs, X, W, t, y, true_ate, n_boot=30)
    assert len(results) == 1
    assert results[0].name == "ipw"


def test_standard_estimators_names():
    assert list(standard_estimators()) == ["naive", "ipw", "aipw", "ipw_att", "matching"]


def test_standard_estimators_recover_truth():
    X, W, t, y, true_ate = _data()
    results = evaluate(standard_estimators(), X, W, t, y, true_ate, n_boot=80)
    for result in results:
        if result.name in ("ipw", "aipw", "ipw_att"):
            assert abs(result.bias) < 0.3
    matching = next(r for r in results if r.name == "matching")
    assert abs(matching.estimate - true_ate) < 0.6


def test_evaluate_rejects_bare_callable():
    X, W, t, y, true_ate = _data(n=500)
    with pytest.raises(TypeError):
        evaluate([ipw_ate], X, W, t, y, true_ate)


def test_evaluate_input_validation():
    X, W, t, y, true_ate = _data(n=500)
    with pytest.raises(ValueError):
        evaluate(_estimators(), X, W[:250], t, y, true_ate)
    with pytest.raises(ValueError):
        evaluate(_estimators(), X, W, t, y[:250], true_ate)
    with pytest.raises(ValueError):
        evaluate(_estimators(), X, W, t, y, true_ate, n_boot=0)


def test_evaluate_counts_bootstrap_failures():
    X, W, t, y, true_ate = _data(n=300, seed=5)
    fragile = {
        "fragile": lambda X, W, t, y, p: (
            _raise_if_thin(t) if t.mean() < 0.52 else ipw_ate(X, t, y, propensity=p)
        )
    }
    results = evaluate(fragile, X, W, t, y, true_ate, n_boot=80, seed=0)
    result = results[0]
    assert 0 < result.bootstrap_failures <= 40
    assert np.isfinite(result.estimate)


def _raise_if_thin(treatment):
    if treatment.mean() < 0.52:
        raise ValueError("too few treated units")


def test_evaluate_raises_when_estimator_always_fails():
    X, W, t, y, true_ate = _data(n=500)
    broken = {"broken": lambda X, W, t, y, p: _raise_if_thin(np.zeros_like(t))}
    with pytest.raises(ValueError):
        evaluate(broken, X, W, t, y, true_ate, n_boot=30)
