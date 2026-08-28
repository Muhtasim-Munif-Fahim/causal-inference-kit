"""Tests for the markdown report renderer."""

import numpy as np
import pytest

from causal_inference.evaluate import EvaluationResult
from causal_inference.generators import simulate_observational_data
from causal_inference.report import (
    assumption_caveats,
    balance_table,
    estimates_table,
    evaluation_table,
    render_report,
)
from causal_inference.estimators import ipw_weights


def _data(n=1500, seed=1):
    return simulate_observational_data(n=n, seed=seed)


def _results():
    return [
        EvaluationResult(name="naive", estimate=3.1, se=0.4, bias=0.9, rmse=1.0),
        EvaluationResult(name="ipw", estimate=2.4, se=0.3, bias=0.2, rmse=0.4),
    ]


def test_balance_table_contains_covariate_names():
    X, W, t, _, _ = _data()
    table = balance_table(X, W, t)
    assert "| Covariate | SMD |" in table
    assert "| X0 |" in table
    assert "| W0 |" in table


def test_balance_table_smd_values():
    X, W, t, _, _ = simulate_observational_data(n=5000, confounding=2.0, seed=2)
    table = balance_table(X, W, t)
    row = next(line for line in table.splitlines() if line.startswith("| X"))
    value = float(row.split("|")[2].strip())
    assert abs(value) > 0.1


def test_balance_table_with_weights_adds_column():
    X, W, t, _, _ = _data()
    weights = ipw_weights(X, t)
    table = balance_table(X, W, t, weights=weights)
    assert "| Weighted SMD |" in table
    separator = table.splitlines()[1]
    assert separator.count("---") == 3


def test_balance_table_requires_covariates():
    X = np.zeros((100, 0))
    W = np.zeros((100, 0))
    t = np.array([1] * 50 + [0] * 50)
    with pytest.raises(ValueError):
        balance_table(X, W, t)


def test_balance_table_custom_names():
    X, W, t, _, _ = _data(n=500)
    table = balance_table(
        X,
        W,
        t,
        x_names=["x0", "x1", "x2"],
        w_names=["w0", "w1"],
    )
    assert "| x0 |" in table
    assert "| w1 |" in table


def test_estimates_table_with_se():
    table = estimates_table([("naive", 3.1, 0.4), ("ipw", 2.4, 0.3)])
    assert "| Estimator | Estimate | SE |" in table
    assert "| naive | 3.100 | 0.400 |" in table
    assert "| ipw | 2.400 | 0.300 |" in table


def test_estimates_table_without_se():
    table = estimates_table([("naive", 3.1)])
    assert "| naive | 3.100 | - |" in table


def test_evaluation_table_renders():
    table = evaluation_table(_results())
    assert "| Estimator | Estimate | SE | Bias | RMSE |" in table
    assert "| ipw | 2.400 | 0.300 | 0.200 | 0.400 |" in table


def test_assumption_caveats_mentions_key_assumptions():
    text = assumption_caveats()
    for keyword in ("Ignorability", "Overlap", "SUTVA", "parallel trends"):
        assert keyword.lower() in text.lower()


def test_render_report_contains_all_sections():
    X, W, t, y, true_ate = _data()
    weights = ipw_weights(X, t)
    report = render_report(X, W, t, y, _results(), true_ate, weights=weights)
    for section in (
        "# Causal inference report",
        "## Covariate balance",
        "## Point estimates",
        "## Bias and RMSE vs ground truth",
        "### Assumption caveats",
    ):
        assert section in report


def test_render_report_data_summary():
    X, W, t, y, true_ate = _data(n=800)
    report = render_report(X, W, t, y, _results(), true_ate)
    assert "800 units" in report
    assert f"{true_ate:.3f}" in report


def test_render_report_appends_extra_markdown():
    X, W, t, y, true_ate = _data(n=500)
    report = render_report(
        X, W, t, y, _results(), true_ate, extra_markdown="## DiD section\n\n0.9"
    )
    assert "## DiD section" in report
    assert report.index("## DiD section") < report.index("### Assumption caveats")
