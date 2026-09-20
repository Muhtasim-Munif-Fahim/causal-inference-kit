"""Markdown report rendering."""

from __future__ import annotations

import numpy as np

from .propensity import (
    standardized_mean_differences,
    weighted_standardized_mean_differences,
)


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def balance_table(X, W, treatment, weights=None, x_names=None, w_names=None) -> str:
    """Render a markdown table of standardized mean differences.

    Columns are the covariate name and the SMD; when ``weights`` is given a
    third column shows the weighted SMD.

    Parameters
    ----------
    X : array-like of shape (n, d_x)
    W : array-like of shape (n, d_w)
    treatment : array-like of shape (n,)
    weights : array-like of shape (n,), optional
        Per-unit weights for the weighted SMD column.
    x_names, w_names : sequence of str, optional
        Covariate labels. Defaults to ``X0, X1, ...`` and ``W0, W1, ...``.

    Returns
    -------
    str
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    d_x = X.shape[1]
    d_w = W.shape[1]
    if d_x + d_w == 0:
        raise ValueError("at least one covariate is required for a balance table")
    if x_names is None:
        x_names = [f"X{i}" for i in range(d_x)]
    if w_names is None:
        w_names = [f"W{i}" for i in range(d_w)]
    names = list(x_names) + list(w_names)
    if len(names) != d_x + d_w:
        raise ValueError("number of names does not match number of covariate columns")

    covariates = np.column_stack([X, W])
    smd = standardized_mean_differences(covariates, treatment)
    if weights is None:
        lines = ["| Covariate | SMD |", "|---|---|"]
        lines += [
            f"| {name} | {_fmt(value)} |" for name, value in zip(names, smd)
        ]
        return "\n".join(lines)

    wsmd = weighted_standardized_mean_differences(covariates, treatment, weights)
    lines = ["| Covariate | SMD | Weighted SMD |", "|---|---|---|"]
    lines += [
        f"| {name} | {_fmt(left)} | {_fmt(right)} |"
        for name, left, right in zip(names, smd, wsmd)
    ]
    return "\n".join(lines)


def estimates_table(rows) -> str:
    """Render a markdown table of point estimates.

    Parameters
    ----------
    rows : iterable of (name, estimate) or (name, estimate, se)

    Returns
    -------
    str
    """
    lines = ["| Estimator | Estimate | SE |", "|---|---|---|"]
    for row in rows:
        if len(row) == 2:
            name, estimate = row
            se = None
        else:
            name, estimate, se = row
        lines.append(f"| {name} | {_fmt(estimate)} | {_fmt(se)} |")
    return "\n".join(lines)


def evaluation_table(results) -> str:
    """Render a bias/RMSE summary table from ``EvaluationResult`` objects.

    Parameters
    ----------
    results : iterable of EvaluationResult

    Returns
    -------
    str
    """
    lines = ["| Estimator | Estimate | SE | Bias | RMSE |", "|---|---|---|---|---|"]
    for result in results:
        lines.append(
            f"| {result.name} | {_fmt(result.estimate)} | {_fmt(result.se)} | "
            f"{_fmt(result.bias)} | {_fmt(result.rmse)} |"
        )
    return "\n".join(lines)


def assumption_caveats() -> str:
    """Standard identifiability caveats as a markdown block."""
    return (
        "### Assumption caveats\n\n"
        "- **Ignorability (unconfoundedness):** treatment assignment must be "
        "independent of the potential outcomes conditional on the observed "
        "covariates. The generator's ``selection_bias`` knob deliberately "
        "violates this, and no adjustment for observed covariates can fix it.\n"
        "- **Overlap (positivity):** every unit must have a non-zero "
        "probability of receiving either treatment. Thin overlap inflates "
        "IPW weights and makes nearest-neighbor matching unreliable.\n"
        "- **SUTVA:** no interference between units and no hidden versions "
        "of the treatment.\n"
        "- **Correct propensity model:** IPW and matching inherit the error "
        "of the estimated propensity score; a misspecified model leaves "
        "residual confounding.\n"
        "- **Double robustness (AIPW):** the augmented IPW estimator remains "
        "consistent if *either* the propensity model *or* the outcome "
        "regressions are correctly specified. Both can be wrong at once, "
        "and then AIPW is biased like any other observational estimator.\n"
        "- **Difference-in-differences:** requires parallel trends between "
        "groups and no anticipation of the treatment in the pre period.\n"
        "- **Synthetic control:** the treated unit's pre-treatment path must "
        "lie in the convex hull of the donor paths, donors must remain "
        "untreated, and there must be no anticipation. In-space placebo "
        "ranks are a diagnostic, not a conventional sampling p-value.\n"
        "- **Regression discontinuity (sharp):** the running variable must "
        "not be manipulated at the cutoff, the conditional means of the "
        "potential outcomes must be continuous at the cutoff, and treatment "
        "must switch deterministically there. The estimate is the jump at "
        "the threshold, not an ATE for the whole sample. Bandwidth choice "
        "trades bias against variance.\n"
    )


def render_report(
    X,
    W,
    treatment,
    outcome,
    results,
    true_ate: float,
    weights=None,
    extra_markdown: str = "",
) -> str:
    """Compose a full markdown report.

    The report contains a data summary, the covariate balance table, point
    estimates, the bias/RMSE evaluation and the assumption caveats. Extra
    markdown (for example a difference-in-differences section) is inserted
    before the caveats.

    Parameters
    ----------
    X, W, treatment, outcome : array-like
    results : iterable of EvaluationResult
    true_ate : float
    weights : array-like of shape (n,), optional
    extra_markdown : str, optional

    Returns
    -------
    str
    """
    treatment = np.asarray(treatment).ravel()
    n = treatment.shape[0]
    treated_share = float(treatment.mean())
    parts = [
        "# Causal inference report",
        "",
        f"Observational data with {n} units, {treated_share:.1%} treated.",
        f"Ground-truth ATE: {_fmt(true_ate)}",
        "",
        "## Covariate balance",
        "",
        balance_table(X, W, treatment, weights=weights),
        "",
        "## Point estimates",
        "",
        estimates_table((r.name, r.estimate, r.se) for r in results),
        "",
        "## Bias and RMSE vs ground truth",
        "",
        evaluation_table(results),
        "",
    ]
    if extra_markdown:
        parts.append(extra_markdown.strip())
        parts.append("")
    parts.append(assumption_caveats())
    return "\n".join(parts)
