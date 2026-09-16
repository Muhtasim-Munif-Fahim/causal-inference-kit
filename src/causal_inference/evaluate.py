"""Evaluation of estimators against a known ground-truth effect."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import aipw_ate, difference_in_means, ipw_ate, ipw_att, propensity_matching
from .propensity import propensity_scores


@dataclass
class EvaluationResult:
    """Summary for one estimator on one dataset.

    Attributes
    ----------
    name : str
    estimate : float
        Point estimate on the original data.
    se : float
        Standard error estimated from the bootstrap distribution.
    bias : float
        Mean bootstrap estimate minus the true effect.
    rmse : float
        Root mean squared error of the bootstrap distribution around the
        true effect.
    bootstrap_failures : int
        Number of bootstrap draws where the estimator raised.
    """

    name: str
    estimate: float
    se: float
    bias: float
    rmse: float
    bootstrap_failures: int = 0


def standard_estimators() -> dict:
    """A dict of ready-made estimators in the ``evaluate`` convention.

    Each callable has the signature ``fn(X, W, treatment, outcome,
    propensity)`` where ``propensity`` is the fixed first-stage propensity
    score from the full sample. Included are the naive difference in means,
    stabilized IPW and doubly robust AIPW for the ATE, IPW for the ATT, and
    nearest-neighbor matching with replacement for the ATT. AIPW's outcome
    regressions use both ``X`` and ``W``.
    """
    return {
        "naive": lambda X, W, t, y, p: difference_in_means(t, y),
        "ipw": lambda X, W, t, y, p: ipw_ate(X, t, y, propensity=p),
        "aipw": lambda X, W, t, y, p: aipw_ate(X, t, y, propensity=p, W=W),
        "ipw_att": lambda X, W, t, y, p: ipw_att(X, t, y, propensity=p),
        "matching": lambda X, W, t, y, p: propensity_matching(
            X, t, y, propensity=p, with_replacement=True
        ),
    }


def _resolve_estimators(estimators):
    if isinstance(estimators, dict):
        return list(estimators.items())
    resolved = []
    for item in estimators:
        if isinstance(item, (tuple, list)) and len(item) == 2 and callable(item[1]):
            resolved.append((item[0], item[1]))
        else:
            raise TypeError("estimators must be a dict or a list of (name, callable) pairs")
    return resolved


def evaluate(
    estimators,
    X,
    W,
    treatment,
    outcome,
    true_ate: float,
    n_boot: int = 200,
    seed: int = 0,
) -> list[EvaluationResult]:
    """Score estimators against a known ground-truth effect.

    Each estimator is a callable ``fn(X, W, treatment, outcome, propensity)
    -> float``. The propensity score is estimated once on the full sample
    (fixed first stage) and its values follow the rows into the bootstrap
    resamples. Outcome regressions used by AIPW are refit on each resample.
    The resampling distribution yields the standard error, the bias and the
    RMSE around the true effect. Bootstrap draws where the estimator raises
    are skipped and counted; if more than half of the draws fail, an error
    is raised.

    Parameters
    ----------
    estimators : dict or list of (name, callable)
        Keys or names label the estimators in the output.
    X, W, treatment, outcome : array-like
    true_ate : float
        Ground-truth average treatment effect.
    n_boot : int
        Number of bootstrap resamples.
    seed : int

    Returns
    -------
    list[EvaluationResult]
        One entry per estimator, in the order given.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    treatment = np.asarray(treatment).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    n = X.shape[0]
    if not (W.shape[0] == treatment.shape[0] == outcome.shape[0] == n):
        raise ValueError("X, W, treatment and outcome must have the same number of rows")
    if n == 0:
        raise ValueError("at least one row of data is required")
    if n_boot < 1:
        raise ValueError("n_boot must be a positive integer")

    resolved = _resolve_estimators(estimators)
    propensity = propensity_scores(X, treatment)[0]
    rng = np.random.default_rng(seed)
    results = []
    for name, fn in resolved:
        estimate = fn(X, W, treatment, outcome, propensity)
        boot = []
        failures = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                boot.append(fn(X[idx], W[idx], treatment[idx], outcome[idx], propensity[idx]))
            except ValueError:
                failures += 1
        if failures > n_boot // 2:
            raise ValueError(
                f"estimator {name!r} failed on more than half of the bootstrap draws"
            )
        boot = np.asarray(boot, dtype=float)
        se = float(boot.std(ddof=1)) if boot.size > 1 else 0.0
        bias = float(boot.mean() - true_ate)
        rmse = float(np.sqrt(np.mean((boot - true_ate) ** 2)))
        results.append(
            EvaluationResult(
                name=name,
                estimate=float(estimate),
                se=se,
                bias=bias,
                rmse=rmse,
                bootstrap_failures=failures,
            )
        )
    return results
