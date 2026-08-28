"""Propensity score estimation and overlap diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticFit:
    """Result of fitting a logistic regression by gradient descent.

    Attributes
    ----------
    coef : ndarray of shape (d,)
        Coefficients on the original covariate scale.
    intercept : float
        Intercept term.
    converged : bool
        True when the gradient tolerance was reached.
    n_iter : int
        Number of gradient steps taken.
    """

    coef: np.ndarray
    intercept: float
    converged: bool
    n_iter: int


@dataclass
class PropensityHistogram:
    """Propensity score bin counts for treated and control groups."""

    edges: np.ndarray
    treated: np.ndarray
    control: np.ndarray


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _validate_treatment(treatment: np.ndarray) -> np.ndarray:
    treatment = np.asarray(treatment).ravel()
    if not np.all((treatment == 0) | (treatment == 1)):
        raise ValueError("treatment must contain only 0 and 1")
    return treatment


def logistic_regression(
    X,
    y,
    l2: float = 1e-4,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> LogisticFit:
    """Fit a logistic regression by gradient descent with backtracking.

    Covariates are standardized internally for numerical stability and the
    coefficients are mapped back to the original scale. The L2 penalty is
    applied to the covariate coefficients only, so a positive ``l2`` keeps
    the solution bounded even under perfect separation.

    Parameters
    ----------
    X : array-like of shape (n, d)
    y : array-like of shape (n,)
        Binary target in {0, 1}.
    l2 : float
        L2 regularization strength.
    max_iter : int
        Maximum number of gradient steps.
    tol : float
        Convergence tolerance on the infinity norm of the gradient.

    Returns
    -------
    LogisticFit
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2d array")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1")

    max_iter = max(max_iter, 1)
    n, d = X.shape
    Xs, mu, sd = _standardize(X)
    design = np.column_stack([np.ones(n), Xs])

    beta = np.zeros(d + 1)
    penalty = np.zeros(d + 1)
    penalty[1:] = l2
    lr = 1.0
    converged = False

    for step in range(1, max_iter + 1):
        z = design @ beta
        p = _sigmoid(z)
        grad = design.T @ (p - y) / n + penalty * beta
        if np.max(np.abs(grad)) < tol:
            converged = True
            break

        loss = _log_loss(y, p) + 0.5 * l2 * (beta[1:] @ beta[1:])
        direction = grad @ grad
        step_size = lr
        while step_size > 1e-12:
            candidate = beta - step_size * grad
            p_c = _sigmoid(design @ candidate)
            loss_c = _log_loss(y, p_c) + 0.5 * l2 * (candidate[1:] @ candidate[1:])
            if loss_c <= loss - 1e-4 * step_size * direction:
                break
            step_size *= 0.5
        beta = beta - step_size * grad
        lr = min(2.0, 2.0 * step_size)

    coef = beta[1:] / sd
    intercept = float(beta[0] - mu @ coef)
    return LogisticFit(
        coef=coef,
        intercept=intercept,
        converged=converged,
        n_iter=step if "step" in locals() else 0,
    )


def propensity_scores(X, treatment, l2: float = 1e-4, max_iter: int = 2000, tol: float = 1e-6):
    """Estimate propensity scores p(T=1 | X) with a logistic model.

    Returns
    -------
    p : ndarray of shape (n,)
        Estimated propensity scores.
    fit : LogisticFit
        The fitted model.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    treatment = _validate_treatment(treatment)
    fit = logistic_regression(X, treatment, l2=l2, max_iter=max_iter, tol=tol)
    z = fit.intercept + X @ fit.coef
    p = _sigmoid(z)
    return np.clip(p, 1e-12, 1.0 - 1e-12), fit


def _as_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2d array")
    return X


def _require_both_groups(t1: np.ndarray, t0: np.ndarray) -> None:
    if not (np.any(t1) and np.any(t0)):
        raise ValueError("both treatment groups must be present")


def standardized_mean_differences(X, treatment) -> np.ndarray:
    """Standardized mean difference per covariate column.

    ``SMD_j = (mean_1j - mean_0j) / sqrt((var_1j + var_0j) / 2)``.

    Columns with zero pooled variance get an SMD of 0.

    Parameters
    ----------
    X : array-like of shape (n, d)
    treatment : array-like of shape (n,)

    Returns
    -------
    smd : ndarray of shape (d,)
    """
    X = _as_matrix(X)
    treatment = _validate_treatment(treatment)
    t1 = treatment == 1
    t0 = ~t1
    _require_both_groups(t1, t0)
    m1 = X[t1].mean(axis=0)
    m0 = X[t0].mean(axis=0)
    v1 = X[t1].var(axis=0)
    v0 = X[t0].var(axis=0)
    pooled = np.sqrt((v1 + v0) / 2.0)
    smd = np.zeros(X.shape[1])
    ok = pooled >= 1e-12
    smd[ok] = (m1[ok] - m0[ok]) / pooled[ok]
    return smd


def weighted_standardized_mean_differences(X, treatment, weights) -> np.ndarray:
    """Weighted standardized mean difference per covariate column.

    Weights are per-unit (for example IPW or matching weights) and must be
    non-negative. Columns with zero pooled weighted variance get an SMD of 0.

    Parameters
    ----------
    X : array-like of shape (n, d)
    treatment : array-like of shape (n,)
    weights : array-like of shape (n,)

    Returns
    -------
    smd : ndarray of shape (d,)
    """
    X = _as_matrix(X)
    treatment = _validate_treatment(treatment)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.shape[0] != X.shape[0]:
        raise ValueError("weights must have one entry per row of X")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    t1 = treatment == 1
    t0 = ~t1
    _require_both_groups(t1, t0)
    smd = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]

        def wstats(rows):
            w = weights[rows]
            total = w.sum()
            if total <= 0:
                return 0.0, 0.0
            mean = np.sum(w * col[rows]) / total
            var = np.sum(w * (col[rows] - mean) ** 2) / total
            return mean, var

        m1, v1 = wstats(t1)
        m0, v0 = wstats(t0)
        pooled = np.sqrt((v1 + v0) / 2.0)
        if pooled >= 1e-12:
            smd[j] = (m1 - m0) / pooled
    return smd


def propensity_histogram(propensity, treatment, bins: int = 10) -> PropensityHistogram:
    """Count propensity scores into fixed-width bins over [0, 1].

    Rows with non-finite propensity are dropped.

    Parameters
    ----------
    propensity : array-like of shape (n,)
    treatment : array-like of shape (n,)
    bins : int
        Number of bins.

    Returns
    -------
    PropensityHistogram
    """
    p = np.asarray(propensity, dtype=float).ravel()
    treatment = _validate_treatment(treatment)
    if p.shape[0] != treatment.shape[0]:
        raise ValueError("propensity and treatment must have the same length")
    if bins < 1:
        raise ValueError("bins must be a positive integer")
    valid = np.isfinite(p)
    treated_counts, edges = np.histogram(p[valid & (treatment == 1)], bins=bins, range=(0.0, 1.0))
    control_counts, _ = np.histogram(p[valid & (treatment == 0)], bins=bins, range=(0.0, 1.0))
    return PropensityHistogram(edges=edges, treated=treated_counts, control=control_counts)
