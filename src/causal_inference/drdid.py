"""Doubly robust difference-in-differences for two-period panels.

Implements the Sant'Anna & Zhao (2020) doubly robust DiD estimator of the
ATT under *conditional* parallel trends, together with its two building
blocks:

* outcome-regression DiD (Heckman, Ichimura & Todd 1997): model the
  control-group trend ``E[dY | X, D=0]`` by OLS and average
  ``dY - m0(X)`` over the treated;
* normalized IPW DiD (Abadie 2005): reweight control trends by
  ``p(X) / (1 - p(X))``.

The doubly robust estimator combines both and is consistent if *either*
the propensity score model or the outcome-trend model is correctly
specified. Standard errors come from the estimator's influence function,
including the estimation effect of the logistic propensity model and the
OLS outcome model (the "traditional" ``drdid_panel`` estimator).

Reference: Sant'Anna, P. H. C. and Zhao, J. (2020). Doubly robust
difference-in-differences estimators. *Journal of Econometrics*, 219(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from statistics import NormalDist

import numpy as np

__all__ = [
    "DRDiDResult",
    "dr_did",
    "ipw_did",
    "or_did",
    "simulate_conditional_did_data",
]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


@dataclass
class DRDiDResult:
    """Doubly robust DiD estimate of the ATT.

    Attributes
    ----------
    att : float
        Doubly robust ATT estimate.
    se : float
        Influence-function standard error.
    ci_low, ci_high : float
        Two-sided normal confidence interval at level ``1 - alpha``.
    p_value : float
        Two-sided p-value for ``H0: ATT = 0``.
    or_att : float
        Outcome-regression DiD point estimate (for comparison).
    ipw_att : float
        Normalized IPW DiD point estimate (for comparison).
    naive_att : float
        Unconditional DiD (difference in mean trends).
    n_treated, n_control : int
        Group sizes.
    n_trimmed : int
        Control units dropped because ``p(X) > 1 - trim``.
    propensity : ndarray of shape (n,)
        Fitted propensity scores.
    influence : ndarray of shape (n,)
        Influence function values; ``se = std(influence) / sqrt(n)``.
    alpha : float
    """

    att: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    or_att: float
    ipw_att: float
    naive_att: float
    n_treated: int
    n_control: int
    n_trimmed: int
    propensity: np.ndarray
    influence: np.ndarray
    alpha: float = 0.05

    def summary(self) -> str:
        level = int(round(100 * (1 - self.alpha)))
        return (
            f"DR-DiD ATT = {self.att:.4f} (SE {self.se:.4f}, "
            f"{level}% CI [{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"p = {self.p_value:.4g}); OR = {self.or_att:.4f}, "
            f"IPW = {self.ipw_att:.4f}, naive = {self.naive_att:.4f}; "
            f"n_treated = {self.n_treated}, n_control = {self.n_control}"
        )


def _prepare(y_pre, y_post, treatment, X):
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    y_post = np.asarray(y_post, dtype=float).ravel()
    d = np.asarray(treatment).ravel()
    if not (y_pre.shape[0] == y_post.shape[0] == d.shape[0]):
        raise ValueError("y_pre, y_post and treatment must have the same length")
    n = d.shape[0]
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("treatment must contain only 0 and 1")
    d = d.astype(float)
    if X is None:
        X = np.empty((n, 0))
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != n:
        raise ValueError("X must have one row per unit")
    dy = y_post - y_pre
    if not (np.all(np.isfinite(dy)) and np.all(np.isfinite(X))):
        raise ValueError("inputs must be finite")
    if d.sum() < 1 or (1 - d).sum() < 1:
        raise ValueError("need at least one treated and one control unit")
    design = np.column_stack([np.ones(n), X])
    return dy, d, design


def _logit_newton(design, d, ridge=1e-8, max_iter=100, tol=1e-10):
    """Unpenalized logistic MLE by Newton-Raphson (tiny ridge for stability)."""
    k = design.shape[1]
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = _expit(design @ beta)
        w = p * (1 - p)
        grad = design.T @ (d - p)
        hess = (design * w[:, None]).T @ design + ridge * np.eye(k)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta, _expit(design @ beta)


def _control_ols(design, dy, d):
    ctrl = d == 0
    Xc = design[ctrl]
    if np.linalg.matrix_rank(Xc) < design.shape[1]:
        raise ValueError("covariates are collinear among control units")
    coef, *_ = np.linalg.lstsq(Xc, dy[ctrl], rcond=None)
    return coef


def or_did(y_pre, y_post, treatment, X=None) -> float:
    """Outcome-regression DiD ATT: ``mean_{D=1}(dY - m0(X))``.

    ``m0`` is an OLS fit of the trend ``dY = Y_post - Y_pre`` on ``X``
    among controls.
    """
    dy, d, design = _prepare(y_pre, y_post, treatment, X)
    coef = _control_ols(design, dy, d)
    t = d == 1
    return float(np.mean(dy[t] - design[t] @ coef))


def ipw_did(y_pre, y_post, treatment, X=None, trim: float = 0.005) -> float:
    """Normalized (Hajek) IPW DiD ATT of Abadie (2005).

    Controls are weighted by ``p(X) / (1 - p(X))`` from a logistic model;
    controls with ``p(X) > 1 - trim`` are dropped.
    """
    dy, d, design = _prepare(y_pre, y_post, treatment, X)
    _, ps = _logit_newton(design, d)
    keep = (d == 1) | (ps <= 1 - trim)
    ps = np.minimum(ps, 1 - 1e-12)
    w_c = keep * (1 - d) * ps / (1 - ps)
    if w_c.sum() <= 0:
        raise ValueError("no control units left after trimming")
    return float(np.sum(d * dy) / d.sum() - np.sum(w_c * dy) / w_c.sum())


def dr_did(
    y_pre,
    y_post,
    treatment,
    X=None,
    alpha: float = 0.05,
    trim: float = 0.005,
) -> DRDiDResult:
    """Sant'Anna & Zhao (2020) doubly robust DiD ATT for a two-period panel.

    Parameters
    ----------
    y_pre, y_post : array-like of shape (n,)
        Outcome of each unit before and after treatment.
    treatment : array-like of shape (n,)
        1 for units treated in the post period, 0 for never-treated.
    X : array-like of shape (n, p), optional
        Pre-treatment covariates for conditional parallel trends. An
        intercept is added. Without covariates the estimator reduces to
        the plain two-period DiD.
    alpha : float
        Significance level for the confidence interval.
    trim : float
        Control units with propensity above ``1 - trim`` get zero weight
        (as in the DRDID R package, default 0.995 cut-off).

    Returns
    -------
    DRDiDResult
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    dy, d, design = _prepare(y_pre, y_post, treatment, X)
    n = d.shape[0]

    _, ps = _logit_newton(design, d)
    keep = (d == 1) | (ps <= 1 - trim)
    ps_c = np.minimum(ps, 1 - 1e-12)
    coef = _control_ols(design, dy, d)
    resid = dy - design @ coef

    w_t = d
    w_c = keep * (1 - d) * ps_c / (1 - ps_c)
    if w_c.sum() <= 0:
        raise ValueError("no control units left after trimming")
    eta_t = np.mean(w_t * resid) / np.mean(w_t)
    eta_c = np.mean(w_c * resid) / np.mean(w_c)
    att = float(eta_t - eta_c)

    # Influence function (Sant'Anna & Zhao 2020, panel DR-DiD).
    w_ols = (1 - d)
    xpx = (design * w_ols[:, None]).T @ design / n
    lin_ols = (w_ols * resid)[:, None] * design @ np.linalg.pinv(xpx)
    pw = ps * (1 - ps)
    hess_ps = np.linalg.pinv((design * pw[:, None]).T @ design / n)
    lin_ps = (d - ps)[:, None] * design @ hess_ps

    inf_t1 = w_t * resid - w_t * eta_t
    m1 = np.mean(w_t[:, None] * design, axis=0)
    inf_t = (inf_t1 - lin_ols @ m1) / np.mean(w_t)

    inf_c1 = w_c * resid - w_c * eta_c
    m2 = np.mean((w_c * (resid - eta_c))[:, None] * design, axis=0)
    m3 = np.mean(w_c[:, None] * design, axis=0)
    inf_c = (inf_c1 + lin_ps @ m2 - lin_ols @ m3) / np.mean(w_c)

    influence = inf_t - inf_c
    se = float(np.std(influence, ddof=1) / sqrt(n))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(att) / se)) if se > 0 else float("nan")

    t = d == 1
    or_att = float(np.mean(resid[t]))
    ipw_att = float(np.sum(d * dy) / d.sum() - np.sum(w_c * dy) / w_c.sum())
    naive = float(dy[t].mean() - dy[~t].mean())
    return DRDiDResult(
        att=att,
        se=se,
        ci_low=att - z * se,
        ci_high=att + z * se,
        p_value=float(p_value),
        or_att=or_att,
        ipw_att=ipw_att,
        naive_att=naive,
        n_treated=int(t.sum()),
        n_control=int((~t).sum()),
        n_trimmed=int(((d == 0) & ~keep).sum()),
        propensity=ps,
        influence=influence,
        alpha=alpha,
    )


def simulate_conditional_did_data(
    n: int = 1000,
    att: float = 1.0,
    confounding: float = 1.0,
    nonlinear_trend: bool = False,
    nonlinear_propensity: bool = False,
    seed: int = 0,
):
    """Simulate a two-period panel where trends depend on covariates.

    Treatment uptake and the untreated trend both depend on ``X``, so the
    unconditional DiD is biased while conditional parallel trends hold.

    Parameters
    ----------
    nonlinear_trend : bool
        Make the control trend nonlinear in ``X`` (misspecifies a linear
        outcome-regression model).
    nonlinear_propensity : bool
        Make the true propensity nonlinear in ``X`` (misspecifies a linear
        logit).

    Returns
    -------
    dict with keys ``y_pre``, ``y_post``, ``treatment``, ``X`` and ``att``.
    """
    if n < 4:
        raise ValueError("n must be at least 4")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    index = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    if nonlinear_propensity:
        index = index + 0.8 * (X[:, 0] ** 2 - 1) - 0.6 * X[:, 0] * X[:, 1]
    d = (rng.uniform(size=n) < _expit(index)).astype(int)
    level = 1.0 + X @ np.array([1.0, 0.5]) + 0.5 * d + rng.normal(size=n)
    trend = 1.0 + confounding * (X[:, 0] - 0.5 * X[:, 1])
    if nonlinear_trend:
        trend = trend + confounding * (X[:, 0] ** 2 + np.sin(2 * X[:, 1]))
    y_pre = level + rng.normal(size=n)
    y_post = level + trend + att * d + rng.normal(size=n)
    return {"y_pre": y_pre, "y_post": y_post, "treatment": d, "X": X, "att": att}
