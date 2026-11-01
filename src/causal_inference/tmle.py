"""Targeted maximum likelihood estimation (TMLE) of the average treatment effect.

TMLE (van der Laan & Rubin, 2006) is a doubly robust, semiparametric
efficient estimator, as AIPW is. It differs from AIPW in *how* it removes
the plug-in bias. AIPW adds the mean residual correction to the
g-computation estimate. TMLE instead **updates the initial outcome model**
along a one-dimensional logistic submodel

    logit Q*(A, X) = logit Q(A, X) + eps * H(A, X),
    H(A, X) = A / g(X) - (1 - A) / (1 - g(X)),

where ``g`` is the propensity score. ``eps`` is fitted by maximum
likelihood, with ``logit Q`` as an offset. The resulting score equation
``sum H (Y - Q*(A, X)) = 0`` is exactly the efficient-influence-function
estimating equation, so the substitution estimator

    psi = mean(Q*(1, X) - Q*(0, X))

is a plug-in estimate that also solves the EIF equation. Because it is a
substitution estimator it always respects the bounds of the outcome space,
which AIPW does not guarantee (for example, a risk difference outside
[-1, 1]). This makes TMLE noticeably more stable under near-positivity
violations.

Continuous outcomes are mapped to [0, 1] with known or observed bounds
``(a, b)``, as in Gruber & van der Laan (2010), and the fluctuation uses
the quasi-binomial logistic loss. Inference uses the estimated influence
curve,

    IC = H (Y - Q*(A, X)) + Q*(1, X) - Q*(0, X) - psi,

with ``se = sd(IC) / sqrt(n)``. For binary outcomes the risk ratio and
odds ratio are also reported, with delta-method intervals on the log scale.

With ``n_folds >= 2`` the nuisances are cross-fitted (pooled CV-TMLE). Each
unit's ``Q`` and ``g`` then come from models fitted on the other folds, so
flexible learners can be used without overfitting the influence curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from statistics import NormalDist
from typing import Optional

import numpy as np

from .dml import LogisticClassifier, PolynomialRidge, _clone, _make_folds, _prepare

__all__ = ["TMLEResult", "tmle_ate"]

_QCLIP = 1e-5


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _logit(p):
    return np.log(p) - np.log1p(-p)


@dataclass
class TMLEResult:
    """TMLE estimate of the ATE with influence-curve inference."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    z: float
    p_value: float
    psi1: float
    psi0: float
    epsilon: float
    initial_estimate: float
    aipw_estimate: float
    outcome_type: str
    bounds: tuple
    n: int
    n_folds: int
    n_truncated: int
    risk_ratio: Optional[float] = None
    rr_ci: Optional[tuple] = None
    odds_ratio: Optional[float] = None
    or_ci: Optional[tuple] = None
    influence: np.ndarray = field(default=None, repr=False)
    q1: np.ndarray = field(default=None, repr=False)
    q0: np.ndarray = field(default=None, repr=False)
    propensity: np.ndarray = field(default=None, repr=False)

    def summary(self) -> str:
        s = (
            f"TMLE ATE: {self.estimate:.4f} (SE {self.se:.4f}, "
            f"CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}], p={self.p_value:.3g}); "
            f"E[Y(1)]={self.psi1:.4f}, E[Y(0)]={self.psi0:.4f}; "
            f"initial g-comp {self.initial_estimate:.4f}, eps={self.epsilon:.4g}"
        )
        if self.risk_ratio is not None:
            s += (
                f"; RR {self.risk_ratio:.4f} [{self.rr_ci[0]:.4f}, {self.rr_ci[1]:.4f}]"
                f", OR {self.odds_ratio:.4f} [{self.or_ci[0]:.4f}, {self.or_ci[1]:.4f}]"
            )
        return s


def _predict_outcome(model, X, binary: bool):
    if binary and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    return np.asarray(model.predict(X), dtype=float).ravel()


def _fit_nuisances(X, a, y, ml_q, ml_g, binary, train, test, propensity):
    t1 = train[a[train] == 1]
    t0 = train[a[train] == 0]
    if t1.size == 0 or t0.size == 0:
        raise ValueError("every training split must contain treated and control units")
    if binary and len(np.unique(y[t1])) < 2:
        m1 = None
    else:
        m1 = _clone(ml_q).fit(X[t1], y[t1])
    if binary and len(np.unique(y[t0])) < 2:
        m0 = None
    else:
        m0 = _clone(ml_q).fit(X[t0], y[t0])
    q1 = np.full(test.size, y[t1].mean()) if m1 is None else _predict_outcome(m1, X[test], binary)
    q0 = np.full(test.size, y[t0].mean()) if m0 is None else _predict_outcome(m0, X[test], binary)
    if propensity is not None:
        g = propensity[test]
    else:
        mg = _clone(ml_g).fit(X[train], a[train])
        if hasattr(mg, "predict_proba"):
            g = np.asarray(mg.predict_proba(X[test]), dtype=float)[:, 1]
        else:
            g = np.asarray(mg.predict(X[test]), dtype=float).ravel()
    return q1, q0, g


def _fluctuate(ys, offset, h, max_iter: int = 100, tol: float = 1e-10) -> float:
    """MLE of ``eps`` in the logistic submodel ``expit(offset + eps h)`` (Newton)."""
    eps = 0.0
    for _ in range(max_iter):
        p = _expit(offset + eps * h)
        score = np.sum(h * (ys - p))
        info = np.sum(h * h * p * (1.0 - p))
        if info <= 0:
            break
        step = score / info
        # damp very large steps for stability; the loss is convex in eps
        if abs(step) > 5.0:
            step = np.sign(step) * 5.0
        eps += step
        if abs(step) < tol:
            break
    return float(eps)


def tmle_ate(
    X,
    treatment,
    outcome,
    ml_q=None,
    ml_g=None,
    propensity=None,
    outcome_type: str = "auto",
    bounds=None,
    truncation: float = 0.01,
    n_folds: int = 1,
    random_state: int = 0,
    alpha: float = 0.05,
) -> TMLEResult:
    """Targeted maximum likelihood estimate of the ATE.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Confounders.
    treatment : array-like of shape (n,)
        Binary treatment in {0, 1}.
    outcome : array-like of shape (n,)
        Continuous or binary outcome.
    ml_q : estimator, optional
        Outcome learner with ``fit``/``predict`` (or ``predict_proba`` for
        binary outcomes), fitted separately in each arm. The default is
        :class:`PolynomialRidge` for continuous outcomes and
        :class:`LogisticClassifier` for binary ones.
    ml_g : estimator, optional
        Propensity classifier with ``predict_proba`` (default
        :class:`LogisticClassifier`). Ignored when ``propensity`` is given.
    propensity : array-like of shape (n,), optional
        Known or pre-computed propensity scores, for example from a
        randomized design.
    outcome_type : {"auto", "continuous", "binary"}
        ``"auto"`` treats a 0/1-valued outcome as binary.
    bounds : (a, b), optional
        Bounds of a continuous outcome used for the [0, 1] scaling. They
        default to the observed range. Supply the known support when it is
        wider.
    truncation : float
        Propensity scores are truncated to ``[truncation, 1 - truncation]``.
    n_folds : int
        ``1`` fits the nuisances on the full sample. ``>= 2`` cross-fits them.
    random_state : int
        Seed for the fold assignment.
    alpha : float
        Confidence intervals have level ``1 - alpha``.

    Returns
    -------
    TMLEResult
    """
    X, a, y = _prepare(X, treatment, outcome)
    n = y.size
    if not np.all(np.isin(a, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1")
    if not (np.any(a == 1) and np.any(a == 0)):
        raise ValueError("both treatment groups must be present")
    if not 0.0 <= truncation < 0.5:
        raise ValueError("truncation must lie in [0, 0.5)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if outcome_type not in ("auto", "continuous", "binary"):
        raise ValueError("outcome_type must be 'auto', 'continuous' or 'binary'")
    is01 = bool(np.all(np.isin(y, (0.0, 1.0))))
    if outcome_type == "auto":
        outcome_type = "binary" if is01 else "continuous"
    binary = outcome_type == "binary"
    if binary and not is01:
        raise ValueError("binary outcome must be 0/1")

    if binary:
        lo, hi = 0.0, 1.0
    elif bounds is None:
        lo, hi = float(y.min()), float(y.max())
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
        if y.min() < lo or y.max() > hi:
            raise ValueError("outcome lies outside the supplied bounds")
    if not hi > lo:
        raise ValueError("outcome bounds must satisfy a < b (constant outcome?)")

    if ml_q is None:
        ml_q = LogisticClassifier() if binary else PolynomialRidge(degree=2, alpha=1.0)
    if ml_g is None:
        ml_g = LogisticClassifier()
    if propensity is not None:
        propensity = np.asarray(propensity, dtype=float).ravel()
        if propensity.size != n or not np.all((propensity > 0) & (propensity < 1)):
            raise ValueError("propensity must have n entries strictly inside (0, 1)")

    q1 = np.empty(n)
    q0 = np.empty(n)
    g = np.empty(n)
    if n_folds == 1:
        everything = np.arange(n)
        splits = [(everything, everything)]
    else:
        splits = _make_folds(n, n_folds, np.random.default_rng(random_state), strata=a)
    for train, test in splits:
        q1[test], q0[test], g[test] = _fit_nuisances(X, a, y, ml_q, ml_g, binary, train, test, propensity)

    n_truncated = int(np.sum((g < truncation) | (g > 1.0 - truncation)))
    g = np.clip(g, max(truncation, 1e-12), 1.0 - max(truncation, 1e-12))

    span = hi - lo
    ys = (y - lo) / span
    q1s = np.clip((q1 - lo) / span, _QCLIP, 1.0 - _QCLIP)
    q0s = np.clip((q0 - lo) / span, _QCLIP, 1.0 - _QCLIP)
    qas = np.where(a == 1, q1s, q0s)

    h1 = 1.0 / g
    h0 = -1.0 / (1.0 - g)
    h = np.where(a == 1, h1, h0)

    initial = span * float(np.mean(q1s - q0s))
    aipw = span * float(np.mean(q1s - q0s + h * (ys - qas)))

    eps = _fluctuate(ys, _logit(qas), h)
    q1u = _expit(_logit(q1s) + eps * h1)
    q0u = _expit(_logit(q0s) + eps * h0)
    qau = np.where(a == 1, q1u, q0u)

    psi1s = float(np.mean(q1u))
    psi0s = float(np.mean(q0u))
    psi = psi1s - psi0s
    resid = ys - qau
    ic1 = (a == 1) * h1 * resid + q1u - psi1s
    ic0 = (a == 0) * (-h0) * resid + q0u - psi0s  # -h0 = 1 / (1 - g)
    ic = ic1 - ic0

    est = span * psi
    se = span * float(np.std(ic, ddof=1) / np.sqrt(n))
    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z = est / se if se > 0 else float("inf") * np.sign(est)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z))) if np.isfinite(z) else 0.0

    rr = rr_ci = odds = or_ci = None
    if binary:
        def _log_ci(log_est, ic_log):
            s = float(np.std(ic_log, ddof=1) / np.sqrt(n))
            return float(np.exp(log_est)), (float(np.exp(log_est - zcrit * s)), float(np.exp(log_est + zcrit * s)))

        rr, rr_ci = _log_ci(np.log(psi1s / psi0s), ic1 / psi1s - ic0 / psi0s)
        log_or = np.log(psi1s / (1 - psi1s)) - np.log(psi0s / (1 - psi0s))
        odds, or_ci = _log_ci(log_or, ic1 / (psi1s * (1 - psi1s)) - ic0 / (psi0s * (1 - psi0s)))

    return TMLEResult(
        estimate=float(est),
        se=se,
        ci_lower=float(est - zcrit * se),
        ci_upper=float(est + zcrit * se),
        z=float(z),
        p_value=float(p_value),
        psi1=lo + span * psi1s,
        psi0=lo + span * psi0s,
        epsilon=eps,
        initial_estimate=initial,
        aipw_estimate=aipw,
        outcome_type=outcome_type,
        bounds=(lo, hi),
        n=n,
        n_folds=n_folds,
        n_truncated=n_truncated,
        risk_ratio=rr,
        rr_ci=rr_ci,
        odds_ratio=odds,
        or_ci=or_ci,
        influence=span * ic,
        q1=lo + span * q1u,
        q0=lo + span * q0u,
        propensity=g,
    )
