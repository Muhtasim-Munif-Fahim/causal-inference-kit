"""Quantile treatment effects by inverse propensity weighting (Firpo, 2007).

Average effects can hide distributional changes: a programme that raises
low incomes and lowers high incomes may have an ATE of zero. The quantile
treatment effect at level ``tau`` is the difference between the ``tau``
quantiles of the two *potential-outcome* distributions,

    QTE(tau) = q_1(tau) - q_0(tau).

Under unconfoundedness and overlap, Firpo (Econometrica 2007) identifies
``q_1(tau)`` as the minimiser of the propensity-weighted check loss

    sum_i T_i / e(X_i) * rho_tau(Y_i - q),

i.e. a weighted quantile of treated outcomes with weights ``1 / e(X)``
(and ``1 / (1 - e(X))`` for controls). For the effect on the treated
(QTT) the treated sample is used as is and controls are re-weighted by
``e(X) / (1 - e(X))``. Inference uses a nonparametric bootstrap that
re-estimates the propensity model in each replicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .estimators import _coerce_arrays, _validate_propensity
from .propensity import propensity_scores

__all__ = [
    "QuantileTreatmentEffectResult",
    "ipw_weighted_quantile",
    "quantile_treatment_effects",
]


def ipw_weighted_quantile(values, weights, quantiles) -> np.ndarray:
    """Weighted quantiles minimising the weighted check loss.

    Returns, for each ``tau``, the smallest value whose normalised
    cumulative weight reaches ``tau`` (the ``inverted_cdf`` rule). With
    equal weights this equals ``np.quantile(values, tau, method="inverted_cdf")``.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    if v.size == 0:
        raise ValueError("at least one value is required")
    if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("quantiles must lie strictly between 0 and 1")
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cdf, q - 1e-12, side="left")
    return v[np.minimum(idx, v.size - 1)]


@dataclass
class QuantileTreatmentEffectResult:
    """Quantile treatment effects with optional bootstrap inference."""

    quantiles: np.ndarray
    qte: np.ndarray
    q_treated: np.ndarray
    q_control: np.ndarray
    target: str
    n_bootstrap: int = 0
    se: Optional[np.ndarray] = None
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None
    ci: float = 0.95
    bootstrap_draws: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        rows = []
        for j, tau in enumerate(self.quantiles):
            row = {
                "quantile": float(tau),
                "qte": float(self.qte[j]),
                "q_treated": float(self.q_treated[j]),
                "q_control": float(self.q_control[j]),
            }
            if self.se is not None:
                row.update(
                    se=float(self.se[j]),
                    ci_lower=float(self.ci_lower[j]),
                    ci_upper=float(self.ci_upper[j]),
                )
            rows.append(row)
        return {"target": self.target, "ci": self.ci, "rows": rows}

    def summary(self) -> str:
        label = "QTE" if self.target == "ate" else "QTT"
        lines = [f"{label} by inverse propensity weighting (Firpo 2007)"]
        for row in self.to_dict()["rows"]:
            line = (
                f"  tau={row['quantile']:.2f}  {label}={row['qte']: .4f}"
                f"  q1={row['q_treated']: .4f}  q0={row['q_control']: .4f}"
            )
            if "se" in row:
                line += (
                    f"  se={row['se']:.4f}  {int(round(self.ci * 100))}% CI "
                    f"[{row['ci_lower']: .4f}, {row['ci_upper']: .4f}]"
                )
            lines.append(line)
        return "\n".join(lines)


def _qte_point(y, t, p, quantiles, target):
    treated = t == 1
    if target == "ate":
        w1 = 1.0 / p[treated]
    else:
        w1 = np.ones(int(treated.sum()))
    w0 = 1.0 / (1.0 - p[~treated]) if target == "ate" else p[~treated] / (1.0 - p[~treated])
    q1 = ipw_weighted_quantile(y[treated], w1, quantiles)
    q0 = ipw_weighted_quantile(y[~treated], w0, quantiles)
    return q1, q0


def quantile_treatment_effects(
    X,
    treatment,
    outcome,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    propensity=None,
    target: str = "ate",
    n_bootstrap: int = 0,
    ci: float = 0.95,
    trim: float = 0.0,
    seed: Optional[int] = None,
) -> QuantileTreatmentEffectResult:
    """Estimate quantile treatment effects (QTE) or effects on the treated (QTT).

    Parameters
    ----------
    X, treatment, outcome : array-like
        Covariates, binary treatment and outcome.
    quantiles : sequence of float
        Quantile levels in ``(0, 1)``.
    propensity : array-like, optional
        Known propensity scores. When omitted a logistic model is fitted
        (and refitted in every bootstrap replicate).
    target : {"ate", "att"}
        ``"ate"`` gives the unconditional QTE, ``"att"`` the QTE on the
        treated (QTT).
    n_bootstrap : int
        Bootstrap replicates for standard errors / percentile intervals.
    ci : float
        Interval coverage.
    trim : float
        Clip propensities to ``[trim, 1 - trim]`` (e.g. 0.01) to limit
        extreme weights. ``0`` disables trimming.
    seed : int, optional
        Bootstrap seed.
    """
    if target not in {"ate", "att"}:
        raise ValueError("target must be 'ate' or 'att'")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie strictly between 0 and 1")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must lie in [0, 0.5)")
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")
    taus = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if taus.size == 0 or np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("quantiles must lie strictly between 0 and 1")
    X, t, y = _coerce_arrays(X, treatment, outcome)
    if not ((t == 1).any() and (t == 0).any()):
        raise ValueError("both treatment groups must be present")
    if not np.all(np.isfinite(y)):
        raise ValueError("outcome must be finite")
    known = propensity is not None
    if known:
        p_all = _validate_propensity(propensity, t.shape[0])
    else:
        p_all, _ = propensity_scores(X, t)

    def clip(p):
        return np.clip(p, trim, 1.0 - trim) if trim > 0 else p

    q1, q0 = _qte_point(y, t, clip(p_all), taus, target)
    result = QuantileTreatmentEffectResult(
        quantiles=taus, qte=q1 - q0, q_treated=q1, q_control=q0, target=target, ci=ci
    )
    if n_bootstrap == 0:
        return result

    rng = np.random.default_rng(seed)
    n = t.shape[0]
    draws = np.empty((int(n_bootstrap), taus.size))
    filled = 0
    while filled < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        tb = t[idx]
        if not ((tb == 1).any() and (tb == 0).any()):
            continue
        pb = p_all[idx] if known else propensity_scores(X[idx], tb)[0]
        b1, b0 = _qte_point(y[idx], tb, clip(pb), taus, target)
        draws[filled] = b1 - b0
        filled += 1
    tail = (1.0 - ci) / 2.0
    result.n_bootstrap = int(n_bootstrap)
    result.bootstrap_draws = draws
    result.se = draws.std(axis=0, ddof=1) if n_bootstrap > 1 else np.zeros(taus.size)
    result.ci_lower = np.quantile(draws, tail, axis=0)
    result.ci_upper = np.quantile(draws, 1.0 - tail, axis=0)
    return result
