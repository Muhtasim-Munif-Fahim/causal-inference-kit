"""Omitted-variable-bias sensitivity for linear regression (Cinelli & Hazlett 2020).

For the OLS estimate of a treatment coefficient in ``Y ~ D + X``, an
omitted confounder ``Z`` changes the estimate by exactly

    |bias| = se(D) * sqrt(dof) * sqrt(R2_{Y~Z|D,X} * R2_{D~Z|X} / (1 - R2_{D~Z|X}))

where ``R2_{.|.}`` are partial R² values. This module reports, with numpy
only:

* ``partial_r2_yd`` -- the share of residual outcome variance explained by
  the treatment; an extreme confounder explaining all residual outcome
  variance would need ``R2_{D~Z|X} >= partial_r2_yd`` to remove the estimate.
* the **robustness value** ``RV_q`` -- the minimum strength of association
  (equal partial R² with treatment and outcome) a confounder needs to
  reduce the estimate by ``100*q`` %, and ``RV_{q,alpha}`` -- the same for
  making the reduced estimate statistically insignificant at level ``alpha``;
* :func:`ovb_adjusted_estimate` -- estimate, SE and CI under a hypothesised
  confounder of given strength (bias assumed to shrink the effect towards 0);
* :func:`ovb_bounds` -- bounds when the confounder is ``kd``/``ky`` times
  as strong as an *observed* benchmark covariate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "OVBBound",
    "OVBSensitivityResult",
    "ovb_adjusted_estimate",
    "ovb_bounds",
    "ovb_sensitivity",
    "partial_r2",
    "robustness_value",
]


# --------------------------------------------------------------- t quantiles
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """Upper tail ``P(T > t)`` of Student's t."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return tail if t >= 0 else 1.0 - tail


def _t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if p == 0.5:
        return 0.0
    upper = p > 0.5
    target = 1.0 - p if upper else p  # lower-tail probability of -|t|
    lo, hi = 0.0, 1.0
    while _t_sf(hi, df) > target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    q = 0.5 * (lo + hi)
    return q if upper else -q


# ----------------------------------------------------------------- core OLS
def _ols(y: np.ndarray, X: np.ndarray):
    n, p = X.shape
    if n <= p:
        raise ValueError("need more observations than regressors")
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise ValueError("design matrix is singular") from exc
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - p
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0.0, None))
    return beta, se, dof


def partial_r2(t_stat: float, dof: float) -> float:
    """Partial R² of a regressor from its t-statistic: ``t² / (t² + dof)``."""
    if dof <= 0:
        raise ValueError("dof must be positive")
    t2 = float(t_stat) ** 2
    return t2 / (t2 + dof)


def robustness_value(t_stat: float, dof: float, q: float = 1.0, alpha: Optional[float] = None) -> float:
    """Cinelli-Hazlett robustness value ``RV_q`` (or ``RV_{q,alpha}``).

    The partial R² that a confounder must have with *both* treatment and
    outcome to reduce the estimate by ``100*q`` % (``alpha=None``) or to
    bring the reduced estimate's ``1 - alpha`` CI to include it
    (``alpha`` given). Returns 0 when the estimate is already insignificant.
    """
    if dof <= 1:
        raise ValueError("dof must exceed 1")
    if not q > 0:
        raise ValueError("q must be positive")
    f = q * abs(float(t_stat)) / math.sqrt(dof)
    if alpha is not None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        f_crit = abs(_t_ppf(alpha / 2.0, dof - 1)) / math.sqrt(dof - 1)
        f = f - f_crit
        if f <= 0:
            return 0.0
    return 0.5 * (math.sqrt(f**4 + 4.0 * f**2) - f**2)


def _prepare(X, treatment, outcome):
    D = np.asarray(treatment, dtype=float).ravel()
    Y = np.asarray(outcome, dtype=float).ravel()
    if X is None:
        Xm = np.empty((D.size, 0))
    else:
        Xm = np.asarray(X, dtype=float)
        if Xm.ndim == 1:
            Xm = Xm.reshape(-1, 1)
    if not (Xm.shape[0] == D.size == Y.size):
        raise ValueError("X, treatment and outcome must have the same number of rows")
    for arr in (Xm, D, Y):
        if not np.all(np.isfinite(arr)):
            raise ValueError("inputs must be finite")
    if np.ptp(D) == 0:
        raise ValueError("treatment must vary")
    return Xm, D, Y


@dataclass
class OVBBound:
    """Bound implied by a confounder ``kd``/``ky`` times as strong as a benchmark."""

    benchmark: str
    kd: float
    ky: float
    r2dz_x: float
    r2yz_dx: float
    adjusted_estimate: float
    adjusted_se: float
    adjusted_lower: float
    adjusted_upper: float


@dataclass
class OVBSensitivityResult:
    estimate: float
    se: float
    t_stat: float
    dof: int
    partial_r2_yd: float
    rv_q: float
    rv_qa: float
    q: float
    alpha: float
    bounds: List[OVBBound] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"estimate {self.estimate:.4g} (se {self.se:.4g}, t {self.t_stat:.3f}, dof {self.dof})",
            f"partial R2 of treatment with outcome: {self.partial_r2_yd:.4f}",
            f"RV_q (q={self.q:g}): {self.rv_q:.4f}",
            f"RV_q,alpha (alpha={self.alpha:g}): {self.rv_qa:.4f}",
        ]
        for b in self.bounds:
            lines.append(
                f"{b.kd:g}x {b.benchmark}: R2dz={b.r2dz_x:.4f} R2yz={b.r2yz_dx:.4f} "
                f"adj {b.adjusted_estimate:.4g} [{b.adjusted_lower:.4g}, {b.adjusted_upper:.4g}]"
            )
        return "\n".join(lines)


def ovb_adjusted_estimate(
    estimate: float,
    se: float,
    dof: int,
    r2dz_x: float,
    r2yz_dx: float,
    alpha: float = 0.05,
    reduce: bool = True,
):
    """Estimate, SE and ``1 - alpha`` CI after adjusting for a hypothetical confounder.

    Returns ``(adjusted_estimate, adjusted_se, lower, upper)``. With
    ``reduce=True`` the bias moves the estimate towards zero.
    """
    if not (0.0 <= r2dz_x < 1.0 and 0.0 <= r2yz_dx <= 1.0):
        raise ValueError("r2dz_x must be in [0, 1) and r2yz_dx in [0, 1]")
    if dof <= 1:
        raise ValueError("dof must exceed 1")
    bias = se * math.sqrt(dof) * math.sqrt(r2yz_dx * r2dz_x / (1.0 - r2dz_x))
    sign = 1.0 if estimate >= 0 else -1.0
    adj = estimate - sign * bias if reduce else estimate + sign * bias
    adj_se = se * math.sqrt((1.0 - r2yz_dx) / (1.0 - r2dz_x)) * math.sqrt(dof / (dof - 1.0))
    crit = abs(_t_ppf(alpha / 2.0, dof - 1))
    return adj, adj_se, adj - crit * adj_se, adj + crit * adj_se


def ovb_bounds(
    r2dxj_x: float,
    r2yxj_dx: float,
    estimate: float,
    se: float,
    dof: int,
    kd: Sequence[float] = (1.0, 2.0, 3.0),
    ky: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    benchmark: str = "benchmark",
) -> List[OVBBound]:
    """Bounds from a benchmark covariate's partial R² (Cinelli & Hazlett 2020, Sec. 6)."""
    kd = [float(k) for k in np.atleast_1d(kd)]
    ky = kd if ky is None else [float(k) for k in np.atleast_1d(ky)]
    if len(ky) != len(kd):
        raise ValueError("kd and ky must have the same length")
    out = []
    for k_d, k_y in zip(kd, ky):
        if k_d < 0 or k_y < 0:
            raise ValueError("kd and ky must be non-negative")
        r2dz = k_d * r2dxj_x / (1.0 - r2dxj_x)
        if r2dz >= 1.0:
            raise ValueError(f"kd={k_d:g} implies R2_D~Z|X >= 1; use a smaller multiplier")
        r2zxj = k_d * r2dxj_x**2 / ((1.0 - k_d * r2dxj_x) * (1.0 - r2dxj_x))
        if r2zxj >= 1.0:
            raise ValueError(f"kd={k_d:g} is impossible for this benchmark")
        r2yz = ((math.sqrt(k_y) + math.sqrt(r2zxj)) / math.sqrt(1.0 - r2zxj)) ** 2 * (
            r2yxj_dx / (1.0 - r2yxj_dx)
        )
        r2yz = min(r2yz, 1.0)
        adj, adj_se, lo, hi = ovb_adjusted_estimate(estimate, se, dof, r2dz, r2yz, alpha=alpha)
        out.append(OVBBound(benchmark, k_d, k_y, r2dz, r2yz, adj, adj_se, lo, hi))
    return out


def ovb_sensitivity(
    X,
    treatment,
    outcome,
    q: float = 1.0,
    alpha: float = 0.05,
    benchmark: Optional[int] = None,
    benchmark_name: Optional[str] = None,
    kd: Sequence[float] = (1.0, 2.0, 3.0),
    ky: Optional[Sequence[float]] = None,
) -> OVBSensitivityResult:
    """Fit ``Y ~ 1 + D + X`` by OLS and run the Cinelli-Hazlett sensitivity analysis.

    ``benchmark`` is a column index of ``X`` used for :func:`ovb_bounds`.
    """
    Xm, D, Y = _prepare(X, treatment, outcome)
    n = D.size
    design = np.column_stack([np.ones(n), D, Xm])
    beta, se, dof = _ols(Y, design)
    if dof <= 1:
        raise ValueError("need at least two residual degrees of freedom")
    est, s = float(beta[1]), float(se[1])
    t = est / s if s > 0 else float("inf")
    result = OVBSensitivityResult(
        estimate=est,
        se=s,
        t_stat=t,
        dof=int(dof),
        partial_r2_yd=partial_r2(t, dof),
        rv_q=robustness_value(t, dof, q=q),
        rv_qa=robustness_value(t, dof, q=q, alpha=alpha),
        q=float(q),
        alpha=float(alpha),
    )
    if benchmark is not None:
        j = int(benchmark)
        if not 0 <= j < Xm.shape[1]:
            raise ValueError("benchmark must index a column of X")
        t_yxj = beta[2 + j] / se[2 + j]
        r2yxj = partial_r2(t_yxj, dof)
        bd, sd, dof_d = _ols(D, np.column_stack([np.ones(n), Xm]))
        r2dxj = partial_r2(bd[1 + j] / sd[1 + j], dof_d)
        result.bounds = ovb_bounds(
            r2dxj, r2yxj, est, s, dof, kd=kd, ky=ky, alpha=alpha,
            benchmark=benchmark_name or f"X{j}",
        )
    return result
