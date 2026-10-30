"""Interrupted time series (segmented regression) with HAC standard errors.

When one unit (a region, a product, a hospital) gets an intervention at a
known time and no untreated comparison group exists, the standard
quasi-experimental design is the *interrupted time series* (Wagner et al.
2002; Bernal, Cummins & Gasparrini 2017). The pre-intervention trend is
projected forward as the counterfactual:

    y_t = b0 + b1 t + b2 D_t + b3 (t - t0) D_t  [+ seasonal terms] + e_t

``D_t = 1`` from the intervention time ``t0`` onward. ``b2`` is the immediate
**level change** and ``b3`` is the **slope change**. Each post-period effect
is ``b2 + b3 (t - t0)``. The module also reports the average and the
cumulative effect over the post period.

Time series residuals are autocorrelated, so the usual OLS standard errors
are too small. Inference here uses Newey-West HAC covariances with Bartlett
weights. The default lag is ``floor(4 (T/100)^(2/9))`` and there is a
``T/(T-k)`` small-sample correction. Effect SEs are exact linear
combinations of the coefficient covariance.

* :func:`interrupted_time_series` -- single-series ITS, with optional Fourier
  seasonality and extra covariates.
* :func:`comparative_interrupted_time_series` -- treated vs control series
  (difference-in-differences of levels and slopes). It fits the ITS to the
  difference ``y_treated - y_control``. With common time points, the
  coefficients are the same as those of the fully interacted stacked model,
  and HAC on the difference series accounts for correlation between the two
  series.
* :func:`its_placebo_test` -- refits the model on pre-period data only, with
  fake interruption times. Large placebo effects point to a misspecified
  trend.

Only ``numpy`` is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .bounds import _norm_cdf, _norm_ppf

__all__ = [
    "ITSResult",
    "newey_west_cov",
    "interrupted_time_series",
    "comparative_interrupted_time_series",
    "its_placebo_test",
]


@dataclass
class ITSResult:
    """Segmented-regression fit and effect estimates."""

    coef: Dict[str, float]
    se: Dict[str, float]
    level_change: float
    level_change_se: float
    slope_change: float
    slope_change_se: float
    pre_slope: float
    post_slope: float
    average_effect: float
    average_effect_se: float
    cumulative_effect: float
    cumulative_effect_se: float
    relative_effect: Optional[float]
    effects: np.ndarray
    effects_se: np.ndarray
    fitted: np.ndarray
    counterfactual: np.ndarray
    residuals: np.ndarray
    time: np.ndarray
    t0: float
    n_pre: int
    n_post: int
    hac_lags: int
    durbin_watson: float
    residual_autocorrelation: float
    alpha: float
    kind: str = "its"
    covariance: np.ndarray = field(default=None, repr=False)

    def ci(self, name: str) -> tuple:
        """``(lo, hi)`` normal CI for ``"level_change"``, ``"slope_change"``,
        ``"average_effect"`` or ``"cumulative_effect"``."""

        est, se = getattr(self, name), getattr(self, f"{name}_se")
        z = _norm_ppf(1.0 - self.alpha / 2.0)
        return est - z * se, est + z * se

    def p_value(self, name: str) -> float:
        est, se = getattr(self, name), getattr(self, f"{name}_se")
        if se <= 0:
            return 0.0 if est != 0 else 1.0
        return 2.0 * (1.0 - _norm_cdf(abs(est / se)))

    def summary(self) -> str:
        pct = f"{1 - self.alpha:.0%}"
        lines = [
            f"{'Comparative interrupted' if self.kind == 'cits' else 'Interrupted'} time series: "
            f"{self.n_pre} pre / {self.n_post} post points, Newey-West lags={self.hac_lags}",
        ]
        for name in ("level_change", "slope_change", "average_effect", "cumulative_effect"):
            lo, hi = self.ci(name)
            lines.append(
                f"  {name:<18} {getattr(self, name):>10.4f}  se {getattr(self, name + '_se'):.4f}  "
                f"{pct} CI [{lo:.4f}, {hi:.4f}]  p={self.p_value(name):.4g}"
            )
        if self.relative_effect is not None:
            lines.append(f"  relative effect    {self.relative_effect:+.2%} of the counterfactual mean")
        lines.append(
            f"  Durbin-Watson {self.durbin_watson:.3f}, lag-1 residual autocorrelation {self.residual_autocorrelation:+.3f}"
        )
        return "\n".join(lines)


def _default_lags(n: int) -> int:
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int, small_sample: bool = True) -> np.ndarray:
    """Newey-West HAC covariance of OLS coefficients (Bartlett kernel).

    ``lags=0`` gives White's HC0 (or HC1 with ``small_sample``).
    """

    X = np.asarray(X, dtype=float)
    u = np.asarray(resid, dtype=float).ravel()
    n, k = X.shape
    if u.size != n:
        raise ValueError("resid must have one entry per row of X")
    if lags < 0:
        raise ValueError("lags must be >= 0")
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, min(int(lags), n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = xu[lag:].T @ xu[:-lag]
        S += w * (G + G.T)
    bread = np.linalg.pinv(X.T @ X)
    V = bread @ S @ bread
    if small_sample and n > k:
        V *= n / (n - k)
    return V


def _design(time, t0, period, n_harmonics, covariates):
    D = (time >= t0).astype(float)
    cols = [np.ones_like(time), time - t0, D, (time - t0) * D]
    names = ["intercept", "pre_slope", "level_change", "slope_change"]
    if period is not None:
        for h in range(1, n_harmonics + 1):
            ang = 2.0 * np.pi * h * time / period
            cols += [np.sin(ang), np.cos(ang)]
            names += [f"sin{h}", f"cos{h}"]
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != time.size:
            raise ValueError("covariates must have one row per time point")
        for j in range(C.shape[1]):
            cols.append(C[:, j])
            names.append(f"cov{j}")
    return np.column_stack(cols), names, D


def interrupted_time_series(
    y: Sequence[float],
    intervention,
    *,
    time: Optional[Sequence[float]] = None,
    period: Optional[float] = None,
    n_harmonics: int = 1,
    covariates=None,
    hac_lags: Optional[int] = None,
    alpha: float = 0.05,
) -> ITSResult:
    """Segmented regression ITS with Newey-West inference.

    Parameters
    ----------
    y:
        Outcome series ordered in time.
    intervention:
        The first post-intervention time ``t0``, on the scale of ``time``
        (which defaults to ``0, 1, ..., T-1``, so an integer index works).
    period, n_harmonics:
        Optional Fourier seasonality (for example ``period=12`` for monthly
        data).
    covariates:
        Optional ``(T,)`` or ``(T, p)`` time-varying controls.
    hac_lags:
        Newey-West truncation lag. Defaults to ``floor(4 (T/100)^(2/9))``.
        Use ``0`` for heteroskedasticity-robust SEs only.
    """

    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    time = np.arange(n, dtype=float) if time is None else np.asarray(time, dtype=float).ravel()
    if time.size != n:
        raise ValueError("time must have the same length as y")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(time))):
        raise ValueError("y and time must be finite")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if period is not None and (period <= 0 or n_harmonics < 1):
        raise ValueError("period must be positive and n_harmonics >= 1")
    t0 = float(intervention)
    X, names, D = _design(time, t0, period, n_harmonics, covariates)
    n_post = int(D.sum())
    n_pre = n - n_post
    if n_pre < 3 or n_post < 2:
        raise ValueError("need at least 3 pre-intervention and 2 post-intervention points")
    if n <= X.shape[1]:
        raise ValueError("too few time points for the number of parameters")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("design matrix is rank deficient")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    lags = _default_lags(n) if hac_lags is None else int(hac_lags)
    if lags < 0:
        raise ValueError("hac_lags must be >= 0")
    V = newey_west_cov(X, resid, lags)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))

    # per-period effects b2 + b3 (t - t0) as linear combinations
    post = D > 0
    L = np.zeros((n_post, X.shape[1]))
    L[:, 2] = 1.0
    L[:, 3] = time[post] - t0
    eff = L @ beta
    eff_se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", L, V, L), 0.0, None))
    l_avg = L.mean(axis=0)
    l_cum = L.sum(axis=0)
    cf = fitted.copy()
    cf[post] -= eff
    cf_mean = float(cf[post].mean())
    rel = float(eff.mean() / cf_mean) if abs(cf_mean) > 1e-12 else None
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2)) if np.any(resid) else float("nan")
    rc = resid - resid.mean()
    rho = float((rc[1:] @ rc[:-1]) / (rc @ rc)) if np.any(rc) else 0.0
    return ITSResult(
        coef=dict(zip(names, map(float, beta))),
        se=dict(zip(names, map(float, se))),
        level_change=float(beta[2]), level_change_se=float(se[2]),
        slope_change=float(beta[3]), slope_change_se=float(se[3]),
        pre_slope=float(beta[1]), post_slope=float(beta[1] + beta[3]),
        average_effect=float(l_avg @ beta), average_effect_se=float(np.sqrt(max(l_avg @ V @ l_avg, 0.0))),
        cumulative_effect=float(l_cum @ beta), cumulative_effect_se=float(np.sqrt(max(l_cum @ V @ l_cum, 0.0))),
        relative_effect=rel, effects=eff, effects_se=eff_se,
        fitted=fitted, counterfactual=cf, residuals=resid, time=time, t0=t0,
        n_pre=n_pre, n_post=n_post, hac_lags=lags, durbin_watson=dw,
        residual_autocorrelation=rho, alpha=alpha, covariance=V,
    )


def comparative_interrupted_time_series(
    y_treated: Sequence[float],
    y_control: Sequence[float],
    intervention,
    **kwargs,
) -> ITSResult:
    """Comparative ITS: the ITS of ``y_treated - y_control``.

    The level and slope changes are differences in differences. They remove
    shocks that hit both series at the intervention time (for example a
    concurrent policy or a seasonal break). ``relative_effect`` is computed
    on the difference scale, so it is set to ``None``.
    """

    yt = np.asarray(y_treated, dtype=float).ravel()
    yc = np.asarray(y_control, dtype=float).ravel()
    if yt.shape != yc.shape:
        raise ValueError("treated and control series must have the same length")
    res = interrupted_time_series(yt - yc, intervention, **kwargs)
    res.kind = "cits"
    res.relative_effect = None
    return res


def its_placebo_test(
    y: Sequence[float],
    intervention,
    placebo_times: Optional[Sequence[float]] = None,
    *,
    time: Optional[Sequence[float]] = None,
    min_segment: int = 6,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Placebo-in-time check that uses pre-intervention data only.

    The model is refit on the pre-period with a fake interruption at each
    placebo time. By default these are every pre-period time that leaves at
    least ``min_segment`` points on both sides. Under a well-specified trend
    the placebo level changes should be centred on zero, with roughly
    ``alpha`` of them significant.

    Returns arrays ``times``, ``level_change``, ``level_change_se``,
    ``p_value`` and the scalar ``rejection_rate``.
    """

    y = np.asarray(y, dtype=float).ravel()
    time = np.arange(y.size, dtype=float) if time is None else np.asarray(time, dtype=float).ravel()
    pre = time < float(intervention)
    yp, tp = y[pre], time[pre]
    if placebo_times is None:
        placebo_times = tp[min_segment: tp.size - min_segment + 1]
    times = np.asarray(placebo_times, dtype=float)
    if times.size == 0:
        raise ValueError("pre-period too short for placebo interruptions")
    lc, lse, pv = [], [], []
    for t in times:
        r = interrupted_time_series(yp, t, time=tp, **kwargs)
        lc.append(r.level_change)
        lse.append(r.level_change_se)
        pv.append(r.p_value("level_change"))
    alpha = kwargs.get("alpha", 0.05)
    pv = np.asarray(pv)
    return {
        "times": times,
        "level_change": np.asarray(lc),
        "level_change_se": np.asarray(lse),
        "p_value": pv,
        "rejection_rate": float(np.mean(pv < alpha)),
    }
