"""Anderson–Rubin inference that stays valid under weak instruments.

Two-stage least squares standard errors lean on a strong first stage. When
the instrument is weak the normal approximation for the TSLS point estimate
breaks down (Staiger & Stock 1997; Stock & Yogo 2005). The Anderson–Rubin
(1949) test of ``H0: beta = beta0`` does not: it asks whether the
instrumented residual ``Y - beta0 * D`` is uncorrelated with the instrument
after projecting out exogenous covariates.

This module covers the just-identified case (one excluded instrument). The
AR statistic is then a robust (HC1) t-squared on the instrument in the
residual regression, asymptotically ``chi^2(1)`` under the null even when
the first stage is arbitrarily weak. Inverting the test over a grid of
``beta0`` values yields an AR confidence set that may be unbounded when the
instrument is too weak to pin down the sign of the effect.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AndersonRubinResult",
    "AndersonRubinCI",
    "anderson_rubin_test",
    "anderson_rubin_ci",
    "stock_yogo_critical_value",
    "weak_instrument_flag",
]


# Stock & Yogo (2005) Table 5.1 — maximal relative bias critical values for
# the first-stage F with one endogenous regressor (worst-case bias 10%).
_STOCK_YOGO_BIAS_10 = {
    1: 16.38,
    2: 19.93,
    3: 22.30,
    4: 24.58,
    5: 26.87,
}


@dataclass(frozen=True)
class AndersonRubinResult:
    beta0: float
    statistic: float
    p_value: float
    reject: bool
    alpha: float
    n: int


@dataclass(frozen=True)
class AndersonRubinCI:
    lower: Optional[float]
    upper: Optional[float]
    alpha: float
    grid: np.ndarray
    accepted: np.ndarray
    unbounded: bool


def _as_1d(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _design(n: int, covariates) -> np.ndarray:
    if covariates is None:
        return np.ones((n, 1))
    c = np.asarray(covariates, dtype=float)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if c.shape[0] != n:
        raise ValueError("covariates must have one row per observation")
    if c.size and not np.all(np.isfinite(c)):
        raise ValueError("covariates must be finite")
    return np.column_stack([np.ones(n), c])


def _ols_resid(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef, coef


def _hc1_t(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    """HC1 t-stat for the coefficient on ``x`` after residualizing on controls."""
    y_r, _ = _ols_resid(y, controls)
    x_r, _ = _ols_resid(x, controls)
    denom = float(np.dot(x_r, x_r))
    if denom <= 0.0:
        raise ValueError("instrument has no residual variation after covariates")
    beta = float(np.dot(x_r, y_r) / denom)
    resid = y_r - beta * x_r
    n, k = controls.shape[0], controls.shape[1] + 1  # +1 for the tested regressor
    if n <= k:
        raise ValueError("not enough observations for HC1 degrees of freedom")
    meat = float(np.sum((x_r ** 2) * (resid ** 2)))
    # HC1 scale: n / (n - k)
    se2 = (n / (n - k)) * meat / (denom ** 2)
    if se2 <= 0.0:
        return 0.0 if beta == 0.0 else np.inf
    return beta / np.sqrt(se2)


def _chi2_1_sf(stat: float) -> float:
    """Survival function of chi^2(1) = Normal(0,1)^2 via erfc."""
    if not np.isfinite(stat) or stat < 0:
        return float("nan")
    z = np.sqrt(stat)
    return float(math.erfc(z / math.sqrt(2.0)))


def anderson_rubin_test(
    instrument,
    treatment,
    outcome,
    beta0: float = 0.0,
    covariates=None,
    alpha: float = 0.05,
) -> AndersonRubinResult:
    """Anderson–Rubin test of ``H0: structural beta = beta0`` (just-identified).

    ``instrument`` must be a single excluded instrument (length-n vector).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    z = _as_1d(instrument, "instrument")
    d = _as_1d(treatment, "treatment")
    y = _as_1d(outcome, "outcome")
    if not (z.size == d.size == y.size):
        raise ValueError("instrument, treatment and outcome must have the same length")
    controls = _design(z.size, covariates)
    residual = y - float(beta0) * d
    t_stat = _hc1_t(residual, z, controls)
    stat = float(t_stat ** 2)
    p = _chi2_1_sf(stat)
    return AndersonRubinResult(
        beta0=float(beta0),
        statistic=stat,
        p_value=p,
        reject=bool(p < alpha),
        alpha=float(alpha),
        n=int(z.size),
    )


def anderson_rubin_ci(
    instrument,
    treatment,
    outcome,
    covariates=None,
    alpha: float = 0.05,
    grid: Optional[Sequence[float]] = None,
    grid_radius: float = 20.0,
    grid_size: int = 401,
) -> AndersonRubinCI:
    """Invert the Anderson–Rubin test over a grid of ``beta0`` values.

    The returned interval is the contiguous block of accepted ``beta0``
    around the TSLS-like midpoints of the grid. When no grid point is
    rejected at either end the set is marked ``unbounded``.
    """
    if grid is None:
        if grid_radius <= 0 or grid_size < 3:
            raise ValueError("grid_radius must be positive and grid_size >= 3")
        grid_arr = np.linspace(-float(grid_radius), float(grid_radius), int(grid_size))
    else:
        grid_arr = np.asarray(list(grid), dtype=float)
        if grid_arr.ndim != 1 or grid_arr.size < 3:
            raise ValueError("grid must be a 1-D sequence of length >= 3")
        if not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid must be finite")
        grid_arr = np.sort(grid_arr)

    accepted = np.empty(grid_arr.size, dtype=bool)
    for i, b0 in enumerate(grid_arr):
        accepted[i] = not anderson_rubin_test(
            instrument, treatment, outcome, beta0=float(b0), covariates=covariates, alpha=alpha
        ).reject

    if not np.any(accepted):
        return AndersonRubinCI(
            lower=None, upper=None, alpha=float(alpha), grid=grid_arr, accepted=accepted, unbounded=True
        )

    # Take the longest run of accepted points (usually unique).
    best_lo = best_hi = None
    best_len = 0
    i = 0
    while i < accepted.size:
        if not accepted[i]:
            i += 1
            continue
        j = i
        while j < accepted.size and accepted[j]:
            j += 1
        if j - i > best_len:
            best_len = j - i
            best_lo, best_hi = i, j - 1
        i = j
    assert best_lo is not None and best_hi is not None
    unbounded = bool(best_lo == 0 or best_hi == accepted.size - 1)
    return AndersonRubinCI(
        lower=None if best_lo == 0 else float(grid_arr[best_lo]),
        upper=None if best_hi == accepted.size - 1 else float(grid_arr[best_hi]),
        alpha=float(alpha),
        grid=grid_arr,
        accepted=accepted,
        unbounded=unbounded,
    )


def stock_yogo_critical_value(n_instruments: int = 1, *, bias: float = 0.10) -> float:
    """Stock–Yogo critical value for maximal relative bias (one endogenous).

    Only the 10% relative-bias column is tabulated here (the common rule of
    thumb). ``n_instruments`` is the number of excluded instruments.
    """
    if bias != 0.10:
        raise ValueError("only bias=0.10 is tabulated in this helper")
    n_instruments = int(n_instruments)
    if n_instruments not in _STOCK_YOGO_BIAS_10:
        raise ValueError("n_instruments must be between 1 and 5 for this table")
    return float(_STOCK_YOGO_BIAS_10[n_instruments])


def weak_instrument_flag(first_stage_f: float, n_instruments: int = 1) -> bool:
    """Return True when the first-stage F is below the Stock–Yogo 10% bias CV."""
    if first_stage_f < 0 or not np.isfinite(first_stage_f):
        raise ValueError("first_stage_f must be a finite non-negative number")
    return bool(first_stage_f < stock_yogo_critical_value(n_instruments))
