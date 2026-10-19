"""Synthetic difference-in-differences for block treatment designs.

Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), "Synthetic
Difference-in-Differences", *American Economic Review* 111(12).

DiD weights every control unit and every pre-period equally and relies on
parallel trends; synthetic control (SC) re-weights control units to match the
treated pre-period path but uses no time weights and no unit fixed effect.
SDID combines both:

- **unit weights** ``omega`` (non-negative, sum to one) make the weighted
  control trend parallel to the treated average *up to a constant*
  (an intercept ``omega_0``), with a ridge penalty
  ``zeta^2 T_pre ||omega||^2`` that spreads weight across similar units;
- **time weights** ``lambda`` (non-negative, sum to one over pre-periods)
  pick the pre-periods whose control outcomes best predict the controls'
  post-period average, again up to a constant;
- the effect is the doubly weighted 2x2 DiD

      tau = (Ybar_tr,post - sum_t lambda_t Ybar_tr,t)
            - sum_i omega_i (Ybar_i,post - sum_t lambda_t Y_i,t).

With ``omega`` uniform and ``lambda`` uniform this is exactly DiD.  The
regularisation ``zeta = (N_tr T_post)^{1/4} sigma`` uses ``sigma``, the
standard deviation of first differences of control pre-period outcomes, as in
the reference implementation (``synthdid``).

Standard errors follow the paper: the **placebo** method (Algorithm 4)
re-assigns treatment to random sets of ``N_tr`` controls and re-estimates
everything (needs more controls than treated units); the **jackknife**
(Algorithm 3) drops one unit at a time with the weights held fixed (needs at
least two treated units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .estimators import _fit_simplex_weights, _panel_from_long, _project_simplex

__all__ = [
    "SyntheticDiDResult",
    "synthetic_did",
    "simulate_sdid_panel",
]


@dataclass
class SyntheticDiDResult:
    """Synthetic DiD estimate with its weights and DiD / SC comparisons.

    Attributes
    ----------
    estimate : float
        SDID average treatment effect on the treated units, post-treatment.
    std_err, ci_lower, ci_upper : float or None
        Standard error and normal 95% interval (``None`` if ``se_method``
        is ``"none"``).
    se_method : str
        ``"placebo"``, ``"jackknife"`` or ``"none"``.
    unit_weights : ndarray of shape (n_control,)
        ``omega`` aligned with ``control_ids``.
    time_weights : ndarray of shape (n_pre,)
        ``lambda`` aligned with ``pre_times``.
    unit_intercept, time_intercept : float
        ``omega_0`` and ``lambda_0`` from the weight fits.
    did_estimate : float
        Plain DiD (uniform unit and time weights) on the same panel.
    sc_estimate : float
        Synthetic control (simplex weights, no intercept, no time weights)
        on the treated-average path.
    zeta : float
        Unit-weight regularisation used.
    noise_level : float
        ``sigma``: std of first differences of control pre-period outcomes.
    """

    estimate: float
    std_err: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    se_method: str
    unit_weights: np.ndarray
    time_weights: np.ndarray
    unit_intercept: float
    time_intercept: float
    control_ids: np.ndarray
    treated_ids: np.ndarray
    pre_times: np.ndarray
    post_times: np.ndarray
    did_estimate: float
    sc_estimate: float
    zeta: float
    noise_level: float
    placebo_estimates: Optional[np.ndarray] = None

    @property
    def effective_controls(self) -> float:
        """Inverse Herfindahl index of the unit weights (1 = one donor)."""
        return float(1.0 / np.sum(self.unit_weights**2))


def _simplex_ridge_intercept(A: np.ndarray, b: np.ndarray, penalty: float, max_iter: int = 20000):
    """Solve ``min_{c, w in simplex} ||c + A w - b||^2 + penalty ||w||^2``.

    The free intercept ``c`` is profiled out by centring the rows of ``A`` and
    ``b``; the remaining simplex-constrained ridge problem is solved by FISTA
    with Euclidean projection onto the simplex.  Returns ``(w, c)``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, k = A.shape
    A_c = A - A.mean(axis=0, keepdims=True)
    b_c = b - b.mean()
    gram = A_c.T @ A_c + penalty * np.eye(k)
    atb = A_c.T @ b_c
    lipschitz = float(np.linalg.eigvalsh(gram).max())
    w = np.full(k, 1.0 / k)
    if lipschitz <= 1e-18:
        return w, float(np.mean(b - A @ w))
    step = 1.0 / lipschitz
    z = w.copy()
    t = 1.0
    for _ in range(max_iter):
        w_new = _project_simplex(z - step * (gram @ z - atb))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < 1e-12:
            w = w_new
            break
        w, t = w_new, t_new
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    return w, float(np.mean(b - A @ w))


def _noise_level(Y_co_pre: np.ndarray) -> float:
    if Y_co_pre.shape[1] < 2:
        return float(np.std(Y_co_pre)) if Y_co_pre.size > 1 else 0.0
    diffs = np.diff(Y_co_pre, axis=1).ravel()
    if diffs.size < 2:
        return 0.0
    return float(np.std(diffs, ddof=1))


def _sdid_core(Y: np.ndarray, n_treated: int, n_pre: int, zeta: Optional[float] = None):
    """SDID on a panel whose last ``n_treated`` rows are treated after ``n_pre``."""
    n_units, n_times = Y.shape
    n_control = n_units - n_treated
    n_post = n_times - n_pre
    Y_co_pre = Y[:n_control, :n_pre]
    Y_co_post = Y[:n_control, n_pre:]
    y_tr = Y[n_control:].mean(axis=0)

    sigma = _noise_level(Y_co_pre)
    if zeta is None:
        zeta = (n_treated * n_post) ** 0.25 * sigma
    zeta_lambda = 1e-6 * sigma

    omega, omega0 = _simplex_ridge_intercept(Y_co_pre.T, y_tr[:n_pre], zeta**2 * n_pre)
    lam, lam0 = _simplex_ridge_intercept(Y_co_pre, Y_co_post.mean(axis=1), zeta_lambda**2 * n_control)

    delta_co = Y_co_post.mean(axis=1) - Y_co_pre @ lam
    delta_tr = y_tr[n_pre:].mean() - y_tr[:n_pre] @ lam
    tau = float(delta_tr - omega @ delta_co)
    return tau, omega, lam, float(omega0), float(lam0), float(zeta), sigma


def synthetic_did(
    unit,
    time,
    outcome,
    treated_units: Iterable,
    treatment_time,
    se_method: str = "auto",
    n_placebo: int = 200,
    zeta: Optional[float] = None,
    seed: int = 0,
) -> SyntheticDiDResult:
    """Synthetic difference-in-differences ATT for a block (simultaneous) adoption.

    Parameters
    ----------
    unit, time, outcome : array-like
        Balanced long-format panel.
    treated_units : iterable
        Unit ids that become treated at ``treatment_time`` (all others are
        never treated).
    treatment_time :
        First post-treatment period; earlier periods are pre-treatment.
    se_method : {"auto", "placebo", "jackknife", "none"}
        ``"auto"`` uses the jackknife with two or more treated units and the
        placebo method otherwise.
    n_placebo : int
        Placebo replications for ``se_method="placebo"``.
    zeta : float, optional
        Override the unit-weight regularisation ``(N_tr T_post)^{1/4} sigma``.
    seed : int
        Seed for placebo draws.
    """
    if se_method not in ("auto", "placebo", "jackknife", "none"):
        raise ValueError("se_method must be 'auto', 'placebo', 'jackknife' or 'none'")
    if zeta is not None and zeta < 0:
        raise ValueError("zeta must be non-negative")
    panel, unit_ids, time_ids = _panel_from_long(unit, time, outcome)
    treated_set = list(treated_units)
    if not treated_set:
        raise ValueError("at least one treated unit is required")
    is_treated = np.isin(unit_ids, np.asarray(treated_set, dtype=unit_ids.dtype))
    if is_treated.sum() != len(set(treated_set)):
        raise ValueError("every treated unit must appear in the panel")
    n_treated = int(is_treated.sum())
    n_control = int((~is_treated).sum())
    if n_control < 2:
        raise ValueError("at least two control units are required")
    pre_mask = time_ids < treatment_time
    n_pre = int(pre_mask.sum())
    if n_pre < 2:
        raise ValueError("at least two pre-treatment periods are required")
    if n_pre == time_ids.size:
        raise ValueError("at least one post-treatment period is required")
    if not np.all(pre_mask[:n_pre]):
        raise ValueError("pre-treatment periods must precede post-treatment periods")

    order = np.concatenate([np.flatnonzero(~is_treated), np.flatnonzero(is_treated)])
    Y = panel[order]
    control_ids = unit_ids[~is_treated]
    treated_ids = unit_ids[is_treated]

    tau, omega, lam, omega0, lam0, zeta_used, sigma = _sdid_core(Y, n_treated, n_pre, zeta)

    # Comparisons on the same panel.
    Y_co, y_tr = Y[:n_control], Y[n_control:].mean(axis=0)
    did = float(
        (y_tr[n_pre:].mean() - y_tr[:n_pre].mean())
        - (Y_co[:, n_pre:].mean() - Y_co[:, :n_pre].mean())
    )
    sc_w = _fit_simplex_weights(Y_co[:, :n_pre].T, y_tr[:n_pre])
    sc = float(np.mean(y_tr[n_pre:] - sc_w @ Y_co[:, n_pre:]))

    method = se_method
    if method == "auto":
        method = "jackknife" if n_treated >= 2 else "placebo"
    std_err = None
    placebo_estimates = None
    if method == "placebo":
        if n_control <= n_treated:
            raise ValueError("placebo standard errors need more control than treated units")
        if n_placebo < 2:
            raise ValueError("n_placebo must be at least 2")
        rng = np.random.default_rng(seed)
        draws = np.empty(n_placebo)
        for b in range(n_placebo):
            perm = rng.permutation(n_control)
            Yb = Y_co[perm]  # last n_treated controls play the treated units
            draws[b] = _sdid_core(Yb, n_treated, n_pre, None)[0]
        placebo_estimates = draws
        std_err = float(np.sqrt(np.var(draws, ddof=1)))
    elif method == "jackknife":
        if n_treated < 2:
            raise ValueError("jackknife standard errors need at least two treated units")
        estimates = []
        for i in range(n_control + n_treated):
            if i < n_control:
                keep = np.ones(n_control, dtype=bool)
                keep[i] = False
                w = omega[keep]
                if w.sum() <= 0:
                    continue  # all weight on the dropped unit: skip, as synthdid does
                w = w / w.sum()
                Yc = Y_co[keep]
                yt = y_tr
            else:
                w = omega
                Yc = Y_co
                tr_keep = np.ones(n_treated, dtype=bool)
                tr_keep[i - n_control] = False
                yt = Y[n_control:][tr_keep].mean(axis=0)
            d_co = Yc[:, n_pre:].mean(axis=1) - Yc[:, :n_pre] @ lam
            d_tr = yt[n_pre:].mean() - yt[:n_pre] @ lam
            estimates.append(d_tr - w @ d_co)
        estimates = np.asarray(estimates)
        n_jk = estimates.size
        std_err = float(np.sqrt((n_jk - 1) / n_jk * np.sum((estimates - estimates.mean()) ** 2)))

    ci_lower = ci_upper = None
    if std_err is not None:
        ci_lower = tau - 1.959963984540054 * std_err
        ci_upper = tau + 1.959963984540054 * std_err

    return SyntheticDiDResult(
        estimate=tau,
        std_err=std_err,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se_method=method,
        unit_weights=omega,
        time_weights=lam,
        unit_intercept=omega0,
        time_intercept=lam0,
        control_ids=control_ids,
        treated_ids=treated_ids,
        pre_times=time_ids[:n_pre],
        post_times=time_ids[n_pre:],
        did_estimate=did,
        sc_estimate=sc,
        zeta=zeta_used,
        noise_level=sigma,
        placebo_estimates=placebo_estimates,
    )


def simulate_sdid_panel(
    n_control: int = 40,
    n_treated: int = 5,
    n_pre: int = 20,
    n_post: int = 6,
    att: float = 2.0,
    n_factors: int = 2,
    selection: float = 0.75,
    noise: float = 0.5,
    seed: int = 0,
):
    """Simulate a block-adoption panel where parallel trends fail.

    ``Y_it = alpha_i + beta_t + gamma_i' F_t + eps_it``.  Treated units load
    more heavily on the trending factors (their loadings are shifted by
    ``selection``), so plain DiD is biased, while a re-weighted set of
    controls can track them.  Treated units adopt at period ``n_pre``.

    Returns ``unit, time, outcome, treated_units, treatment_time, true_att``.
    """
    if n_control < 2 or n_treated < 1:
        raise ValueError("need at least two control units and one treated unit")
    if n_pre < 2 or n_post < 1:
        raise ValueError("need at least two pre-periods and one post-period")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    rng = np.random.default_rng(seed)
    n_units = n_control + n_treated
    n_times = n_pre + n_post
    t = np.arange(n_times, dtype=float)
    factors = np.vstack(
        [np.linspace(-1.0, 1.0, n_times) * (k + 1) + 0.3 * rng.normal(size=n_times) for k in range(n_factors)]
    )
    loadings = rng.normal(size=(n_units, n_factors))
    loadings[n_control:] += selection
    alpha = rng.normal(size=n_units)
    alpha[n_control:] += 1.0
    beta = 0.2 * t + 0.5 * np.sin(t / 2.0)
    panel = alpha[:, None] + beta[None, :] + loadings @ factors + noise * rng.normal(size=(n_units, n_times))
    panel[n_control:, n_pre:] += att
    unit = np.repeat(np.arange(n_units), n_times)
    time = np.tile(np.arange(n_times), n_units)
    treated_units = list(range(n_control, n_units))
    return unit, time, panel.reshape(-1), treated_units, int(n_pre), float(att)
