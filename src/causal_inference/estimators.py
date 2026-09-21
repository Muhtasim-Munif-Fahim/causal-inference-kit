"""Treatment-effect estimators: IPW, AIPW, matching, DiD, synthetic control, RD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .propensity import _validate_treatment, propensity_scores


def _validate_propensity(propensity, n):
    p = np.asarray(propensity, dtype=float).ravel()
    if p.shape[0] != n:
        raise ValueError("propensity must have one entry per unit")
    if not np.all(np.isfinite(p)):
        raise ValueError("propensity must be finite")
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("propensity must lie strictly between 0 and 1")
    return p


def _as_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("expected a 1d or 2d array")
    return X


def _coerce_arrays(X, treatment, outcome):
    X = _as_2d(X)
    treatment = _validate_treatment(treatment)
    outcome = np.asarray(outcome, dtype=float).ravel()
    if X.shape[0] != treatment.shape[0] or treatment.shape[0] != outcome.shape[0]:
        raise ValueError("X, treatment and outcome must have the same number of rows")
    if X.shape[0] == 0:
        raise ValueError("at least one row of data is required")
    return X, treatment, outcome


def _outcome_features(X, W=None):
    """Covariates for the outcome regression: ``X``, plus ``W`` when given."""
    if W is None:
        return X
    W = _as_2d(W)
    if W.shape[0] != X.shape[0]:
        raise ValueError("W must have the same number of rows as X")
    if W.shape[1] == 0:
        return X
    return np.column_stack([X, W])


def _ols_predict(Z_train, y_train, Z_pred):
    """OLS with intercept: fit on ``Z_train`` and predict on ``Z_pred``."""
    n_train = Z_train.shape[0]
    if n_train == 0:
        raise ValueError("both treatment groups must be present")
    design = np.column_stack([np.ones(n_train), Z_train])
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    design_pred = np.column_stack([np.ones(Z_pred.shape[0]), Z_pred])
    return design_pred @ coef


def difference_in_means(treatment, outcome) -> float:
    """Difference in observed means between treated and untreated."""
    treatment = _validate_treatment(treatment)
    outcome = np.asarray(outcome, dtype=float).ravel()
    if outcome.shape[0] != treatment.shape[0]:
        raise ValueError("treatment and outcome must have the same length")
    if treatment.sum() in (0, treatment.shape[0]):
        raise ValueError("both treatment groups must be present")
    return float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def ipw_weights(X, treatment, propensity=None, stabilized: bool = True) -> np.ndarray:
    """Inverse probability weights, one per unit.

    Raw weights are ``1 / p`` for treated and ``1 / (1 - p)`` for control
    units. Stabilized weights rescale each group by its marginal share,
    ``P(T) / p`` and ``(1 - P(T)) / (1 - p)``, which keeps the weights close
    to one.

    Parameters
    ----------
    X : array-like of shape (n, d)
    treatment : array-like of shape (n,)
    propensity : array-like of shape (n,), optional
    stabilized : bool

    Returns
    -------
    ndarray of shape (n,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    treatment = _validate_treatment(treatment)
    if X.shape[0] != treatment.shape[0]:
        raise ValueError("X and treatment must have the same number of rows")
    if X.shape[0] == 0:
        raise ValueError("at least one row of data is required")
    if propensity is None:
        p, _ = propensity_scores(X, treatment)
    else:
        p = _validate_propensity(propensity, treatment.shape[0])
    treated = treatment == 1
    if not (treated.any() and (~treated).any()):
        raise ValueError("both treatment groups must be present")
    if stabilized:
        p_treat = treatment.mean()
        return np.where(treated, p_treat / p, (1.0 - p_treat) / (1.0 - p))
    return np.where(treated, 1.0 / p, 1.0 / (1.0 - p))


def ipw_ate(
    X,
    treatment,
    outcome,
    propensity=None,
    stabilized: bool = True,
    normalized: bool = True,
) -> float:
    """Inverse probability weighting estimate of the average treatment effect.

    Each unit is weighted by the inverse of its probability of receiving the
    treatment it actually got. When ``stabilized`` is true the marginal
    treatment probability is folded in, which keeps weights closer to one;
    stabilization only makes sense with the normalized (Hajek) form. When
    ``normalized`` is true the estimate is the Hajek difference of weighted
    group means; otherwise it is the Horvitz-Thompson difference of raw
    inverse-probability means.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Covariates used to estimate the propensity score when ``propensity``
        is not supplied.
    treatment : array-like of shape (n,)
    outcome : array-like of shape (n,)
    propensity : array-like of shape (n,), optional
        Pre-computed propensity scores.
    stabilized : bool
    normalized : bool

    Returns
    -------
    float
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    if stabilized and not normalized:
        raise ValueError("stabilized weights require normalized=True")
    weights = ipw_weights(X, treatment, propensity=propensity, stabilized=stabilized)
    treated = treatment == 1
    control = ~treated
    if normalized:
        w_t = weights[treated]
        w_c = weights[control]
        if w_t.sum() <= 0 or w_c.sum() <= 0:
            raise ValueError("IPW weights sum to zero in one of the groups")
        estimate = np.sum(w_t * outcome[treated]) / w_t.sum() - np.sum(w_c * outcome[control]) / w_c.sum()
    else:
        estimate = (
            np.sum(weights[treated] * outcome[treated])
            - np.sum(weights[control] * outcome[control])
        ) / treatment.size
    return float(estimate)


def ipw_att(X, treatment, outcome, propensity=None) -> float:
    """Inverse probability weighting estimate of the effect on the treated.

    Control units are reweighted by the odds ``p / (1 - p)`` so the weighted
    control outcome matches the treated group's counterfactual.

    Parameters
    ----------
    X : array-like of shape (n, d)
    treatment : array-like of shape (n,)
    outcome : array-like of shape (n,)
    propensity : array-like of shape (n,), optional

    Returns
    -------
    float
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    if propensity is None:
        p, _ = propensity_scores(X, treatment)
    else:
        p = _validate_propensity(propensity, treatment.shape[0])
    treated = treatment == 1
    control = ~treated
    if not (treated.any() and control.any()):
        raise ValueError("both treatment groups must be present")
    w_control = p[control] / (1.0 - p[control])
    if w_control.sum() <= 0:
        raise ValueError("ATT control weights sum to zero")
    return float(
        outcome[treated].mean() - np.sum(w_control * outcome[control]) / np.sum(w_control)
    )


def outcome_regression(X, treatment, outcome, W=None):
    """Separate OLS outcome models for the treated and control groups.

    Fits ``E[Y | T=1, Z]`` and ``E[Y | T=0, Z]`` by least squares with an
    intercept, then predicts both regressions for every unit. ``Z`` is ``X``
    concatenated with ``W`` when ``W`` is supplied. Outcome-only covariates
    ``W`` belong here rather than in the propensity score: they predict
    ``Y`` but not treatment assignment.

    Parameters
    ----------
    X : array-like of shape (n, d)
    treatment : array-like of shape (n,)
    outcome : array-like of shape (n,)
    W : array-like of shape (n, d_w), optional
        Extra outcome covariates stacked onto ``X``.

    Returns
    -------
    mu1, mu0 : ndarray of shape (n,)
        Predicted treated and control potential outcomes.
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    Z = _outcome_features(X, W)
    treated = treatment == 1
    control = ~treated
    if not (treated.any() and control.any()):
        raise ValueError("both treatment groups must be present")
    mu1 = _ols_predict(Z[treated], outcome[treated], Z)
    mu0 = _ols_predict(Z[control], outcome[control], Z)
    return mu1, mu0


def aipw_ate(
    X,
    treatment,
    outcome,
    propensity=None,
    W=None,
    normalized: bool = False,
) -> float:
    """Augmented IPW (doubly robust) estimate of the average treatment effect.

    Combines an outcome regression with inverse-probability weighting. The
    unnormalized form is the sample mean of the uncentered efficient
    influence function

    ``m1(Z) - m0(Z) + T / e(X) * (Y - m1(Z)) - (1 - T) / (1 - e(X)) * (Y - m0(Z))``,

    where ``m_t`` is an OLS regression of ``Y`` on ``(X, W)`` in treatment
    arm ``t`` and ``e(X)`` is the propensity score. The estimator is
    *doubly robust*: it is consistent if either the propensity model or
    both outcome regressions are correctly specified, not necessarily both.

    When ``normalized`` is true the IPW residual corrections are
    Hájek-normalized by the sum of the inverse-propensity weights in each
    arm, which can reduce variance when weights are heavy-tailed.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Confounders used to estimate the propensity score when
        ``propensity`` is not supplied, and as outcome-regression
        covariates.
    treatment : array-like of shape (n,)
    outcome : array-like of shape (n,)
    propensity : array-like of shape (n,), optional
        Pre-computed propensity scores.
    W : array-like of shape (n, d_w), optional
        Extra outcome-only covariates for the regressions ``m_t``.
    normalized : bool

    Returns
    -------
    float
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    if propensity is None:
        p, _ = propensity_scores(X, treatment)
    else:
        p = _validate_propensity(propensity, treatment.shape[0])
    treated = treatment == 1
    control = ~treated
    if not (treated.any() and control.any()):
        raise ValueError("both treatment groups must be present")
    mu1, mu0 = outcome_regression(X, treatment, outcome, W=W)
    residual_1 = treated * (outcome - mu1) / p
    residual_0 = (~treated) * (outcome - mu0) / (1.0 - p)
    if normalized:
        w1 = treated / p
        w0 = (~treated) / (1.0 - p)
        if w1.sum() <= 0 or w0.sum() <= 0:
            raise ValueError("AIPW weights sum to zero in one of the groups")
        mu1_hat = mu1.mean() + residual_1.sum() / w1.sum()
        mu0_hat = mu0.mean() + residual_0.sum() / w0.sum()
        return float(mu1_hat - mu0_hat)
    return float(np.mean(mu1 - mu0 + residual_1 - residual_0))


def _nearest_k(sorted_vals, target, k, available=None, caliper=None):
    """Indices (in the sorted array) of the k values nearest to ``target``.

    Distances are scanned outward from the insertion point of ``target``, so
    the returned positions are ordered by distance. ``available`` is an
    optional boolean mask over the sorted positions; masked-out positions are
    skipped. ``caliper`` stops the scan once the distance exceeds the limit.
    """
    pos = np.searchsorted(sorted_vals, target)
    left = pos - 1
    right = pos
    found = []
    while len(found) < k and (left >= 0 or right < sorted_vals.size):
        d_left = abs(sorted_vals[left] - target) if left >= 0 else np.inf
        d_right = abs(sorted_vals[right] - target) if right < sorted_vals.size else np.inf
        if d_left <= d_right:
            position = left
            left -= 1
        else:
            position = right
            right += 1
        if available is not None and not available[position]:
            continue
        if caliper is not None and abs(sorted_vals[position] - target) > caliper:
            break
        found.append(position)
    return found


def _match_att(logit, y, treated, control, caliper, with_replacement, n_neighbors):
    order = treated[np.argsort(logit[treated])[::-1]]
    sorted_pos = np.argsort(logit[control])
    sorted_logit = logit[control][sorted_pos]
    sorted_y = y[control][sorted_pos]
    if with_replacement and n_neighbors == 1:
        return _match_att_vectorized(logit[order], y[order], sorted_logit, sorted_y, caliper)
    contributions = []
    if with_replacement:
        for ti in order:
            nearest = _nearest_k(sorted_logit, logit[ti], n_neighbors, caliper=caliper)
            if not nearest:
                continue
            contributions.append(y[ti] - np.mean(sorted_y[nearest]))
    else:
        available = np.ones(control.size, dtype=bool)
        for ti in order:
            nearest = _nearest_k(
                sorted_logit, logit[ti], n_neighbors, available=available, caliper=caliper
            )
            if not nearest:
                continue
            available[nearest] = False
            contributions.append(y[ti] - np.mean(sorted_y[nearest]))
    if not contributions:
        raise ValueError("no treated units could be matched (check the caliper)")
    return float(np.mean(contributions))


def _match_att_vectorized(target_logit, target_y, sorted_logit, sorted_y, caliper):
    """Nearest control per treated unit via searchsorted, no replacement."""
    size = sorted_logit.size
    pos = np.searchsorted(sorted_logit, target_logit)
    left = pos - 1
    right = pos
    left_ok = left >= 0
    right_ok = right < size
    d_left = np.where(left_ok, np.abs(sorted_logit[np.clip(left, 0, size - 1)] - target_logit), np.inf)
    d_right = np.where(right_ok, np.abs(sorted_logit[np.clip(right, 0, size - 1)] - target_logit), np.inf)
    best = np.where(d_left <= d_right, left, right)
    best_d = np.minimum(d_left, d_right)
    keep = np.ones(target_logit.size, dtype=bool)
    if caliper is not None:
        keep = best_d <= caliper
    if not keep.any():
        raise ValueError("no treated units could be matched (check the caliper)")
    best = np.clip(best, 0, size - 1)
    differences = target_y - sorted_y[best]
    return float(np.mean(differences[keep]))


def _match_ate(logit, y, treated, control, caliper, n_neighbors):
    is_treated = np.zeros(logit.shape[0], dtype=bool)
    is_treated[treated] = True
    c_pos = np.argsort(logit[control])
    t_pos = np.argsort(logit[treated])
    sorted_logit_c = logit[control][c_pos]
    sorted_y_c = y[control][c_pos]
    sorted_logit_t = logit[treated][t_pos]
    sorted_y_t = y[treated][t_pos]
    n = logit.shape[0]
    y1_hat = np.empty(n)
    y0_hat = np.empty(n)
    for i in range(n):
        if is_treated[i]:
            nearest = _nearest_k(sorted_logit_c, logit[i], n_neighbors, caliper=caliper)
            if not nearest:
                raise ValueError("no matches found within the caliper")
            y0_hat[i] = np.mean(sorted_y_c[nearest])
            y1_hat[i] = y[i]
        else:
            nearest = _nearest_k(sorted_logit_t, logit[i], n_neighbors, caliper=caliper)
            if not nearest:
                raise ValueError("no matches found within the caliper")
            y1_hat[i] = np.mean(sorted_y_t[nearest])
            y0_hat[i] = y[i]
    return float(np.mean(y1_hat - y0_hat))


def propensity_matching(
    X,
    treatment,
    outcome,
    propensity=None,
    estimand: str = "att",
    caliper: float | None = None,
    with_replacement: bool = False,
    n_neighbors: int = 1,
) -> float:
    """Nearest-neighbor propensity-score matching.

    Treated units are processed from highest to lowest propensity so that
    hard-to-match units pick controls first. Distance is measured in logit
    propensity space. Without replacement each control is used at most once;
    with replacement controls may be reused.

    Parameters
    ----------
    X : array-like of shape (n, d)
    treatment : array-like of shape (n,)
    outcome : array-like of shape (n,)
    propensity : array-like of shape (n,), optional
        Pre-computed propensity scores.
    estimand : {"att", "ate"}
        ``"att"`` matches treated units to controls. ``"ate"`` imputes the
        counterfactual outcome for every unit from its nearest opposite-group
        matches and requires ``with_replacement=True``.
    caliper : float, optional
        Maximum allowed logit-propensity distance. Units with no match inside
        the caliper are dropped; a value that drops every unit raises an error.
    with_replacement : bool
    n_neighbors : int
        Number of matches per unit.

    Returns
    -------
    float
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    if estimand not in ("att", "ate"):
        raise ValueError("estimand must be 'att' or 'ate'")
    if estimand == "ate" and not with_replacement:
        raise ValueError("ATE via matching requires with_replacement=True")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if propensity is None:
        p, _ = propensity_scores(X, treatment)
    else:
        p = _validate_propensity(propensity, treatment.shape[0])
    logit = np.log(p / (1.0 - p))
    treated = np.flatnonzero(treatment == 1)
    control = np.flatnonzero(treatment == 0)
    if treated.size == 0 or control.size == 0:
        raise ValueError("both treatment groups must be present")
    if estimand == "att":
        return _match_att(logit, outcome, treated, control, caliper, with_replacement, n_neighbors)
    return _match_ate(logit, outcome, treated, control, caliper, n_neighbors)


@dataclass
class DifferenceInDifferencesResult:
    """Two-period or two-way fixed-effects difference-in-differences fit.

    Attributes
    ----------
    estimate : float
        ATT under parallel trends: the two-by-two interaction, or the
        coefficient on the treatment indicator in a two-way fixed-effects
        regression.
    se : float
        Standard error. Clustered by unit when units are supplied and
        ``cluster`` is true; otherwise heteroskedasticity-robust (four-cell
        or HC1).
    method : {"2x2", "twfe"}
        ``"2x2"`` when there are two periods; ``"twfe"`` when more than two
        periods are used with unit and time fixed effects.
    se_type : {"cluster", "hc1"}
        Variance estimator.
    n : int
        Number of observations.
    n_units : int or None
        Number of distinct units, or ``None`` when ``unit`` was omitted.
    n_times : int
        Number of distinct time periods.
    n_clusters : int or None
        Number of clusters used for the SE, or ``None`` for HC1.
    n_treated_post : int
        Observations with treatment switched on.
    treated_mean_pre, treated_mean_post : float or None
        Cell means in a two-period design; ``None`` for multi-period TWFE.
    control_mean_pre, control_mean_post : float or None
        Control-group cell means in a two-period design.
    """

    estimate: float
    se: float
    method: str
    se_type: str
    n: int
    n_units: int | None
    n_times: int
    n_clusters: int | None
    n_treated_post: int
    treated_mean_pre: float | None = None
    treated_mean_post: float | None = None
    control_mean_pre: float | None = None
    control_mean_post: float | None = None


def _did_cell_mean_var(values: np.ndarray):
    if values.size == 0:
        raise ValueError("one of the group-period cells is empty")
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0, int(values.size)
    return mean, float(values.var(ddof=1) / values.size), int(values.size)


def _two_by_two_cells(group, period, outcome):
    """Four-cell means, observation-robust SE, and the 2x2 ATT."""
    times = np.unique(period)
    if times.size != 2:
        raise ValueError("two-period DiD requires exactly two distinct periods")
    pre, post = times[0], times[1]

    def stats(gval, pval):
        return _did_cell_mean_var(outcome[(group == gval) & (period == pval)])

    m11, v11, n11 = stats(1, post)
    m10, v10, n10 = stats(1, pre)
    m01, v01, n01 = stats(0, post)
    m00, v00, n00 = stats(0, pre)
    estimate = m11 - m10 - m01 + m00
    se = float(np.sqrt(max(v11 + v10 + v01 + v00, 0.0)))
    return {
        "estimate": float(estimate),
        "se": se,
        "treated_mean_pre": m10,
        "treated_mean_post": m11,
        "control_mean_pre": m00,
        "control_mean_post": m01,
        "n_treated_post": n11,
        "pre": pre,
        "post": post,
    }


def _group_constant_within_unit(group, unit_index, n_units):
    gmin = np.full(n_units, np.inf)
    gmax = np.full(n_units, -np.inf)
    np.minimum.at(gmin, unit_index, group)
    np.maximum.at(gmax, unit_index, group)
    if np.any(gmin != gmax):
        raise ValueError("group must be constant within unit")


def _balanced_first_diff(unit, group, period, outcome):
    """Unit-level first-difference ATT and clustered SE, or ``None``."""
    unit_ids, unit_index = np.unique(unit, return_inverse=True)
    n_units = unit_ids.size
    counts = np.bincount(unit_index, minlength=n_units)
    if n_units < 2 or not np.all(counts == 2):
        return None
    times = np.unique(period)
    if times.size != 2:
        return None
    group = np.asarray(group, dtype=float)
    _group_constant_within_unit(group, unit_index, n_units)
    order = np.lexsort((period, unit_index))
    p_sorted = period[order]
    pre, post = times[0], times[1]
    if not (np.all(p_sorted[0::2] == pre) and np.all(p_sorted[1::2] == post)):
        return None
    y_sorted = outcome[order]
    g_sorted = group[order]
    delta = y_sorted[1::2] - y_sorted[0::2]
    g = g_sorted[0::2]
    treated = g == 1
    if not treated.any() or not (~treated).any():
        raise ValueError("both treatment groups must be present")
    d1 = delta[treated]
    d0 = delta[~treated]
    estimate = float(d1.mean() - d0.mean())
    v1 = 0.0 if d1.size < 2 else float(d1.var(ddof=1) / d1.size)
    v0 = 0.0 if d0.size < 2 else float(d0.var(ddof=1) / d0.size)
    se = float(np.sqrt(max(v1 + v0, 0.0)))
    return estimate, se, n_units


def _demean_within(values, unit_index, n_units, counts):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        sums = np.bincount(unit_index, weights=values, minlength=n_units)
        return values - (sums / counts)[unit_index]
    out = np.empty_like(values, dtype=float)
    for j in range(values.shape[1]):
        sums = np.bincount(unit_index, weights=values[:, j], minlength=n_units)
        out[:, j] = values[:, j] - (sums / counts)[unit_index]
    return out


def _sandwich_se(X, resid, cluster_index=None):
    """HC1 or cluster-robust SE of the last OLS coefficient."""
    n, k = X.shape
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    if cluster_index is None:
        meat = (X * (resid ** 2)[:, None]).T @ X
        scale = n / max(n - k, 1)
        n_clusters = None
    else:
        n_clusters = int(np.max(cluster_index)) + 1
        scores = np.zeros((n_clusters, k))
        np.add.at(scores, cluster_index, X * resid[:, None])
        meat = scores.T @ scores
        g = n_clusters
        scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    variance = float(np.real((scale * (xtx_inv @ meat @ xtx_inv))[-1, -1]))
    if not np.isfinite(variance):
        variance = 0.0
    return float(np.sqrt(max(variance, 0.0))), n_clusters


def _twfe(unit, time, treatment, outcome, cluster: bool):
    """Unit and time FE regression of ``outcome`` on ``treatment``."""
    unit_ids, unit_index = np.unique(unit, return_inverse=True)
    time_ids, time_index = np.unique(time, return_inverse=True)
    n_units = unit_ids.size
    n_times = time_ids.size
    if n_times < 2:
        raise ValueError("at least two time periods are required")
    if n_units < 2:
        raise ValueError("at least two units are required")
    counts = np.bincount(unit_index, minlength=n_units).astype(float)
    y = _demean_within(outcome, unit_index, n_units, counts)
    time_dummies = np.eye(n_times, dtype=float)[time_index][:, 1:]
    d = np.asarray(treatment, dtype=float)
    X = np.column_stack(
        [
            _demean_within(time_dummies, unit_index, n_units, counts),
            _demean_within(d, unit_index, n_units, counts),
        ]
    )
    if X.shape[1] > 1:
        others = X[:, :-1]
        d_resid = X[:, -1] - others @ np.linalg.lstsq(others, X[:, -1], rcond=None)[0]
    else:
        d_resid = X[:, -1]
    if float(np.dot(d_resid, d_resid)) < 1e-12:
        raise ValueError("treatment is collinear with unit and time fixed effects")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    se, n_clusters = _sandwich_se(
        X, resid, cluster_index=unit_index if cluster else None
    )
    return float(coef[-1]), se, n_units, n_times, n_clusters, int((d == 1).sum())


def difference_in_differences(
    group,
    period,
    outcome,
    unit=None,
    treatment=None,
    treatment_time=None,
    cluster: bool | None = None,
) -> DifferenceInDifferencesResult:
    """Difference-in-differences ATT with clustered or robust standard errors.

    In the two-period, two-group design the estimate is

    ``(Ybar_treated_post - Ybar_treated_pre) - (Ybar_control_post -
    Ybar_control_pre)``.

    When ``unit`` is supplied and each unit is observed once in each of two
    periods, the same ATT is formed from unit-level first differences and
    the standard error is clustered at the unit (the usual panel DiD SE).
    Without unit identifiers the four cells are treated as independent
    samples and the SE is heteroskedasticity-robust.

    With more than two periods the estimator is two-way fixed effects:
    ``Y_it = a_i + b_t + tau D_it + e_it``, with ``D_it`` equal to
    ``treatment`` when given, otherwise ``group * 1{period >= treatment_time}``.
    The coefficient ``tau`` is the ATT under parallel trends and canonical
    (simultaneous) adoption. Staggered adoption can produce negative TWFE
    weights; pass a two-period panel or a canonical post indicator instead.

    Parameters
    ----------
    group : array-like of shape (n,)
        Group indicator, 1 = treated (ever-treated). Must be constant
        within ``unit`` when units are supplied.
    period : array-like of shape (n,)
        Period identifier. In a two-period design the later period is
        treated as post unless ``treatment_time`` is given.
    outcome : array-like of shape (n,)
    unit : array-like of shape (n,), optional
        Unit identifier. Enables unit fixed effects / first differences
        and cluster-robust standard errors.
    treatment : array-like of shape (n,), optional
        Treatment status ``D_it``. If omitted, ``D_it`` is
        ``group * 1{post}`` for two periods, or
        ``group * 1{period >= treatment_time}`` for multiple periods.
    treatment_time : optional
        First post-treatment period when ``treatment`` is omitted.
    cluster : bool, optional
        If true, cluster the SE by ``unit``. If omitted, cluster when
        ``unit`` is supplied and use HC1 otherwise.

    Returns
    -------
    DifferenceInDifferencesResult
    """
    group = _validate_treatment(group)
    period = np.asarray(period).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    if not (group.shape[0] == period.shape[0] == outcome.shape[0]):
        raise ValueError("group, period and outcome must have the same length")
    if group.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(outcome)):
        raise ValueError("outcome must be finite")
    if unit is not None:
        unit = np.asarray(unit).ravel()
        if unit.shape[0] != group.shape[0]:
            raise ValueError("unit must have the same length as group")
    if cluster is None:
        cluster = unit is not None
    if cluster and unit is None:
        raise ValueError("clustered standard errors require unit identifiers")

    times = np.unique(period)
    n_times = int(times.size)
    if n_times < 2:
        raise ValueError("at least two time periods are required")

    if treatment is not None:
        treatment = _validate_treatment(treatment)
        if treatment.shape[0] != group.shape[0]:
            raise ValueError("treatment must have the same length as group")
    elif n_times == 2 and treatment_time is None:
        treatment = ((group == 1) & (period == times[-1])).astype(float)
    elif treatment_time is not None:
        treatment = ((group == 1) & (period >= treatment_time)).astype(float)
    else:
        raise ValueError(
            "multi-period DiD requires treatment or treatment_time when there "
            "are more than two distinct periods"
        )
    if not ((treatment == 1).any() and (treatment == 0).any()):
        raise ValueError("both treated and untreated observations must be present")

    n = int(group.shape[0])
    cell = None
    cell_error = None
    if n_times == 2:
        try:
            cell = _two_by_two_cells(group, period, outcome)
        except ValueError as exc:
            cell_error = exc

    if unit is None:
        if cell is not None:
            return DifferenceInDifferencesResult(
                estimate=cell["estimate"],
                se=cell["se"],
                method="2x2",
                se_type="hc1",
                n=n,
                n_units=None,
                n_times=n_times,
                n_clusters=None,
                n_treated_post=int((treatment == 1).sum()),
                treated_mean_pre=cell["treated_mean_pre"],
                treated_mean_post=cell["treated_mean_post"],
                control_mean_pre=cell["control_mean_pre"],
                control_mean_post=cell["control_mean_post"],
            )
        if cell_error is not None:
            raise cell_error
        raise ValueError("unit identifiers are required for multi-period TWFE")

    method = "2x2" if n_times == 2 else "twfe"
    first_diff = None
    if n_times == 2 and cluster:
        first_diff = _balanced_first_diff(unit, group, period, outcome)

    if first_diff is not None:
        estimate, se, n_units = first_diff
        n_clusters = n_units
        se_type = "cluster"
        n_treated_post = int((treatment == 1).sum())
    else:
        estimate, se, n_units, n_times_fit, n_clusters, n_treated_post = _twfe(
            unit, period, treatment, outcome, cluster=cluster
        )
        n_times = n_times_fit
        se_type = "cluster" if cluster else "hc1"
        if not cluster:
            n_clusters = None

    return DifferenceInDifferencesResult(
        estimate=estimate,
        se=se,
        method=method,
        se_type=se_type,
        n=n,
        n_units=int(n_units),
        n_times=int(n_times),
        n_clusters=None if n_clusters is None else int(n_clusters),
        n_treated_post=n_treated_post,
        treated_mean_pre=None if cell is None else cell["treated_mean_pre"],
        treated_mean_post=None if cell is None else cell["treated_mean_post"],
        control_mean_pre=None if cell is None else cell["control_mean_pre"],
        control_mean_post=None if cell is None else cell["control_mean_post"],
    )


@dataclass
class SyntheticControlResult:
    """Abadie-style synthetic control fit for a single treated unit.

    Attributes
    ----------
    estimate : float
        Average post-treatment gap between the treated unit and the
        synthetic control.
    weights : ndarray of shape (n_donors,)
        Non-negative donor weights summing to 1, aligned with ``donor_ids``.
    donor_ids : ndarray of shape (n_donors,)
        Unit identifiers of the donor pool, in sorted order.
    times : ndarray of shape (n_times,)
        Sorted period identifiers.
    treated_outcome : ndarray of shape (n_times,)
        Observed outcome path of the treated unit.
    synthetic_outcome : ndarray of shape (n_times,)
        Donor-weighted synthetic path.
    gap : ndarray of shape (n_times,)
        ``treated_outcome - synthetic_outcome`` in every period.
    pre_rmspe : float
        Root mean squared pre-treatment gap.
    post_rmspe : float
        Root mean squared post-treatment gap.
    placebo_p_value : float or None
        In-space placebo rank p-value, or ``None`` when placebos were not run.
    placebo_estimates : ndarray or None
        Average post-treatment gap for each donor placebo.
    placebo_ratios : ndarray or None
        Post/pre RMSPE ratios for each donor placebo.
    """

    estimate: float
    weights: np.ndarray
    donor_ids: np.ndarray
    times: np.ndarray
    treated_outcome: np.ndarray
    synthetic_outcome: np.ndarray
    gap: np.ndarray
    pre_rmspe: float
    post_rmspe: float
    placebo_p_value: float | None = None
    placebo_estimates: np.ndarray | None = None
    placebo_ratios: np.ndarray | None = None


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, sum(w) = 1}`` (Duchi et al.)."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        raise ValueError("at least one donor unit is required")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    total = w.sum()
    if total <= 0:
        return np.ones(n) / n
    return w / total


def _fit_simplex_weights(X0: np.ndarray, x1: np.ndarray, max_iter: int = 20000) -> np.ndarray:
    """Non-negative weights summing to 1 that match ``X0 w`` to ``x1``.

    Solves ``min_{w in simplex} ||X0 w - x1||^2`` by accelerated projected
    gradient descent (FISTA) on the probability simplex. ``X0`` is the
    ``(n_pre, n_donors)`` matrix of donor pre-treatment outcomes and ``x1``
    is the treated unit's pre-treatment path.
    """
    X0 = np.asarray(X0, dtype=float)
    x1 = np.asarray(x1, dtype=float).ravel()
    if X0.ndim != 2:
        raise ValueError("donor pre-treatment outcomes must be a 2d array")
    n_pre, n_donors = X0.shape
    if n_donors < 1:
        raise ValueError("at least one donor unit is required")
    if n_pre < 1:
        raise ValueError("at least one pre-treatment period is required")
    if x1.shape[0] != n_pre:
        raise ValueError("treated and donor pre-treatment lengths differ")

    lipschitz = float(np.linalg.norm(X0, 2)) ** 2
    if not np.isfinite(lipschitz) or lipschitz <= 1e-18:
        return np.ones(n_donors) / n_donors
    step = 1.0 / lipschitz
    gram = X0.T @ X0
    xtx1 = X0.T @ x1

    w = np.ones(n_donors) / n_donors
    z = w.copy()
    t = 1.0
    for _ in range(max_iter):
        grad = gram @ z - xtx1
        w_new = _project_simplex(z - step * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < 1e-12:
            w = w_new
            break
        w, t = w_new, t_new
    w = np.maximum(w, 0.0)
    return w / w.sum()


def _rmspe(gap: np.ndarray) -> float:
    gap = np.asarray(gap, dtype=float).ravel()
    if gap.size == 0:
        raise ValueError("cannot compute RMSPE of an empty gap")
    return float(np.sqrt(np.mean(gap ** 2)))


def _rmspe_ratio(pre_gap: np.ndarray, post_gap: np.ndarray) -> float:
    pre = _rmspe(pre_gap)
    post = _rmspe(post_gap)
    if pre < 1e-15:
        return float(np.inf) if post > 1e-15 else 0.0
    return post / pre


def _panel_from_long(unit, time, outcome):
    """Pivot a long panel into a balanced ``(n_units, n_times)`` matrix."""
    unit = np.asarray(unit)
    time = np.asarray(time)
    outcome = np.asarray(outcome, dtype=float).ravel()
    if not (unit.shape[0] == time.shape[0] == outcome.shape[0]):
        raise ValueError("unit, time and outcome must have the same length")
    if unit.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(outcome)):
        raise ValueError("outcome must be finite")

    unit_ids, unit_index = np.unique(unit, return_inverse=True)
    time_ids, time_index = np.unique(time, return_inverse=True)
    n_units = unit_ids.size
    n_times = time_ids.size
    n_unique = np.unique(np.column_stack([unit_index, time_index]), axis=0).shape[0]
    if n_unique < unit.shape[0]:
        raise ValueError("duplicate unit-time observations")
    if n_unique != n_units * n_times:
        raise ValueError("unbalanced panel: every unit must be observed at every time")

    panel = np.empty((n_units, n_times), dtype=float)
    panel[unit_index, time_index] = outcome
    return panel, unit_ids, time_ids


def _match_id(ids: np.ndarray, value):
    matches = np.flatnonzero(ids == value)
    if matches.size == 0 and np.issubdtype(ids.dtype, np.number):
        try:
            matches = np.flatnonzero(ids == type(ids[0])(value))
        except (TypeError, ValueError):
            matches = np.array([], dtype=int)
    if matches.size != 1:
        raise ValueError(f"treated_unit {value!r} must identify exactly one unit")
    return int(matches[0])


def synthetic_control(
    unit,
    time,
    outcome,
    treated_unit,
    treatment_time,
    placebo: bool = False,
) -> SyntheticControlResult:
    """Abadie synthetic control for a single treated unit.

    Donor weights are constrained to the probability simplex (non-negative
    and summing to one) and chosen to match the treated unit's
    pre-treatment outcomes. The estimate is the average post-treatment
    gap between the treated series and the synthetic series
    ``sum_j w_j Y_jt``.

    When ``placebo`` is true, the same procedure is applied to every
    donor (leaving the originally treated unit out of the donor pool).
    The reported p-value is the rank of the treated unit's post/pre
    RMSPE ratio among these in-space placebos,
    ``(1 + #{placebos with ratio >= treated}) / (1 + n_donors)``.

    Parameters
    ----------
    unit : array-like of shape (n,)
        Unit identifier for each observation.
    time : array-like of shape (n,)
        Period identifier. Compared against ``treatment_time``.
    outcome : array-like of shape (n,)
    treated_unit :
        Identifier of the single treated unit.
    treatment_time :
        First post-treatment period. Periods strictly before this value
        are used to fit the weights.
    placebo : bool
        If true, run in-space placebos on every donor.

    Returns
    -------
    SyntheticControlResult
    """
    panel, unit_ids, time_ids = _panel_from_long(unit, time, outcome)
    treated_index = _match_id(unit_ids, treated_unit)
    pre_mask = time_ids < treatment_time
    post_mask = time_ids >= treatment_time
    if not pre_mask.any():
        raise ValueError("at least one pre-treatment period is required")
    if not post_mask.any():
        raise ValueError("at least one post-treatment period is required")

    donor_index = np.flatnonzero(np.arange(unit_ids.size) != treated_index)
    if donor_index.size < 1:
        raise ValueError("at least one donor unit is required")
    if placebo and donor_index.size < 2:
        raise ValueError("placebo checks require at least two donor units")

    treated_y = panel[treated_index]
    donor_y = panel[donor_index]
    weights = _fit_simplex_weights(donor_y[:, pre_mask].T, treated_y[pre_mask])
    synthetic = weights @ donor_y
    gap = treated_y - synthetic
    estimate = float(gap[post_mask].mean())
    pre_rmspe = _rmspe(gap[pre_mask])
    post_rmspe = _rmspe(gap[post_mask])

    placebo_p_value = None
    placebo_estimates = None
    placebo_ratios = None
    if placebo:
        treated_ratio = _rmspe_ratio(gap[pre_mask], gap[post_mask])
        placebo_estimates = np.empty(donor_index.size)
        placebo_ratios = np.empty(donor_index.size)
        for k, j in enumerate(donor_index):
            others = np.delete(donor_index, k)
            p_weights = _fit_simplex_weights(
                panel[others][:, pre_mask].T, panel[j, pre_mask]
            )
            p_synthetic = p_weights @ panel[others]
            p_gap = panel[j] - p_synthetic
            placebo_estimates[k] = p_gap[post_mask].mean()
            placebo_ratios[k] = _rmspe_ratio(p_gap[pre_mask], p_gap[post_mask])
        n_extreme = int(np.sum(placebo_ratios >= treated_ratio))
        placebo_p_value = (1.0 + n_extreme) / (1.0 + donor_index.size)

    return SyntheticControlResult(
        estimate=estimate,
        weights=weights,
        donor_ids=unit_ids[donor_index],
        times=time_ids,
        treated_outcome=treated_y,
        synthetic_outcome=synthetic,
        gap=gap,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        placebo_p_value=placebo_p_value,
        placebo_estimates=placebo_estimates,
        placebo_ratios=placebo_ratios,
    )


_RD_KERNELS = ("triangular", "uniform", "epanechnikov")
_IK_KERNEL_CONSTANT = {
    "triangular": 3.43754,
    "uniform": 5.40384,
    "epanechnikov": 3.1999,
}


@dataclass
class RegressionDiscontinuityResult:
    """Sharp local-linear regression-discontinuity fit at a cutoff.

    Attributes
    ----------
    estimate : float
        Jump in the conditional mean at the cutoff: the treatment effect
        for units at the threshold. Equal to ``intercept_right -
        intercept_left`` when treated units sit above the cutoff, and
        the opposite when ``treated_above`` is false.
    cutoff : float
        Threshold used to assign treatment.
    bandwidth : float
        Half-width of the window around the cutoff. Observations with
        ``|running - cutoff|`` larger than this (or with zero kernel
        weight) are dropped.
    kernel : str
        Kernel used for the local-linear weights.
    n_left : int
        Number of control-side observations with positive kernel weight.
    n_right : int
        Number of treated-side observations with positive kernel weight.
    intercept_left : float
        Local-linear intercept approaching the cutoff from the left.
    intercept_right : float
        Local-linear intercept approaching the cutoff from the right.
    slope_left : float
        Local-linear slope on the left of the cutoff.
    slope_right : float
        Local-linear slope on the right of the cutoff.
    se : float
        Conventional homoskedastic standard error of the jump.
    treated_above : bool
        If true, units with ``running >= cutoff`` are treated.
    """

    estimate: float
    cutoff: float
    bandwidth: float
    kernel: str
    n_left: int
    n_right: int
    intercept_left: float
    intercept_right: float
    slope_left: float
    slope_right: float
    se: float
    treated_above: bool = True


def _coerce_rd_arrays(running, outcome):
    running = np.asarray(running, dtype=float).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    if running.shape[0] != outcome.shape[0]:
        raise ValueError("running and outcome must have the same length")
    if running.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(running)):
        raise ValueError("running variable must be finite")
    if not np.all(np.isfinite(outcome)):
        raise ValueError("outcome must be finite")
    return running, outcome


def _normalize_rd_kernel(kernel: str) -> str:
    if kernel == "rectangular":
        kernel = "uniform"
    if kernel not in _RD_KERNELS:
        raise ValueError(f"kernel must be one of {list(_RD_KERNELS)}")
    return kernel


def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel weights on the scaled running variable ``u = (R - c) / h``."""
    u = np.asarray(u, dtype=float).ravel()
    abs_u = np.abs(u)
    if kernel == "triangular":
        return np.where(abs_u < 1.0, 1.0 - abs_u, 0.0)
    if kernel == "uniform":
        return np.where(abs_u <= 1.0, 1.0, 0.0)
    return np.where(abs_u <= 1.0, 0.75 * (1.0 - u * u), 0.0)


def _weighted_local_linear(x, y, cutoff, weights):
    """Intercept, slope and intercept variance from weighted local linear."""
    n = x.shape[0]
    if n < 2:
        raise ValueError("each side of the cutoff needs at least two observations")
    z = x - cutoff
    design = np.column_stack([np.ones(n), z])
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    resid = y - design @ coef
    n_pos = int(np.sum(weights > 0))
    if n_pos < 2:
        raise ValueError("each side of the cutoff needs at least two observations")
    if n_pos == 2:
        sigma2 = 0.0
    else:
        sigma2 = float(np.sum(weights * resid ** 2) / (n_pos - 2))
    xtwx = design.T @ (weights[:, None] * design)
    try:
        inv = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(xtwx)
    var_intercept = float(max(sigma2 * inv[0, 0], 0.0))
    return float(coef[0]), float(coef[1]), var_intercept


def _ik_bandwidth(running, outcome, cutoff, kernel: str) -> float:
    """Imbens–Kalyanaraman (2012) MSE-optimal bandwidth for local linear RD.

    Follows the practical selector in IK (Review of Economic Studies) as
    implemented by the ``rdd`` R package: a uniform-kernel pilot for the
    density and residual variance, a cubic for the third derivative, local
    quadratics for the second derivatives, and a regularization term that
    keeps the denominator away from zero when the estimated curvatures
    nearly cancel.
    """
    x = np.asarray(running, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    n = x.size
    sx = float(np.std(x, ddof=1))
    if not np.isfinite(sx) or sx <= 0:
        raise ValueError("running variable has no variation")

    h1 = 1.84 * sx * n ** (-0.2)
    left_pilot = (x >= cutoff - h1) & (x <= cutoff)
    right_pilot = (x > cutoff) & (x <= cutoff + h1)
    n_lp = int(left_pilot.sum())
    n_rp = int(right_pilot.sum())
    if n_lp < 1 or n_rp < 1:
        raise ValueError("insufficient data near the cutoff to choose a bandwidth")

    fbar = (n_lp + n_rp) / (2.0 * n * h1)
    if fbar <= 0:
        raise ValueError("insufficient data near the cutoff to choose a bandwidth")
    var_y = (
        np.sum((y[left_pilot] - y[left_pilot].mean()) ** 2)
        + np.sum((y[right_pilot] - y[right_pilot].mean()) ** 2)
    ) / (n_lp + n_rp)

    left_all = x <= cutoff
    right_all = x > cutoff
    if int(left_all.sum()) < 2 or int(right_all.sum()) < 2:
        raise ValueError("both sides of the cutoff must have observations")
    med_left = float(np.median(x[left_all]))
    med_right = float(np.median(x[right_all]))
    cubic_mask = (x >= med_left) & (x <= med_right)
    if int((x[left_pilot] > med_left).sum()) == 0 or int((x[right_pilot] < med_right).sum()) == 0:
        raise ValueError("insufficient data near the cutoff to choose a bandwidth")
    if int(cubic_mask.sum()) < 5:
        raise ValueError("insufficient data near the cutoff to choose a bandwidth")

    z = x - cutoff
    dummy = (x >= cutoff).astype(float)
    cubic_design = np.column_stack([np.ones(n), dummy, z, z ** 2, z ** 3])
    cubic_coef, *_ = np.linalg.lstsq(cubic_design[cubic_mask], y[cubic_mask], rcond=None)
    m3 = 6.0 * float(cubic_coef[4])
    m3_sq = max(m3 ** 2, 0.01)

    n_left_all = float((x < cutoff).sum())
    n_right_all = float((x >= cutoff).sum())
    h2_l = 3.56 * (n_left_all ** (-1.0 / 7.0)) * (var_y / (fbar * m3_sq)) ** (1.0 / 7.0)
    h2_r = 3.56 * (n_right_all ** (-1.0 / 7.0)) * (var_y / (fbar * m3_sq)) ** (1.0 / 7.0)

    left_h2 = (x >= cutoff - h2_l) & (x < cutoff)
    right_h2 = (x >= cutoff) & (x <= cutoff + h2_r)
    n_l2 = int(left_h2.sum())
    n_r2 = int(right_h2.sum())
    if n_l2 < 3 or n_r2 < 3:
        raise ValueError("insufficient data near the cutoff to choose a bandwidth")

    def _quadratic_second_deriv(mask):
        zz = z[mask]
        design = np.column_stack([np.ones(mask.sum()), zz, zz ** 2])
        coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        return 2.0 * float(coef[2])

    m2_l = _quadratic_second_deriv(left_h2)
    m2_r = _quadratic_second_deriv(right_h2)
    r_l = 720.0 * var_y / (n_l2 * h2_l ** 4)
    r_r = 720.0 * var_y / (n_r2 * h2_r ** 4)
    denom = fbar * ((m2_r - m2_l) ** 2 + r_l + r_r)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError("automatic bandwidth selector failed")
    ck = _IK_KERNEL_CONSTANT[kernel]
    bandwidth = ck * (2.0 * var_y / denom) ** 0.2 * n ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("automatic bandwidth selector failed")
    return float(bandwidth)


def regression_discontinuity(
    running,
    outcome,
    cutoff: float = 0.0,
    bandwidth=None,
    kernel: str = "triangular",
    treatment=None,
    treated_above: bool = True,
) -> RegressionDiscontinuityResult:
    """Sharp local-linear regression discontinuity at a cutoff.

    Treatment is a deterministic function of a running variable: units
    with ``running >= cutoff`` are treated when ``treated_above`` is
    true (the usual score-above-threshold design). Within a bandwidth
    ``h`` the estimator fits kernel-weighted linear regressions of the
    outcome on ``(running - cutoff)`` separately on each side. The
    intercepts are the left and right limits of ``E[Y | running]`` at
    the cutoff; their difference is the treatment effect for units at
    the threshold.

    When ``bandwidth`` is omitted the Imbens–Kalyanaraman (2012)
    MSE-optimal bandwidth is used. The default triangular kernel is the
    MSE-optimal kernel for local linear RD; uniform and Epanechnikov
    kernels are also available.

    The estimand is local. Continuity of the potential-outcome
    conditional means at the cutoff, no manipulation of the running
    variable, and sharp assignment are required; the estimate is not an
    ATE for the whole sample.

    Parameters
    ----------
    running : array-like of shape (n,)
        Running (forcing) variable.
    outcome : array-like of shape (n,)
    cutoff : float
        Treatment threshold.
    bandwidth : float, optional
        Window half-width. If omitted, the IK selector is used.
    kernel : {"triangular", "uniform", "epanechnikov"}
    treatment : array-like of shape (n,), optional
        If given, must match sharp assignment at ``cutoff``.
    treated_above : bool
        If true, ``running >= cutoff`` is the treated side.

    Returns
    -------
    RegressionDiscontinuityResult
    """
    running, outcome = _coerce_rd_arrays(running, outcome)
    kernel = _normalize_rd_kernel(kernel)
    if treatment is not None:
        treatment = _validate_treatment(treatment)
        if treatment.shape[0] != running.shape[0]:
            raise ValueError("treatment must have one entry per running-variable unit")
        assigned = (running >= cutoff) if treated_above else (running < cutoff)
        if not np.array_equal(treatment, assigned.astype(float)):
            raise ValueError(
                "treatment is not a sharp function of the running variable at the cutoff"
            )
    if bandwidth is None:
        bandwidth = _ik_bandwidth(running, outcome, cutoff, kernel)
    else:
        bandwidth = float(bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be a positive finite number")

    weights = _kernel_weights((running - cutoff) / bandwidth, kernel)
    left = (running < cutoff) & (weights > 0)
    right = (running >= cutoff) & (weights > 0)
    if int(left.sum()) < 2 or int(right.sum()) < 2:
        raise ValueError("each side of the cutoff needs at least two observations")

    a_left, b_left, var_left = _weighted_local_linear(
        running[left], outcome[left], cutoff, weights[left]
    )
    a_right, b_right, var_right = _weighted_local_linear(
        running[right], outcome[right], cutoff, weights[right]
    )
    jump = a_right - a_left
    estimate = jump if treated_above else -jump
    return RegressionDiscontinuityResult(
        estimate=float(estimate),
        cutoff=float(cutoff),
        bandwidth=float(bandwidth),
        kernel=kernel,
        n_left=int(left.sum()),
        n_right=int(right.sum()),
        intercept_left=a_left,
        intercept_right=a_right,
        slope_left=b_left,
        slope_right=b_right,
        se=float(np.sqrt(var_left + var_right)),
        treated_above=bool(treated_above),
    )
