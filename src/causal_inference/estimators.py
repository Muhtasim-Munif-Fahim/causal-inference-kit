"""Treatment-effect estimators: IPW, AIPW, propensity matching, difference-in-differences."""

from __future__ import annotations

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


def difference_in_differences(group, period, outcome) -> float:
    """Two-period, two-group difference-in-differences estimate.

    Returns ``(Ybar_treated_post - Ybar_treated_pre) - (Ybar_control_post -
    Ybar_control_pre)``.

    Parameters
    ----------
    group : array-like of shape (n,)
        Group indicator, 1 = treated.
    period : array-like of shape (n,)
        Period indicator, 1 = post.
    outcome : array-like of shape (n,)

    Returns
    -------
    float
    """
    group = _validate_treatment(group)
    period = _validate_treatment(period)
    outcome = np.asarray(outcome, dtype=float).ravel()
    if not (group.shape[0] == period.shape[0] == outcome.shape[0]):
        raise ValueError("group, period and outcome must have the same length")

    def cell_mean(sel):
        values = outcome[sel]
        if values.size == 0:
            raise ValueError("one of the group-period cells is empty")
        return values.mean()

    return float(
        cell_mean((group == 1) & (period == 1))
        - cell_mean((group == 1) & (period == 0))
        - cell_mean((group == 0) & (period == 1))
        + cell_mean((group == 0) & (period == 0))
    )
