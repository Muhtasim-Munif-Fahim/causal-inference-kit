# causal-inference-kit

A small, dependency-light toolkit for estimating treatment effects from
observational data. It includes a synthetic data generator with known
ground-truth effects, a propensity-score model fitted by gradient descent,
IPW / matching / AIPW / difference-in-differences / event-study DiD / synthetic-control /
regression-discontinuity / two-stage least squares estimators, bootstrap
evaluation against the ground truth, and markdown report rendering. Only
`numpy` and `pandas` are required.

## Contents

| Module | Purpose |
| --- | --- |
| `causal_inference.generators` | Synthetic observational data with known ATE, confounding, effect heterogeneity and a hidden-confounding (selection bias) knob; a two-period or multi-period panel for DiD / TWFE; a donor panel for synthetic control; a running-variable sample for sharp RD; an encouragement design for instrumental variables |
| `causal_inference.propensity` | Logistic regression by gradient descent (L2, backtracking), propensity scores, SMD and overlap diagnostics |
| `causal_inference.estimators` | IPW (ATE/ATT, stabilized and Hájek variants), AIPW (doubly robust ATE), nearest-neighbor propensity matching, two-period DiD and optional multi-period TWFE (clustered or robust SE), event-study / dynamic DiD relative-time coefficients, Abadie synthetic control, sharp local-linear regression discontinuity, 2SLS instrumental variables (HC1 SE, robust first-stage F) |
| `causal_inference.evaluate` | Bootstrap bias / RMSE / standard error of a list of estimators against the true effect |
| `causal_inference.report` | Markdown renderer: balance table, point estimates, evaluation summary, assumption caveats |
| `causal_inference.cli` | `simulate`, `estimate` and `report` subcommands |

## Estimators

| Estimator | Estimand | Notes |
| --- | --- | --- |
| Naive difference in means | ATE | Baseline; biased under confounding |
| IPW (stabilized, Hájek) | ATE | Inverse probability weighting with stabilized weights |
| IPW (raw, Horvitz-Thompson) | ATE | Unnormalized inverse-probability difference |
| IPW | ATT | Controls reweighted by the odds |
| AIPW (augmented IPW) | ATE | Doubly robust: OLS outcome regression plus IPW residual correction |
| Propensity matching | ATT | Greedy nearest-neighbor in logit space, with/without replacement, optional caliper |
| Propensity matching (imputation) | ATE | Counterfactual imputation, with replacement |
| Difference-in-differences | ATT (DiD) | Two-period 2x2 or multi-period TWFE; clustered-by-unit or HC1 robust SE |
| Event-study DiD | Dynamic ATT | Relative-time coefficients around adoption; unit and time FE; omit `k=-1` by default |
| Synthetic control | ATT (SC) | Non-negative donor weights summing to 1; pre-treatment fit, post-treatment gap, optional in-space placebo |
| Regression discontinuity (sharp) | LATE at cutoff | Local linear on each side of a threshold; user or Imbens–Kalyanaraman bandwidth |
| Two-stage least squares | LATE / ATT | IV coefficient on an endogenous treatment; HC1 robust SE and robust first-stage F |


## Event-study / dynamic DiD

`event_study_did` estimates relative-time coefficients around a common
adoption date `T0`:

```text
Y_it = a_i + b_t + sum_{k != ref} tau_k * 1{G_i=1, t - T0 = k} + e_it
```

The coefficient at `reference` (default `-1`, the last pre-period) is
normalized to zero. Under parallel trends the leads are zero in expectation
and the lags are dynamic ATTs. This sits alongside `difference_in_differences`,
which reports a single 2x2 or TWFE ATT.

```python
from causal_inference import event_study_did, simulate_did_data

unit, group, period, outcome, true_att = simulate_did_data(
    n=800, ate=2.0, n_pre=3, n_post=3, seed=3
)
result = event_study_did(
    unit, group, period, outcome, treatment_time=3, reference=-1
)
for k, coef, se in zip(result.relative_times, result.coefficients, result.ses):
    print(f"k={k:+d}: {coef:.3f} (se {se:.3f})")
```

```bash
python -m causal_inference.cli event-study --n 500 --n-pre 3 --n-post 3 --ate 1.5
```

## Installation

```bash
pip install -r requirements.txt   # numpy, pandas
pip install -e .                  # optional: installs the package
```

Requires Python 3.9+.

## Quickstart (CLI)

```bash
# 1. simulate a confounded dataset with a known ATE
python -m causal_inference.cli simulate --n 5000 --confounding 1.5 --seed 7 --out data.csv

# 2. print point estimates
python -m causal_inference.cli estimate --data data.csv --true-ate 2.25

# 3. write a markdown evaluation report
python -m causal_inference.cli report --data data.csv --true-ate 2.25 --bootstrap 200 --out report.md
```

## Quickstart (Python)

```python
import numpy as np
from causal_inference import (
    simulate_observational_data,
    propensity_scores,
    ipw_ate,
    aipw_ate,
    evaluate,
    standard_estimators,
)

X, W, treatment, outcome, true_ate = simulate_observational_data(
    n=5000, confounding=1.5, seed=7
)
p, fit = propensity_scores(X, treatment)
ipw = ipw_ate(X, treatment, outcome, propensity=p)
aipw = aipw_ate(X, treatment, outcome, propensity=p, W=W)
print(f"IPW  estimate {ipw:.3f} vs true ATE {true_ate:.3f}")
print(f"AIPW estimate {aipw:.3f} vs true ATE {true_ate:.3f}")

results = evaluate(standard_estimators(), X, W, treatment, outcome, true_ate, n_boot=200)
for r in results:
    print(f"{r.name:<10} estimate {r.estimate:7.3f}  bias {r.bias:7.3f}  rmse {r.rmse:7.3f}")
```

## Demo

```bash
python examples/run_demo.py
```

simulates a confounded sample, runs every estimator, evaluates them with a
bootstrap, runs a DiD / TWFE panel, a synthetic-control donor panel, a sharp
RD sample and an instrumental-variables encouragement design, and writes
`examples/output/demo_report.md`.

### Difference-in-differences (Python)

```python
from causal_inference import simulate_did_data, difference_in_differences

unit, group, period, outcome, true_att = simulate_did_data(n=1000, seed=7)
result = difference_in_differences(group, period, outcome, unit=unit)
print(result.estimate, true_att, result.se, result.se_type)

# multi-period two-way fixed effects (canonical adoption at t = n_pre)
unit, group, period, outcome, true_att = simulate_did_data(
    n=400, n_pre=3, n_post=3, ate=1.5, seed=7
)
result = difference_in_differences(
    group, period, outcome, unit=unit, treatment_time=3
)
print(result.estimate, result.se, result.method)
```

The two-period estimator is the usual 2x2 interaction. Passing `unit` uses
unit-level first differences and clusters the standard error at the unit;
omit `unit` for a repeated cross-section with a four-cell robust SE. With
more than two periods the same function fits two-way fixed effects
`Y_it = a_i + b_t + tau D_it + e_it` and reports a cluster-robust SE.
That TWFE coefficient is an ATT under parallel trends and simultaneous
adoption; staggered timing can produce negative weights.

### Synthetic control (Python)

```python
from causal_inference import simulate_synthetic_control_data, synthetic_control

unit, time, outcome, treated, t0, true_effect = simulate_synthetic_control_data(
    n_donors=8, n_pre=12, n_post=8, ate=5.0, seed=7
)
result = synthetic_control(unit, time, outcome, treated, t0, placebo=True)
print(result.estimate, true_effect, result.weights, result.placebo_p_value)
```

Donor weights are constrained to the simplex (non-negative, sum to one) and
chosen so the synthetic series matches the treated unit before `t0`. The
estimate is the average post-treatment gap. With `placebo=True` each donor is
treated as if it were treated; the rank of the treated unit's post/pre RMSPE
ratio among those placebos is returned as `placebo_p_value`.

### Regression discontinuity (Python)

```python
from causal_inference import simulate_rd_data, regression_discontinuity

running, treatment, outcome, cutoff, true_effect = simulate_rd_data(
    n=4000, cutoff=0.0, ate=2.0, slope=1.0, seed=7
)
result = regression_discontinuity(running, outcome, cutoff=cutoff)
print(result.estimate, true_effect, result.bandwidth, result.n_left, result.n_right)
```

Local linear regressions of the outcome on `(running - cutoff)` are fit on
each side of the threshold, with triangular-kernel weights by default. The
estimate is the intercept jump at the cutoff. Omit `bandwidth` to use the
Imbens–Kalyanaraman (2012) selector, or pass a positive window half-width.

### Instrumental variables (Python)

```python
from causal_inference import simulate_iv_data, two_stage_least_squares

instrument, treatment, outcome, covariates, true_late = simulate_iv_data(
    n=4000, late=2.0, compliance=0.5, confounding=1.5, seed=7
)
result = two_stage_least_squares(instrument, treatment, outcome)
print(result.estimate, true_late, result.se, result.first_stage_f)
```

The instrument is a randomized encouragement. Treatment is taken only by
compliers, and only when they are encouraged, so the 2SLS coefficient is
the complier LATE and, because there are no always-takers, the ATT.
`result.se` is a heteroskedasticity-robust (HC1) standard error that uses
the structural residual. `result.first_stage_f` is the robust Wald F on
the excluded instruments. Pass exogenous controls with `covariates=`;
they enter both stages. A continuous or over-identified instrument is
accepted as a 1d or 2d array.

## Identifiability caveats

Every estimator in this toolkit recovers a causal effect only under
assumptions that are not testable from the data alone:

- **Ignorability (unconfoundedness):** treatment must be independent of the
  potential outcomes given the observed covariates. The generator's
  `selection_bias` knob produces data where this fails, and no adjustment for
  observed covariates can repair it.
- **Overlap (positivity):** every unit must have a non-zero probability of
  receiving either treatment. Thin overlap inflates IPW weights and makes
  nearest-neighbor matching unreliable.
- **SUTVA:** no interference between units and no hidden versions of the
  treatment.
- **Correct propensity model:** IPW and matching inherit the error of the
  estimated propensity score; a misspecified model leaves residual
  confounding.
- **Double robustness (AIPW):** AIPW remains consistent if *either* the
  propensity model *or* the outcome regressions are correctly specified.
  Both can be wrong at once, and then AIPW is biased like any other
  observational estimator.
- **Difference-in-differences:** requires parallel trends between groups and
  no anticipation of the treatment in the pre period. Two-way fixed effects
  additionally assume canonical (simultaneous) adoption; staggered timing
  can assign negative weights to some treatment effects.
- **Synthetic control:** the treated unit's pre-treatment path must be
  well approximated by a convex combination of untreated donors, with no
  anticipation. In-space placebo ranks are a diagnostic, not a conventional
  sampling p-value.
- **Regression discontinuity (sharp):** the running variable must not be
  manipulated at the cutoff, potential-outcome conditional means must be
  continuous there, and treatment must switch deterministically. The
  estimate is local to the threshold, not an ATE for the whole sample.
  Bandwidth choice trades bias against variance.
- **Instrumental variables (2SLS):** the instrument must be relevant
  (a small first-stage F signals a weak instrument), must affect the
  outcome only through treatment (exclusion), and must be as good as
  randomly assigned given the covariates (independence). With a binary
  instrument the estimate is a complier LATE. One-sided noncompliance
  makes that LATE equal the ATT. The reported standard error is
  heteroskedasticity-robust (HC1); weak instruments invalidate the
  usual normal approximation.

## Tests

```bash
python -m pytest tests -q -c pyproject.toml
```

## License

MIT
