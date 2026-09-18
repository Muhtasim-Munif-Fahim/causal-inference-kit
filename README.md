# causal-inference-kit

A small, dependency-light toolkit for estimating treatment effects from
observational data. It includes a synthetic data generator with known
ground-truth effects, a propensity-score model fitted by gradient descent,
IPW / matching / AIPW / difference-in-differences / synthetic-control
estimators, bootstrap evaluation against the ground truth, and markdown
report rendering. Only `numpy` and `pandas` are required.

## Contents

| Module | Purpose |
| --- | --- |
| `causal_inference.generators` | Synthetic observational data with known ATE, confounding, effect heterogeneity and a hidden-confounding (selection bias) knob; a two-period panel for DiD; a donor panel for synthetic control |
| `causal_inference.propensity` | Logistic regression by gradient descent (L2, backtracking), propensity scores, SMD and overlap diagnostics |
| `causal_inference.estimators` | IPW (ATE/ATT, stabilized and Hájek variants), AIPW (doubly robust ATE), nearest-neighbor propensity matching, difference-in-differences, Abadie synthetic control |
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
| Difference-in-differences | ATT (DiD) | Two-period, two-group |
| Synthetic control | ATT (SC) | Non-negative donor weights summing to 1; pre-treatment fit, post-treatment gap, optional in-space placebo |

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
bootstrap, runs a DiD panel and a synthetic-control donor panel, and writes
`examples/output/demo_report.md`.

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
  no anticipation of the treatment in the pre period.
- **Synthetic control:** the treated unit's pre-treatment path must be
  well approximated by a convex combination of untreated donors, with no
  anticipation. In-space placebo ranks are a diagnostic, not a conventional
  sampling p-value.

## Tests

```bash
python -m pytest tests -q -c pyproject.toml
```

## License

MIT
