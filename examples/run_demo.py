"""End-to-end demo: simulate, estimate, evaluate and write a report."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_inference.estimators import (
    aipw_ate,
    difference_in_differences,
    difference_in_means,
    ipw_ate,
    ipw_att,
    ipw_weights,
    propensity_matching,
)
from causal_inference.evaluate import evaluate, standard_estimators
from causal_inference.generators import simulate_did_data, simulate_observational_data
from causal_inference.propensity import propensity_scores
from causal_inference.report import render_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8000, help="sample size")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="directory for the generated report",
    )
    args = parser.parse_args(argv)

    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=args.n, confounding=1.5, selection_bias=0.0, seed=args.seed
    )

    print("=" * 60)
    print("Causal inference demo")
    print("=" * 60)
    print(f"synthetic sample: n={len(treatment)}, treated {treatment.mean():.1%}")
    print(f"ground-truth ATE: {true_ate:.3f}\n")

    p, _ = propensity_scores(X, treatment)
    estimates = [
        ("naive difference", difference_in_means(treatment, outcome)),
        ("ipw (stabilized)", ipw_ate(X, treatment, outcome, propensity=p)),
        ("aipw (doubly robust)", aipw_ate(X, treatment, outcome, propensity=p, W=W)),
        ("ipw (ATT)", ipw_att(X, treatment, outcome, propensity=p)),
        (
            "matching (ATT)",
            propensity_matching(X, treatment, outcome, propensity=p, with_replacement=True),
        ),
    ]
    print(f"{'estimator':<24}{'estimate':>10}{'bias':>10}")
    print("-" * 44)
    for name, estimate in estimates:
        print(f"{name:<24}{estimate:>10.3f}{estimate - true_ate:>10.3f}")

    print(f"\nbootstrap evaluation ({args.bootstrap} resamples, fixed first stage):")
    results = evaluate(
        standard_estimators(),
        X,
        W,
        treatment,
        outcome,
        true_ate,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    print(f"{'estimator':<12}{'estimate':>10}{'se':>8}{'bias':>8}{'rmse':>8}")
    print("-" * 46)
    for result in results:
        print(
            f"{result.name:<12}{result.estimate:>10.3f}{result.se:>8.3f}"
            f"{result.bias:>8.3f}{result.rmse:>8.3f}"
        )

    group, period, did_outcome, true_did = simulate_did_data(n=args.n, seed=args.seed)
    did = difference_in_differences(group, period, did_outcome)
    print(f"\ndifference-in-differences on a simulated panel: {did:.3f} (true {true_did:.3f})")

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "demo_report.md")
    did_section = (
        "## Difference-in-differences\n\n"
        f"On a simulated two-period two-group panel the DiD estimate is "
        f"**{did:.3f}** against a true effect of **{true_did:.3f}**."
    )
    markdown = render_report(
        X,
        W,
        treatment,
        outcome,
        results,
        true_ate,
        weights=ipw_weights(X, treatment),
        extra_markdown=did_section,
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(f"\nreport written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
