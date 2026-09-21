"""causal_inference: a small toolkit for treatment-effect estimation."""

from .evaluate import EvaluationResult, evaluate, standard_estimators
from .estimators import (
    DifferenceInDifferencesResult,
    RegressionDiscontinuityResult,
    SyntheticControlResult,
    aipw_ate,
    difference_in_differences,
    difference_in_means,
    ipw_ate,
    ipw_att,
    ipw_weights,
    outcome_regression,
    propensity_matching,
    regression_discontinuity,
    synthetic_control,
)
from .generators import (
    simulate_did_data,
    simulate_observational_data,
    simulate_rd_data,
    simulate_synthetic_control_data,
)
from .propensity import (
    LogisticFit,
    PropensityHistogram,
    logistic_regression,
    propensity_histogram,
    propensity_scores,
    standardized_mean_differences,
    weighted_standardized_mean_differences,
)
from .report import render_report

__version__ = "0.1.0"

__all__ = [
    "DifferenceInDifferencesResult",
    "EvaluationResult",
    "LogisticFit",
    "PropensityHistogram",
    "RegressionDiscontinuityResult",
    "SyntheticControlResult",
    "aipw_ate",
    "difference_in_differences",
    "difference_in_means",
    "evaluate",
    "ipw_ate",
    "ipw_att",
    "ipw_weights",
    "logistic_regression",
    "outcome_regression",
    "propensity_histogram",
    "propensity_matching",
    "propensity_scores",
    "regression_discontinuity",
    "render_report",
    "simulate_did_data",
    "simulate_observational_data",
    "simulate_rd_data",
    "simulate_synthetic_control_data",
    "standard_estimators",
    "standardized_mean_differences",
    "synthetic_control",
    "weighted_standardized_mean_differences",
]
