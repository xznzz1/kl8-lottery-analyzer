"""快乐8策略 v2 的独立探索性研究模块。"""

from .bayesian import (
    DECAY_GRID,
    FAIR_PROBABILITY,
    PRIOR_STRENGTH_GRID,
    BayesianParameters,
    PosteriorPrediction,
    candidate_sets,
    discounted_beta_bernoulli,
    rank_probabilities,
)
from .evaluation import (
    COMPARATOR_STRATEGIES,
    NestedEvaluationConfig,
    NestedEvaluationResult,
    evaluate_nested_walk_forward,
)
from .metrics import (
    CalibrationBin,
    CalibrationResult,
    bernoulli_log_loss,
    brier_score,
    calibration_summary,
    top_k_hits,
)

__all__ = [
    "COMPARATOR_STRATEGIES",
    "DECAY_GRID",
    "FAIR_PROBABILITY",
    "PRIOR_STRENGTH_GRID",
    "BayesianParameters",
    "CalibrationBin",
    "CalibrationResult",
    "NestedEvaluationConfig",
    "NestedEvaluationResult",
    "PosteriorPrediction",
    "bernoulli_log_loss",
    "brier_score",
    "calibration_summary",
    "candidate_sets",
    "discounted_beta_bernoulli",
    "evaluate_nested_walk_forward",
    "rank_probabilities",
    "top_k_hits",
]
