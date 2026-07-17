"""变点贝叶斯模型的严格因果嵌套时间滚动评估。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution

from .bayesian import DRAW_SIZE, FAIR_PROBABILITY, NUMBER_COUNT, rank_probabilities
from .changepoint import (
    CHANGEPOINT_PRIOR_STRENGTH,
    HIGH_CHANGE_DECAY,
    MINIMUM_EFFECTIVE_HISTORY,
    NORMAL_DECAY,
    ChangepointParameters,
    precompute_change_scores,
    preregistered_changepoint_parameters,
)
from .evaluation import (
    DYNAMIC_STRATEGY,
    MINIMUM_INITIAL_HISTORY,
    MINIMUM_INNER_OBSERVATIONS,
    NestedEvaluationConfig,
    NestedEvaluationResult,
    evaluate_nested_walk_forward,
    validate_issue_draws,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

CHANGEPOINT_STRATEGY = "changepoint_bayesian"
FIXED_NORMAL_STRATEGY = "fixed_normal_bayesian"
FIXED_HIGH_STRATEGY = "fixed_high_bayesian"
PHASE2_COMPARATOR_STRATEGIES = (
    "uniform_random",
    "rolling_frequency",
    "exponential_frequency",
    "hybrid",
    "repository_advanced",
    DYNAMIC_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    FIXED_HIGH_STRATEGY,
    CHANGEPOINT_STRATEGY,
)


@dataclass(frozen=True)
class ChangepointEvaluationConfig:
    """第二阶段固定时间设计；只允许关闭昂贵比较器以便单元测试。"""

    minimum_history: int = MINIMUM_INITIAL_HISTORY
    minimum_inner_observations: int = MINIMUM_INNER_OBSERVATIONS
    include_comparators: bool = True

    def __post_init__(self) -> None:
        if self.minimum_history != MINIMUM_INITIAL_HISTORY:
            raise ValueError("第二阶段初始历史固定为365期")
        if self.minimum_inner_observations != MINIMUM_INNER_OBSERVATIONS:
            raise ValueError("第二阶段初始内层验证固定为120期")


@dataclass(frozen=True)
class _AdaptivePosteriorGrid:
    normal_alpha: FloatArray
    normal_beta: FloatArray
    normal_mean: FloatArray
    high_alpha: FloatArray
    high_beta: FloatArray
    high_mean: FloatArray


@dataclass(frozen=True)
class ChangepointEvaluationResult:
    """第二阶段逐外层期的概率、状态、选窗与比较结果。"""

    phase1_result: NestedEvaluationResult
    parameters: tuple[ChangepointParameters, ...]
    selected_parameter_indices: IntArray
    inner_mean_brier: FloatArray
    candidate_change_scores: FloatArray
    candidate_thresholds: FloatArray
    candidate_high_change: BoolArray
    threshold_training_first_indices: IntArray
    threshold_training_last_indices: IntArray
    alpha: FloatArray
    beta: FloatArray
    posterior_mean: FloatArray
    posterior_variance: FloatArray
    credible_interval_lower: FloatArray
    credible_interval_upper: FloatArray
    fixed_normal_posterior_mean: FloatArray
    fixed_high_posterior_mean: FloatArray
    high_change: BoolArray
    active_decay: FloatArray
    effective_history_length: IntArray
    rankings: dict[str, IntArray]

    @property
    def issues(self) -> IntArray:
        return self.phase1_result.issues

    @property
    def draws(self) -> IntArray:
        return self.phase1_result.draws

    @property
    def indicators(self) -> FloatArray:
        return self.phase1_result.indicators

    @property
    def outer_indices(self) -> IntArray:
        return self.phase1_result.outer_indices

    @property
    def outer_issues(self) -> IntArray:
        return self.phase1_result.outer_issues

    @property
    def outer_outcomes(self) -> FloatArray:
        return self.phase1_result.outer_outcomes

    @property
    def uniform_random_mean_hits(self) -> FloatArray:
        return self.phase1_result.uniform_random_mean_hits

    def selected_parameters(self, outer_offset: int) -> ChangepointParameters:
        index = int(self.selected_parameter_indices[outer_offset])
        return self.parameters[index]

    @property
    def ranking_differs_from_fixed_normal(self) -> BoolArray:
        """逐期标记自适应排名是否不同于始终 normal 的消融基线。"""

        return cast(
            BoolArray,
            np.any(
                self.rankings[CHANGEPOINT_STRATEGY]
                != self.rankings[FIXED_NORMAL_STRATEGY],
                axis=1,
            ),
        )

    @property
    def ranking_differs_from_fixed_high(self) -> BoolArray:
        """逐期标记自适应排名是否不同于始终 high 的消融基线。"""

        return cast(
            BoolArray,
            np.any(
                self.rankings[CHANGEPOINT_STRATEGY]
                != self.rankings[FIXED_HIGH_STRATEGY],
                axis=1,
            ),
        )


def _precompute_one_decay(
    indicators: FloatArray, *, decay: float, history_limit: int | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    count = len(indicators)
    alpha = np.empty((count, NUMBER_COUNT), dtype=np.float64)
    beta_values = np.empty_like(alpha)
    means = np.empty_like(alpha)
    successes = np.zeros(NUMBER_COUNT, dtype=np.float64)
    total_weight = 0.0
    removal_weight = decay**history_limit if history_limit is not None else 0.0

    for target_index in range(count):
        current_alpha = CHANGEPOINT_PRIOR_STRENGTH * FAIR_PROBABILITY + successes
        current_beta = (
            CHANGEPOINT_PRIOR_STRENGTH * (1.0 - FAIR_PROBABILITY)
            + total_weight
            - successes
        )
        alpha[target_index] = current_alpha
        beta_values[target_index] = current_beta
        means[target_index] = current_alpha / (current_alpha + current_beta)

        successes = decay * successes + indicators[target_index]
        total_weight = decay * total_weight + 1.0
        if history_limit is not None and target_index >= history_limit:
            successes = (
                successes - removal_weight * indicators[target_index - history_limit]
            )
            total_weight -= removal_weight

    if not np.isfinite(means).all():
        raise FloatingPointError("自适应后验网格包含NaN或无穷值")
    if not np.all((means > 0.0) & (means < 1.0)):
        raise FloatingPointError("自适应后验概率必须严格位于(0, 1)")
    if not np.allclose(means.sum(axis=1), DRAW_SIZE, rtol=0.0, atol=1e-9):
        raise FloatingPointError("自适应后验网格的80个概率之和不等于20")
    return alpha, beta_values, means


def _precompute_adaptive_posteriors(indicators: FloatArray) -> _AdaptivePosteriorGrid:
    normal_alpha, normal_beta, normal_mean = _precompute_one_decay(
        indicators, decay=NORMAL_DECAY, history_limit=None
    )
    high_alpha, high_beta, high_mean = _precompute_one_decay(
        indicators,
        decay=HIGH_CHANGE_DECAY,
        history_limit=MINIMUM_EFFECTIVE_HISTORY,
    )
    return _AdaptivePosteriorGrid(
        normal_alpha=normal_alpha,
        normal_beta=normal_beta,
        normal_mean=normal_mean,
        high_alpha=high_alpha,
        high_beta=high_beta,
        high_mean=high_mean,
    )


def _causal_thresholds_and_candidates(
    indicators: FloatArray,
    parameters: tuple[ChangepointParameters, ...],
    grid: _AdaptivePosteriorGrid,
    *,
    minimum_history: int,
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray, FloatArray]:
    parameter_count = len(parameters)
    period_count = len(indicators)
    scores = np.full((parameter_count, period_count), np.nan, dtype=np.float64)
    thresholds = np.full_like(scores, np.nan)
    high_change = np.zeros((parameter_count, period_count), dtype=np.bool_)
    candidate_means = np.full(
        (parameter_count, period_count, NUMBER_COUNT), np.nan, dtype=np.float64
    )
    candidate_brier = np.full((parameter_count, period_count), np.nan, dtype=np.float64)

    for parameter_index, parameter in enumerate(parameters):
        parameter_scores = precompute_change_scores(indicators, parameter)
        scores[parameter_index] = parameter_scores
        first_score_index = parameter.required_score_history
        for target_index in range(minimum_history, period_count):
            historical_scores = parameter_scores[first_score_index:target_index]
            if len(historical_scores) == 0 or not np.isfinite(historical_scores).all():
                raise ValueError("变化阈值没有足够的目标期前历史分数")
            threshold = float(
                np.quantile(
                    historical_scores, parameter.change_quantile, method="linear"
                )
            )
            thresholds[parameter_index, target_index] = threshold
            is_high = bool(parameter_scores[target_index] > threshold)
            high_change[parameter_index, target_index] = is_high
            means = grid.high_mean if is_high else grid.normal_mean
            candidate_means[parameter_index, target_index] = means[target_index]
            candidate_brier[parameter_index, target_index] = float(
                np.mean(np.square(means[target_index] - indicators[target_index]))
            )
    return scores, thresholds, high_change, candidate_means, candidate_brier


def _select_recent_windows(
    candidate_brier: FloatArray,
    outer_indices: IntArray,
    *,
    minimum_history: int,
) -> tuple[IntArray, FloatArray]:
    parameter_count = candidate_brier.shape[0]
    selected = np.empty(len(outer_indices), dtype=np.int64)
    inner_means = np.empty((len(outer_indices), parameter_count), dtype=np.float64)
    for outer_offset, raw_target_index in enumerate(outer_indices):
        target_index = int(raw_target_index)
        means = cast(
            FloatArray,
            candidate_brier[:, minimum_history:target_index].mean(axis=1),
        )
        if not np.isfinite(means).all():
            raise FloatingPointError("近期窗口内层Brier均值包含NaN或无穷值")
        inner_means[outer_offset] = means
        selected[outer_offset] = int(np.argmin(means))
    return selected, inner_means


def _selected_posterior_arrays(
    grid: _AdaptivePosteriorGrid,
    outer_indices: IntArray,
    selected_parameter_indices: IntArray,
    high_change_grid: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray, FloatArray, IntArray]:
    count = len(outer_indices)
    alpha = np.empty((count, NUMBER_COUNT), dtype=np.float64)
    beta_values = np.empty_like(alpha)
    means = np.empty_like(alpha)
    selected_high = np.empty(count, dtype=np.bool_)
    active_decay = np.empty(count, dtype=np.float64)
    effective_history = np.empty(count, dtype=np.int64)
    for outer_offset, (raw_target, raw_parameter) in enumerate(
        zip(outer_indices, selected_parameter_indices, strict=True)
    ):
        target_index = int(raw_target)
        parameter_index = int(raw_parameter)
        is_high = bool(high_change_grid[parameter_index, target_index])
        selected_high[outer_offset] = is_high
        if is_high:
            alpha[outer_offset] = grid.high_alpha[target_index]
            beta_values[outer_offset] = grid.high_beta[target_index]
            means[outer_offset] = grid.high_mean[target_index]
            active_decay[outer_offset] = HIGH_CHANGE_DECAY
            effective_history[outer_offset] = MINIMUM_EFFECTIVE_HISTORY
        else:
            alpha[outer_offset] = grid.normal_alpha[target_index]
            beta_values[outer_offset] = grid.normal_beta[target_index]
            means[outer_offset] = grid.normal_mean[target_index]
            active_decay[outer_offset] = NORMAL_DECAY
            effective_history[outer_offset] = target_index
    return alpha, beta_values, means, selected_high, active_decay, effective_history


def _posterior_uncertainty(
    alpha: FloatArray, beta_values: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    total = cast(FloatArray, alpha + beta_values)
    variance = cast(
        FloatArray,
        alpha * beta_values / (np.square(total) * (total + 1.0)),
    )
    lower = cast(
        FloatArray,
        np.asarray(beta_distribution.ppf(0.025, alpha, beta_values), dtype=np.float64),
    )
    upper = cast(
        FloatArray,
        np.asarray(beta_distribution.ppf(0.975, alpha, beta_values), dtype=np.float64),
    )
    if not all(np.isfinite(values).all() for values in (variance, lower, upper)):
        raise FloatingPointError("变点后验不确定性包含NaN或无穷值")
    return variance, lower, upper


def evaluate_changepoint_walk_forward(
    issues: IntArray,
    draws: IntArray,
    *,
    config: ChangepointEvaluationConfig | None = None,
) -> ChangepointEvaluationResult:
    """执行三窗口、90%历史阈值和自适应权重的嵌套时间评估。"""

    settings = config or ChangepointEvaluationConfig()
    issue_values, draw_values = validate_issue_draws(issues, draws)
    required = settings.minimum_history + settings.minimum_inner_observations + 1
    if len(issue_values) < required:
        raise ValueError(f"至少需要{required}期数据才能形成一个外层测试期")

    phase1 = evaluate_nested_walk_forward(
        issue_values,
        draw_values,
        config=NestedEvaluationConfig(
            minimum_history=settings.minimum_history,
            minimum_inner_observations=settings.minimum_inner_observations,
            include_comparators=settings.include_comparators,
        ),
    )
    parameters = preregistered_changepoint_parameters()
    adaptive_grid = _precompute_adaptive_posteriors(phase1.indicators)
    scores, thresholds, high_grid, _, candidate_brier = (
        _causal_thresholds_and_candidates(
            phase1.indicators,
            parameters,
            adaptive_grid,
            minimum_history=settings.minimum_history,
        )
    )
    selected, inner_means = _select_recent_windows(
        candidate_brier,
        phase1.outer_indices,
        minimum_history=settings.minimum_history,
    )
    alpha, beta_values, means, high_change, active_decay, effective_history = (
        _selected_posterior_arrays(
            adaptive_grid,
            phase1.outer_indices,
            selected,
            high_grid,
        )
    )
    variance, lower, upper = _posterior_uncertainty(alpha, beta_values)

    outer = phase1.outer_indices
    fixed_normal_means = adaptive_grid.normal_mean[outer].copy()
    fixed_high_means = adaptive_grid.high_mean[outer].copy()
    normal_mask = np.logical_not(high_change)
    if not np.array_equal(means[normal_mask], fixed_normal_means[normal_mask]):
        raise FloatingPointError("normal状态未逐项复用fixed_normal后验")
    if not np.array_equal(means[high_change], fixed_high_means[high_change]):
        raise FloatingPointError("high_change状态未逐项复用fixed_high后验")

    rankings: dict[str, IntArray] = {
        DYNAMIC_STRATEGY: phase1.rankings[DYNAMIC_STRATEGY],
        FIXED_NORMAL_STRATEGY: np.vstack(
            [rank_probabilities(probabilities) for probabilities in fixed_normal_means]
        ).astype(np.int64),
        FIXED_HIGH_STRATEGY: np.vstack(
            [rank_probabilities(probabilities) for probabilities in fixed_high_means]
        ).astype(np.int64),
        CHANGEPOINT_STRATEGY: np.vstack(
            [rank_probabilities(probabilities) for probabilities in means]
        ).astype(np.int64),
    }
    if settings.include_comparators:
        for strategy in PHASE2_COMPARATOR_STRATEGIES:
            if strategy not in (
                "uniform_random",
                CHANGEPOINT_STRATEGY,
                FIXED_NORMAL_STRATEGY,
                FIXED_HIGH_STRATEGY,
            ):
                rankings[strategy] = phase1.rankings[strategy]

    candidate_scores = scores[:, outer].T
    candidate_thresholds = thresholds[:, outer].T
    candidate_high = high_grid[:, outer].T
    first_threshold_indices = np.asarray(
        [parameter.required_score_history for parameter in parameters],
        dtype=np.int64,
    )
    last_threshold_indices = cast(IntArray, outer - 1)

    return ChangepointEvaluationResult(
        phase1_result=phase1,
        parameters=parameters,
        selected_parameter_indices=selected,
        inner_mean_brier=inner_means,
        candidate_change_scores=candidate_scores,
        candidate_thresholds=candidate_thresholds,
        candidate_high_change=candidate_high,
        threshold_training_first_indices=first_threshold_indices,
        threshold_training_last_indices=last_threshold_indices,
        alpha=alpha,
        beta=beta_values,
        posterior_mean=means,
        posterior_variance=variance,
        credible_interval_lower=lower,
        credible_interval_upper=upper,
        fixed_normal_posterior_mean=fixed_normal_means,
        fixed_high_posterior_mean=fixed_high_means,
        high_change=high_change,
        active_decay=active_decay,
        effective_history_length=effective_history,
        rankings=rankings,
    )


__all__ = [
    "CHANGEPOINT_STRATEGY",
    "FIXED_HIGH_STRATEGY",
    "FIXED_NORMAL_STRATEGY",
    "PHASE2_COMPARATOR_STRATEGIES",
    "ChangepointEvaluationConfig",
    "ChangepointEvaluationResult",
    "evaluate_changepoint_walk_forward",
]
