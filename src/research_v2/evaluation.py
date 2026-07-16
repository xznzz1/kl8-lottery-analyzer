"""动态贝叶斯模型的嵌套时间滚动评估。"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution

from .bayesian import (
    DECAY_GRID,
    DRAW_SIZE,
    FAIR_PROBABILITY,
    NUMBER_COUNT,
    PRIOR_STRENGTH_GRID,
    BayesianParameters,
    rank_probabilities,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MINIMUM_INITIAL_HISTORY = 365
MINIMUM_INNER_OBSERVATIONS = 120
UNIFORM_RANDOM_SEEDS = tuple(range(202601, 202621))
DYNAMIC_STRATEGY = "dynamic_bayesian"
BASELINE_RANKING_STRATEGIES = (
    "historical_frequency",
    "rolling_frequency",
    "exponential_frequency",
    "hybrid",
    "repository_advanced",
)
COMPARATOR_STRATEGIES = (
    "uniform_random",
    *BASELINE_RANKING_STRATEGIES,
    DYNAMIC_STRATEGY,
)
RANKING_STRATEGIES = (*BASELINE_RANKING_STRATEGIES, DYNAMIC_STRATEGY)


class _StrategyParametersFactory(Protocol):
    def __call__(
        self,
        *,
        rolling_window: int,
        decay: float,
        hybrid_weights: tuple[float, float, float],
    ) -> object: ...


class _ScoreNumbers(Protocol):
    def __call__(
        self, history: IntArray, strategy: str, parameters: object
    ) -> FloatArray: ...


class _RankedNumbers(Protocol):
    def __call__(self, scores: FloatArray) -> IntArray: ...


class _ComputeEnhancedScores(Protocol):
    def __call__(
        self,
        draws: IntArray,
        limit: int,
        recent_window: int,
        reference_window: int,
        decay: float,
        weights: tuple[float, float, float],
        dirichlet_weight: float,
        pca_components: int,
        use_pca: bool,
        use_graph_embeddings: bool,
    ) -> tuple[list[tuple[int, float]], object]: ...


@dataclass(frozen=True)
class _V1ComparatorAdapter:
    score_numbers: _ScoreNumbers
    ranked_numbers: _RankedNumbers
    compute_enhanced_scores: _ComputeEnhancedScores
    parameters: object


def _load_v1_comparator_adapter() -> _V1ComparatorAdapter:
    """运行时只读加载 v1 比较器，避免把冻结源码纳入 v2 类型改造。"""

    module = import_module("src.scientific.strategies")
    feature_module = import_module("src.analysis.feature_enhancer")
    strategies = cast(tuple[str, ...], module.DETERMINISTIC_STRATEGIES)
    if strategies != BASELINE_RANKING_STRATEGIES:
        raise ValueError("v1确定性策略集合与v2比较契约不一致")
    factory = cast(_StrategyParametersFactory, module.StrategyParameters)
    parameters = factory(
        rolling_window=120,
        decay=0.99,
        hybrid_weights=(0.25, 0.50, 0.25),
    )
    return _V1ComparatorAdapter(
        score_numbers=cast(_ScoreNumbers, module.score_numbers),
        ranked_numbers=cast(_RankedNumbers, module.ranked_numbers),
        compute_enhanced_scores=cast(
            _ComputeEnhancedScores, feature_module.compute_enhanced_scores
        ),
        parameters=parameters,
    )


def deterministic_repository_advanced_scores(
    history: IntArray, adapter: _V1ComparatorAdapter
) -> FloatArray:
    """复用 v1 高级特征，并消除 PCA 主成分的符号不唯一性。"""

    recent_first = history[::-1]
    synthetic_issues = np.arange(len(recent_first), 0, -1, dtype=np.int64)[:, None]
    advanced_input = np.hstack([synthetic_issues, recent_first])
    base_ranked, _ = adapter.compute_enhanced_scores(
        advanced_input,
        limit=len(advanced_input),
        recent_window=40,
        reference_window=160,
        decay=0.97,
        weights=(0.45, 0.25, 0.30),
        dirichlet_weight=0.22,
        pca_components=1,
        use_pca=False,
        use_graph_embeddings=False,
    )
    base_scores = np.zeros(NUMBER_COUNT, dtype=np.float64)
    for number, value in base_ranked:
        base_scores[number - 1] = value

    indicators = np.zeros((len(recent_first), NUMBER_COUNT), dtype=np.float64)
    indicators[
        np.arange(len(recent_first), dtype=np.int64)[:, None],
        recent_first - 1,
    ] = 1.0
    pca_scores = np.zeros(NUMBER_COUNT, dtype=np.float64)
    centered = indicators - indicators.mean(axis=0)
    if float(np.var(indicators, axis=0).sum()) > 1e-12:
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, -1]
        pivot = int(np.argmax(np.abs(principal)))
        if principal[pivot] < 0.0:
            principal = -principal
        spread = float(principal.max() - principal.min())
        if spread >= 1e-9:
            pca_scores = (principal - principal.min()) / spread

    # v1中PCA权重固定为0.18；量化只用于稳定极近同分的排序。
    return cast(
        FloatArray,
        np.asarray(
            np.round(base_scores + 0.18 * pca_scores, decimals=12),
            dtype=np.float64,
        ),
    )


@dataclass(frozen=True)
class NestedEvaluationConfig:
    """第一阶段预先固定的时间验证设计。"""

    minimum_history: int = MINIMUM_INITIAL_HISTORY
    minimum_inner_observations: int = MINIMUM_INNER_OBSERVATIONS
    include_comparators: bool = True

    def __post_init__(self) -> None:
        if self.minimum_history < MINIMUM_INITIAL_HISTORY:
            raise ValueError("最小初始历史不得少于365期")
        if self.minimum_inner_observations <= 0:
            raise ValueError("内层验证观察数必须为正整数")


@dataclass(frozen=True)
class PosteriorGrid:
    """所有预注册参数在每个目标时点之前形成的后验网格。

    九个网格点是九种概率与不确定性设定；排序只由三个 ``decay`` 决定，
    因为同一 ``decay`` 下的三个 ``prior_strength`` 不改变号码次序。
    """

    parameters: tuple[BayesianParameters, ...]
    alpha: FloatArray
    beta: FloatArray
    posterior_mean: FloatArray
    brier_score: FloatArray


@dataclass(frozen=True)
class NestedEvaluationResult:
    """逐外层期选参后得到的探索性评估结果。"""

    issues: IntArray
    draws: IntArray
    indicators: FloatArray
    outer_indices: IntArray
    parameters: tuple[BayesianParameters, ...]
    selected_parameter_indices: IntArray
    inner_mean_brier: FloatArray
    alpha: FloatArray
    beta: FloatArray
    posterior_mean: FloatArray
    posterior_variance: FloatArray
    credible_interval_lower: FloatArray
    credible_interval_upper: FloatArray
    rankings: dict[str, IntArray]
    uniform_random_mean_hits: FloatArray

    @property
    def outer_issues(self) -> IntArray:
        """返回外层目标期号。"""

        return self.issues[self.outer_indices]

    @property
    def outer_outcomes(self) -> FloatArray:
        """返回外层目标期的80维二元结果。"""

        return self.indicators[self.outer_indices]

    def selected_parameters(self, outer_offset: int) -> BayesianParameters:
        """返回给定外层记录在预测前选定的参数。"""

        parameter_index = int(self.selected_parameter_indices[outer_offset])
        return self.parameters[parameter_index]


def preregistered_parameter_grid() -> tuple[BayesianParameters, ...]:
    """严格返回用户预注册的3×3小网格，不允许动态扩展。"""

    return tuple(
        BayesianParameters(decay=decay, prior_strength=prior_strength)
        for decay in DECAY_GRID
        for prior_strength in PRIOR_STRENGTH_GRID
    )


def validate_issue_draws(
    issues: IntArray, draws: IntArray
) -> tuple[IntArray, IntArray]:
    """校验按期号严格升序排列的快乐8开奖记录。"""

    issue_values = cast(IntArray, np.asarray(issues, dtype=np.int64))
    draw_values = cast(IntArray, np.asarray(draws, dtype=np.int64))
    if issue_values.ndim != 1 or draw_values.shape != (
        len(issue_values),
        DRAW_SIZE,
    ):
        raise ValueError("期号必须为一维，开奖号码必须为(n, 20)")
    if len(issue_values) == 0:
        raise ValueError("开奖记录不能为空")
    if np.any(np.diff(issue_values) <= 0):
        raise ValueError("期号必须严格递增且不得重复")
    if np.any((draw_values < 1) | (draw_values > NUMBER_COUNT)):
        raise ValueError("开奖号码必须位于1至80")
    if any(len(set(map(int, row))) != DRAW_SIZE for row in draw_values):
        raise ValueError("每期20个开奖号码必须互不重复")
    return issue_values, draw_values


def draws_to_indicators(draws: IntArray) -> FloatArray:
    """把 ``(n, 20)`` 开奖号码转换为 ``(n, 80)`` 二元矩阵。"""

    draw_values = cast(IntArray, np.asarray(draws, dtype=np.int64))
    if draw_values.ndim != 2 or draw_values.shape[1] != DRAW_SIZE:
        raise ValueError("开奖号码必须为(n, 20)")
    indicators = np.zeros((len(draw_values), NUMBER_COUNT), dtype=np.float64)
    indicators[
        np.arange(len(draw_values), dtype=np.int64)[:, None], draw_values - 1
    ] = 1.0
    if not np.all(indicators.sum(axis=1) == DRAW_SIZE):
        raise ValueError("存在重复或非法开奖号码")
    return indicators


def precompute_posterior_grid(indicators: FloatArray) -> PosteriorGrid:
    """按时间递推充分统计量，且目标期结果只在该期预测之后进入状态。"""

    outcomes = cast(FloatArray, np.asarray(indicators, dtype=np.float64))
    if outcomes.ndim != 2 or outcomes.shape[1] != NUMBER_COUNT:
        raise ValueError("二元结果必须为(n, 80)")
    if not np.logical_or(outcomes == 0.0, outcomes == 1.0).all():
        raise ValueError("二元结果只能包含0或1")
    if not np.all(outcomes.sum(axis=1) == DRAW_SIZE):
        raise ValueError("每期必须恰好有20个出现号码")

    parameters = preregistered_parameter_grid()
    shape = (len(parameters), len(outcomes), NUMBER_COUNT)
    alpha = np.empty(shape, dtype=np.float64)
    beta_values = np.empty(shape, dtype=np.float64)
    posterior_mean = np.empty(shape, dtype=np.float64)

    for parameter_index, parameter in enumerate(parameters):
        discounted_successes = np.zeros(NUMBER_COUNT, dtype=np.float64)
        total_weight = 0.0
        for target_index in range(len(outcomes)):
            current_alpha = (
                parameter.prior_strength * FAIR_PROBABILITY + discounted_successes
            )
            current_beta = (
                parameter.prior_strength * (1.0 - FAIR_PROBABILITY)
                + total_weight
                - discounted_successes
            )
            alpha[parameter_index, target_index] = current_alpha
            beta_values[parameter_index, target_index] = current_beta
            posterior_mean[parameter_index, target_index] = current_alpha / (
                current_alpha + current_beta
            )

            # 当前目标期结果只在预测完成后加入，供下一期及以后使用。
            discounted_successes = (
                parameter.decay * discounted_successes + outcomes[target_index]
            )
            total_weight = parameter.decay * total_weight + 1.0

    if not np.isfinite(posterior_mean).all():
        raise FloatingPointError("预计算后验包含NaN或无穷值")
    if not np.all((posterior_mean > 0.0) & (posterior_mean < 1.0)):
        raise FloatingPointError("后验均值必须严格位于(0, 1)")
    if not np.allclose(posterior_mean.sum(axis=2), DRAW_SIZE, rtol=0.0, atol=1e-9):
        raise FloatingPointError("任一目标期的80个概率之和不等于20")
    scores = np.mean(np.square(posterior_mean - outcomes[None, :, :]), axis=2)
    return PosteriorGrid(
        parameters=parameters,
        alpha=alpha,
        beta=beta_values,
        posterior_mean=posterior_mean,
        brier_score=scores,
    )


def _select_parameters_before_outer_issue(
    grid: PosteriorGrid,
    outer_indices: IntArray,
    *,
    minimum_history: int,
) -> tuple[IntArray, FloatArray]:
    selected = np.empty(len(outer_indices), dtype=np.int64)
    inner_scores = np.empty(
        (len(outer_indices), len(grid.parameters)), dtype=np.float64
    )
    for outer_offset, raw_target_index in enumerate(outer_indices):
        target_index = int(raw_target_index)
        # 内层只包含当前外层期之前、且已有至少minimum_history历史的目标期。
        scores = cast(
            FloatArray,
            grid.brier_score[:, minimum_history:target_index].mean(axis=1),
        )
        inner_scores[outer_offset] = scores
        # np.argmin在同分时返回预注册网格中的第一项，保证完全可复现。
        selected[outer_offset] = int(np.argmin(scores))
    return selected, inner_scores


def _selected_posterior_arrays(
    grid: PosteriorGrid,
    outer_indices: IntArray,
    selected_parameter_indices: IntArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    selected_alpha = grid.alpha[selected_parameter_indices, outer_indices]
    selected_beta = grid.beta[selected_parameter_indices, outer_indices]
    selected_mean = grid.posterior_mean[selected_parameter_indices, outer_indices]
    total = cast(FloatArray, selected_alpha + selected_beta)
    variance = cast(
        FloatArray,
        selected_alpha * selected_beta / (np.square(total) * (total + 1.0)),
    )
    lower = cast(
        FloatArray,
        np.asarray(
            beta_distribution.ppf(0.025, selected_alpha, selected_beta),
            dtype=np.float64,
        ),
    )
    upper = cast(
        FloatArray,
        np.asarray(
            beta_distribution.ppf(0.975, selected_alpha, selected_beta),
            dtype=np.float64,
        ),
    )
    if not all(
        np.isfinite(array).all()
        for array in (
            selected_alpha,
            selected_beta,
            selected_mean,
            variance,
            lower,
            upper,
        )
    ):
        raise FloatingPointError("选定后验产生NaN或无穷值")
    return selected_alpha, selected_beta, selected_mean, variance, lower, upper


def _compute_rankings(
    issues: IntArray,
    draws: IntArray,
    outer_indices: IntArray,
    selected_mean: FloatArray,
    *,
    include_comparators: bool,
) -> tuple[dict[str, IntArray], FloatArray]:
    count = len(outer_indices)
    rankings: dict[str, IntArray] = {
        DYNAMIC_STRATEGY: np.vstack(
            [rank_probabilities(probabilities) for probabilities in selected_mean]
        ).astype(np.int64)
    }
    uniform_mean_hits = np.full((count, 10), np.nan, dtype=np.float64)
    if not include_comparators:
        return rankings, uniform_mean_hits

    adapter = _load_v1_comparator_adapter()
    for strategy in BASELINE_RANKING_STRATEGIES:
        strategy_rankings = np.empty((count, NUMBER_COUNT), dtype=np.int64)
        for outer_offset, raw_target_index in enumerate(outer_indices):
            target_index = int(raw_target_index)
            if strategy == "repository_advanced":
                scores = deterministic_repository_advanced_scores(
                    draws[:target_index], adapter
                )
            else:
                scores = adapter.score_numbers(
                    draws[:target_index], strategy, adapter.parameters
                )
            strategy_rankings[outer_offset] = adapter.ranked_numbers(scores)
        rankings[strategy] = strategy_rankings

    for outer_offset, raw_target_index in enumerate(outer_indices):
        target_index = int(raw_target_index)
        issue = int(issues[target_index])
        winning = set(map(int, draws[target_index]))
        seed_hits = np.empty((len(UNIFORM_RANDOM_SEEDS), 10), dtype=np.float64)
        for seed_offset, seed in enumerate(UNIFORM_RANDOM_SEEDS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, issue]))
            permutation = rng.permutation(np.arange(1, NUMBER_COUNT + 1))
            cumulative_hits = np.cumsum(
                np.fromiter(
                    (number in winning for number in permutation),
                    dtype=np.int64,
                    count=NUMBER_COUNT,
                )
            )
            seed_hits[seed_offset] = cumulative_hits[:10]
        # 20个seed只在期内聚合；独立时间簇仍然是开奖期。
        uniform_mean_hits[outer_offset] = seed_hits.mean(axis=0)
    return rankings, uniform_mean_hits


def evaluate_nested_walk_forward(
    issues: IntArray,
    draws: IntArray,
    *,
    config: NestedEvaluationConfig | None = None,
) -> NestedEvaluationResult:
    """执行不打乱期号、逐外层期重新选参的嵌套时间评估。"""

    settings = config or NestedEvaluationConfig()
    issue_values, draw_values = validate_issue_draws(issues, draws)
    required = settings.minimum_history + settings.minimum_inner_observations + 1
    if len(issue_values) < required:
        raise ValueError(f"至少需要{required}期数据才能形成一个外层测试期")
    indicators = draws_to_indicators(draw_values)
    grid = precompute_posterior_grid(indicators)
    first_outer_index = settings.minimum_history + settings.minimum_inner_observations
    outer_indices = np.arange(first_outer_index, len(issue_values), dtype=np.int64)
    selected_indices, inner_scores = _select_parameters_before_outer_issue(
        grid,
        outer_indices,
        minimum_history=settings.minimum_history,
    )
    alpha, beta_values, means, variance, lower, upper = _selected_posterior_arrays(
        grid, outer_indices, selected_indices
    )
    rankings, uniform_hits = _compute_rankings(
        issue_values,
        draw_values,
        outer_indices,
        means,
        include_comparators=settings.include_comparators,
    )
    return NestedEvaluationResult(
        issues=issue_values,
        draws=draw_values,
        indicators=indicators,
        outer_indices=outer_indices,
        parameters=grid.parameters,
        selected_parameter_indices=selected_indices,
        inner_mean_brier=inner_scores,
        alpha=alpha,
        beta=beta_values,
        posterior_mean=means,
        posterior_variance=variance,
        credible_interval_lower=lower,
        credible_interval_upper=upper,
        rankings=rankings,
        uniform_random_mean_hits=uniform_hits,
    )


__all__ = [
    "BASELINE_RANKING_STRATEGIES",
    "COMPARATOR_STRATEGIES",
    "DYNAMIC_STRATEGY",
    "MINIMUM_INITIAL_HISTORY",
    "MINIMUM_INNER_OBSERVATIONS",
    "RANKING_STRATEGIES",
    "UNIFORM_RANDOM_SEEDS",
    "NestedEvaluationConfig",
    "NestedEvaluationResult",
    "PosteriorGrid",
    "draws_to_indicators",
    "deterministic_repository_advanced_scores",
    "evaluate_nested_walk_forward",
    "precompute_posterior_grid",
    "preregistered_parameter_grid",
    "validate_issue_draws",
]
