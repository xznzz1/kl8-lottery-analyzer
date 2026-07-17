"""快乐8 v2 第二阶段的在线变点检测与自适应 Beta-Bernoulli 模型。

变化分数比较互不重叠的近期窗口与其之前的参考窗口。该分数只描述历史
频率结构的偏离，不证明开奖机制发生变化；公平彩票的随机波动也会产生
假变点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution

from .bayesian import (
    DRAW_SIZE,
    FAIR_PROBABILITY,
    NUMBER_COUNT,
    rank_probabilities,
    validate_indicator_history,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

RECENT_WINDOW_GRID = (30, 60, 120)
REFERENCE_WINDOW = 240
MINIMUM_EFFECTIVE_HISTORY = 120
CHANGE_QUANTILE = 0.90
HIGH_CHANGE_DECAY = 0.95
NORMAL_DECAY = 0.995
CHANGEPOINT_PRIOR_STRENGTH = 80.0


@dataclass(frozen=True, order=True)
class ChangepointParameters:
    """第二阶段预注册配置；只有近期窗口允许在三项小网格中选择。"""

    recent_window: int
    reference_window: int = REFERENCE_WINDOW
    minimum_effective_history: int = MINIMUM_EFFECTIVE_HISTORY
    change_quantile: float = CHANGE_QUANTILE
    high_change_decay: float = HIGH_CHANGE_DECAY
    normal_decay: float = NORMAL_DECAY
    prior_strength: float = CHANGEPOINT_PRIOR_STRENGTH

    def __post_init__(self) -> None:
        expected = (
            self.reference_window == REFERENCE_WINDOW
            and self.minimum_effective_history == MINIMUM_EFFECTIVE_HISTORY
            and self.change_quantile == CHANGE_QUANTILE
            and self.high_change_decay == HIGH_CHANGE_DECAY
            and self.normal_decay == NORMAL_DECAY
            and self.prior_strength == CHANGEPOINT_PRIOR_STRENGTH
        )
        if self.recent_window not in RECENT_WINDOW_GRID or not expected:
            raise ValueError("变点参数必须严格来自第二阶段预注册配置")

    @property
    def required_score_history(self) -> int:
        """形成一次变化分数所需的最少历史期数。"""

        return self.recent_window + self.reference_window


@dataclass(frozen=True)
class ChangepointPrediction:
    """单个预测时点的变点状态及80维后验。"""

    alpha: FloatArray
    beta: FloatArray
    posterior_mean: FloatArray
    posterior_variance: FloatArray
    credible_interval_lower: FloatArray
    credible_interval_upper: FloatArray
    change_score: float
    change_threshold: float
    high_change: bool
    decay: float
    effective_history_length: int

    @property
    def ranking(self) -> IntArray:
        """按概率降序、同分号码升序返回完整排名。"""

        return rank_probabilities(self.posterior_mean)


def preregistered_changepoint_parameters() -> tuple[ChangepointParameters, ...]:
    """按固定顺序返回三个预注册近期窗口。"""

    return tuple(ChangepointParameters(window) for window in RECENT_WINDOW_GRID)


def frequency_change_score(
    history: FloatArray, parameters: ChangepointParameters
) -> float:
    """计算近期与此前参考窗口的80维频率 L1 距离。

    ``history`` 必须只包含目标期之前的开奖。参考窗口紧邻近期窗口但不与
    其重叠，避免近期样本同时出现在比较两侧。
    """

    array = validate_indicator_history(history)
    required = parameters.required_score_history
    if len(array) < required:
        raise ValueError(f"变化分数至少需要{required}期历史")
    recent = array[-parameters.recent_window :]
    reference = array[-parameters.required_score_history : -parameters.recent_window]
    recent_frequency = cast(FloatArray, recent.mean(axis=0))
    reference_frequency = cast(FloatArray, reference.mean(axis=0))
    score = float(np.abs(recent_frequency - reference_frequency).sum())
    if not np.isfinite(score) or score < 0.0:
        raise FloatingPointError("变化分数必须是有限非负数")
    return score


def precompute_change_scores(
    indicators: FloatArray, parameters: ChangepointParameters
) -> FloatArray:
    """为每个目标索引计算只使用该索引之前历史的变化分数。"""

    array = validate_indicator_history(indicators)
    scores = np.full(len(array), np.nan, dtype=np.float64)
    cumulative = np.vstack(
        (
            np.zeros((1, NUMBER_COUNT), dtype=np.float64),
            np.cumsum(array, axis=0, dtype=np.float64),
        )
    )
    window = parameters.recent_window
    reference = parameters.reference_window
    for target_index in range(parameters.required_score_history, len(array)):
        recent_counts = cumulative[target_index] - cumulative[target_index - window]
        reference_end = target_index - window
        reference_counts = (
            cumulative[reference_end] - cumulative[reference_end - reference]
        )
        recent_frequency = recent_counts / window
        reference_frequency = reference_counts / reference
        scores[target_index] = float(
            np.abs(recent_frequency - reference_frequency).sum()
        )
    if not np.isfinite(scores[parameters.required_score_history :]).all():
        raise FloatingPointError("预计算变化分数包含NaN或无穷值")
    return scores


def _posterior_from_weighted_history(
    history: FloatArray,
    *,
    decay: float,
    prior_strength: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    ages = np.arange(len(history) - 1, -1, -1, dtype=np.float64)
    weights = cast(FloatArray, np.power(decay, ages))
    successes = cast(FloatArray, weights @ history)
    total_weight = float(weights.sum())
    alpha = cast(FloatArray, prior_strength * FAIR_PROBABILITY + successes)
    beta_values = cast(
        FloatArray,
        prior_strength * (1.0 - FAIR_PROBABILITY) + total_weight - successes,
    )
    posterior_mean = cast(FloatArray, alpha / (alpha + beta_values))
    return alpha, beta_values, posterior_mean


def _validate_posterior(
    alpha: FloatArray, beta_values: FloatArray, posterior_mean: FloatArray
) -> None:
    if not all(
        np.isfinite(values).all() for values in (alpha, beta_values, posterior_mean)
    ):
        raise FloatingPointError("变点后验产生NaN或无穷值")
    if not np.all((posterior_mean > 0.0) & (posterior_mean < 1.0)):
        raise FloatingPointError("后验概率必须严格位于(0, 1)")
    if not np.isclose(posterior_mean.sum(), DRAW_SIZE, rtol=0.0, atol=1e-9):
        raise FloatingPointError("80个后验概率之和不等于20")


def changepoint_beta_bernoulli(
    history: FloatArray,
    *,
    change_threshold: float,
    parameters: ChangepointParameters,
) -> ChangepointPrediction:
    """只用目标期之前历史形成变点状态和自适应后验。"""

    array = validate_indicator_history(history)
    if not np.isfinite(change_threshold) or change_threshold < 0.0:
        raise ValueError("变化阈值必须是有限非负数")
    score = frequency_change_score(array, parameters)
    high_change = score > change_threshold
    if high_change:
        effective = min(len(array), parameters.minimum_effective_history)
        effective_history = array[-effective:]
        decay = parameters.high_change_decay
    else:
        effective = len(array)
        effective_history = array
        decay = parameters.normal_decay

    alpha, beta_values, posterior_mean = _posterior_from_weighted_history(
        effective_history,
        decay=decay,
        prior_strength=parameters.prior_strength,
    )
    _validate_posterior(alpha, beta_values, posterior_mean)
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
        raise FloatingPointError("变点后验不确定性计算产生NaN或无穷值")
    return ChangepointPrediction(
        alpha=alpha,
        beta=beta_values,
        posterior_mean=posterior_mean,
        posterior_variance=variance,
        credible_interval_lower=lower,
        credible_interval_upper=upper,
        change_score=score,
        change_threshold=float(change_threshold),
        high_change=high_change,
        decay=decay,
        effective_history_length=effective,
    )


__all__ = [
    "CHANGEPOINT_PRIOR_STRENGTH",
    "CHANGE_QUANTILE",
    "HIGH_CHANGE_DECAY",
    "MINIMUM_EFFECTIVE_HISTORY",
    "NORMAL_DECAY",
    "RECENT_WINDOW_GRID",
    "REFERENCE_WINDOW",
    "ChangepointParameters",
    "ChangepointPrediction",
    "changepoint_beta_bernoulli",
    "frequency_change_score",
    "precompute_change_scores",
    "preregistered_changepoint_parameters",
]
