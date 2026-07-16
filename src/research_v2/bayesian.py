"""动态折扣 Beta-Bernoulli 概率模型。

固定 ``decay`` 时，令 ``S_i`` 为号码 ``i`` 的折扣出现次数、``W`` 为折扣
总权重、``s`` 为 ``prior_strength``，则
``posterior_mean_i = (s * 0.25 + S_i) / (s + W)``。这是 ``S_i`` 的正仿射
变换，因此与使用相同 ``decay`` 的指数频率 ``S_i / W`` 产生完全相同的
号码排序。``prior_strength`` 只改变概率收缩、方差、可信区间和概率评分，
不改变排序；3×3概率参数网格只有3种不同的排序机制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution

FloatArray = NDArray[np.float64]

NUMBER_COUNT = 80
DRAW_SIZE = 20
FAIR_PROBABILITY = DRAW_SIZE / NUMBER_COUNT
DECAY_GRID = (0.97, 0.99, 0.995)
PRIOR_STRENGTH_GRID = (5.0, 20.0, 80.0)


@dataclass(frozen=True, order=True)
class BayesianParameters:
    """预注册的动态贝叶斯参数。"""

    decay: float
    prior_strength: float

    def __post_init__(self) -> None:
        if self.decay not in DECAY_GRID:
            raise ValueError(f"decay必须来自预注册网格：{DECAY_GRID}")
        if self.prior_strength not in PRIOR_STRENGTH_GRID:
            raise ValueError(f"prior_strength必须来自预注册网格：{PRIOR_STRENGTH_GRID}")


@dataclass(frozen=True)
class PosteriorPrediction:
    """单个预测时点的80维后验结果。"""

    alpha: FloatArray
    beta: FloatArray
    posterior_mean: FloatArray
    posterior_variance: FloatArray
    credible_interval_lower: FloatArray
    credible_interval_upper: FloatArray


def validate_indicator_history(history: FloatArray) -> FloatArray:
    """校验按时间升序排列的 ``(n, 80)`` 二元开奖矩阵。"""

    array = cast(FloatArray, np.asarray(history, dtype=np.float64))
    if array.ndim != 2 or array.shape[1] != NUMBER_COUNT:
        raise ValueError(f"历史矩阵必须为(n, 80)，收到：{array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("历史矩阵包含NaN或无穷值")
    if not np.logical_or(array == 0.0, array == 1.0).all():
        raise ValueError("历史矩阵只能包含0或1")
    if len(array) and not np.all(array.sum(axis=1) == DRAW_SIZE):
        raise ValueError("每期必须恰好包含20个出现号码")
    return array


def _posterior_from_sufficient_statistics(
    discounted_successes: FloatArray,
    discounted_total_weight: float,
    parameters: BayesianParameters,
) -> PosteriorPrediction:
    successes = cast(FloatArray, np.asarray(discounted_successes, dtype=np.float64))
    if successes.shape != (NUMBER_COUNT,):
        raise ValueError("折扣成功次数必须恰好包含80项")
    if discounted_total_weight < 0 or not np.isfinite(discounted_total_weight):
        raise ValueError("折扣总权重必须为有限非负数")
    if not np.isfinite(successes).all():
        raise ValueError("折扣成功次数包含NaN或无穷值")
    tolerance = 1e-10
    if np.any(successes < -tolerance) or np.any(
        successes > discounted_total_weight + tolerance
    ):
        raise ValueError("折扣成功次数超出合法范围")

    alpha = cast(
        FloatArray,
        parameters.prior_strength * FAIR_PROBABILITY + successes,
    )
    beta = cast(
        FloatArray,
        parameters.prior_strength * (1.0 - FAIR_PROBABILITY)
        + discounted_total_weight
        - successes,
    )
    total = cast(FloatArray, alpha + beta)
    posterior_mean = cast(FloatArray, alpha / total)
    posterior_variance = cast(
        FloatArray, alpha * beta / (np.square(total) * (total + 1.0))
    )
    credible_interval_lower = cast(
        FloatArray,
        np.asarray(beta_distribution.ppf(0.025, alpha, beta), dtype=np.float64),
    )
    credible_interval_upper = cast(
        FloatArray,
        np.asarray(beta_distribution.ppf(0.975, alpha, beta), dtype=np.float64),
    )

    arrays = (
        alpha,
        beta,
        posterior_mean,
        posterior_variance,
        credible_interval_lower,
        credible_interval_upper,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise FloatingPointError("后验计算产生NaN或无穷值")
    if not np.all((posterior_mean > 0.0) & (posterior_mean < 1.0)):
        raise FloatingPointError("后验均值必须严格位于(0, 1)")
    if not np.isclose(posterior_mean.sum(), DRAW_SIZE, rtol=0.0, atol=1e-9):
        raise FloatingPointError("80个后验概率之和不等于20")
    if not np.all(
        (credible_interval_lower > 0.0)
        & (credible_interval_lower < credible_interval_upper)
        & (credible_interval_upper < 1.0)
    ):
        raise FloatingPointError("95%可信区间超出合法范围")

    return PosteriorPrediction(
        alpha=alpha,
        beta=beta,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        credible_interval_lower=credible_interval_lower,
        credible_interval_upper=credible_interval_upper,
    )


def discounted_beta_bernoulli(
    history: FloatArray, parameters: BayesianParameters
) -> PosteriorPrediction:
    """只用给定历史计算目标期的动态折扣 Beta-Bernoulli 后验。

    对固定 ``decay``，后验均值是折扣出现次数的正仿射变换；所以更换
    ``prior_strength`` 不会改变完整80位排名，只会改变概率及不确定性。
    """

    array = validate_indicator_history(history)
    if len(array) == 0:
        successes = np.zeros(NUMBER_COUNT, dtype=np.float64)
        total_weight = 0.0
    else:
        ages = np.arange(len(array) - 1, -1, -1, dtype=np.float64)
        weights = cast(FloatArray, np.power(parameters.decay, ages))
        successes = cast(FloatArray, weights @ array)
        total_weight = float(weights.sum())
    return _posterior_from_sufficient_statistics(successes, total_weight, parameters)


def rank_probabilities(probabilities: FloatArray) -> NDArray[np.int64]:
    """按后验均值降序、同分号码升序返回1至80的完整排序。"""

    values = cast(FloatArray, np.asarray(probabilities, dtype=np.float64))
    if values.shape != (NUMBER_COUNT,) or not np.isfinite(values).all():
        raise ValueError("概率必须是80项有限数值")
    numbers = np.arange(1, NUMBER_COUNT + 1, dtype=np.int64)
    return cast(NDArray[np.int64], numbers[np.lexsort((numbers, -values))])


def candidate_sets(probabilities: FloatArray) -> dict[int, tuple[int, ...]]:
    """生成选一至选十的确定性候选集合。"""

    ranking = rank_probabilities(probabilities)
    return {play: tuple(map(int, ranking[:play])) for play in range(1, 11)}


def posterior_from_state(
    discounted_successes: FloatArray,
    discounted_total_weight: float,
    parameters: BayesianParameters,
) -> PosteriorPrediction:
    """供时间滚动评估复用的充分统计量入口。"""

    return _posterior_from_sufficient_statistics(
        discounted_successes, discounted_total_weight, parameters
    )


__all__ = [
    "DECAY_GRID",
    "DRAW_SIZE",
    "FAIR_PROBABILITY",
    "NUMBER_COUNT",
    "PRIOR_STRENGTH_GRID",
    "BayesianParameters",
    "PosteriorPrediction",
    "candidate_sets",
    "discounted_beta_bernoulli",
    "posterior_from_state",
    "rank_probabilities",
    "validate_indicator_history",
]
