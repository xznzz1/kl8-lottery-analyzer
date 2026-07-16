"""快乐8 v2 的概率、校准与 Top-k 指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LOG_LOSS_EPSILON = 1e-9
DEFAULT_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class CalibrationBin:
    """一个固定概率分箱的校准统计。"""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_probability: float | None
    actual_rate: float | None
    absolute_gap: float | None
    weighted_gap: float


@dataclass(frozen=True)
class CalibrationResult:
    """完整固定分箱及 expected calibration error。"""

    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float


def _validated_pair(
    probabilities: FloatArray, outcomes: FloatArray
) -> tuple[FloatArray, FloatArray]:
    predicted = cast(FloatArray, np.asarray(probabilities, dtype=np.float64))
    actual = cast(FloatArray, np.asarray(outcomes, dtype=np.float64))
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("概率和结果必须是形状相同的非空数组")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("概率和结果不能包含NaN或无穷值")
    if np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("概率必须位于[0, 1]")
    if not np.logical_or(actual == 0.0, actual == 1.0).all():
        raise ValueError("结果只能包含0或1")
    return predicted, actual


def brier_score(probabilities: FloatArray, outcomes: FloatArray) -> float:
    """返回80维或任意同形数组的逐元素均值 Brier score。"""

    predicted, actual = _validated_pair(probabilities, outcomes)
    return float(np.mean(np.square(predicted - actual)))


def bernoulli_log_loss(
    probabilities: FloatArray,
    outcomes: FloatArray,
    *,
    epsilon: float = LOG_LOSS_EPSILON,
) -> float:
    """先截断概率，再计算数值稳定的 Bernoulli log loss。"""

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon必须位于(0, 0.5)")
    predicted, actual = _validated_pair(probabilities, outcomes)
    clipped = cast(FloatArray, np.clip(predicted, epsilon, 1.0 - epsilon))
    losses = cast(
        FloatArray,
        -(actual * np.log(clipped) + (1.0 - actual) * np.log1p(-clipped)),
    )
    value = float(np.mean(losses))
    if not np.isfinite(value):
        raise FloatingPointError("log loss产生NaN或无穷值")
    return value


def calibration_summary(
    probabilities: FloatArray,
    outcomes: FloatArray,
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationResult:
    """使用覆盖[0, 1]的固定等宽分箱计算校准和 ECE。"""

    if bins <= 1:
        raise ValueError("校准分箱数必须大于1")
    predicted, actual = _validated_pair(probabilities, outcomes)
    predicted_flat = predicted.ravel()
    actual_flat = actual.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    assignments = np.searchsorted(edges, predicted_flat, side="right") - 1
    assignments = np.clip(assignments, 0, bins - 1)
    total = predicted_flat.size
    rows: list[CalibrationBin] = []
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if count == 0:
            rows.append(
                CalibrationBin(
                    bin_index=index,
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    mean_predicted_probability=None,
                    actual_rate=None,
                    absolute_gap=None,
                    weighted_gap=0.0,
                )
            )
            continue
        mean_probability = float(predicted_flat[mask].mean())
        actual_rate = float(actual_flat[mask].mean())
        gap = abs(mean_probability - actual_rate)
        weighted_gap = gap * count / total
        ece += weighted_gap
        rows.append(
            CalibrationBin(
                bin_index=index,
                lower_bound=lower,
                upper_bound=upper,
                count=count,
                mean_predicted_probability=mean_probability,
                actual_rate=actual_rate,
                absolute_gap=gap,
                weighted_gap=weighted_gap,
            )
        )
    return CalibrationResult(bins=tuple(rows), expected_calibration_error=float(ece))


def top_k_hits(
    ranking: NDArray[np.int64], outcomes: FloatArray, k: int
) -> tuple[int, float, float]:
    """返回实际命中数、随机理论期望 ``k/4`` 和超额命中。"""

    ordered = cast(NDArray[np.int64], np.asarray(ranking, dtype=np.int64))
    actual = cast(FloatArray, np.asarray(outcomes, dtype=np.float64))
    if ordered.shape != (80,) or set(map(int, ordered)) != set(range(1, 81)):
        raise ValueError("ranking必须是1至80的完整排列")
    if actual.shape != (80,) or not np.logical_or(actual == 0.0, actual == 1.0).all():
        raise ValueError("outcomes必须是80项二元结果")
    if k not in range(1, 11):
        raise ValueError("k必须为1至10")
    hits = int(actual[ordered[:k] - 1].sum())
    expected = k / 4.0
    return hits, expected, hits - expected


__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "LOG_LOSS_EPSILON",
    "CalibrationBin",
    "CalibrationResult",
    "bernoulli_log_loss",
    "brier_score",
    "calibration_summary",
    "top_k_hits",
]
