# -*- coding: utf-8 -*-
"""
特征增强工具集。

该模块面向 `kl8_analysis*.py`，提供基于最新开奖历史的高级特征分析能力，
包括：
1. 近期趋势动量分析；
2. 号码共现谱分析；
3. 组合分数的归一化与汇总。

实现目标：
- 提升号码挑选阶段的特征维度；
- 提供结构化的调试信息，便于在日志中输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Number = int
Score = float


@dataclass(frozen=True)
class FeatureDebugInfo:
    """用于日志/调试的结构化特征信息。"""

    recency_scores: Dict[Number, Score]
    momentum_scores: Dict[Number, Score]
    co_occurrence_scores: Dict[Number, Score]
    combined_scores: List[Tuple[Number, Score]]


def _iter_recent_draws(draws: np.ndarray, limit: int) -> Iterable[np.ndarray]:
    """返回按时间倒序截断后的开奖数据。"""

    if limit <= 0 or limit > len(draws):
        limit = len(draws)
    for row in draws[:limit]:
        yield row


def _normalise(values: np.ndarray) -> np.ndarray:
    """将数组缩放到 0-1 区间，避免除零报错。"""

    vmax = values.max()
    vmin = values.min()
    if vmax - vmin < 1e-9:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _extract_numbers(draw_row: Sequence[Number]) -> List[Number]:
    """去除行首的期号，仅返回 20 个开奖号码。"""

    return [int(n) for n in draw_row[1:] if int(n) > 0]


def compute_recency_and_momentum_scores(
    draws: np.ndarray,
    limit: int,
    recent_window: int = 40,
    reference_window: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算近期频率与动量得分。

    - `recent_window`：近期窗口长度，默认 40 期；
    - `reference_window`：参考窗口长度，默认 160 期。
    """

    max_number = 80
    recency_counts = np.zeros(max_number + 1, dtype=float)
    reference_counts = np.zeros(max_number + 1, dtype=float)

    limited_draws = list(_iter_recent_draws(draws, limit))
    recent_slice = limited_draws[:recent_window]
    reference_slice = limited_draws[:reference_window]

    # 近期权重：越新的期数权重越大
    for idx, row in enumerate(recent_slice):
        weight = 1.0 - (idx / max(len(recent_slice), 1))
        for value in _extract_numbers(row):
            recency_counts[value] += max(weight, 0.1)

    # 长期基线
    for row in reference_slice:
        for value in _extract_numbers(row):
            reference_counts[value] += 1.0

    recency_scores = _normalise(recency_counts)
    momentum_raw = recency_counts - reference_counts * (
        len(recent_slice) / max(len(reference_slice), 1)
    )
    momentum_scores = _normalise(momentum_raw)

    return recency_scores, momentum_scores


def compute_co_occurrence_scores(
    draws: np.ndarray,
    limit: int,
    decay: float = 0.97,
) -> np.ndarray:
    """
    基于共现矩阵的谱分析得分。

    参数：
        draws: 历史开奖二维数组。
        limit: 统计范围（按最新往前）。
        decay: 权重衰减因子，越接近 1 越强调新数据。
    """

    max_number = 80
    matrix = np.zeros((max_number + 1, max_number + 1), dtype=float)

    for idx, row in enumerate(_iter_recent_draws(draws, limit)):
        numbers = _extract_numbers(row)
        if len(numbers) < 2:
            continue
        weight = decay ** idx
        for i in range(len(numbers)):
            ni = numbers[i]
            for j in range(i + 1, len(numbers)):
                nj = numbers[j]
                matrix[ni, nj] += weight
                matrix[nj, ni] += weight

    if not np.any(matrix):
        return np.zeros(max_number + 1, dtype=float)

    # 使用特征向量对应的主成分作为中心性指标
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    principal_vector = np.abs(eigenvectors[:, -1])
    co_occurrence_scores = _normalise(principal_vector)
    return co_occurrence_scores


def compute_enhanced_scores(
    draws: np.ndarray,
    limit: int,
    recent_window: int = 40,
    reference_window: int = 160,
    decay: float = 0.97,
    weights: Tuple[float, float, float] = (0.45, 0.25, 0.30),
) -> Tuple[List[Tuple[Number, Score]], FeatureDebugInfo]:
    """
    汇总多源特征，返回排序后的号码得分列表与调试信息。
    """

    if draws.size == 0:
        empty = FeatureDebugInfo({}, {}, {}, [])
        return [], empty

    recency_scores, momentum_scores = compute_recency_and_momentum_scores(
        draws,
        limit=limit,
        recent_window=recent_window,
        reference_window=reference_window,
    )
    co_occurrence_scores = compute_co_occurrence_scores(
        draws,
        limit=limit,
        decay=decay,
    )

    w_recency, w_momentum, w_co = weights
    combined = (
        w_recency * recency_scores
        + w_momentum * momentum_scores
        + w_co * co_occurrence_scores
    )

    ranked = sorted(
        ((number, float(combined[number])) for number in range(1, 81)),
        key=lambda item: item[1],
        reverse=True,
    )

    debug_info = FeatureDebugInfo(
        recency_scores={i: float(recency_scores[i]) for i in range(1, 81)},
        momentum_scores={i: float(momentum_scores[i]) for i in range(1, 81)},
        co_occurrence_scores={i: float(co_occurrence_scores[i]) for i in range(1, 81)},
        combined_scores=ranked,
    )

    return ranked, debug_info

