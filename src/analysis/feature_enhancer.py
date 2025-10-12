# -*- coding: utf-8 -*-
"""
特征增强工具集
----------------
面向 `kl8_analysis*.py` 脚本提供基于历史开奖的高级特征计算：
1. 近期频率与动量；
2. 号码共现谱分析；
3. Dirichlet-Multinomial 分层平滑；
4. PCA 主成分辅助特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover - optional依赖
    PCA = None

try:
    from ..config import DIRICHLET_CONFIG
except Exception:  # pragma: no cover - 脚本直跑时的路径回退
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import DIRICHLET_CONFIG  # type: ignore


Number = int
Score = float


@dataclass(frozen=True)
class FeatureDebugInfo:
    """用于日志输出的结构化特征信息。"""

    recency_scores: Dict[Number, Score]
    momentum_scores: Dict[Number, Score]
    co_occurrence_scores: Dict[Number, Score]
    dirichlet_mean: Dict[Number, Score]
    dirichlet_variance: Dict[Number, Score]
    dirichlet_scores: Dict[Number, Score]
    combined_scores: List[Tuple[Number, Score]]


def _iter_recent_draws(draws: np.ndarray, limit: int) -> Iterable[np.ndarray]:
    """返回按时间倒序截断后的开奖数据。"""

    if limit <= 0 or limit > len(draws):
        limit = len(draws)
    for row in draws[:limit]:
        yield row


def _normalise(values: np.ndarray) -> np.ndarray:
    """将数组缩放到 0-1 区间，避免除零异常。"""

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

    - ``recent_window``：近期窗口长度；
    - ``reference_window``：长期参考窗口长度。
    """

    max_number = 80
    recency_counts = np.zeros(max_number + 1, dtype=float)
    reference_counts = np.zeros(max_number + 1, dtype=float)

    limited_draws = list(_iter_recent_draws(draws, limit))
    recent_slice = limited_draws[:recent_window]
    reference_slice = limited_draws[:reference_window]

    for idx, row in enumerate(recent_slice):
        weight = 1.0 - (idx / max(len(recent_slice), 1))
        for value in _extract_numbers(row):
            recency_counts[value] += max(weight, 0.1)

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
    """构建共现矩阵并返回谱中心性得分。"""

    max_number = 80
    matrix = np.zeros((max_number + 1, max_number + 1), dtype=float)

    for idx, row in enumerate(_iter_recent_draws(draws, limit)):
        numbers = _extract_numbers(row)
        if len(numbers) < 2:
            continue
        weight = decay**idx
        for i in range(len(numbers)):
            ni = numbers[i]
            for j in range(i + 1, len(numbers)):
                nj = numbers[j]
                matrix[ni, nj] += weight
                matrix[nj, ni] += weight

    if not np.any(matrix):
        return np.zeros(max_number + 1, dtype=float)

    _, eigenvectors = np.linalg.eigh(matrix)
    principal_vector = np.abs(eigenvectors[:, -1])
    return _normalise(principal_vector)


def _compute_dirichlet_scores(
    draws: np.ndarray,
    limit: int,
    window_size: int,
    prior_strength: float,
    variance_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于 Dirichlet-Multinomial 后验计算得分、均值与方差。"""

    max_number = 80
    effective_limit = min(limit, draws.shape[0])
    local_counts = np.zeros(max_number + 1, dtype=float)
    global_counts = np.zeros(max_number + 1, dtype=float)

    for idx, row in enumerate(_iter_recent_draws(draws, effective_limit)):
        numbers = _extract_numbers(row)
        for value in numbers:
            global_counts[value] += 1.0
            if idx < window_size:
                local_counts[value] += 1.0

    global_total = global_counts.sum()
    if global_total <= 0:
        base_prior = np.ones(max_number + 1, dtype=float)
    else:
        freq = global_counts / global_total
        base_prior = np.maximum(freq * prior_strength * max_number, 1e-6)

    posterior_alpha = base_prior + local_counts
    alpha_sum = posterior_alpha.sum()
    if alpha_sum <= 0:
        zero = np.zeros(max_number + 1, dtype=float)
        return zero, zero, zero

    posterior_mean = posterior_alpha / alpha_sum
    variance = (posterior_alpha * (alpha_sum - posterior_alpha)) / (alpha_sum**2 * (alpha_sum + 1.0))
    adjusted = posterior_mean - variance_weight * np.sqrt(np.maximum(variance, 0.0))
    dirichlet_scores = _normalise(adjusted)

    return dirichlet_scores, posterior_mean, variance


def compute_enhanced_scores(
    draws: np.ndarray,
    limit: int,
    recent_window: int = 40,
    reference_window: int = 160,
    decay: float = 0.97,
    weights: Tuple[float, float, float] = (0.45, 0.25, 0.30),
    dirichlet_weight: float | None = None,
    pca_components: int = 1,
    use_pca: bool = True,
) -> Tuple[List[Tuple[Number, Score]], FeatureDebugInfo]:
    """汇总多源特征，返回排序结果与调试信息。"""

    if draws.size == 0:
        empty = FeatureDebugInfo({}, {}, {}, {}, {}, {}, [])
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

    dirichlet_scores, dirichlet_mean, dirichlet_variance = _compute_dirichlet_scores(
        draws=draws,
        limit=limit,
        window_size=max(1, min(DIRICHLET_CONFIG["window_size"], limit)),
        prior_strength=max(DIRICHLET_CONFIG["prior_strength"], 1e-6),
        variance_weight=max(DIRICHLET_CONFIG["variance_weight"], 0.0),
    )

    pca_scores = np.zeros(81, dtype=float)
    if use_pca and PCA is not None:
        numbers_matrix = np.zeros((min(limit, draws.shape[0]), 80), dtype=float)
        for idx, row in enumerate(draws[: numbers_matrix.shape[0]]):
            nums = set(_extract_numbers(row))
            for n in nums:
                numbers_matrix[idx, n - 1] = 1.0
        try:
            pca = PCA(n_components=pca_components)
            pca.fit(numbers_matrix)
            if pca_components == 1:
                pc1 = pca.components_[0]
                pc1_norm = _normalise(pc1)
                for n in range(1, 81):
                    pca_scores[n] = float(pc1_norm[n - 1])
            else:
                pc_sum = np.sum(np.abs(pca.components_), axis=0)
                pc_sum_norm = _normalise(pc_sum)
                for n in range(1, 81):
                    pca_scores[n] = float(pc_sum_norm[n - 1])
        except Exception:  # pragma: no cover - 防御 fallback
            pca_scores = np.zeros(81, dtype=float)

    w_recency, w_momentum, w_co = weights
    w_pca = 0.18
    w_dirichlet = dirichlet_weight if dirichlet_weight is not None else 0.22

    combined = (
        w_recency * recency_scores
        + w_momentum * momentum_scores
        + w_co * co_occurrence_scores
        + w_dirichlet * dirichlet_scores
        + w_pca * pca_scores
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
        dirichlet_mean={i: float(dirichlet_mean[i]) for i in range(1, 81)},
        dirichlet_variance={i: float(dirichlet_variance[i]) for i in range(1, 81)},
        dirichlet_scores={i: float(dirichlet_scores[i]) for i in range(1, 81)},
        combined_scores=ranked,
    )

    return ranked, debug_info

