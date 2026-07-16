"""只依赖预测时点之前历史的快乐8号码排序策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.analysis.feature_enhancer import compute_enhanced_scores

DETERMINISTIC_STRATEGIES = (
    "historical_frequency",
    "rolling_frequency",
    "exponential_frequency",
    "hybrid",
    "repository_advanced",
)


@dataclass(frozen=True)
class StrategyParameters:
    """经验证区间选择、随后冻结的策略参数。"""

    rolling_window: int = 120
    decay: float = 0.97
    hybrid_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)

    def __post_init__(self) -> None:
        if self.rolling_window <= 0:
            raise ValueError("rolling_window必须为正整数")
        if not 0 < self.decay <= 1:
            raise ValueError("decay必须在(0, 1]范围内")
        if len(self.hybrid_weights) != 3 or any(w < 0 for w in self.hybrid_weights):
            raise ValueError("hybrid_weights必须是三个非负权重")
        if not np.isclose(sum(self.hybrid_weights), 1.0):
            raise ValueError("hybrid_weights之和必须为1")


def validate_draws(draws: np.ndarray) -> np.ndarray:
    """校验并返回 ``(期数, 20)`` 的整数开奖号码矩阵。"""

    array = np.asarray(draws, dtype=int)
    if array.ndim != 2 or array.shape[1] != 20:
        raise ValueError(f"开奖号码矩阵必须是(n, 20)，收到：{array.shape}")
    if len(array) == 0:
        raise ValueError("开奖号码矩阵不能为空")
    if np.any((array < 1) | (array > 80)):
        raise ValueError("开奖号码必须位于1至80")
    if any(len(set(map(int, row))) != 20 for row in array):
        raise ValueError("每期20个开奖号码必须互不重复")
    return array


def _frequency_scores(history: np.ndarray) -> np.ndarray:
    counts = np.bincount(history.ravel(), minlength=81).astype(float)
    return counts[1:] / max(len(history), 1)


def _normalise(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    spread = scores.max() - scores.min()
    if spread <= 1e-12:
        return np.zeros_like(scores)
    return (scores - scores.min()) / spread


def score_numbers(
    history: np.ndarray,
    strategy: str,
    parameters: StrategyParameters,
) -> np.ndarray:
    """仅使用传入历史计算1至80的得分。

    调用方负责保证 ``history`` 不含目标期及更晚数据。高级策略显式禁用外部
    图嵌入缓存，因为缓存的训练截止点通常不可证明，可能造成未来数据泄漏。
    """

    history = validate_draws(history)
    if strategy == "repository_advanced":
        # feature_enhancer约定最近期在前，且每行第0列为期号占位。
        recent_first = history[::-1]
        synthetic_issues = np.arange(len(recent_first), 0, -1, dtype=int)[:, None]
        advanced_input = np.hstack([synthetic_issues, recent_first])
        ranked, _ = compute_enhanced_scores(
            advanced_input,
            limit=len(advanced_input),
            recent_window=40,
            reference_window=160,
            decay=0.97,
            use_pca=True,
            use_graph_embeddings=False,
        )
        scores = np.zeros(80, dtype=float)
        for number, value in ranked:
            scores[number - 1] = value
        return scores

    historical = _frequency_scores(history)
    if strategy == "historical_frequency":
        return historical

    rolling = _frequency_scores(history[-parameters.rolling_window :])
    if strategy == "rolling_frequency":
        return rolling

    ages = np.arange(len(history) - 1, -1, -1, dtype=float)
    weights = np.power(parameters.decay, ages)
    indicators = np.zeros((len(history), 80), dtype=float)
    indicators[np.arange(len(history))[:, None], history - 1] = 1.0
    exponential_counts = weights @ indicators
    exponential = exponential_counts / weights.sum()

    if strategy == "exponential_frequency":
        return exponential
    if strategy == "hybrid":
        w_history, w_rolling, w_exponential = parameters.hybrid_weights
        return (
            w_history * _normalise(historical)
            + w_rolling * _normalise(rolling)
            + w_exponential * _normalise(exponential)
        )
    raise ValueError(f"未知策略：{strategy}")


def ranked_numbers(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """按得分降序、号码升序做确定性排序。"""

    array = np.asarray(scores, dtype=float)
    if array.shape != (80,):
        raise ValueError("号码得分必须恰好包含80项")
    numbers = np.arange(1, 81)
    return numbers[np.lexsort((numbers, -array))]


def deterministic_tickets(
    scores: Sequence[float] | np.ndarray, play: int, ticket_mode: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """从同一预测时点得分生成两注合法票面。"""

    if play not in range(1, 11):
        raise ValueError("play必须为1至10")
    ranked = ranked_numbers(scores)
    first = tuple(sorted(map(int, ranked[:play])))
    if ticket_mode == "disjoint":
        second = tuple(sorted(map(int, ranked[play : 2 * play])))
    elif ticket_mode == "independent":
        # 两注分别优化且允许重叠；确定性得分下最优集合相同。
        second = tuple(sorted(map(int, ranked[:play])))
    else:
        raise ValueError(f"未知出票方式：{ticket_mode}")
    return first, second


def random_tickets(
    play: int,
    ticket_mode: str,
    *,
    seed: int,
    issue: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按seed、期号、玩法和出票方式确定性派生随机票面。"""

    if play not in range(1, 11):
        raise ValueError("play必须为1至10")
    mode_code = 0 if ticket_mode == "disjoint" else 1
    if ticket_mode not in {"disjoint", "independent"}:
        raise ValueError(f"未知出票方式：{ticket_mode}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, issue, play, mode_code]))
    if ticket_mode == "disjoint":
        chosen = rng.choice(np.arange(1, 81), size=2 * play, replace=False)
        first_values, second_values = chosen[:play], chosen[play:]
    else:
        first_values = rng.choice(np.arange(1, 81), size=play, replace=False)
        second_values = rng.choice(np.arange(1, 81), size=play, replace=False)
    return (
        tuple(sorted(map(int, first_values))),
        tuple(sorted(map(int, second_values))),
    )


def assert_legal_tickets(
    tickets: tuple[tuple[int, ...], tuple[int, ...]],
    play: int,
    ticket_mode: str,
) -> None:
    """验证注数、号码范围、注内唯一性及无重叠约束。"""

    if len(tickets) != 2:
        raise ValueError("每期必须恰好两注")
    for ticket in tickets:
        if len(ticket) != play or len(set(ticket)) != play:
            raise ValueError("票面号码数量或唯一性不合法")
        if any(number < 1 or number > 80 for number in ticket):
            raise ValueError("票面号码必须位于1至80")
    if ticket_mode == "disjoint" and set(tickets[0]) & set(tickets[1]):
        raise ValueError("无重叠出票方式的两注出现重复号码")


__all__ = [
    "DETERMINISTIC_STRATEGIES",
    "StrategyParameters",
    "assert_legal_tickets",
    "deterministic_tickets",
    "random_tickets",
    "ranked_numbers",
    "score_numbers",
    "validate_draws",
]
