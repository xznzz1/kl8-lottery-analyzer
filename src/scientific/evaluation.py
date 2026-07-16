"""固定预算快乐8的时间滚动评估与验证区间调参。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .prizes import PrizeScenario, prize_for_hits
from .strategies import (
    DETERMINISTIC_STRATEGIES,
    StrategyParameters,
    assert_legal_tickets,
    deterministic_tickets,
    random_tickets,
    score_numbers,
    validate_draws,
)

PRE_REGISTERED_WINDOWS = (30, 60, 120, 240)
PRE_REGISTERED_DECAYS = (0.90, 0.94, 0.97, 0.99)
PRE_REGISTERED_HYBRID_WEIGHTS = (
    (0.50, 0.30, 0.20),
    (0.34, 0.33, 0.33),
    (0.25, 0.50, 0.25),
    (0.25, 0.25, 0.50),
)


@dataclass(frozen=True)
class TemporalSplit:
    """开发、滚动验证和最终未接触holdout的边界。"""

    validation_start: int
    holdout_start: int
    total_issues: int

    @property
    def validation_indices(self) -> range:
        return range(self.validation_start, self.holdout_start)

    @property
    def holdout_indices(self) -> range:
        return range(self.holdout_start, self.total_issues)


def make_temporal_split(
    total_issues: int,
    *,
    holdout_fraction: float = 0.20,
    validation_fraction_of_pre_holdout: float = 0.20,
    minimum_history: int = 365,
) -> TemporalSplit:
    """建立固定时间切分；最后20%在参数冻结前不可见。"""

    if total_issues <= minimum_history + 2:
        raise ValueError("数据不足以同时建立历史、验证和holdout区间")
    holdout_size = max(1, int(round(total_issues * holdout_fraction)))
    holdout_start = total_issues - holdout_size
    validation_size = max(
        1, int(round(holdout_start * validation_fraction_of_pre_holdout))
    )
    validation_start = holdout_start - validation_size
    if validation_start < minimum_history:
        raise ValueError("切分后预测历史少于minimum_history")
    return TemporalSplit(validation_start, holdout_start, total_issues)


def load_history_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取CSV、严格校验并按期号从早到晚返回。"""

    frame = pd.read_csv(path)
    number_columns = [f"红球_{index}" for index in range(1, 21)]
    required = ["期数", *number_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"历史CSV缺少字段：{missing}")
    selected = frame[required].apply(pd.to_numeric, errors="raise")
    if selected.isna().any().any():
        raise ValueError("历史CSV包含缺失值")
    selected = selected.sort_values("期数", kind="stable").reset_index(drop=True)
    issues = selected["期数"].to_numpy(dtype=int)
    if len(np.unique(issues)) != len(issues):
        raise ValueError("历史CSV包含重复期号")
    if np.any(np.diff(issues) <= 0):
        raise ValueError("期号无法形成严格递增时间顺序")
    draws = validate_draws(selected[number_columns].to_numpy(dtype=int))
    return issues, draws


def _validation_hit_score(
    draws: np.ndarray,
    indices: Iterable[int],
    strategy: str,
    parameters: StrategyParameters,
) -> float:
    """预注册目标：所有玩法与两种出票方式的单注平均命中数。"""

    total_hits = 0
    total_bets = 0
    for target_index in indices:
        scores = score_numbers(draws[:target_index], strategy, parameters)
        winning = set(map(int, draws[target_index]))
        for play in range(1, 11):
            for ticket_mode in ("disjoint", "independent"):
                tickets = deterministic_tickets(scores, play, ticket_mode)
                assert_legal_tickets(tickets, play, ticket_mode)
                total_hits += sum(len(set(ticket) & winning) for ticket in tickets)
                total_bets += 2
    return total_hits / total_bets


def tune_on_validation(
    draws: np.ndarray,
    split: TemporalSplit,
) -> tuple[StrategyParameters, pd.DataFrame]:
    """仅在滚动验证区间上选择预注册网格，随后冻结参数。"""

    draws = validate_draws(draws)
    validation = tuple(split.validation_indices)
    rows: list[dict[str, object]] = []

    window_scores: list[float] = []
    for window in PRE_REGISTERED_WINDOWS:
        parameters = StrategyParameters(rolling_window=window)
        score = _validation_hit_score(
            draws, validation, "rolling_frequency", parameters
        )
        window_scores.append(score)
        rows.append(
            {
                "parameter": "rolling_window",
                "value": str(window),
                "validation_mean_hits_per_bet": score,
            }
        )
    selected_window = PRE_REGISTERED_WINDOWS[int(np.argmax(window_scores))]

    decay_scores: list[float] = []
    for decay in PRE_REGISTERED_DECAYS:
        parameters = StrategyParameters(rolling_window=selected_window, decay=decay)
        score = _validation_hit_score(
            draws, validation, "exponential_frequency", parameters
        )
        decay_scores.append(score)
        rows.append(
            {
                "parameter": "decay",
                "value": str(decay),
                "validation_mean_hits_per_bet": score,
            }
        )
    selected_decay = PRE_REGISTERED_DECAYS[int(np.argmax(decay_scores))]

    hybrid_scores: list[float] = []
    for weights in PRE_REGISTERED_HYBRID_WEIGHTS:
        parameters = StrategyParameters(selected_window, selected_decay, weights)
        score = _validation_hit_score(draws, validation, "hybrid", parameters)
        hybrid_scores.append(score)
        rows.append(
            {
                "parameter": "hybrid_weights(history,rolling,decay)",
                "value": "/".join(f"{weight:.2f}" for weight in weights),
                "validation_mean_hits_per_bet": score,
            }
        )
    selected_weights = PRE_REGISTERED_HYBRID_WEIGHTS[int(np.argmax(hybrid_scores))]
    selected = StrategyParameters(selected_window, selected_decay, selected_weights)

    sensitivity = pd.DataFrame(rows)
    selected_values = {
        "rolling_window": str(selected_window),
        "decay": str(selected_decay),
        "hybrid_weights(history,rolling,decay)": "/".join(
            f"{weight:.2f}" for weight in selected_weights
        ),
    }
    sensitivity["selected"] = sensitivity.apply(
        lambda row: str(row["value"]) == selected_values[str(row["parameter"])], axis=1
    )
    return selected, sensitivity


def _record_for_tickets(
    *,
    issue: int,
    draw: np.ndarray,
    strategy: str,
    seed: int | None,
    play: int,
    ticket_mode: str,
    tickets: tuple[tuple[int, ...], tuple[int, ...]],
    scenario: PrizeScenario,
) -> dict[str, object]:
    winning = set(map(int, draw))
    hits1 = len(set(tickets[0]) & winning)
    hits2 = len(set(tickets[1]) & winning)
    prize1 = prize_for_hits(play, hits1, scenario, issue=issue)
    prize2 = prize_for_hits(play, hits2, scenario, issue=issue)
    return {
        "issue": issue,
        "strategy": strategy,
        "seed": seed,
        "play": play,
        "ticket_mode": ticket_mode,
        "ticket1": " ".join(map(str, tickets[0])),
        "ticket2": " ".join(map(str, tickets[1])),
        "hits1": hits1,
        "hits2": hits2,
        "prize1": prize1,
        "prize2": prize2,
        "total_prize": prize1 + prize2,
    }


def evaluate_indices(
    issues: np.ndarray,
    draws: np.ndarray,
    indices: Sequence[int] | range,
    parameters: StrategyParameters,
    scenario: PrizeScenario,
    *,
    random_seeds: Sequence[int],
) -> pd.DataFrame:
    """逐期预测；目标期 ``t`` 的策略输入严格为 ``draws[:t]``。"""

    draws = validate_draws(draws)
    if len(issues) != len(draws):
        raise ValueError("期号与开奖记录长度不一致")
    rows: list[dict[str, object]] = []
    for target_index in indices:
        if target_index <= 0 or target_index >= len(draws):
            raise ValueError(f"目标索引越界：{target_index}")
        issue = int(issues[target_index])
        draw = draws[target_index]
        history = draws[:target_index]

        for strategy in DETERMINISTIC_STRATEGIES:
            scores = score_numbers(history, strategy, parameters)
            for play in range(1, 11):
                for ticket_mode in ("disjoint", "independent"):
                    tickets = deterministic_tickets(scores, play, ticket_mode)
                    assert_legal_tickets(tickets, play, ticket_mode)
                    rows.append(
                        _record_for_tickets(
                            issue=issue,
                            draw=draw,
                            strategy=strategy,
                            seed=None,
                            play=play,
                            ticket_mode=ticket_mode,
                            tickets=tickets,
                            scenario=scenario,
                        )
                    )

        for seed in random_seeds:
            for play in range(1, 11):
                for ticket_mode in ("disjoint", "independent"):
                    tickets = random_tickets(
                        play, ticket_mode, seed=int(seed), issue=issue
                    )
                    assert_legal_tickets(tickets, play, ticket_mode)
                    rows.append(
                        _record_for_tickets(
                            issue=issue,
                            draw=draw,
                            strategy="uniform_random",
                            seed=int(seed),
                            play=play,
                            ticket_mode=ticket_mode,
                            tickets=tickets,
                            scenario=scenario,
                        )
                    )
    return pd.DataFrame(rows)


__all__ = [
    "PRE_REGISTERED_DECAYS",
    "PRE_REGISTERED_HYBRID_WEIGHTS",
    "PRE_REGISTERED_WINDOWS",
    "TemporalSplit",
    "evaluate_indices",
    "load_history_csv",
    "make_temporal_split",
    "tune_on_validation",
]
