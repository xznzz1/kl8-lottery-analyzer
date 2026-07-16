"""科学回测的汇总、置信区间与配对随机化检验。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def wilson_score_interval(
    successes: float, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """返回比例的Wilson 95%区间，零事件时仍给出正上界。

    ``successes``允许为小数，以支持把每期跨seed事件率作为一个聚类观测；
    此时``trials``必须是独立期数，而不是seed×期数。
    """

    if trials <= 0 or not 0.0 <= successes <= trials:
        raise ValueError("Wilson区间要求0<=成功数<=正试验数")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))


def _bootstrap_means(
    arrays: dict[str, np.ndarray], samples: int, seed: int
) -> dict[str, tuple[float, float]]:
    """按期有放回抽样，返回各均值指标的95%百分位区间。"""

    if samples <= 0:
        return {name: (float("nan"), float("nan")) for name in arrays}
    length = len(next(iter(arrays.values())))
    if length == 0 or any(len(values) != length for values in arrays.values()):
        raise ValueError("bootstrap输入必须是等长非空数组")
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[np.ndarray]] = {name: [] for name in arrays}
    chunk_size = min(250, samples)
    remaining = samples
    while remaining:
        size = min(chunk_size, remaining)
        indices = rng.integers(0, length, size=(size, length))
        for name, values in arrays.items():
            estimates[name].append(values[indices].mean(axis=1))
        remaining -= size
    return {
        name: _percentile_interval(np.concatenate(chunks))
        for name, chunks in estimates.items()
    }


def longest_true_run(values: Iterable[bool]) -> int:
    """返回布尔序列中最长连续True长度。"""

    longest = current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def risk_metrics(
    total_prizes: np.ndarray, budget_per_issue: float = 4.0
) -> dict[str, float | int]:
    """计算以每期净收益为基础的最大回撤与最长连续亏损期。"""

    prizes = np.asarray(total_prizes, dtype=float)
    net = prizes - budget_per_issue
    cumulative = np.cumsum(net)
    with_origin = np.concatenate(([0.0], cumulative))
    peaks = np.maximum.accumulate(with_origin)
    drawdowns = peaks[1:] - cumulative
    return {
        "max_drawdown": float(drawdowns.max(initial=0.0)),
        "longest_losing_streak": longest_true_run(prizes < budget_per_issue),
        "ending_profit": float(net.sum()),
    }


def summarise_period_records(
    records: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260716,
) -> dict[str, float | int]:
    """汇总一个策略/玩法/出票方式在连续若干期上的指标。"""

    required = {"issue", "hits1", "hits2", "prize1", "prize2", "total_prize"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"回测记录缺少字段：{sorted(missing)}")
    if records.empty or records["issue"].duplicated().any():
        raise ValueError("汇总输入必须每期恰好一行且不能为空")

    hits = (records["hits1"].to_numpy(float) + records["hits2"].to_numpy(float)) / 2.0
    prizes = records["total_prize"].to_numpy(float)
    any_prize = (prizes > 0).astype(float)
    profit = (prizes > 4.0).astype(float)
    p100 = (prizes >= 100.0).astype(float)
    p1000 = (prizes >= 1000.0).astype(float)
    p10000 = (prizes >= 10000.0).astype(float)
    roi_per_issue = prizes / 4.0 - 1.0

    bootstrap_intervals = _bootstrap_means(
        {
            "mean_hits_per_bet": hits,
            "expected_prize": prizes,
            "roi": roi_per_issue,
        },
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    event_intervals = {
        name: wilson_score_interval(float(values.sum()), len(values))
        for name, values in {
            "any_prize_probability": any_prize,
            "profit_probability": profit,
            "prize_at_least_100_probability": p100,
            "prize_at_least_1000_probability": p1000,
            "prize_at_least_10000_probability": p10000,
        }.items()
    }
    risk = risk_metrics(prizes)
    result: dict[str, float | int] = {
        "issues": int(len(records)),
        "mean_hits_per_bet": float(hits.mean()),
        "any_prize_probability": float(any_prize.mean()),
        "profit_probability": float(profit.mean()),
        "expected_prize": float(prizes.mean()),
        "roi": float(roi_per_issue.mean()),
        "prize_at_least_100_probability": float(p100.mean()),
        "prize_at_least_1000_probability": float(p1000.mean()),
        "prize_at_least_10000_probability": float(p10000.mean()),
        **risk,
    }
    for name, (low, high) in {**bootstrap_intervals, **event_intervals}.items():
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high
    return result


def summarise_seed_ensemble(
    records: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260716,
) -> dict[str, float | int]:
    """汇总多seed随机基线，不对奖金均值做非线性事件判定。

    概率和均值先在每个seed×期上计算，再按期平均；风险指标则先沿每个
    seed的真实奖金路径计算，最后对seed取平均，避免构造不存在的“均值票”。
    """

    required = {
        "issue",
        "seed",
        "hits1",
        "hits2",
        "prize1",
        "prize2",
        "total_prize",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"随机ensemble记录缺少字段：{sorted(missing)}")
    if records.empty or records["seed"].isna().any():
        raise ValueError("随机ensemble输入必须包含非空seed记录")
    if records.duplicated(["seed", "issue"]).any():
        raise ValueError("随机ensemble中每个seed×期必须恰好一行")

    seed_count = int(records["seed"].nunique())
    issue_count = int(records["issue"].nunique())
    per_seed_counts = records.groupby("seed")["issue"].nunique()
    per_issue_counts = records.groupby("issue")["seed"].nunique()
    if (
        not (per_seed_counts == issue_count).all()
        or not (per_issue_counts == seed_count).all()
    ):
        raise ValueError("随机ensemble的所有seed必须覆盖完全相同的期号")

    derived = pd.DataFrame({"issue": records["issue"].to_numpy()})
    prizes = records["total_prize"].to_numpy(float)
    derived["mean_hits_per_bet"] = (
        records["hits1"].to_numpy(float) + records["hits2"].to_numpy(float)
    ) / 2.0
    derived["any_prize_probability"] = (prizes > 0).astype(float)
    derived["profit_probability"] = (prizes > 4.0).astype(float)
    derived["expected_prize"] = prizes
    derived["roi"] = prizes / 4.0 - 1.0
    derived["prize_at_least_100_probability"] = (prizes >= 100.0).astype(float)
    derived["prize_at_least_1000_probability"] = (prizes >= 1000.0).astype(float)
    derived["prize_at_least_10000_probability"] = (prizes >= 10000.0).astype(float)
    per_issue = derived.groupby("issue", sort=True).mean()

    bootstrap_columns = {"mean_hits_per_bet", "expected_prize", "roi"}
    bootstrap_intervals = _bootstrap_means(
        {name: per_issue[name].to_numpy(float) for name in bootstrap_columns},
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    event_intervals = {
        name: wilson_score_interval(float(per_issue[name].sum()), issue_count)
        for name in per_issue.columns
        if name not in bootstrap_columns
    }
    seed_risks = [
        risk_metrics(group.sort_values("issue")["total_prize"].to_numpy(float))
        for _, group in records.groupby("seed", sort=True)
    ]
    result: dict[str, float | int] = {
        "issues": issue_count,
        **{name: float(per_issue[name].mean()) for name in per_issue.columns},
        "max_drawdown": float(np.mean([risk["max_drawdown"] for risk in seed_risks])),
        "longest_losing_streak": float(
            np.mean([risk["longest_losing_streak"] for risk in seed_risks])
        ),
        "ending_profit": float(np.mean([risk["ending_profit"] for risk in seed_risks])),
    }
    for name, (low, high) in {**bootstrap_intervals, **event_intervals}.items():
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high
    return result


def paired_randomisation_test(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int = 20_000,
    seed: int = 20260716,
) -> tuple[float, float]:
    """配对符号翻转检验：备择假设为candidate均值高于baseline。

    该检验把配对差值视为在零假设下可交换符号；严格有效性依赖差值分布
    对称/符号可交换，调用方必须把这一条件作为推断限制披露。
    """

    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or len(candidate) == 0:
        raise ValueError("配对检验要求两个等长非空一维数组")
    differences = candidate - baseline
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = samples
    while remaining:
        size = min(1000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(differences)))
        simulated = (signs * differences).mean(axis=1)
        exceedances += int(np.count_nonzero(simulated >= observed - 1e-15))
        remaining -= size
    return observed, (exceedances + 1) / (samples + 1)


def paired_mean_bootstrap_interval(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 20260716,
) -> tuple[float, float]:
    """对逐期配对均值差按期重抽样，返回95%百分位区间。"""

    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or len(candidate) == 0:
        raise ValueError("配对区间要求两个等长非空一维数组")
    return _bootstrap_means(
        {"effect": candidate - baseline}, samples=samples, seed=seed
    )["effect"]


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm逐步校正，控制多玩法、多策略和出票方式的家族错误率。"""

    values = np.asarray(list(p_values), dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p值必须位于0至1")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def hit_distribution(
    records: pd.DataFrame, max_hits: int | None = None
) -> pd.DataFrame:
    """输出两注合并后的单注命中数分布，包含零计数档位。"""

    values = pd.concat([records["hits1"], records["hits2"]], ignore_index=True)
    counts = values.value_counts().sort_index()
    if max_hits is not None:
        if max_hits < 0:
            raise ValueError("max_hits不能为负数")
        counts = counts.reindex(range(max_hits + 1), fill_value=0)
    return pd.DataFrame(
        {
            "hit_count": counts.index.astype(int),
            "ticket_count": counts.to_numpy(int),
            "probability": counts.to_numpy(float) / len(values),
        }
    )


__all__ = [
    "hit_distribution",
    "holm_adjust",
    "longest_true_run",
    "paired_mean_bootstrap_interval",
    "paired_randomisation_test",
    "risk_metrics",
    "summarise_period_records",
    "summarise_seed_ensemble",
    "wilson_score_interval",
]
