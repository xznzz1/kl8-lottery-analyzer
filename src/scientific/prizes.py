"""快乐8当前规则下的单注奖金与两注精确概率。

固定奖表采用 2026 年已执行规则。选九中九和选十中十为浮动奖，必须由
``PrizeScenario`` 显式给出情景值；因此本模块计算的是规则情景，不是历史实付奖金。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Dict, Iterable, Mapping

OLD_FIXED_PRIZES: Dict[int, Dict[int, float]] = {
    1: {1: 4.6},
    2: {2: 19.0},
    3: {3: 53.0, 2: 3.0},
    4: {4: 100.0, 3: 5.0, 2: 3.0},
    5: {5: 1000.0, 4: 21.0, 3: 3.0},
    6: {6: 3000.0, 5: 30.0, 4: 10.0, 3: 3.0},
    7: {7: 10000.0, 6: 288.0, 5: 28.0, 4: 4.0, 0: 2.0},
    8: {8: 50000.0, 7: 800.0, 6: 88.0, 5: 10.0, 4: 3.0, 0: 2.0},
    9: {9: 300000.0, 8: 2000.0, 7: 200.0, 6: 20.0, 5: 5.0, 4: 3.0, 0: 2.0},
    10: {9: 8000.0, 8: 800.0, 7: 80.0, 6: 5.0, 5: 3.0, 0: 2.0},
}


CURRENT_FIXED_PRIZES: Dict[int, Dict[int, float]] = {
    1: {1: 4.5},
    2: {2: 19.0},
    3: {3: 52.0, 2: 3.0},
    4: {4: 93.0, 3: 5.0, 2: 3.0},
    5: {5: 1000.0, 4: 20.0, 3: 3.0},
    6: {6: 2880.0, 5: 30.0, 4: 10.0, 3: 3.0},
    7: {7: 8500.0, 6: 300.0, 5: 30.0, 4: 4.0, 0: 2.0},
    8: {8: 50000.0, 7: 800.0, 6: 80.0, 5: 10.0, 4: 3.0, 0: 2.0},
    9: {8: 2000.0, 7: 225.0, 6: 22.0, 5: 5.0, 4: 3.0, 0: 2.0},
    10: {9: 8000.0, 8: 720.0, 7: 80.0, 6: 5.0, 5: 3.0, 0: 2.0},
}

# 2025-12-30开奖的第2025350期起执行财综〔2025〕46号批准的新规则。
CURRENT_RULE_START_ISSUE = 2025350


class MissingFloatingPrizeError(ValueError):
    """需要浮动奖但未提供逐期值或明确情景时抛出。"""


@dataclass(frozen=True)
class PrizeScenario:
    """浮动奖情景。

    默认不提供任何浮动奖，命中时会明确失败。封顶情景只能通过
    :meth:`cap_scenario` 显式构造，且不代表每期实际兑付。
    """

    pick9_jackpot: float | None = None
    pick10_jackpot: float | None = None
    label: str = "missing_floating_prizes"
    per_issue: Mapping[int, Mapping[int, float]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        values = (self.pick9_jackpot, self.pick10_jackpot)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("浮动奖情景值不能为负数")
        if not self.label.strip():
            raise ValueError("浮动奖情景必须提供非空标签")

    @classmethod
    def cap_scenario(cls) -> "PrizeScenario":
        """构造现行规则封顶情景；调用方必须显式选择。"""

        return cls(
            pick9_jackpot=250_000.0,
            pick10_jackpot=5_000_000.0,
            label="explicit_rule_cap_scenario_not_actual_payout",
        )

    def floating_prize(self, play: int, issue: int | None) -> float:
        if (
            issue is not None
            and issue in self.per_issue
            and play in self.per_issue[issue]
        ):
            return float(self.per_issue[issue][play])
        fallback = self.pick9_jackpot if play == 9 else self.pick10_jackpot
        if fallback is None:
            raise MissingFloatingPrizeError(
                f"期号{issue or '未知'}选{play}浮动奖缺失；请提供逐期官方值或显式情景"
            )
        return float(fallback)


def rule_version_for_issue(issue: int) -> str:
    """按期号返回 ``old`` 或 ``current`` 奖金规则版本。"""

    if issue <= 0:
        raise ValueError("期号必须为正整数")
    return "current" if issue >= CURRENT_RULE_START_ISSUE else "old"


def prize_for_hits(
    play: int,
    hits: int,
    scenario: PrizeScenario,
    *,
    issue: int | None = None,
    rule_version: str | None = None,
) -> float:
    """返回一注在指定命中数下的奖金（元），每注只取一个奖级。"""

    if play not in range(1, 11):
        raise ValueError(f"玩法必须为选一至选十，收到：{play}")
    if hits < 0 or hits > play:
        raise ValueError(f"命中数必须在0至{play}之间，收到：{hits}")
    if rule_version is None:
        if issue is None:
            raise ValueError("必须提供issue或显式rule_version")
        rule_version = rule_version_for_issue(issue)
    if rule_version not in {"old", "current"}:
        raise ValueError(f"未知奖金规则版本：{rule_version}")

    if play == 10 and hits == 10:
        return scenario.floating_prize(play, issue)
    if rule_version == "current" and play == 9 and hits == 9:
        return scenario.floating_prize(play, issue)
    table = CURRENT_FIXED_PRIZES if rule_version == "current" else OLD_FIXED_PRIZES
    return float(table[play].get(hits, 0.0))


def _draw_probability_for_overlap(
    play: int, overlap: int
) -> Iterable[tuple[int, int, float]]:
    """给定两注重叠号码数，枚举两注命中数的联合概率。"""

    if overlap < 0 or overlap > play:
        raise ValueError("两注重叠数超出合法范围")
    common = overlap
    only_each = play - overlap
    outside = 80 - (2 * play - overlap)
    denominator = comb(80, 20)

    for common_drawn in range(common + 1):
        for first_only_drawn in range(only_each + 1):
            for second_only_drawn in range(only_each + 1):
                outside_drawn = 20 - common_drawn - first_only_drawn - second_only_drawn
                if not 0 <= outside_drawn <= outside:
                    continue
                ways = (
                    comb(common, common_drawn)
                    * comb(only_each, first_only_drawn)
                    * comb(only_each, second_only_drawn)
                    * comb(outside, outside_drawn)
                )
                if ways:
                    yield (
                        common_drawn + first_only_drawn,
                        common_drawn + second_only_drawn,
                        ways / denominator,
                    )


def _overlap_distribution(play: int, ticket_mode: str) -> Iterable[tuple[int, float]]:
    if ticket_mode == "disjoint":
        yield 0, 1.0
        return
    if ticket_mode != "independent":
        raise ValueError(f"未知出票方式：{ticket_mode}")

    denominator = comb(80, play)
    for overlap in range(play + 1):
        if play - overlap > 80 - play:
            continue
        probability = (
            comb(play, overlap) * comb(80 - play, play - overlap) / denominator
        )
        if probability:
            yield overlap, probability


def exact_two_ticket_metrics(
    play: int,
    ticket_mode: str,
    scenario: PrizeScenario,
) -> dict[str, float | int | str]:
    """精确计算固定每期两注（4元）的随机选号结构性指标。

    ``disjoint`` 强制两注无重叠；``independent`` 表示两注分别从80个号码中
    均匀抽取，因此对所有可能的重叠数进行精确加权。
    """

    expected_prize = 0.0
    any_prize = 0.0
    profit = 0.0
    at_least_100 = 0.0
    at_least_1000 = 0.0
    at_least_10000 = 0.0
    probability_total = 0.0

    for overlap, overlap_probability in _overlap_distribution(play, ticket_mode):
        for hits1, hits2, draw_probability in _draw_probability_for_overlap(
            play, overlap
        ):
            probability = overlap_probability * draw_probability
            payout = prize_for_hits(
                play, hits1, scenario, rule_version="current"
            ) + prize_for_hits(play, hits2, scenario, rule_version="current")
            probability_total += probability
            expected_prize += probability * payout
            any_prize += probability * (payout > 0)
            profit += probability * (payout > 4.0)
            at_least_100 += probability * (payout >= 100.0)
            at_least_1000 += probability * (payout >= 1000.0)
            at_least_10000 += probability * (payout >= 10000.0)

    if abs(probability_total - 1.0) > 1e-10:
        raise AssertionError(f"联合概率未归一：{probability_total}")

    return {
        "play": play,
        "ticket_mode": ticket_mode,
        "budget_per_issue": 4.0,
        "mean_hits_per_bet": play * 20 / 80,
        "any_prize_probability": any_prize,
        "profit_probability": profit,
        "expected_prize": expected_prize,
        "roi": expected_prize / 4.0 - 1.0,
        "prize_at_least_100_probability": at_least_100,
        "prize_at_least_1000_probability": at_least_1000,
        "prize_at_least_10000_probability": at_least_10000,
        "prize_scenario": scenario.label,
    }


__all__ = [
    "CURRENT_FIXED_PRIZES",
    "CURRENT_RULE_START_ISSUE",
    "MissingFloatingPrizeError",
    "OLD_FIXED_PRIZES",
    "PrizeScenario",
    "exact_two_ticket_metrics",
    "prize_for_hits",
    "rule_version_for_issue",
]
