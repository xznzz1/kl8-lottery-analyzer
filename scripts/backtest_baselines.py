#!/usr/bin/env python3
"""生成快乐8固定4元预算的严格时间滚动科学评估。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scientific.evaluation import TemporalSplit  # noqa: E402
from src.scientific.evaluation import (
    evaluate_indices,
    load_history_csv,
    make_temporal_split,
    tune_on_validation,
)
from src.scientific.prizes import PrizeScenario  # noqa: E402
from src.scientific.prizes import exact_two_ticket_metrics
from src.scientific.statistics import hit_distribution  # noqa: E402
from src.scientific.statistics import (
    holm_adjust,
    paired_mean_bootstrap_interval,
    paired_randomisation_test,
    summarise_period_records,
    summarise_seed_ensemble,
)
from src.scientific.strategies import StrategyParameters  # noqa: E402

DEFAULT_RANDOM_SEEDS = tuple(range(202601, 202621))


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) < 5 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("至少提供5个互不重复的随机seed")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="快乐8无未来泄漏滚动回测；固定每期两注、每注2元。"
    )
    parser.add_argument("--data", default="data/kl8/data.csv", help="历史CSV路径")
    parser.add_argument(
        "--output-dir", default="results/scientific", help="CSV输出目录"
    )
    parser.add_argument(
        "--report", default="reports/kl8_scientific_report.md", help="Markdown报告路径"
    )
    parser.add_argument(
        "--floating-prize-mode",
        choices=("cap-scenario", "custom-scenario"),
        required=True,
        help="必须显式选择浮动奖情景；封顶情景不代表历史实付奖金",
    )
    parser.add_argument(
        "--pick9-jackpot", type=float, help="自定义新规则选九中九情景值"
    )
    parser.add_argument("--pick10-jackpot", type=float, help="自定义选十中十情景值")
    parser.add_argument("--scenario-label", default="custom_floating_prize_scenario")
    parser.add_argument(
        "--random-seeds",
        type=_parse_seeds,
        default=DEFAULT_RANDOM_SEEDS,
        help="逗号分隔的随机seed，至少5个",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    return parser


def _scenario_from_args(args: argparse.Namespace) -> PrizeScenario:
    if args.floating_prize_mode == "cap-scenario":
        return PrizeScenario.cap_scenario()
    if args.pick9_jackpot is None or args.pick10_jackpot is None:
        raise SystemExit("custom-scenario必须同时提供--pick9-jackpot和--pick10-jackpot")
    return PrizeScenario(
        pick9_jackpot=args.pick9_jackpot,
        pick10_jackpot=args.pick10_jackpot,
        label=args.scenario_label,
    )


def _add_group_summary(
    rows: list[dict[str, object]],
    group: pd.DataFrame,
    *,
    split_name: str,
    strategy: str,
    seed: object,
    play: int,
    ticket_mode: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    scenario: PrizeScenario,
) -> None:
    summary = summarise_period_records(
        group,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rows.append(
        {
            "split": split_name,
            "strategy": strategy,
            "seed": seed,
            "play": play,
            "ticket_mode": ticket_mode,
            "prize_scenario": scenario.label,
            **summary,
        }
    )


def summarise_strategy_records(
    records: pd.DataFrame,
    *,
    split_name: str,
    bootstrap_samples: int,
    scenario: PrizeScenario,
) -> pd.DataFrame:
    """保留每个随机seed，并另建逐期seed均值基线。"""

    rows: list[dict[str, object]] = []
    grouped = records.groupby(["strategy", "seed", "play", "ticket_mode"], dropna=False)
    for group_index, ((strategy, seed, play, ticket_mode), group) in enumerate(grouped):
        is_seed_replication = strategy == "uniform_random" and pd.notna(seed)
        _add_group_summary(
            rows,
            group.sort_values("issue"),
            split_name=split_name,
            strategy=str(strategy),
            seed=int(seed) if pd.notna(seed) else "",
            play=int(play),
            ticket_mode=str(ticket_mode),
            bootstrap_samples=0 if is_seed_replication else bootstrap_samples,
            bootstrap_seed=20260716 + group_index,
            scenario=scenario,
        )

    random_records = records[records["strategy"] == "uniform_random"]
    for group_index, ((play, ticket_mode), group) in enumerate(
        random_records.groupby(["play", "ticket_mode"])
    ):
        summary = summarise_seed_ensemble(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20261716 + group_index,
        )
        rows.append(
            {
                "split": split_name,
                "strategy": "uniform_random",
                "seed": "ensemble_mean",
                "play": int(play),
                "ticket_mode": str(ticket_mode),
                "prize_scenario": scenario.label,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _attach_randomisation_tests(
    final_summary: pd.DataFrame,
    final_records: pd.DataFrame,
    *,
    samples: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    tests: list[dict[str, object]] = []
    random_records = final_records[final_records["strategy"] == "uniform_random"].copy()
    random_records["mean_hits_per_bet_issue"] = (
        random_records["hits1"] + random_records["hits2"]
    ) / 2.0
    baseline = random_records.groupby(["issue", "play", "ticket_mode"], as_index=False)[
        "mean_hits_per_bet_issue"
    ].mean()

    deterministic = final_records[final_records["strategy"] != "uniform_random"].copy()
    deterministic["mean_hits_per_bet_issue"] = (
        deterministic["hits1"] + deterministic["hits2"]
    ) / 2.0
    for test_index, ((strategy, play, ticket_mode), group) in enumerate(
        deterministic.groupby(["strategy", "play", "ticket_mode"])
    ):
        paired = group[["issue", "mean_hits_per_bet_issue"]].merge(
            baseline[
                (baseline["play"] == play) & (baseline["ticket_mode"] == ticket_mode)
            ][["issue", "mean_hits_per_bet_issue"]],
            on="issue",
            suffixes=("_candidate", "_random"),
            validate="one_to_one",
        )
        candidate = paired["mean_hits_per_bet_issue_candidate"].to_numpy()
        random_baseline = paired["mean_hits_per_bet_issue_random"].to_numpy()
        effect, p_value = paired_randomisation_test(
            candidate,
            random_baseline,
            samples=samples,
            seed=20260716 + test_index,
        )
        effect_low, effect_high = paired_mean_bootstrap_interval(
            candidate,
            random_baseline,
            samples=bootstrap_samples,
            seed=20270716 + test_index,
        )
        tests.append(
            {
                "strategy": strategy,
                "play": int(play),
                "ticket_mode": ticket_mode,
                "mean_hits_effect_vs_random": effect,
                "mean_hits_effect_ci95_low": effect_low,
                "mean_hits_effect_ci95_high": effect_high,
                "randomisation_p_value": p_value,
            }
        )
    tests_frame = pd.DataFrame(tests)
    tests_frame["holm_adjusted_p_value"] = holm_adjust(
        tests_frame["randomisation_p_value"]
    )
    tests_frame["significant_at_0_05"] = (
        tests_frame["holm_adjusted_p_value"] < 0.05
    ) & (tests_frame["mean_hits_effect_vs_random"] > 0)
    return final_summary.merge(
        tests_frame,
        on=["strategy", "play", "ticket_mode"],
        how="left",
    )


def _all_plays_summary(scenario: PrizeScenario) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            exact_two_ticket_metrics(play, ticket_mode, scenario)
            for play in range(1, 11)
            for ticket_mode in ("disjoint", "independent")
        ]
    )
    frame["any_prize_rank"] = (
        frame["any_prize_probability"].rank(ascending=False, method="min").astype(int)
    )
    frame["prize_at_least_1000_rank"] = (
        frame["prize_at_least_1000_probability"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    def utility(column: str) -> pd.Series:
        values = frame[column]
        spread = values.max() - values.min()
        return (
            (values - values.min()) / spread
            if spread
            else pd.Series(0.0, index=frame.index)
        )

    frame["win_probability_utility"] = utility("any_prize_probability")
    frame["expected_prize_utility"] = utility("expected_prize")
    for win_weight in (0.25, 0.50, 0.75):
        ev_weight = 1.0 - win_weight
        suffix = f"w{int(win_weight * 100)}_win_w{int(ev_weight * 100)}_ev"
        frame[f"balanced_score_{suffix}"] = (
            win_weight * frame["win_probability_utility"]
            + ev_weight * frame["expected_prize_utility"]
        )
        frame[f"balanced_rank_{suffix}"] = (
            frame[f"balanced_score_{suffix}"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
    return frame.sort_values(["play", "ticket_mode"]).reset_index(drop=True)


def _hit_distributions(final_records: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    deterministic = final_records[final_records["strategy"] != "uniform_random"]
    for (strategy, play, ticket_mode), group in deterministic.groupby(
        ["strategy", "play", "ticket_mode"]
    ):
        distribution = hit_distribution(group, max_hits=int(play))
        distribution.insert(0, "ticket_mode", ticket_mode)
        distribution.insert(0, "play", int(play))
        distribution.insert(0, "seed", "")
        distribution.insert(0, "strategy", strategy)
        rows.append(distribution)

    random_records = final_records[final_records["strategy"] == "uniform_random"]
    for (play, ticket_mode), group in random_records.groupby(["play", "ticket_mode"]):
        distribution = hit_distribution(group, max_hits=int(play))
        distribution.insert(0, "ticket_mode", ticket_mode)
        distribution.insert(0, "play", int(play))
        distribution.insert(0, "seed", "pooled_seeds")
        distribution.insert(0, "strategy", "uniform_random")
        rows.append(distribution)
    return pd.concat(rows, ignore_index=True)


def _markdown_table(
    frame: pd.DataFrame, columns: Sequence[str], digits: int = 4
) -> str:
    labels = [str(column) for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_metadata(data_path: Path) -> dict[str, object]:
    metadata_path = data_path.parent / "download_meta.json"
    if not metadata_path.exists():
        return {"metadata_status": "download_meta.json缺失"}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_report(
    report_path: Path,
    *,
    metadata: dict[str, object],
    issues: np.ndarray,
    split: TemporalSplit,
    parameters: StrategyParameters,
    sensitivity: pd.DataFrame,
    plays: pd.DataFrame,
    final_results: pd.DataFrame,
    random_seed_count: int,
    scenario: PrizeScenario,
) -> None:
    significant = final_results[
        final_results.get(
            "significant_at_0_05", pd.Series(False, index=final_results.index)
        ).fillna(False)
    ]
    superiority = (
        "没有策略在最终未接触holdout上、经过Holm多重检验校正后，单注平均命中数显著优于多seed均匀随机选号。"
        if significant.empty
        else f"有{len(significant)}个策略×玩法×出票方式组合在校正后显著优于随机；需结合效应量和复现验证解读。"
    )
    final_deterministic = final_results[
        (final_results["strategy"] != "uniform_random")
        & (final_results["seed"].astype(str).isin(["", "nan"]))
    ].copy()
    if final_deterministic.empty:
        final_deterministic = final_results[
            final_results["strategy"] != "uniform_random"
        ].copy()
    comparison = final_deterministic.sort_values(
        ["holm_adjusted_p_value", "mean_hits_effect_vs_random"],
        ascending=[True, False],
    ).head(15)

    play_view = plays[
        [
            "play",
            "ticket_mode",
            "any_prize_probability",
            "profit_probability",
            "expected_prize",
            "roi",
            "prize_at_least_1000_probability",
            "any_prize_rank",
            "balanced_rank_w50_win_w50_ev",
            "prize_at_least_1000_rank",
        ]
    ]
    comparison_columns = [
        "strategy",
        "play",
        "ticket_mode",
        "mean_hits_effect_vs_random",
        "mean_hits_effect_ci95_low",
        "mean_hits_effect_ci95_high",
        "randomisation_p_value",
        "holm_adjusted_p_value",
        "roi",
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)

    report = f"""# 快乐8固定4元科学评估报告

## 执行摘要

- **最终结论：{superiority}** 这是一项样本外检验，不构成可预测随机开奖的主张。
- 每期预算固定为4元，即两注、每注2元；选一至选十和两种出票方式使用同一时间区间配对比较。
- “更容易中奖”按每期任意奖金概率衡量；“更有高奖金可能性”按每期奖金至少1000元概率衡量，两者分别排名，不能互换。
- 奖金和ROI采用情景 `{scenario.label}`。浮动奖、限赔、派奖和税后金额并非仅凭开奖号码可还原，因此ROI是规则情景结果，不是历史实付收益。

## 数据来源与完整性

开奖数据来自500.com快乐8走势页面，实时HTML结构和拒绝行规则见 `docs/data_source_validation.md`。
本次输入覆盖 `{int(issues[0])}` 至 `{int(issues[-1])}`，共 `{len(issues)}` 期；下载元信息如下（路径仅保留可移植文件名）：

```json
{metadata_json}
```

数据粒度为每期一行、20个互不重复的1至80号码。完整下载CSV与HTML不提交；复现时重新下载并以元信息中的SHA-256固定输入快照。

## 方法与指标定义

比较策略包括：多seed均匀随机、全历史频率、滚动窗口频率、指数衰减频率、验证区间选择权重的hybrid，以及原仓库可改造成无泄漏的高级特征组合。原仓库的深度特征、全样本图嵌入和规则缓存因已识别的时间泄漏风险未纳入。

每个目标期先生成两注合法票面，再记录单注平均命中数、命中分布、任意中奖、4元后盈利、期望奖金、ROI、100/1000/10000元门槛、最大回撤和最长连续亏损。概率事件按两注合计奖金定义。

## 防未来数据泄漏与反过拟合

- 预测索引 `t` 的唯一输入为 `draws[:t]`；CSV先按期号严格升序，绝不以期号数值差代替行索引。
- 最后20%（索引 `{split.holdout_start}` 至 `{split.total_issues - 1}`）是最终holdout；在冻结参数前不参与选择。
- 在holdout之前的滚动验证区间（索引 `{split.validation_start}` 至 `{split.holdout_start - 1}`）上，仅比较预注册窗口、衰减和hybrid权重网格。
- 选中参数：窗口 `{parameters.rolling_window}`，衰减 `{parameters.decay}`，hybrid权重（全历史/滚动/衰减）`{parameters.hybrid_weights}`。
- 高级方法每个时点重新只用既往历史计算，并禁用无法证明训练截止点的图嵌入缓存。

参数敏感性：

{_markdown_table(sensitivity, ["parameter", "value", "validation_mean_hits_per_bet", "selected"], 6)}

## 官方规则与奖金假设

完整来源、访问日期、旧版/新版边界和奖级表见 `docs/official_rules.md`。第2025350期起切换新规则；选十中十始终为浮动奖，新规则下选九中九也为浮动奖。名义固定奖仍可能受限额赔付影响。本报告用显式情景而非把浮动奖封顶值当作确定值。

## 选一至选十总表（当前规则结构性精确概率）

下表对均匀随机票面做组合数学精确计算，不依赖有限seed抽样。两个效用先在20个玩法×出票选项内做min-max归一化。
平衡排名主版本公开使用50%任意中奖效用 + 50%期望奖金效用，CSV同时给出25/75和75/25权重敏感性。

{_markdown_table(play_view, list(play_view.columns), 6)}

## 策略比较（最终未接触holdout）

主检验端点是每期两注的单注平均命中数。候选策略与 `{random_seed_count}` 个seed的逐期随机均值做配对符号翻转检验，并对5策略×10玩法×2出票方式的检验族使用Holm校正。符号翻转检验依赖零假设下配对差值符号可交换（通常以差值近似对称解释），因此这里将其作为有明确假设的近似随机化检验，而不是无条件精确检验。最靠近显著性的15项如下：

{_markdown_table(comparison, comparison_columns, 6)}

**结论仍为：{superiority}** ROI、中奖频率和高奖金事件继续作为描述性及置信区间指标报告，不用稀有大奖的偶然出现替代预注册主检验。

## 统计不确定性

平均命中、期望奖金、ROI及配对命中效应提供按期重抽样的95%百分位bootstrap区间；任意中奖、盈利及100/1000/10000元事件概率使用Wilson 95%区间，零次观测仍保留正上界。随机基线同时保留每个seed结果和ensemble汇总：事件先在每个seed×期上判定后平均，ensemble的Wilson区间以期数而不是seed×期数作为有效聚类数；回撤与最长亏损先沿各seed路径计算后平均，不构造不存在的“平均奖金票”。只展示一次随机选号会严重低估基线方差。配对设计消除了同期开奖难度的期间错配，但有限holdout对千元及万元事件仍可能非常稀疏。

## 限制条件与稳健性边界

- 开奖机制按随机过程解释；观察到的冷热、共现或PCA结构不等于可预测信号。
- 当前规则结构性表使用显式浮动奖情景；历史逐期实付浮动奖、限赔、派奖、地区活动和税后奖金未完整输入。
- 两版奖表按期号切换，但没有票面销售地，无法重建地方活动。
- 对1000元、10000元门槛的样本外估计受稀有事件限制，应优先参考精确组合概率和区间，不应只看回测点估计。
- 本次网格是预注册的小网格；更大的搜索空间必须另开新的未接触holdout，不能重复使用本报告holdout。

## 针对每期4元玩家的大白话结论

**想“更容易中奖”**，看 `any_prize_rank`，它包含返还2元但整期仍亏钱的情况；所以中奖不等于盈利。

**想“更有高奖金可能性”**，看 `prize_at_least_1000_rank`。这类玩法通常中奖更少、连续亏损更长，不能拿“偶尔可能中得多”包装成“更容易赢”。

平衡选择的权重不是隐藏分数：主表是50%中奖概率、50%期望奖金，并已输出两侧权重敏感性。无论采用哪种偏好，当前证据都不支持把历史频率或高级特征宣称为能稳定战胜随机选号。

## 推荐下一步与复现命令

```powershell
.venv\\Scripts\\python.exe scripts/get_data.py --name kl8
.venv\\Scripts\\python.exe scripts/backtest_baselines.py --data data/kl8/data.csv --floating-prize-mode cap-scenario
```

若获得逐期官方浮动奖、限赔及派奖数据，应改用明确标注的自定义情景或扩展逐期输入，再生成新的报告；不得覆盖本次holdout后重新调参。

生成时间（UTC）：`{generated_at}`。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = _scenario_from_args(args)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    issues, draws = load_history_csv(data_path)
    split = make_temporal_split(len(draws))
    parameters, sensitivity = tune_on_validation(draws, split)

    validation_records = evaluate_indices(
        issues,
        draws,
        split.validation_indices,
        parameters,
        scenario,
        random_seeds=args.random_seeds,
    )
    final_records = evaluate_indices(
        issues,
        draws,
        split.holdout_indices,
        parameters,
        scenario,
        random_seeds=args.random_seeds,
    )

    validation_summary = summarise_strategy_records(
        validation_records,
        split_name="validation",
        bootstrap_samples=args.bootstrap_samples,
        scenario=scenario,
    )
    final_summary = summarise_strategy_records(
        final_records,
        split_name="final_holdout",
        bootstrap_samples=args.bootstrap_samples,
        scenario=scenario,
    )
    final_summary = _attach_randomisation_tests(
        final_summary,
        final_records,
        samples=args.permutation_samples,
        bootstrap_samples=args.bootstrap_samples,
    )
    strategy_comparison = pd.concat(
        [validation_summary, final_summary], ignore_index=True, sort=False
    )
    final_holdout = final_summary[
        (final_summary["strategy"] != "uniform_random")
        | (final_summary["seed"] == "ensemble_mean")
    ].copy()
    plays = _all_plays_summary(scenario)
    distributions = _hit_distributions(final_records)
    risk = final_summary[
        [
            "strategy",
            "seed",
            "play",
            "ticket_mode",
            "max_drawdown",
            "longest_losing_streak",
            "ending_profit",
        ]
    ].copy()

    plays.to_csv(output_dir / "all_plays_summary.csv", index=False, encoding="utf-8")
    strategy_comparison.to_csv(
        output_dir / "strategy_comparison.csv", index=False, encoding="utf-8"
    )
    distributions.to_csv(
        output_dir / "hit_distributions.csv", index=False, encoding="utf-8"
    )
    risk.to_csv(output_dir / "risk_metrics.csv", index=False, encoding="utf-8")
    final_holdout.to_csv(
        output_dir / "final_holdout_results.csv", index=False, encoding="utf-8"
    )
    sensitivity.to_csv(
        output_dir / "parameter_sensitivity.csv", index=False, encoding="utf-8"
    )
    (output_dir / "selected_parameters.json").write_text(
        json.dumps(
            {
                "rolling_window": parameters.rolling_window,
                "decay": parameters.decay,
                "hybrid_weights": parameters.hybrid_weights,
                "validation_start_issue": int(issues[split.validation_start]),
                "validation_end_issue": int(issues[split.holdout_start - 1]),
                "final_holdout_start_issue": int(issues[split.holdout_start]),
                "final_holdout_end_issue": int(issues[-1]),
                "random_seeds": args.random_seeds,
                "prize_scenario": scenario.label,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_report(
        report_path,
        metadata=_load_metadata(data_path),
        issues=issues,
        split=split,
        parameters=parameters,
        sensitivity=sensitivity,
        plays=plays,
        final_results=final_holdout,
        random_seed_count=len(args.random_seeds),
        scenario=scenario,
    )
    print(
        f"完成：{len(draws)}期，验证{len(split.validation_indices)}期，"
        f"最终holdout {len(split.holdout_indices)}期；结果写入{output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
