"""运行快乐8策略 v2 第一阶段探索性嵌套时间回测。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

# v1 repository_advanced包含PCA；在导入NumPy/Scipy前固定线性代数线程，
# 避免不同进程的浮点归约顺序改变近似同分号码的最终排名。
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_v2.evaluation import COMPARATOR_STRATEGIES  # noqa: E402
from src.research_v2.evaluation import (  # noqa: E402
    DYNAMIC_STRATEGY,
    MINIMUM_INITIAL_HISTORY,
    MINIMUM_INNER_OBSERVATIONS,
    NestedEvaluationConfig,
    NestedEvaluationResult,
    evaluate_nested_walk_forward,
    validate_issue_draws,
)
from src.research_v2.metrics import CalibrationResult  # noqa: E402
from src.research_v2.metrics import (  # noqa: E402
    bernoulli_log_loss,
    calibration_summary,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROBABILITY_STRATEGIES = ("uniform_random", DYNAMIC_STRATEGY)
OUTPUT_FILENAMES = (
    "issue_probabilities.csv",
    "issue_metrics.csv",
    "topk_metrics.csv",
    "hyperparameter_history.csv",
    "calibration.csv",
)


@dataclass(frozen=True)
class ProbabilityMetrics:
    """逐期概率指标数组。"""

    dynamic_brier: FloatArray
    dynamic_log_loss: FloatArray
    uniform_brier: FloatArray
    uniform_log_loss: FloatArray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行动态Beta-Bernoulli嵌套时间滚动研究"
    )
    parser.add_argument("--data", default="data_cache/kl8/data.csv")
    parser.add_argument("--output-dir", default="results/research_v2")
    parser.add_argument("--report", default="reports/kl8_v2_research_report.md")
    return parser.parse_args()


def _resolve_within(root: Path, raw: str, label: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label}必须位于项目目录内")
    return resolved


def load_history_csv(path: Path) -> tuple[IntArray, IntArray]:
    """读取并按期号升序返回快乐8历史数据。"""

    number_columns = [f"红球_{index}" for index in range(1, 21)]
    rows: list[tuple[int, tuple[int, ...]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(number_columns).issubset(
            reader.fieldnames
        ):
            raise ValueError("快乐8数据缺少期数或红球列")
        if "期数" not in reader.fieldnames:
            raise ValueError("快乐8数据缺少期数列")
        for row in reader:
            issue = int(row["期数"])
            numbers = tuple(int(row[column]) for column in number_columns)
            rows.append((issue, numbers))
    rows.sort(key=lambda item: item[0])
    issues = cast(IntArray, np.asarray([row[0] for row in rows], dtype=np.int64))
    draws = cast(IntArray, np.asarray([row[1] for row in rows], dtype=np.int64))
    return validate_issue_draws(issues, draws)


def _float(value: float) -> str:
    return format(float(value), ".12g")


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_issue_probabilities(path: Path, result: NestedEvaluationResult) -> None:
    fields = (
        "issue",
        "history_through_issue",
        "history_issue_count",
        "number",
        "outcome",
        "rank",
        "selected_top10",
        "decay",
        "prior_strength",
        "alpha",
        "beta",
        "posterior_mean",
        "posterior_variance",
        "credible_interval_95_lower",
        "credible_interval_95_upper",
        "probability_sum",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        rankings = result.rankings[DYNAMIC_STRATEGY]
        for outer_offset, raw_target_index in enumerate(result.outer_indices):
            target_index = int(raw_target_index)
            parameter = result.selected_parameters(outer_offset)
            inverse_rank = np.empty(80, dtype=np.int64)
            inverse_rank[rankings[outer_offset] - 1] = np.arange(1, 81, dtype=np.int64)
            probability_sum = float(result.posterior_mean[outer_offset].sum())
            for number_index in range(80):
                rank = int(inverse_rank[number_index])
                yield {
                    "issue": int(result.issues[target_index]),
                    "history_through_issue": int(result.issues[target_index - 1]),
                    "history_issue_count": target_index,
                    "number": number_index + 1,
                    "outcome": int(result.indicators[target_index, number_index]),
                    "rank": rank,
                    "selected_top10": rank <= 10,
                    "decay": _float(parameter.decay),
                    "prior_strength": _float(parameter.prior_strength),
                    "alpha": _float(result.alpha[outer_offset, number_index]),
                    "beta": _float(result.beta[outer_offset, number_index]),
                    "posterior_mean": _float(
                        result.posterior_mean[outer_offset, number_index]
                    ),
                    "posterior_variance": _float(
                        result.posterior_variance[outer_offset, number_index]
                    ),
                    "credible_interval_95_lower": _float(
                        result.credible_interval_lower[outer_offset, number_index]
                    ),
                    "credible_interval_95_upper": _float(
                        result.credible_interval_upper[outer_offset, number_index]
                    ),
                    "probability_sum": _float(probability_sum),
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def calculate_probability_metrics(
    result: NestedEvaluationResult,
) -> ProbabilityMetrics:
    outcomes = result.outer_outcomes
    dynamic_brier = cast(
        FloatArray,
        np.mean(np.square(result.posterior_mean - outcomes), axis=1),
    )
    uniform_probabilities = np.full_like(result.posterior_mean, 0.25, dtype=np.float64)
    uniform_brier = cast(
        FloatArray,
        np.mean(np.square(uniform_probabilities - outcomes), axis=1),
    )
    dynamic_log_loss = np.asarray(
        [
            bernoulli_log_loss(probabilities, outcome)
            for probabilities, outcome in zip(
                result.posterior_mean, outcomes, strict=True
            )
        ],
        dtype=np.float64,
    )
    uniform_log_loss = np.asarray(
        [
            bernoulli_log_loss(probabilities, outcome)
            for probabilities, outcome in zip(
                uniform_probabilities, outcomes, strict=True
            )
        ],
        dtype=np.float64,
    )
    return ProbabilityMetrics(
        dynamic_brier=dynamic_brier,
        dynamic_log_loss=dynamic_log_loss,
        uniform_brier=uniform_brier,
        uniform_log_loss=uniform_log_loss,
    )


def write_issue_metrics(
    path: Path,
    result: NestedEvaluationResult,
    metrics: ProbabilityMetrics,
) -> None:
    fields = (
        "issue",
        "strategy",
        "brier_score",
        "bernoulli_log_loss",
        "brier_delta_vs_uniform",
        "log_loss_delta_vs_uniform",
        "probability_sum",
        "decay",
        "prior_strength",
        "independent_time_cluster",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        for outer_offset, issue in enumerate(result.outer_issues):
            parameter = result.selected_parameters(outer_offset)
            dynamic_brier = float(metrics.dynamic_brier[outer_offset])
            dynamic_log_loss = float(metrics.dynamic_log_loss[outer_offset])
            uniform_brier = float(metrics.uniform_brier[outer_offset])
            uniform_log_loss = float(metrics.uniform_log_loss[outer_offset])
            common = {
                "issue": int(issue),
                "independent_time_cluster": int(issue),
                "evidence_status": "exploratory_development_evidence",
            }
            yield {
                **common,
                "strategy": "uniform_random",
                "brier_score": _float(uniform_brier),
                "bernoulli_log_loss": _float(uniform_log_loss),
                "brier_delta_vs_uniform": _float(0.0),
                "log_loss_delta_vs_uniform": _float(0.0),
                "probability_sum": _float(20.0),
                "decay": "",
                "prior_strength": "",
            }
            yield {
                **common,
                "strategy": DYNAMIC_STRATEGY,
                "brier_score": _float(dynamic_brier),
                "bernoulli_log_loss": _float(dynamic_log_loss),
                "brier_delta_vs_uniform": _float(dynamic_brier - uniform_brier),
                "log_loss_delta_vs_uniform": _float(
                    dynamic_log_loss - uniform_log_loss
                ),
                "probability_sum": _float(
                    float(result.posterior_mean[outer_offset].sum())
                ),
                "decay": _float(parameter.decay),
                "prior_strength": _float(parameter.prior_strength),
            }

    _write_csv(path, fields, rows())


def calculate_topk_hits(
    result: NestedEvaluationResult,
) -> dict[str, FloatArray]:
    hits_by_strategy: dict[str, FloatArray] = {
        "uniform_random": result.uniform_random_mean_hits.copy()
    }
    outcomes = result.outer_outcomes
    for strategy, rankings in result.rankings.items():
        hits = np.empty((len(result.outer_indices), 10), dtype=np.float64)
        for outer_offset, ranking in enumerate(rankings):
            selected_outcomes = outcomes[outer_offset, ranking[:10] - 1]
            hits[outer_offset] = np.cumsum(selected_outcomes)
        hits_by_strategy[strategy] = hits
    if set(hits_by_strategy) != set(COMPARATOR_STRATEGIES):
        raise ValueError("Top-k比较策略集合不完整")
    return hits_by_strategy


def write_topk_metrics(
    path: Path,
    result: NestedEvaluationResult,
    hits_by_strategy: dict[str, FloatArray],
) -> None:
    fields = (
        "issue",
        "strategy",
        "k",
        "candidate_numbers",
        "actual_hits",
        "random_theoretical_expected_hits",
        "excess_hits",
        "delta_vs_uniform_seed_ensemble",
        "within_issue_aggregation",
        "independent_time_cluster",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        for outer_offset, issue in enumerate(result.outer_issues):
            for strategy in COMPARATOR_STRATEGIES:
                ranking = result.rankings.get(strategy)
                aggregation = (
                    "mean_over_20_seeds_within_issue"
                    if strategy == "uniform_random"
                    else "single_deterministic_ranking"
                )
                for k in range(1, 11):
                    hits = float(hits_by_strategy[strategy][outer_offset, k - 1])
                    uniform_hits = float(
                        hits_by_strategy["uniform_random"][outer_offset, k - 1]
                    )
                    expected = k / 4.0
                    candidates = ""
                    if ranking is not None:
                        candidates = " ".join(map(str, ranking[outer_offset, :k]))
                    yield {
                        "issue": int(issue),
                        "strategy": strategy,
                        "k": k,
                        "candidate_numbers": candidates,
                        "actual_hits": _float(hits),
                        "random_theoretical_expected_hits": _float(expected),
                        "excess_hits": _float(hits - expected),
                        "delta_vs_uniform_seed_ensemble": _float(hits - uniform_hits),
                        "within_issue_aggregation": aggregation,
                        "independent_time_cluster": int(issue),
                        "evidence_status": "exploratory_development_evidence",
                    }

    _write_csv(path, fields, rows())


def write_hyperparameter_history(
    path: Path, result: NestedEvaluationResult, config: NestedEvaluationConfig
) -> None:
    fields = (
        "outer_issue",
        "history_through_issue",
        "inner_first_target_issue",
        "inner_last_target_issue",
        "inner_validation_issue_count",
        "decay",
        "prior_strength",
        "inner_mean_brier",
        "selected",
        "selection_metric",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        inner_first_issue = int(result.issues[config.minimum_history])
        for outer_offset, raw_target_index in enumerate(result.outer_indices):
            target_index = int(raw_target_index)
            selected = int(result.selected_parameter_indices[outer_offset])
            for parameter_index, parameter in enumerate(result.parameters):
                yield {
                    "outer_issue": int(result.issues[target_index]),
                    "history_through_issue": int(result.issues[target_index - 1]),
                    "inner_first_target_issue": inner_first_issue,
                    "inner_last_target_issue": int(result.issues[target_index - 1]),
                    "inner_validation_issue_count": target_index
                    - config.minimum_history,
                    "decay": _float(parameter.decay),
                    "prior_strength": _float(parameter.prior_strength),
                    "inner_mean_brier": _float(
                        result.inner_mean_brier[outer_offset, parameter_index]
                    ),
                    "selected": parameter_index == selected,
                    "selection_metric": "mean_brier_score",
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def calculate_calibration(
    result: NestedEvaluationResult,
) -> dict[str, CalibrationResult]:
    outcomes = result.outer_outcomes
    return {
        "uniform_random": calibration_summary(
            np.full_like(outcomes, 0.25, dtype=np.float64), outcomes
        ),
        DYNAMIC_STRATEGY: calibration_summary(result.posterior_mean, outcomes),
    }


def write_calibration(path: Path, calibrations: dict[str, CalibrationResult]) -> None:
    fields = (
        "strategy",
        "bin_index",
        "lower_bound_inclusive",
        "upper_bound",
        "upper_bound_inclusive",
        "count",
        "mean_predicted_probability",
        "actual_rate",
        "absolute_gap",
        "weighted_gap",
        "expected_calibration_error",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        for strategy in PROBABILITY_STRATEGIES:
            result = calibrations[strategy]
            for row in result.bins:
                yield {
                    "strategy": strategy,
                    "bin_index": row.bin_index,
                    "lower_bound_inclusive": _float(row.lower_bound),
                    "upper_bound": _float(row.upper_bound),
                    "upper_bound_inclusive": row.bin_index == len(result.bins) - 1,
                    "count": row.count,
                    "mean_predicted_probability": (
                        ""
                        if row.mean_predicted_probability is None
                        else _float(row.mean_predicted_probability)
                    ),
                    "actual_rate": (
                        "" if row.actual_rate is None else _float(row.actual_rate)
                    ),
                    "absolute_gap": (
                        "" if row.absolute_gap is None else _float(row.absolute_gap)
                    ),
                    "weighted_gap": _float(row.weighted_gap),
                    "expected_calibration_error": _float(
                        result.expected_calibration_error
                    ),
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def _cluster_mean_interval(values: FloatArray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, math.nan, math.nan, math.nan
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return (
        mean,
        standard_error,
        mean - 1.96 * standard_error,
        mean + 1.96 * standard_error,
    )


def _markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    result: NestedEvaluationResult,
    config: NestedEvaluationConfig,
    metrics: ProbabilityMetrics,
    hits_by_strategy: dict[str, FloatArray],
    calibrations: dict[str, CalibrationResult],
    *,
    data_path: Path,
) -> str:
    dynamic_brier = _cluster_mean_interval(metrics.dynamic_brier)
    uniform_brier = _cluster_mean_interval(metrics.uniform_brier)
    brier_delta = _cluster_mean_interval(
        cast(FloatArray, metrics.dynamic_brier - metrics.uniform_brier)
    )
    dynamic_log = _cluster_mean_interval(metrics.dynamic_log_loss)
    uniform_log = _cluster_mean_interval(metrics.uniform_log_loss)
    log_delta = _cluster_mean_interval(
        cast(FloatArray, metrics.dynamic_log_loss - metrics.uniform_log_loss)
    )
    probability_error = float(np.max(np.abs(result.posterior_mean.sum(axis=1) - 20.0)))
    selected_counter = Counter(
        (
            result.parameters[int(index)].decay,
            result.parameters[int(index)].prior_strength,
        )
        for index in result.selected_parameter_indices
    )

    probability_table = _markdown_table(
        ("概率模型", "平均Brier", "相对随机", "平均log loss", "相对随机", "ECE"),
        (
            (
                "uniform_random",
                f"{uniform_brier[0]:.9f}",
                "0",
                f"{uniform_log[0]:.9f}",
                "0",
                f"{calibrations['uniform_random'].expected_calibration_error:.9f}",
            ),
            (
                DYNAMIC_STRATEGY,
                f"{dynamic_brier[0]:.9f}",
                f"{brier_delta[0]:+.9f}",
                f"{dynamic_log[0]:.9f}",
                f"{log_delta[0]:+.9f}",
                f"{calibrations[DYNAMIC_STRATEGY].expected_calibration_error:.9f}",
            ),
        ),
    )
    topk_rows: list[tuple[str, ...]] = []
    uniform_top10 = float(hits_by_strategy["uniform_random"][:, 9].mean())
    for strategy in COMPARATOR_STRATEGIES:
        mean_hits = hits_by_strategy[strategy].mean(axis=0)
        topk_rows.append(
            (
                strategy,
                f"{mean_hits[0]:.6f}",
                f"{mean_hits[4]:.6f}",
                f"{mean_hits[9]:.6f}",
                f"{mean_hits[9] - 2.5:+.6f}",
                f"{mean_hits[9] - uniform_top10:+.6f}",
            )
        )
    topk_table = _markdown_table(
        (
            "策略",
            "平均Top-1命中",
            "平均Top-5命中",
            "平均Top-10命中",
            "Top-10超额",
            "相对期内随机",
        ),
        topk_rows,
    )
    parameter_table = _markdown_table(
        ("decay", "prior_strength", "被选外层期数", "占比"),
        (
            (
                f"{decay:g}",
                f"{prior:g}",
                str(count),
                f"{count / len(result.outer_indices):.2%}",
            )
            for (decay, prior), count in sorted(selected_counter.items())
        ),
    )
    calibration_table = _markdown_table(
        ("概率箱", "记录数", "预测均值", "实际出现率", "绝对差"),
        (
            (
                f"[{row.lower_bound:.1f}, {row.upper_bound:.1f}{']' if row.bin_index == 9 else ')'}",
                str(row.count),
                (
                    "—"
                    if row.mean_predicted_probability is None
                    else f"{row.mean_predicted_probability:.6f}"
                ),
                "—" if row.actual_rate is None else f"{row.actual_rate:.6f}",
                "—" if row.absolute_gap is None else f"{row.absolute_gap:.6f}",
            )
            for row in calibrations[DYNAMIC_STRATEGY].bins
        ),
    )
    best_top10_strategy = max(
        COMPARATOR_STRATEGIES,
        key=lambda strategy: float(hits_by_strategy[strategy][:, 9].mean()),
    )
    best_top10_hits = float(hits_by_strategy[best_top10_strategy][:, 9].mean())
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    relative_data_path = data_path.relative_to(PROJECT_ROOT).as_posix()
    first_outer = int(result.outer_issues[0])
    last_outer = int(result.outer_issues[-1])
    first_inner = int(result.issues[config.minimum_history])

    return f"""# 快乐8策略 v2 第一阶段探索性研究报告

## 技术摘要

- 本报告是 **探索性 v2 研究（exploratory development evidence）**。旧 final holdout 已经被查看，本次历史时间滚动结果不是新的独立 holdout，也不是确认性证据。
- 动态 Beta-Bernoulli 在 `{len(result.outer_indices)}` 个外层开奖期上的平均 Brier score 为 `{dynamic_brier[0]:.9f}`，相对公平随机概率基线的差值为 `{brier_delta[0]:+.9f}`；按开奖期聚类的近似95%区间为 `[{brier_delta[2]:+.9f}, {brier_delta[3]:+.9f}]`。Brier 和 log loss 都是越低越好，因此这里的正差表示动态模型在这段已查看历史上略差；该差异仍只是探索性描述。
- 平均 Bernoulli log loss 为 `{dynamic_log[0]:.9f}`，相对随机基线差值 `{log_delta[0]:+.9f}`；动态模型 ECE 为 `{calibrations[DYNAMIC_STRATEGY].expected_calibration_error:.9f}`。
- Top-10 历史平均命中最高的比较项为 `{best_top10_strategy}`（`{best_top10_hits:.6f}`），随机理论期望为 `2.5`。该排序经过同一历史开发样本比较，不能据此宣称真实预测能力提高。
- 每个目标期的80个预测概率之和与20的最大绝对误差为 `{probability_error:.3e}`，满足快乐8每期固定开出20个号码的约束。

## 历史概率表现没有提供确认性优势证据

概率指标只比较具有合法80维概率解释的 `dynamic_bayesian` 与公平基线 `uniform_random=0.25`。其余现有策略输出的是排序分数，未通过事后变换伪装成校准概率；它们在后续 Top-k 表中比较。

{probability_table}

所有均值和差值都以开奖期为独立时间簇。没有把同一期80个号码、20个随机seed或多个k值当成独立样本，也没有报告机会性p值。

## 所有七个策略按同一期 Top-k 口径比较

确定性策略每期开奖期只有一个排名；`uniform_random` 的20个固定seed先在期内求平均，再把开奖期作为时间簇。`repository_advanced` 复用 v1 高级特征与权重，但在 v2 只读适配层中固定 PCA 主成分符号和同分顺序，以消除旧实现的跨进程排名漂移。完整 k=1 至10逐期结果见 `results/research_v2/topk_metrics.csv`。

{topk_table}

`excess_hits = actual_hits - k/4`；“相对期内随机”以每期开奖期先聚合20个固定seed后再求期均值。正差仅是已查看历史样本中的描述性结果，不是可持续优势证明。

## 数据范围、预测单位与指标定义

- 输入数据：`{relative_data_path}`，SHA-256 `{data_hash}`。
- 历史共 `{len(result.issues)}` 期，期号 `{int(result.issues[0])}` 至 `{int(result.issues[-1])}`。
- 内层首次验证目标期为 `{first_inner}`；外层探索期为 `{first_outer}` 至 `{last_outer}`，共 `{len(result.outer_indices)}` 期。
- 每个外层期产生80个目标期前概率、后验方差、95%可信区间和选一至选十排名。
- Brier score：每期开奖期 `mean((p_i-y_i)^2)`，80个号码只构成该期开奖期内的联合评分。
- Log loss：概率先截断到 `[1e-9, 1-1e-9]`。
- Calibration：固定10个等宽概率箱，ECE按每箱记录占比加权。

## 动态 Beta-Bernoulli 模型严格使用目标期之前的数据

公平基准为 `p0=20/80=0.25`。对目标期 `t`、号码 `i`，最近一期历史的 `age=0`：

```text
alpha_i,t = prior_strength × 0.25 + Σ decay^age × y_i
beta_i,t  = prior_strength × 0.75 + Σ decay^age × (1-y_i)
p_i,t     = alpha_i,t / (alpha_i,t + beta_i,t)
variance  = alpha_i,t × beta_i,t / ((alpha_i,t+beta_i,t)^2 × (alpha_i,t+beta_i,t+1))
```

95%可信区间使用对应 Beta 后验的2.5%与97.5%分位数。候选按 `posterior_mean` 降序排列，同分时号码升序；选一至选十是该排序的前 k 个号码。

预注册参数网格严格限定为：

- `decay ∈ {{0.97, 0.99, 0.995}}`
- `prior_strength ∈ {{5, 20, 80}}`

没有扩大网格，也没有根据外层或最终结果追加参数。

## 嵌套扩展式时间验证在每个外层期重新选参

1. 最初 `{MINIMUM_INITIAL_HISTORY}` 期只作为历史输入。
2. 接下来至少 `{MINIMUM_INNER_OBSERVATIONS}` 个目标期形成初始内层验证证据；每个内层预测也只使用其目标期之前的数据。
3. 对每个外层目标期，用从第 `{MINIMUM_INITIAL_HISTORY + 1}` 个目标开始到外层前一期的全部历史内层 Brier score，选择均值最低的参数。
4. 同分时按预注册网格顺序（decay顺序，再prior_strength顺序）选择，保证重复运行一致。
5. 选参后才预测当前外层期；当前开奖结果随后才进入下一期状态。期号从未随机打乱。

参数实际选择历史如下：

{parameter_table}

## 校准结果主要集中在接近0.25的概率箱

下面保留全部固定分箱，包括空箱；完整机器可读结果见 `results/research_v2/calibration.csv`。

{calibration_table}

## 限制、不确定性与稳健性检查

- **旧 holdout 已查看。** 所有历史结果只能称为 exploratory development evidence，不能称为新的独立 holdout或确认性证据。
- **没有因果或真实预测提升主张。** 同一历史样本承担了开发、嵌套选参和探索性比较；即使按时间防泄漏，也不能消除研究者自由度与历史选择偏差。
- **彩票公平性是主导假设。** 如果开奖公平且独立，任何历史模型都不应存在稳定优势；短期正负差异都可能是随机波动。
- **概率比较范围受限。** 历史、滚动、指数、混合和仓库高级基线是排序器，不具备未经额外校准的概率含义，因此只比较Top-k，没有事后把分数缩放为概率。
- **高级基线使用确定性适配。** v1 `repository_advanced` 的 PCA 主成分存在数学上的符号不唯一性，原实现跨进程可产生不同近似同分排名；v2 没有修改 v1，而是用等价协方差特征分解、固定主成分符号并将合成分数量化到12位小数。该比较应理解为旧策略的确定性研究适配版本。
- **随机seed不扩大样本量。** 20个seed仅用于同期开奖期内稳定随机参考，统计单位仍是开奖期。
- **区间是描述性的。** 报告中的差值区间按开奖期均值的常规标准误近似，不是预注册确认性检验，也未做多重比较推断。
- **表格优先于图形。** 第一阶段交付强调精确、可复核的CSV与紧凑表格，没有增加派生图片资产；后续若需要趋势诊断，应从逐期结果生成并保持开奖期粒度。

## 下一步应先冻结 v2，再依靠未来预先封存开奖确认

1. 审阅第一阶段实现、时间边界和参数选择记录，不根据本报告结果扩大网格。
2. 若决定继续，建立独立、外部可验证的 v2 冻结契约和候选 manifest；信任锚不能与可编辑哈希清单共存于同一未认证文件。
3. 冻结后只在未来开奖到达前预先封存概率与候选，逐期追加，不回写历史预测。
4. 未来样本累积到预先约定数量后，仍以开奖期为时间簇评价 Brier、log loss、ECE 与 Top-k 差值。

## 仍需回答的问题

- v2 的独立信任锚、签名或外部存证机制由谁维护？
- 未来确认期数、停止规则和主要指标应在何时冻结？
- 是否仅冻结动态概率模型，还是同时冻结候选出票映射与预算规则？

本报告不能依据历史回测声称提高了真实预测能力。真正确认必须在模型冻结后，依靠未来、预先封存且未被模型开发查看的开奖数据。
"""


def run() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_path = _resolve_within(root, str(args.data), "数据文件")
    output_dir = _resolve_within(root, str(args.output_dir), "结果目录")
    report_path = _resolve_within(root, str(args.report), "报告文件")
    if output_dir != (root / "results" / "research_v2").resolve():
        raise ValueError("第一阶段结果目录必须为results/research_v2")
    if report_path != (root / "reports" / "kl8_v2_research_report.md").resolve():
        raise ValueError("第一阶段报告必须为reports/kl8_v2_research_report.md")

    issues, draws = load_history_csv(data_path)
    config = NestedEvaluationConfig()
    result = evaluate_nested_walk_forward(issues, draws, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    probability_metrics = calculate_probability_metrics(result)
    topk_hits = calculate_topk_hits(result)
    calibrations = calculate_calibration(result)
    write_issue_probabilities(output_dir / OUTPUT_FILENAMES[0], result)
    write_issue_metrics(output_dir / OUTPUT_FILENAMES[1], result, probability_metrics)
    write_topk_metrics(output_dir / OUTPUT_FILENAMES[2], result, topk_hits)
    write_hyperparameter_history(output_dir / OUTPUT_FILENAMES[3], result, config)
    write_calibration(output_dir / OUTPUT_FILENAMES[4], calibrations)
    report = build_report(
        result,
        config,
        probability_metrics,
        topk_hits,
        calibrations,
        data_path=data_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    print(
        "完成："
        f"历史={len(issues)}期，外层探索={len(result.outer_indices)}期，"
        f"概率记录={len(result.outer_indices) * 80}，"
        f"报告={report_path.relative_to(root).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
