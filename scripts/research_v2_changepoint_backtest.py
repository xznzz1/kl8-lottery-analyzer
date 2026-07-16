"""运行快乐8策略 v2 第二阶段在线变点探索性回测。"""

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

from src.research_v2.changepoint import (  # noqa: E402
    CHANGEPOINT_PRIOR_STRENGTH,
    HIGH_CHANGE_DECAY,
    MINIMUM_EFFECTIVE_HISTORY,
    NORMAL_DECAY,
    RECENT_WINDOW_GRID,
    REFERENCE_WINDOW,
)
from src.research_v2.changepoint_evaluation import (  # noqa: E402
    CHANGEPOINT_STRATEGY,
    PHASE2_COMPARATOR_STRATEGIES,
    ChangepointEvaluationConfig,
    ChangepointEvaluationResult,
    evaluate_changepoint_walk_forward,
)
from src.research_v2.evaluation import DYNAMIC_STRATEGY  # noqa: E402
from src.research_v2.evaluation import validate_issue_draws  # noqa: E402
from src.research_v2.metrics import (  # noqa: E402
    CalibrationResult,
    bernoulli_log_loss,
    calibration_summary,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROBABILITY_STRATEGIES = (
    "uniform_random",
    DYNAMIC_STRATEGY,
    CHANGEPOINT_STRATEGY,
)
OUTPUT_FILENAMES = (
    "issue_probabilities.csv",
    "issue_metrics.csv",
    "topk_metrics.csv",
    "changepoint_history.csv",
    "state_metrics.csv",
    "calibration.csv",
    "trigger_intervals.csv",
)


@dataclass(frozen=True)
class ProbabilityMetrics:
    """逐期开奖期的三组概率评分。"""

    changepoint_brier: FloatArray
    changepoint_log_loss: FloatArray
    dynamic_brier: FloatArray
    dynamic_log_loss: FloatArray
    uniform_brier: FloatArray
    uniform_log_loss: FloatArray


@dataclass(frozen=True)
class TriggerInterval:
    """连续 high_change 外层期段。"""

    start_issue: int
    end_issue: int
    issue_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行在线变点Beta-Bernoulli嵌套时间滚动研究"
    )
    parser.add_argument("--data", default="data_cache/kl8/data.csv")
    parser.add_argument("--output-dir", default="results/research_v2_changepoint")
    parser.add_argument("--report", default="reports/kl8_v2_changepoint_report.md")
    return parser.parse_args()


def _resolve_within(root: Path, raw: str, label: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label}必须位于项目目录内")
    return resolved


def load_history_csv(path: Path) -> tuple[IntArray, IntArray]:
    """读取正式或临时快乐8 CSV，并按期号升序校验。"""

    number_columns = [f"红球_{index}" for index in range(1, 21)]
    rows: list[tuple[int, tuple[int, ...]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "期数" not in reader.fieldnames:
            raise ValueError("快乐8数据缺少期数列")
        if not set(number_columns).issubset(reader.fieldnames):
            raise ValueError("快乐8数据缺少红球列")
        for row in reader:
            rows.append(
                (
                    int(row["期数"]),
                    tuple(int(row[column]) for column in number_columns),
                )
            )
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


def calculate_probability_metrics(
    result: ChangepointEvaluationResult,
) -> ProbabilityMetrics:
    outcomes = result.outer_outcomes
    changepoint_brier = cast(
        FloatArray,
        np.mean(np.square(result.posterior_mean - outcomes), axis=1),
    )
    dynamic_probabilities = result.phase1_result.posterior_mean
    dynamic_brier = cast(
        FloatArray,
        np.mean(np.square(dynamic_probabilities - outcomes), axis=1),
    )
    uniform_probabilities = np.full_like(outcomes, 0.25, dtype=np.float64)
    uniform_brier = cast(
        FloatArray,
        np.mean(np.square(uniform_probabilities - outcomes), axis=1),
    )

    def log_losses(probabilities: FloatArray) -> FloatArray:
        return np.asarray(
            [
                bernoulli_log_loss(predicted, actual)
                for predicted, actual in zip(probabilities, outcomes, strict=True)
            ],
            dtype=np.float64,
        )

    return ProbabilityMetrics(
        changepoint_brier=changepoint_brier,
        changepoint_log_loss=log_losses(result.posterior_mean),
        dynamic_brier=dynamic_brier,
        dynamic_log_loss=log_losses(dynamic_probabilities),
        uniform_brier=uniform_brier,
        uniform_log_loss=log_losses(uniform_probabilities),
    )


def calculate_topk_hits(
    result: ChangepointEvaluationResult,
) -> dict[str, FloatArray]:
    hits_by_strategy: dict[str, FloatArray] = {
        "uniform_random": result.uniform_random_mean_hits.copy()
    }
    outcomes = result.outer_outcomes
    for strategy, rankings in result.rankings.items():
        hits = np.empty((len(result.outer_indices), 10), dtype=np.float64)
        for outer_offset, ranking in enumerate(rankings):
            hits[outer_offset] = np.cumsum(outcomes[outer_offset, ranking[:10] - 1])
        hits_by_strategy[strategy] = hits
    if set(hits_by_strategy) != set(PHASE2_COMPARATOR_STRATEGIES):
        raise ValueError("第二阶段Top-k比较策略集合不完整")
    return hits_by_strategy


def calculate_calibration(
    result: ChangepointEvaluationResult,
) -> dict[str, CalibrationResult]:
    outcomes = result.outer_outcomes
    return {
        "uniform_random": calibration_summary(
            np.full_like(outcomes, 0.25, dtype=np.float64), outcomes
        ),
        DYNAMIC_STRATEGY: calibration_summary(
            result.phase1_result.posterior_mean, outcomes
        ),
        CHANGEPOINT_STRATEGY: calibration_summary(result.posterior_mean, outcomes),
    }


def find_trigger_intervals(
    result: ChangepointEvaluationResult,
) -> tuple[TriggerInterval, ...]:
    intervals: list[TriggerInterval] = []
    start: int | None = None
    for offset, triggered in enumerate(result.high_change):
        if bool(triggered) and start is None:
            start = offset
        is_last = offset == len(result.high_change) - 1
        if start is not None and (not bool(triggered) or is_last):
            end = offset if bool(triggered) and is_last else offset - 1
            intervals.append(
                TriggerInterval(
                    start_issue=int(result.outer_issues[start]),
                    end_issue=int(result.outer_issues[end]),
                    issue_count=end - start + 1,
                )
            )
            start = None
    return tuple(intervals)


def write_issue_probabilities(path: Path, result: ChangepointEvaluationResult) -> None:
    fields = (
        "issue",
        "history_through_issue",
        "history_issue_count",
        "number",
        "outcome",
        "posterior_mean",
        "posterior_variance",
        "credible_interval_lower",
        "credible_interval_upper",
        "rank",
        "recent_window",
        "change_score",
        "change_threshold",
        "high_change",
        "active_decay",
        "effective_history_length",
        "probability_sum",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        for outer_offset, raw_target_index in enumerate(result.outer_indices):
            target_index = int(raw_target_index)
            parameter_index = int(result.selected_parameter_indices[outer_offset])
            parameter = result.parameters[parameter_index]
            ranking = result.rankings[CHANGEPOINT_STRATEGY][outer_offset]
            rank_positions = np.empty(80, dtype=np.int64)
            rank_positions[ranking - 1] = np.arange(1, 81, dtype=np.int64)
            for number_index in range(80):
                yield {
                    "issue": int(result.issues[target_index]),
                    "history_through_issue": int(result.issues[target_index - 1]),
                    "history_issue_count": target_index,
                    "number": number_index + 1,
                    "outcome": int(result.outer_outcomes[outer_offset, number_index]),
                    "posterior_mean": _float(
                        result.posterior_mean[outer_offset, number_index]
                    ),
                    "posterior_variance": _float(
                        result.posterior_variance[outer_offset, number_index]
                    ),
                    "credible_interval_lower": _float(
                        result.credible_interval_lower[outer_offset, number_index]
                    ),
                    "credible_interval_upper": _float(
                        result.credible_interval_upper[outer_offset, number_index]
                    ),
                    "rank": int(rank_positions[number_index]),
                    "recent_window": parameter.recent_window,
                    "change_score": _float(
                        result.candidate_change_scores[outer_offset, parameter_index]
                    ),
                    "change_threshold": _float(
                        result.candidate_thresholds[outer_offset, parameter_index]
                    ),
                    "high_change": bool(result.high_change[outer_offset]),
                    "active_decay": _float(result.active_decay[outer_offset]),
                    "effective_history_length": int(
                        result.effective_history_length[outer_offset]
                    ),
                    "probability_sum": _float(
                        float(result.posterior_mean[outer_offset].sum())
                    ),
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def write_issue_metrics(
    path: Path,
    result: ChangepointEvaluationResult,
    metrics: ProbabilityMetrics,
) -> None:
    fields = (
        "issue",
        "strategy",
        "brier_score",
        "bernoulli_log_loss",
        "brier_delta_vs_uniform",
        "brier_delta_vs_dynamic_bayesian",
        "log_loss_delta_vs_uniform",
        "log_loss_delta_vs_dynamic_bayesian",
        "high_change",
        "independent_time_cluster",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        arrays = {
            "uniform_random": (metrics.uniform_brier, metrics.uniform_log_loss),
            DYNAMIC_STRATEGY: (metrics.dynamic_brier, metrics.dynamic_log_loss),
            CHANGEPOINT_STRATEGY: (
                metrics.changepoint_brier,
                metrics.changepoint_log_loss,
            ),
        }
        for offset, issue in enumerate(result.outer_issues):
            for strategy in PROBABILITY_STRATEGIES:
                brier_values, log_values = arrays[strategy]
                brier = float(brier_values[offset])
                log_loss = float(log_values[offset])
                yield {
                    "issue": int(issue),
                    "strategy": strategy,
                    "brier_score": _float(brier),
                    "bernoulli_log_loss": _float(log_loss),
                    "brier_delta_vs_uniform": _float(
                        brier - float(metrics.uniform_brier[offset])
                    ),
                    "brier_delta_vs_dynamic_bayesian": _float(
                        brier - float(metrics.dynamic_brier[offset])
                    ),
                    "log_loss_delta_vs_uniform": _float(
                        log_loss - float(metrics.uniform_log_loss[offset])
                    ),
                    "log_loss_delta_vs_dynamic_bayesian": _float(
                        log_loss - float(metrics.dynamic_log_loss[offset])
                    ),
                    "high_change": bool(result.high_change[offset]),
                    "independent_time_cluster": int(issue),
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def write_topk_metrics(
    path: Path,
    result: ChangepointEvaluationResult,
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
        "high_change",
        "within_issue_aggregation",
        "independent_time_cluster",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        for offset, issue in enumerate(result.outer_issues):
            for strategy in PHASE2_COMPARATOR_STRATEGIES:
                ranking = result.rankings.get(strategy)
                aggregation = (
                    "mean_over_20_seeds_within_issue"
                    if strategy == "uniform_random"
                    else "single_deterministic_ranking"
                )
                for k in range(1, 11):
                    hits = float(hits_by_strategy[strategy][offset, k - 1])
                    uniform = float(hits_by_strategy["uniform_random"][offset, k - 1])
                    candidates = ""
                    if ranking is not None:
                        candidates = " ".join(map(str, ranking[offset, :k]))
                    yield {
                        "issue": int(issue),
                        "strategy": strategy,
                        "k": k,
                        "candidate_numbers": candidates,
                        "actual_hits": _float(hits),
                        "random_theoretical_expected_hits": _float(k / 4.0),
                        "excess_hits": _float(hits - k / 4.0),
                        "delta_vs_uniform_seed_ensemble": _float(hits - uniform),
                        "high_change": bool(result.high_change[offset]),
                        "within_issue_aggregation": aggregation,
                        "independent_time_cluster": int(issue),
                        "evidence_status": "exploratory_development_evidence",
                    }

    _write_csv(path, fields, rows())


def write_changepoint_history(path: Path, result: ChangepointEvaluationResult) -> None:
    fields = (
        "outer_issue",
        "history_through_issue",
        "recent_window",
        "reference_window",
        "inner_mean_brier",
        "selected",
        "change_score",
        "change_threshold_quantile",
        "change_threshold",
        "candidate_high_change",
        "threshold_first_score_target_issue",
        "threshold_last_score_target_issue",
        "threshold_data_through_issue",
        "threshold_score_count",
        "selection_metric",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        for offset, raw_target_index in enumerate(result.outer_indices):
            target_index = int(raw_target_index)
            selected = int(result.selected_parameter_indices[offset])
            for parameter_index, parameter in enumerate(result.parameters):
                first_index = int(
                    result.threshold_training_first_indices[parameter_index]
                )
                last_index = int(result.threshold_training_last_indices[offset])
                yield {
                    "outer_issue": int(result.issues[target_index]),
                    "history_through_issue": int(result.issues[target_index - 1]),
                    "recent_window": parameter.recent_window,
                    "reference_window": parameter.reference_window,
                    "inner_mean_brier": _float(
                        result.inner_mean_brier[offset, parameter_index]
                    ),
                    "selected": parameter_index == selected,
                    "change_score": _float(
                        result.candidate_change_scores[offset, parameter_index]
                    ),
                    "change_threshold_quantile": _float(parameter.change_quantile),
                    "change_threshold": _float(
                        result.candidate_thresholds[offset, parameter_index]
                    ),
                    "candidate_high_change": bool(
                        result.candidate_high_change[offset, parameter_index]
                    ),
                    "threshold_first_score_target_issue": int(
                        result.issues[first_index]
                    ),
                    "threshold_last_score_target_issue": int(result.issues[last_index]),
                    "threshold_data_through_issue": int(result.issues[last_index - 1]),
                    "threshold_score_count": last_index - first_index + 1,
                    "selection_metric": "inner_mean_brier_score",
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def _state_rows(
    result: ChangepointEvaluationResult, metrics: ProbabilityMetrics
) -> Iterable[dict[str, object]]:
    fields = {
        "high_change": result.high_change,
        "normal": np.logical_not(result.high_change),
    }
    for state, mask in fields.items():
        count = int(mask.sum())

        def mean(values: FloatArray) -> str:
            return "" if count == 0 else _float(float(values[mask].mean()))

        yield {
            "state": state,
            "issue_count": count,
            "issue_proportion": _float(count / len(result.outer_indices)),
            "changepoint_brier": mean(metrics.changepoint_brier),
            "dynamic_brier": mean(metrics.dynamic_brier),
            "uniform_brier": mean(metrics.uniform_brier),
            "changepoint_brier_delta_vs_dynamic": mean(
                cast(FloatArray, metrics.changepoint_brier - metrics.dynamic_brier)
            ),
            "changepoint_brier_delta_vs_uniform": mean(
                cast(FloatArray, metrics.changepoint_brier - metrics.uniform_brier)
            ),
            "changepoint_log_loss": mean(metrics.changepoint_log_loss),
            "dynamic_log_loss": mean(metrics.dynamic_log_loss),
            "uniform_log_loss": mean(metrics.uniform_log_loss),
            "changepoint_log_loss_delta_vs_dynamic": mean(
                cast(
                    FloatArray,
                    metrics.changepoint_log_loss - metrics.dynamic_log_loss,
                )
            ),
            "changepoint_log_loss_delta_vs_uniform": mean(
                cast(
                    FloatArray,
                    metrics.changepoint_log_loss - metrics.uniform_log_loss,
                )
            ),
            "independent_unit": "issue",
            "evidence_status": "exploratory_development_evidence",
        }


def write_state_metrics(
    path: Path,
    result: ChangepointEvaluationResult,
    metrics: ProbabilityMetrics,
) -> None:
    fields = (
        "state",
        "issue_count",
        "issue_proportion",
        "changepoint_brier",
        "dynamic_brier",
        "uniform_brier",
        "changepoint_brier_delta_vs_dynamic",
        "changepoint_brier_delta_vs_uniform",
        "changepoint_log_loss",
        "dynamic_log_loss",
        "uniform_log_loss",
        "changepoint_log_loss_delta_vs_dynamic",
        "changepoint_log_loss_delta_vs_uniform",
        "independent_unit",
        "evidence_status",
    )
    _write_csv(path, fields, _state_rows(result, metrics))


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
            summary = calibrations[strategy]
            for row in summary.bins:
                yield {
                    "strategy": strategy,
                    "bin_index": row.bin_index,
                    "lower_bound_inclusive": _float(row.lower_bound),
                    "upper_bound": _float(row.upper_bound),
                    "upper_bound_inclusive": row.bin_index == len(summary.bins) - 1,
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
                        summary.expected_calibration_error
                    ),
                    "evidence_status": "exploratory_development_evidence",
                }

    _write_csv(path, fields, rows())


def write_trigger_intervals(path: Path, intervals: tuple[TriggerInterval, ...]) -> None:
    fields = (
        "interval_index",
        "start_issue",
        "end_issue",
        "issue_count",
        "evidence_status",
    )
    _write_csv(
        path,
        fields,
        (
            {
                "interval_index": index,
                "start_issue": interval.start_issue,
                "end_issue": interval.end_issue,
                "issue_count": interval.issue_count,
                "evidence_status": "exploratory_development_evidence",
            }
            for index, interval in enumerate(intervals, start=1)
        ),
    )


def _ordinary_mean_interval(values: FloatArray) -> tuple[float, float, float, float]:
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
    result: ChangepointEvaluationResult,
    config: ChangepointEvaluationConfig,
    metrics: ProbabilityMetrics,
    hits_by_strategy: dict[str, FloatArray],
    calibrations: dict[str, CalibrationResult],
    intervals: tuple[TriggerInterval, ...],
    *,
    data_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> str:
    changepoint_brier = _ordinary_mean_interval(metrics.changepoint_brier)
    dynamic_brier = _ordinary_mean_interval(metrics.dynamic_brier)
    uniform_brier = _ordinary_mean_interval(metrics.uniform_brier)
    brier_vs_dynamic = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_brier - metrics.dynamic_brier)
    )
    brier_vs_uniform = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_brier - metrics.uniform_brier)
    )
    changepoint_log = _ordinary_mean_interval(metrics.changepoint_log_loss)
    dynamic_log = _ordinary_mean_interval(metrics.dynamic_log_loss)
    uniform_log = _ordinary_mean_interval(metrics.uniform_log_loss)
    log_vs_dynamic = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_log_loss - metrics.dynamic_log_loss)
    )
    log_vs_uniform = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_log_loss - metrics.uniform_log_loss)
    )
    trigger_count = int(result.high_change.sum())
    rank_differences = result.ranking_differs_from_fixed_exponential
    rank_difference_count = int(rank_differences.sum())
    period_count = len(result.outer_indices)
    probability_error = float(np.max(np.abs(result.posterior_mean.sum(axis=1) - 20.0)))
    window_counts = Counter(
        result.parameters[int(index)].recent_window
        for index in result.selected_parameter_indices
    )

    probability_table = _markdown_table(
        ("概率模型", "平均Brier", "相对动态", "相对随机", "平均log loss", "ECE"),
        (
            (
                "uniform_random",
                f"{uniform_brier[0]:.9f}",
                f"{uniform_brier[0] - dynamic_brier[0]:+.9f}",
                "0",
                f"{uniform_log[0]:.9f}",
                f"{calibrations['uniform_random'].expected_calibration_error:.9f}",
            ),
            (
                DYNAMIC_STRATEGY,
                f"{dynamic_brier[0]:.9f}",
                "0",
                f"{dynamic_brier[0] - uniform_brier[0]:+.9f}",
                f"{dynamic_log[0]:.9f}",
                f"{calibrations[DYNAMIC_STRATEGY].expected_calibration_error:.9f}",
            ),
            (
                CHANGEPOINT_STRATEGY,
                f"{changepoint_brier[0]:.9f}",
                f"{brier_vs_dynamic[0]:+.9f}",
                f"{brier_vs_uniform[0]:+.9f}",
                f"{changepoint_log[0]:.9f}",
                f"{calibrations[CHANGEPOINT_STRATEGY].expected_calibration_error:.9f}",
            ),
        ),
    )
    topk_table = _markdown_table(
        ("策略", "Top-1", "Top-5", "Top-10", "Top-10相对理论", "Top-10相对随机seed"),
        (
            (
                strategy,
                f"{hits_by_strategy[strategy][:, 0].mean():.6f}",
                f"{hits_by_strategy[strategy][:, 4].mean():.6f}",
                f"{hits_by_strategy[strategy][:, 9].mean():.6f}",
                f"{hits_by_strategy[strategy][:, 9].mean() - 2.5:+.6f}",
                f"{hits_by_strategy[strategy][:, 9].mean() - hits_by_strategy['uniform_random'][:, 9].mean():+.6f}",
            )
            for strategy in PHASE2_COMPARATOR_STRATEGIES
        ),
    )
    changepoint_topk_table = _markdown_table(
        ("k", "平均命中", "随机理论k/4", "excess_hits", "相对随机seed"),
        (
            (
                str(k),
                f"{hits_by_strategy[CHANGEPOINT_STRATEGY][:, k - 1].mean():.6f}",
                f"{k / 4.0:.6f}",
                f"{hits_by_strategy[CHANGEPOINT_STRATEGY][:, k - 1].mean() - k / 4.0:+.6f}",
                f"{hits_by_strategy[CHANGEPOINT_STRATEGY][:, k - 1].mean() - hits_by_strategy['uniform_random'][:, k - 1].mean():+.6f}",
            )
            for k in range(1, 11)
        ),
    )
    state_rows = list(_state_rows(result, metrics))
    state_table = _markdown_table(
        ("状态", "期数", "占比", "变点Brier", "相对动态", "相对随机", "变点log loss"),
        (
            (
                str(row["state"]),
                str(row["issue_count"]),
                f"{float(cast(str, row['issue_proportion'])):.2%}",
                str(row["changepoint_brier"]) or "—",
                str(row["changepoint_brier_delta_vs_dynamic"]) or "—",
                str(row["changepoint_brier_delta_vs_uniform"]) or "—",
                str(row["changepoint_log_loss"]) or "—",
            )
            for row in state_rows
        ),
    )
    interval_table = _markdown_table(
        ("序号", "起始期", "结束期", "连续期数"),
        (
            (
                str(index),
                str(interval.start_issue),
                str(interval.end_issue),
                str(interval.issue_count),
            )
            for index, interval in enumerate(intervals, start=1)
        ),
    )
    window_table = _markdown_table(
        ("recent_window", "被选外层期数", "占比"),
        (
            (
                str(window),
                str(window_counts.get(window, 0)),
                f"{window_counts.get(window, 0) / period_count:.2%}",
            )
            for window in RECENT_WINDOW_GRID
        ),
    )
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    relative_data_path = data_path.relative_to(project_root).as_posix()
    first_outer = int(result.outer_issues[0])
    last_outer = int(result.outer_issues[-1])
    rolling_top10 = float(hits_by_strategy["rolling_frequency"][:, 9].mean())
    cp_top10 = float(hits_by_strategy[CHANGEPOINT_STRATEGY][:, 9].mean())

    return f"""# 快乐8策略 v2 第二阶段在线变点探索性研究报告

## 技术摘要

- 本报告仅是 **exploratory development evidence（探索性开发证据）**。旧 final holdout 已被查看，结果不是新的独立 holdout、确认性证据或显著性检验。
- `{CHANGEPOINT_STRATEGY}` 在 `{period_count}` 个外层开奖期的平均 Brier score 为 `{changepoint_brier[0]:.9f}`；相对 `{DYNAMIC_STRATEGY}` 为 `{brier_vs_dynamic[0]:+.9f}`，相对公平随机概率为 `{brier_vs_uniform[0]:+.9f}`。Brier 越低越好。
- 平均 Bernoulli log loss 为 `{changepoint_log[0]:.9f}`；相对动态贝叶斯 `{log_vs_dynamic[0]:+.9f}`，相对随机 `{log_vs_uniform[0]:+.9f}`。这些差值的普通均值标准误近似区间未校正时间相关性，只作描述。
- 共 `{trigger_count}` 期触发 high_change，占 `{trigger_count / period_count:.2%}`；完整80位排名有 `{rank_difference_count}` 期（`{rank_difference_count / period_count:.2%}`）不同于固定 `decay=0.99` 的指数频率。若该计数为零，第二阶段实现按预注册标准视为失败。
- 变点模型 Top-10 平均命中 `{cp_top10:.6f}`，rolling_frequency 为 `{rolling_top10:.6f}`，期内随机 seed 集成为 `{hits_by_strategy['uniform_random'][:, 9].mean():.6f}`；这些已查看历史上的差异不能证明真实预测能力提高。

## 概率表现未构成确认性优势证据

概率比较以每期开奖期的80维联合评分为单位。负差表示相对基线更低；没有把同一期80个号码、多个 k 或20个随机 seed 当作独立样本。

{probability_table}

逐期 Brier 差值的普通均值标准误近似区间：相对动态贝叶斯 `[{brier_vs_dynamic[2]:+.9f}, {brier_vs_dynamic[3]:+.9f}]`，相对随机 `[{brier_vs_uniform[2]:+.9f}, {brier_vs_uniform[3]:+.9f}]`。逐期 log loss 差值对应区间为 `[{log_vs_dynamic[2]:+.9f}, {log_vs_dynamic[3]:+.9f}]` 和 `[{log_vs_uniform[2]:+.9f}, {log_vs_uniform[3]:+.9f}]`。这些区间未校正潜在时间相关性，不是确认性置信区间或显著性检验。

## 七个策略的 Top-k 历史比较

`uniform_random` 的20个固定 seed 先在同期开奖期内求均值；其他策略每期只有一个确定性排名。独立统计单位始终是开奖期。

{topk_table}

变点模型完整 Top-1 至 Top-10：

{changepoint_topk_table}

## high_change 触发了可审计的自适应路径

高变化与正常状态的概率表现分开如下；状态是预测前形成的，不使用该期结果。

{state_table}

连续 high_change 时间段：

{interval_table}

三个预注册近期窗口的内层选择结果：

{window_table}

## 数据范围与指标定义

- 输入：`{relative_data_path}`，SHA-256 `{data_hash}`。
- 历史共 `{len(result.issues)}` 期，期号 `{int(result.issues[0])}` 至 `{int(result.issues[-1])}`；外层期 `{first_outer}` 至 `{last_outer}`。
- Brier：逐期开奖 `mean((p_i-y_i)^2)`；log loss 先把概率截断到 `[1e-9, 1-1e-9]`。
- Calibration：固定10个等宽概率箱；ECE 按每箱记录数占比加权。
- Top-k：实际命中、理论期望 `k/4` 及 `excess_hits=hits-k/4`。
- 每期80个变点概率之和与20的最大绝对误差为 `{probability_error:.3e}`。

## 模型用频率偏离切换旧数据权重

对目标期 `t`，近期窗口长度 `r ∈ {{{', '.join(map(str, RECENT_WINDOW_GRID))}}}`，参考窗口固定 `{REFERENCE_WINDOW}` 期且紧邻近期窗之前、互不重叠：

```text
p_recent_i,t    = mean(y_i) over [t-r, t)
p_reference_i,t = mean(y_i) over [t-r-240, t-r)
change_score_t  = Σ_i |p_recent_i,t - p_reference_i,t|
```

该分数不证明开奖机制改变，只检测历史频率结构是否偏离；公平彩票的随机波动本身也会制造假变点。

对每个窗口和目标期，阈值是该目标期之前所有可得历史变化分数的固定90%分位数，当前分数不进入自身阈值。`change_score_t > threshold_t` 时使用最近 `{MINIMUM_EFFECTIVE_HISTORY}` 期、`decay={HIGH_CHANGE_DECAY}`；否则保留全部可得历史、`decay={NORMAL_DECAY}`。先验强度固定 `{CHANGEPOINT_PRIOR_STRENGTH:g}`，公平中心 `0.25`：

```text
alpha_i,t = 80 × 0.25 + Σ adaptive_weight(age) × y_i
beta_i,t  = 80 × 0.75 + Σ adaptive_weight(age) × (1-y_i)
p_i,t     = alpha_i,t / (alpha_i,t + beta_i,t)
```

号码按 `p_i,t` 降序、同分号码升序。自适应衰减与高变化时的120期截断使排序可以不同于固定 decay 指数频率；本次历史中不同排名期数为 `{rank_difference_count}`。

## 嵌套时间验证没有读取外层目标结果

1. 前 `{config.minimum_history}` 期仅作初始历史。
2. 接下来 `{config.minimum_inner_observations}` 个目标期形成首次窗口选择证据；每个内层预测的变化分数、阈值和概率也只用该目标之前的数据。
3. 每个外层期分别计算三个窗口在全部更早内层目标上的平均 Brier，选择最低者；同分按 `30, 60, 120` 顺序。
4. 选窗与阈值固定后预测外层期，当前结果随后才进入下一期历史。没有随机打乱期号，没有用 Top-10 选窗，也没有扩大预注册窗口网格。

## 限制、不确定性与失败模式

- **旧 holdout 已查看。** 全部历史结果只能称为 exploratory development evidence。
- **没有真实预测提升主张。** 如果彩票公平且独立，任何历史模型都不应有稳定优势；正负差异都可能是随机波动。
- **变化分数会误报。** 90%历史分位数按设计会在稳定随机序列中产生部分 high_change；触发不是机制变化的证据。
- **窗口仍经过历史开发比较。** 虽然只在外层前的内层 Brier 上从三个预注册窗口选择，研究者自由度和历史选择偏差仍存在。
- **区间仅描述。** 普通均值标准误近似未校正时间相关性，也未进行多重比较校正。
- **排名差异不等于优势。** 排名改变只证明自适应机制不是固定指数频率的重命名，不证明改变方向有用。
- **确认必须面向未来。** 真正确认需要在模型和输出契约冻结后，依靠未来、预先封存且未被开发查看的开奖。

## 后续步骤

1. 审核 `changepoint_history.csv` 的阈值时间边界、窗口选择记录和触发段，不根据结果扩网格。
2. 若继续研究，在任何未来开奖前冻结模型源码、数据截止点、概率和候选 manifest。
3. 预先规定未来确认期数、主要指标、停止规则和变点误报解释，再逐期追加不可回写记录。

## 仍需回答的问题

- 90%分位阈值在未来冻结期应保持扩展更新，还是在冻结时固定数值？
- 未来确认的主要指标应是 Brier、log loss，还是预先指定的 Top-k？
- 变点状态是否需要外部机制证据才能触发任何实际决策？

本报告不能依据历史回测声称提高了真实预测能力。
"""


def execute_backtest(
    *,
    data_path: Path,
    output_dir: Path,
    report_path: Path,
    project_root: Path,
) -> ChangepointEvaluationResult:
    root = project_root.resolve()
    resolved_data = _resolve_within(root, str(data_path), "数据文件")
    resolved_output = _resolve_within(root, str(output_dir), "结果目录")
    resolved_report = _resolve_within(root, str(report_path), "报告文件")
    if resolved_output != (root / "results" / "research_v2_changepoint").resolve():
        raise ValueError("第二阶段结果目录必须为results/research_v2_changepoint")
    if resolved_report != (root / "reports" / "kl8_v2_changepoint_report.md").resolve():
        raise ValueError("第二阶段报告必须为reports/kl8_v2_changepoint_report.md")

    issues, draws = load_history_csv(resolved_data)
    config = ChangepointEvaluationConfig()
    result = evaluate_changepoint_walk_forward(issues, draws, config=config)
    metrics = calculate_probability_metrics(result)
    topk_hits = calculate_topk_hits(result)
    calibrations = calculate_calibration(result)
    intervals = find_trigger_intervals(result)
    if not bool(result.ranking_differs_from_fixed_exponential.any()):
        raise RuntimeError("变点排名从未不同于固定指数频率，第二阶段实现失败")

    resolved_output.mkdir(parents=True, exist_ok=True)
    write_issue_probabilities(resolved_output / OUTPUT_FILENAMES[0], result)
    write_issue_metrics(resolved_output / OUTPUT_FILENAMES[1], result, metrics)
    write_topk_metrics(resolved_output / OUTPUT_FILENAMES[2], result, topk_hits)
    write_changepoint_history(resolved_output / OUTPUT_FILENAMES[3], result)
    write_state_metrics(resolved_output / OUTPUT_FILENAMES[4], result, metrics)
    write_calibration(resolved_output / OUTPUT_FILENAMES[5], calibrations)
    write_trigger_intervals(resolved_output / OUTPUT_FILENAMES[6], intervals)
    report = build_report(
        result,
        config,
        metrics,
        topk_hits,
        calibrations,
        intervals,
        data_path=resolved_data,
        project_root=root,
    )
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = resolved_report.with_suffix(resolved_report.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(resolved_report)
    return result


def run() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    result = execute_backtest(
        data_path=Path(str(args.data)),
        output_dir=Path(str(args.output_dir)),
        report_path=Path(str(args.report)),
        project_root=root,
    )
    print(
        "完成："
        f"历史={len(result.issues)}期，外层探索={len(result.outer_indices)}期，"
        f"high_change={int(result.high_change.sum())}期，"
        f"排名变化={int(result.ranking_differs_from_fixed_exponential.sum())}期，"
        "报告=reports/kl8_v2_changepoint_report.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
