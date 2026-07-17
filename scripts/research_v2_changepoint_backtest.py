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
    FIXED_HIGH_STRATEGY,
    FIXED_NORMAL_STRATEGY,
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
    FIXED_NORMAL_STRATEGY,
    FIXED_HIGH_STRATEGY,
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
    "switch_diagnostics.csv",
)


@dataclass(frozen=True)
class ProbabilityMetrics:
    """逐期开奖期的五组概率评分。"""

    changepoint_brier: FloatArray
    changepoint_log_loss: FloatArray
    dynamic_brier: FloatArray
    dynamic_log_loss: FloatArray
    fixed_normal_brier: FloatArray
    fixed_normal_log_loss: FloatArray
    fixed_high_brier: FloatArray
    fixed_high_log_loss: FloatArray
    uniform_brier: FloatArray
    uniform_log_loss: FloatArray


@dataclass(frozen=True)
class TriggerInterval:
    """连续 high_change 外层期段。"""

    start_issue: int
    end_issue: int
    issue_count: int


@dataclass(frozen=True)
class SwitchDiagnostic:
    """自适应模型相对固定消融基线的状态内诊断。"""

    comparator: str
    scope: str
    issue_count: int
    ranking_difference_count: int
    ranking_difference_proportion: float
    probabilities_exactly_equal: bool


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
        fixed_normal_brier=cast(
            FloatArray,
            np.mean(np.square(result.fixed_normal_posterior_mean - outcomes), axis=1),
        ),
        fixed_normal_log_loss=log_losses(result.fixed_normal_posterior_mean),
        fixed_high_brier=cast(
            FloatArray,
            np.mean(np.square(result.fixed_high_posterior_mean - outcomes), axis=1),
        ),
        fixed_high_log_loss=log_losses(result.fixed_high_posterior_mean),
        uniform_brier=uniform_brier,
        uniform_log_loss=log_losses(uniform_probabilities),
    )


def _probability_metric_arrays(
    metrics: ProbabilityMetrics,
) -> dict[str, tuple[FloatArray, FloatArray]]:
    return {
        "uniform_random": (metrics.uniform_brier, metrics.uniform_log_loss),
        DYNAMIC_STRATEGY: (metrics.dynamic_brier, metrics.dynamic_log_loss),
        FIXED_NORMAL_STRATEGY: (
            metrics.fixed_normal_brier,
            metrics.fixed_normal_log_loss,
        ),
        FIXED_HIGH_STRATEGY: (
            metrics.fixed_high_brier,
            metrics.fixed_high_log_loss,
        ),
        CHANGEPOINT_STRATEGY: (
            metrics.changepoint_brier,
            metrics.changepoint_log_loss,
        ),
    }


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
        FIXED_NORMAL_STRATEGY: calibration_summary(
            result.fixed_normal_posterior_mean, outcomes
        ),
        FIXED_HIGH_STRATEGY: calibration_summary(
            result.fixed_high_posterior_mean, outcomes
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


def calculate_switch_diagnostics(
    result: ChangepointEvaluationResult,
) -> tuple[SwitchDiagnostic, ...]:
    """按全部、high_change和normal期比较自适应与两个固定消融模型。"""

    scopes = (
        ("all", np.ones(len(result.outer_indices), dtype=np.bool_)),
        ("high_change", result.high_change),
        ("normal", np.logical_not(result.high_change)),
    )
    comparisons = (
        (
            FIXED_NORMAL_STRATEGY,
            result.ranking_differs_from_fixed_normal,
            result.fixed_normal_posterior_mean,
        ),
        (
            FIXED_HIGH_STRATEGY,
            result.ranking_differs_from_fixed_high,
            result.fixed_high_posterior_mean,
        ),
    )
    rows: list[SwitchDiagnostic] = []
    for comparator, ranking_differences, fixed_probabilities in comparisons:
        for scope, mask in scopes:
            count = int(mask.sum())
            difference_count = int(ranking_differences[mask].sum())
            rows.append(
                SwitchDiagnostic(
                    comparator=comparator,
                    scope=scope,
                    issue_count=count,
                    ranking_difference_count=difference_count,
                    ranking_difference_proportion=(
                        0.0 if count == 0 else difference_count / count
                    ),
                    probabilities_exactly_equal=bool(
                        np.array_equal(
                            result.posterior_mean[mask], fixed_probabilities[mask]
                        )
                    ),
                )
            )
    return tuple(rows)


def write_issue_probabilities(path: Path, result: ChangepointEvaluationResult) -> None:
    fields = (
        "issue",
        "history_through_issue",
        "history_issue_count",
        "number",
        "outcome",
        "posterior_mean",
        "fixed_normal_posterior_mean",
        "fixed_high_posterior_mean",
        "equals_active_fixed_probability",
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
                    "fixed_normal_posterior_mean": _float(
                        result.fixed_normal_posterior_mean[outer_offset, number_index]
                    ),
                    "fixed_high_posterior_mean": _float(
                        result.fixed_high_posterior_mean[outer_offset, number_index]
                    ),
                    "equals_active_fixed_probability": bool(
                        result.posterior_mean[outer_offset, number_index]
                        == (
                            result.fixed_high_posterior_mean[outer_offset, number_index]
                            if result.high_change[outer_offset]
                            else result.fixed_normal_posterior_mean[
                                outer_offset, number_index
                            ]
                        )
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
        "brier_delta_vs_fixed_normal_bayesian",
        "brier_delta_vs_fixed_high_bayesian",
        "log_loss_delta_vs_uniform",
        "log_loss_delta_vs_dynamic_bayesian",
        "log_loss_delta_vs_fixed_normal_bayesian",
        "log_loss_delta_vs_fixed_high_bayesian",
        "high_change",
        "independent_time_cluster",
        "evidence_status",
    )

    def rows() -> Iterable[dict[str, object]]:
        arrays = _probability_metric_arrays(metrics)
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
                    "brier_delta_vs_fixed_normal_bayesian": _float(
                        brier - float(metrics.fixed_normal_brier[offset])
                    ),
                    "brier_delta_vs_fixed_high_bayesian": _float(
                        brier - float(metrics.fixed_high_brier[offset])
                    ),
                    "log_loss_delta_vs_uniform": _float(
                        log_loss - float(metrics.uniform_log_loss[offset])
                    ),
                    "log_loss_delta_vs_dynamic_bayesian": _float(
                        log_loss - float(metrics.dynamic_log_loss[offset])
                    ),
                    "log_loss_delta_vs_fixed_normal_bayesian": _float(
                        log_loss - float(metrics.fixed_normal_log_loss[offset])
                    ),
                    "log_loss_delta_vs_fixed_high_bayesian": _float(
                        log_loss - float(metrics.fixed_high_log_loss[offset])
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
            "fixed_normal_brier": mean(metrics.fixed_normal_brier),
            "fixed_high_brier": mean(metrics.fixed_high_brier),
            "uniform_brier": mean(metrics.uniform_brier),
            "changepoint_brier_delta_vs_dynamic": mean(
                cast(FloatArray, metrics.changepoint_brier - metrics.dynamic_brier)
            ),
            "changepoint_brier_delta_vs_uniform": mean(
                cast(FloatArray, metrics.changepoint_brier - metrics.uniform_brier)
            ),
            "changepoint_brier_delta_vs_fixed_normal": mean(
                cast(
                    FloatArray,
                    metrics.changepoint_brier - metrics.fixed_normal_brier,
                )
            ),
            "changepoint_brier_delta_vs_fixed_high": mean(
                cast(
                    FloatArray,
                    metrics.changepoint_brier - metrics.fixed_high_brier,
                )
            ),
            "changepoint_log_loss": mean(metrics.changepoint_log_loss),
            "dynamic_log_loss": mean(metrics.dynamic_log_loss),
            "fixed_normal_log_loss": mean(metrics.fixed_normal_log_loss),
            "fixed_high_log_loss": mean(metrics.fixed_high_log_loss),
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
            "changepoint_log_loss_delta_vs_fixed_normal": mean(
                cast(
                    FloatArray,
                    metrics.changepoint_log_loss - metrics.fixed_normal_log_loss,
                )
            ),
            "changepoint_log_loss_delta_vs_fixed_high": mean(
                cast(
                    FloatArray,
                    metrics.changepoint_log_loss - metrics.fixed_high_log_loss,
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
        "fixed_normal_brier",
        "fixed_high_brier",
        "uniform_brier",
        "changepoint_brier_delta_vs_dynamic",
        "changepoint_brier_delta_vs_uniform",
        "changepoint_brier_delta_vs_fixed_normal",
        "changepoint_brier_delta_vs_fixed_high",
        "changepoint_log_loss",
        "dynamic_log_loss",
        "fixed_normal_log_loss",
        "fixed_high_log_loss",
        "uniform_log_loss",
        "changepoint_log_loss_delta_vs_dynamic",
        "changepoint_log_loss_delta_vs_uniform",
        "changepoint_log_loss_delta_vs_fixed_normal",
        "changepoint_log_loss_delta_vs_fixed_high",
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


def write_switch_diagnostics(
    path: Path, diagnostics: tuple[SwitchDiagnostic, ...]
) -> None:
    fields = (
        "comparator",
        "scope",
        "issue_count",
        "ranking_difference_count",
        "ranking_difference_proportion",
        "probabilities_exactly_equal",
        "evidence_status",
    )
    _write_csv(
        path,
        fields,
        (
            {
                "comparator": row.comparator,
                "scope": row.scope,
                "issue_count": row.issue_count,
                "ranking_difference_count": row.ranking_difference_count,
                "ranking_difference_proportion": _float(
                    row.ranking_difference_proportion
                ),
                "probabilities_exactly_equal": row.probabilities_exactly_equal,
                "evidence_status": "exploratory_development_evidence",
            }
            for row in diagnostics
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
    diagnostics: tuple[SwitchDiagnostic, ...],
    *,
    data_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> str:
    metric_arrays = _probability_metric_arrays(metrics)
    brier_intervals = {
        strategy: _ordinary_mean_interval(values[0])
        for strategy, values in metric_arrays.items()
    }
    log_intervals = {
        strategy: _ordinary_mean_interval(values[1])
        for strategy, values in metric_arrays.items()
    }
    brier_vs_normal = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_brier - metrics.fixed_normal_brier)
    )
    brier_vs_high = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_brier - metrics.fixed_high_brier)
    )
    log_vs_normal = _ordinary_mean_interval(
        cast(
            FloatArray,
            metrics.changepoint_log_loss - metrics.fixed_normal_log_loss,
        )
    )
    log_vs_high = _ordinary_mean_interval(
        cast(
            FloatArray,
            metrics.changepoint_log_loss - metrics.fixed_high_log_loss,
        )
    )
    brier_vs_dynamic = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_brier - metrics.dynamic_brier)
    )
    brier_vs_uniform = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_brier - metrics.uniform_brier)
    )
    log_vs_dynamic = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_log_loss - metrics.dynamic_log_loss)
    )
    log_vs_uniform = _ordinary_mean_interval(
        cast(FloatArray, metrics.changepoint_log_loss - metrics.uniform_log_loss)
    )
    trigger_count = int(result.high_change.sum())
    period_count = len(result.outer_indices)
    probability_error = float(np.max(np.abs(result.posterior_mean.sum(axis=1) - 20.0)))
    window_counts = Counter(
        result.parameters[int(index)].recent_window
        for index in result.selected_parameter_indices
    )
    diagnostic_lookup = {(row.comparator, row.scope): row for row in diagnostics}
    normal_all = diagnostic_lookup[(FIXED_NORMAL_STRATEGY, "all")]
    normal_high = diagnostic_lookup[(FIXED_NORMAL_STRATEGY, "high_change")]
    normal_normal = diagnostic_lookup[(FIXED_NORMAL_STRATEGY, "normal")]
    high_all = diagnostic_lookup[(FIXED_HIGH_STRATEGY, "all")]
    high_high = diagnostic_lookup[(FIXED_HIGH_STRATEGY, "high_change")]
    high_normal = diagnostic_lookup[(FIXED_HIGH_STRATEGY, "normal")]
    if not normal_normal.probabilities_exactly_equal:
        raise ValueError("报告前校验失败：normal状态不等于fixed_normal")
    if not high_high.probabilities_exactly_equal:
        raise ValueError("报告前校验失败：high_change状态不等于fixed_high")

    simultaneously_better = (
        brier_vs_normal[0] < 0.0
        and brier_vs_high[0] < 0.0
        and log_vs_normal[0] < 0.0
        and log_vs_high[0] < 0.0
    )
    if simultaneously_better:
        ablation_conclusion = (
            "在这段已查看历史中，变点切换的平均 Brier 和 log loss 均低于两个固定消融基线；"
            "这仍然只是探索性历史消融结果，不能证明未来优势。"
        )
    else:
        ablation_conclusion = (
            "当前历史结果没有证明变化检测和自适应切换优于简单固定权重方案："
            "变点模型没有在平均 Brier 与 log loss 上同时优于两个固定消融基线。"
        )

    probability_table = _markdown_table(
        ("概率模型", "平均Brier", "平均log loss", "ECE"),
        (
            (
                strategy,
                f"{brier_intervals[strategy][0]:.9f}",
                f"{log_intervals[strategy][0]:.9f}",
                f"{calibrations[strategy].expected_calibration_error:.9f}",
            )
            for strategy in PROBABILITY_STRATEGIES
        ),
    )
    ablation_delta_table = _markdown_table(
        ("指标", "相对fixed_normal", "普通近似区间", "相对fixed_high", "普通近似区间"),
        (
            (
                "Brier",
                f"{brier_vs_normal[0]:+.9f}",
                f"[{brier_vs_normal[2]:+.9f}, {brier_vs_normal[3]:+.9f}]",
                f"{brier_vs_high[0]:+.9f}",
                f"[{brier_vs_high[2]:+.9f}, {brier_vs_high[3]:+.9f}]",
            ),
            (
                "Bernoulli log loss",
                f"{log_vs_normal[0]:+.9f}",
                f"[{log_vs_normal[2]:+.9f}, {log_vs_normal[3]:+.9f}]",
                f"{log_vs_high[0]:+.9f}",
                f"[{log_vs_high[2]:+.9f}, {log_vs_high[3]:+.9f}]",
            ),
            (
                "ECE",
                f"{calibrations[CHANGEPOINT_STRATEGY].expected_calibration_error - calibrations[FIXED_NORMAL_STRATEGY].expected_calibration_error:+.9f}",
                "—",
                f"{calibrations[CHANGEPOINT_STRATEGY].expected_calibration_error - calibrations[FIXED_HIGH_STRATEGY].expected_calibration_error:+.9f}",
                "—",
            ),
        ),
    )
    topk_table = _markdown_table(
        ("策略", *(f"Top-{k}" for k in range(1, 11))),
        (
            (
                strategy,
                *(
                    f"{hits_by_strategy[strategy][:, k - 1].mean():.6f}"
                    for k in range(1, 11)
                ),
            )
            for strategy in PHASE2_COMPARATOR_STRATEGIES
        ),
    )
    topk_ablation_table = _markdown_table(
        ("k", "changepoint", "相对fixed_normal", "相对fixed_high"),
        (
            (
                str(k),
                f"{hits_by_strategy[CHANGEPOINT_STRATEGY][:, k - 1].mean():.6f}",
                f"{hits_by_strategy[CHANGEPOINT_STRATEGY][:, k - 1].mean() - hits_by_strategy[FIXED_NORMAL_STRATEGY][:, k - 1].mean():+.6f}",
                f"{hits_by_strategy[CHANGEPOINT_STRATEGY][:, k - 1].mean() - hits_by_strategy[FIXED_HIGH_STRATEGY][:, k - 1].mean():+.6f}",
            )
            for k in range(1, 11)
        ),
    )
    switch_table = _markdown_table(
        ("比较", "范围", "范围期数", "排名不同期数", "比例"),
        (
            (
                "changepoint vs fixed_normal",
                "全部外层期",
                str(normal_all.issue_count),
                str(normal_all.ranking_difference_count),
                f"{normal_all.ranking_difference_proportion:.2%}",
            ),
            (
                "changepoint vs fixed_normal",
                "high_change期",
                str(normal_high.issue_count),
                str(normal_high.ranking_difference_count),
                f"{normal_high.ranking_difference_proportion:.2%}",
            ),
            (
                "changepoint vs fixed_high",
                "全部外层期",
                str(high_all.issue_count),
                str(high_all.ranking_difference_count),
                f"{high_all.ranking_difference_proportion:.2%}",
            ),
            (
                "changepoint vs fixed_high",
                "normal期",
                str(high_normal.issue_count),
                str(high_normal.ranking_difference_count),
                f"{high_normal.ranking_difference_proportion:.2%}",
            ),
        ),
    )
    state_rows = list(_state_rows(result, metrics))
    state_table = _markdown_table(
        (
            "状态",
            "期数",
            "占比",
            "变点Brier",
            "相对fixed_normal",
            "相对fixed_high",
            "变点log loss",
        ),
        (
            (
                str(row["state"]),
                str(row["issue_count"]),
                f"{float(cast(str, row['issue_proportion'])):.2%}",
                str(row["changepoint_brier"]) or "—",
                str(row["changepoint_brier_delta_vs_fixed_normal"]) or "—",
                str(row["changepoint_brier_delta_vs_fixed_high"]) or "—",
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
- `{CHANGEPOINT_STRATEGY}` 的唯一新增机制，是用目标期前变化分数在 `{FIXED_NORMAL_STRATEGY}` 与 `{FIXED_HIGH_STRATEGY}` 两个预先固定模型之间切换；两个消融基线完整报告，没有按结果择一。
- 平均 Brier 为 `{brier_intervals[CHANGEPOINT_STRATEGY][0]:.9f}`，相对 fixed_normal `{brier_vs_normal[0]:+.9f}`、相对 fixed_high `{brier_vs_high[0]:+.9f}`；平均 log loss 为 `{log_intervals[CHANGEPOINT_STRATEGY][0]:.9f}`，相对两基线分别为 `{log_vs_normal[0]:+.9f}` 与 `{log_vs_high[0]:+.9f}`。
- {ablation_conclusion}
- 共 `{trigger_count}` 期触发 high_change，占 `{trigger_count / period_count:.2%}`。normal 期自适应概率逐项等于 fixed_normal：`{normal_normal.probabilities_exactly_equal}`；high_change 期逐项等于 fixed_high：`{high_high.probabilities_exactly_equal}`。
- 变点模型 Top-10 平均命中 `{cp_top10:.6f}`，rolling_frequency 为 `{rolling_top10:.6f}`，期内随机 seed 集成为 `{hits_by_strategy['uniform_random'][:, 9].mean():.6f}`；这些已查看历史上的差异不能证明真实预测能力提高。

## 两个固定消融基线隔离了状态切换的贡献

概率比较以每期开奖期的80维联合评分为单位。负差表示自适应模型相对基线更低；没有把同一期80个号码、多个 k 或20个随机 seed 当作独立样本。

{probability_table}

{ablation_delta_table}

这些是逐期开奖差值的普通均值标准误近似区间，未校正潜在时间相关性，只作描述；ECE 是整体校准汇总，未构造伪逐期区间。本报告不计算或报告机会性 p 值。作为上下文，changepoint 相对 dynamic_bayesian 的 Brier/log loss 差值为 `{brier_vs_dynamic[0]:+.9f}`/`{log_vs_dynamic[0]:+.9f}`，相对 uniform_random 为 `{brier_vs_uniform[0]:+.9f}`/`{log_vs_uniform[0]:+.9f}`。

## 九个策略完整 Top-1 至 Top-10 历史比较

`uniform_random` 的20个固定 seed 先在同期开奖期内求均值；其他策略每期只有一个确定性排名。独立统计单位始终是开奖期。

{topk_table}

变点模型相对两个固定消融基线的完整 Top-1 至 Top-10 差值：

{topk_ablation_table}

## 真正由状态切换造成的排名差异只出现在相反固定状态

{switch_table}

normal 状态中 changepoint 概率逐项等于 fixed_normal，high_change 状态中逐项等于 fixed_high。与固定 `decay=0.99` 指数频率的旧“100%不同”检查不能隔离状态切换，因为 normal 本身使用 `0.995`、high 本身使用 `0.95` 和120期截断；该数字已从机制有效性判据中删除。现在的检查直接比较两个被切换的固定模型。

## high_change 触发路径及状态内表现可审计

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

两个消融基线不调参、不选优，直接复用同一 posterior grid：

- `{FIXED_NORMAL_STRATEGY}`：始终使用全部目标期前历史、`decay={NORMAL_DECAY}`、`prior_strength={CHANGEPOINT_PRIOR_STRENGTH:g}`。
- `{FIXED_HIGH_STRATEGY}`：始终使用最近 `{MINIMUM_EFFECTIVE_HISTORY}` 期目标期前历史、`decay={HIGH_CHANGE_DECAY}`、`prior_strength={CHANGEPOINT_PRIOR_STRENGTH:g}`。

对每个窗口和目标期，阈值是该目标期之前所有可得历史变化分数的固定90%分位数，当前分数不进入自身阈值。`change_score_t > threshold_t` 时选择 fixed_high，否则选择 fixed_normal。先验公平中心为 `0.25`：

```text
alpha_i,t = 80 × 0.25 + Σ adaptive_weight(age) × y_i
beta_i,t  = 80 × 0.75 + Σ adaptive_weight(age) × (1-y_i)
p_i,t     = alpha_i,t / (alpha_i,t + beta_i,t)
```

号码按 `p_i,t` 降序、同分号码升序。changepoint 没有第三套概率公式；它的唯一新增机制是根据目标期前状态在两个固定概率向量之间切换，其价值必须由上面的完整消融比较判断。

## 嵌套时间验证没有读取外层目标结果

1. 前 `{config.minimum_history}` 期仅作初始历史。
2. 接下来 `{config.minimum_inner_observations}` 个目标期形成首次窗口选择证据；每个内层预测的变化分数、阈值和概率也只用该目标之前的数据。
3. 每个外层期分别计算三个窗口在全部更早内层目标上的平均 Brier，选择最低者；同分按 `30, 60, 120` 顺序。
4. 选窗与阈值固定后预测外层期，当前结果随后才进入下一期历史。没有随机打乱期号，没有用 Top-10 选窗，也没有扩大预注册窗口网格。

## 限制、不确定性与失败模式

- **旧 holdout 已查看。** 全部历史结果只能称为 exploratory development evidence。
- **没有真实预测提升主张。** 如果彩票公平且独立，任何历史模型都不应有稳定优势；正负差异都可能是随机波动。
- **变化分数会误报。** 90%历史分位数按设计会在稳定随机序列中产生部分 high_change；触发不是机制变化的证据。
- **消融结论仍是探索性的。** 即使 changepoint 优于某个固定基线，也只能称为历史探索性消融结果；{ablation_conclusion}
- **窗口仍经过历史开发比较。** 虽然只在外层前的内层 Brier 上从三个预注册窗口选择，研究者自由度和历史选择偏差仍存在。
- **区间仅描述。** 普通均值标准误近似未校正时间相关性，也未进行多重比较校正。
- **排名差异不等于优势。** 相对两个固定状态的排名改变只证明切换实际改变了输出，不证明改变方向有用。
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
    diagnostics = calculate_switch_diagnostics(result)

    resolved_output.mkdir(parents=True, exist_ok=True)
    write_issue_probabilities(resolved_output / OUTPUT_FILENAMES[0], result)
    write_issue_metrics(resolved_output / OUTPUT_FILENAMES[1], result, metrics)
    write_topk_metrics(resolved_output / OUTPUT_FILENAMES[2], result, topk_hits)
    write_changepoint_history(resolved_output / OUTPUT_FILENAMES[3], result)
    write_state_metrics(resolved_output / OUTPUT_FILENAMES[4], result, metrics)
    write_calibration(resolved_output / OUTPUT_FILENAMES[5], calibrations)
    write_trigger_intervals(resolved_output / OUTPUT_FILENAMES[6], intervals)
    write_switch_diagnostics(resolved_output / OUTPUT_FILENAMES[7], diagnostics)
    report = build_report(
        result,
        config,
        metrics,
        topk_hits,
        calibrations,
        intervals,
        diagnostics,
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
        "相对fixed_normal排名变化="
        f"{int(result.ranking_differs_from_fixed_normal.sum())}期，"
        "相对fixed_high排名变化="
        f"{int(result.ranking_differs_from_fixed_high.sum())}期，"
        "报告=reports/kl8_v2_changepoint_report.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
