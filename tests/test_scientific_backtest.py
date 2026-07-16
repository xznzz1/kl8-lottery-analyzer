# -*- coding: utf-8 -*-
"""科学回测的奖金、时间顺序、随机性和汇总回归测试。"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_baselines import _load_metadata, _resolve_scoped_path
from src.scientific.evaluation import (
    evaluate_indices,
    make_temporal_split,
    tune_on_validation,
)
from src.scientific.prizes import (
    MissingFloatingPrizeError,
    PrizeScenario,
    exact_two_ticket_metrics,
    prize_for_hits,
    rule_version_for_issue,
)
from src.scientific.statistics import (
    hit_distribution,
    holm_adjust,
    paired_mean_bootstrap_interval,
    risk_metrics,
    summarise_period_records,
    summarise_seed_ensemble,
    wilson_score_interval,
)
from src.scientific.strategies import (
    StrategyParameters,
    assert_legal_tickets,
    deterministic_tickets,
    random_tickets,
    score_numbers,
)


def _draws(rows: int = 50, offset: int = 0) -> np.ndarray:
    return np.asarray(
        [
            [((offset + row + index) % 80) + 1 for index in range(20)]
            for row in range(rows)
        ],
        dtype=int,
    )


def test_prize_rule_boundary_and_floating_missing_failure():
    scenario = PrizeScenario(pick10_jackpot=123456.0, label="测试情景")
    assert rule_version_for_issue(2025349) == "old"
    assert rule_version_for_issue(2025350) == "current"
    assert prize_for_hits(1, 1, scenario, issue=2025349) == 4.6
    assert prize_for_hits(1, 1, scenario, issue=2025350) == 4.5
    assert prize_for_hits(9, 9, scenario, issue=2025349) == 300000.0
    with pytest.raises(MissingFloatingPrizeError):
        prize_for_hits(9, 9, scenario, issue=2025350)
    assert prize_for_hits(10, 10, scenario, issue=2024001) == 123456.0


def test_current_rule_prize_table_and_exact_probabilities():
    scenario = PrizeScenario.cap_scenario()
    assert prize_for_hits(10, 8, scenario, rule_version="current") == 720.0
    assert prize_for_hits(9, 7, scenario, rule_version="current") == 225.0
    assert prize_for_hits(7, 0, scenario, rule_version="current") == 2.0
    metrics = exact_two_ticket_metrics(5, "disjoint", scenario)
    assert 0.0 <= metrics["any_prize_probability"] <= 1.0
    assert 0.0 <= metrics["profit_probability"] <= 1.0
    assert metrics["mean_hits_per_bet"] == pytest.approx(1.25)
    assert metrics["roi"] == pytest.approx(metrics["expected_prize"] / 4.0 - 1.0)


@pytest.mark.parametrize("play", range(1, 11))
@pytest.mark.parametrize("ticket_mode", ["disjoint", "independent"])
def test_ticket_legality_and_random_reproducibility(play, ticket_mode):
    scores = np.linspace(0.0, 1.0, 80)
    deterministic = deterministic_tickets(scores, play, ticket_mode)
    assert_legal_tickets(deterministic, play, ticket_mode)
    first = random_tickets(play, ticket_mode, seed=123, issue=2026001)
    second = random_tickets(play, ticket_mode, seed=123, issue=2026001)
    assert first == second
    assert_legal_tickets(first, play, ticket_mode)


def test_future_rows_do_not_change_prediction_for_target_issue():
    draws = _draws(30)
    issues = np.arange(2026001, 2026031)
    split_future = draws.copy()
    split_future[16:] = _draws(14, offset=37)
    parameters = StrategyParameters(rolling_window=10, decay=0.94)
    scenario = PrizeScenario.cap_scenario()

    first = evaluate_indices(
        issues, draws, [15], parameters, scenario, random_seeds=[1, 2, 3, 4, 5]
    )
    second = evaluate_indices(
        issues,
        split_future,
        [15],
        parameters,
        scenario,
        random_seeds=[1, 2, 3, 4, 5],
    )
    columns = ["strategy", "seed", "play", "ticket_mode", "ticket1", "ticket2"]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_validation_tuning_never_reads_final_holdout():
    draws = _draws(80)
    split = make_temporal_split(
        80,
        holdout_fraction=0.2,
        validation_fraction_of_pre_holdout=0.2,
        minimum_history=30,
    )
    changed_holdout = draws.copy()
    changed_holdout[split.holdout_start :] = _draws(
        len(changed_holdout) - split.holdout_start, offset=43
    )
    first, sensitivity1 = tune_on_validation(draws, split)
    second, sensitivity2 = tune_on_validation(changed_holdout, split)
    assert first == second
    pd.testing.assert_frame_equal(sensitivity1, sensitivity2)


def test_summary_and_risk_metrics_use_four_yuan_period_budget():
    records = pd.DataFrame(
        {
            "issue": [1, 2, 3, 4],
            "hits1": [1, 0, 2, 0],
            "hits2": [0, 1, 0, 0],
            "prize1": [4.5, 0.0, 19.0, 0.0],
            "prize2": [0.0, 4.5, 0.0, 0.0],
            "total_prize": [4.5, 4.5, 19.0, 0.0],
        }
    )
    summary = summarise_period_records(records, bootstrap_samples=50)
    assert summary["mean_hits_per_bet"] == pytest.approx(0.5)
    assert summary["profit_probability"] == pytest.approx(0.75)
    assert summary["expected_prize"] == pytest.approx(7.0)
    assert summary["roi"] == pytest.approx(0.75)
    assert summary["prize_at_least_1000_probability"] == 0.0
    assert summary["prize_at_least_1000_probability_ci95_high"] > 0.0
    risk = risk_metrics(records["total_prize"].to_numpy())
    assert risk["longest_losing_streak"] == 1


def test_random_ensemble_averages_events_before_summary_not_prize_amounts():
    records = pd.DataFrame(
        {
            "issue": [1, 2, 1, 2],
            "seed": [11, 11, 22, 22],
            "hits1": [1, 2, 0, 0],
            "hits2": [0, 0, 0, 0],
            "prize1": [8.0, 2000.0, 0.0, 0.0],
            "prize2": [0.0, 0.0, 0.0, 0.0],
            "total_prize": [8.0, 2000.0, 0.0, 0.0],
        }
    )
    summary = summarise_seed_ensemble(records, bootstrap_samples=50)
    assert summary["issues"] == 2
    assert summary["any_prize_probability"] == pytest.approx(0.5)
    assert summary["prize_at_least_1000_probability"] == pytest.approx(0.25)
    assert summary["expected_prize"] == pytest.approx(502.0)
    assert "prize_at_least_1000_probability_ci95_low" in summary
    assert summary["prize_at_least_10000_probability"] == 0.0
    assert summary["prize_at_least_10000_probability_ci95_high"] > 0.0


def test_wilson_interval_does_not_claim_zero_risk_after_zero_events():
    low, high = wilson_score_interval(0.0, 326)
    assert low == 0.0
    assert 0.0 < high < 0.02


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    order = np.argsort([0.01, 0.04, 0.03, 0.5])
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_paired_effect_bootstrap_interval_uses_period_differences():
    candidate = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.array([0.5, 1.5, 2.5, 3.5])
    low, high = paired_mean_bootstrap_interval(candidate, baseline, samples=100, seed=7)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_hit_distribution_includes_unobserved_legal_bins():
    records = pd.DataFrame({"hits1": [0, 1], "hits2": [1, 0]})
    distribution = hit_distribution(records, max_hits=3)
    assert list(distribution["hit_count"]) == [0, 1, 2, 3]
    assert list(distribution["ticket_count"]) == [2, 2, 0, 0]
    assert distribution["probability"].sum() == pytest.approx(1.0)


def test_scientific_modules_do_not_import_torch():
    assert "torch" not in sys.modules
    history = _draws(20)
    scores = score_numbers(history, "historical_frequency", StrategyParameters())
    assert scores.shape == (80,)


def test_report_metadata_removes_absolute_local_path(tmp_path):
    data_path = tmp_path / "data.csv"
    (tmp_path / "download_meta.json").write_text(
        '{"saved_path":"X:/outside/data.csv","csv_file":"nested/data.csv"}',
        encoding="utf-8",
    )
    metadata = _load_metadata(data_path)
    assert metadata["saved_path"] == "data.csv"
    assert metadata["csv_file"] == "data.csv"


def test_backtest_output_path_rejects_outside_scoped_directory():
    with pytest.raises(SystemExit, match="结果目录必须位于"):
        _resolve_scoped_path("../outside", "results", "结果目录")
