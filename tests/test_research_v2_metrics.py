"""快乐8 v2 概率与校准指标测试。"""

from __future__ import annotations

import numpy as np

from src.research_v2.metrics import (
    bernoulli_log_loss,
    brier_score,
    calibration_summary,
    top_k_hits,
)


def test_brier_score_matches_manual_mean_squared_error() -> None:
    probabilities = np.array([0.25, 0.75], dtype=np.float64)
    outcomes = np.array([0.0, 1.0], dtype=np.float64)

    assert np.isclose(brier_score(probabilities, outcomes), 0.0625)


def test_log_loss_is_stable_for_extreme_probabilities() -> None:
    probabilities = np.array([0.0, 1.0, 1e-300, 1.0], dtype=np.float64)
    outcomes = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

    value = bernoulli_log_loss(probabilities, outcomes)

    assert np.isfinite(value)
    assert value > 0.0


def test_calibration_returns_all_fixed_bins_including_empty_bins() -> None:
    probabilities = np.array([0.05, 0.15, 0.25, 0.95], dtype=np.float64)
    outcomes = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    result = calibration_summary(probabilities, outcomes, bins=10)

    assert len(result.bins) == 10
    assert sum(row.count for row in result.bins) == 4
    assert result.bins[3].count == 0
    assert result.bins[3].mean_predicted_probability is None
    assert 0.0 <= result.expected_calibration_error <= 1.0


def test_perfect_calibration_has_zero_ece() -> None:
    probabilities = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    outcomes = probabilities.copy()

    result = calibration_summary(probabilities, outcomes, bins=10)

    assert np.isclose(result.expected_calibration_error, 0.0)


def test_top_k_reports_actual_expected_and_excess_hits() -> None:
    ranking = np.arange(1, 81, dtype=np.int64)
    outcomes = np.zeros(80, dtype=np.float64)
    outcomes[[0, 2, 20, 40]] = 1.0

    hits, expected, excess = top_k_hits(ranking, outcomes, 4)

    assert hits == 2
    assert expected == 1.0
    assert excess == 1.0
