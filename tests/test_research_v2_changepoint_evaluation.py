"""第二阶段嵌套时间边界、状态与 v1 不变性测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.research_v2.changepoint import changepoint_beta_bernoulli
from src.research_v2.changepoint_evaluation import (
    CHANGEPOINT_STRATEGY,
    ChangepointEvaluationConfig,
    evaluate_changepoint_walk_forward,
)
from src.research_v2.evaluation import (
    NestedEvaluationConfig,
    evaluate_nested_walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
V1_SOURCE_HASHES = {
    "config/config.yaml": "03bef7cf395afe7cd8e62adf9e2eb06f2b25b7c27ce7f0e1ccf4154cf0f4f731",
    "scripts/prospective_evaluate.py": "621ebf4b30158b3566a51d6e0ba188bd7ba04cf45d47053c8315c12e8cd11923",
    "src/analysis/feature_enhancer.py": "46fa09871bbf791633ff6cdb078d99d3917e072db598c4319abae5542a9be010",
    "src/config.py": "ecc9a216d5d954a75c36df0bc7a36a77d1be18ae35a33025d2ac87f17686702c",
    "src/scientific/evaluation.py": "d0d8a859ceddb1776df5bd59f578be3e5e576ea0dce1170aaa23d069b2660fd8",
    "src/scientific/prizes.py": "19d9ebe0e31cf599707e4c02ab0deef7cbb2c4fd94fd6c042cc5eac3c021645b",
    "src/scientific/prospective.py": "6f67c35001d056c361f25b29424e9f9e2349c8feab949b4b6b6db559de22aeb3",
    "src/scientific/statistics.py": "21573d7d4ff73ed2be79d4450166d3310736191e638fe8327acfeeba5faf542e",
    "src/scientific/strategies.py": "72c430166233036a403f82be185f0cb9eae4a3f517274eb936ad1d473e043870",
}
MANIFEST_NORMALISED_HASH = (
    "ed26dd0a1aa6841d0fd82a8a21157767a0ee399f0d57922bcfffced27c06df55"
)
FREEZE_CONFIG_NORMALISED_HASH = (
    "2c2e8bf33a7be2c11656b3375a7804b3e22a9f2b58ca5bf7fd39ff7aaa92940d"
)


def _normalised_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rotating_draws(periods: int) -> tuple[np.ndarray, np.ndarray]:
    issues = np.arange(2020001, 2020001 + periods, dtype=np.int64)
    draws = np.empty((periods, 20), dtype=np.int64)
    for period in range(periods):
        draws[period] = ((np.arange(20) * 7 + period * 3) % 80) + 1
    return issues, draws


def _stable_draws(periods: int) -> tuple[np.ndarray, np.ndarray]:
    issues = np.arange(2020001, 2020001 + periods, dtype=np.int64)
    draws = np.empty((periods, 20), dtype=np.int64)
    for period in range(periods):
        start = 20 * (period % 4)
        draws[period] = np.arange(start + 1, start + 21)
    return issues, draws


def _changed_draws(periods: int) -> tuple[np.ndarray, np.ndarray]:
    issues = np.arange(2020001, 2020001 + periods, dtype=np.int64)
    draws = np.empty((periods, 20), dtype=np.int64)
    draws[:440] = np.arange(1, 21, dtype=np.int64)
    draws[440:] = np.arange(61, 81, dtype=np.int64)
    return issues, draws


def test_outer_target_and_future_do_not_change_first_prediction_or_threshold() -> None:
    issues, draws = _rotating_draws(490)
    config = ChangepointEvaluationConfig(include_comparators=False)
    original = evaluate_changepoint_walk_forward(issues, draws, config=config)
    modified = draws.copy()
    first_outer = config.minimum_history + config.minimum_inner_observations
    modified[first_outer:] = ((modified[first_outer:] - 1 + 23) % 80) + 1
    changed = evaluate_changepoint_walk_forward(issues, modified, config=config)

    np.testing.assert_array_equal(
        original.selected_parameter_indices[:1],
        changed.selected_parameter_indices[:1],
    )
    np.testing.assert_array_equal(
        original.candidate_change_scores[:1], changed.candidate_change_scores[:1]
    )
    np.testing.assert_array_equal(
        original.candidate_thresholds[:1], changed.candidate_thresholds[:1]
    )
    np.testing.assert_array_equal(
        original.posterior_mean[:1], changed.posterior_mean[:1]
    )


def test_threshold_history_and_inner_selection_end_before_outer_target() -> None:
    issues, draws = _rotating_draws(490)
    result = evaluate_changepoint_walk_forward(
        issues,
        draws,
        config=ChangepointEvaluationConfig(include_comparators=False),
    )
    first_outer = int(result.outer_indices[0])
    assert np.all(result.threshold_training_first_indices < first_outer)
    assert int(result.threshold_training_last_indices[0]) == first_outer - 1
    assert result.inner_mean_brier.shape == (5, 3)
    assert int(result.selected_parameter_indices[0]) == int(
        np.argmin(result.inner_mean_brier[0])
    )


def test_no_change_history_is_mainly_normal() -> None:
    issues, draws = _stable_draws(550)
    result = evaluate_changepoint_walk_forward(
        issues,
        draws,
        config=ChangepointEvaluationConfig(include_comparators=False),
    )
    assert float(result.high_change.mean()) < 0.25
    assert np.all(result.active_decay[~result.high_change] == 0.995)


def test_clear_frequency_shift_triggers_high_change() -> None:
    issues, draws = _changed_draws(520)
    result = evaluate_changepoint_walk_forward(
        issues,
        draws,
        config=ChangepointEvaluationConfig(include_comparators=False),
    )
    assert bool(result.high_change[:5].all())
    assert np.all(result.active_decay[:5] == 0.95)
    assert np.all(result.effective_history_length[:5] == 120)


def test_probabilities_sum_to_twenty_and_ranking_repeats() -> None:
    issues, draws = _rotating_draws(490)
    config = ChangepointEvaluationConfig(include_comparators=False)
    first = evaluate_changepoint_walk_forward(issues, draws, config=config)
    second = evaluate_changepoint_walk_forward(issues, draws, config=config)
    assert np.all((first.posterior_mean > 0.0) & (first.posterior_mean < 1.0))
    np.testing.assert_allclose(first.posterior_mean.sum(axis=1), 20.0, atol=1e-9)
    np.testing.assert_array_equal(first.posterior_mean, second.posterior_mean)
    np.testing.assert_array_equal(
        first.rankings[CHANGEPOINT_STRATEGY], second.rankings[CHANGEPOINT_STRATEGY]
    )


def test_recursive_grid_matches_direct_target_history_calculation() -> None:
    issues, draws = _rotating_draws(490)
    result = evaluate_changepoint_walk_forward(
        issues,
        draws,
        config=ChangepointEvaluationConfig(include_comparators=False),
    )
    offset = 0
    target_index = int(result.outer_indices[offset])
    parameter_index = int(result.selected_parameter_indices[offset])
    direct = changepoint_beta_bernoulli(
        result.indicators[:target_index],
        change_threshold=float(result.candidate_thresholds[offset, parameter_index]),
        parameters=result.parameters[parameter_index],
    )
    np.testing.assert_allclose(
        direct.posterior_mean, result.posterior_mean[offset], atol=1e-14
    )
    assert direct.high_change == bool(result.high_change[offset])
    assert (
        direct.change_score == result.candidate_change_scores[offset, parameter_index]
    )


def test_some_rankings_differ_from_fixed_exponential_frequency() -> None:
    issues, draws = _rotating_draws(490)
    result = evaluate_changepoint_walk_forward(issues, draws)
    assert bool(result.ranking_differs_from_fixed_exponential.any())


def test_phase1_dynamic_bayesian_behavior_is_unchanged() -> None:
    issues, draws = _rotating_draws(490)
    phase1 = evaluate_nested_walk_forward(
        issues, draws, config=NestedEvaluationConfig(include_comparators=False)
    )
    phase2 = evaluate_changepoint_walk_forward(
        issues,
        draws,
        config=ChangepointEvaluationConfig(include_comparators=False),
    )
    np.testing.assert_array_equal(
        phase1.selected_parameter_indices,
        phase2.phase1_result.selected_parameter_indices,
    )
    np.testing.assert_array_equal(
        phase1.posterior_mean, phase2.phase1_result.posterior_mean
    )
    np.testing.assert_array_equal(
        phase1.rankings["dynamic_bayesian"],
        phase2.phase1_result.rankings["dynamic_bayesian"],
    )


def test_production_time_design_cannot_be_changed() -> None:
    with np.testing.assert_raises(ValueError):
        ChangepointEvaluationConfig(minimum_history=364)
    with np.testing.assert_raises(ValueError):
        ChangepointEvaluationConfig(minimum_inner_observations=119)


def test_v1_sources_freeze_config_and_manifest_are_unchanged() -> None:
    actual = {
        relative: _normalised_hash(ROOT / relative) for relative in V1_SOURCE_HASHES
    }
    assert actual == V1_SOURCE_HASHES
    assert (
        _normalised_hash(ROOT / "config/scientific_freeze.json")
        == FREEZE_CONFIG_NORMALISED_HASH
    )
    assert (
        _normalised_hash(ROOT / "reports/prospective_manifests/2026188.json")
        == MANIFEST_NORMALISED_HASH
    )
