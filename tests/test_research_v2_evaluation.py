"""嵌套时间验证与 v1 不变边界测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.research_v2.bayesian import DECAY_GRID, PRIOR_STRENGTH_GRID
from src.research_v2.evaluation import (
    NestedEvaluationConfig,
    evaluate_nested_walk_forward,
    precompute_posterior_grid,
    preregistered_parameter_grid,
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
MANIFEST_ALLOWED_RAW_HASHES = {
    MANIFEST_NORMALISED_HASH,
    "8d98fff7219fcc5fcdbdcd60ef91468af61a99eefbaf92c6f20490221ffad4d6",
}


def _normalised_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_history(periods: int = 490) -> tuple[np.ndarray, np.ndarray]:
    issues = np.arange(2020001, 2020001 + periods, dtype=np.int64)
    draws = np.empty((periods, 20), dtype=np.int64)
    base = np.arange(20, dtype=np.int64)
    for index in range(periods):
        draws[index] = 1 + (base + 7 * index) % 80
    return issues, draws


def test_parameter_grid_is_exactly_the_preregistered_nine_combinations() -> None:
    grid = preregistered_parameter_grid()

    assert len(grid) == 9
    assert [(row.decay, row.prior_strength) for row in grid] == [
        (decay, prior) for decay in DECAY_GRID for prior in PRIOR_STRENGTH_GRID
    ]


def test_outer_target_and_future_draws_never_enter_target_prediction() -> None:
    issues, draws = _synthetic_history()
    config = NestedEvaluationConfig(include_comparators=False)
    original = evaluate_nested_walk_forward(issues, draws, config=config)

    modified = draws.copy()
    first_outer = config.minimum_history + config.minimum_inner_observations
    base = np.arange(20, dtype=np.int64)
    for index in range(first_outer, len(modified)):
        modified[index] = 1 + (base + 7 * index + 31) % 80
    changed = evaluate_nested_walk_forward(issues, modified, config=config)

    assert original.outer_indices[0] == first_outer
    assert np.array_equal(
        original.selected_parameter_indices[:1],
        changed.selected_parameter_indices[:1],
    )
    assert np.array_equal(original.posterior_mean[:1], changed.posterior_mean[:1])


def test_inner_selection_uses_only_earlier_temporal_targets() -> None:
    issues, draws = _synthetic_history()
    config = NestedEvaluationConfig(include_comparators=False)
    result = evaluate_nested_walk_forward(issues, draws, config=config)
    grid = precompute_posterior_grid(result.indicators)
    first_outer = int(result.outer_indices[0])
    expected_scores = grid.brier_score[:, config.minimum_history : first_outer].mean(
        axis=1
    )

    assert first_outer - config.minimum_history == 120
    assert np.allclose(result.inner_mean_brier[0], expected_scores)
    assert int(result.selected_parameter_indices[0]) == int(np.argmin(expected_scores))


def test_repeated_nested_runs_are_identical() -> None:
    issues, draws = _synthetic_history()
    config = NestedEvaluationConfig(include_comparators=False)

    first = evaluate_nested_walk_forward(issues, draws, config=config)
    second = evaluate_nested_walk_forward(issues, draws, config=config)

    assert np.array_equal(
        first.selected_parameter_indices, second.selected_parameter_indices
    )
    assert np.array_equal(first.posterior_mean, second.posterior_mean)
    assert np.array_equal(first.credible_interval_lower, second.credible_interval_lower)
    assert np.array_equal(
        first.rankings["dynamic_bayesian"], second.rankings["dynamic_bayesian"]
    )


def test_deterministic_repository_advanced_adapter_repeats_identically() -> None:
    issues, draws = _synthetic_history(periods=486)

    first = evaluate_nested_walk_forward(issues, draws)
    second = evaluate_nested_walk_forward(issues, draws)

    assert np.array_equal(
        first.rankings["repository_advanced"],
        second.rankings["repository_advanced"],
    )


def test_production_minimum_history_cannot_be_reduced_below_365() -> None:
    try:
        NestedEvaluationConfig(minimum_history=364)
    except ValueError as error:
        assert "365" in str(error)
    else:
        raise AssertionError("minimum_history=364应被拒绝")


def test_v1_nine_file_source_manifest_hashes_are_unchanged() -> None:
    actual = {
        relative: _normalised_hash(ROOT / relative) for relative in V1_SOURCE_HASHES
    }

    assert actual == V1_SOURCE_HASHES


def test_existing_2026188_manifest_bytes_are_unchanged() -> None:
    path = ROOT / "reports/prospective_manifests/2026188.json"
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    assert raw_hash in MANIFEST_ALLOWED_RAW_HASHES
    assert _normalised_hash(path) == MANIFEST_NORMALISED_HASH
