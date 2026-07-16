from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.scientific.evaluation as evaluation
from scripts.prospective_evaluate import build_parser
from src.scientific.prospective import (
    EXPECTED_RANDOM_SEEDS,
    ProspectivePaths,
    canonical_history_sha256,
    evaluate_new_issues,
    load_scientific_freeze,
    resolve_prospective_paths,
    run_prospective_evaluation,
)
from src.scientific.strategies import assert_legal_tickets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMBER_COLUMNS = [f"红球_{index}" for index in range(1, 21)]


def _draws(rows: int, *, offset: int = 0) -> np.ndarray:
    return np.asarray(
        [
            ((np.arange(20, dtype=int) + (row + offset) * 7) % 80) + 1
            for row in range(rows)
        ],
        dtype=int,
    )


def _normalised_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_history(path: Path, issues: np.ndarray, draws: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(draws, columns=NUMBER_COLUMNS)
    frame.insert(0, "期数", issues)
    frame.sort_values("期数", ascending=False).to_csv(
        path, index=False, encoding="utf-8"
    )


def _fixture_issues(post_freeze: int = 1, history_rows: int = 50) -> np.ndarray:
    frozen = np.arange(2026186 - history_rows + 1, 2026187, dtype=int)
    future = np.arange(2026187, 2026187 + post_freeze, dtype=int)
    return np.concatenate([frozen, future])


def _prepare_project(
    tmp_path: Path, *, post_freeze: int = 1, history_rows: int = 50
) -> tuple[ProspectivePaths, np.ndarray, np.ndarray]:
    root = tmp_path / "repo"
    issues = _fixture_issues(post_freeze=post_freeze, history_rows=history_rows)
    draws = _draws(len(issues))
    data_path = root / "data_cache" / "kl8" / "data.csv"
    _write_history(data_path, issues, draws)

    report = root / "reports" / "kl8_scientific_report.md"
    final_holdout = root / "results" / "scientific" / "final_holdout_results.csv"
    selected = root / "results" / "scientific" / "selected_parameters.json"
    for path, text in (
        (report, "frozen scientific report\n"),
        (final_holdout, "frozen final holdout\n"),
        (selected, '{"frozen": true}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    payload = json.loads(
        (PROJECT_ROOT / "config" / "scientific_freeze.json").read_text(encoding="utf-8")
    )
    frozen_mask = issues <= 2026186
    snapshot = payload["original_data_snapshot"]
    snapshot.update(
        {
            "earliest_issue": int(issues[frozen_mask][0]),
            "latest_issue": 2026186,
            "record_count": int(frozen_mask.sum()),
            "canonical_records_sha256": canonical_history_sha256(
                issues[frozen_mask], draws[frozen_mask]
            ),
        }
    )
    artifacts = payload["frozen_artifacts"]
    artifacts["scientific_report"]["sha256_utf8_normalized_lf"] = _normalised_sha256(
        report
    )
    artifacts["final_holdout_results"]["sha256_utf8_normalized_lf"] = (
        _normalised_sha256(final_holdout)
    )
    artifacts["selected_parameters"]["sha256_utf8_normalized_lf"] = _normalised_sha256(
        selected
    )
    freeze_path = root / "config" / "scientific_freeze.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    paths = resolve_prospective_paths(root, temp_root=tmp_path / ".tmp")
    return paths, issues, draws


def _read_records(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype={"seed": str})


def test_target_prediction_only_receives_prior_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, issues, draws = _prepare_project(tmp_path, post_freeze=2)
    freeze = load_scientific_freeze(paths.freeze_config)
    observed_histories: list[np.ndarray] = []
    original = evaluation.score_numbers

    def spy_score_numbers(
        history: np.ndarray, strategy: str, parameters: Any
    ) -> np.ndarray:
        observed_histories.append(history.copy())
        return original(history, strategy, parameters)

    monkeypatch.setattr(evaluation, "score_numbers", spy_score_numbers)
    baseline = evaluate_new_issues(issues, draws, freeze)
    assert [len(history) for history in observed_histories] == [50] * 5 + [51] * 5
    assert all(
        np.array_equal(history[-1], draws[49]) for history in observed_histories[:5]
    )
    assert all(
        np.array_equal(history[-1], draws[50]) for history in observed_histories[5:]
    )

    changed = draws.copy()
    changed[50] = _draws(1, offset=31)[0]
    changed[51] = _draws(1, offset=43)[0]
    modified = evaluate_new_issues(issues, changed, freeze)
    ticket_columns = [
        "strategy",
        "seed",
        "play",
        "ticket_mode",
        "ticket1",
        "ticket2",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["issue"] == 2026187, ticket_columns].reset_index(
            drop=True
        ),
        modified.loc[modified["issue"] == 2026187, ticket_columns].reset_index(
            drop=True
        ),
    )


def test_freeze_config_is_exact_and_not_cli_tunable() -> None:
    freeze = load_scientific_freeze(PROJECT_ROOT / "config" / "scientific_freeze.json")
    assert freeze.frozen_through_issue == 2026186
    assert freeze.rolling_window == 120
    assert freeze.decay == 0.99
    assert freeze.hybrid_weights == (0.25, 0.50, 0.25)
    assert freeze.random_seeds == EXPECTED_RANDOM_SEEDS
    with pytest.raises(FrozenInstanceError):
        freeze.decay = 0.97  # type: ignore[misc]
    destinations = {action.dest for action in build_parser()._actions}
    assert (
        not {"rolling_window", "decay", "hybrid_weights", "random_seeds"} & destinations
    )


def test_tampered_freeze_parameter_fails(tmp_path: Path) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    payload = json.loads(paths.freeze_config.read_text(encoding="utf-8"))
    payload["rolling_window"] = 60
    paths.freeze_config.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rolling_window"):
        load_scientific_freeze(paths.freeze_config)


def test_tampered_freeze_environment_fails(tmp_path: Path) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    payload = json.loads(paths.freeze_config.read_text(encoding="utf-8"))
    payload["repository_advanced"]["environment_observed"]["scikit_learn"] = "0.0"
    paths.freeze_config.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="环境版本"):
        load_scientific_freeze(paths.freeze_config)


def test_prospective_path_never_uses_split_tuning_or_final_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    for relative in (
        "src/scientific/prospective.py",
        "scripts/prospective_evaluate.py",
    ):
        tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "make_temporal_split" not in called_names
        assert "tune_on_validation" not in called_names

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("不得重新切分或调参")

    monkeypatch.setattr(evaluation, "make_temporal_split", forbidden)
    monkeypatch.setattr(evaluation, "tune_on_validation", forbidden)
    run_prospective_evaluation(paths, bootstrap_samples=10)


def test_repeat_run_is_idempotent_and_single_issue_is_descriptive(
    tmp_path: Path,
) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    first = run_prospective_evaluation(paths, bootstrap_samples=10)
    first_bytes = paths.records.read_bytes()
    second = run_prospective_evaluation(paths, bootstrap_samples=10)
    assert first == second
    assert paths.records.read_bytes() == first_bytes
    records = _read_records(paths.records)
    summary = pd.read_csv(paths.summary)
    assert len(records) == 500
    assert not records.duplicated(
        ["issue", "strategy", "seed", "play", "ticket_mode"]
    ).any()
    assert len(summary) == 120
    assert set(summary["inference_status"]) == {"not_run_single_issue"}
    assert set(summary["interval_method"]) == {"not_computed_single_issue"}
    interval_columns = [
        column for column in summary if column.endswith(("_ci95_low", "_ci95_high"))
    ]
    assert summary[interval_columns].isna().all().all()
    report = paths.report.read_text(encoding="utf-8")
    assert "目前只有1个新增期，不能得出统计显著性结论" in report
    assert "既不能证明任何策略优于随机，也不能证明策略与随机等效" in report


def test_existing_record_conflict_fails_before_other_outputs_change(
    tmp_path: Path,
) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    run_prospective_evaluation(paths, bootstrap_samples=10)
    records = _read_records(paths.records)
    records.loc[0, "total_prize"] = float(records.loc[0, "total_prize"]) + 1.0
    records.to_csv(paths.records, index=False, encoding="utf-8")
    corrupted_records = paths.records.read_bytes()
    protected = {
        path: path.read_bytes()
        for path in (paths.summary, paths.candidates, paths.report)
    }
    with pytest.raises(ValueError, match="冲突"):
        run_prospective_evaluation(paths, bootstrap_samples=10)
    assert paths.records.read_bytes() == corrupted_records
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_adding_one_draw_only_adds_that_issue(tmp_path: Path) -> None:
    paths, issues, draws = _prepare_project(tmp_path)
    run_prospective_evaluation(paths, bootstrap_samples=10)
    old_records = _read_records(paths.records)

    new_issues = np.append(issues, 2026188)
    new_draws = np.vstack([draws, _draws(1, offset=71)])
    _write_history(paths.data, new_issues, new_draws)
    result = run_prospective_evaluation(paths, bootstrap_samples=10, next_issue=2026189)
    updated = _read_records(paths.records)
    assert result.prospective_issues == (2026187, 2026188)
    assert len(updated) == 1000
    assert len(updated[updated["issue"] == 2026188]) == 500
    assert "目前共有 `2` 个真正的前瞻时间簇" in paths.report.read_text(encoding="utf-8")
    pd.testing.assert_frame_equal(
        old_records.sort_values(list(old_records.columns)).reset_index(drop=True),
        updated[updated["issue"] == 2026187]
        .sort_values(list(old_records.columns))
        .reset_index(drop=True),
        check_dtype=False,
    )


def test_frozen_scientific_outputs_are_never_touched(tmp_path: Path) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    protected = {
        path: path.read_bytes()
        for path in (
            paths.frozen_report,
            paths.frozen_final_holdout,
            paths.frozen_selected_parameters,
        )
    }
    run_prospective_evaluation(paths, bootstrap_samples=10)
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("data", "../data.csv"),
        ("data", r"C:\lottery\data.csv"),
        ("output_dir", "results/prospective-evil"),
        ("output_dir", "results/scientific"),
        ("report", "reports/kl8_scientific_report.md"),
        ("report", r"D:\lottery\outside.md"),
    ],
)
def test_output_paths_cannot_escape_or_overwrite_frozen_files(
    tmp_path: Path, keyword: str, value: str
) -> None:
    root = tmp_path / "repo"
    arguments: dict[str, str] = {keyword: value}
    with pytest.raises(ValueError):
        resolve_prospective_paths(root, temp_root=tmp_path / ".tmp", **arguments)


def test_temporary_path_is_fixed_to_repository_parent_tmp(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    accepted = resolve_prospective_paths(root, temp_root=tmp_path / ".tmp")
    assert accepted.temp_root == (tmp_path / ".tmp").resolve()
    with pytest.raises(ValueError, match="临时目录固定"):
        resolve_prospective_paths(root, temp_root=tmp_path / "other-temp")


def test_next_issue_candidates_are_complete_and_legal(tmp_path: Path) -> None:
    paths, _, _ = _prepare_project(tmp_path)
    result = run_prospective_evaluation(paths, bootstrap_samples=10)
    candidates = pd.read_csv(paths.candidates, keep_default_na=False)
    assert result.next_issue == 2026188
    assert len(candidates) == 100
    assert set(candidates["target_issue"]) == {2026188}
    assert set(candidates["history_through_issue"]) == {2026187}
    assert "uniform_random" not in set(candidates["strategy"])
    assert (
        candidates[["strategy", "play", "ticket_mode"]].drop_duplicates().shape[0]
        == 100
    )
    for row in candidates.itertuples(index=False):
        tickets = (
            tuple(int(value) for value in row.ticket1.split()),
            tuple(int(value) for value in row.ticket2.split()),
        )
        assert_legal_tickets(tickets, int(row.play), str(row.ticket_mode))
        assert int(row.overlap_count) == len(set(tickets[0]) & set(tickets[1]))
    assert set(candidates["output_status"]) == {
        "frozen_strategy_output_not_prediction_or_validity_evidence"
    }
