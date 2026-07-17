"""v2 第三阶段未来前瞻冻结、评价和防覆盖测试。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.research_v2.prospective_monitor import (
    CHANGEPOINT_STRATEGY,
    CONFIRMATION_ISSUE_COUNT,
    DATA_RELATIVE_PATH,
    DYNAMIC_STRATEGY,
    EVIDENCE_STATUS,
    FIXED_HIGH_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    FREEZE_RELATIVE_PATH,
    MANIFEST_RELATIVE_DIR,
    PROBABILITY_STRATEGIES,
    RESULT_RELATIVE_DIR,
    UNIFORM_STRATEGY,
    build_evaluation_record,
    build_formal_primary_summary,
    build_manifest,
    canonical_json_bytes,
    holm_adjust,
    load_and_verify_freeze_config,
    load_history_csv,
    normalized_lf_sha256,
    predict_frozen_models,
    require_contract_path,
    source_manifest_fingerprint,
    write_evaluation_exclusive,
    write_manifest_exclusive,
)

IntArray = NDArray[np.int64]

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / FREEZE_RELATIVE_PATH
IMMUTABLE_PATHS = (
    "config/config.yaml",
    "scripts/prospective_evaluate.py",
    "src/analysis/feature_enhancer.py",
    "src/config.py",
    "src/scientific/evaluation.py",
    "src/scientific/prizes.py",
    "src/scientific/prospective.py",
    "src/scientific/statistics.py",
    "src/scientific/strategies.py",
    "config/scientific_freeze.json",
    "reports/prospective_manifests/2026188.json",
    "src/research_v2/bayesian.py",
    "src/research_v2/evaluation.py",
    "src/research_v2/metrics.py",
    "src/research_v2/changepoint.py",
    "src/research_v2/changepoint_evaluation.py",
    "scripts/research_v2_backtest.py",
    "scripts/research_v2_changepoint_backtest.py",
    "reports/kl8_v2_research_report.md",
    "reports/kl8_v2_changepoint_report.md",
)


def _draw_for_period(period: int) -> tuple[int, ...]:
    return tuple(int(value) for value in ((np.arange(20) * 7 + period * 3) % 80) + 1)


def _write_data(path: Path, periods: int = 486) -> tuple[IntArray, IntArray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["期数", *(f"红球_{index}" for index in range(1, 21))]
    start_issue = 2020001
    rows: list[dict[str, int]] = []
    for period in range(periods):
        row = {"期数": start_issue + period}
        row.update(
            {
                f"红球_{index}": number
                for index, number in enumerate(_draw_for_period(period), start=1)
            }
        )
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return load_history_csv(path)


def _prepare_project(
    tmp_path: Path,
) -> tuple[Path, Path, Path, IntArray, IntArray, int]:
    root = tmp_path / "project"
    data_path = root / DATA_RELATIVE_PATH
    issues, draws = _write_data(data_path)
    source_path = root / "src/frozen_for_test.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("FROZEN = True\n", encoding="utf-8")

    parsed: object = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    config = cast(dict[str, Any], copy.deepcopy(parsed))
    files = {"src/frozen_for_test.py": normalized_lf_sha256(source_path)}
    config["source_manifest"] = {
        "algorithm": "sha256_utf8_normalized_lf",
        "files": files,
        "fingerprint": source_manifest_fingerprint(files),
        "change_policy": "test fixture",
    }
    config_path = root / FREEZE_RELATIVE_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(canonical_json_bytes(config))
    target_issue = int(issues[-1]) + 1
    return root, data_path, config_path, issues, draws, target_issue


def _build_test_manifest(
    root: Path,
    data_path: Path,
    config_path: Path,
    issues: IntArray,
    draws: IntArray,
    target_issue: int,
) -> dict[str, object]:
    return build_manifest(
        project_root=root,
        data_path=data_path,
        config_path=config_path,
        issues=issues,
        draws=draws,
        target_issue=target_issue,
        generated_at_utc="2026-07-01T10:00:00Z",
        git_commit_sha="a" * 40,
        official_source_url="https://example.gov.cn/kl8/next-issue",
        official_confirmed_at_utc="2026-07-01T09:55:00Z",
    )


def _write_test_manifest(
    root: Path,
    data_path: Path,
    config_path: Path,
    issues: IntArray,
    draws: IntArray,
    target_issue: int,
) -> tuple[dict[str, object], Path]:
    manifest = _build_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    manifest_dir = root / MANIFEST_RELATIVE_DIR
    return manifest, write_manifest_exclusive(manifest, manifest_dir)


def test_five_models_are_legal_and_rankings_are_reproducible(tmp_path: Path) -> None:
    root, _, _, issues, draws, target_issue = _prepare_project(tmp_path)
    first = predict_frozen_models(issues, draws, target_issue=target_issue)
    second = predict_frozen_models(issues, draws, target_issue=target_issue)
    assert root.is_dir()
    assert set(first.probabilities) == set(PROBABILITY_STRATEGIES)
    for strategy in PROBABILITY_STRATEGIES:
        probabilities = first.probabilities[strategy]
        assert probabilities.shape == (80,)
        assert np.all((probabilities > 0.0) & (probabilities < 1.0))
        assert float(probabilities.sum()) == pytest.approx(20.0, abs=1e-9)
        assert np.array_equal(first.rankings[strategy], second.rankings[strategy])
        assert set(map(int, first.rankings[strategy])) == set(range(1, 81))


def test_target_result_and_future_data_never_change_current_manifest(
    tmp_path: Path,
) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path / "first"
    )
    base = _build_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    extended_issues = np.concatenate(
        (issues, np.asarray([target_issue, target_issue + 1], dtype=np.int64))
    )
    extended_draws_a = np.vstack((draws, np.arange(1, 21), np.arange(21, 41)))
    extended_draws_b = np.vstack((draws, np.arange(61, 81), np.arange(41, 61)))
    with_target_a = _build_test_manifest(
        root,
        data_path,
        config_path,
        extended_issues,
        extended_draws_a,
        target_issue,
    )
    with_target_b = _build_test_manifest(
        root,
        data_path,
        config_path,
        extended_issues,
        extended_draws_b,
        target_issue,
    )
    assert canonical_json_bytes(base) == canonical_json_bytes(with_target_a)
    assert canonical_json_bytes(base) == canonical_json_bytes(with_target_b)


def test_same_input_generates_byte_identical_manifest(tmp_path: Path) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    first = _build_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    (
        second_root,
        second_data,
        second_config,
        second_issues,
        second_draws,
        second_target,
    ) = _prepare_project(tmp_path / "second")
    second = _build_test_manifest(
        second_root,
        second_data,
        second_config,
        second_issues,
        second_draws,
        second_target,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["history_issue_count"] == 486
    assert first["evidence_status"] == EVIDENCE_STATUS
    models = cast(dict[str, Any], first["models"])
    for model in models.values():
        assert len(model["probabilities"]) == 80
        assert len(model["ranking"]) == 80
        assert set(model["top_k_candidates"]) == {str(k) for k in range(1, 11)}


def test_existing_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    manifest = _build_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    path = write_manifest_exclusive(manifest, root / MANIFEST_RELATIVE_DIR)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        write_manifest_exclusive(manifest, root / MANIFEST_RELATIVE_DIR)
    assert path.read_bytes() == before


def test_evaluation_uses_manifest_and_cannot_be_written_twice(tmp_path: Path) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    _, manifest_path = _write_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    actual = np.arange(1, 21, dtype=np.int64)
    record = build_evaluation_record(
        project_root=root,
        config_path=config_path,
        manifest_path=manifest_path,
        actual_numbers=actual,
        official_result_published_at_utc="2026-07-02T10:00:00Z",
        evaluated_at_utc="2026-07-02T10:05:00Z",
    )
    result_path = write_evaluation_exclusive(record, root / RESULT_RELATIVE_DIR)
    before = result_path.read_bytes()
    with pytest.raises(FileExistsError, match="拒绝重复写入"):
        write_evaluation_exclusive(record, root / RESULT_RELATIVE_DIR)
    assert result_path.read_bytes() == before
    assert record["actual_numbers"] == list(range(1, 21))
    manifest_info = cast(dict[str, Any], record["manifest"])
    assert manifest_info["sealed_before_official_result"] is True
    model_metrics = cast(dict[str, Any], record["models"])
    assert set(model_metrics) == set(PROBABILITY_STRATEGIES)
    for metrics in model_metrics.values():
        assert set(metrics["top_k"]) == {str(k) for k in range(1, 11)}


def test_manifest_must_predate_official_result_and_evaluation_follows_it(
    tmp_path: Path,
) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    _, manifest_path = _write_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    actual = np.arange(1, 21, dtype=np.int64)
    with pytest.raises(ValueError, match="不得早于官方结果"):
        build_evaluation_record(
            project_root=root,
            config_path=config_path,
            manifest_path=manifest_path,
            actual_numbers=actual,
            official_result_published_at_utc="2026-07-02T10:00:00Z",
            evaluated_at_utc="2026-07-02T09:59:59Z",
        )
    late_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    late_manifest["generated_at_utc"] = "2026-07-02T10:00:00Z"
    late_path = root / MANIFEST_RELATIVE_DIR / "late.json"
    late_path.write_bytes(canonical_json_bytes(late_manifest))
    with pytest.raises(ValueError, match="未在官方结果发布前"):
        build_evaluation_record(
            project_root=root,
            config_path=config_path,
            manifest_path=late_path,
            actual_numbers=actual,
            official_result_published_at_utc="2026-07-02T10:00:00Z",
            evaluated_at_utc="2026-07-02T10:01:00Z",
        )


def test_source_drift_refuses_manifest_and_evaluation(tmp_path: Path) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    _, manifest_path = _write_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    (root / "src/frozen_for_test.py").write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="冻结源码漂移"):
        _build_test_manifest(root, data_path, config_path, issues, draws, target_issue)
    with pytest.raises(ValueError, match="冻结源码漂移"):
        build_evaluation_record(
            project_root=root,
            config_path=config_path,
            manifest_path=manifest_path,
            actual_numbers=np.arange(1, 21, dtype=np.int64),
            official_result_published_at_utc="2026-07-02T10:00:00Z",
            evaluated_at_utc="2026-07-02T10:05:00Z",
        )


def test_paths_are_confined_to_fixed_project_locations(tmp_path: Path) -> None:
    root, _, _, _, _, _ = _prepare_project(tmp_path)
    assert (
        require_contract_path(
            root, MANIFEST_RELATIVE_DIR, MANIFEST_RELATIVE_DIR, "manifest目录"
        )
        == (root / MANIFEST_RELATIVE_DIR).resolve()
    )
    with pytest.raises(ValueError, match="必须固定"):
        require_contract_path(
            root,
            Path("reports/wrong"),
            MANIFEST_RELATIVE_DIR,
            "manifest目录",
        )
    with pytest.raises(ValueError, match="项目目录内"):
        require_contract_path(
            root,
            root.parent / "escape",
            RESULT_RELATIVE_DIR,
            "结果目录",
        )


def test_unit_workflow_never_uses_real_data_cache(tmp_path: Path) -> None:
    real_data = ROOT / DATA_RELATIVE_PATH
    before = (
        hashlib.sha256(real_data.read_bytes()).hexdigest()
        if real_data.exists()
        else None
    )
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    assert tmp_path.resolve() in data_path.resolve().parents
    _write_test_manifest(root, data_path, config_path, issues, draws, target_issue)
    after = (
        hashlib.sha256(real_data.read_bytes()).hexdigest()
        if real_data.exists()
        else None
    )
    assert after == before


def test_freeze_contract_rejects_model_parameter_changes(tmp_path: Path) -> None:
    root, _, config_path, _, _, _ = _prepare_project(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"][FIXED_NORMAL_STRATEGY]["decay"] = 0.99
    config_path.write_bytes(canonical_json_bytes(config))
    with pytest.raises(ValueError, match="fixed_normal_bayesian冻结参数漂移"):
        load_and_verify_freeze_config(root, config_path)


def test_freeze_contract_rejects_early_stopping_or_comparison_changes(
    tmp_path: Path,
) -> None:
    root, _, config_path, _, _, _ = _prepare_project(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["confirmation_protocol"]["early_stopping_forbidden"] = False
    config_path.write_bytes(canonical_json_bytes(config))
    with pytest.raises(ValueError, match="禁止提前停止"):
        load_and_verify_freeze_config(root, config_path)


def test_formal_summary_is_blocked_before_365_unique_presealed_issues(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="恰好365"):
        build_formal_primary_summary(results_dir)
    record = {
        "evidence_status": EVIDENCE_STATUS,
        "target_issue": 1,
        "manifest": {
            "sealed_before_official_result": True,
            "sha256": "b" * 64,
        },
        "comparisons": {
            "versus_uniform": {
                strategy: {"brier_difference_model_minus_uniform": -0.001}
                for strategy in (
                    DYNAMIC_STRATEGY,
                    FIXED_NORMAL_STRATEGY,
                    FIXED_HIGH_STRATEGY,
                    CHANGEPOINT_STRATEGY,
                )
            }
        },
    }
    (results_dir / "1.json").write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="恰好365"):
        build_formal_primary_summary(results_dir)


def test_formal_summary_uses_all_four_comparisons_and_holm(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    comparison_models = (
        DYNAMIC_STRATEGY,
        FIXED_NORMAL_STRATEGY,
        FIXED_HIGH_STRATEGY,
        CHANGEPOINT_STRATEGY,
    )
    for offset in range(CONFIRMATION_ISSUE_COUNT):
        versus_uniform = {
            strategy: {
                "brier_difference_model_minus_uniform": (
                    -0.001 - model_index * 0.0001 + (offset % 7) * 0.000001
                )
            }
            for model_index, strategy in enumerate(comparison_models)
        }
        record = {
            "evidence_status": EVIDENCE_STATUS,
            "target_issue": 2027001 + offset,
            "manifest": {
                "sealed_before_official_result": True,
                "sha256": "b" * 64,
            },
            "comparisons": {"versus_uniform": versus_uniform},
        }
        (results_dir / f"{2027001 + offset}.json").write_bytes(
            canonical_json_bytes(record)
        )
    summary = build_formal_primary_summary(results_dir)
    assert summary["issue_count"] == 365
    comparisons = cast(dict[str, Any], summary["comparisons"])
    assert set(comparisons) == set(comparison_models)
    assert all(
        0.0 <= row["holm_adjusted_p_value"] <= 1.0 for row in comparisons.values()
    )


def test_holm_requires_exact_preregistered_comparison_family() -> None:
    with pytest.raises(ValueError, match="全部四项"):
        holm_adjust({DYNAMIC_STRATEGY: 0.01})
    adjusted = holm_adjust(
        {
            DYNAMIC_STRATEGY: 0.01,
            FIXED_NORMAL_STRATEGY: 0.02,
            FIXED_HIGH_STRATEGY: 0.03,
            CHANGEPOINT_STRATEGY: 0.04,
        }
    )
    assert adjusted[DYNAMIC_STRATEGY] == pytest.approx(0.04)
    assert adjusted[FIXED_NORMAL_STRATEGY] == pytest.approx(0.06)
    assert adjusted[FIXED_HIGH_STRATEGY] == pytest.approx(0.06)
    assert adjusted[CHANGEPOINT_STRATEGY] == pytest.approx(0.06)


def test_v1_existing_manifests_and_phase1_phase2_remain_byte_identical(
    tmp_path: Path,
) -> None:
    before = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in IMMUTABLE_PATHS
    }
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    _write_test_manifest(root, data_path, config_path, issues, draws, target_issue)
    after = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in IMMUTABLE_PATHS
    }
    assert after == before


def test_manifest_records_dynamic_and_changepoint_target_prior_selection(
    tmp_path: Path,
) -> None:
    root, data_path, config_path, issues, draws, target_issue = _prepare_project(
        tmp_path
    )
    manifest = _build_test_manifest(
        root, data_path, config_path, issues, draws, target_issue
    )
    selected = cast(dict[str, Any], manifest["selected_target_prior_parameters"])
    assert selected[DYNAMIC_STRATEGY]["decay"] in (0.97, 0.99, 0.995)
    assert selected[DYNAMIC_STRATEGY]["prior_strength"] in (5.0, 20.0, 80.0)
    assert selected[CHANGEPOINT_STRATEGY]["recent_window"] in (30, 60, 120)
    assert selected[CHANGEPOINT_STRATEGY]["state"] in ("normal", "high_change")
    models = cast(dict[str, Any], manifest["models"])
    assert models[UNIFORM_STRATEGY]["probabilities"] == [0.25] * 80
