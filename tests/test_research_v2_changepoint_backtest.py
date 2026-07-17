"""第二阶段脚本输出、自包含输入和路径边界测试。"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import research_v2_changepoint_backtest as backtest

ROOT = Path(__file__).resolve().parents[1]
V1_PATHS = (
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
    "scripts/research_v2_backtest.py",
    "reports/kl8_v2_research_report.md",
)


def _data_bytes(periods: int = 486) -> bytes:
    header = ["期数", *(f"红球_{index}" for index in range(1, 21))]
    lines = [",".join(header)]
    for period in range(periods):
        issue = 2020001 + period
        numbers = ((np.arange(20) * 7 + period * 3) % 80) + 1
        lines.append(",".join((str(issue), *(str(int(number)) for number in numbers))))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _prepare_root(root: Path) -> tuple[Path, Path, Path]:
    data_path = root / "data_cache/kl8/data.csv"
    output_dir = root / "results/research_v2_changepoint"
    report_path = root / "reports/kl8_v2_changepoint_report.md"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(_data_bytes())
    return data_path, output_dir, report_path


def _execute(root: Path) -> tuple[Path, Path, Path]:
    data_path, output_dir, report_path = _prepare_root(root)
    backtest.execute_backtest(
        data_path=data_path,
        output_dir=output_dir,
        report_path=report_path,
        project_root=root,
    )
    return data_path, output_dir, report_path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def test_output_csv_fields_and_row_counts(tmp_path: Path) -> None:
    _, output_dir, report_path = _execute(tmp_path)
    expected_counts = {
        "issue_probabilities.csv": 80,
        "issue_metrics.csv": 5,
        "topk_metrics.csv": 90,
        "changepoint_history.csv": 3,
        "state_metrics.csv": 2,
        "calibration.csv": 50,
        "switch_diagnostics.csv": 6,
    }
    for filename, expected in expected_counts.items():
        fields, rows = _read_csv(output_dir / filename)
        assert fields
        assert len(rows) == expected
    interval_fields, _ = _read_csv(output_dir / "trigger_intervals.csv")
    assert interval_fields == [
        "interval_index",
        "start_issue",
        "end_issue",
        "issue_count",
        "evidence_status",
    ]
    probability_fields, _ = _read_csv(output_dir / "issue_probabilities.csv")
    assert {
        "posterior_mean",
        "fixed_normal_posterior_mean",
        "fixed_high_posterior_mean",
        "equals_active_fixed_probability",
        "change_score",
        "change_threshold",
        "high_change",
        "active_decay",
    }.issubset(probability_fields)
    assert report_path.is_file()


def test_same_temporary_input_writes_byte_identical_outputs(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _, first_output, first_report = _execute(first_root)
    _, second_output, second_report = _execute(second_root)
    for filename in backtest.OUTPUT_FILENAMES:
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()


def test_report_values_match_machine_readable_summaries(tmp_path: Path) -> None:
    _, output_dir, report_path = _execute(tmp_path)
    _, issue_rows = _read_csv(output_dir / "issue_metrics.csv")
    changepoint = [
        row for row in issue_rows if row["strategy"] == "changepoint_bayesian"
    ]
    fixed_normal = [
        row for row in issue_rows if row["strategy"] == "fixed_normal_bayesian"
    ]
    fixed_high = [row for row in issue_rows if row["strategy"] == "fixed_high_bayesian"]
    expected_brier = float(np.mean([float(row["brier_score"]) for row in changepoint]))
    expected_vs_normal = expected_brier - float(
        np.mean([float(row["brier_score"]) for row in fixed_normal])
    )
    expected_vs_high = expected_brier - float(
        np.mean([float(row["brier_score"]) for row in fixed_high])
    )
    _, topk_rows = _read_csv(output_dir / "topk_metrics.csv")
    top10 = [
        float(row["actual_hits"])
        for row in topk_rows
        if row["strategy"] == "changepoint_bayesian" and row["k"] == "10"
    ]
    report = report_path.read_text(encoding="utf-8")
    assert f"{expected_brier:.9f}" in report
    assert f"{expected_vs_normal:+.9f}" in report
    assert f"{expected_vs_high:+.9f}" in report
    assert f"{np.mean(top10):.6f}" in report


def test_switch_csv_and_report_summaries_are_consistent(tmp_path: Path) -> None:
    _, output_dir, report_path = _execute(tmp_path)
    _, rows = _read_csv(output_dir / "switch_diagnostics.csv")
    lookup = {(row["comparator"], row["scope"]): row for row in rows}
    normal_all = lookup[("fixed_normal_bayesian", "all")]
    normal_high = lookup[("fixed_normal_bayesian", "high_change")]
    high_all = lookup[("fixed_high_bayesian", "all")]
    high_normal = lookup[("fixed_high_bayesian", "normal")]
    report = report_path.read_text(encoding="utf-8")
    for row in (normal_all, normal_high, high_all, high_normal):
        assert row["ranking_difference_count"] in report
        assert f"{float(row['ranking_difference_proportion']):.2%}" in report
    assert (
        lookup[("fixed_normal_bayesian", "normal")]["probabilities_exactly_equal"]
        == "True"
    )
    assert (
        lookup[("fixed_high_bayesian", "high_change")]["probabilities_exactly_equal"]
        == "True"
    )


def test_all_five_probability_models_and_nine_topk_strategies_are_reported(
    tmp_path: Path,
) -> None:
    _, output_dir, report_path = _execute(tmp_path)
    _, issue_rows = _read_csv(output_dir / "issue_metrics.csv")
    _, topk_rows = _read_csv(output_dir / "topk_metrics.csv")
    assert {row["strategy"] for row in issue_rows} == set(
        backtest.PROBABILITY_STRATEGIES
    )
    assert {row["strategy"] for row in topk_rows} == set(
        backtest.PHASE2_COMPARATOR_STRATEGIES
    )
    report = report_path.read_text(encoding="utf-8")
    for strategy in backtest.PROBABILITY_STRATEGIES:
        assert strategy in report
    for strategy in backtest.PHASE2_COMPARATOR_STRATEGIES:
        assert strategy in report


def test_tmp_execution_never_reads_real_data_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path, output_dir, report_path = _prepare_root(tmp_path)
    original_loader = backtest.load_history_csv
    seen: list[Path] = []

    def guarded_loader(path: Path) -> tuple[np.ndarray, np.ndarray]:
        resolved = path.resolve()
        assert tmp_path.resolve() in resolved.parents
        assert resolved != (ROOT / "data_cache/kl8/data.csv").resolve()
        seen.append(resolved)
        return original_loader(path)

    monkeypatch.setattr(backtest, "load_history_csv", guarded_loader)
    backtest.execute_backtest(
        data_path=data_path,
        output_dir=output_dir,
        report_path=report_path,
        project_root=tmp_path,
    )
    assert seen == [data_path.resolve()]


def test_path_escape_and_output_contract_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="项目目录内"):
        backtest._resolve_within(tmp_path, str(tmp_path.parent / "escape.csv"), "数据")
    data_path, _, report_path = _prepare_root(tmp_path)
    with pytest.raises(ValueError, match="结果目录"):
        backtest.execute_backtest(
            data_path=data_path,
            output_dir=tmp_path / "results/wrong",
            report_path=report_path,
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="报告"):
        backtest.execute_backtest(
            data_path=data_path,
            output_dir=tmp_path / "results/research_v2_changepoint",
            report_path=tmp_path / "reports/wrong.md",
            project_root=tmp_path,
        )


def test_script_execution_preserves_v1_and_manifest_bytes(tmp_path: Path) -> None:
    before = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in V1_PATHS
    }
    _execute(tmp_path)
    after = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in V1_PATHS
    }
    assert after == before


def test_makefile_routes_phase2_through_quality_and_help() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/research_v2_changepoint_backtest.py" in makefile
    assert "tests/test_research_v2_changepoint.py" in makefile
    assert "tests/test_research_v2_changepoint_evaluation.py" in makefile
    assert "tests/test_research_v2_changepoint_backtest.py" in makefile
    assert "research-v2-changepoint-backtest:" in makefile
