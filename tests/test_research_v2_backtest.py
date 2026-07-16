"""v2回测脚本的输出契约、可复现性与路径边界测试。"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
import pytest

from scripts import research_v2_backtest as backtest
from src.research_v2.evaluation import (
    COMPARATOR_STRATEGIES,
    NestedEvaluationConfig,
    NestedEvaluationResult,
    evaluate_nested_walk_forward,
)
from src.research_v2.metrics import CalibrationResult

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_BYTES = b"issue,number_1,number_2\n" b"2020001,1,2\n" b"2020002,3,4\n"
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


@dataclass(frozen=True)
class ScriptFixture:
    result: NestedEvaluationResult
    config: NestedEvaluationConfig
    probability_metrics: backtest.ProbabilityMetrics
    topk_hits: dict[str, np.ndarray]
    calibrations: dict[str, CalibrationResult]


def _normalised_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_history(periods: int = 486) -> tuple[np.ndarray, np.ndarray]:
    issues = np.arange(2020001, 2020001 + periods, dtype=np.int64)
    draws = np.empty((periods, 20), dtype=np.int64)
    base = np.arange(20, dtype=np.int64)
    for index in range(periods):
        draws[index] = 1 + (base + 7 * index) % 80
    return issues, draws


@pytest.fixture(scope="module")
def script_fixture() -> ScriptFixture:
    issues, draws = _synthetic_history()
    config = NestedEvaluationConfig()
    result = evaluate_nested_walk_forward(issues, draws, config=config)
    return ScriptFixture(
        result=result,
        config=config,
        probability_metrics=backtest.calculate_probability_metrics(result),
        topk_hits=backtest.calculate_topk_hits(result),
        calibrations=backtest.calculate_calibration(result),
    )


def _write_test_data(project_root: Path) -> Path:
    data_path = project_root / "input" / "data.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(TEST_DATA_BYTES)
    return data_path


def _write_outputs(
    directory: Path,
    fixture: ScriptFixture,
    *,
    data_path: Path,
    project_root: Path,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    backtest.write_issue_probabilities(
        directory / backtest.OUTPUT_FILENAMES[0], fixture.result
    )
    backtest.write_issue_metrics(
        directory / backtest.OUTPUT_FILENAMES[1],
        fixture.result,
        fixture.probability_metrics,
    )
    backtest.write_topk_metrics(
        directory / backtest.OUTPUT_FILENAMES[2], fixture.result, fixture.topk_hits
    )
    backtest.write_hyperparameter_history(
        directory / backtest.OUTPUT_FILENAMES[3], fixture.result, fixture.config
    )
    backtest.write_calibration(
        directory / backtest.OUTPUT_FILENAMES[4], fixture.calibrations
    )
    report = backtest.build_report(
        fixture.result,
        fixture.config,
        fixture.probability_metrics,
        fixture.topk_hits,
        fixture.calibrations,
        data_path=data_path,
        project_root=project_root,
    )
    (directory / "report.md").write_text(report, encoding="utf-8")
    return report


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def test_output_csv_fields_and_row_counts(
    tmp_path: Path, script_fixture: ScriptFixture
) -> None:
    data_path = _write_test_data(tmp_path)
    _write_outputs(
        tmp_path / "outputs",
        script_fixture,
        data_path=data_path,
        project_root=tmp_path,
    )
    output_dir = tmp_path / "outputs"
    outer_count = len(script_fixture.result.outer_indices)
    expected = {
        "issue_probabilities.csv": (
            80 * outer_count,
            {"issue", "number", "posterior_mean", "posterior_variance", "rank"},
        ),
        "issue_metrics.csv": (
            2 * outer_count,
            {"issue", "strategy", "brier_score", "bernoulli_log_loss"},
        ),
        "topk_metrics.csv": (
            len(COMPARATOR_STRATEGIES) * 10 * outer_count,
            {"issue", "strategy", "k", "actual_hits", "excess_hits"},
        ),
        "hyperparameter_history.csv": (
            9 * outer_count,
            {"outer_issue", "decay", "prior_strength", "selected"},
        ),
        "calibration.csv": (
            20,
            {"strategy", "bin_index", "count", "expected_calibration_error"},
        ),
    }
    for filename, (row_count, required_fields) in expected.items():
        fields, rows = _read_csv(output_dir / filename)
        assert len(rows) == row_count
        assert required_fields.issubset(fields)


def test_same_input_writes_byte_identical_outputs(
    tmp_path: Path, script_fixture: ScriptFixture
) -> None:
    data_path = _write_test_data(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_outputs(
        first,
        script_fixture,
        data_path=data_path,
        project_root=tmp_path,
    )
    _write_outputs(
        second,
        script_fixture,
        data_path=data_path,
        project_root=tmp_path,
    )

    for filename in (*backtest.OUTPUT_FILENAMES, "report.md"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_report_matches_issue_and_topk_csv_summaries(
    tmp_path: Path, script_fixture: ScriptFixture
) -> None:
    data_path = _write_test_data(tmp_path)
    output_dir = tmp_path / "outputs"
    report = _write_outputs(
        output_dir,
        script_fixture,
        data_path=data_path,
        project_root=tmp_path,
    )
    _, issue_rows = _read_csv(output_dir / "issue_metrics.csv")
    dynamic_rows = [row for row in issue_rows if row["strategy"] == "dynamic_bayesian"]
    mean_brier = fmean(float(row["brier_score"]) for row in dynamic_rows)
    mean_log_loss = fmean(float(row["bernoulli_log_loss"]) for row in dynamic_rows)
    assert f"平均 Brier score 为 `{mean_brier:.9f}`" in report
    assert f"平均 Bernoulli log loss 为 `{mean_log_loss:.9f}`" in report

    _, topk_rows = _read_csv(output_dir / "topk_metrics.csv")
    means = {
        strategy: fmean(
            float(row["actual_hits"])
            for row in topk_rows
            if row["strategy"] == strategy and row["k"] == "10"
        )
        for strategy in COMPARATOR_STRATEGIES
    }
    best_strategy = max(COMPARATOR_STRATEGIES, key=means.__getitem__)
    assert (
        f"Top-10 历史平均命中最高的比较项为 `{best_strategy}`"
        f"（`{means[best_strategy]:.6f}`）"
    ) in report


def test_outputs_do_not_read_repository_data_cache(
    tmp_path: Path,
    script_fixture: ScriptFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_data = (ROOT / "data_cache/kl8/data.csv").resolve()
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == repository_data:
            raise FileNotFoundError(repository_data)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    data_path = _write_test_data(tmp_path)
    report = _write_outputs(
        tmp_path / "outputs",
        script_fixture,
        data_path=data_path,
        project_root=tmp_path,
    )

    expected_hash = hashlib.sha256(TEST_DATA_BYTES).hexdigest()
    assert "输入数据：`input/data.csv`" in report
    assert expected_hash in report


def test_production_cli_defaults_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["research_v2_backtest.py"])

    arguments = backtest.parse_args()

    assert arguments.data == "data_cache/kl8/data.csv"
    assert arguments.output_dir == "results/research_v2"
    assert arguments.report == "reports/kl8_v2_research_report.md"


def test_path_escape_is_rejected() -> None:
    with pytest.raises(ValueError, match="项目目录内"):
        backtest._resolve_within(ROOT, "../escaped.csv", "测试路径")


@pytest.mark.parametrize(
    ("output_dir", "report", "message"),
    (
        (
            "results/not-research-v2",
            "reports/kl8_v2_research_report.md",
            "结果目录必须",
        ),
        (
            "results/research_v2",
            "reports/not-kl8-v2.md",
            "报告必须",
        ),
    ),
)
def test_result_and_report_path_restrictions_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: str,
    report: str,
    message: str,
) -> None:
    arguments = argparse.Namespace(
        data="data_cache/kl8/data.csv",
        output_dir=output_dir,
        report=report,
    )
    monkeypatch.setattr(backtest, "parse_args", lambda: arguments)

    with pytest.raises(ValueError, match=message):
        backtest.run()


def test_script_layer_preserves_v1_sources_and_2026188_manifest() -> None:
    actual = {
        relative: _normalised_hash(ROOT / relative) for relative in V1_SOURCE_HASHES
    }
    manifest = ROOT / "reports/prospective_manifests/2026188.json"

    assert actual == V1_SOURCE_HASHES
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest() in MANIFEST_ALLOWED_RAW_HASHES
    )
    assert _normalised_hash(manifest) == MANIFEST_NORMALISED_HASH


def test_makefile_ci_routes_through_all_research_v2_quality_paths() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    expected_paths = (
        "src/research_v2",
        "scripts/research_v2_backtest.py",
        "tests/test_research_v2_bayesian.py",
        "tests/test_research_v2_metrics.py",
        "tests/test_research_v2_evaluation.py",
        "tests/test_research_v2_backtest.py",
    )

    assert all(path in makefile for path in expected_paths)
    assert "mypy --strict src/research_v2 scripts/research_v2_backtest.py" in makefile
    assert "research-v2-backtest:" in makefile
    assert "ci: fmt lint test build" in makefile
