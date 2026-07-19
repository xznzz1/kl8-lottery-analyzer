"""v2 第三阶段未来前瞻冻结、远程封存、顺序链和汇总测试。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts import research_v2_prospective as prospective_cli
from src.research_v2 import prospective_monitor as monitor
from src.research_v2.prospective_monitor import (
    CHANGEPOINT_STRATEGY,
    CONFIRMATION_ISSUE_COUNT,
    DATA_RELATIVE_PATH,
    DYNAMIC_STRATEGY,
    EVALUATION_EVIDENCE_STATUS,
    FINAL_EVALUATION_SEAL_EVIDENCE_STATUS,
    FINAL_EVALUATION_SEAL_FILENAME,
    FIXED_HIGH_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    FREEZE_ACTIVE,
    FREEZE_ID,
    FREEZE_PENDING,
    FREEZE_RELATIVE_PATH,
    FROZEN_SOURCE_PATHS,
    GITHUB_BASE_BRANCH,
    GITHUB_REPOSITORY,
    MANIFEST_EVIDENCE_STATUS,
    MANIFEST_RELATIVE_DIR,
    PRIMARY_COMPARISON_MODELS,
    PROBABILITY_STRATEGIES,
    RESULT_RELATIVE_DIR,
    UNIFORM_BASE_SEEDS,
    UNIFORM_STRATEGY,
    ManifestChainEntry,
    ProspectivePrediction,
    build_evaluation_record,
    build_final_evaluation_seal,
    build_formal_summary,
    build_manifest,
    calculate_evaluation_comparisons,
    calculate_manifest_metrics,
    canonical_json_bytes,
    configuration_fingerprint,
    holm_adjust,
    load_and_verify_freeze_config,
    load_history_csv,
    moving_block_bootstrap_inference,
    next_chain_position,
    normalized_lf_sha256,
    predict_frozen_models,
    raw_sha256,
    require_active_freeze,
    require_contract_path,
    source_manifest_fingerprint,
    uniform_seed_rankings,
    validate_unpublished_target,
    write_evaluation_exclusive,
    write_final_evaluation_seal_exclusive,
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
    rows: list[dict[str, int]] = []
    for period in range(periods):
        row = {"期数": 2020001 + period}
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


def _load_config_template() -> dict[str, Any]:
    parsed: object = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], copy.deepcopy(parsed))


def _finish_config(config: dict[str, Any]) -> None:
    config["configuration_sha256"] = configuration_fingerprint(config)


def _prepare_project(
    tmp_path: Path, *, status: str = FREEZE_ACTIVE
) -> tuple[Path, Path, Path, IntArray, IntArray, int, dict[str, Any]]:
    root = tmp_path / "project"
    data_path = root / DATA_RELATIVE_PATH
    issues, draws = _write_data(data_path)
    files: dict[str, str] = {}
    for relative in FROZEN_SOURCE_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
        files[relative] = normalized_lf_sha256(destination)
    config = _load_config_template()
    config["freeze_status"] = status
    source_manifest = cast(dict[str, Any], config["source_manifest"])
    source_manifest["files"] = files
    source_manifest["fingerprint"] = source_manifest_fingerprint(files)
    _finish_config(config)
    config_path = root / FREEZE_RELATIVE_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(canonical_json_bytes(config))
    return (
        root,
        data_path,
        config_path,
        issues,
        draws,
        int(issues[-1]) + 1,
        config,
    )


def _fake_prediction(target_issue: int) -> ProspectivePrediction:
    del target_issue
    probabilities = {
        strategy: np.full(80, 0.25, dtype=np.float64)
        for strategy in PROBABILITY_STRATEGIES
    }
    numbers = np.arange(1, 81, dtype=np.int64)
    rankings = {
        strategy: np.roll(numbers, offset).copy()
        for offset, strategy in enumerate(PROBABILITY_STRATEGIES[1:], start=1)
    }
    return ProspectivePrediction(
        probabilities=probabilities,
        rankings=rankings,
        uniform_rankings=uniform_seed_rankings(2020487),
        dynamic_decay=0.99,
        dynamic_prior_strength=20.0,
        changepoint_recent_window=60,
        changepoint_change_score=1.0,
        changepoint_threshold=1.1,
        changepoint_high_change=False,
        changepoint_active_decay=0.995,
        changepoint_effective_history_length=486,
    )


def _install_fake_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(
        issues: IntArray, draws: IntArray, *, target_issue: int
    ) -> ProspectivePrediction:
        del issues, draws
        prediction = _fake_prediction(target_issue)
        return ProspectivePrediction(
            probabilities=prediction.probabilities,
            rankings=prediction.rankings,
            uniform_rankings=uniform_seed_rankings(target_issue),
            dynamic_decay=prediction.dynamic_decay,
            dynamic_prior_strength=prediction.dynamic_prior_strength,
            changepoint_recent_window=prediction.changepoint_recent_window,
            changepoint_change_score=prediction.changepoint_change_score,
            changepoint_threshold=prediction.changepoint_threshold,
            changepoint_high_change=prediction.changepoint_high_change,
            changepoint_active_decay=prediction.changepoint_active_decay,
            changepoint_effective_history_length=(
                prediction.changepoint_effective_history_length
            ),
        )

    monkeypatch.setattr(monitor, "predict_frozen_models", fake)


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
        manifest_dir=root / MANIFEST_RELATIVE_DIR,
        results_dir=root / RESULT_RELATIVE_DIR,
        issues=issues,
        draws=draws,
        target_issue=target_issue,
        local_manifest_generated_at_utc="2026-07-01T10:00:00Z",
        git_commit_sha="a" * 40,
        official_source_url="https://example.gov.cn/kl8/next-issue",
        official_confirmed_at_utc="2026-07-01T09:55:00Z",
    )


class _SealClient:
    def __init__(
        self,
        manifest_bytes: bytes,
        *,
        state: str = "MERGED",
        base: str = GITHUB_BASE_BRANCH,
        merged_at: str = "2026-07-01T11:00:00Z",
        merge_commit: str = "b" * 40,
        repository: str = GITHUB_REPOSITORY,
        file_bytes: Mapping[str, bytes] | None = None,
    ) -> None:
        self.manifest_bytes = manifest_bytes
        self.file_bytes = file_bytes
        self.state = state
        self.base = base
        self.merged_at = merged_at
        self.merge_commit = merge_commit
        self.repository = repository
        self.pull_request_calls = 0
        self.file_calls = 0

    def get_pull_request(self, repository: str, pr_number: int) -> Mapping[str, object]:
        del repository
        self.pull_request_calls += 1
        return {
            "repository": self.repository,
            "number": pr_number,
            "url": f"https://github.com/{GITHUB_REPOSITORY}/pull/{pr_number}",
            "state": self.state,
            "baseRefName": self.base,
            "mergedAt": self.merged_at,
            "mergeCommit": {"oid": self.merge_commit},
        }

    def get_file_bytes(self, repository: str, path: str, commit_sha: str) -> bytes:
        del repository, commit_sha
        self.file_calls += 1
        if self.file_bytes is not None:
            return self.file_bytes[path]
        return self.manifest_bytes


class _FailingSealClient(_SealClient):
    def get_pull_request(self, repository: str, pr_number: int) -> Mapping[str, object]:
        del repository, pr_number
        raise OSError("offline")


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
    return manifest, write_manifest_exclusive(manifest, root / MANIFEST_RELATIVE_DIR)


def _build_verified_evaluation(
    root: Path,
    config_path: Path,
    manifest_path: Path,
    client: _SealClient | None = None,
) -> dict[str, object]:
    return build_evaluation_record(
        project_root=root,
        config_path=config_path,
        manifest_path=manifest_path,
        actual_numbers=np.arange(1, 21, dtype=np.int64),
        seal_pr_number=17,
        official_result_source_url="https://example.gov.cn/kl8/result",
        official_result_published_at_utc="2026-07-02T10:00:00Z",
        evaluated_at_utc="2026-07-02T10:05:00Z",
        seal_client=client or _SealClient(manifest_path.read_bytes()),
    )


def _completed_first_issue(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    IntArray,
    IntArray,
    int,
    Path,
]:
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    _, manifest_path = _write_test_manifest(root, data, config, issues, draws, target)
    evaluation = _build_verified_evaluation(root, config, manifest_path)
    evaluation_path = write_evaluation_exclusive(evaluation, root / RESULT_RELATIVE_DIR)
    next_issues = np.append(issues, target)
    next_draws = np.vstack((draws, np.arange(1, 21, dtype=np.int64)))
    return root, data, config, next_issues, next_draws, target, evaluation_path


def _second_manifest_ready(
    tmp_path: Path,
) -> tuple[Path, Path, Path, int, Path, Path]:
    root, data, config, issues, draws, previous_target, evaluation_path = (
        _completed_first_issue(tmp_path)
    )
    target = previous_target + 10
    _, manifest_path = _write_test_manifest(root, data, config, issues, draws, target)
    return root, config, data, target, manifest_path, evaluation_path


def test_pending_status_blocks_all_four_production_commands(tmp_path: Path) -> None:
    root, _, _, _, _, target, _ = _prepare_project(tmp_path, status=FREEZE_PENDING)
    manifest_args = prospective_cli.parse_args(
        [
            "--project-root",
            str(root),
            "manifest",
            "--target-issue",
            str(target),
            "--official-source-url",
            "https://example.gov.cn/next",
            "--official-confirmed-at-utc",
            "2026-07-01T09:00:00Z",
        ]
    )
    evaluate_args = prospective_cli.parse_args(
        [
            "--project-root",
            str(root),
            "evaluate",
            "--target-issue",
            str(target),
            "--seal-pr-number",
            "1",
            "--official-result-source-url",
            "https://example.gov.cn/result",
            "--official-result-published-at-utc",
            "2026-07-02T10:00:00Z",
        ]
    )
    summary_args = prospective_cli.parse_args(["--project-root", str(root), "summary"])
    finalize_args = prospective_cli.parse_args(
        [
            "--project-root",
            str(root),
            "finalize-evaluation",
            "--target-issue",
            str(target),
            "--evaluation-seal-pr-number",
            "1",
        ]
    )
    for command, args in (
        (prospective_cli._manifest_command, manifest_args),
        (prospective_cli._evaluate_command, evaluate_args),
        (prospective_cli._finalize_evaluation_command, finalize_args),
        (prospective_cli._summary_command, summary_args),
    ):
        with pytest.raises(RuntimeError, match="pending"):
            command(args, root)


def test_cli_has_no_manual_time_or_git_sha_overrides() -> None:
    base = [
        "manifest",
        "--target-issue",
        "2027001",
        "--official-source-url",
        "https://example.gov.cn/next",
        "--official-confirmed-at-utc",
        "2026-07-01T09:00:00Z",
    ]
    with pytest.raises(SystemExit):
        prospective_cli.parse_args([*base, "--generated-at-utc", "x"])
    with pytest.raises(SystemExit):
        prospective_cli.parse_args([*base, "--git-commit-sha", "a" * 40])
    with pytest.raises(SystemExit):
        prospective_cli.parse_args(["summary", "--data", "other.csv"])


def test_active_freeze_requires_existing_ancestor_tag() -> None:
    config: dict[str, Any] = {"freeze_status": FREEZE_ACTIVE, "freeze_tag": "tag"}

    def accepted(arguments: tuple[str, ...]) -> str:
        if arguments[:2] == ("rev-parse", "--verify"):
            return "a" * 40
        if arguments == ("rev-parse", "HEAD"):
            return "b" * 40
        return ""

    require_active_freeze(ROOT, config, git_command=accepted)

    def rejected(arguments: tuple[str, ...]) -> str:
        if arguments[0] == "merge-base":
            raise subprocess.CalledProcessError(1, list(arguments))
        return "a" * 40

    with pytest.raises(RuntimeError, match="标签"):
        require_active_freeze(ROOT, config, git_command=rejected)


def test_unpublished_target_validation_rejects_target_and_future_rows(
    tmp_path: Path,
) -> None:
    _, _, _, issues, draws, target, _ = _prepare_project(tmp_path)
    validate_unpublished_target(issues, draws, target)
    with pytest.raises(ValueError, match="已经存在"):
        validate_unpublished_target(
            np.append(issues, target), np.vstack((draws, np.arange(1, 21))), target
        )
    with pytest.raises(ValueError, match="晚于target_issue"):
        validate_unpublished_target(
            np.append(issues, target + 1),
            np.vstack((draws, np.arange(1, 21))),
            target,
        )
    with pytest.raises(ValueError, match="已经存在"):
        validate_unpublished_target(issues, draws, int(issues[-1]))


def test_manifest_records_latest_history_and_deterministic_uniform_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    first = _build_test_manifest(root, data, config, issues, draws, target)
    second = _build_test_manifest(root, data, config, issues, draws, target)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["history_through_issue"] == int(issues[-1])
    assert first["history_issue_count"] == len(issues)
    assert first["confirmation_index"] == 1
    assert first["previous_manifest_sha256"] is None
    assert first["evidence_status"] == MANIFEST_EVIDENCE_STATUS
    assert first["remote_preseal_verified"] is False
    models = cast(dict[str, Any], first["models"])
    uniform = cast(dict[str, Any], models[UNIFORM_STRATEGY])
    assert uniform["probabilities"] == [0.25] * 80
    assert len(uniform["rankings_by_seed"]) == 20
    assert all(len(row["ranking"]) == 80 for row in uniform["rankings_by_seed"])


def test_uniform_rankings_change_by_target_and_are_reproducible() -> None:
    first = uniform_seed_rankings(2027001)
    repeat = uniform_seed_rankings(2027001)
    next_issue = uniform_seed_rankings(2027002)
    assert len(first) == len(UNIFORM_BASE_SEEDS) == 20
    assert all(np.array_equal(a, b) for a, b in zip(first, repeat, strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(first, next_issue, strict=True))
    assert all(set(map(int, ranking)) == set(range(1, 81)) for ranking in first)


def test_uniform_topk_is_averaged_within_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    manifest = _build_test_manifest(root, data, config, issues, draws, target)
    metrics = calculate_manifest_metrics(
        cast(dict[str, Any], manifest), np.arange(1, 21, dtype=np.int64)
    )
    models = cast(dict[str, Any], manifest["models"])
    seeded = models[UNIFORM_STRATEGY]["rankings_by_seed"]
    expected = np.mean(
        [sum(number <= 20 for number in row["ranking"][:10]) for row in seeded]
    )
    row = cast(dict[str, Any], metrics[UNIFORM_STRATEGY])["top_k"]["10"]
    assert row["hits"] == pytest.approx(expected)
    assert row["aggregation"] == "within_issue_mean_over_20_seeds"
    assert row["seed_count"] == 20


def test_real_model_prediction_is_legal_and_reproducible(tmp_path: Path) -> None:
    _, _, _, issues, draws, target, _ = _prepare_project(tmp_path)
    first = predict_frozen_models(issues, draws, target_issue=target)
    second = predict_frozen_models(issues, draws, target_issue=target)
    for strategy in PROBABILITY_STRATEGIES:
        assert np.array_equal(
            first.probabilities[strategy], second.probabilities[strategy]
        )
        probabilities = first.probabilities[strategy]
        assert np.all((probabilities > 0.0) & (probabilities < 1.0))
        assert float(probabilities.sum()) == pytest.approx(20.0, abs=1e-9)
    assert len(first.uniform_rankings) == 20


def test_manifest_chain_requires_previous_evaluation_and_links_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config_path, issues, draws, target, config = _prepare_project(tmp_path)
    _, first_path = _write_test_manifest(root, data, config_path, issues, draws, target)
    with pytest.raises(ValueError, match="一一对应"):
        _build_test_manifest(root, data, config_path, issues, draws, target + 1)
    evaluation = _build_verified_evaluation(root, config_path, first_path)
    results = root / RESULT_RELATIVE_DIR
    evaluation_path = write_evaluation_exclusive(evaluation, results)
    next_issues = np.append(issues, target)
    next_draws = np.vstack((draws, np.arange(1, 21, dtype=np.int64)))
    second = _build_test_manifest(
        root, data, config_path, next_issues, next_draws, target + 10
    )
    assert second["confirmation_index"] == 2
    assert second["protocol_start_target_issue"] == target
    assert second["previous_target_issue"] == target
    assert second["previous_manifest_sha256"] == raw_sha256(first_path)
    assert second["history_through_issue"] == target
    assert second["previous_evaluation_target_issue"] == target
    assert second["previous_evaluation_confirmation_index"] == 1
    assert (
        second["previous_evaluation_path"]
        == (RESULT_RELATIVE_DIR / f"{target}.json").as_posix()
    )
    assert second["previous_evaluation_sha256"] == raw_sha256(evaluation_path)
    second["previous_manifest_sha256"] = "0" * 64
    write_manifest_exclusive(second, root / MANIFEST_RELATIVE_DIR)
    with pytest.raises(ValueError, match="SHA-256链断裂"):
        next_chain_position(
            project_root=root,
            manifest_dir=root / MANIFEST_RELATIVE_DIR,
            results_dir=results,
            config=config,
            target_issue=target + 20,
            history_issues=next_issues,
            history_draws=next_draws,
        )


def test_same_freeze_cannot_skip_official_draw_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, previous_target, _ = _completed_first_issue(
        tmp_path
    )
    skipped_issue = previous_target + 5
    skipped_issues = np.append(issues, skipped_issue)
    skipped_draws = np.vstack((draws, np.arange(21, 41, dtype=np.int64)))
    for proposed_target in (previous_target + 10, previous_target + 100):
        with pytest.raises(ValueError, match="同一freeze_id禁止跳过后恢复"):
            _build_test_manifest(
                root,
                data,
                config,
                skipped_issues,
                skipped_draws,
                proposed_target,
            )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw_bytes", "原始字节不是规范"),
        ("actual_numbers", "actual_numbers"),
        ("models", "models"),
        ("comparisons", "comparisons"),
    ],
)
def test_modified_previous_evaluation_refuses_next_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, previous_target, evaluation_path = (
        _completed_first_issue(tmp_path)
    )
    if mutation == "raw_bytes":
        evaluation_path.write_bytes(evaluation_path.read_bytes() + b" ")
    else:
        record = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if mutation == "actual_numbers":
            record["actual_numbers"] = list(range(2, 22))
        elif mutation == "models":
            record["models"][DYNAMIC_STRATEGY]["brier_score"] += 0.1
        else:
            record["comparisons"]["versus_uniform"][DYNAMIC_STRATEGY][
                "brier_difference_model_minus_uniform"
            ] += 0.1
        evaluation_path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match=message):
        _build_test_manifest(root, data, config, issues, draws, previous_target + 10)


def test_existing_manifest_and_evaluation_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    manifest, path = _write_test_manifest(root, data, config, issues, draws, target)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        write_manifest_exclusive(manifest, root / MANIFEST_RELATIVE_DIR)
    assert path.read_bytes() == before
    record = _build_verified_evaluation(root, config, path)
    result_path = write_evaluation_exclusive(record, root / RESULT_RELATIVE_DIR)
    result_before = result_path.read_bytes()
    with pytest.raises(FileExistsError, match="拒绝重复写入"):
        write_evaluation_exclusive(record, root / RESULT_RELATIVE_DIR)
    assert result_path.read_bytes() == result_before


@pytest.mark.parametrize(
    ("client_factory", "message"),
    [
        (lambda raw: _SealClient(raw, state="OPEN"), "尚未合并"),
        (
            lambda raw: _SealClient(raw, merged_at="2026-07-02T10:00:00Z"),
            "不早于",
        ),
        (lambda raw: _SealClient(raw, base="main"), "base分支错误"),
        (lambda raw: _SealClient(b"wrong"), "与本地文件不符"),
        (
            lambda raw: _SealClient(raw, repository="someone/else"),
            "不属于冻结GitHub仓库",
        ),
    ],
)
def test_remote_preseal_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Any,
    message: str,
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    _, path = _write_test_manifest(root, data, config, issues, draws, target)
    with pytest.raises(ValueError, match=message):
        _build_verified_evaluation(
            root, config, path, client_factory(path.read_bytes())
        )
    assert not (root / RESULT_RELATIVE_DIR).exists()


def test_remote_preseal_network_failure_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    _, path = _write_test_manifest(root, data, config, issues, draws, target)
    with pytest.raises(RuntimeError, match="API不可用"):
        _build_verified_evaluation(
            root, config, path, _FailingSealClient(path.read_bytes())
        )
    assert not (root / RESULT_RELATIVE_DIR).exists()


def test_verified_evaluation_records_complete_remote_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    _, path = _write_test_manifest(root, data, config, issues, draws, target)
    record = _build_verified_evaluation(root, config, path)
    assert record["remote_preseal_verified"] is True
    assert record["sealed_before_official_result"] is True
    assert record["seal_pr_number"] == 17
    assert record["seal_merge_commit_sha"] == "b" * 40
    assert record["seal_merged_at_utc"] == "2026-07-01T11:00:00Z"
    assert record["manifest_sha256_at_merge"] == raw_sha256(path)
    assert record["official_result_source_url"].startswith("https://")
    assert record["evaluation_locally_created"] is True
    assert record["remote_evaluation_anchor_pending"] is True
    assert isinstance(record["evaluation_payload_sha256"], str)


@pytest.mark.parametrize(
    ("previous_bytes", "message"),
    [
        (None, "不含上一evaluation"),
        (b"wrong", "上一evaluation SHA-256不匹配"),
    ],
)
def test_next_manifest_seal_must_anchor_previous_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    previous_bytes: bytes | None,
    message: str,
) -> None:
    _install_fake_prediction(monkeypatch)
    root, config, _, _, manifest_path, evaluation_path = _second_manifest_ready(
        tmp_path
    )
    manifest_repository_path = manifest_path.relative_to(root).as_posix()
    evaluation_repository_path = evaluation_path.relative_to(root).as_posix()
    files = {manifest_repository_path: manifest_path.read_bytes()}
    if previous_bytes is not None:
        files[evaluation_repository_path] = previous_bytes
    client = _SealClient(manifest_path.read_bytes(), file_bytes=files)
    with pytest.raises((RuntimeError, ValueError), match=message):
        _build_verified_evaluation(root, config, manifest_path, client)


def test_next_evaluation_records_previous_remote_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    root, config, _, _, manifest_path, evaluation_path = _second_manifest_ready(
        tmp_path
    )
    files = {
        manifest_path.relative_to(root).as_posix(): manifest_path.read_bytes(),
        evaluation_path.relative_to(root).as_posix(): evaluation_path.read_bytes(),
    }
    record = _build_verified_evaluation(
        root,
        config,
        manifest_path,
        _SealClient(manifest_path.read_bytes(), file_bytes=files),
    )
    assert record["previous_evaluation_remote_anchor_verified"] is True
    anchor = cast(dict[str, Any], record["previous_evaluation_anchor"])
    assert anchor["sha256"] == raw_sha256(evaluation_path)
    assert anchor["sha256_at_merge"] == raw_sha256(evaluation_path)


def test_finalize_rejects_arbitrary_evaluation_fixtures_before_github(
    tmp_path: Path,
) -> None:
    root, _, config_path, issues, draws, _, _ = _prepare_project(tmp_path)
    results = root / RESULT_RELATIVE_DIR
    results.mkdir(parents=True)
    for offset in range(CONFIRMATION_ISSUE_COUNT):
        (results / f"{2030001 + offset}.json").write_bytes(
            canonical_json_bytes({"fixture": offset})
        )
    client = _SealClient(b"unused")
    with pytest.raises(ValueError, match="365份manifest"):
        build_final_evaluation_seal(
            project_root=root,
            config_path=config_path,
            manifest_dir=root / MANIFEST_RELATIVE_DIR,
            results_dir=results,
            official_issues=issues,
            official_draws=draws,
            target_issue=2030365,
            evaluation_seal_pr_number=99,
            seal_client=client,
        )
    assert client.pull_request_calls == 0
    assert client.file_calls == 0


def test_summary_rejects_missing_manifest_or_evaluation(tmp_path: Path) -> None:
    root, _, _, issues, draws, _, config = _prepare_project(tmp_path)
    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    manifests.mkdir()
    results.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="365份manifest"):
        build_formal_summary(
            manifest_dir=manifests,
            results_dir=results,
            config=config,
            official_issues=issues,
            official_draws=draws,
            seal_client=_SealClient(b"unused"),
        )

    previous_target: int | None = None
    previous_digest: str | None = None
    first_target = 2030001
    for offset in range(CONFIRMATION_ISSUE_COUNT):
        target = first_target + offset
        payload = {
            "confirmation_index": offset + 1,
            "evidence_status": MANIFEST_EVIDENCE_STATUS,
            "freeze_config_sha256": config["configuration_sha256"],
            "freeze_id": config["freeze_id"],
            "frozen_parameters": config["models"],
            "history_through_issue": previous_target,
            "previous_manifest_sha256": previous_digest,
            "previous_target_issue": previous_target,
            "previous_evaluation_target_issue": previous_target,
            "previous_evaluation_confirmation_index": (offset if offset > 0 else None),
            "previous_evaluation_path": (
                (RESULT_RELATIVE_DIR / f"{previous_target}.json").as_posix()
                if previous_target is not None
                else None
            ),
            "previous_evaluation_sha256": ("1" * 64 if offset > 0 else None),
            "protocol_start_target_issue": first_target,
            "remote_preseal_verified": False,
            "schema_version": 2,
            "source_manifest": config["source_manifest"],
            "target_issue": target,
        }
        path = manifests / f"{target}.json"
        path.write_bytes(canonical_json_bytes(payload))
        previous_target = target
        previous_digest = raw_sha256(path)
    with pytest.raises(ValueError, match="365份一一对应的evaluation"):
        build_formal_summary(
            manifest_dir=manifests,
            results_dir=results,
            config=config,
            official_issues=issues,
            official_draws=draws,
            seal_client=_SealClient(b"unused"),
        )
    with pytest.raises(ValueError, match="365期确认链已满"):
        next_chain_position(
            project_root=root,
            manifest_dir=manifests,
            results_dir=results,
            config=config,
            target_issue=first_target + CONFIRMATION_ISSUE_COUNT,
            history_issues=issues,
            history_draws=draws,
        )


def _write_compact_completed_chain(
    root: Path, config: dict[str, Any], *, include_final_seal: bool
) -> tuple[Path, Path, IntArray, IntArray]:
    manifests = root / MANIFEST_RELATIVE_DIR
    results = root / RESULT_RELATIVE_DIR
    manifests.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    first_target = 2030001
    previous_target: int | None = None
    previous_manifest_digest: str | None = None
    previous_evaluation_digest: str | None = None
    previous_evaluation_path: Path | None = None
    issues: list[int] = []
    draws: list[IntArray] = []
    for offset in range(CONFIRMATION_ISSUE_COUNT):
        target = first_target + offset
        actual = np.asarray(_draw_for_period(offset), dtype=np.int64)
        manifest = _synthetic_manifest(target, offset)
        manifest.update(
            {
                "schema_version": 2,
                "evidence_status": MANIFEST_EVIDENCE_STATUS,
                "remote_preseal_verified": False,
                "freeze_id": FREEZE_ID,
                "freeze_config_sha256": config["configuration_sha256"],
                "confirmation_index": offset + 1,
                "protocol_start_target_issue": first_target,
                "previous_target_issue": previous_target,
                "previous_manifest_sha256": previous_manifest_digest,
                "previous_evaluation_target_issue": previous_target,
                "previous_evaluation_confirmation_index": (
                    offset if offset > 0 else None
                ),
                "previous_evaluation_path": (
                    previous_evaluation_path.relative_to(root).as_posix()
                    if previous_evaluation_path is not None
                    else None
                ),
                "previous_evaluation_sha256": previous_evaluation_digest,
                "source_manifest": config["source_manifest"],
                "frozen_parameters": config["models"],
                "history_through_issue": (
                    previous_target if previous_target is not None else first_target - 1
                ),
            }
        )
        manifest_path = manifests / f"{target}.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        manifest_digest = raw_sha256(manifest_path)
        metrics = calculate_manifest_metrics(manifest, actual)
        evaluation = {
            "schema_version": 2,
            "evidence_status": EVALUATION_EVIDENCE_STATUS,
            "freeze_id": FREEZE_ID,
            "confirmation_index": offset + 1,
            "target_issue": target,
            "actual_numbers": sorted(map(int, actual)),
            "remote_preseal_verified": True,
            "sealed_before_official_result": True,
            "evaluation_locally_created": True,
            "remote_evaluation_anchor_pending": True,
            "manifest_sha256_at_merge": manifest_digest,
            "manifest": {"sha256": manifest_digest},
            "previous_evaluation_remote_anchor_verified": (
                True if offset > 0 else None
            ),
            "previous_evaluation_anchor": (
                {
                    "target_issue": previous_target,
                    "confirmation_index": offset,
                    "path": previous_evaluation_path.relative_to(root).as_posix(),
                    "sha256": previous_evaluation_digest,
                    "sha256_at_merge": previous_evaluation_digest,
                }
                if previous_evaluation_path is not None
                else None
            ),
            "changepoint_state": "high_change" if offset % 5 == 0 else "normal",
            "models": metrics,
            "comparisons": calculate_evaluation_comparisons(metrics),
        }
        evaluation["evaluation_payload_sha256"] = (
            monitor._evaluation_payload_fingerprint(evaluation)
        )
        evaluation_path = results / f"{target}.json"
        evaluation_path.write_bytes(canonical_json_bytes(evaluation))
        previous_target = target
        previous_manifest_digest = manifest_digest
        previous_evaluation_path = evaluation_path
        previous_evaluation_digest = raw_sha256(evaluation_path)
        issues.append(target)
        draws.append(actual)
    if include_final_seal:
        assert previous_target is not None
        assert previous_evaluation_path is not None
        assert previous_evaluation_digest is not None
        final_seal = {
            "schema_version": 2,
            "evidence_status": FINAL_EVALUATION_SEAL_EVIDENCE_STATUS,
            "freeze_id": FREEZE_ID,
            "target_issue": previous_target,
            "confirmation_index": CONFIRMATION_ISSUE_COUNT,
            "evaluation_path": previous_evaluation_path.relative_to(root).as_posix(),
            "evaluation_sha256": previous_evaluation_digest,
            "seal_pr_number": 999,
            "seal_pr_url": f"https://github.com/{GITHUB_REPOSITORY}/pull/999",
            "seal_merge_commit_sha": "c" * 40,
            "seal_merged_at_utc": "2028-01-01T00:00:00Z",
            "remote_evaluation_anchor_verified": True,
        }
        final_seal["final_seal_payload_sha256"] = (
            monitor._final_seal_payload_fingerprint(final_seal)
        )
        (results / FINAL_EVALUATION_SEAL_FILENAME).write_bytes(
            canonical_json_bytes(final_seal)
        )
    return (
        manifests,
        results,
        np.asarray(issues, dtype=np.int64),
        np.asarray(draws, dtype=np.int64),
    )


def test_completed_chain_requires_final_seal_and_official_data_match(
    tmp_path: Path,
) -> None:
    root, _, _, _, _, _, config = _prepare_project(tmp_path)
    manifests, results, issues, draws = _write_compact_completed_chain(
        root, config, include_final_seal=False
    )
    with pytest.raises(ValueError, match="缺少final evaluation seal"):
        monitor._validate_completed_chain(
            manifests, results, config, issues, draws, _SealClient(b"unused")
        )
    last_target = int(issues[-1])
    last_evaluation = results / f"{last_target}.json"
    seal = {
        "schema_version": 2,
        "evidence_status": FINAL_EVALUATION_SEAL_EVIDENCE_STATUS,
        "freeze_id": FREEZE_ID,
        "target_issue": last_target,
        "confirmation_index": CONFIRMATION_ISSUE_COUNT,
        "evaluation_path": last_evaluation.relative_to(root).as_posix(),
        "evaluation_sha256": raw_sha256(last_evaluation),
        "seal_pr_number": 999,
        "seal_pr_url": f"https://github.com/{GITHUB_REPOSITORY}/pull/999",
        "seal_merge_commit_sha": "c" * 40,
        "seal_merged_at_utc": "2028-01-01T00:00:00Z",
        "remote_evaluation_anchor_verified": True,
    }
    seal["final_seal_payload_sha256"] = monitor._final_seal_payload_fingerprint(seal)
    (results / FINAL_EVALUATION_SEAL_FILENAME).write_bytes(canonical_json_bytes(seal))
    (results / "formal_summary.json").write_bytes(
        canonical_json_bytes({"fixture": True})
    )
    paired = monitor._validate_completed_chain(
        manifests,
        results,
        config,
        issues,
        draws,
        _SealClient(
            last_evaluation.read_bytes(),
            merged_at="2028-01-01T00:00:00Z",
            merge_commit="c" * 40,
        ),
    )
    assert len(paired) == CONFIRMATION_ISSUE_COUNT
    changed_draws = draws.copy()
    changed_draws[10] = np.arange(21, 41, dtype=np.int64)
    with pytest.raises(ValueError, match="actual_numbers与正式输入数据不一致"):
        monitor._validate_completed_chain(
            manifests,
            results,
            config,
            issues,
            changed_draws,
            _SealClient(b"unused"),
        )
    (results / "unexpected.json").write_bytes(canonical_json_bytes({"bad": True}))
    with pytest.raises(ValueError, match="额外协议JSON"):
        monitor._validate_completed_chain(
            manifests, results, config, issues, draws, _SealClient(b"unused")
        )


def test_numeric_evaluation_enumeration_excludes_protocol_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    for name in ("2030001.json", FINAL_EVALUATION_SEAL_FILENAME, "formal_summary.json"):
        (results / name).write_bytes(canonical_json_bytes({"name": name}))
    assert [path.name for path in monitor._numeric_evaluation_paths(results)] == [
        "2030001.json"
    ]


def _synthetic_manifest(target: int, issue_offset: int) -> dict[str, Any]:
    numbers = np.arange(1, 81, dtype=np.int64)
    models: dict[str, Any] = {}
    for model_offset, strategy in enumerate(PROBABILITY_STRATEGIES):
        if strategy == UNIFORM_STRATEGY:
            probabilities = np.full(80, 0.25, dtype=np.float64)
        else:
            centered = ((numbers - 1 + issue_offset + model_offset) % 80) - 39.5
            probabilities = 0.25 + (0.015 + model_offset * 0.003) * centered / 39.5
        model: dict[str, Any] = {
            "number_index": list(range(1, 81)),
            "probabilities": list(map(float, probabilities)),
        }
        if strategy == UNIFORM_STRATEGY:
            seeded: list[dict[str, Any]] = []
            for base_seed, ranking in zip(
                UNIFORM_BASE_SEEDS, uniform_seed_rankings(target), strict=True
            ):
                material = f"kl8-v2-prospective-365-v1|{target}|{base_seed}".encode(
                    "ascii"
                )
                digest = hashlib.sha256(material).digest()
                seeded.append(
                    {
                        "base_seed": base_seed,
                        "seed_material_sha256": digest.hex(),
                        "seed_integer": int.from_bytes(digest[:8], "big"),
                        "ranking": list(map(int, ranking)),
                        "top_k_candidates": {
                            str(k): list(map(int, ranking[:k])) for k in range(1, 11)
                        },
                    }
                )
            model["rankings_by_seed"] = seeded
        else:
            ranking = np.lexsort((numbers, -probabilities)) + 1
            model["ranking"] = list(map(int, ranking))
            model["top_k_candidates"] = {
                str(k): list(map(int, ranking[:k])) for k in range(1, 11)
            }
        models[strategy] = model
    return {"target_issue": target, "models": models}


def _write_local_final_seal(
    *,
    root: Path,
    results_dir: Path,
    target_issue: int,
    pr_number: int = 999,
    merge_commit: str = "c" * 40,
    merged_at: str = "2028-01-01T00:00:00Z",
) -> tuple[Path, Path, dict[str, Any]]:
    evaluation_path = results_dir / f"{target_issue}.json"
    seal: dict[str, Any] = {
        "schema_version": 2,
        "evidence_status": FINAL_EVALUATION_SEAL_EVIDENCE_STATUS,
        "freeze_id": FREEZE_ID,
        "target_issue": target_issue,
        "confirmation_index": CONFIRMATION_ISSUE_COUNT,
        "evaluation_path": evaluation_path.relative_to(root).as_posix(),
        "evaluation_sha256": raw_sha256(evaluation_path),
        "seal_pr_number": pr_number,
        "seal_pr_url": f"https://github.com/{GITHUB_REPOSITORY}/pull/{pr_number}",
        "seal_merge_commit_sha": merge_commit,
        "seal_merged_at_utc": merged_at,
        "remote_evaluation_anchor_verified": True,
    }
    seal["final_seal_payload_sha256"] = monitor._final_seal_payload_fingerprint(seal)
    seal_path = results_dir / FINAL_EVALUATION_SEAL_FILENAME
    seal_path.write_bytes(canonical_json_bytes(seal))
    return seal_path, evaluation_path, seal


def test_finalize_prevalidates_complete_chain_before_github_and_writes_once(
    tmp_path: Path,
) -> None:
    root, _, config_path, _, _, _, config = _prepare_project(tmp_path)
    manifests, results, issues, draws = _write_compact_completed_chain(
        root, config, include_final_seal=False
    )
    target = int(issues[-1])

    def rejected(message: str, check_draws: IntArray = draws) -> None:
        client = _SealClient(b"unused")
        with pytest.raises(ValueError, match=message):
            build_final_evaluation_seal(
                project_root=root,
                config_path=config_path,
                manifest_dir=manifests,
                results_dir=results,
                official_issues=issues,
                official_draws=check_draws,
                target_issue=target,
                evaluation_seal_pr_number=999,
                seal_client=client,
            )
        assert client.pull_request_calls == 0
        assert client.file_calls == 0

    missing_manifest = manifests / f"{int(issues[20])}.json"
    original_manifest = missing_manifest.read_bytes()
    missing_manifest.unlink()
    rejected("manifest")
    missing_manifest.write_bytes(original_manifest)

    chained_manifest = manifests / f"{int(issues[20])}.json"
    original_manifest = chained_manifest.read_bytes()
    changed_manifest = cast(
        dict[str, Any], json.loads(original_manifest.decode("utf-8"))
    )
    changed_manifest["previous_manifest_sha256"] = "0" * 64
    chained_manifest.write_bytes(canonical_json_bytes(changed_manifest))
    rejected("manifest SHA-256链断裂")
    chained_manifest.write_bytes(original_manifest)

    anchored_evaluation = results / f"{int(issues[1])}.json"
    original_evaluation = anchored_evaluation.read_bytes()
    changed_evaluation = cast(
        dict[str, Any], json.loads(original_evaluation.decode("utf-8"))
    )
    changed_evaluation["previous_evaluation_remote_anchor_verified"] = False
    changed_evaluation["evaluation_payload_sha256"] = (
        monitor._evaluation_payload_fingerprint(changed_evaluation)
    )
    anchored_evaluation.write_bytes(canonical_json_bytes(changed_evaluation))
    rejected("上一evaluation未由下一manifest seal PR远程锚定")
    anchored_evaluation.write_bytes(original_evaluation)

    changed_draws = draws.copy()
    changed_draws[10] = np.arange(21, 41, dtype=np.int64)
    rejected("actual_numbers与正式输入数据不一致", changed_draws)

    final_evaluation = results / f"{target}.json"
    original_final = final_evaluation.read_bytes()
    changed_final = cast(dict[str, Any], json.loads(original_final.decode("utf-8")))
    models = cast(dict[str, Any], changed_final["models"])
    uniform = cast(dict[str, Any], models[UNIFORM_STRATEGY])
    uniform["brier_score"] = float(uniform["brier_score"]) + 0.01
    changed_final["evaluation_payload_sha256"] = (
        monitor._evaluation_payload_fingerprint(changed_final)
    )
    final_evaluation.write_bytes(canonical_json_bytes(changed_final))
    rejected("models不能从manifest和正式数据独立复算")
    final_evaluation.write_bytes(original_final)

    changed_final = cast(dict[str, Any], json.loads(original_final.decode("utf-8")))
    comparisons = cast(dict[str, Any], changed_final["comparisons"])
    comparisons["fixture_tamper"] = True
    changed_final["evaluation_payload_sha256"] = (
        monitor._evaluation_payload_fingerprint(changed_final)
    )
    final_evaluation.write_bytes(canonical_json_bytes(changed_final))
    rejected("comparisons不能从正式数据独立复算")
    final_evaluation.write_bytes(original_final)

    client = _SealClient(final_evaluation.read_bytes())
    seal = build_final_evaluation_seal(
        project_root=root,
        config_path=config_path,
        manifest_dir=manifests,
        results_dir=results,
        official_issues=issues,
        official_draws=draws,
        target_issue=target,
        evaluation_seal_pr_number=999,
        seal_client=client,
    )
    assert client.pull_request_calls == 1
    assert client.file_calls == 1
    assert seal["evaluation_sha256"] == raw_sha256(final_evaluation)
    output = write_final_evaluation_seal_exclusive(seal, results)
    assert output.name == FINAL_EVALUATION_SEAL_FILENAME
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        write_final_evaluation_seal_exclusive(seal, results)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unmerged", "尚未合并"),
        ("wrong_base", "base分支错误"),
        ("wrong_commit", "合并提交与本地记录不一致"),
        ("wrong_time", "合并时间与本地记录不一致"),
        ("missing_file", "不含第365期evaluation"),
        ("wrong_file", "与本地不符"),
        ("offline", "GitHub API不可用"),
    ],
)
def test_summary_revalidates_final_seal_with_github_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    root, _, _, _, _, _, config = _prepare_project(tmp_path)
    results = root / RESULT_RELATIVE_DIR
    results.mkdir(parents=True)
    target = 2030365
    evaluation_path = results / f"{target}.json"
    evaluation_path.write_bytes(canonical_json_bytes({"target_issue": target}))
    entry = ManifestChainEntry(
        path=root / MANIFEST_RELATIVE_DIR / f"{target}.json",
        payload=_synthetic_manifest(target, 364),
        sha256="a" * 64,
        confirmation_index=CONFIRMATION_ISSUE_COUNT,
        target_issue=target,
    )
    paired = ((entry, {}, np.asarray(_draw_for_period(364), dtype=np.int64)),)
    monkeypatch.setattr(
        monitor,
        "_validate_completed_chain_without_final_seal",
        lambda *args, **kwargs: paired,
    )
    _write_local_final_seal(root=root, results_dir=results, target_issue=target)
    raw = evaluation_path.read_bytes()
    if case == "unmerged":
        client: _SealClient = _SealClient(raw, state="OPEN")
    elif case == "wrong_base":
        client = _SealClient(raw, base="main")
    elif case == "wrong_commit":
        client = _SealClient(
            raw, merged_at="2028-01-01T00:00:00Z", merge_commit="b" * 40
        )
    elif case == "wrong_time":
        client = _SealClient(raw, merge_commit="c" * 40)
    elif case == "missing_file":
        client = _SealClient(raw, file_bytes={})
    elif case == "wrong_file":
        client = _SealClient(b"wrong")
    else:
        client = _FailingSealClient(raw)
    with pytest.raises((RuntimeError, ValueError), match=message):
        build_formal_summary(
            manifest_dir=root / MANIFEST_RELATIVE_DIR,
            results_dir=results,
            config=config,
            official_issues=np.asarray([target], dtype=np.int64),
            official_draws=np.asarray([_draw_for_period(364)], dtype=np.int64),
            seal_client=client,
        )


def test_summary_rejects_locally_forged_final_seal_before_github(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _, _, _, _, config = _prepare_project(tmp_path)
    results = root / RESULT_RELATIVE_DIR
    results.mkdir(parents=True)
    target = 2030365
    evaluation_path = results / f"{target}.json"
    evaluation_path.write_bytes(canonical_json_bytes({"target_issue": target}))
    entry = ManifestChainEntry(
        path=root / MANIFEST_RELATIVE_DIR / f"{target}.json",
        payload=_synthetic_manifest(target, 364),
        sha256="a" * 64,
        confirmation_index=CONFIRMATION_ISSUE_COUNT,
        target_issue=target,
    )
    monkeypatch.setattr(
        monitor,
        "_validate_completed_chain_without_final_seal",
        lambda *args, **kwargs: (
            (entry, {}, np.asarray(_draw_for_period(364), dtype=np.int64)),
        ),
    )
    seal_path, _, seal = _write_local_final_seal(
        root=root, results_dir=results, target_issue=target
    )
    seal["seal_pr_number"] = 12345
    seal_path.write_bytes(canonical_json_bytes(seal))
    client = _SealClient(evaluation_path.read_bytes())
    with pytest.raises(ValueError, match="内容SHA-256不匹配"):
        build_formal_summary(
            manifest_dir=root / MANIFEST_RELATIVE_DIR,
            results_dir=results,
            config=config,
            official_issues=np.asarray([target], dtype=np.int64),
            official_draws=np.asarray([_draw_for_period(364)], dtype=np.int64),
            seal_client=client,
        )
    assert client.pull_request_calls == 0
    assert client.file_calls == 0


def test_complete_summary_recomputes_secondary_metrics_and_primary_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _, _, _, _, config = _prepare_project(tmp_path)
    paired: list[tuple[ManifestChainEntry, dict[str, Any], IntArray]] = []
    results_dir = root / RESULT_RELATIVE_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    official_issues: list[int] = []
    official_draws: list[IntArray] = []
    for offset in range(CONFIRMATION_ISSUE_COUNT):
        target = 2030001 + offset
        manifest = _synthetic_manifest(target, offset)
        actual = np.asarray(_draw_for_period(offset), dtype=np.int64)
        metrics = calculate_manifest_metrics(manifest, actual)
        digest = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        entry = ManifestChainEntry(
            path=tmp_path / f"{target}.json",
            payload=manifest,
            sha256=digest,
            confirmation_index=offset + 1,
            target_issue=target,
        )
        record: dict[str, Any] = {
            "actual_numbers": sorted(map(int, actual)),
            "changepoint_state": "high_change" if offset % 5 == 0 else "normal",
            "models": metrics,
            "comparisons": calculate_evaluation_comparisons(metrics),
        }
        (results_dir / f"{target}.json").write_bytes(canonical_json_bytes(record))
        paired.append((entry, record, actual))
        official_issues.append(target)
        official_draws.append(actual)
    monkeypatch.setattr(
        monitor,
        "_validate_completed_chain_without_final_seal",
        lambda *args, **kwargs: tuple(paired),
    )
    final_target = int(official_issues[-1])
    _, final_evaluation, _ = _write_local_final_seal(
        root=root, results_dir=results_dir, target_issue=final_target
    )
    client = _SealClient(
        final_evaluation.read_bytes(),
        merged_at="2028-01-01T00:00:00Z",
        merge_commit="c" * 40,
    )
    summary = build_formal_summary(
        manifest_dir=root / MANIFEST_RELATIVE_DIR,
        results_dir=results_dir,
        config=config,
        official_issues=np.asarray(official_issues, dtype=np.int64),
        official_draws=np.asarray(official_draws, dtype=np.int64),
        seal_client=client,
    )
    assert client.pull_request_calls == 1
    assert client.file_calls == 1
    assert summary["issue_count"] == 365
    primary = cast(dict[str, Any], summary["primary_inference"])
    assert set(primary["comparisons"]) == set(PRIMARY_COMPARISON_MODELS)
    assert primary["block_length"] == 30
    assert primary["resample_count"] == 20000
    assert primary["seed"] == 20260717
    secondary = cast(dict[str, Any], summary["secondary_metrics"])
    assert secondary["status"] == "descriptive_not_for_model_selection"
    assert set(secondary["models"]) == set(PROBABILITY_STRATEGIES)
    assert len(secondary["issue_index"]) == 365
    assert secondary["high_change_trigger_count"] == 73
    for model in secondary["models"].values():
        assert len(model["calibration"]["bins"]) == 10
        assert set(model["mean_top_k_hits"]) == {str(k) for k in range(1, 11)}
    first_metrics = cast(dict[str, Any], paired[0][1]["models"])
    first_dynamic = cast(dict[str, Any], first_metrics[DYNAMIC_STRATEGY])
    assert first_dynamic["brier_score"] >= 0.0


def test_moving_block_bootstrap_and_holm_are_reproducible() -> None:
    values = np.linspace(-0.002, 0.001, 365, dtype=np.float64)
    first = moving_block_bootstrap_inference(values)
    second = moving_block_bootstrap_inference(values)
    assert first == second
    assert first.observed_mean == pytest.approx(float(values.mean()))
    assert 0.0 <= first.raw_p_value <= 1.0
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


def test_freeze_contract_rejects_any_config_or_source_drift(tmp_path: Path) -> None:
    root, _, config_path, _, _, _, config = _prepare_project(tmp_path)
    load_and_verify_freeze_config(root, config_path)
    changed = copy.deepcopy(config)
    changed["confirmation_protocol"]["bootstrap"]["block_length"] = 31
    _finish_config(changed)
    config_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="主要检验"):
        load_and_verify_freeze_config(root, config_path)
    config_path.write_bytes(canonical_json_bytes(config))
    source = root / FROZEN_SOURCE_PATHS[0]
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="冻结源码漂移"):
        load_and_verify_freeze_config(root, config_path)


def test_paths_and_unit_io_are_confined_to_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_prediction(monkeypatch)
    real_data = ROOT / DATA_RELATIVE_PATH
    before = (
        hashlib.sha256(real_data.read_bytes()).hexdigest()
        if real_data.exists()
        else None
    )
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    _write_test_manifest(root, data, config, issues, draws, target)
    assert tmp_path.resolve() in data.resolve().parents
    assert (
        require_contract_path(
            root, MANIFEST_RELATIVE_DIR, MANIFEST_RELATIVE_DIR, "manifest目录"
        )
        == (root / MANIFEST_RELATIVE_DIR).resolve()
    )
    with pytest.raises(ValueError, match="必须固定"):
        require_contract_path(
            root, Path("reports/wrong"), MANIFEST_RELATIVE_DIR, "manifest目录"
        )
    with pytest.raises(ValueError, match="项目目录内"):
        require_contract_path(
            root, root.parent / "escape", RESULT_RELATIVE_DIR, "结果目录"
        )
    after = (
        hashlib.sha256(real_data.read_bytes()).hexdigest()
        if real_data.exists()
        else None
    )
    assert after == before


def test_v1_and_existing_v2_files_remain_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in IMMUTABLE_PATHS
    }
    _install_fake_prediction(monkeypatch)
    root, data, config, issues, draws, target, _ = _prepare_project(tmp_path)
    _write_test_manifest(root, data, config, issues, draws, target)
    after = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in IMMUTABLE_PATHS
    }
    assert after == before
