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
    FIXED_HIGH_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    FREEZE_ACTIVE,
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
    build_formal_summary,
    build_manifest,
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
        repository: str = GITHUB_REPOSITORY,
    ) -> None:
        self.manifest_bytes = manifest_bytes
        self.state = state
        self.base = base
        self.merged_at = merged_at
        self.repository = repository

    def get_pull_request(self, repository: str, pr_number: int) -> Mapping[str, object]:
        del repository
        return {
            "repository": self.repository,
            "number": pr_number,
            "url": f"https://github.com/{GITHUB_REPOSITORY}/pull/{pr_number}",
            "state": self.state,
            "baseRefName": self.base,
            "mergedAt": self.merged_at,
            "mergeCommit": {"oid": "b" * 40},
        }

    def get_file_bytes(self, repository: str, path: str, commit_sha: str) -> bytes:
        del repository, path, commit_sha
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


def test_pending_status_blocks_all_three_production_commands(tmp_path: Path) -> None:
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
    for command, args in (
        (prospective_cli._manifest_command, manifest_args),
        (prospective_cli._evaluate_command, evaluate_args),
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
    evaluation = {
        "evidence_status": EVALUATION_EVIDENCE_STATUS,
        "freeze_id": config["freeze_id"],
        "target_issue": target,
        "confirmation_index": 1,
        "remote_preseal_verified": True,
        "sealed_before_official_result": True,
        "manifest_sha256_at_merge": raw_sha256(first_path),
        "manifest": {"sha256": raw_sha256(first_path)},
    }
    results = root / RESULT_RELATIVE_DIR
    results.mkdir(parents=True)
    (results / f"{target}.json").write_bytes(canonical_json_bytes(evaluation))
    second = _build_test_manifest(root, data, config_path, issues, draws, target + 1)
    assert second["confirmation_index"] == 2
    assert second["protocol_start_target_issue"] == target
    assert second["previous_target_issue"] == target
    assert second["previous_manifest_sha256"] == raw_sha256(first_path)
    second["previous_manifest_sha256"] = "0" * 64
    write_manifest_exclusive(second, root / MANIFEST_RELATIVE_DIR)
    with pytest.raises(ValueError, match="SHA-256链断裂"):
        next_chain_position(
            manifest_dir=root / MANIFEST_RELATIVE_DIR,
            results_dir=results,
            config=config,
            target_issue=target + 2,
        )


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


def test_summary_rejects_missing_manifest_or_evaluation(tmp_path: Path) -> None:
    _, _, _, _, _, _, config = _prepare_project(tmp_path)
    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    manifests.mkdir()
    results.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="365份manifest"):
        build_formal_summary(manifest_dir=manifests, results_dir=results, config=config)

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
            "previous_manifest_sha256": previous_digest,
            "previous_target_issue": previous_target,
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
        build_formal_summary(manifest_dir=manifests, results_dir=results, config=config)
    with pytest.raises(ValueError, match="365期确认链已满"):
        next_chain_position(
            manifest_dir=manifests,
            results_dir=results,
            config=config,
            target_issue=first_target + CONFIRMATION_ISSUE_COUNT,
        )


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


def test_complete_summary_recomputes_secondary_metrics_and_primary_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, _, _, config = _prepare_project(tmp_path)
    paired: list[tuple[ManifestChainEntry, dict[str, Any]]] = []
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
        }
        paired.append((entry, record))
    monkeypatch.setattr(
        monitor, "_validate_completed_chain", lambda *args: tuple(paired)
    )
    summary = build_formal_summary(
        manifest_dir=tmp_path / "manifests",
        results_dir=tmp_path / "results",
        config=config,
    )
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
