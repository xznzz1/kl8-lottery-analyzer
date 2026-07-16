"""冻结参数后的快乐8真正前瞻监测。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.analysis.feature_enhancer import PCA
from src.config import DIRICHLET_CONFIG

from .evaluation import evaluate_indices, load_history_csv
from .prizes import PrizeScenario
from .statistics import summarise_period_records, summarise_seed_ensemble
from .strategies import (
    DETERMINISTIC_STRATEGIES,
    REPOSITORY_ADVANCED_PARAMETERS,
    StrategyParameters,
    assert_legal_tickets,
    deterministic_tickets,
    score_numbers,
)

EXPECTED_WINDOWS_ROOT = Path(r"D:\lottery\kl8-lottery-analyzer")
EXPECTED_FROZEN_THROUGH_ISSUE = 2026186
EXPECTED_ROLLING_WINDOW = 120
EXPECTED_DECAY = 0.99
EXPECTED_HYBRID_WEIGHTS = (0.25, 0.50, 0.25)
EXPECTED_RANDOM_SEEDS = tuple(range(202601, 202621))
EXPECTED_PRIZE_SCENARIO = "explicit_rule_cap_scenario_not_actual_payout"
EXPECTED_IMPLEMENTATION_BASELINE_COMMIT = "25ee7c410c3d2916b81f32dfac7afa6b6b44202f"
EXPECTED_ENVIRONMENT = {
    "python": "3.11",
    "numpy": "1.26.4",
    "pandas": "3.0.3",
    "scikit_learn": "1.9.0",
}
SOURCE_MANIFEST_ALGORITHM = "sha256_utf8_normalized_lf"
EXPECTED_SOURCE_FILES = (
    "config/config.yaml",
    "src/analysis/feature_enhancer.py",
    "src/config.py",
    "src/scientific/evaluation.py",
    "src/scientific/prizes.py",
    "src/scientific/strategies.py",
)
EXPECTED_FIRST_PROSPECTIVE_ISSUE = 2026187
EXPECTED_FIRST_PRESEALED_CANDIDATE_ISSUE = 2026188

RECORD_COLUMNS = (
    "issue",
    "strategy",
    "seed",
    "play",
    "ticket_mode",
    "ticket1",
    "ticket2",
    "hits1",
    "hits2",
    "prize1",
    "prize2",
    "total_prize",
)
RECORD_KEY_COLUMNS = ("issue", "strategy", "seed", "play", "ticket_mode")
CI_SUFFIXES = ("_ci95_low", "_ci95_high")
OUTPUT_STATUS = "frozen_strategy_output_not_prediction_or_validity_evidence"


@dataclass(frozen=True)
class ScientificFreeze:
    """从版本化JSON加载并校验的冻结契约。"""

    frozen_through_issue: int
    rolling_window: int
    decay: float
    hybrid_weights: tuple[float, float, float]
    random_seeds: tuple[int, ...]
    source_manifest_files: tuple[tuple[str, str], ...]
    source_manifest_fingerprint: str
    prize_scenario_label: str
    original_data_path: str
    original_earliest_issue: int
    original_latest_issue: int
    original_record_count: int
    original_csv_sha256: str
    original_canonical_sha256: str
    report_path: str
    report_normalized_sha256: str
    final_holdout_path: str
    final_holdout_normalized_sha256: str
    selected_parameters_path: str
    selected_parameters_normalized_sha256: str
    next_issue: int
    first_fully_presealed_candidate_issue: int
    fingerprint: str

    @property
    def strategy_parameters(self) -> StrategyParameters:
        """返回不可变的前瞻策略参数。"""

        return StrategyParameters(
            rolling_window=self.rolling_window,
            decay=self.decay,
            hybrid_weights=self.hybrid_weights,
        )


@dataclass(frozen=True)
class ProspectivePaths:
    """前瞻运行允许访问的固定路径。"""

    project_root: Path
    data: Path
    freeze_config: Path
    output_dir: Path
    records: Path
    summary: Path
    candidates: Path
    report: Path
    manifests_dir: Path
    temp_root: Path
    frozen_report: Path
    frozen_final_holdout: Path
    frozen_selected_parameters: Path


@dataclass(frozen=True)
class ProspectiveRunResult:
    """单次前瞻运行的可审计结果。"""

    prospective_issues: tuple[int, ...]
    records: int
    summary_rows: int
    candidate_rows: int
    next_issue: int
    candidate_manifest: Path


def _as_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"冻结配置字段{label}必须是对象")
    return cast(dict[str, object], value)


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"冻结配置字段{label}必须是数组")
    return cast(list[object], value)


def _canonical_json_sha256(payload: dict[str, object]) -> str:
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalised_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify_source_manifest(
    payload: dict[str, object], project_root: Path
) -> tuple[tuple[tuple[str, str], ...], str]:
    """校验冻结策略源码；任何漂移都要求建立新的策略版本。"""

    manifest = _as_dict(payload.get("source_manifest"), "source_manifest")
    if manifest.get("algorithm") != SOURCE_MANIFEST_ALGORITHM:
        raise ValueError("source_manifest算法与冻结契约不一致")
    raw_files = _as_dict(manifest.get("files"), "source_manifest.files")
    if set(raw_files) != set(EXPECTED_SOURCE_FILES):
        raise ValueError(
            "source_manifest文件集合与冻结契约不一致；必须建立新的策略版本"
        )
    expected_files = tuple(
        sorted(
            (
                relative,
                _require_sha256(
                    raw_files[relative], f"source_manifest.files.{relative}"
                ),
            )
            for relative in EXPECTED_SOURCE_FILES
        )
    )
    fingerprint_payload: dict[str, object] = {
        "algorithm": SOURCE_MANIFEST_ALGORITHM,
        "files": dict(expected_files),
    }
    configured_fingerprint = _require_sha256(
        manifest.get("fingerprint"), "source_manifest.fingerprint"
    )
    computed_fingerprint = _canonical_json_sha256(fingerprint_payload)
    if configured_fingerprint != computed_fingerprint:
        raise ValueError("source_manifest指纹不一致；必须建立新的策略版本，拒绝继续")

    root = project_root.resolve()
    for relative, expected_hash in expected_files:
        source_path = (root / relative).resolve()
        if root not in source_path.parents or not source_path.is_file():
            raise ValueError(
                f"冻结策略源码缺失或越界：{relative}；必须建立新的策略版本"
            )
        actual_hash = _normalised_text_sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"冻结策略源码已变化：{relative}；必须建立新的策略版本，" "不能静默继续"
            )
    return expected_files, configured_fingerprint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"冻结配置字段{label}不是合法SHA-256")
    return text


def _as_int(value: object) -> int:
    return int(str(value))


def _as_float(value: object) -> float:
    return float(str(value))


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"冻结配置字段{label}必须是布尔值")
    return value


def load_scientific_freeze(path: Path) -> ScientificFreeze:
    """读取冻结配置并拒绝任何策略参数漂移。"""

    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if _as_int(payload.get("schema_version", 0)) != 1:
        raise ValueError("不支持的科学冻结配置版本")
    if payload.get("freeze_status") != "viewed_and_permanently_frozen":
        raise ValueError("final holdout必须标记为已查看且永久冻结")

    # 必须先校验源码，再读取运行参数或执行任何前瞻计算。
    source_manifest_files, source_manifest_fingerprint = _verify_source_manifest(
        payload, path.resolve().parent.parent
    )

    frozen_through = _as_int(payload["frozen_through_issue"])
    rolling_window = _as_int(payload["rolling_window"])
    decay = _as_float(payload["decay"])
    hybrid_weights = tuple(
        _as_float(value)
        for value in _as_sequence(payload["hybrid_weights"], "hybrid_weights")
    )
    random_seeds = tuple(
        _as_int(value)
        for value in _as_sequence(payload["random_seeds"], "random_seeds")
    )
    if frozen_through != EXPECTED_FROZEN_THROUGH_ISSUE:
        raise ValueError("frozen_through_issue与永久冻结契约不一致")
    if rolling_window != EXPECTED_ROLLING_WINDOW:
        raise ValueError("rolling_window已偏离冻结值120")
    if not math.isclose(decay, EXPECTED_DECAY, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("decay已偏离冻结值0.99")
    if hybrid_weights != EXPECTED_HYBRID_WEIGHTS:
        raise ValueError("hybrid_weights已偏离冻结值(0.25, 0.50, 0.25)")
    if random_seeds != EXPECTED_RANDOM_SEEDS:
        raise ValueError("random_seeds已偏离冻结的20个seed")

    scenario = _as_dict(payload["prize_scenario"], "prize_scenario")
    scenario_label = str(scenario["label"])
    if (
        scenario.get("mode") != "cap_scenario"
        or scenario_label != EXPECTED_PRIZE_SCENARIO
    ):
        raise ValueError("奖金情景已偏离冻结的显式封顶情景")

    advanced = _as_dict(payload["repository_advanced"], "repository_advanced")
    if (
        advanced.get("implementation_baseline_commit")
        != EXPECTED_IMPLEMENTATION_BASELINE_COMMIT
    ):
        raise ValueError("repository_advanced实现基线提交与冻结契约不一致")
    expected_advanced: dict[str, object] = {
        "recent_window": REPOSITORY_ADVANCED_PARAMETERS.recent_window,
        "reference_window": REPOSITORY_ADVANCED_PARAMETERS.reference_window,
        "decay": REPOSITORY_ADVANCED_PARAMETERS.decay,
        "feature_weights": list(REPOSITORY_ADVANCED_PARAMETERS.feature_weights),
        "dirichlet_weight": REPOSITORY_ADVANCED_PARAMETERS.dirichlet_weight,
        "pca_components": REPOSITORY_ADVANCED_PARAMETERS.pca_components,
        "use_pca": REPOSITORY_ADVANCED_PARAMETERS.use_pca,
        "use_graph_embeddings": REPOSITORY_ADVANCED_PARAMETERS.use_graph_embeddings,
    }
    for key, expected in expected_advanced.items():
        actual = advanced.get(key)
        if isinstance(expected, float):
            if not math.isclose(
                _as_float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"repository_advanced.{key}已偏离冻结值")
        elif isinstance(expected, list):
            values = [
                _as_float(value)
                for value in _as_sequence(actual, f"repository_advanced.{key}")
            ]
            if values != expected:
                raise ValueError(f"repository_advanced.{key}已偏离冻结值")
        elif actual != expected:
            raise ValueError(f"repository_advanced.{key}已偏离冻结值")
    frozen_dirichlet = _as_dict(
        advanced["dirichlet_config"], "repository_advanced.dirichlet_config"
    )
    for key in ("prior_strength", "window_size", "variance_weight"):
        if not math.isclose(
            _as_float(frozen_dirichlet[key]),
            float(DIRICHLET_CONFIG[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"运行环境Dirichlet配置与冻结值不一致：{key}")
    if _as_bool(advanced["use_pca"], "repository_advanced.use_pca") and PCA is None:
        raise RuntimeError(
            "冻结的repository_advanced要求scikit-learn PCA，当前环境不可用"
        )
    frozen_environment = _as_dict(
        advanced["environment_observed"], "repository_advanced.environment_observed"
    )
    configured_environment = {
        key: str(frozen_environment.get(key)) for key in EXPECTED_ENVIRONMENT
    }
    if configured_environment != EXPECTED_ENVIRONMENT:
        raise ValueError("repository_advanced环境版本已偏离冻结契约")
    runtime_environment = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": metadata.version("scikit-learn"),
    }
    if runtime_environment != EXPECTED_ENVIRONMENT:
        raise RuntimeError(f"运行环境版本与冻结契约不一致：{runtime_environment}")

    snapshot = _as_dict(payload["original_data_snapshot"], "original_data_snapshot")
    if snapshot.get("path") != "data_cache/kl8/data.csv":
        raise ValueError("原始数据快照路径与冻结契约不一致")
    artifacts = _as_dict(payload["frozen_artifacts"], "frozen_artifacts")
    report = _as_dict(artifacts["scientific_report"], "scientific_report")
    final_holdout = _as_dict(
        artifacts["final_holdout_results"], "final_holdout_results"
    )
    selected = _as_dict(artifacts["selected_parameters"], "selected_parameters")
    policy = _as_dict(payload["prospective_policy"], "prospective_policy")
    if _as_int(policy.get("first_target_issue", 0)) != EXPECTED_FIRST_PROSPECTIVE_ISSUE:
        raise ValueError("prospective_policy.first_target_issue与冻结契约不一致")
    next_issue = _as_int(policy.get("next_issue", 0))
    first_presealed_issue = _as_int(
        policy.get("first_fully_presealed_candidate_issue", 0)
    )
    if first_presealed_issue != EXPECTED_FIRST_PRESEALED_CANDIDATE_ISSUE:
        raise ValueError("首个完整预先封存候选期号与冻结契约不一致")
    if next_issue <= frozen_through:
        raise ValueError("prospective_policy.next_issue必须晚于冻结截止期")
    return ScientificFreeze(
        frozen_through_issue=frozen_through,
        rolling_window=rolling_window,
        decay=decay,
        hybrid_weights=hybrid_weights,
        random_seeds=random_seeds,
        source_manifest_files=source_manifest_files,
        source_manifest_fingerprint=source_manifest_fingerprint,
        prize_scenario_label=scenario_label,
        original_data_path=str(snapshot["path"]),
        original_earliest_issue=_as_int(snapshot["earliest_issue"]),
        original_latest_issue=_as_int(snapshot["latest_issue"]),
        original_record_count=_as_int(snapshot["record_count"]),
        original_csv_sha256=_require_sha256(snapshot["csv_sha256"], "csv_sha256"),
        original_canonical_sha256=_require_sha256(
            snapshot["canonical_records_sha256"], "canonical_records_sha256"
        ),
        report_path=str(report["path"]),
        report_normalized_sha256=_require_sha256(
            report["sha256_utf8_normalized_lf"], "scientific_report.sha256"
        ),
        final_holdout_path=str(final_holdout["path"]),
        final_holdout_normalized_sha256=_require_sha256(
            final_holdout["sha256_utf8_normalized_lf"],
            "final_holdout_results.sha256",
        ),
        selected_parameters_path=str(selected["path"]),
        selected_parameters_normalized_sha256=_require_sha256(
            selected["sha256_utf8_normalized_lf"], "selected_parameters.sha256"
        ),
        next_issue=next_issue,
        first_fully_presealed_candidate_issue=first_presealed_issue,
        fingerprint=_canonical_json_sha256(payload),
    )


def validate_runtime_storage(project_root: Path) -> None:
    """Windows自动运行必须位于指定D盘仓库并使用仓库虚拟环境。"""

    root = project_root.resolve()
    if os.name != "nt":
        return
    if root != EXPECTED_WINDOWS_ROOT.resolve():
        raise RuntimeError(f"Windows仓库根必须为{EXPECTED_WINDOWS_ROOT}，当前为{root}")
    expected_python = (root / ".venv" / "Scripts" / "python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"必须使用项目虚拟环境：{expected_python}")


def _resolve_exact_path(
    project_root: Path, raw: str | Path, expected_relative: str, label: str
) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    expected = (project_root / expected_relative).resolve()
    root = project_root.resolve()
    if resolved != expected or (resolved != root and root not in resolved.parents):
        raise ValueError(f"{label}固定为{expected}，拒绝路径：{resolved}")
    return resolved


def resolve_prospective_paths(
    project_root: Path,
    *,
    data: str | Path = "data_cache/kl8/data.csv",
    freeze_config: str | Path = "config/scientific_freeze.json",
    output_dir: str | Path = "results/prospective",
    report: str | Path = "reports/kl8_prospective_report.md",
    temp_root: Path | None = None,
) -> ProspectivePaths:
    """把所有输入输出锁定到约定的仓库内边界。"""

    root = project_root.resolve()
    resolved_output = _resolve_exact_path(
        root, output_dir, "results/prospective", "结果目录"
    )
    resolved_report = _resolve_exact_path(
        root, report, "reports/kl8_prospective_report.md", "前瞻报告"
    )
    resolved_manifests = (root / "reports" / "prospective_manifests").resolve()
    if root not in resolved_manifests.parents:
        raise ValueError("候选封存目录必须位于仓库reports目录内")
    resolved_temp = (temp_root or root.parent / ".tmp").resolve()
    expected_temp = (root.parent / ".tmp").resolve()
    if resolved_temp != expected_temp:
        raise ValueError(f"临时目录固定为{expected_temp}")
    return ProspectivePaths(
        project_root=root,
        data=_resolve_exact_path(root, data, "data_cache/kl8/data.csv", "历史数据"),
        freeze_config=_resolve_exact_path(
            root, freeze_config, "config/scientific_freeze.json", "冻结配置"
        ),
        output_dir=resolved_output,
        records=resolved_output / "prospective_records.csv",
        summary=resolved_output / "prospective_summary.csv",
        candidates=resolved_output / "next_issue_candidates.csv",
        report=resolved_report,
        manifests_dir=resolved_manifests,
        temp_root=resolved_temp,
        frozen_report=root / "reports" / "kl8_scientific_report.md",
        frozen_final_holdout=root
        / "results"
        / "scientific"
        / "final_holdout_results.csv",
        frozen_selected_parameters=root
        / "results"
        / "scientific"
        / "selected_parameters.json",
    )


def validate_prospective_paths_integrity(paths: ProspectivePaths) -> None:
    """拒绝绕过resolver手工构造的越界或覆盖路径。"""

    expected = resolve_prospective_paths(
        paths.project_root,
        temp_root=paths.temp_root,
    )
    if paths != expected:
        mismatches = [
            field
            for field in ProspectivePaths.__dataclass_fields__
            if getattr(paths, field) != getattr(expected, field)
        ]
        raise ValueError(f"前瞻路径契约被绕过：{', '.join(mismatches)}")


def canonical_history_sha256(issues: np.ndarray, draws: np.ndarray) -> str:
    """对升序期号与20个号码生成跨平台规范化SHA-256。"""

    header = ["期数", *(f"红球_{index}" for index in range(1, 21))]
    rows = [",".join(header)]
    for issue, draw in zip(issues, draws):
        rows.append(",".join(map(str, (int(issue), *map(int, draw)))))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def verify_frozen_inputs(
    freeze: ScientificFreeze,
    paths: ProspectivePaths,
    issues: np.ndarray,
    draws: np.ndarray,
) -> None:
    """在写入前验证旧报告、旧结果（若存在）和冻结数据前缀。"""

    expected_pairs = (
        (
            freeze.report_path,
            paths.frozen_report,
            freeze.report_normalized_sha256,
            True,
        ),
        (
            freeze.final_holdout_path,
            paths.frozen_final_holdout,
            freeze.final_holdout_normalized_sha256,
            False,
        ),
        (
            freeze.selected_parameters_path,
            paths.frozen_selected_parameters,
            freeze.selected_parameters_normalized_sha256,
            False,
        ),
    )
    for configured, actual_path, expected_hash, required in expected_pairs:
        configured_path = (paths.project_root / configured).resolve()
        if configured_path != actual_path.resolve():
            raise ValueError(f"冻结产物路径与契约不一致：{configured}")
        if not actual_path.exists():
            if required:
                raise FileNotFoundError(f"冻结科学报告不存在：{actual_path}")
            continue
        actual_hash = _normalised_text_sha256(actual_path)
        if actual_hash != expected_hash:
            raise ValueError(f"冻结产物内容已变化，拒绝前瞻运行：{actual_path.name}")

    if paths.report.resolve() == paths.frozen_report.resolve():
        raise ValueError("前瞻报告不得覆盖冻结科学报告")
    if paths.output_dir.resolve() == paths.frozen_final_holdout.parent.resolve():
        raise ValueError("前瞻结果不得写入冻结科学结果目录")

    frozen_mask = issues <= freeze.frozen_through_issue
    frozen_issues = issues[frozen_mask]
    frozen_draws = draws[frozen_mask]
    if len(frozen_issues) != freeze.original_record_count:
        raise ValueError("冻结数据前缀期数与原快照不一致")
    if (
        int(frozen_issues[0]) != freeze.original_earliest_issue
        or int(frozen_issues[-1]) != freeze.original_latest_issue
    ):
        raise ValueError("冻结数据前缀期号范围与原快照不一致")
    if (
        canonical_history_sha256(frozen_issues, frozen_draws)
        != freeze.original_canonical_sha256
    ):
        raise ValueError("冻结数据前缀内容与原快照冲突")


def _normalise_seed(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    numeric = float(text)
    if not numeric.is_integer():
        raise ValueError(f"seed必须为整数：{value}")
    return str(int(numeric))


def normalise_records(records: pd.DataFrame) -> pd.DataFrame:
    """规范记录类型、顺序和票面合法性。"""

    missing = set(RECORD_COLUMNS) - set(records.columns)
    extra = set(records.columns) - set(RECORD_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"前瞻记录字段不一致，缺少={sorted(missing)}，多出={sorted(extra)}"
        )
    frame = records.loc[:, RECORD_COLUMNS].copy()
    for column in ("issue", "play", "hits1", "hits2"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("prize1", "prize2", "total_prize"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    for column in ("strategy", "ticket_mode", "ticket1", "ticket2"):
        frame[column] = frame[column].astype(str)
    frame["seed"] = frame["seed"].map(_normalise_seed)
    if frame.duplicated(list(RECORD_KEY_COLUMNS)).any():
        raise ValueError("前瞻记录主键重复")
    for row in frame.itertuples(index=False):
        tickets = (
            tuple(int(value) for value in str(row.ticket1).split()),
            tuple(int(value) for value in str(row.ticket2).split()),
        )
        assert_legal_tickets(tickets, int(row.play), str(row.ticket_mode))
    return frame.sort_values(list(RECORD_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _validate_record_grid(records: pd.DataFrame, freeze: ScientificFreeze) -> None:
    for issue, group in records.groupby("issue", sort=True):
        expected_keys = {
            (strategy, "", play, mode)
            for strategy in DETERMINISTIC_STRATEGIES
            for play in range(1, 11)
            for mode in ("disjoint", "independent")
        }
        expected_keys.update(
            {
                ("uniform_random", str(seed), play, mode)
                for seed in freeze.random_seeds
                for play in range(1, 11)
                for mode in ("disjoint", "independent")
            }
        )
        actual_keys = set(
            group[["strategy", "seed", "play", "ticket_mode"]].itertuples(
                index=False, name=None
            )
        )
        if actual_keys != expected_keys:
            raise ValueError(f"期号{int(issue)}的前瞻记录网格不完整")


def evaluate_new_issues(
    issues: np.ndarray,
    draws: np.ndarray,
    freeze: ScientificFreeze,
) -> pd.DataFrame:
    """只评估冻结截止期之后的数据，并逐目标截断未来数组。"""

    if freeze.frozen_through_issue not in set(map(int, issues)):
        raise ValueError("历史数据缺少冻结截止期号")
    target_indices = np.flatnonzero(issues > freeze.frozen_through_issue)
    if len(target_indices) == 0:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    scenario = PrizeScenario.cap_scenario()
    if scenario.label != freeze.prize_scenario_label:
        raise ValueError("运行时奖金情景与冻结配置不一致")

    batches: list[pd.DataFrame] = []
    for raw_index in target_indices:
        target_index = int(raw_index)
        # 每次只把截至目标期的数组交给评估器，未来期开奖在调用边界之外。
        batches.append(
            evaluate_indices(
                issues[: target_index + 1],
                draws[: target_index + 1],
                [target_index],
                freeze.strategy_parameters,
                scenario,
                random_seeds=freeze.random_seeds,
            )
        )
    records = normalise_records(pd.concat(batches, ignore_index=True))
    if (records["issue"] <= freeze.frozen_through_issue).any():
        raise AssertionError("前瞻记录包含冻结区间")
    _validate_record_grid(records, freeze)
    return records


def reconcile_records(
    expected: pd.DataFrame, existing_path: Path, freeze: ScientificFreeze
) -> pd.DataFrame:
    """重算校验已有记录，仅追加全新的目标期。"""

    expected = normalise_records(expected)
    if not existing_path.exists():
        return expected
    existing = normalise_records(
        pd.read_csv(existing_path, keep_default_na=False, dtype={"seed": str})
    )
    if (existing["issue"] <= freeze.frozen_through_issue).any():
        raise ValueError("已有前瞻记录错误包含冻结期或更早数据")
    _validate_record_grid(existing, freeze)

    expected_indexed = expected.set_index(list(RECORD_KEY_COLUMNS)).sort_index()
    existing_indexed = existing.set_index(list(RECORD_KEY_COLUMNS)).sort_index()
    missing_now = existing_indexed.index.difference(expected_indexed.index)
    if len(missing_now):
        raise ValueError("已有前瞻记录在当前数据中缺失，可能发生数据倒退或配置冲突")
    recomputed = expected_indexed.loc[existing_indexed.index]
    if not existing_indexed.equals(recomputed):
        differences = existing_indexed.ne(recomputed)
        difference_index = differences.stack().loc[lambda values: values].index[0]
        *key_values, first_column = difference_index
        first_key = tuple(key_values)
        raise ValueError(
            f"已有前瞻记录与重算结果冲突：key={first_key}，字段={first_column}"
        )
    combined = pd.concat(
        [
            existing,
            expected_indexed.loc[
                expected_indexed.index.difference(existing_indexed.index)
            ].reset_index(),
        ],
        ignore_index=True,
    )
    combined = normalise_records(combined)
    _validate_record_grid(combined, freeze)
    return combined


def _summary_row(
    source: dict[str, float | int],
    *,
    strategy: str,
    play: int,
    ticket_mode: str,
    summary_level: str,
    seed_count: int,
    prize_scenario: str,
) -> dict[str, object]:
    name_map = {
        "mean_hits_per_bet": "observed_mean_hits_per_bet",
        "any_prize_probability": "observed_any_prize_rate",
        "profit_probability": "observed_profit_rate",
        "expected_prize": "realized_mean_prize_yuan",
        "roi": "observed_mean_roi",
        "prize_at_least_100_probability": "observed_prize_at_least_100_rate",
        "prize_at_least_1000_probability": "observed_prize_at_least_1000_rate",
        "prize_at_least_10000_probability": "observed_prize_at_least_10000_rate",
        "max_drawdown": "max_drawdown_yuan",
        "longest_losing_streak": "longest_losing_streak_issues",
        "ending_profit": "ending_profit_yuan",
    }
    issue_count = int(source["issues"])
    row: dict[str, object] = {
        "strategy": strategy,
        "summary_level": summary_level,
        "seed_count": seed_count,
        "play": play,
        "ticket_mode": ticket_mode,
        "budget_per_issue_yuan": 4.0,
        "prize_scenario": prize_scenario,
        "n_issue_clusters": issue_count,
        "inference_status": (
            "not_run_single_issue"
            if issue_count == 1
            else "descriptive_only_no_hypothesis_test"
        ),
        "comparison_to_random_tested": False,
    }
    for source_name, output_name in name_map.items():
        row[output_name] = source[source_name]
        for suffix in CI_SUFFIXES:
            input_interval = f"{source_name}{suffix}"
            if input_interval in source:
                row[f"{output_name}{suffix}"] = (
                    np.nan if issue_count == 1 else source[input_interval]
                )
    row["interval_method"] = (
        "not_computed_single_issue"
        if issue_count == 1
        else (
            "issue_cluster_bootstrap"
            if summary_level == "random_seed_ensemble"
            else "wilson_events_and_issue_bootstrap_means"
        )
    )
    return row


def summarise_prospective_records(
    records: pd.DataFrame,
    freeze: ScientificFreeze,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """生成不含显著性检验和机会性排名的前瞻描述摘要。"""

    if records.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    deterministic = records[records["strategy"] != "uniform_random"]
    for group_index, ((strategy, play, ticket_mode), group) in enumerate(
        deterministic.groupby(["strategy", "play", "ticket_mode"], sort=True)
    ):
        source = summarise_period_records(
            group.sort_values("issue"),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20261870 + group_index,
        )
        rows.append(
            _summary_row(
                source,
                strategy=str(strategy),
                play=int(play),
                ticket_mode=str(ticket_mode),
                summary_level="deterministic_strategy",
                seed_count=0,
                prize_scenario=freeze.prize_scenario_label,
            )
        )

    random_records = records[records["strategy"] == "uniform_random"]
    for group_index, ((play, ticket_mode), group) in enumerate(
        random_records.groupby(["play", "ticket_mode"], sort=True)
    ):
        source = summarise_seed_ensemble(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20262870 + group_index,
        )
        rows.append(
            _summary_row(
                source,
                strategy="uniform_random",
                play=int(play),
                ticket_mode=str(ticket_mode),
                summary_level="random_seed_ensemble",
                seed_count=len(freeze.random_seeds),
                prize_scenario=freeze.prize_scenario_label,
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["strategy", "play", "ticket_mode"], kind="stable")
        .reset_index(drop=True)
    )


def generate_next_issue_candidates(
    issues: np.ndarray,
    draws: np.ndarray,
    freeze: ScientificFreeze,
    *,
    next_issue: int,
) -> pd.DataFrame:
    """用全部已到达历史生成下一期冻结策略票面，不作效果宣称。"""

    latest_issue = int(issues[-1])
    if next_issue <= latest_issue:
        raise ValueError("候选目标期必须晚于最新已知期开奖")
    history_fingerprint = canonical_history_sha256(issues, draws)
    rows: list[dict[str, object]] = []
    for strategy in DETERMINISTIC_STRATEGIES:
        scores = score_numbers(draws, strategy, freeze.strategy_parameters)
        for play in range(1, 11):
            for ticket_mode in ("disjoint", "independent"):
                tickets = deterministic_tickets(scores, play, ticket_mode)
                assert_legal_tickets(tickets, play, ticket_mode)
                rows.append(
                    {
                        "target_issue": next_issue,
                        "history_through_issue": latest_issue,
                        "history_issue_count": len(issues),
                        "strategy": strategy,
                        "play": play,
                        "ticket_mode": ticket_mode,
                        "ticket1": " ".join(map(str, tickets[0])),
                        "ticket2": " ".join(map(str, tickets[1])),
                        "overlap_count": len(set(tickets[0]) & set(tickets[1])),
                        "ticket_count": 2,
                        "budget_yuan": 4.0,
                        "freeze_fingerprint": freeze.fingerprint,
                        "history_fingerprint": history_fingerprint,
                        "output_status": OUTPUT_STATUS,
                    }
                )
    candidates = (
        pd.DataFrame(rows)
        .sort_values(["strategy", "play", "ticket_mode"], kind="stable")
        .reset_index(drop=True)
    )
    if len(candidates) != len(DETERMINISTIC_STRATEGIES) * 10 * 2:
        raise AssertionError("下一期候选票面网格不完整")
    return candidates


def _csv_text(frame: pd.DataFrame) -> str:
    stream = io.StringIO(newline="")
    frame.to_csv(
        stream,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    )
    return stream.getvalue()


def _build_candidate_manifest(
    *,
    freeze: ScientificFreeze,
    issues: np.ndarray,
    draws: np.ndarray,
    candidates: pd.DataFrame,
    generated_at_utc: str,
) -> dict[str, object]:
    """把完整候选CSV封装为可版本控制、可重建的规范化清单。"""

    candidate_text = _csv_text(candidates)
    candidate_records = list(csv.DictReader(io.StringIO(candidate_text)))
    if len(candidate_records) != 100:
        raise ValueError("候选封存清单必须恰好包含100行候选")
    return {
        "schema_version": 1,
        "target_issue": int(candidates["target_issue"].iloc[0]),
        "generated_at_utc": generated_at_utc,
        "history_through_issue": int(issues[-1]),
        "history_issue_count": len(issues),
        "history_canonical_sha256": canonical_history_sha256(issues, draws),
        "freeze_fingerprint": freeze.fingerprint,
        "source_manifest_fingerprint": freeze.source_manifest_fingerprint,
        "candidate_count": len(candidate_records),
        "next_issue_candidates_csv_normalized_sha256": hashlib.sha256(
            candidate_text.encode("utf-8")
        ).hexdigest(),
        "candidate_columns": list(candidates.columns),
        "candidates": candidate_records,
    }


def _verify_existing_candidate_seal(
    paths: ProspectivePaths,
    freeze: ScientificFreeze,
    upcoming_target_issue: int,
) -> None:
    """在覆盖当前候选CSV前，先用其所属期号的版本化manifest验封。"""

    if not paths.candidates.exists():
        return
    try:
        existing_candidates = pd.read_csv(
            paths.candidates, keep_default_na=False, dtype=str
        )
        target_values = {
            int(value) for value in existing_candidates["target_issue"].unique()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("已有next_issue_candidates.csv无法识别目标期号") from error
    if len(target_values) != 1:
        raise ValueError("已有next_issue_candidates.csv包含多个目标期号")
    existing_target = target_values.pop()
    existing_manifest_path = paths.manifests_dir / f"{existing_target}.json"
    if not existing_manifest_path.exists():
        if existing_target == upcoming_target_issue:
            # 兼容本功能上线前已生成但尚未封存的同一期候选；本轮将严格重算封存。
            return
        raise ValueError(
            f"已有期号{existing_target}候选尚无版本化manifest，禁止覆盖为新一期"
        )
    try:
        existing_manifest = _as_dict(
            json.loads(existing_manifest_path.read_text(encoding="utf-8")),
            "candidate_manifest",
        )
        expected_hash = _require_sha256(
            existing_manifest["next_issue_candidates_csv_normalized_sha256"],
            "candidate_manifest.candidate_csv_sha256",
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"期号{existing_target}的候选封存清单无法验证，禁止覆盖"
        ) from error
    normalised_candidate_text = (
        paths.candidates.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    candidate_records = list(csv.DictReader(io.StringIO(normalised_candidate_text)))
    generated_at_utc = str(existing_manifest.get("generated_at_utc", ""))
    try:
        generated_timestamp = datetime.fromisoformat(generated_at_utc)
    except ValueError as error:
        raise ValueError(
            f"期号{existing_target}的候选封存时间不是合法UTC时间，禁止覆盖"
        ) from error
    try:
        candidate_history_issues = {
            _as_int(row["history_through_issue"]) for row in candidate_records
        }
        candidate_history_counts = {
            _as_int(row["history_issue_count"]) for row in candidate_records
        }
        candidate_history_hashes = {
            str(row["history_fingerprint"]) for row in candidate_records
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"期号{existing_target}的候选CSV审计字段无效，禁止覆盖"
        ) from error
    if (
        _as_int(existing_manifest.get("schema_version", 0)) != 1
        or _as_int(existing_manifest.get("target_issue", 0)) != existing_target
        or _as_int(existing_manifest.get("candidate_count", 0)) != 100
        or len(_as_sequence(existing_manifest.get("candidates"), "candidates")) != 100
        or generated_timestamp.utcoffset() != timedelta(0)
        or existing_manifest.get("candidate_columns")
        != list(existing_candidates.columns)
        or existing_manifest.get("candidates") != candidate_records
        or candidate_history_issues
        != {_as_int(existing_manifest.get("history_through_issue", 0))}
        or candidate_history_counts
        != {_as_int(existing_manifest.get("history_issue_count", 0))}
        or candidate_history_hashes
        != {
            _require_sha256(
                existing_manifest.get("history_canonical_sha256"),
                "candidate_manifest.history_canonical_sha256",
            )
        }
        or existing_manifest.get("freeze_fingerprint") != freeze.fingerprint
        or existing_manifest.get("source_manifest_fingerprint")
        != freeze.source_manifest_fingerprint
    ):
        raise ValueError(
            f"期号{existing_target}的候选封存清单内容不一致或不完整，禁止覆盖"
        )
    if _normalised_text_sha256(paths.candidates) != expected_hash:
        raise ValueError(
            f"期号{existing_target}的next_issue_candidates.csv与候选封存清单"
            "哈希不一致，禁止覆盖"
        )


def _prepare_candidate_manifest(
    *,
    paths: ProspectivePaths,
    freeze: ScientificFreeze,
    issues: np.ndarray,
    draws: np.ndarray,
    candidates: pd.DataFrame,
) -> tuple[Path, dict[str, object], bool]:
    """首建manifest；已有manifest只读并逐字段核对，绝不覆盖。"""

    target_issue = int(candidates["target_issue"].iloc[0])
    _verify_existing_candidate_seal(paths, freeze, target_issue)
    manifest_path = (paths.manifests_dir / f"{target_issue}.json").resolve()
    if manifest_path.parent != paths.manifests_dir.resolve():
        raise ValueError("候选封存清单路径越界")

    existing: dict[str, object] | None = None
    if manifest_path.exists():
        try:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"已有候选封存清单无法解析：{manifest_path}") from error
        existing = _as_dict(parsed, "candidate_manifest")
        generated_at_utc = str(existing.get("generated_at_utc", ""))
        if not generated_at_utc:
            raise ValueError("已有候选封存清单缺少generated_at_utc，禁止覆盖")
    else:
        generated_at_utc = datetime.now(timezone.utc).isoformat()

    expected = _build_candidate_manifest(
        freeze=freeze,
        issues=issues,
        draws=draws,
        candidates=candidates,
        generated_at_utc=generated_at_utc,
    )
    if existing is not None and existing != expected:
        raise ValueError(
            f"期号{target_issue}的已有候选封存清单与重算结果不一致，"
            "禁止覆盖；请建立新的期号manifest"
        )
    if existing is not None and paths.candidates.exists():
        expected_hash = str(existing["next_issue_candidates_csv_normalized_sha256"])
        actual_hash = _normalised_text_sha256(paths.candidates)
        if actual_hash != expected_hash:
            raise ValueError(
                "next_issue_candidates.csv与已封存清单哈希不一致，" "禁止静默覆盖"
            )
    return manifest_path, expected, existing is None


def _create_text_exclusive(path: Path, text: str, temp_root: Path) -> None:
    """通过同卷硬链接只创建新文件，竞态下也不会覆盖既有manifest。"""

    temp_root.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=temp_root,
            prefix="kl8-manifest-",
            suffix=".tmp",
        ) as stream:
            stream.write(text)
            stream.flush()
            temporary = Path(stream.name)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"候选封存清单已存在，禁止覆盖：{path}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, text: str, temp_root: Path) -> None:
    temp_root.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=temp_root,
            prefix="kl8-prospective-",
            suffix=".tmp",
        ) as stream:
            stream.write(text)
            stream.flush()
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_prospective_report(
    *,
    freeze: ScientificFreeze,
    issues: np.ndarray,
    records: pd.DataFrame,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    data_sha256: str,
    candidate_manifest_path: Path,
    candidate_manifest: dict[str, object],
) -> str:
    """生成强调单期限制和非预测性质的前瞻报告。"""

    prospective_issues = tuple(sorted(map(int, records["issue"].unique())))
    issue_text = (
        "、".join(map(str, prospective_issues)) if prospective_issues else "暂无"
    )
    completeness = (
        records.groupby("strategy", sort=True).size().reset_index(name="records")
    )
    lines = ["| 策略 | 原始记录数 |", "| --- | ---: |"]
    lines.extend(
        f"| {row.strategy} | {int(row.records)} |"
        for row in completeness.itertuples(index=False)
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    single_issue_caveat = (
        "目前只有1个新增期，不能得出统计显著性结论。当前差异仅是单期描述；"
        "既不能证明任何策略优于随机，也不能证明策略与随机等效。"
        if len(prospective_issues) == 1
        else "本报告只提供前瞻描述性监测，未运行新的策略选择或显著性检验。"
    )
    return f"""# 快乐8冻结策略前瞻监测报告

## 执行摘要

冻结截止期号为 `{freeze.frozen_through_issue}`。本次只评估截止期之后已经真实到达的期开奖：`{issue_text}`。

**{single_issue_caveat}**

本报告不重新划分holdout、不重新调参，也不重新评估旧final holdout。所有候选仅是冻结策略输出，不是投注推荐、中奖概率预测或策略有效性证据。

## 冻结契约

- rolling window：`{freeze.rolling_window}`
- exponential decay：`{freeze.decay}`
- hybrid weights（全历史/滚动/指数）：`{freeze.hybrid_weights}`
- 随机基线：`{len(freeze.random_seeds)}` 个冻结seed；这些seed是同期开奖条件下的Monte Carlo重复，不是独立期开奖样本。
- 冻结数据快照：`{freeze.original_earliest_issue}`—`{freeze.original_latest_issue}`，`{freeze.original_record_count}`期，原CSV SHA-256 `{freeze.original_csv_sha256}`。
- 冻结配置指纹：`{freeze.fingerprint}`。
- 冻结策略源码清单指纹：`{freeze.source_manifest_fingerprint}`；运行在读取开奖数据和执行策略前逐文件复核源码。
- 奖金采用 `{freeze.prize_scenario_label}`；这是规则封顶情景，不代表逐期实际兑付奖金。

## 当前数据与方法

- 当前输入共 `{len(issues)}` 期，范围 `{int(issues[0])}`—`{int(issues[-1])}`，当前CSV SHA-256 `{data_sha256}`。
- 预测任意目标期 `t` 时，评估器只接收截至该目标期的截断数组，策略输入严格为目标期之前的历史。
- 已有前瞻记录会按冻结配置全量重算并逐字段校验；同一主键重复运行不追加，任何冲突在写文件前明确失败。
- 每期固定两注、每注2元。`disjoint`两注无重叠；`independent`在本项目中表示分别优化且允许重叠，并非统计独立，确定性策略通常产生两注相同票面。

## 前瞻证据分级

- `2026187`是**首个冻结后样本外观察**：参数、seed和策略集合已冻结，并按截至`2026186`的历史重放；但其具体票面没有在开奖前通过版本控制公开封存，因此不能表述为已经提前公开封存了具体票面的预测。
- `2026188`在版本化清单提交后，是**首个具有完整预先封存候选集的期号**。该清单包含完整100行规范化候选、历史指纹、源码清单指纹和候选CSV哈希。

## 前瞻记录完整性

{chr(10).join(lines)}

完整逐票结果见 `results/prospective/prospective_records.csv`；描述摘要见 `results/prospective/prospective_summary.csv`。当前摘要共 `{len(summary)}` 行，不含p值、显著性判断或机会性排名。单期置信区间未计算，相关字段留空。

## 下一期冻结策略票面

已用截至 `{int(issues[-1])}` 的历史为期号 `{int(candidates['target_issue'].iloc[0])}` 生成 `{len(candidates)}` 行票面，覆盖选一至选十、两种出票方式及五个冻结确定性策略。文件为 `results/prospective/next_issue_candidates.csv`。

本期不可变候选清单为 `{candidate_manifest_path.as_posix()}`，候选CSV规范化SHA-256为 `{candidate_manifest['next_issue_candidates_csv_normalized_sha256']}`。已有期号的清单只读，重算逐字段不一致时立即失败，绝不覆盖。

这些票面只说明冻结代码在当前历史上的确定性输出，不宣称彩票开奖可预测，也不构成投注建议。

## 限制与后续判定边界

- 目前共有 `{len(prospective_issues)}` 个冻结后样本外观察期，20个随机seed不能替代更多期开奖；其中2026187不具有开奖前具体票面封存证据。
- 不因本期命中、奖金或ROI调整窗口、衰减、权重、高级特征参数、策略集合或seed。
- 只有在更多未来开奖自然到达后，才能按预先固定的比较方法累积证据；任何新策略主张需要另一个预先声明且未接触的数据段。
- 浮动奖使用封顶情景，收益字段不能解释为历史实付或确定性ROI。

生成时间（UTC）：`{generated_at}`。
"""


def run_prospective_evaluation(
    paths: ProspectivePaths,
    *,
    bootstrap_samples: int = 1000,
    next_issue: int | None = None,
) -> ProspectiveRunResult:
    """执行冻结验证、幂等合并、摘要和下一期候选生成。"""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples必须为正整数")
    validate_prospective_paths_integrity(paths)
    freeze = load_scientific_freeze(paths.freeze_config)
    issues, draws = load_history_csv(paths.data)
    verify_frozen_inputs(freeze, paths, issues, draws)

    expected_records = evaluate_new_issues(issues, draws, freeze)
    combined_records = reconcile_records(expected_records, paths.records, freeze)
    if combined_records.empty:
        raise ValueError("当前数据没有冻结截止期之后的开奖记录")
    summary = summarise_prospective_records(
        combined_records, freeze, bootstrap_samples=bootstrap_samples
    )
    candidate_issue = int(next_issue) if next_issue is not None else freeze.next_issue
    candidates = generate_next_issue_candidates(
        issues, draws, freeze, next_issue=candidate_issue
    )
    manifest_path, candidate_manifest, create_manifest = _prepare_candidate_manifest(
        paths=paths,
        freeze=freeze,
        issues=issues,
        draws=draws,
        candidates=candidates,
    )
    relative_manifest_path = manifest_path.relative_to(paths.project_root)
    report = build_prospective_report(
        freeze=freeze,
        issues=issues,
        records=combined_records,
        summary=summary,
        candidates=candidates,
        data_sha256=_file_sha256(paths.data),
        candidate_manifest_path=relative_manifest_path,
        candidate_manifest=candidate_manifest,
    )

    # 所有冲突检查先完成；新manifest先排他创建，失败时尚未改写任何输出。
    manifest_created = False
    if create_manifest:
        manifest_text = (
            json.dumps(candidate_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        _create_text_exclusive(manifest_path, manifest_text, paths.temp_root)
        manifest_created = True
    try:
        _atomic_write_text(paths.records, _csv_text(combined_records), paths.temp_root)
        _atomic_write_text(paths.summary, _csv_text(summary), paths.temp_root)
        _atomic_write_text(paths.candidates, _csv_text(candidates), paths.temp_root)
        _atomic_write_text(paths.report, report, paths.temp_root)
    except BaseException:
        # 只回滚本次刚创建、尚未形成完整输出状态的manifest；既有manifest永不删除。
        if manifest_created and manifest_path.exists():
            manifest_path.unlink()
        raise
    prospective_issues = tuple(sorted(map(int, combined_records["issue"].unique())))
    return ProspectiveRunResult(
        prospective_issues=prospective_issues,
        records=len(combined_records),
        summary_rows=len(summary),
        candidate_rows=len(candidates),
        next_issue=candidate_issue,
        candidate_manifest=manifest_path,
    )


__all__ = [
    "EXPECTED_DECAY",
    "EXPECTED_FROZEN_THROUGH_ISSUE",
    "EXPECTED_HYBRID_WEIGHTS",
    "EXPECTED_RANDOM_SEEDS",
    "EXPECTED_ROLLING_WINDOW",
    "EXPECTED_SOURCE_FILES",
    "OUTPUT_STATUS",
    "ProspectivePaths",
    "ProspectiveRunResult",
    "ScientificFreeze",
    "build_prospective_report",
    "canonical_history_sha256",
    "evaluate_new_issues",
    "generate_next_issue_candidates",
    "load_scientific_freeze",
    "normalise_records",
    "reconcile_records",
    "resolve_prospective_paths",
    "run_prospective_evaluation",
    "SOURCE_MANIFEST_ALGORITHM",
    "summarise_prospective_records",
    "validate_prospective_paths_integrity",
    "validate_runtime_storage",
    "verify_frozen_inputs",
]
