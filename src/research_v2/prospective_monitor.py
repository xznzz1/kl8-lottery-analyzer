"""快乐8 v2 冻结模型的未来前瞻封存、评价和正式汇总基础设施。

模块只编排已合并的第一、第二阶段模型。生产状态机严格区分本地生成、GitHub
远程合并封存和开奖后评价；任何缺失、漂移、网络失败或顺序链断裂都必须
fail closed，且不得覆盖既有 manifest 或评价。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .bayesian import DRAW_SIZE, FAIR_PROBABILITY, NUMBER_COUNT
from .changepoint import (
    CHANGE_QUANTILE,
    CHANGEPOINT_PRIOR_STRENGTH,
    HIGH_CHANGE_DECAY,
    MINIMUM_EFFECTIVE_HISTORY,
    NORMAL_DECAY,
    RECENT_WINDOW_GRID,
    REFERENCE_WINDOW,
)
from .changepoint_evaluation import (
    CHANGEPOINT_STRATEGY,
    FIXED_HIGH_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    ChangepointEvaluationConfig,
    evaluate_changepoint_walk_forward,
)
from .evaluation import (
    DYNAMIC_STRATEGY,
    MINIMUM_INITIAL_HISTORY,
    MINIMUM_INNER_OBSERVATIONS,
    validate_issue_draws,
)
from .metrics import bernoulli_log_loss, brier_score, calibration_summary, top_k_hits

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SCHEMA_VERSION = 2
FREEZE_ID = "kl8-v2-prospective-365-v1"
FREEZE_PENDING = "research_model_and_protocol_frozen_pending_code_audit"
FREEZE_ACTIVE = "active"
FREEZE_TAG = "kl8-v2-prospective-365-v1"
GITHUB_REPOSITORY = "xznzz1/kl8-lottery-analyzer"
GITHUB_BASE_BRANCH = "scientific-model"
MANIFEST_EVIDENCE_STATUS = "prospective_manifest_pending_remote_seal"
EVALUATION_EVIDENCE_STATUS = "prospective_presealed_research"
SUMMARY_EVIDENCE_STATUS = "completed_prospective_confirmation_summary"
FINAL_EVALUATION_SEAL_EVIDENCE_STATUS = "final_evaluation_remote_anchor_verified"
CONFIRMATION_ISSUE_COUNT = 365
UNIFORM_STRATEGY = "uniform_random"
UNIFORM_BASE_SEEDS = tuple(range(202601, 202621))
BOOTSTRAP_BLOCK_LENGTH = 30
BOOTSTRAP_RESAMPLE_COUNT = 20_000
BOOTSTRAP_SEED = 20_260_717
PROBABILITY_STRATEGIES = (
    UNIFORM_STRATEGY,
    DYNAMIC_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    FIXED_HIGH_STRATEGY,
    CHANGEPOINT_STRATEGY,
)
PRIMARY_COMPARISON_MODELS = (
    DYNAMIC_STRATEGY,
    FIXED_NORMAL_STRATEGY,
    FIXED_HIGH_STRATEGY,
    CHANGEPOINT_STRATEGY,
)
PRIMARY_COMPARISONS = (
    "dynamic_bayesian_minus_uniform_random",
    "fixed_normal_bayesian_minus_uniform_random",
    "fixed_high_bayesian_minus_uniform_random",
    "changepoint_bayesian_minus_uniform_random",
)
SECONDARY_METRICS = (
    "mean_brier_all_five_models",
    "mean_bernoulli_log_loss_all_five_models",
    "fixed_10_bin_calibration_and_ece",
    "mean_top_1_through_top_10_hits_all_five_models",
    "uniform_20_seed_within_issue_mean_top_k",
    "changepoint_minus_fixed_normal_brier_and_log_loss",
    "changepoint_minus_fixed_high_brier_and_log_loss",
    "high_change_trigger_count_and_proportion",
    "per_issue_index_and_manifest_sha256",
)
MANIFEST_RELATIVE_DIR = Path("reports/research_v2_prospective_manifests")
RESULT_RELATIVE_DIR = Path("results/research_v2_prospective")
FREEZE_RELATIVE_PATH = Path("config/research_v2_prospective_freeze.json")
DATA_RELATIVE_PATH = Path("data_cache/kl8/data.csv")
FINAL_EVALUATION_SEAL_FILENAME = "final_evaluation_seal.json"
FORMAL_SUMMARY_FILENAME = "formal_summary.json"
FROZEN_SOURCE_PATHS = (
    ".gitattributes",
    "scripts/research_v2_prospective.py",
    "src/research_v2/bayesian.py",
    "src/research_v2/changepoint.py",
    "src/research_v2/changepoint_evaluation.py",
    "src/research_v2/evaluation.py",
    "src/research_v2/metrics.py",
    "src/research_v2/prospective_monitor.py",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
NUMERIC_EVALUATION_PATTERN = re.compile(r"^[1-9][0-9]*\.json$")


@dataclass(frozen=True)
class ProspectivePrediction:
    """五个冻结模型在一个未开奖目标期的目标期前预测。"""

    probabilities: dict[str, FloatArray]
    rankings: dict[str, IntArray]
    uniform_rankings: tuple[IntArray, ...]
    dynamic_decay: float
    dynamic_prior_strength: float
    changepoint_recent_window: int
    changepoint_change_score: float
    changepoint_threshold: float
    changepoint_high_change: bool
    changepoint_active_decay: float
    changepoint_effective_history_length: int


@dataclass(frozen=True)
class ChainPosition:
    """下一份 manifest 在不可选择顺序链中的位置。"""

    confirmation_index: int
    protocol_start_target_issue: int
    previous_target_issue: int | None
    previous_manifest_sha256: str | None
    previous_evaluation_target_issue: int | None
    previous_evaluation_confirmation_index: int | None
    previous_evaluation_path: str | None
    previous_evaluation_sha256: str | None


@dataclass(frozen=True)
class ManifestChainEntry:
    """一份已经落盘并通过顺序链校验的 manifest。"""

    path: Path
    payload: dict[str, Any]
    sha256: str
    confirmation_index: int
    target_issue: int


@dataclass(frozen=True)
class RemoteSealEvidence:
    """通过 GitHub API 验证的远程合并封存证据。"""

    seal_pr_number: int
    seal_pr_url: str
    seal_merge_commit_sha: str
    seal_merged_at_utc: str
    manifest_sha256_at_merge: str
    previous_evaluation_sha256_at_merge: str | None


@dataclass(frozen=True)
class RemoteFileAnchorEvidence:
    """一个文件在固定仓库已合并 PR 中的远程原始字节锚点。"""

    seal_pr_number: int
    seal_pr_url: str
    seal_merge_commit_sha: str
    seal_merged_at_utc: str
    file_sha256_at_merge: str


@dataclass(frozen=True)
class BootstrapInference:
    """固定循环移动分块 bootstrap 的单项主要比较结果。"""

    observed_mean: float
    ordinary_standard_error: float
    interval_lower: float
    interval_upper: float
    raw_p_value: float


class GitHubSealClient(Protocol):
    """远程封存验证所需的最小、可 mock GitHub API 接口。"""

    def get_pull_request(
        self, repository: str, pr_number: int
    ) -> Mapping[str, object]: ...

    def get_file_bytes(self, repository: str, path: str, commit_sha: str) -> bytes: ...


GitCommand = Callable[[Sequence[str]], str]


def normalized_lf_sha256(path: Path) -> str:
    """按 UTF-8 解码并统一 LF 后计算源码 SHA-256。"""

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_sha256(path: Path) -> str:
    """分块计算文件原始字节 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_fingerprint(files: Mapping[str, str]) -> str:
    """为已排序的 ``路径=NUL=哈希`` 清单形成确定性指纹。"""

    canonical = "".join(
        f"{relative}\0{digest}\n" for relative, digest in sorted(files.items())
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """返回跨重复运行稳定的 UTF-8 JSON 字节。"""

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (rendered + "\n").encode("utf-8")


def configuration_fingerprint(config: Mapping[str, object]) -> str:
    """计算排除自哈希字段后的完整冻结配置 SHA-256。"""

    copied = dict(config)
    copied.pop("configuration_sha256", None)
    return hashlib.sha256(canonical_json_bytes(copied)).hexdigest()


def _evaluation_payload_fingerprint(record: Mapping[str, object]) -> str:
    copied = dict(record)
    copied.pop("evaluation_payload_sha256", None)
    return hashlib.sha256(canonical_json_bytes(copied)).hexdigest()


def _validate_evaluation_payload_fingerprint(record: Mapping[str, object]) -> None:
    configured = record.get("evaluation_payload_sha256")
    if (
        not isinstance(configured, str)
        or not SHA256_PATTERN.fullmatch(configured)
        or configured != _evaluation_payload_fingerprint(record)
    ):
        raise ValueError("evaluation内容SHA-256不匹配，文件已被修改")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取合法JSON：{path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return cast(dict[str, Any], parsed)


def _load_canonical_json_object(path: Path, label: str) -> dict[str, Any]:
    """读取协议产物，并拒绝任何非规范或逐字节改写。"""

    payload = _load_json_object(path)
    if path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError(f"{label}原始字节不是规范不可变JSON")
    return payload


def _numeric_evaluation_paths(results_dir: Path) -> list[Path]:
    """只枚举 ``<target_issue>.json``，排除final seal与summary。"""

    if not results_dir.exists():
        return []
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_file() and NUMERIC_EVALUATION_PATTERN.fullmatch(path.name)
    )


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是JSON对象")
    return cast(dict[str, Any], value)


def _canonical_utc(value: str, label: str) -> tuple[str, datetime]:
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}必须是带时区的ISO 8601时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label}必须包含时区")
    utc = parsed.astimezone(timezone.utc)
    canonical = utc.isoformat(timespec="seconds").replace("+00:00", "Z")
    return canonical, utc


def utc_now_string() -> str:
    """返回精确到秒的规范 UTC 时间。"""

    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def resolve_within(project_root: Path, path: Path, label: str) -> Path:
    """解析路径并拒绝越出项目根目录。"""

    root = project_root.resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label}必须位于项目目录内")
    return resolved


def require_contract_path(
    project_root: Path,
    path: Path,
    expected_relative: Path,
    label: str,
) -> Path:
    """拒绝把正式冻结产物写入约定目录之外。"""

    resolved = resolve_within(project_root, path, label)
    expected = (project_root.resolve() / expected_relative).resolve()
    if resolved != expected:
        raise ValueError(f"{label}必须固定为{expected_relative.as_posix()}")
    return resolved


def load_history_csv(path: Path) -> tuple[IntArray, IntArray]:
    """读取快乐8 CSV，并按期号升序执行完整票面校验。"""

    number_columns = [f"红球_{index}" for index in range(1, DRAW_SIZE + 1)]
    rows: list[tuple[int, tuple[int, ...]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "期数" not in reader.fieldnames:
            raise ValueError("快乐8数据缺少期数列")
        if not set(number_columns).issubset(reader.fieldnames):
            raise ValueError("快乐8数据缺少红球列")
        for row in reader:
            rows.append(
                (
                    int(row["期数"]),
                    tuple(int(row[column]) for column in number_columns),
                )
            )
    rows.sort(key=lambda item: item[0])
    issues = cast(IntArray, np.asarray([row[0] for row in rows], dtype=np.int64))
    draws = cast(IntArray, np.asarray([row[1] for row in rows], dtype=np.int64))
    return validate_issue_draws(issues, draws)


def _expected_models() -> dict[str, object]:
    return {
        UNIFORM_STRATEGY: {
            "base_seeds": list(UNIFORM_BASE_SEEDS),
            "probability": FAIR_PROBABILITY,
            "ranking_method": "sha256_seeded_numpy_permutation",
            "seed_material_format": (
                "kl8-v2-prospective-365-v1|<target_issue>|<base_seed>"
            ),
            "seed_integer_derivation": "sha256_first_8_bytes_unsigned_big_endian",
            "top_k_aggregation": "within_issue_mean_over_20_seeds",
        },
        DYNAMIC_STRATEGY: {
            "decay_grid": [0.97, 0.99, 0.995],
            "minimum_initial_history": MINIMUM_INITIAL_HISTORY,
            "minimum_inner_observations": MINIMUM_INNER_OBSERVATIONS,
            "parameter_selection": (
                "target_prior_inner_mean_brier_argmin_preregistered_grid_order_tiebreak"
            ),
            "prior_strength_grid": [5.0, 20.0, 80.0],
        },
        FIXED_NORMAL_STRATEGY: {
            "decay": NORMAL_DECAY,
            "history": "all_target_prior_history",
            "prior_strength": CHANGEPOINT_PRIOR_STRENGTH,
        },
        FIXED_HIGH_STRATEGY: {
            "decay": HIGH_CHANGE_DECAY,
            "history": f"last_{MINIMUM_EFFECTIVE_HISTORY}_target_prior_issues",
            "prior_strength": CHANGEPOINT_PRIOR_STRENGTH,
        },
        CHANGEPOINT_STRATEGY: {
            "change_quantile": CHANGE_QUANTILE,
            "fixed_high_model": FIXED_HIGH_STRATEGY,
            "fixed_normal_model": FIXED_NORMAL_STRATEGY,
            "minimum_effective_history": MINIMUM_EFFECTIVE_HISTORY,
            "recent_window_grid": list(RECENT_WINDOW_GRID),
            "reference_window": REFERENCE_WINDOW,
            "selection_metric": "inner_mean_brier",
            "threshold_training": "target_prior_change_scores_only",
        },
    }


def _validate_freeze_contract(config: Mapping[str, Any]) -> None:
    expected_top_level = {
        "schema_version",
        "freeze_id",
        "freeze_status",
        "freeze_tag",
        "configuration_sha256",
        "evidence_status",
        "confirmation_issue_count",
        "github",
        "models",
        "confirmation_protocol",
        "manifest_policy",
        "evaluation_policy",
        "summary_policy",
        "claim_boundary",
        "source_manifest",
    }
    if set(config) != expected_top_level:
        raise ValueError("v2前瞻冻结配置顶层字段漂移")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v2前瞻冻结配置schema_version不受支持")
    if config.get("freeze_id") != FREEZE_ID:
        raise ValueError("freeze_id漂移")
    if config.get("freeze_status") not in (FREEZE_PENDING, FREEZE_ACTIVE):
        raise ValueError("freeze_status必须是pending或active")
    if config.get("freeze_tag") != FREEZE_TAG:
        raise ValueError("freeze_tag漂移")
    configured_fingerprint = config.get("configuration_sha256")
    if (
        not isinstance(configured_fingerprint, str)
        or not SHA256_PATTERN.fullmatch(configured_fingerprint)
        or configured_fingerprint != configuration_fingerprint(config)
    ):
        raise ValueError("冻结配置SHA-256不匹配")
    if config.get("evidence_status") != {
        "evaluation_verified": EVALUATION_EVIDENCE_STATUS,
        "final_evaluation_seal": FINAL_EVALUATION_SEAL_EVIDENCE_STATUS,
        "manifest_local": MANIFEST_EVIDENCE_STATUS,
        "summary_completed": SUMMARY_EVIDENCE_STATUS,
    }:
        raise ValueError("证据状态契约漂移")
    if config.get("confirmation_issue_count") != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("正式确认期数必须固定为365")
    if config.get("github") != {
        "base_branch": GITHUB_BASE_BRANCH,
        "repository": GITHUB_REPOSITORY,
    }:
        raise ValueError("GitHub仓库或base分支漂移")
    if config.get("models") != _expected_models():
        raise ValueError("五个冻结模型或uniform seed列表漂移")

    protocol = _require_mapping(
        config.get("confirmation_protocol"), "confirmation_protocol"
    )
    expected_protocol = {
        "bootstrap": {
            "alternative": "mean_model_minus_uniform_brier_less_than_zero",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "method": "circular_moving_block_bootstrap",
            "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
            "seed": BOOTSTRAP_SEED,
        },
        "early_success_claims_forbidden": True,
        "early_stopping_forbidden": True,
        "formal_summary_after_exact_issue_count": CONFIRMATION_ISSUE_COUNT,
        "issue_is_statistical_unit": True,
        "model_changes_during_confirmation_forbidden": True,
        "multiplicity_correction": "Holm",
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "primary_metric": "per_issue_80_dimensional_brier_score",
        "secondary_metrics": list(SECONDARY_METRICS),
        "secondary_metrics_are_descriptive": True,
        "secondary_metrics_cannot_select_models": True,
    }
    if protocol != expected_protocol:
        raise ValueError("主要检验、Holm、次要指标或禁止提前停止契约漂移")
    if config.get("manifest_policy") != {
        "continuity_rule": "previous_target_must_equal_current_data_latest_issue",
        "directory": MANIFEST_RELATIVE_DIR.as_posix(),
        "existing_manifest_policy": "refuse_overwrite_rewrite_or_delete",
        "previous_evaluation_chain": "raw_sha256_and_independent_recalculation",
        "remote_claim_at_generation": False,
        "sequence": "immutable_manifest_and_evaluation_hash_chain_1_to_365",
    }:
        raise ValueError("manifest目录、哈希链或防覆盖策略漂移")
    if config.get("evaluation_policy") != {
        "directory": RESULT_RELATIVE_DIR.as_posix(),
        "duplicate_target_policy": "refuse_overwrite",
        "final_evaluation_seal_filename": FINAL_EVALUATION_SEAL_FILENAME,
        "local_creation_state": "remote_evaluation_anchor_pending",
        "previous_evaluation_remote_anchor": "next_manifest_seal_pr",
        "record_format": "one_immutable_json_per_target_issue",
        "remote_verification": "github_merged_pr_fail_closed",
    }:
        raise ValueError("评价目录、防覆盖或远程验证策略漂移")
    if config.get("summary_policy") != {
        "data_path": DATA_RELATIVE_PATH.as_posix(),
        "evaluation_enumeration": "numeric_target_issue_json_only",
        "final_evaluation_anchor_required": True,
        "formal_summary_filename": FORMAL_SUMMARY_FILENAME,
        "official_data_recalculation_required": True,
    }:
        raise ValueError("正式summary数据复核、文件枚举或final seal策略漂移")
    if config.get("claim_boundary") != {
        "fair_lottery_note": "若彩票公平且独立，历史模型不应存在稳定预测优势。",
        "historical_v2_status": "exploratory_development_evidence",
        "prospective_status_before_365": (
            "ongoing_prospective_presealed_research_no_success_claim"
        ),
    }:
        raise ValueError("claim_boundary漂移")


def load_and_verify_freeze_config(
    project_root: Path, config_path: Path
) -> dict[str, Any]:
    """读取完整冻结契约，并在任何生产操作前拒绝配置或源码漂移。"""

    resolved = require_contract_path(
        project_root, config_path, FREEZE_RELATIVE_PATH, "冻结配置路径"
    )
    config = _load_json_object(resolved)
    _validate_freeze_contract(config)
    source_manifest = _require_mapping(config.get("source_manifest"), "source_manifest")
    if source_manifest.get("algorithm") != "sha256_utf8_normalized_lf":
        raise ValueError("source manifest哈希算法不受支持")
    if source_manifest.get("change_policy") != (
        "任一冻结源码变化时拒绝生成manifest、评价或正式汇总；必须建立新的协议版本。"
    ):
        raise ValueError("source manifest变更策略漂移")
    files = _require_mapping(source_manifest.get("files"), "source_manifest.files")
    if set(files) != set(FROZEN_SOURCE_PATHS):
        raise ValueError("source manifest冻结文件集合漂移")
    verified: dict[str, str] = {}
    for relative in FROZEN_SOURCE_PATHS:
        raw_digest = files.get(relative)
        if not isinstance(raw_digest, str) or not SHA256_PATTERN.fullmatch(raw_digest):
            raise ValueError(f"source manifest含非法SHA-256：{relative}")
        source_path = resolve_within(project_root, Path(relative), "冻结源码")
        if not source_path.is_file():
            raise ValueError(f"冻结源码不存在：{relative}")
        if normalized_lf_sha256(source_path) != raw_digest:
            raise ValueError(f"冻结源码漂移，拒绝运行：{relative}")
        verified[relative] = raw_digest
    if source_manifest.get("fingerprint") != source_manifest_fingerprint(verified):
        raise ValueError("source manifest总指纹不匹配")
    return config


def _default_git_command(project_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_active_freeze(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    git_command: GitCommand | None = None,
) -> None:
    """pending时拒绝；active时验证冻结标签存在且是当前HEAD祖先。"""

    if config.get("freeze_status") != FREEZE_ACTIVE:
        raise RuntimeError("冻结配置仍为pending，生产操作全部拒绝运行")
    command = git_command or (lambda args: _default_git_command(project_root, args))
    tag = str(config["freeze_tag"])
    try:
        tag_commit = command(("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"))
        head = command(("rev-parse", "HEAD"))
        command(("merge-base", "--is-ancestor", tag_commit, head))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("冻结标签不存在或不是当前HEAD祖先，拒绝运行") from exc
    if not GIT_SHA_PATTERN.fullmatch(tag_commit) or not GIT_SHA_PATTERN.fullmatch(head):
        raise RuntimeError("Git返回了非法提交SHA，拒绝运行")


def canonical_history_sha256(issues: IntArray, draws: IntArray) -> str:
    """对完整目标期前历史形成与行序、号码列顺序无关的规范哈希。"""

    issue_values, draw_values = validate_issue_draws(issues, draws)
    lines = []
    for issue, draw in zip(issue_values, draw_values, strict=True):
        numbers = ",".join(f"{number:02d}" for number in sorted(map(int, draw)))
        lines.append(f"{int(issue)},{numbers}\n")
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def validate_unpublished_target(
    issues: IntArray, draws: IntArray, target_issue: int
) -> tuple[IntArray, IntArray]:
    """要求输入完整截止于最新开奖，且目标期严格尚未出现在数据中。"""

    issue_values, draw_values = validate_issue_draws(issues, draws)
    if target_issue <= 0:
        raise ValueError("目标期号必须为正整数")
    if np.any(issue_values == target_issue):
        raise ValueError("target_issue已经存在于输入数据，拒绝生成manifest")
    if np.any(issue_values > target_issue):
        raise ValueError("输入数据包含晚于target_issue的记录")
    latest_issue = int(issue_values[-1])
    if target_issue <= latest_issue:
        raise ValueError("target_issue必须严格晚于最新已开奖期")
    required = MINIMUM_INITIAL_HISTORY + MINIMUM_INNER_OBSERVATIONS
    if len(issue_values) < required:
        raise ValueError(f"目标期前至少需要{required}期历史")
    return issue_values, draw_values


def _uniform_seed_material(target_issue: int, base_seed: int) -> bytes:
    return f"{FREEZE_ID}|{target_issue}|{base_seed}".encode("ascii")


def uniform_seed_rankings(target_issue: int) -> tuple[IntArray, ...]:
    """按20个预注册seed生成目标期特定、完全可复现的随机排列。"""

    rankings: list[IntArray] = []
    numbers = np.arange(1, NUMBER_COUNT + 1, dtype=np.int64)
    for base_seed in UNIFORM_BASE_SEEDS:
        digest = hashlib.sha256(
            _uniform_seed_material(target_issue, base_seed)
        ).digest()
        seed_integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed_integer)
        rankings.append(cast(IntArray, rng.permutation(numbers)))
    return tuple(rankings)


def predict_frozen_models(
    issues: IntArray, draws: IntArray, *, target_issue: int
) -> ProspectivePrediction:
    """严格复用已合并 v2 实现，形成未开奖目标期的五模型预测。"""

    historical_issues, historical_draws = validate_unpublished_target(
        issues, draws, target_issue
    )
    sentinel_draw = np.arange(1, DRAW_SIZE + 1, dtype=np.int64)
    evaluation_issues = cast(
        IntArray,
        np.concatenate((historical_issues, np.asarray([target_issue], dtype=np.int64))),
    )
    evaluation_draws = cast(
        IntArray, np.vstack((historical_draws, sentinel_draw)).astype(np.int64)
    )
    result = evaluate_changepoint_walk_forward(
        evaluation_issues,
        evaluation_draws,
        config=ChangepointEvaluationConfig(include_comparators=False),
    )
    if int(result.outer_issues[-1]) != target_issue:
        raise RuntimeError("现有v2接口没有返回目标期预测")
    offset = len(result.outer_indices) - 1
    phase1_parameters = result.phase1_result.selected_parameters(offset)
    changepoint_parameters = result.selected_parameters(offset)
    probabilities: dict[str, FloatArray] = {
        UNIFORM_STRATEGY: np.full(NUMBER_COUNT, FAIR_PROBABILITY, dtype=np.float64),
        DYNAMIC_STRATEGY: result.phase1_result.posterior_mean[offset].copy(),
        FIXED_NORMAL_STRATEGY: result.fixed_normal_posterior_mean[offset].copy(),
        FIXED_HIGH_STRATEGY: result.fixed_high_posterior_mean[offset].copy(),
        CHANGEPOINT_STRATEGY: result.posterior_mean[offset].copy(),
    }
    rankings: dict[str, IntArray] = {
        DYNAMIC_STRATEGY: result.rankings[DYNAMIC_STRATEGY][offset].copy(),
        FIXED_NORMAL_STRATEGY: result.rankings[FIXED_NORMAL_STRATEGY][offset].copy(),
        FIXED_HIGH_STRATEGY: result.rankings[FIXED_HIGH_STRATEGY][offset].copy(),
        CHANGEPOINT_STRATEGY: result.rankings[CHANGEPOINT_STRATEGY][offset].copy(),
    }
    for strategy, model_probabilities in probabilities.items():
        if model_probabilities.shape != (NUMBER_COUNT,) or not np.all(
            (model_probabilities > 0.0) & (model_probabilities < 1.0)
        ):
            raise FloatingPointError(f"{strategy}概率不严格位于(0,1)")
        if not np.isclose(model_probabilities.sum(), DRAW_SIZE, rtol=0.0, atol=1e-9):
            raise FloatingPointError(f"{strategy}的80个概率之和不等于20")
    for strategy, ranking in rankings.items():
        if set(map(int, ranking)) != set(range(1, NUMBER_COUNT + 1)):
            raise FloatingPointError(f"{strategy}排名不是1至80完整排列")
    return ProspectivePrediction(
        probabilities=probabilities,
        rankings=rankings,
        uniform_rankings=uniform_seed_rankings(target_issue),
        dynamic_decay=phase1_parameters.decay,
        dynamic_prior_strength=phase1_parameters.prior_strength,
        changepoint_recent_window=changepoint_parameters.recent_window,
        changepoint_change_score=float(
            result.candidate_change_scores[
                offset, int(result.selected_parameter_indices[offset])
            ]
        ),
        changepoint_threshold=float(
            result.candidate_thresholds[
                offset, int(result.selected_parameter_indices[offset])
            ]
        ),
        changepoint_high_change=bool(result.high_change[offset]),
        changepoint_active_decay=float(result.active_decay[offset]),
        changepoint_effective_history_length=int(
            result.effective_history_length[offset]
        ),
    )


def _relative_posix(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _top_k_candidates(ranking: IntArray) -> dict[str, list[int]]:
    return {str(k): list(map(int, ranking[:k])) for k in range(1, 11)}


def _load_manifest_chain(
    manifest_dir: Path, config: Mapping[str, Any]
) -> tuple[ManifestChainEntry, ...]:
    unsorted_paths = list(manifest_dir.glob("*.json")) if manifest_dir.exists() else []
    indexed_paths: list[tuple[int, Path]] = []
    for path in unsorted_paths:
        payload = _load_canonical_json_object(path, "manifest")
        index = payload.get("confirmation_index")
        if not isinstance(index, int):
            raise ValueError("manifest confirmation_index非法")
        indexed_paths.append((index, path))
    paths = [path for _, path in sorted(indexed_paths, key=lambda item: item[0])]
    entries: list[ManifestChainEntry] = []
    expected_start: int | None = None
    previous_target: int | None = None
    previous_digest: str | None = None
    for expected_index, path in enumerate(paths, start=1):
        payload = _load_canonical_json_object(path, "manifest")
        index = payload.get("confirmation_index")
        target = payload.get("target_issue")
        if index != expected_index:
            raise ValueError("manifest confirmation_index不连续")
        if not isinstance(target, int) or target <= 0 or path.stem != str(target):
            raise ValueError("manifest文件名与target_issue不一致")
        if payload.get("freeze_id") != config.get("freeze_id"):
            raise ValueError("manifest freeze_id不匹配")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("manifest schema_version不匹配")
        if payload.get("evidence_status") != MANIFEST_EVIDENCE_STATUS:
            raise ValueError("manifest本地证据状态不匹配")
        if payload.get("remote_preseal_verified") is not False:
            raise ValueError("manifest不得自行声称远程封存")
        if payload.get("freeze_config_sha256") != config.get("configuration_sha256"):
            raise ValueError("manifest冻结配置SHA-256不匹配")
        if payload.get("source_manifest") != config.get("source_manifest"):
            raise ValueError("manifest source manifest与冻结配置不一致")
        if payload.get("frozen_parameters") != config.get("models"):
            raise ValueError("manifest五模型冻结参数与配置不一致")
        start = payload.get("protocol_start_target_issue")
        if expected_index == 1:
            expected_start = target
            if start != target:
                raise ValueError("首期protocol_start_target_issue必须等于target_issue")
            if payload.get("previous_target_issue") is not None:
                raise ValueError("首期previous_target_issue必须为null")
            if payload.get("previous_manifest_sha256") is not None:
                raise ValueError("首期previous_manifest_sha256必须为null")
            if payload.get("previous_evaluation_target_issue") is not None:
                raise ValueError("首期previous_evaluation_target_issue必须为null")
            if payload.get("previous_evaluation_confirmation_index") is not None:
                raise ValueError("首期previous_evaluation_confirmation_index必须为null")
            if payload.get("previous_evaluation_path") is not None:
                raise ValueError("首期previous_evaluation_path必须为null")
            if payload.get("previous_evaluation_sha256") is not None:
                raise ValueError("首期previous_evaluation_sha256必须为null")
        else:
            if start != expected_start:
                raise ValueError("protocol_start_target_issue发生漂移")
            if payload.get("previous_target_issue") != previous_target:
                raise ValueError("manifest previous_target_issue链断裂")
            if payload.get("previous_manifest_sha256") != previous_digest:
                raise ValueError("manifest SHA-256链断裂")
            if previous_target is not None and target <= previous_target:
                raise ValueError("manifest target_issue必须严格递增")
            if payload.get("history_through_issue") != previous_target:
                raise ValueError("后续manifest history_through_issue必须等于上一目标期")
            if payload.get("previous_evaluation_target_issue") != previous_target:
                raise ValueError("manifest上一evaluation target_issue链断裂")
            if payload.get("previous_evaluation_confirmation_index") != (
                expected_index - 1
            ):
                raise ValueError("manifest上一evaluation confirmation_index链断裂")
            expected_evaluation_path = (
                RESULT_RELATIVE_DIR / f"{previous_target}.json"
            ).as_posix()
            if payload.get("previous_evaluation_path") != expected_evaluation_path:
                raise ValueError("manifest上一evaluation路径链断裂")
            previous_evaluation_digest = payload.get("previous_evaluation_sha256")
            if not isinstance(
                previous_evaluation_digest, str
            ) or not SHA256_PATTERN.fullmatch(previous_evaluation_digest):
                raise ValueError("manifest上一evaluation SHA-256非法")
        digest = raw_sha256(path)
        entries.append(
            ManifestChainEntry(
                path=path,
                payload=payload,
                sha256=digest,
                confirmation_index=expected_index,
                target_issue=target,
            )
        )
        previous_target = target
        previous_digest = digest
    return tuple(entries)


def _validate_prior_evaluations(
    entries: Sequence[ManifestChainEntry], results_dir: Path
) -> dict[int, tuple[Path, dict[str, Any]]]:
    paths = _numeric_evaluation_paths(results_dir)
    expected_names = {f"{entry.target_issue}.json" for entry in entries}
    if {path.name for path in paths} != expected_names:
        raise ValueError("既有manifest与evaluation不一一对应，禁止继续")
    by_target = {entry.target_issue: entry for entry in entries}
    validated: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        record = _load_canonical_json_object(path, "evaluation")
        target = record.get("target_issue")
        if not isinstance(target, int) or target not in by_target:
            raise ValueError("evaluation包含非法或额外target_issue")
        entry = by_target[target]
        if record.get("confirmation_index") != entry.confirmation_index:
            raise ValueError("evaluation confirmation_index不匹配")
        manifest = _require_mapping(record.get("manifest"), "evaluation.manifest")
        if manifest.get("sha256") != entry.sha256:
            raise ValueError("evaluation记录的manifest SHA-256不匹配")
        if record.get("remote_preseal_verified") is not True:
            raise ValueError("既有evaluation未通过远程封存验证")
        if record.get("sealed_before_official_result") is not True:
            raise ValueError("既有evaluation不是开奖前远程封存")
        if record.get("evidence_status") != EVALUATION_EVIDENCE_STATUS:
            raise ValueError("既有evaluation证据状态不正确")
        if record.get("freeze_id") != FREEZE_ID:
            raise ValueError("既有evaluation freeze_id不匹配")
        if record.get("manifest_sha256_at_merge") != entry.sha256:
            raise ValueError("既有evaluation合并提交manifest SHA-256不匹配")
        if record.get("evaluation_locally_created") is not True:
            raise ValueError("既有evaluation缺少本地独占创建标记")
        if record.get("remote_evaluation_anchor_pending") is not True:
            raise ValueError("既有evaluation缺少待远程锚定标记")
        validated[target] = (path, record)
    for offset in range(1, len(entries)):
        previous_entry = entries[offset - 1]
        current_entry = entries[offset]
        previous_path, _ = validated[previous_entry.target_issue]
        _, current_record = validated[current_entry.target_issue]
        previous_digest = raw_sha256(previous_path)
        if current_entry.payload.get("previous_evaluation_sha256") != previous_digest:
            raise ValueError("既有manifest记录的上一evaluation SHA-256不匹配")
        if current_record.get("previous_evaluation_remote_anchor_verified") is not True:
            raise ValueError("既有evaluation远程锚定链不完整")
        expected_anchor = {
            "target_issue": previous_entry.target_issue,
            "confirmation_index": previous_entry.confirmation_index,
            "path": (
                RESULT_RELATIVE_DIR / f"{previous_entry.target_issue}.json"
            ).as_posix(),
            "sha256": previous_digest,
            "sha256_at_merge": previous_digest,
        }
        if current_record.get("previous_evaluation_anchor") != expected_anchor:
            raise ValueError("既有evaluation远程锚点与上一evaluation不一致")
    return validated


def next_chain_position(
    *,
    project_root: Path,
    manifest_dir: Path,
    results_dir: Path,
    config: Mapping[str, Any],
    target_issue: int,
    history_issues: IntArray,
    history_draws: IntArray,
) -> ChainPosition:
    """形成下一位置，并强制上一目标期是当前数据最新官方开奖。"""

    entries = _load_manifest_chain(manifest_dir, config)
    if len(entries) >= CONFIRMATION_ISSUE_COUNT:
        raise ValueError("365期确认链已满，拒绝生成额外manifest")
    evaluations = _validate_prior_evaluations(entries, results_dir)
    if not entries:
        return ChainPosition(1, target_issue, None, None, None, None, None, None)
    previous = entries[-1]
    if target_issue <= previous.target_issue:
        raise ValueError("下一target_issue必须严格晚于上一记录")
    latest_issue = int(history_issues[-1])
    if latest_issue != previous.target_issue:
        raise ValueError(
            "上一目标期不是当前数据最新开奖，说明官方开奖链不连续；"
            "同一freeze_id禁止跳过后恢复，必须建立新协议版本"
        )
    matches = np.flatnonzero(history_issues == previous.target_issue)
    if len(matches) != 1:
        raise ValueError("当前完整数据必须恰好包含一次上一目标期开奖")
    previous_path, previous_record = evaluations[previous.target_issue]
    official_actual = cast(IntArray, history_draws[int(matches[0])].copy())
    if previous_record.get("actual_numbers") != sorted(map(int, official_actual)):
        raise ValueError("上一evaluation actual_numbers与当前正式数据不一致")
    recomputed_models = calculate_manifest_metrics(previous.payload, official_actual)
    if previous_record.get("models") != recomputed_models:
        raise ValueError("上一evaluation models无法从manifest和正式结果独立复算")
    recomputed_comparisons = calculate_evaluation_comparisons(recomputed_models)
    if previous_record.get("comparisons") != recomputed_comparisons:
        raise ValueError("上一evaluation comparisons无法独立复算")
    _validate_evaluation_payload_fingerprint(previous_record)
    previous_evaluation_digest = raw_sha256(previous_path)
    return ChainPosition(
        confirmation_index=len(entries) + 1,
        protocol_start_target_issue=int(
            entries[0].payload["protocol_start_target_issue"]
        ),
        previous_target_issue=previous.target_issue,
        previous_manifest_sha256=previous.sha256,
        previous_evaluation_target_issue=previous.target_issue,
        previous_evaluation_confirmation_index=previous.confirmation_index,
        previous_evaluation_path=_relative_posix(project_root, previous_path),
        previous_evaluation_sha256=previous_evaluation_digest,
    )


def build_manifest(
    *,
    project_root: Path,
    data_path: Path,
    config_path: Path,
    manifest_dir: Path,
    results_dir: Path,
    issues: IntArray,
    draws: IntArray,
    target_issue: int,
    local_manifest_generated_at_utc: str,
    git_commit_sha: str,
    official_source_url: str,
    official_confirmed_at_utc: str,
) -> dict[str, object]:
    """构造本地、尚未声称远程封存的目标期前 manifest。"""

    config = load_and_verify_freeze_config(project_root, config_path)
    if config.get("freeze_status") != FREEZE_ACTIVE:
        raise RuntimeError("冻结配置仍为pending，拒绝构造生产manifest")
    resolved_data = resolve_within(project_root, data_path, "输入数据")
    generated_text, generated_time = _canonical_utc(
        local_manifest_generated_at_utc, "local_manifest_generated_at_utc"
    )
    confirmed_text, confirmed_time = _canonical_utc(
        official_confirmed_at_utc, "official_confirmed_at_utc"
    )
    if confirmed_time > generated_time:
        raise ValueError("官方期号确认时间不得晚于manifest生成时间")
    if not official_source_url.startswith("https://"):
        raise ValueError("官方期号来源必须是HTTPS URL")
    if not GIT_SHA_PATTERN.fullmatch(git_commit_sha):
        raise ValueError("当前Git提交SHA必须是40位小写十六进制")
    history_issues, history_draws = validate_unpublished_target(
        issues, draws, target_issue
    )
    latest_issue = int(history_issues[-1])
    position = next_chain_position(
        project_root=project_root,
        manifest_dir=manifest_dir,
        results_dir=results_dir,
        config=config,
        target_issue=target_issue,
        history_issues=history_issues,
        history_draws=history_draws,
    )
    prediction = predict_frozen_models(issues, draws, target_issue=target_issue)

    model_payload: dict[str, object] = {}
    for strategy in PROBABILITY_STRATEGIES:
        payload: dict[str, object] = {
            "number_index": list(range(1, NUMBER_COUNT + 1)),
            "probabilities": [
                float(value) for value in prediction.probabilities[strategy]
            ],
        }
        if strategy == UNIFORM_STRATEGY:
            seeded: list[dict[str, object]] = []
            for base_seed, ranking in zip(
                UNIFORM_BASE_SEEDS, prediction.uniform_rankings, strict=True
            ):
                digest = hashlib.sha256(
                    _uniform_seed_material(target_issue, base_seed)
                ).digest()
                seeded.append(
                    {
                        "base_seed": base_seed,
                        "seed_material_sha256": digest.hex(),
                        "seed_integer": int.from_bytes(
                            digest[:8], byteorder="big", signed=False
                        ),
                        "ranking": list(map(int, ranking)),
                        "top_k_candidates": _top_k_candidates(ranking),
                    }
                )
            payload["rankings_by_seed"] = seeded
        else:
            ranking = prediction.rankings[strategy]
            payload["ranking"] = list(map(int, ranking))
            payload["top_k_candidates"] = _top_k_candidates(ranking)
        model_payload[strategy] = payload

    if latest_issue != int(history_issues[-1]):
        raise RuntimeError("history_through_issue未使用输入数据最新期号")
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": MANIFEST_EVIDENCE_STATUS,
        "remote_preseal_verified": False,
        "freeze_id": FREEZE_ID,
        "freeze_config_sha256": config["configuration_sha256"],
        "confirmation_index": position.confirmation_index,
        "protocol_start_target_issue": position.protocol_start_target_issue,
        "previous_target_issue": position.previous_target_issue,
        "previous_manifest_sha256": position.previous_manifest_sha256,
        "previous_evaluation_target_issue": (position.previous_evaluation_target_issue),
        "previous_evaluation_confirmation_index": (
            position.previous_evaluation_confirmation_index
        ),
        "previous_evaluation_path": position.previous_evaluation_path,
        "previous_evaluation_sha256": position.previous_evaluation_sha256,
        "target_issue": target_issue,
        "local_manifest_generated_at_utc": generated_text,
        "history_through_issue": latest_issue,
        "history_issue_count": len(history_issues),
        "input_data": {
            "path": _relative_posix(project_root, resolved_data),
            "canonical_target_prior_sha256": canonical_history_sha256(
                history_issues, history_draws
            ),
            "canonicalization": "issue_ascending_numbers_ascending_utf8_lf",
        },
        "git_commit_sha": git_commit_sha,
        "official_issue_confirmation": {
            "source_url": official_source_url,
            "confirmed_at_utc": confirmed_text,
            "confirmation_method": "explicit_official_source_not_integer_inference",
        },
        "source_manifest": config["source_manifest"],
        "frozen_parameters": config["models"],
        "selected_target_prior_parameters": {
            DYNAMIC_STRATEGY: {
                "decay": prediction.dynamic_decay,
                "prior_strength": prediction.dynamic_prior_strength,
            },
            CHANGEPOINT_STRATEGY: {
                "recent_window": prediction.changepoint_recent_window,
                "change_score": prediction.changepoint_change_score,
                "change_threshold": prediction.changepoint_threshold,
                "state": (
                    "high_change" if prediction.changepoint_high_change else "normal"
                ),
                "active_decay": prediction.changepoint_active_decay,
                "effective_history_length": (
                    prediction.changepoint_effective_history_length
                ),
            },
        },
        "models": model_payload,
    }
    return record


def write_manifest_exclusive(
    manifest: Mapping[str, object], manifest_dir: Path
) -> Path:
    """以独占创建模式写入 manifest，已存在时拒绝覆盖。"""

    target_issue = manifest.get("target_issue")
    if not isinstance(target_issue, int) or target_issue <= 0:
        raise ValueError("manifest缺少合法target_issue")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{target_issue}.json"
    try:
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"manifest已存在，拒绝覆盖：{path}") from exc
    return path


def _manifest_model_arrays(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, FloatArray], dict[str, IntArray], tuple[IntArray, ...]]:
    models = _require_mapping(manifest.get("models"), "manifest.models")
    if tuple(sorted(models)) != tuple(sorted(PROBABILITY_STRATEGIES)):
        raise ValueError("manifest未完整包含五个冻结模型")
    probabilities: dict[str, FloatArray] = {}
    rankings: dict[str, IntArray] = {}
    uniform_rankings: list[IntArray] = []
    target = manifest.get("target_issue")
    if not isinstance(target, int):
        raise ValueError("manifest target_issue非法")
    for strategy in PROBABILITY_STRATEGIES:
        model = _require_mapping(models[strategy], f"manifest.models.{strategy}")
        probability_array = cast(
            FloatArray, np.asarray(model.get("probabilities"), dtype=np.float64)
        )
        if probability_array.shape != (NUMBER_COUNT,) or not np.all(
            (probability_array > 0.0) & (probability_array < 1.0)
        ):
            raise ValueError(f"manifest中的{strategy}概率非法")
        if not np.isclose(probability_array.sum(), DRAW_SIZE, atol=1e-9, rtol=0.0):
            raise ValueError(f"manifest中的{strategy}概率和不等于20")
        probabilities[strategy] = probability_array
        if strategy == UNIFORM_STRATEGY:
            seeded = model.get("rankings_by_seed")
            if not isinstance(seeded, list) or len(seeded) != len(UNIFORM_BASE_SEEDS):
                raise ValueError("uniform必须包含20组预注册seed排名")
            expected_rankings = uniform_seed_rankings(target)
            for offset, (raw, base_seed, expected) in enumerate(
                zip(seeded, UNIFORM_BASE_SEEDS, expected_rankings, strict=True)
            ):
                row = _require_mapping(raw, f"uniform.seed[{offset}]")
                ranking = cast(IntArray, np.asarray(row.get("ranking"), dtype=np.int64))
                digest = hashlib.sha256(
                    _uniform_seed_material(target, base_seed)
                ).digest()
                if row.get("base_seed") != base_seed:
                    raise ValueError("uniform base_seed顺序或取值漂移")
                if row.get("seed_material_sha256") != digest.hex():
                    raise ValueError("uniform seed material哈希不匹配")
                if row.get("seed_integer") != int.from_bytes(
                    digest[:8], byteorder="big", signed=False
                ):
                    raise ValueError("uniform seed integer不匹配")
                if not np.array_equal(ranking, expected):
                    raise ValueError("uniform随机排名不可复现或被篡改")
                uniform_rankings.append(ranking)
        else:
            ranking = cast(IntArray, np.asarray(model.get("ranking"), dtype=np.int64))
            if set(map(int, ranking)) != set(range(1, NUMBER_COUNT + 1)):
                raise ValueError(f"manifest中的{strategy}排名非法")
            rankings[strategy] = ranking
    return probabilities, rankings, tuple(uniform_rankings)


def _outcomes(actual_numbers: IntArray, target_issue: int) -> FloatArray:
    numbers = cast(IntArray, np.asarray(actual_numbers, dtype=np.int64))
    if numbers.shape != (DRAW_SIZE,):
        raise ValueError("实际开奖号码必须恰好20个")
    validate_issue_draws(
        np.asarray([target_issue], dtype=np.int64), numbers.reshape(1, DRAW_SIZE)
    )
    outcomes = np.zeros(NUMBER_COUNT, dtype=np.float64)
    outcomes[numbers - 1] = 1.0
    return outcomes


def calculate_manifest_metrics(
    manifest: Mapping[str, Any], actual_numbers: IntArray
) -> dict[str, object]:
    """从manifest概率与排名独立复算五模型逐期指标。"""

    target = manifest.get("target_issue")
    if not isinstance(target, int):
        raise ValueError("manifest目标期非法")
    outcomes = _outcomes(actual_numbers, target)
    probabilities, rankings, uniform_rankings = _manifest_model_arrays(manifest)
    model_metrics: dict[str, object] = {}
    for strategy in PROBABILITY_STRATEGIES:
        topk: dict[str, object] = {}
        for k in range(1, 11):
            if strategy == UNIFORM_STRATEGY:
                seed_hits = np.asarray(
                    [
                        top_k_hits(ranking, outcomes, k)[0]
                        for ranking in uniform_rankings
                    ],
                    dtype=np.float64,
                )
                hits = float(seed_hits.mean())
                topk[str(k)] = {
                    "hits": hits,
                    "random_expected_hits": k / 4.0,
                    "excess_hits": hits - k / 4.0,
                    "aggregation": "within_issue_mean_over_20_seeds",
                    "seed_count": len(UNIFORM_BASE_SEEDS),
                }
            else:
                hits, expected, excess = top_k_hits(rankings[strategy], outcomes, k)
                topk[str(k)] = {
                    "hits": hits,
                    "random_expected_hits": expected,
                    "excess_hits": excess,
                }
        model_metrics[strategy] = {
            "brier_score": brier_score(probabilities[strategy], outcomes),
            "bernoulli_log_loss": bernoulli_log_loss(probabilities[strategy], outcomes),
            "top_k": topk,
        }
    return model_metrics


def calculate_evaluation_comparisons(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """从五模型逐期指标确定性复算所有预注册差值。"""

    uniform_metrics = _require_mapping(metrics[UNIFORM_STRATEGY], UNIFORM_STRATEGY)
    uniform_brier = float(uniform_metrics["brier_score"])
    uniform_log_loss = float(uniform_metrics["bernoulli_log_loss"])
    versus_uniform: dict[str, object] = {}
    for strategy in PRIMARY_COMPARISON_MODELS:
        model = _require_mapping(metrics[strategy], strategy)
        versus_uniform[strategy] = {
            "brier_difference_model_minus_uniform": (
                float(model["brier_score"]) - uniform_brier
            ),
            "log_loss_difference_model_minus_uniform": (
                float(model["bernoulli_log_loss"]) - uniform_log_loss
            ),
        }
    changepoint = _require_mapping(metrics[CHANGEPOINT_STRATEGY], CHANGEPOINT_STRATEGY)
    versus_fixed: dict[str, object] = {}
    for strategy in (FIXED_NORMAL_STRATEGY, FIXED_HIGH_STRATEGY):
        fixed = _require_mapping(metrics[strategy], strategy)
        versus_fixed[strategy] = {
            "brier_difference_changepoint_minus_fixed": (
                float(changepoint["brier_score"]) - float(fixed["brier_score"])
            ),
            "log_loss_difference_changepoint_minus_fixed": (
                float(changepoint["bernoulli_log_loss"])
                - float(fixed["bernoulli_log_loss"])
            ),
        }
    return {
        "versus_uniform": versus_uniform,
        "changepoint_versus_fixed_baselines": versus_fixed,
    }


def verify_remote_preseal(
    *,
    client: GitHubSealClient,
    repository: str,
    base_branch: str,
    pr_number: int,
    manifest_path: Path,
    manifest_repository_path: str,
    official_result_published_at_utc: str,
    previous_evaluation_path: Path | None = None,
    previous_evaluation_repository_path: str | None = None,
    previous_evaluation_expected_sha256: str | None = None,
) -> RemoteSealEvidence:
    """通过GitHub API验证合并时文件原始字节，任一失败都抛错。"""

    if pr_number <= 0:
        raise ValueError("seal_pr_number必须为正整数")
    published_text, published_time = _canonical_utc(
        official_result_published_at_utc, "official_result_published_at_utc"
    )
    try:
        metadata = client.get_pull_request(repository, pr_number)
    except Exception as exc:
        raise RuntimeError("GitHub API不可用，远程封存验证失败") from exc
    if metadata.get("repository") != repository:
        raise ValueError("seal PR不属于冻结GitHub仓库")
    if metadata.get("number") != pr_number:
        raise ValueError("GitHub返回的PR编号不匹配")
    if metadata.get("state") != "MERGED":
        raise ValueError("seal PR尚未合并")
    if metadata.get("baseRefName") != base_branch:
        raise ValueError("seal PR base分支错误")
    merged_at = metadata.get("mergedAt")
    if not isinstance(merged_at, str):
        raise ValueError("seal PR缺少合并时间")
    merged_text, merged_time = _canonical_utc(merged_at, "seal_merged_at_utc")
    if merged_time >= published_time:
        raise ValueError("seal PR合并时间不早于官方结果发布时间")
    merge_commit = metadata.get("mergeCommit")
    if isinstance(merge_commit, dict):
        merge_commit = merge_commit.get("oid")
    if not isinstance(merge_commit, str) or not GIT_SHA_PATTERN.fullmatch(merge_commit):
        raise ValueError("seal PR缺少合法合并提交SHA")
    try:
        remote_bytes = client.get_file_bytes(
            repository, manifest_repository_path, merge_commit
        )
    except Exception as exc:
        raise RuntimeError("合并提交不含该期manifest或GitHub API不可用") from exc
    local_digest = raw_sha256(manifest_path)
    remote_digest = hashlib.sha256(remote_bytes).hexdigest()
    if remote_digest != local_digest:
        raise ValueError("合并提交中的manifest SHA-256与本地文件不符")
    previous_digest_at_merge: str | None = None
    previous_values = (
        previous_evaluation_path,
        previous_evaluation_repository_path,
        previous_evaluation_expected_sha256,
    )
    if any(value is not None for value in previous_values):
        if not all(value is not None for value in previous_values):
            raise ValueError("上一evaluation远程锚定参数不完整")
        assert previous_evaluation_path is not None
        assert previous_evaluation_repository_path is not None
        assert previous_evaluation_expected_sha256 is not None
        if raw_sha256(previous_evaluation_path) != previous_evaluation_expected_sha256:
            raise ValueError("上一evaluation本地SHA-256与当前manifest记录不符")
        try:
            previous_remote_bytes = client.get_file_bytes(
                repository, previous_evaluation_repository_path, merge_commit
            )
        except Exception as exc:
            raise RuntimeError("合并提交不含上一evaluation或GitHub API不可用") from exc
        previous_digest_at_merge = hashlib.sha256(previous_remote_bytes).hexdigest()
        if previous_digest_at_merge != previous_evaluation_expected_sha256:
            raise ValueError("合并提交中的上一evaluation SHA-256不匹配")
    url = metadata.get("url")
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        raise ValueError("seal PR URL非法")
    return RemoteSealEvidence(
        seal_pr_number=pr_number,
        seal_pr_url=url,
        seal_merge_commit_sha=merge_commit,
        seal_merged_at_utc=merged_text,
        manifest_sha256_at_merge=remote_digest,
        previous_evaluation_sha256_at_merge=previous_digest_at_merge,
    )


def verify_remote_file_anchor(
    *,
    client: GitHubSealClient,
    repository: str,
    base_branch: str,
    pr_number: int,
    local_path: Path,
    repository_path: str,
) -> RemoteFileAnchorEvidence:
    """验证任一协议文件原始字节存在于固定仓库已合并PR。"""

    if pr_number <= 0:
        raise ValueError("evaluation_seal_pr_number必须为正整数")
    try:
        metadata = client.get_pull_request(repository, pr_number)
    except Exception as exc:
        raise RuntimeError("GitHub API不可用，final evaluation seal失败") from exc
    if metadata.get("repository") != repository:
        raise ValueError("final seal PR不属于冻结GitHub仓库")
    if metadata.get("number") != pr_number:
        raise ValueError("GitHub返回的final seal PR编号不匹配")
    if metadata.get("state") != "MERGED":
        raise ValueError("final seal PR尚未合并")
    if metadata.get("baseRefName") != base_branch:
        raise ValueError("final seal PR base分支错误")
    merged_at = metadata.get("mergedAt")
    if not isinstance(merged_at, str):
        raise ValueError("final seal PR缺少合并时间")
    merged_text, _ = _canonical_utc(merged_at, "final_seal_merged_at_utc")
    merge_commit = metadata.get("mergeCommit")
    if isinstance(merge_commit, dict):
        merge_commit = merge_commit.get("oid")
    if not isinstance(merge_commit, str) or not GIT_SHA_PATTERN.fullmatch(merge_commit):
        raise ValueError("final seal PR缺少合法合并提交SHA")
    try:
        remote_bytes = client.get_file_bytes(repository, repository_path, merge_commit)
    except Exception as exc:
        raise RuntimeError(
            "final seal合并提交不含第365期evaluation或GitHub API不可用"
        ) from exc
    local_digest = raw_sha256(local_path)
    remote_digest = hashlib.sha256(remote_bytes).hexdigest()
    if remote_digest != local_digest:
        raise ValueError("final seal合并提交中的evaluation SHA-256与本地不符")
    url = metadata.get("url")
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        raise ValueError("final seal PR URL非法")
    return RemoteFileAnchorEvidence(
        seal_pr_number=pr_number,
        seal_pr_url=url,
        seal_merge_commit_sha=merge_commit,
        seal_merged_at_utc=merged_text,
        file_sha256_at_merge=remote_digest,
    )


def build_final_evaluation_seal(
    *,
    project_root: Path,
    config_path: Path,
    results_dir: Path,
    target_issue: int,
    evaluation_seal_pr_number: int,
    seal_client: GitHubSealClient,
) -> dict[str, object]:
    """为没有下一份manifest的第365期evaluation形成最终远程锚点。"""

    config = load_and_verify_freeze_config(project_root, config_path)
    if config.get("freeze_status") != FREEZE_ACTIVE:
        raise RuntimeError("冻结配置仍为pending，拒绝finalize evaluation")
    paths = _numeric_evaluation_paths(results_dir)
    if len(paths) != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("finalize前必须恰好存在365份逐期evaluation")
    evaluation_path = results_dir / f"{target_issue}.json"
    if evaluation_path not in paths:
        raise ValueError("finalize目标evaluation不存在")
    evaluation = _load_canonical_json_object(evaluation_path, "第365期evaluation")
    if evaluation.get("target_issue") != target_issue:
        raise ValueError("finalize target_issue与evaluation不一致")
    if evaluation.get("confirmation_index") != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("finalize只允许第365期evaluation")
    if evaluation.get("freeze_id") != FREEZE_ID:
        raise ValueError("第365期evaluation freeze_id不匹配")
    if evaluation.get("evidence_status") != EVALUATION_EVIDENCE_STATUS:
        raise ValueError("第365期evaluation证据状态不匹配")
    if evaluation.get("remote_preseal_verified") is not True:
        raise ValueError("第365期manifest未通过开奖前远程封存")
    if evaluation.get("sealed_before_official_result") is not True:
        raise ValueError("第365期manifest不是开奖前远程封存")
    if evaluation.get("evaluation_locally_created") is not True:
        raise ValueError("第365期evaluation缺少本地创建标记")
    if evaluation.get("remote_evaluation_anchor_pending") is not True:
        raise ValueError("第365期evaluation不处于待远程锚定状态")
    _validate_evaluation_payload_fingerprint(evaluation)
    repository_path = _relative_posix(project_root, evaluation_path)
    github = _require_mapping(config["github"], "github")
    anchor = verify_remote_file_anchor(
        client=seal_client,
        repository=str(github["repository"]),
        base_branch=str(github["base_branch"]),
        pr_number=evaluation_seal_pr_number,
        local_path=evaluation_path,
        repository_path=repository_path,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": FINAL_EVALUATION_SEAL_EVIDENCE_STATUS,
        "freeze_id": FREEZE_ID,
        "target_issue": target_issue,
        "confirmation_index": CONFIRMATION_ISSUE_COUNT,
        "evaluation_path": repository_path,
        "evaluation_sha256": raw_sha256(evaluation_path),
        "seal_pr_number": anchor.seal_pr_number,
        "seal_pr_url": anchor.seal_pr_url,
        "seal_merge_commit_sha": anchor.seal_merge_commit_sha,
        "seal_merged_at_utc": anchor.seal_merged_at_utc,
        "remote_evaluation_anchor_verified": True,
    }


def write_final_evaluation_seal_exclusive(
    seal: Mapping[str, object], results_dir: Path
) -> Path:
    """独占写入第365期final evaluation seal。"""

    if seal.get("remote_evaluation_anchor_verified") is not True:
        raise ValueError("未通过远程锚定验证的final seal不得写入")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / FINAL_EVALUATION_SEAL_FILENAME
    try:
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(seal))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"final evaluation seal已存在，拒绝覆盖：{path}") from exc
    return path


def build_evaluation_record(
    *,
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    actual_numbers: IntArray,
    seal_pr_number: int,
    official_result_source_url: str,
    official_result_published_at_utc: str,
    evaluated_at_utc: str,
    seal_client: GitHubSealClient,
) -> dict[str, object]:
    """远程封存验证成功后，才从已封存概率构造开奖后评价。"""

    config = load_and_verify_freeze_config(project_root, config_path)
    if config.get("freeze_status") != FREEZE_ACTIVE:
        raise RuntimeError("冻结配置仍为pending，拒绝构造生产evaluation")
    if not official_result_source_url.startswith("https://"):
        raise ValueError("官方结果来源必须是HTTPS URL")
    resolved_manifest = resolve_within(project_root, manifest_path, "manifest")
    manifest = _load_json_object(resolved_manifest)
    if manifest.get("evidence_status") != MANIFEST_EVIDENCE_STATUS:
        raise ValueError("manifest证据状态不符合本地待远程封存协议")
    if manifest.get("remote_preseal_verified") is not False:
        raise ValueError("manifest不得预先声称远程封存")
    if manifest.get("freeze_id") != config.get("freeze_id"):
        raise ValueError("manifest freeze_id与当前冻结配置不一致")
    if manifest.get("freeze_config_sha256") != config.get("configuration_sha256"):
        raise ValueError("manifest冻结配置SHA-256与当前配置不一致")
    if manifest.get("source_manifest") != config.get("source_manifest"):
        raise ValueError("manifest source manifest与当前冻结配置不一致")
    target_issue = manifest.get("target_issue")
    if not isinstance(target_issue, int) or target_issue <= 0:
        raise ValueError("manifest目标期号非法")
    chain_entries = _load_manifest_chain(resolved_manifest.parent, config)
    matching_entries = [
        entry for entry in chain_entries if entry.path.resolve() == resolved_manifest
    ]
    if len(matching_entries) != 1:
        raise ValueError("待评价manifest不在完整有效的365期顺序链中")
    confirmation_index = manifest.get("confirmation_index")
    if not isinstance(confirmation_index, int):
        raise ValueError("manifest confirmation_index非法")
    published_text, published_time = _canonical_utc(
        official_result_published_at_utc, "official_result_published_at_utc"
    )
    evaluated_text, evaluated_time = _canonical_utc(
        evaluated_at_utc, "evaluated_at_utc"
    )
    if evaluated_time < published_time:
        raise ValueError("评价时间不得早于官方结果发布时间")
    repository_path = _relative_posix(project_root, resolved_manifest)
    previous_evaluation_path: Path | None = None
    previous_evaluation_repository_path: str | None = None
    previous_evaluation_sha256: str | None = None
    if confirmation_index > 1:
        raw_previous_path = manifest.get("previous_evaluation_path")
        raw_previous_sha = manifest.get("previous_evaluation_sha256")
        if not isinstance(raw_previous_path, str) or not isinstance(
            raw_previous_sha, str
        ):
            raise ValueError("当前manifest缺少上一evaluation锚定字段")
        previous_evaluation_path = resolve_within(
            project_root, Path(raw_previous_path), "上一evaluation"
        )
        previous_evaluation_repository_path = raw_previous_path
        previous_evaluation_sha256 = raw_previous_sha
    github = _require_mapping(config["github"], "github")
    seal = verify_remote_preseal(
        client=seal_client,
        repository=str(github["repository"]),
        base_branch=str(github["base_branch"]),
        pr_number=seal_pr_number,
        manifest_path=resolved_manifest,
        manifest_repository_path=repository_path,
        official_result_published_at_utc=published_text,
        previous_evaluation_path=previous_evaluation_path,
        previous_evaluation_repository_path=previous_evaluation_repository_path,
        previous_evaluation_expected_sha256=previous_evaluation_sha256,
    )
    numbers = cast(IntArray, np.asarray(actual_numbers, dtype=np.int64))
    metrics = calculate_manifest_metrics(manifest, numbers)
    comparisons = calculate_evaluation_comparisons(metrics)
    selected = _require_mapping(
        manifest.get("selected_target_prior_parameters"),
        "selected_target_prior_parameters",
    )
    changepoint_selected = _require_mapping(
        selected.get(CHANGEPOINT_STRATEGY), CHANGEPOINT_STRATEGY
    )
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVALUATION_EVIDENCE_STATUS,
        "freeze_id": FREEZE_ID,
        "confirmation_index": manifest["confirmation_index"],
        "target_issue": target_issue,
        "evaluated_at_utc": evaluated_text,
        "official_result_source_url": official_result_source_url,
        "official_result_published_at_utc": published_text,
        "actual_numbers": sorted(map(int, numbers)),
        "evaluation_locally_created": True,
        "remote_evaluation_anchor_pending": True,
        "remote_preseal_verified": True,
        "sealed_before_official_result": True,
        "seal_pr_number": seal.seal_pr_number,
        "seal_pr_url": seal.seal_pr_url,
        "seal_merge_commit_sha": seal.seal_merge_commit_sha,
        "seal_merged_at_utc": seal.seal_merged_at_utc,
        "manifest_sha256_at_merge": seal.manifest_sha256_at_merge,
        "previous_evaluation_remote_anchor_verified": (
            True if confirmation_index > 1 else None
        ),
        "previous_evaluation_anchor": (
            {
                "target_issue": manifest["previous_evaluation_target_issue"],
                "confirmation_index": manifest[
                    "previous_evaluation_confirmation_index"
                ],
                "path": previous_evaluation_repository_path,
                "sha256": previous_evaluation_sha256,
                "sha256_at_merge": seal.previous_evaluation_sha256_at_merge,
            }
            if confirmation_index > 1
            else None
        ),
        "manifest": {
            "path": repository_path,
            "sha256": raw_sha256(resolved_manifest),
            "local_manifest_generated_at_utc": manifest[
                "local_manifest_generated_at_utc"
            ],
        },
        "changepoint_state": changepoint_selected.get("state"),
        "models": metrics,
        "comparisons": comparisons,
    }
    record["evaluation_payload_sha256"] = _evaluation_payload_fingerprint(record)
    return record


def write_evaluation_exclusive(record: Mapping[str, object], results_dir: Path) -> Path:
    """每期使用独立不可覆盖 JSON；同一期第二次写入必定失败。"""

    target_issue = record.get("target_issue")
    if not isinstance(target_issue, int) or target_issue <= 0:
        raise ValueError("评价记录缺少合法target_issue")
    if record.get("remote_preseal_verified") is not True:
        raise ValueError("未经远程封存验证的评价不得写入")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{target_issue}.json"
    try:
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"该期评价已存在，拒绝重复写入：{path}") from exc
    return path


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """对固定的四项主要比较执行确定性 Holm 校正。"""

    if tuple(sorted(p_values)) != tuple(sorted(PRIMARY_COMPARISON_MODELS)):
        raise ValueError("Holm校正必须一次包含全部四项主要比较")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for index, (name, raw_value) in enumerate(ordered):
        if not np.isfinite(raw_value) or not 0.0 <= raw_value <= 1.0:
            raise ValueError("原始p值必须位于[0,1]")
        candidate = min(1.0, (count - index) * raw_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def moving_block_bootstrap_inference(values: FloatArray) -> BootstrapInference:
    """按固定30期循环连续块、20000次重采样执行单侧主要检验。"""

    deltas = cast(FloatArray, np.asarray(values, dtype=np.float64))
    if deltas.shape != (CONFIRMATION_ISSUE_COUNT,) or not np.isfinite(deltas).all():
        raise ValueError("移动分块bootstrap必须接收365个有限逐期差值")
    observed = float(deltas.mean())
    centered = deltas - observed
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    block_count = math.ceil(CONFIRMATION_ISSUE_COUNT / BOOTSTRAP_BLOCK_LENGTH)
    sample_means = np.empty(BOOTSTRAP_RESAMPLE_COUNT, dtype=np.float64)
    null_means = np.empty_like(sample_means)
    offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH, dtype=np.int64)
    batch_size = 500
    for first in range(0, BOOTSTRAP_RESAMPLE_COUNT, batch_size):
        size = min(batch_size, BOOTSTRAP_RESAMPLE_COUNT - first)
        starts = rng.integers(
            0,
            CONFIRMATION_ISSUE_COUNT,
            size=(size, block_count),
            dtype=np.int64,
        )
        indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % CONFIRMATION_ISSUE_COUNT
        flattened = indices.reshape(size, -1)[:, :CONFIRMATION_ISSUE_COUNT]
        sample_means[first : first + size] = deltas[flattened].mean(axis=1)
        null_means[first : first + size] = centered[flattened].mean(axis=1)
    raw_p = (1.0 + float(np.count_nonzero(null_means <= observed))) / (
        BOOTSTRAP_RESAMPLE_COUNT + 1.0
    )
    lower, upper = np.quantile(sample_means, [0.025, 0.975], method="linear")
    return BootstrapInference(
        observed_mean=observed,
        ordinary_standard_error=float(
            deltas.std(ddof=1) / np.sqrt(CONFIRMATION_ISSUE_COUNT)
        ),
        interval_lower=float(lower),
        interval_upper=float(upper),
        raw_p_value=raw_p,
    )


def _validate_completed_chain(
    manifest_dir: Path,
    results_dir: Path,
    config: Mapping[str, Any],
    official_issues: IntArray,
    official_draws: IntArray,
) -> tuple[tuple[ManifestChainEntry, dict[str, Any], IntArray], ...]:
    issue_values, draw_values = validate_issue_draws(official_issues, official_draws)
    entries = _load_manifest_chain(manifest_dir, config)
    if len(entries) != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("正式汇总必须恰好包含365份manifest")
    paths = _numeric_evaluation_paths(results_dir)
    expected_names = {f"{entry.target_issue}.json" for entry in entries}
    if (
        len(paths) != CONFIRMATION_ISSUE_COUNT
        or {path.name for path in paths} != expected_names
    ):
        raise ValueError("正式汇总必须恰好包含365份一一对应的evaluation")
    allowed_non_evaluation = {
        FINAL_EVALUATION_SEAL_FILENAME,
        FORMAL_SUMMARY_FILENAME,
    }
    all_json_names = (
        {path.name for path in results_dir.glob("*.json")}
        if results_dir.exists()
        else set()
    )
    unexpected = all_json_names - expected_names - allowed_non_evaluation
    if unexpected:
        raise ValueError("结果目录包含额外协议JSON文件")
    records = {
        path.stem: _load_canonical_json_object(path, "evaluation") for path in paths
    }
    paired: list[tuple[ManifestChainEntry, dict[str, Any], IntArray]] = []
    for entry in entries:
        record = records[str(entry.target_issue)]
        if record.get("target_issue") != entry.target_issue:
            raise ValueError("evaluation target_issue与manifest不匹配")
        if record.get("confirmation_index") != entry.confirmation_index:
            raise ValueError("evaluation confirmation_index与manifest不匹配")
        if record.get("freeze_id") != FREEZE_ID:
            raise ValueError("evaluation freeze_id不匹配")
        if record.get("evidence_status") != EVALUATION_EVIDENCE_STATUS:
            raise ValueError("evaluation证据状态不正确")
        if record.get("remote_preseal_verified") is not True:
            raise ValueError("evaluation未通过远程封存验证")
        if record.get("sealed_before_official_result") is not True:
            raise ValueError("evaluation不是开奖前远程封存")
        if record.get("evaluation_locally_created") is not True:
            raise ValueError("evaluation缺少本地独占创建标记")
        if record.get("remote_evaluation_anchor_pending") is not True:
            raise ValueError("evaluation缺少待远程锚定标记")
        _validate_evaluation_payload_fingerprint(record)
        manifest_info = _require_mapping(record.get("manifest"), "evaluation.manifest")
        if manifest_info.get("sha256") != entry.sha256:
            raise ValueError("evaluation记录的manifest SHA与真实文件不一致")
        if record.get("manifest_sha256_at_merge") != entry.sha256:
            raise ValueError("合并提交manifest SHA与真实文件不一致")
        matches = np.flatnonzero(issue_values == entry.target_issue)
        if len(matches) != 1:
            raise ValueError("正式数据中365个目标期必须各恰好出现一次")
        official_actual = cast(IntArray, draw_values[int(matches[0])].copy())
        if record.get("actual_numbers") != sorted(map(int, official_actual)):
            raise ValueError("evaluation actual_numbers与正式输入数据不一致")
        paired.append((entry, record, official_actual))

    for offset in range(CONFIRMATION_ISSUE_COUNT - 1):
        entry, _, _ = paired[offset]
        next_entry, next_record, _ = paired[offset + 1]
        evaluation_path = results_dir / f"{entry.target_issue}.json"
        evaluation_digest = raw_sha256(evaluation_path)
        expected_path = (RESULT_RELATIVE_DIR / f"{entry.target_issue}.json").as_posix()
        next_manifest = next_entry.payload
        if next_manifest.get("previous_evaluation_target_issue") != entry.target_issue:
            raise ValueError("下一manifest未锚定上一evaluation target_issue")
        if next_manifest.get("previous_evaluation_confirmation_index") != (
            entry.confirmation_index
        ):
            raise ValueError("下一manifest未锚定上一evaluation confirmation_index")
        if next_manifest.get("previous_evaluation_path") != expected_path:
            raise ValueError("下一manifest未锚定上一evaluation路径")
        if next_manifest.get("previous_evaluation_sha256") != evaluation_digest:
            raise ValueError("下一manifest记录的上一evaluation SHA-256不匹配")
        if next_record.get("previous_evaluation_remote_anchor_verified") is not True:
            raise ValueError("上一evaluation未由下一manifest seal PR远程锚定")
        anchor = _require_mapping(
            next_record.get("previous_evaluation_anchor"),
            "previous_evaluation_anchor",
        )
        if anchor != {
            "target_issue": entry.target_issue,
            "confirmation_index": entry.confirmation_index,
            "path": expected_path,
            "sha256": evaluation_digest,
            "sha256_at_merge": evaluation_digest,
        }:
            raise ValueError("下一manifest seal PR的上一evaluation远程锚点不匹配")

    final_entry, _, _ = paired[-1]
    final_evaluation_path = results_dir / f"{final_entry.target_issue}.json"
    final_digest = raw_sha256(final_evaluation_path)
    final_seal_path = results_dir / FINAL_EVALUATION_SEAL_FILENAME
    if not final_seal_path.is_file():
        raise ValueError("第365期evaluation缺少final evaluation seal")
    final_seal = _load_canonical_json_object(final_seal_path, "final evaluation seal")
    if final_seal.get("evidence_status") != FINAL_EVALUATION_SEAL_EVIDENCE_STATUS:
        raise ValueError("final evaluation seal证据状态不匹配")
    if final_seal.get("freeze_id") != FREEZE_ID:
        raise ValueError("final evaluation seal freeze_id不匹配")
    if final_seal.get("target_issue") != final_entry.target_issue:
        raise ValueError("final evaluation seal target_issue不匹配")
    if final_seal.get("confirmation_index") != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("final evaluation seal confirmation_index不匹配")
    expected_final_path = (
        RESULT_RELATIVE_DIR / f"{final_entry.target_issue}.json"
    ).as_posix()
    if final_seal.get("evaluation_path") != expected_final_path:
        raise ValueError("final evaluation seal路径不匹配")
    if final_seal.get("evaluation_sha256") != final_digest:
        raise ValueError("final evaluation seal SHA-256与第365期evaluation不匹配")
    if final_seal.get("remote_evaluation_anchor_verified") is not True:
        raise ValueError("第365期evaluation未通过最终远程锚定")
    if (
        not isinstance(final_seal.get("seal_pr_number"), int)
        or int(final_seal["seal_pr_number"]) <= 0
    ):
        raise ValueError("final evaluation seal PR编号非法")
    if not isinstance(final_seal.get("seal_pr_url"), str) or not str(
        final_seal["seal_pr_url"]
    ).startswith("https://github.com/"):
        raise ValueError("final evaluation seal PR URL非法")
    merge_sha = final_seal.get("seal_merge_commit_sha")
    if not isinstance(merge_sha, str) or not GIT_SHA_PATTERN.fullmatch(merge_sha):
        raise ValueError("final evaluation seal合并提交SHA非法")
    merged_at = final_seal.get("seal_merged_at_utc")
    if not isinstance(merged_at, str):
        raise ValueError("final evaluation seal缺少合并时间")
    _canonical_utc(merged_at, "final evaluation seal合并时间")
    return tuple(paired)


def _serialize_calibration(
    probabilities: FloatArray, outcomes: FloatArray
) -> dict[str, object]:
    result = calibration_summary(probabilities, outcomes, bins=10)
    return {
        "expected_calibration_error": result.expected_calibration_error,
        "bins": [
            {
                "bin_index": row.bin_index,
                "lower_bound": row.lower_bound,
                "upper_bound": row.upper_bound,
                "count": row.count,
                "mean_predicted_probability": row.mean_predicted_probability,
                "actual_rate": row.actual_rate,
                "absolute_gap": row.absolute_gap,
                "weighted_gap": row.weighted_gap,
            }
            for row in result.bins
        ],
    }


def build_formal_summary(
    *,
    manifest_dir: Path,
    results_dir: Path,
    config: Mapping[str, Any],
    official_issues: IntArray,
    official_draws: IntArray,
) -> dict[str, object]:
    """验证完整365期链，并独立复算主要与全部次要指标。"""

    _validate_freeze_contract(config)
    if config.get("freeze_status") != FREEZE_ACTIVE:
        raise RuntimeError("冻结配置仍为pending，拒绝正式summary")
    paired = _validate_completed_chain(
        manifest_dir, results_dir, config, official_issues, official_draws
    )
    probabilities: dict[str, list[FloatArray]] = {
        strategy: [] for strategy in PROBABILITY_STRATEGIES
    }
    outcomes: list[FloatArray] = []
    brier_values: dict[str, list[float]] = {
        strategy: [] for strategy in PROBABILITY_STRATEGIES
    }
    log_values: dict[str, list[float]] = {
        strategy: [] for strategy in PROBABILITY_STRATEGIES
    }
    topk_values: dict[str, dict[int, list[float]]] = {
        strategy: {k: [] for k in range(1, 11)} for strategy in PROBABILITY_STRATEGIES
    }
    issue_index: list[dict[str, object]] = []
    high_change_count = 0
    changepoint_deltas: dict[str, dict[str, list[float]]] = {
        FIXED_NORMAL_STRATEGY: {"brier": [], "log_loss": []},
        FIXED_HIGH_STRATEGY: {"brier": [], "log_loss": []},
    }
    for entry, record, actual in paired:
        outcome = _outcomes(actual, entry.target_issue)
        outcomes.append(outcome)
        model_probabilities, _, _ = _manifest_model_arrays(entry.payload)
        recomputed = calculate_manifest_metrics(entry.payload, actual)
        if record.get("models") != recomputed:
            raise ValueError("evaluation指标不能从manifest和实际结果独立复算")
        if record.get("comparisons") != calculate_evaluation_comparisons(recomputed):
            raise ValueError("evaluation comparisons不能从正式数据独立复算")
        for strategy in PROBABILITY_STRATEGIES:
            probabilities[strategy].append(model_probabilities[strategy])
            metrics = _require_mapping(recomputed[strategy], strategy)
            brier_values[strategy].append(float(metrics["brier_score"]))
            log_values[strategy].append(float(metrics["bernoulli_log_loss"]))
            topk = _require_mapping(metrics["top_k"], f"{strategy}.top_k")
            for k in range(1, 11):
                row = _require_mapping(topk[str(k)], f"{strategy}.top_k.{k}")
                topk_values[strategy][k].append(float(row["hits"]))
        if record.get("changepoint_state") == "high_change":
            high_change_count += 1
        cp_brier = brier_values[CHANGEPOINT_STRATEGY][-1]
        cp_log = log_values[CHANGEPOINT_STRATEGY][-1]
        for baseline in (FIXED_NORMAL_STRATEGY, FIXED_HIGH_STRATEGY):
            changepoint_deltas[baseline]["brier"].append(
                cp_brier - brier_values[baseline][-1]
            )
            changepoint_deltas[baseline]["log_loss"].append(
                cp_log - log_values[baseline][-1]
            )
        issue_index.append(
            {
                "confirmation_index": entry.confirmation_index,
                "target_issue": entry.target_issue,
                "manifest_sha256": entry.sha256,
                "evaluation_sha256": raw_sha256(
                    results_dir / f"{entry.target_issue}.json"
                ),
            }
        )

    outcome_matrix = cast(FloatArray, np.vstack(outcomes))
    model_summaries: dict[str, object] = {}
    for strategy in PROBABILITY_STRATEGIES:
        probability_matrix = cast(FloatArray, np.vstack(probabilities[strategy]))
        model_summaries[strategy] = {
            "mean_brier_score": float(np.mean(brier_values[strategy])),
            "mean_bernoulli_log_loss": float(np.mean(log_values[strategy])),
            "calibration": _serialize_calibration(probability_matrix, outcome_matrix),
            "mean_top_k_hits": {
                str(k): float(np.mean(topk_values[strategy][k])) for k in range(1, 11)
            },
            "top_k_aggregation": (
                "within_issue_mean_over_20_seeds_then_mean_over_365_issues"
                if strategy == UNIFORM_STRATEGY
                else "mean_over_365_issues"
            ),
        }

    bootstrap_results: dict[str, BootstrapInference] = {}
    raw_p_values: dict[str, float] = {}
    for strategy in PRIMARY_COMPARISON_MODELS:
        deltas = cast(
            FloatArray,
            np.asarray(brier_values[strategy], dtype=np.float64)
            - np.asarray(brier_values[UNIFORM_STRATEGY], dtype=np.float64),
        )
        inference = moving_block_bootstrap_inference(deltas)
        bootstrap_results[strategy] = inference
        raw_p_values[strategy] = inference.raw_p_value
    adjusted = holm_adjust(raw_p_values)
    primary: dict[str, object] = {}
    for strategy, inference in bootstrap_results.items():
        primary[strategy] = {
            "mean_brier_difference_model_minus_uniform": inference.observed_mean,
            "ordinary_standard_error": inference.ordinary_standard_error,
            "moving_block_bootstrap_95_interval": [
                inference.interval_lower,
                inference.interval_upper,
            ],
            "one_sided_p_value_unadjusted": inference.raw_p_value,
            "holm_adjusted_p_value": adjusted[strategy],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        "evidence_status": SUMMARY_EVIDENCE_STATUS,
        "issue_count": CONFIRMATION_ISSUE_COUNT,
        "primary_metric": "per_issue_80_dimensional_brier_score",
        "issue_is_statistical_unit": True,
        "primary_inference": {
            "alternative": "mean_model_minus_uniform_brier_less_than_zero",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "comparison_count": len(PRIMARY_COMPARISON_MODELS),
            "method": "circular_moving_block_bootstrap",
            "multiplicity_correction": "Holm",
            "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
            "seed": BOOTSTRAP_SEED,
            "comparisons": primary,
            "limitations": (
                "结论依赖时间平稳性和预注册30期分块长度假设；不得根据结果改变分块长度或另选检验。"
            ),
        },
        "secondary_metrics": {
            "status": "descriptive_not_for_model_selection",
            "models": model_summaries,
            "changepoint_versus_fixed_baselines": {
                baseline: {
                    "mean_brier_difference_changepoint_minus_fixed": float(
                        np.mean(values["brier"])
                    ),
                    "mean_log_loss_difference_changepoint_minus_fixed": float(
                        np.mean(values["log_loss"])
                    ),
                }
                for baseline, values in changepoint_deltas.items()
            },
            "high_change_trigger_count": high_change_count,
            "high_change_trigger_proportion": (
                high_change_count / CONFIRMATION_ISSUE_COUNT
            ),
            "issue_index": issue_index,
        },
    }


__all__ = [
    "BOOTSTRAP_BLOCK_LENGTH",
    "BOOTSTRAP_RESAMPLE_COUNT",
    "BOOTSTRAP_SEED",
    "CHANGEPOINT_STRATEGY",
    "CONFIRMATION_ISSUE_COUNT",
    "DATA_RELATIVE_PATH",
    "DYNAMIC_STRATEGY",
    "EVALUATION_EVIDENCE_STATUS",
    "FINAL_EVALUATION_SEAL_EVIDENCE_STATUS",
    "FINAL_EVALUATION_SEAL_FILENAME",
    "FIXED_HIGH_STRATEGY",
    "FIXED_NORMAL_STRATEGY",
    "FREEZE_ACTIVE",
    "FREEZE_ID",
    "FREEZE_PENDING",
    "FREEZE_RELATIVE_PATH",
    "FREEZE_TAG",
    "FORMAL_SUMMARY_FILENAME",
    "FROZEN_SOURCE_PATHS",
    "GITHUB_BASE_BRANCH",
    "GITHUB_REPOSITORY",
    "MANIFEST_EVIDENCE_STATUS",
    "MANIFEST_RELATIVE_DIR",
    "PRIMARY_COMPARISON_MODELS",
    "PROBABILITY_STRATEGIES",
    "RESULT_RELATIVE_DIR",
    "SCHEMA_VERSION",
    "SECONDARY_METRICS",
    "SUMMARY_EVIDENCE_STATUS",
    "UNIFORM_BASE_SEEDS",
    "UNIFORM_STRATEGY",
    "BootstrapInference",
    "GitHubSealClient",
    "ProspectivePrediction",
    "RemoteSealEvidence",
    "RemoteFileAnchorEvidence",
    "build_evaluation_record",
    "build_final_evaluation_seal",
    "build_formal_summary",
    "build_manifest",
    "calculate_manifest_metrics",
    "calculate_evaluation_comparisons",
    "canonical_history_sha256",
    "canonical_json_bytes",
    "configuration_fingerprint",
    "holm_adjust",
    "load_and_verify_freeze_config",
    "load_history_csv",
    "moving_block_bootstrap_inference",
    "next_chain_position",
    "normalized_lf_sha256",
    "predict_frozen_models",
    "raw_sha256",
    "require_active_freeze",
    "require_contract_path",
    "resolve_within",
    "source_manifest_fingerprint",
    "uniform_seed_rankings",
    "utc_now_string",
    "validate_unpublished_target",
    "verify_remote_preseal",
    "verify_remote_file_anchor",
    "write_evaluation_exclusive",
    "write_final_evaluation_seal_exclusive",
    "write_manifest_exclusive",
]
