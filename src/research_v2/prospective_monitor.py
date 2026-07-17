"""快乐8 v2 冻结模型的未来前瞻封存与逐期评价基础设施。

本模块只编排已经合并的第一、第二阶段模型，不重新定义或调整模型参数。
每个 manifest 只使用目标期之前的数据；开奖后评价只读取已封存 manifest 中的
概率，绝不重新预测或覆盖既有记录。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ttest_1samp

from .bayesian import DRAW_SIZE, FAIR_PROBABILITY, NUMBER_COUNT, candidate_sets
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
from .metrics import bernoulli_log_loss, brier_score, top_k_hits

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SCHEMA_VERSION = 1
EVIDENCE_STATUS = "prospective_presealed_research"
CONFIRMATION_ISSUE_COUNT = 365
UNIFORM_STRATEGY = "uniform_random"
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
MANIFEST_RELATIVE_DIR = Path("reports/research_v2_prospective_manifests")
RESULT_RELATIVE_DIR = Path("results/research_v2_prospective")
FREEZE_RELATIVE_PATH = Path("config/research_v2_prospective_freeze.json")
DATA_RELATIVE_PATH = Path("data_cache/kl8/data.csv")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ProspectivePrediction:
    """五个冻结模型在一个目标期的目标期前预测。"""

    probabilities: dict[str, FloatArray]
    rankings: dict[str, IntArray]
    dynamic_decay: float
    dynamic_prior_strength: float
    changepoint_recent_window: int
    changepoint_change_score: float
    changepoint_threshold: float
    changepoint_high_change: bool
    changepoint_active_decay: float
    changepoint_effective_history_length: int


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


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取合法JSON：{path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return cast(dict[str, Any], parsed)


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


def _validate_freeze_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v2前瞻冻结配置schema_version不受支持")
    if config.get("evidence_status") != EVIDENCE_STATUS:
        raise ValueError("v2前瞻证据状态不是预注册值")
    if config.get("confirmation_issue_count") != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("正式确认期数必须固定为365")
    models = _require_mapping(config.get("models"), "models")
    if tuple(sorted(models)) != tuple(sorted(PROBABILITY_STRATEGIES)):
        raise ValueError("冻结配置必须完整保留五个概率模型")
    if models[UNIFORM_STRATEGY] != {"probability": FAIR_PROBABILITY}:
        raise ValueError("uniform_random冻结参数漂移")
    expected_normal = {
        "decay": NORMAL_DECAY,
        "history": "all_target_prior_history",
        "prior_strength": CHANGEPOINT_PRIOR_STRENGTH,
    }
    expected_high = {
        "decay": HIGH_CHANGE_DECAY,
        "history": f"last_{MINIMUM_EFFECTIVE_HISTORY}_target_prior_issues",
        "prior_strength": CHANGEPOINT_PRIOR_STRENGTH,
    }
    if models[FIXED_NORMAL_STRATEGY] != expected_normal:
        raise ValueError("fixed_normal_bayesian冻结参数漂移")
    if models[FIXED_HIGH_STRATEGY] != expected_high:
        raise ValueError("fixed_high_bayesian冻结参数漂移")
    dynamic = _require_mapping(models[DYNAMIC_STRATEGY], DYNAMIC_STRATEGY)
    if dynamic.get("decay_grid") != [0.97, 0.99, 0.995]:
        raise ValueError("dynamic_bayesian decay网格漂移")
    if dynamic.get("prior_strength_grid") != [5.0, 20.0, 80.0]:
        raise ValueError("dynamic_bayesian prior网格漂移")
    if dynamic.get("minimum_initial_history") != MINIMUM_INITIAL_HISTORY:
        raise ValueError("dynamic_bayesian初始历史漂移")
    if dynamic.get("minimum_inner_observations") != MINIMUM_INNER_OBSERVATIONS:
        raise ValueError("dynamic_bayesian内层观察数漂移")
    changepoint = _require_mapping(models[CHANGEPOINT_STRATEGY], CHANGEPOINT_STRATEGY)
    expected_changepoint = {
        "change_quantile": CHANGE_QUANTILE,
        "fixed_high_model": FIXED_HIGH_STRATEGY,
        "fixed_normal_model": FIXED_NORMAL_STRATEGY,
        "minimum_effective_history": MINIMUM_EFFECTIVE_HISTORY,
        "recent_window_grid": list(RECENT_WINDOW_GRID),
        "reference_window": REFERENCE_WINDOW,
        "selection_metric": "inner_mean_brier",
        "threshold_training": "target_prior_change_scores_only",
    }
    if changepoint != expected_changepoint:
        raise ValueError("changepoint_bayesian冻结参数漂移")
    protocol = _require_mapping(
        config.get("confirmation_protocol"), "confirmation_protocol"
    )
    expected_primary = [
        "dynamic_bayesian_minus_uniform_random",
        "fixed_normal_bayesian_minus_uniform_random",
        "fixed_high_bayesian_minus_uniform_random",
        "changepoint_bayesian_minus_uniform_random",
    ]
    expected_secondary = [
        "bernoulli_log_loss",
        "fixed_bin_calibration_and_ece",
        "top_1_through_top_10_hits",
        "changepoint_minus_fixed_normal",
        "changepoint_minus_fixed_high",
        "high_change_trigger_proportion",
    ]
    expected_multiplicity = {
        "alternative": "mean_model_minus_uniform_brier_less_than_zero",
        "comparison_count": 4,
        "correction": "Holm",
        "raw_test": "one_sided_paired_issue_level_t_test",
    }
    if protocol.get("primary_metric") != "per_issue_80_dimensional_brier_score":
        raise ValueError("主要指标冻结契约漂移")
    if protocol.get("primary_comparisons") != expected_primary:
        raise ValueError("四项主要比较冻结契约漂移")
    if protocol.get("secondary_metrics") != expected_secondary:
        raise ValueError("次要指标冻结契约漂移")
    if protocol.get("multiplicity") != expected_multiplicity:
        raise ValueError("Holm多重比较冻结契约漂移")
    if any(
        protocol.get(key) is not True
        for key in (
            "early_success_claims_forbidden",
            "early_stopping_forbidden",
            "issue_is_statistical_unit",
            "model_changes_during_confirmation_forbidden",
            "secondary_metrics_cannot_select_models",
        )
    ):
        raise ValueError("365期禁止提前停止或改模的冻结契约漂移")
    if protocol.get("formal_summary_after_exact_issue_count") != (
        CONFIRMATION_ISSUE_COUNT
    ):
        raise ValueError("正式汇总期数冻结契约漂移")
    manifest_policy = _require_mapping(config.get("manifest_policy"), "manifest_policy")
    if manifest_policy != {
        "directory": MANIFEST_RELATIVE_DIR.as_posix(),
        "existing_manifest_policy": "refuse_overwrite_rewrite_or_delete",
        "must_commit_and_push_before_official_result": True,
    }:
        raise ValueError("manifest防覆盖或预封存策略漂移")
    evaluation_policy = _require_mapping(
        config.get("evaluation_policy"), "evaluation_policy"
    )
    if evaluation_policy != {
        "directory": RESULT_RELATIVE_DIR.as_posix(),
        "record_format": "one_immutable_json_per_target_issue",
        "duplicate_target_policy": "refuse_overwrite",
    }:
        raise ValueError("逐期开奖评价防覆盖策略漂移")


def load_and_verify_freeze_config(
    project_root: Path, config_path: Path
) -> dict[str, Any]:
    """读取冻结配置，并在任何预测或评价前拒绝源码漂移。"""

    resolved = require_contract_path(
        project_root, config_path, FREEZE_RELATIVE_PATH, "冻结配置路径"
    )
    config = _load_json_object(resolved)
    _validate_freeze_contract(config)
    source_manifest = _require_mapping(config.get("source_manifest"), "source_manifest")
    if source_manifest.get("algorithm") != "sha256_utf8_normalized_lf":
        raise ValueError("source manifest哈希算法不受支持")
    files = _require_mapping(source_manifest.get("files"), "source_manifest.files")
    expected_files: dict[str, str] = {}
    for raw_relative, raw_digest in files.items():
        if not isinstance(raw_relative, str) or not isinstance(raw_digest, str):
            raise ValueError("source manifest路径和哈希必须是字符串")
        if not SHA256_PATTERN.fullmatch(raw_digest):
            raise ValueError(f"source manifest含非法SHA-256：{raw_relative}")
        source_path = resolve_within(project_root, Path(raw_relative), "冻结源码")
        if not source_path.is_file():
            raise ValueError(f"冻结源码不存在：{raw_relative}")
        actual = normalized_lf_sha256(source_path)
        if actual != raw_digest:
            raise ValueError(f"冻结源码漂移，拒绝运行：{raw_relative}")
        expected_files[raw_relative] = raw_digest
    fingerprint = source_manifest_fingerprint(expected_files)
    if source_manifest.get("fingerprint") != fingerprint:
        raise ValueError("source manifest总指纹不匹配")
    return config


def canonical_history_sha256(issues: IntArray, draws: IntArray) -> str:
    """对目标期前历史形成与CSV行序和号码列顺序无关的规范哈希。"""

    issue_values, draw_values = validate_issue_draws(issues, draws)
    lines = []
    for issue, draw in zip(issue_values, draw_values, strict=True):
        numbers = ",".join(f"{number:02d}" for number in sorted(map(int, draw)))
        lines.append(f"{int(issue)},{numbers}\n")
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def _target_prior_history(
    issues: IntArray, draws: IntArray, target_issue: int
) -> tuple[IntArray, IntArray]:
    issue_values, draw_values = validate_issue_draws(issues, draws)
    if target_issue <= 0:
        raise ValueError("目标期号必须为正整数")
    mask = issue_values < target_issue
    historical_issues = issue_values[mask]
    historical_draws = draw_values[mask]
    required = MINIMUM_INITIAL_HISTORY + MINIMUM_INNER_OBSERVATIONS
    if len(historical_issues) < required:
        raise ValueError(f"目标期前至少需要{required}期历史")
    if len(historical_issues) == 0 or int(historical_issues[-1]) >= target_issue:
        raise ValueError("历史截止期必须早于目标期")
    return historical_issues, historical_draws


def predict_frozen_models(
    issues: IntArray, draws: IntArray, *, target_issue: int
) -> ProspectivePrediction:
    """严格复用已合并 v2 实现，形成五个冻结模型的目标期前预测。

    为调用现有 walk-forward 公共接口，函数在目标位置放入一个确定性哨兵票面；
    既有实现先预测、后吸收当前结果，因此哨兵内容不会影响当前预测。测试同时
    验证目标期和未来结果改变时输出不变。
    """

    historical_issues, historical_draws = _target_prior_history(
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
    uniform = np.full(NUMBER_COUNT, FAIR_PROBABILITY, dtype=np.float64)
    probabilities: dict[str, FloatArray] = {
        UNIFORM_STRATEGY: uniform,
        DYNAMIC_STRATEGY: result.phase1_result.posterior_mean[offset].copy(),
        FIXED_NORMAL_STRATEGY: result.fixed_normal_posterior_mean[offset].copy(),
        FIXED_HIGH_STRATEGY: result.fixed_high_posterior_mean[offset].copy(),
        CHANGEPOINT_STRATEGY: result.posterior_mean[offset].copy(),
    }
    rankings: dict[str, IntArray] = {
        UNIFORM_STRATEGY: np.arange(1, NUMBER_COUNT + 1, dtype=np.int64),
        DYNAMIC_STRATEGY: result.rankings[DYNAMIC_STRATEGY][offset].copy(),
        FIXED_NORMAL_STRATEGY: result.rankings[FIXED_NORMAL_STRATEGY][offset].copy(),
        FIXED_HIGH_STRATEGY: result.rankings[FIXED_HIGH_STRATEGY][offset].copy(),
        CHANGEPOINT_STRATEGY: result.rankings[CHANGEPOINT_STRATEGY][offset].copy(),
    }
    for strategy in PROBABILITY_STRATEGIES:
        model_probabilities = probabilities[strategy]
        if model_probabilities.shape != (NUMBER_COUNT,) or not np.all(
            (model_probabilities > 0.0) & (model_probabilities < 1.0)
        ):
            raise FloatingPointError(f"{strategy}概率不严格位于(0,1)")
        if not np.isclose(model_probabilities.sum(), DRAW_SIZE, rtol=0.0, atol=1e-9):
            raise FloatingPointError(f"{strategy}的80个概率之和不等于20")
        if set(map(int, rankings[strategy])) != set(range(1, NUMBER_COUNT + 1)):
            raise FloatingPointError(f"{strategy}排名不是1至80完整排列")
    return ProspectivePrediction(
        probabilities=probabilities,
        rankings=rankings,
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


def build_manifest(
    *,
    project_root: Path,
    data_path: Path,
    config_path: Path,
    issues: IntArray,
    draws: IntArray,
    target_issue: int,
    generated_at_utc: str,
    git_commit_sha: str,
    official_source_url: str,
    official_confirmed_at_utc: str,
) -> dict[str, object]:
    """构造一个确定性、尚未落盘的目标期前 manifest。"""

    config = load_and_verify_freeze_config(project_root, config_path)
    resolved_data = resolve_within(project_root, data_path, "输入数据")
    generated_text, generated_time = _canonical_utc(
        generated_at_utc, "generated_at_utc"
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
    historical_issues, historical_draws = _target_prior_history(
        issues, draws, target_issue
    )
    prediction = predict_frozen_models(issues, draws, target_issue=target_issue)

    model_payload: dict[str, object] = {}
    for strategy in PROBABILITY_STRATEGIES:
        probabilities = [float(value) for value in prediction.probabilities[strategy]]
        ranking = [int(value) for value in prediction.rankings[strategy]]
        candidates = candidate_sets(prediction.probabilities[strategy])
        model_payload[strategy] = {
            "number_index": list(range(1, NUMBER_COUNT + 1)),
            "probabilities": probabilities,
            "ranking": ranking,
            "top_k_candidates": {str(k): list(candidates[k]) for k in range(1, 11)},
        }

    source_manifest = _require_mapping(config["source_manifest"], "source_manifest")
    frozen_parameters = _require_mapping(config["models"], "models")
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "target_issue": target_issue,
        "generated_at_utc": generated_text,
        "history_through_issue": int(historical_issues[-1]),
        "history_issue_count": len(historical_issues),
        "input_data": {
            "path": _relative_posix(project_root, resolved_data),
            "canonical_target_prior_sha256": canonical_history_sha256(
                historical_issues, historical_draws
            ),
            "canonicalization": "issue_ascending_numbers_ascending_utf8_lf",
        },
        "git_commit_sha": git_commit_sha,
        "official_issue_confirmation": {
            "source_url": official_source_url,
            "confirmed_at_utc": confirmed_text,
        },
        "source_manifest": source_manifest,
        "frozen_parameters": frozen_parameters,
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
        "confirmation_protocol": {
            "fixed_future_issue_count": CONFIRMATION_ISSUE_COUNT,
            "primary_metric": "per_issue_80_dimensional_brier_score",
            "formal_summary_only_after_issue_count": CONFIRMATION_ISSUE_COUNT,
            "multiplicity_correction": "holm_four_primary_comparisons",
            "early_stopping": False,
        },
    }


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
) -> tuple[dict[str, FloatArray], dict[str, IntArray]]:
    models = _require_mapping(manifest.get("models"), "manifest.models")
    if tuple(sorted(models)) != tuple(sorted(PROBABILITY_STRATEGIES)):
        raise ValueError("manifest未完整包含五个冻结模型")
    probabilities: dict[str, FloatArray] = {}
    rankings: dict[str, IntArray] = {}
    for strategy in PROBABILITY_STRATEGIES:
        model = _require_mapping(models[strategy], f"manifest.models.{strategy}")
        probability_array = cast(
            FloatArray, np.asarray(model.get("probabilities"), dtype=np.float64)
        )
        ranking_array = cast(IntArray, np.asarray(model.get("ranking"), dtype=np.int64))
        if probability_array.shape != (NUMBER_COUNT,) or not np.all(
            (probability_array > 0.0) & (probability_array < 1.0)
        ):
            raise ValueError(f"manifest中的{strategy}概率非法")
        if not np.isclose(probability_array.sum(), DRAW_SIZE, atol=1e-9, rtol=0.0):
            raise ValueError(f"manifest中的{strategy}概率和不等于20")
        if set(map(int, ranking_array)) != set(range(1, NUMBER_COUNT + 1)):
            raise ValueError(f"manifest中的{strategy}排名非法")
        probabilities[strategy] = probability_array
        rankings[strategy] = ranking_array
    return probabilities, rankings


def build_evaluation_record(
    *,
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    actual_numbers: IntArray,
    official_result_published_at_utc: str,
    evaluated_at_utc: str,
) -> dict[str, object]:
    """只用已封存概率和实际开奖号码构造开奖后评价记录。"""

    config = load_and_verify_freeze_config(project_root, config_path)
    resolved_manifest = resolve_within(project_root, manifest_path, "manifest")
    manifest = _load_json_object(resolved_manifest)
    if manifest.get("evidence_status") != EVIDENCE_STATUS:
        raise ValueError("manifest证据状态不符合前瞻协议")
    if manifest.get("source_manifest") != config.get("source_manifest"):
        raise ValueError("manifest source manifest与当前冻结配置不一致")
    target_issue = manifest.get("target_issue")
    if not isinstance(target_issue, int) or target_issue <= 0:
        raise ValueError("manifest目标期号非法")
    generated_text = manifest.get("generated_at_utc")
    if not isinstance(generated_text, str):
        raise ValueError("manifest缺少generated_at_utc")
    _, generated_time = _canonical_utc(generated_text, "generated_at_utc")
    published_text, published_time = _canonical_utc(
        official_result_published_at_utc, "official_result_published_at_utc"
    )
    evaluated_text, evaluated_time = _canonical_utc(
        evaluated_at_utc, "evaluated_at_utc"
    )
    if evaluated_time < published_time:
        raise ValueError("评价时间不得早于官方结果发布时间")
    if generated_time >= published_time:
        raise ValueError("manifest未在官方结果发布前封存")
    numbers = cast(IntArray, np.asarray(actual_numbers, dtype=np.int64))
    if numbers.shape != (DRAW_SIZE,):
        raise ValueError("实际开奖号码必须恰好20个")
    validate_issue_draws(
        np.asarray([target_issue], dtype=np.int64), numbers.reshape(1, DRAW_SIZE)
    )
    outcomes = np.zeros(NUMBER_COUNT, dtype=np.float64)
    outcomes[numbers - 1] = 1.0
    probabilities, rankings = _manifest_model_arrays(manifest)

    model_metrics: dict[str, object] = {}
    for strategy in PROBABILITY_STRATEGIES:
        topk: dict[str, object] = {}
        for k in range(1, 11):
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

    uniform_metrics = _require_mapping(
        model_metrics[UNIFORM_STRATEGY], UNIFORM_STRATEGY
    )
    uniform_brier = float(uniform_metrics["brier_score"])
    uniform_log_loss = float(uniform_metrics["bernoulli_log_loss"])
    versus_uniform: dict[str, object] = {}
    for strategy in PRIMARY_COMPARISON_MODELS:
        metrics = _require_mapping(model_metrics[strategy], strategy)
        versus_uniform[strategy] = {
            "brier_difference_model_minus_uniform": (
                float(metrics["brier_score"]) - uniform_brier
            ),
            "log_loss_difference_model_minus_uniform": (
                float(metrics["bernoulli_log_loss"]) - uniform_log_loss
            ),
        }
    changepoint_metrics = _require_mapping(
        model_metrics[CHANGEPOINT_STRATEGY], CHANGEPOINT_STRATEGY
    )
    versus_fixed: dict[str, object] = {}
    for strategy in (FIXED_NORMAL_STRATEGY, FIXED_HIGH_STRATEGY):
        metrics = _require_mapping(model_metrics[strategy], strategy)
        versus_fixed[strategy] = {
            "brier_difference_changepoint_minus_fixed": (
                float(changepoint_metrics["brier_score"])
                - float(metrics["brier_score"])
            ),
            "log_loss_difference_changepoint_minus_fixed": (
                float(changepoint_metrics["bernoulli_log_loss"])
                - float(metrics["bernoulli_log_loss"])
            ),
        }
    selected = _require_mapping(
        manifest.get("selected_target_prior_parameters"),
        "selected_target_prior_parameters",
    )
    changepoint_selected = _require_mapping(
        selected.get(CHANGEPOINT_STRATEGY), CHANGEPOINT_STRATEGY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "target_issue": target_issue,
        "evaluated_at_utc": evaluated_text,
        "official_result_published_at_utc": published_text,
        "actual_numbers": sorted(map(int, numbers)),
        "manifest": {
            "path": _relative_posix(project_root, resolved_manifest),
            "sha256": raw_sha256(resolved_manifest),
            "generated_at_utc": generated_text,
            "sealed_before_official_result": True,
        },
        "changepoint_state": changepoint_selected.get("state"),
        "models": model_metrics,
        "comparisons": {
            "versus_uniform": versus_uniform,
            "changepoint_versus_fixed_baselines": versus_fixed,
        },
    }


def write_evaluation_exclusive(record: Mapping[str, object], results_dir: Path) -> Path:
    """每期使用独立不可覆盖 JSON；同一期第二次写入必定失败。"""

    target_issue = record.get("target_issue")
    if not isinstance(target_issue, int) or target_issue <= 0:
        raise ValueError("评价记录缺少合法target_issue")
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


def build_formal_primary_summary(results_dir: Path) -> dict[str, object]:
    """仅在恰好365条预封存记录时形成四项主要比较的正式汇总。"""

    records = [_load_json_object(path) for path in sorted(results_dir.glob("*.json"))]
    issue_values = [record.get("target_issue") for record in records]
    if len(records) != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("正式汇总只能在恰好365个未来开奖期完成后运行")
    if any(not isinstance(issue, int) or issue <= 0 for issue in issue_values):
        raise ValueError("正式汇总含非法目标期号")
    if len(set(issue_values)) != CONFIRMATION_ISSUE_COUNT:
        raise ValueError("正式汇总含重复目标期")
    for record in records:
        if record.get("evidence_status") != EVIDENCE_STATUS:
            raise ValueError("正式汇总只能包含v2预封存研究记录")
        manifest = _require_mapping(record.get("manifest"), "record.manifest")
        if manifest.get("sealed_before_official_result") is not True:
            raise ValueError("正式汇总只能包含开奖前封存记录")
        manifest_digest = manifest.get("sha256")
        if not isinstance(manifest_digest, str) or not SHA256_PATTERN.fullmatch(
            manifest_digest
        ):
            raise ValueError("正式汇总记录缺少合法manifest SHA-256")

    deltas_by_model: dict[str, FloatArray] = {}
    raw_p_values: dict[str, float] = {}
    summaries: dict[str, object] = {}
    for strategy in PRIMARY_COMPARISON_MODELS:
        values: list[float] = []
        for record in records:
            comparisons = _require_mapping(record.get("comparisons"), "comparisons")
            versus_uniform = _require_mapping(
                comparisons.get("versus_uniform"), "versus_uniform"
            )
            model = _require_mapping(versus_uniform.get(strategy), strategy)
            values.append(float(model["brier_difference_model_minus_uniform"]))
        deltas = cast(FloatArray, np.asarray(values, dtype=np.float64))
        deltas_by_model[strategy] = deltas
        test = ttest_1samp(deltas, popmean=0.0, alternative="less")
        raw_p = float(test.pvalue)
        raw_p_values[strategy] = raw_p
        summaries[strategy] = {
            "issue_count": len(deltas),
            "mean_brier_difference_model_minus_uniform": float(deltas.mean()),
            "ordinary_standard_error": float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
            "one_sided_p_value_unadjusted": raw_p,
        }
    adjusted = holm_adjust(raw_p_values)
    for strategy in PRIMARY_COMPARISON_MODELS:
        model_summary = _require_mapping(summaries[strategy], strategy)
        model_summary["holm_adjusted_p_value"] = adjusted[strategy]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "completed_prospective_confirmation_summary",
        "issue_count": CONFIRMATION_ISSUE_COUNT,
        "primary_metric": "per_issue_80_dimensional_brier_score",
        "issue_is_statistical_unit": True,
        "test": {
            "name": "one_sided_paired_issue_level_t_test",
            "alternative": "mean_model_minus_uniform_brier_less_than_zero",
            "multiplicity_correction": "holm",
            "comparison_count": len(PRIMARY_COMPARISON_MODELS),
        },
        "comparisons": summaries,
    }


__all__ = [
    "CONFIRMATION_ISSUE_COUNT",
    "DATA_RELATIVE_PATH",
    "EVIDENCE_STATUS",
    "FREEZE_RELATIVE_PATH",
    "MANIFEST_RELATIVE_DIR",
    "PRIMARY_COMPARISON_MODELS",
    "PROBABILITY_STRATEGIES",
    "RESULT_RELATIVE_DIR",
    "SCHEMA_VERSION",
    "UNIFORM_STRATEGY",
    "ProspectivePrediction",
    "build_evaluation_record",
    "build_formal_primary_summary",
    "build_manifest",
    "canonical_history_sha256",
    "canonical_json_bytes",
    "holm_adjust",
    "load_and_verify_freeze_config",
    "load_history_csv",
    "normalized_lf_sha256",
    "predict_frozen_models",
    "raw_sha256",
    "require_contract_path",
    "resolve_within",
    "source_manifest_fingerprint",
    "utc_now_string",
    "write_evaluation_exclusive",
    "write_manifest_exclusive",
]
