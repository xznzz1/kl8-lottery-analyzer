# -*- coding: utf-8 -*-
"""
项目配置（精简版）。

本版本仅保留快乐8分析所需的最小配置集合，移除了与建模训练相关的冗余定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"
_ALLOWED_OUTPUT_ROOT_NAMES = ("data_cache", "results", "reports")


def _load_yaml_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open(encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}
    # 提供兜底默认值，避免在最小环境下启动失败
    return {}


YAML_CONFIG: Dict[str, Any] = _load_yaml_config()


def _resolve_output_path(value: object, default: Path) -> Path:
    """解析项目输出路径，并拒绝落到仓库允许目录之外。"""

    candidate = Path(str(value)) if value not in (None, "") else default
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    resolved = candidate.resolve()
    allowed_roots = [(BASE_DIR / name).resolve() for name in _ALLOWED_OUTPUT_ROOT_NAMES]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"输出路径必须位于data_cache、results或reports内：{resolved}")
    return resolved


def resolve_scoped_output_path(value: object, default: Path, root_name: str) -> Path:
    """解析并限制输出到指定的项目目录。

    ``root_name`` 仅接受 ``data_cache``、``results`` 或 ``reports``。相对路径
    始终以仓库根目录为基准，防止从其他工作目录启动脚本时写到 C 盘或仓库外。
    """

    if root_name not in _ALLOWED_OUTPUT_ROOT_NAMES:
        raise ValueError(f"未知的输出目录边界：{root_name}")
    resolved = _resolve_output_path(value, default)
    root = (BASE_DIR / root_name).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"输出路径必须位于{root_name}内：{resolved}")
    return resolved


_RAW_PATH_SECTION = YAML_CONFIG.get("paths", {})
_PATH_SECTION: Dict[str, Any] = (
    _RAW_PATH_SECTION if isinstance(_RAW_PATH_SECTION, dict) else {}
)
PATHS = {
    "data": _resolve_output_path(_PATH_SECTION.get("data"), BASE_DIR / "data_cache"),
    "results": _resolve_output_path(_PATH_SECTION.get("results"), BASE_DIR / "results"),
    "reports": _resolve_output_path(_PATH_SECTION.get("reports"), BASE_DIR / "reports"),
    "logs": _resolve_output_path(
        _PATH_SECTION.get("logs"), BASE_DIR / "results" / "logs"
    ),
    "data_cache": _resolve_output_path(
        _PATH_SECTION.get("data_cache"), BASE_DIR / "data_cache"
    ),
}

_RAW_NETWORK_SECTION = YAML_CONFIG.get("network", {})
_NETWORK_SECTION: Dict[str, Any] = (
    _RAW_NETWORK_SECTION if isinstance(_RAW_NETWORK_SECTION, dict) else {}
)
NETWORK_CONFIG = {
    "timeout": _NETWORK_SECTION.get("timeout", 20),
    "retry_count": _NETWORK_SECTION.get("retry_count", 3),
    "backoff_factor": _NETWORK_SECTION.get("backoff_factor", 0.6),
    "user_agent": _NETWORK_SECTION.get(
        "user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    ),
}

ALLOWED_DOMAINS = {"datachart.500.com", "data.917500.cn"}

DATA_FILE_NAME = "data.csv"
MODEL_METADATA_FILE = "metadata.json"

_RAW_ANALYSIS_SECTION = YAML_CONFIG.get("analysis", {})
_ANALYSIS_SECTION: Dict[str, Any] = (
    _RAW_ANALYSIS_SECTION if isinstance(_RAW_ANALYSIS_SECTION, dict) else {}
)


def _nested_section(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key, {})
    return value if isinstance(value, dict) else {}


_DIRICHLET_SECTION = _nested_section(_ANALYSIS_SECTION, "dirichlet")
_RULE_SECTION = _nested_section(_ANALYSIS_SECTION, "rules")
_COPULA_SECTION = _nested_section(_ANALYSIS_SECTION, "copula")
_GRAPH_SECTION = _nested_section(_ANALYSIS_SECTION, "graph_embedding")

DIRICHLET_CONFIG = {
    "prior_strength": float(_DIRICHLET_SECTION.get("prior_strength", 0.5)),
    "window_size": int(_DIRICHLET_SECTION.get("window_size", 120)),
    "variance_weight": float(_DIRICHLET_SECTION.get("variance_weight", 0.3)),
}

RULE_MINER_CONFIG = {
    "min_support": float(_RULE_SECTION.get("min_support", 0.08)),
    "min_confidence": float(_RULE_SECTION.get("min_confidence", 0.6)),
    "max_itemset_size": int(_RULE_SECTION.get("max_itemset_size", 3)),
    "cache_ttl_seconds": int(_RULE_SECTION.get("cache_ttl_seconds", 24 * 60 * 60)),
    "soft_penalty_weight": float(_RULE_SECTION.get("soft_penalty_weight", 0.4)),
}

_COPULA_RANDOM_SEED = _COPULA_SECTION.get("random_seed")
COPULA_CONFIG = {
    "enabled": bool(_COPULA_SECTION.get("enabled", True)),
    "min_draws": int(_COPULA_SECTION.get("min_draws", 200)),
    "samples": int(_COPULA_SECTION.get("samples", 48)),
    "shrinkage": float(_COPULA_SECTION.get("shrinkage", 0.12)),
    "topk_multiplier": float(_COPULA_SECTION.get("topk_multiplier", 1.3)),
    "random_seed": (
        None
        if _COPULA_RANDOM_SEED in (None, "", "null")
        else int(str(_COPULA_RANDOM_SEED))
    ),
}

GRAPH_EMBED_CONFIG = {
    "enabled": bool(_GRAPH_SECTION.get("enabled", True)),
    "embedding_dim": int(_GRAPH_SECTION.get("embedding_dim", 32)),
    "cache_file": str(
        resolve_scoped_output_path(
            _GRAPH_SECTION.get("cache_file"),
            PATHS["data_cache"] / "graph_embeddings.npz",
            "data_cache",
        )
    ),
    "weight": float(_GRAPH_SECTION.get("weight", 0.16)),
    "metric": str(_GRAPH_SECTION.get("metric", "norm")),
    "decay": float(_GRAPH_SECTION.get("decay", 0.92)),
}


@dataclass(frozen=True)
class SequenceModelSpec:
    """描述快乐8单个序列模型的结构参数。"""

    sequence_len: int
    num_classes: int
    embedding_dim: int
    hidden_units: Iterable[int]
    dropout: float = 0.2


@dataclass(frozen=True)
class LotteryModelConfig:
    """描述快乐8分析脚本需要的基本参数。"""

    code: str
    name: str
    red: SequenceModelSpec
    default_window: int = 6
    default_batch_size: int = 48
    default_red_epochs: int = 40
    learning_rate: float = 5e-4
    allow_sequence_order: bool = True


LOTTERY_CONFIGS: Dict[str, LotteryModelConfig] = {
    "kl8": LotteryModelConfig(
        code="kl8",
        name="快乐8",
        red=SequenceModelSpec(
            sequence_len=20,
            num_classes=80,
            embedding_dim=48,
            hidden_units=(128, 128, 64),
            dropout=0.35,
        ),
    )
}


def ensure_runtime_directories() -> None:
    """确保数据、结果、日志目录存在。"""

    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


def get_lottery_config(code: str) -> LotteryModelConfig:
    """根据代号获取配置。"""

    normalized = code.lower().strip()
    if normalized not in LOTTERY_CONFIGS:
        raise ValueError(f"未知的彩票类型：{code}")
    return LOTTERY_CONFIGS[normalized]


name_path = {
    code: {
        "name": cfg.name,
        "path": f"{(PATHS['data'] / code).as_posix()}/",
    }
    for code, cfg in LOTTERY_CONFIGS.items()
}
predict_path = f"{(PATHS['results']).as_posix()}/"
data_file_name = DATA_FILE_NAME


__all__ = [
    "ALLOWED_DOMAINS",
    "BASE_DIR",
    "CONFIG_FILE",
    "DATA_FILE_NAME",
    "LotteryModelConfig",
    "LOTTERY_CONFIGS",
    "MODEL_METADATA_FILE",
    "DIRICHLET_CONFIG",
    "RULE_MINER_CONFIG",
    "COPULA_CONFIG",
    "GRAPH_EMBED_CONFIG",
    "NETWORK_CONFIG",
    "PATHS",
    "SequenceModelSpec",
    "data_file_name",
    "ensure_runtime_directories",
    "get_lottery_config",
    "name_path",
    "predict_path",
    "resolve_scoped_output_path",
]

ball_name = [("红球", "red"), ("蓝球", "blue")]

data_file_name = "data.csv"
data_cq_file_name = "data_cq.csv"
predict_path = f"{(PATHS['results'] / 'predict').as_posix()}/"
result_path = f"{PATHS['results'].as_posix()}/"

# Network configuration for crawlers
HTTP_ALLOWLIST = [
    "datachart.500.com",
    "917500.cn",
]

# Local cache directory for crawler fallback
HTTP_CACHE_DIR = str(PATHS["data_cache"] / "http")

# HTTP retry/backoff defaults (used by src.common.get_http_session_with_backoff)
HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 0.5
HTTP_REQUEST_DELAY = 0.0  # seconds between retries (additional to exponential backoff)

name_path = {
    "ssq": {"name": "双色球", "path": f"{(PATHS['data'] / 'ssq').as_posix()}/"},
    "dlt": {"name": "大乐透", "path": f"{(PATHS['data'] / 'dlt').as_posix()}/"},
    "qxc": {"name": "七星彩", "path": f"{(PATHS['data'] / 'qxc').as_posix()}/"},
    "pls": {"name": "排列三", "path": f"{(PATHS['data'] / 'pls').as_posix()}/"},
    "kl8": {"name": "快乐8", "path": f"{(PATHS['data'] / 'kl8').as_posix()}/"},
}

model_path = f"{(PATHS['data_cache'] / 'models').as_posix()}/"

model_args = {
    "kl8": {
        "model_args": {
            "seq_len": 3,
            "batch_size": 1,
            "red_sequence_len": 20,
            "sequence_len": 20,
            "red_n_class": 80,
            "red_epochs": 1,
            "red_embedding_size": 20,
            "red_hidden_size": 20,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 0,
            "blue_epochs": 0,
            "blue_embedding_size": 0,
            "blue_hidden_size": 0,
            "blue_layer_size": 0,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08,
        },
        "path": {
            "red": model_path + "/kl8/red_ball_model/",
            "blue": model_path + "/kl8/blue_ball_model/",
        },
        "pathname": {"name": "/kl8/"},
        "subpath": {"red": "/red_ball_model/", "blue": "/blue_ball_model/"},
    },
    "pls": {
        "model_args": {
            "seq_len": 3,
            "batch_size": 1,
            "red_sequence_len": 3,
            "sequence_len": 3,
            "red_n_class": 10,
            "red_epochs": 1,
            "red_embedding_size": 10,
            "red_hidden_size": 10,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 0,
            "blue_epochs": 0,
            "blue_embedding_size": 0,
            "blue_hidden_size": 0,
            "blue_layer_size": 0,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08,
        },
        "path": {
            "red": model_path + "/pls/red_ball_model/",
            "blue": model_path + "/pls/blue_ball_model/",
        },
        "pathname": {"name": "/pls/"},
        "subpath": {"red": "/red_ball_model/", "blue": "/blue_ball_model/"},
    },
    "ssq": {
        "model_args": {
            "seq_len": 3,
            "batch_size": 1,
            "red_sequence_len": 6,
            "sequence_len": 6,
            "red_n_class": 33,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 16,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08,
        },
        "path": {
            "red": model_path + "/ssq/red_ball_model/",
            "blue": model_path + "/ssq/blue_ball_model/",
        },
        "pathname": {"name": "/ssq/"},
        "subpath": {"red": "/red_ball_model/", "blue": "/blue_ball_model/"},
    },
    "dlt": {
        "model_args": {
            "seq_len": 3,
            "batch_size": 1,
            "red_sequence_len": 5,
            "red_n_class": 35,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 2,
            "blue_n_class": 12,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08,
        },
        "path": {
            "red": model_path + "/dlt/red_ball_model/",
            "blue": model_path + "/dlt/blue_ball_model/",
        },
        "pathname": {"name": "/dlt/"},
        "subpath": {"red": "/red_ball_model/", "blue": "/blue_ball_model/"},
    },
}

# 模型名
pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"
