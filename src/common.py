# -*- coding: utf-8 -*-
"""快乐8公共接口。

当前主流程保持轻量，不在模块导入阶段强制加载PyTorch。
旧版深度学习功能保存在 common_legacy.py 中，并按需延迟加载。
"""

from __future__ import annotations

from importlib import import_module
from typing import Optional

from .data_fetcher import download_history, get_current_issue, load_history

SUPPORTED_LOTTERIES = {"kl8"}


def _ensure_supported_lottery(code: str) -> str:
    normalized = code.lower().strip()
    if normalized not in SUPPORTED_LOTTERIES:
        raise ValueError(f"当前仅支持快乐8玩法，收到：{code}")
    return normalized


def get_data_run(
    name: str,
    cq: int = 0,
    *,
    sequence_mode: Optional[bool] = None,
    start_issue: int | str | None = None,
    end_issue: int | str | None = None,
):
    """下载快乐8历史数据。

    sequence_mode优先于旧参数cq：
    - False：普通排序号码
    - True：出球顺序号码
    """

    code = _ensure_supported_lottery(name)

    use_sequence_order = bool(cq) if sequence_mode is None else bool(sequence_mode)

    def coerce_issue(value: int | str | None) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text.isdigit():
            raise ValueError(f"期号必须为正整数，收到：{value!r}")
        return int(text)

    return download_history(
        code,
        start=coerce_issue(start_issue),
        end=coerce_issue(end_issue),
        use_sequence_order=use_sequence_order,
    )


def get_current_number(name: str) -> str:
    """返回快乐8最新期号。"""

    code = _ensure_supported_lottery(name)
    return get_current_issue(code)


def __getattr__(name: str):
    """按需加载旧版深度学习接口。

    只有真正调用旧训练功能时才要求安装PyTorch。
    """

    # importlib、unittest.mock等工具会查询__path__等模块元数据。
    # 这些特殊属性不能触发旧版深度学习模块加载。
    if name.startswith("__"):
        raise AttributeError(name)

    try:
        legacy = import_module("src.common_legacy")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                f"{name}属于旧版深度学习功能，需要额外安装PyTorch；"
                "数据下载和统计回测不需要PyTorch。"
            ) from exc
        raise

    try:
        return getattr(legacy, name)
    except AttributeError as exc:
        raise AttributeError(f"模块src.common没有属性{name!r}") from exc


__all__ = [
    "SUPPORTED_LOTTERIES",
    "download_history",
    "get_current_issue",
    "get_current_number",
    "get_data_run",
    "load_history",
]
