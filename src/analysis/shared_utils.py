# -*- coding:utf-8 -*-
"""
通用分析工具函数（供 kl8_analysis 与 kl8_analysis_plus 复用）
- 目录/路径工具
- 数学与序列小工具
"""

from __future__ import annotations

import os
import random
import re
import threading
from multiprocessing import Process
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
_SAFE_LABEL = re.compile(r"^[\w.-]+$", flags=re.UNICODE)


def _resolve_results_dir(file_dir: str | os.PathLike[str]) -> Path:
    """解析结果目录，并拒绝仓库 ``results`` 之外的写入。"""

    candidate = Path(file_dir)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    root = RESULTS_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"结果目录必须位于results内：{resolved}")
    return resolved


def _resolve_results_file(file_path: str | os.PathLike[str]) -> Path:
    """最终写入前再次确认文件仍位于 ``results`` 内。"""

    resolved = Path(file_path).resolve()
    root = RESULTS_ROOT.resolve()
    if root not in resolved.parents:
        raise ValueError(f"结果文件必须位于results内：{resolved}")
    return resolved


def compute_output_dir(random_mode: int, path_label: str) -> str:
    """根据 random_mode 与路径标签生成输出目录路径。
    random_mode: 0 -> results/legacy，1 -> results/random
    path_label: 作为输出根目录下的单层子目录；拒绝路径穿越。
    """
    base = RESULTS_ROOT / ("legacy" if int(random_mode) == 0 else "random")
    label = path_label.strip()
    if label:
        if label in {".", ".."} or _SAFE_LABEL.fullmatch(label) is None:
            raise ValueError("输出标签只能包含字母、数字、下划线、点或连字符")
        base = base / label
    return f"{base.resolve().as_posix()}/"


def ensure_dir(path: str) -> None:
    """确保目录存在（等幂）。"""
    os.makedirs(path, exist_ok=True)


def check_odd_even(lst: List[int]) -> Tuple[int, int]:
    """统计列表中的奇偶个数。"""
    odd = 0
    even = 0
    for item in lst:
        if item % 2 == 0:
            even += 1
        else:
            odd += 1
    return odd, even


def find_consecutive_number(numbers: List[int]) -> List[Tuple[int, ...]]:
    """找出连续号码的组合段。
    输入需为升序列表。
    """
    if not numbers:
        return []
    consecutive_group: List[Tuple[int, ...]] = []
    group: List[int] = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] == 1:
            group.append(numbers[i])
        else:
            if len(group) > 1:
                consecutive_group.append(tuple(group))
            group = [numbers[i]]
    if len(group) > 1:
        consecutive_group.append(tuple(group))
    return consecutive_group


def _build_unique_filename(
    file_dir: str, prefix: str, current_time_str: str, cal_nums: int, period_num: str
) -> str:
    """生成不重复的结果文件名，格式：prefix_time_calnums_period.csv"""
    if prefix in {".", ".."} or _SAFE_LABEL.fullmatch(prefix) is None:
        raise ValueError("结果文件前缀只能包含字母、数字、下划线、点或连字符")
    safe_dir = _resolve_results_dir(file_dir)
    file_name = _resolve_results_file(
        safe_dir / f"{prefix}_{current_time_str}_{cal_nums}_{period_num}.csv"
    )
    # 如果存在则添加随机扰动重试
    while os.path.exists(file_name):
        rnd = random.randint(0, 999999)
        file_name = _resolve_results_file(
            safe_dir
            / f"{prefix}_{int(current_time_str) + rnd}_{cal_nums}_{period_num}.csv"
        )
    return str(file_name)


def write_results_core(
    rows: List[List[int]],
    file_dir: str,
    file_prefix: str,
    cal_nums: int,
    total_create: int,
    multiple: int,
    multiple_ratio: str,
    period_num: str,
    current_time_str: str,
) -> None:
    """同步写入结果文件（供线程/进程后端调用）。"""
    safe_file_dir = _resolve_results_dir(file_dir)
    file_dir = f"{safe_file_dir.as_posix()}/"
    ensure_dir(file_dir)
    file_name = _build_unique_filename(
        file_dir, file_prefix, current_time_str, cal_nums, str(period_num)
    )
    with open(file_name, "w") as f:
        # 写表头
        for i in range(cal_nums - 1):
            f.write(f"b{i+1},")
        f.write(f"b{cal_nums}\n")

        # 写内容
        cnt = 0
        item_index = 0
        if multiple > 1:
            div_nums = multiple_ratio.split(",")
            div_a = int(div_nums[0])
            div_b = int(div_nums[1])
        for item in rows:
            if multiple > 1:
                item_index += 1
                if item_index % div_a != div_b:
                    continue
                cnt += 1
            for idx in range(len(item) - 1):
                f.write(f"{item[idx]},")
            f.write(f"{item[-1]}\n")
            if multiple > 1 and cnt >= total_create:
                break


def write_results_async(
    rows: List[List[int]],
    file_dir: str,
    file_prefix: str,
    cal_nums: int,
    total_create: int,
    multiple: int,
    multiple_ratio: str,
    period_num: str,
    current_time_str: str,
    backend: str = "thread",
) -> None:
    """
    异步写入结果文件。

    注意：为兼容 Windows 上的 multiprocessing（spawn 启动方式），目标函数必须
    是顶层可 picklable 的可调用，不能使用 lambda/闭包。

    backend 支持：
    - 'thread'：使用 threading.Thread，适用于 I/O 为主（默认）。
    - 'process'：使用 multiprocessing.Process，避免 GIL 影响。
    """
    safe_file_dir = f"{_resolve_results_dir(file_dir).as_posix()}/"
    target_args = (
        rows,
        safe_file_dir,
        file_prefix,
        cal_nums,
        total_create,
        multiple,
        multiple_ratio,
        period_num,
        current_time_str,
    )
    if backend == "process":
        p = Process(target=write_results_core, args=target_args)
        p.start()
    else:
        t = threading.Thread(target=write_results_core, args=target_args)
        t.start()
