# -*- coding: utf-8 -*-
"""
KL8专用通用函数模块
提取原common.py中KL8相关的功能并进行优化
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

from .config import get_kl8_config, ensure_runtime_directories
from .data_fetcher import download_kl8_history, get_kl8_current_issue


def get_data_run(
    code: str = "kl8",
    cq: int = 0,
    start_issue: Optional[int] = None,
    end_issue: Optional[int] = None,
) -> None:
    """下载KL8历史数据。
    
    Args:
        code: 彩票类型代码（兼容参数，当前只支持kl8）
        cq: 是否使用序列顺序 (KL8专用)
        start_issue: 起始期号
        end_issue: 结束期号
    """
    
    # 验证彩票类型（当前只支持KL8）
    if code.lower() != "kl8":
        raise ValueError(f"不支持的彩票类型：{code}，当前只支持 kl8")
    
    ensure_runtime_directories()
    config = get_kl8_config()
    
    logger.info("开始下载{}历史数据...", config.name)
    
    # KL8支持序列顺序下载
    use_sequence = bool(cq)
    download_kl8_history(
        start=start_issue, 
        end=end_issue, 
        use_sequence_order=use_sequence
    )
    
    logger.success("{}历史数据下载完成", config.name)


def get_current_number() -> str:
    """返回KL8的当前期号。
    
    Returns:
        当前期号字符串
    """
    
    config = get_kl8_config()
    logger.info("获取{}当前期号...", config.name)
    
    current_issue = get_kl8_current_issue()
    logger.info("{}当前期号: {}", config.name, current_issue)
    
    return current_issue


def format_prediction_results(predictions: Dict[str, list]) -> Dict[str, list]:
    """格式化预测结果。
    
    Args:
        predictions: 原始预测结果
        
    Returns:
        格式化后的预测结果
    """
    
    config = get_kl8_config()
    readable = {}
    
    for key, value in predictions.items():
        if hasattr(value, 'tolist'):
            # numpy数组转换
            readable[key] = list(map(int, value.tolist()))
        else:
            # 普通列表
            readable[key] = list(map(int, value))
    
    logger.info("【{}】预测结果: {}", config.name, readable)
    return readable


def validate_kl8_numbers(numbers: list) -> bool:
    """验证KL8号码是否有效。
    
    Args:
        numbers: 号码列表
        
    Returns:
        是否有效
    """
    
    if not isinstance(numbers, list):
        return False
    
    if len(numbers) != 20:
        return False
    
    # 检查号码范围 (1-80)
    for num in numbers:
        if not isinstance(num, int) or num < 1 or num > 80:
            return False
    
    # 检查是否有重复号码
    if len(set(numbers)) != len(numbers):
        return False
    
    return True


def calculate_kl8_statistics(numbers: list) -> Dict[str, float]:
    """计算KL8号码统计信息。
    
    Args:
        numbers: 号码列表
        
    Returns:
        统计信息字典
    """
    
    if not validate_kl8_numbers(numbers):
        raise ValueError("无效的KL8号码")
    
    stats = {}
    
    # 基础统计
    stats['mean'] = sum(numbers) / len(numbers)
    stats['max'] = max(numbers)
    stats['min'] = min(numbers)
    stats['range'] = stats['max'] - stats['min']
    
    # 奇偶分析
    odd_count = sum(1 for num in numbers if num % 2 == 1)
    stats['odd_count'] = odd_count
    stats['even_count'] = 20 - odd_count
    stats['odd_ratio'] = odd_count / 20
    
    # 大小分析 (以40为界)
    big_count = sum(1 for num in numbers if num > 40)
    stats['big_count'] = big_count
    stats['small_count'] = 20 - big_count
    stats['big_ratio'] = big_count / 20
    
    # 区间分析
    zone_counts = [0, 0, 0, 0]  # 1-20, 21-40, 41-60, 61-80
    for num in numbers:
        zone_idx = (num - 1) // 20
        zone_counts[zone_idx] += 1
    
    stats['zone_1_count'] = zone_counts[0]  # 1-20
    stats['zone_2_count'] = zone_counts[1]  # 21-40
    stats['zone_3_count'] = zone_counts[2]  # 41-60
    stats['zone_4_count'] = zone_counts[3]  # 61-80
    
    return stats


__all__ = [
    "get_data_run",
    "get_current_number", 
    "format_prediction_results",
    "validate_kl8_numbers",
    "calculate_kl8_statistics",
]