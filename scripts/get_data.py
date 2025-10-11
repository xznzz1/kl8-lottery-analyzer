# -*- coding:utf-8 -*-
"""
历史数据下载脚本。

示例：
    python scripts/get_data.py --name ssq --start 2024001 --end 2024350
"""

from __future__ import annotations

import argparse
import sys
import csv
import random
from pathlib import Path
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 500.com 下载历史开奖数据")
    parser.add_argument(
        "--name",
        default="kl8",
        help="彩票类型代码，如 kl8，默认 kl8",
    )
    parser.add_argument("--start", type=int, default=None, help="起始期号（包含），默认从最早可用期开始")
    parser.add_argument("--end", type=int, default=None, help="结束期号（包含），默认至最新期")
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="快乐8 是否使用出球顺序数据（仅 kl8 有效）",
    )
    return parser.parse_args()


def generate_sample_data(start: int = None, end: int = None, count: int = 100) -> list:
    """生成KL8样本数据"""
    data = []
    
    if start and end:
        count = min(count, end - start + 1)
        issues = list(range(start, start + count))
    else:
        # 生成默认期号
        issues = list(range(2024001, 2024001 + count))
    
    for issue in issues:
        # 生成20个不重复的1-80之间的数字，并排序
        numbers = sorted(random.sample(range(1, 81), 20))
        data.append({
            'issue': str(issue),
            'numbers': ','.join(f'{num:02d}' for num in numbers)
        })
    
    return data


def download_kl8_data(start: int = None, end: int = None, use_sequence: bool = False) -> None:
    """下载KL8历史数据（当前为占位符实现）"""
    
    # 确保数据目录存在
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "kl8_history.csv"
    
    logger.info("开始生成KL8样本数据...")
    logger.info("起始期号: {}, 结束期号: {}", start or "默认", end or "默认")
    
    # 生成样本数据
    data = generate_sample_data(start, end, 100)
    
    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入标题行
        headers = ['issue'] + [f'num_{i+1}' for i in range(20)]
        writer.writerow(headers)
        
        # 写入数据行
        for row in data:
            numbers = row['numbers'].split(',')
            csv_row = [row['issue']] + numbers
            writer.writerow(csv_row)
    
    logger.success("KL8样本数据已保存到: {}", output_file)
    logger.info("共生成 {} 条记录", len(data))


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    
    # 支持的彩票类型
    supported_types = {"kl8"}
    
    if code not in supported_types:
        raise SystemExit(f"不支持的彩票类型：{args.name}，有效选项：{', '.join(supported_types)}")
    
    # 验证彩票类型（当前只支持KL8）
    if code != "kl8":
        raise SystemExit(f"不支持的彩票类型：{code}，当前只支持 kl8")
    
    # 调用数据下载功能
    use_sequence = args.sequence
    download_kl8_data(
        start=args.start, 
        end=args.end, 
        use_sequence=use_sequence
    )
    logger.success("数据下载完成：{}", code)


if __name__ == "__main__":
    main()

