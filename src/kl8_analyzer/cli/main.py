#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KL8分析器主入口
提供统一的命令行接口
"""

import argparse
import sys
from typing import List, Optional

from loguru import logger

from kl8_analyzer.utils.config import get_kl8_config


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    
    parser = argparse.ArgumentParser(
        prog="kl8-analyzer",
        description="KL8快乐8彩票高级分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  kl8-analyzer analysis --advanced_mode 2 --cal_nums 20
  kl8-analyzer cash --start_date 2025-01-01 --end_date 2025-01-31
  kl8-analyzer runner --mode batch --total_create 500
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="KL8 Lottery Analyzer v1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 分析命令
    analysis_parser = subparsers.add_parser("analysis", help="运行KL8分析")
    analysis_parser.add_argument("--advanced_mode", type=int, default=2, help="高级模式 (0-7)")
    analysis_parser.add_argument("--cal_nums", type=int, default=20, help="计算号码数量")
    analysis_parser.add_argument("--window_size", type=int, default=6, help="窗口大小")
    
    # 收益分析命令
    cash_parser = subparsers.add_parser("cash", help="收益分析")
    cash_parser.add_argument("--start_date", type=str, help="开始日期 (YYYY-MM-DD)")
    cash_parser.add_argument("--end_date", type=str, help="结束日期 (YYYY-MM-DD)")
    cash_parser.add_argument("--investment", type=float, default=100.0, help="投资金额")
    
    # 批量运行命令
    runner_parser = subparsers.add_parser("runner", help="批量运行任务")
    runner_parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single", help="运行模式")
    runner_parser.add_argument("--total_create", type=int, default=100, help="总创建数量")
    runner_parser.add_argument("--max_attempts", type=int, default=1000, help="最大尝试次数")
    
    # 数据管理命令
    data_parser = subparsers.add_parser("data", help="数据管理")
    data_parser.add_argument("--download", action="store_true", help="下载历史数据")
    data_parser.add_argument("--start_issue", type=int, help="起始期号")
    data_parser.add_argument("--end_issue", type=int, help="结束期号")
    data_parser.add_argument("--use_sequence", action="store_true", help="使用序列顺序")
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """主入口函数"""
    
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    config = get_kl8_config()
    logger.info("KL8分析器启动 - {}", config.name)
    
    try:
        if parsed_args.command == "analysis":
            from kl8_analyzer.cli.analysis_cli import run_analysis
            return run_analysis(parsed_args)
            
        elif parsed_args.command == "cash":
            from kl8_analyzer.cli.cash_cli import run_cash_analysis
            return run_cash_analysis(parsed_args)
            
        elif parsed_args.command == "runner":
            from kl8_analyzer.cli.runner_cli import run_batch_tasks
            return run_batch_tasks(parsed_args)
            
        elif parsed_args.command == "data":
            from kl8_analyzer.cli.data_cli import run_data_management
            return run_data_management(parsed_args)
            
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        return 130
        
    except Exception as e:
        logger.error("运行出错: {}", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())