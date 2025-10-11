# -*- coding: utf-8 -*-
"""
KL8数据管理CLI命令
"""

import argparse
from loguru import logger

def run_data_management(args: argparse.Namespace) -> int:
    """运行KL8数据管理"""
    
    if args.download:
        logger.info("开始下载KL8历史数据")
        
        try:
            from kl8_analyzer.utils.common import get_data_run
            
            get_data_run(
                cq=1 if args.use_sequence else 0,
                start_issue=args.start_issue,
                end_issue=args.end_issue
            )
            
            logger.success("KL8数据下载完成")
            return 0
            
        except Exception as e:
            logger.error("KL8数据下载失败: {}", e)
            return 1
    
    else:
        logger.info("请指定数据管理操作 (如 --download)")
        return 1


def main(args=None):
    """数据管理命令的独立入口"""
    parser = argparse.ArgumentParser(description="KL8数据管理工具")
    parser.add_argument("--download", action="store_true", help="下载历史数据")
    parser.add_argument("--start_issue", type=int, help="起始期号")
    parser.add_argument("--end_issue", type=int, help="结束期号")
    parser.add_argument("--use_sequence", action="store_true", help="使用序列顺序")
    
    parsed_args = parser.parse_args(args)
    return run_data_management(parsed_args)