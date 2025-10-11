# -*- coding: utf-8 -*-
"""
KL8批量运行CLI命令
"""

import argparse
from loguru import logger

def run_batch_tasks(args: argparse.Namespace) -> int:
    """运行KL8批量任务"""
    
    logger.info("开始KL8批量任务 - 模式: {}, 总数量: {}", args.mode, args.total_create)
    
    try:
        logger.info("批量任务参数: 模式={}, 总创建数={}, 最大尝试数={}", 
                   args.mode, args.total_create, args.max_attempts)
        
        # TODO: 集成实际的批量运行逻辑
        # from kl8_analyzer.core.runner import KL8Runner
        # runner = KL8Runner()
        # results = runner.run_batch(
        #     mode=args.mode,
        #     total_create=args.total_create,
        #     max_attempts=args.max_attempts
        # )
        
        logger.success("KL8批量任务完成")
        return 0
        
    except Exception as e:
        logger.error("KL8批量任务失败: {}", e)
        return 1


def main(args=None):
    """批量运行命令的独立入口"""
    parser = argparse.ArgumentParser(description="KL8批量运行工具")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single", help="运行模式")
    parser.add_argument("--total_create", type=int, default=100, help="总创建数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="最大尝试次数")
    
    parsed_args = parser.parse_args(args)
    return run_batch_tasks(parsed_args)