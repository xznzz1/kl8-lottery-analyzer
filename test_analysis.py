#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试CLI分析命令
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("测试分析命令...")

try:
    from src.kl8_analyzer.cli.main import main
    
    # 测试分析命令
    result = main(["analysis", "--advanced_mode", "2", "--cal_nums", "5"])
    print(f"分析命令结果：{result}")
    
except Exception as e:
    print(f"命令执行错误：{e}")
    import traceback
    traceback.print_exc()