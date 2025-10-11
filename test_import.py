#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试导入是否会触发argparse冲突
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("开始测试模块导入...")

try:
    print("导入config...")
    from src.kl8_analyzer.utils.config import get_kl8_config
    
    print("导入main...")
    from src.kl8_analyzer.cli.main import main
    
    print("测试main函数调用...")
    result = main(["--help"])
    print(f"结果：{result}")
    
except Exception as e:
    print(f"导入错误：{e}")
    import traceback
    traceback.print_exc()