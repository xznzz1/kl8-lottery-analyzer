# -*- coding: utf-8 -*-
"""
KL8 (快乐8) 数据分析工具集。

当前仓库聚焦于：
- 历史数据下载与加载 (`src.data_fetcher`)
- 顶层脚本共享接口 (`src.common`)
- 多种分析/回测脚本 (`src.analysis` 包)
"""
from .config import *
from . import common
__all__ = ["analysis", "common", "config", "data_fetcher"]
