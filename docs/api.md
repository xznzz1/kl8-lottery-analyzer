# 公共 API 一览（🚀 包含多线程优化版本）

## 核心 API

| 模块 | 函数 | 参数 | 返回值 | 说明 |
|------|------|------|--------|------|
| `src.common` | `get_data_run(name, sequence_mode=False, start_issue=None, end_issue=None)` | `name`: 彩票代号；`sequence_mode`: 是否抓取顺序数据；`start_issue`/`end_issue`: 期号区间 | `None` | 下载快乐 8 历史数据并写入 `data/kl8/data.csv`。 |
| `src.common` | `get_current_number(name)` | `name`: 彩票代号 | `str` | 返回快乐 8 最新一期的期号。 |
| `src.common` | `load_history(name)` | `name`: 彩票代号 | `pandas.DataFrame` | 从本地 CSV 读取历史数据。 |
| `src.data_fetcher` | `download_history(code, start=None, end=None, use_sequence_order=False, client=None)` | 同上 | `DownloadResult` | 带重试的网络抓取实现，可选顺序数据模式。 |
| `src.data_fetcher` | `get_current_issue(code, client=None)` | 彩票代号、可选客户端 | `str` | 读取 500.com 页面上的最新期号。 |
| `src.data_fetcher` | `load_history(code)` | 彩票代号 | `pandas.DataFrame` | 与 `common.load_history` 等价，直接暴露底层能力。 |

## 🚀 多线程优化 API（Plus版本）

### kl8_analysis_plus.py
```python
# 关键函数签名
def download_data_if_needed(download, cal_nums):
    """单线程数据下载函数，避免重复下载"""
    pass

def sub_process(args_tuple):
    """线程池工作函数，线程安全的号码生成"""
    pass

# 使用示例
from concurrent.futures import ThreadPoolExecutor
import threading

# 线程池配置
max_workers = 8
results_lock = threading.Lock()
shared_results = []

# 并发执行
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(sub_process, args) for args in task_list]
```

### kl8_cash_plus.py
```python
# 关键函数签名
def check_lottery(file_path, file_name, data_dict, download_flag):
    """线程安全的收益分析函数"""
    pass

def process_files_parallel(file_list, max_workers=4):
    """并行处理多个预测文件"""
    pass

# 使用示例
results = process_files_parallel(
    file_list=["result1.txt", "result2.txt"], 
    max_workers=6
)
```

## 线程安全特性

### 共享资源保护
```python
import threading

# 全局锁定义
results_lock = threading.Lock()
progress_lock = threading.Lock()

# 安全访问模式
with results_lock:
    shared_results.append(new_result)
```

### 错误处理机制
```python
# 线程级错误隔离
try:
    result = worker_function(args)
    with results_lock:
        shared_results.append(result)
except Exception as e:
    logger.error(f"工作线程错误: {e}")
    # 单线程失败不影响其他线程
```

> **使用建议**：对于大规模批量处理，推荐使用 Plus 版本的多线程优化 API。传统版本适合单次小规模分析。
> 
> **调用方式**：由于 `src/analysis` 下的脚本仍以 CLI 方式存在，若要在代码中复用，请改用 `subprocess` 调用并传入完整参数。
