# 公共 API 一览

| 模块 | 函数 | 参数 | 返回值 | 说明 |
|------|------|------|--------|------|
| `src.common` | `get_data_run(name, sequence_mode=False, start_issue=None, end_issue=None)` | `name`: 彩票代号；`sequence_mode`: 是否抓取顺序数据；`start_issue`/`end_issue`: 期号区间 | `None` | 下载快乐 8 历史数据并写入 `data/kl8/data.csv`。 |
| `src.common` | `get_current_number(name)` | `name`: 彩票代号 | `str` | 返回快乐 8 最新一期的期号。 |
| `src.common` | `load_history(name)` | `name`: 彩票代号 | `pandas.DataFrame` | 从本地 CSV 读取历史数据。 |
| `src.data_fetcher` | `download_history(code, start=None, end=None, use_sequence_order=False, client=None)` | 同上 | `DownloadResult` | 带重试的网络抓取实现，可选顺序数据模式。 |
| `src.data_fetcher` | `get_current_issue(code, client=None)` | 彩票代号、可选客户端 | `str` | 读取 500.com 页面上的最新期号。 |
| `src.data_fetcher` | `load_history(code)` | 彩票代号 | `pandas.DataFrame` | 与 `common.load_history` 等价，直接暴露底层能力。 |

> 由于 `src/analysis` 下的大型脚本仍以 CLI 方式存在，若要在代码中复用，请改用 `subprocess` 调用并传入完整参数。
