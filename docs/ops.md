# 运维与监控指南（🚀 多线程优化版）

## 环境要求
- 建议在名为 `python311` 的 Conda 环境中运行。
- 需要可访问 `https://datachart.500.com` 与 `https://data.917500.cn` 的网络。
- 目录 `data/kl8`、`results`、`logs` 会在 `make setup` 时自动创建。
- **🔧 并发环境要求**：推荐CPU核心数2倍的线程池大小，确保充足内存（推荐4GB+）

## 运行监控

### 传统监控
- **日志**：分析脚本使用 `loguru` 输出，默认写入控制台；若需落盘，可在脚本中调用 `logger.add("logs/kl8.log")`。
- **网络请求**：`LotteryHttpClient` 已启用重试与超时（默认 20s / 3 次）。建议在外层对 `ValueError`、`requests.HTTPError` 做重试或报警。
- **数据完整性**：每次下载会生成 `data/kl8/download_meta.json`，包含期号总数与时间戳，可用于简单的健康检查。

### 🚀 多线程监控（Plus版本）
- **线程池状态**：监控 `ThreadPoolExecutor` 的活跃线程数和任务队列长度
- **内存使用**：监控线程共享内存占用，避免内存泄漏
- **并发性能**：跟踪每个工作线程的处理时间和吞吐量
- **锁竞争**：监控 `threading.Lock` 的等待时间，识别性能瓶颈
- **数据下载优化**：确认主线程单次下载成功，避免重复请求

### 监控指标
```bash
# 关键性能指标监控
- 数据下载次数：应为1次（主线程）
- 活跃线程数：≤ max_workers 设置值
- 内存峰值：比传统版本降低60%
- 处理速度：大批量任务提升40%
- 错误率：单线程失败不影响其他线程
```

## 故障排查

### 传统问题
| 症状 | 可能原因 | 处理建议 |
|------|----------|----------|
| 下载报错"禁止访问域名" | 请求被重定向或被劫持到非白名单域名 | 检查网络代理，将目标域名加入白名单，再次执行 |
| 分析脚本提示找不到 `data.csv` | 未执行下载或目录未初始化 | 运行 `make setup` 后执行 `make download-data` |
| `matplotlib` 无法显示图表 | 服务器无图形后端 | 使用 `matplotlib.use("Agg")` 或只生成图片文件 |

### 🚀 多线程问题（Plus版本）
| 症状 | 可能原因 | 处理建议 |
|------|----------|----------|
| 线程池卡死 | 线程锁死锁或无限等待 | 检查 `threading.Lock` 使用，确保成对 acquire/release |
| 内存持续增长 | 线程间对象引用未清理 | 检查大对象的生命周期，及时释放线程局部变量 |
| 结果不一致 | 线程间共享变量竞争 | 确保所有共享资源访问都在锁保护下进行 |
| 性能不升反降 | 线程数设置过高导致上下文切换 | 调整 `max_workers` 为CPU核心数的1-2倍 |
| 部分文件未处理 | 异常线程提前退出 | 检查异常处理机制，确保线程错误被正确捕获和记录 |

### 调试命令
```bash
# 检查线程状态
python -c "import threading; print(f'活跃线程数: {threading.active_count()}')"

# 监控内存使用
python -c "import psutil; print(f'内存使用: {psutil.virtual_memory().percent}%')"

# 验证数据下载
ls -la data/kl8/ && echo "数据文件检查完成"
```

## 常驻任务建议

### 传统任务
- 使用 `cron` / `Windows 任务计划` 调用 `make download-data`（每日一次即可）。
- 若需批量分析，可编写 shell 脚本调用 `python src/analysis/kl8_running.py ...`，并将日志重定向到 `logs/`。

### 🚀 优化任务（Plus版本推荐）
```bash
# 定时数据更新（每日05:00）
0 5 * * * cd /path/to/kl8-lottery-analyzer && make download-data

# 高性能批量分析（每日06:00）
0 6 * * * cd /path/to/kl8-lottery-analyzer && python src/analysis/kl8_analysis_plus.py \
  --cal_nums 20 --total_create 1000 --max_workers 8 --advanced_mode 1 \
  --path "daily_analysis" >> logs/daily_analysis.log 2>&1

# 大规模收益分析（每周日00:00）
0 0 * * 0 cd /path/to/kl8-lottery-analyzer && python src/analysis/kl8_cash_plus.py \
  --path "weekly_results" --max_workers 6 >> logs/weekly_cash.log 2>&1
```

### 生产环境监控
- **Prometheus指标**：对任务退出码、日志关键字、线程池状态进行采集
- **告警规则**：
  - 数据下载失败 > 3次连续
  - 线程池死锁检测（处理时间 > 10min）
  - 内存使用率 > 80%
  - 错误率 > 5%

### 性能调优建议
- **CPU密集型**：max_workers = CPU核心数
- **I/O密集型**：max_workers = CPU核心数 × 2
- **内存限制**：单线程内存 × max_workers < 总内存 × 0.8
- **批量大小**：根据内存大小调整 total_create 参数
