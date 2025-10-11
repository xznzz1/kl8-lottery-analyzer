# 运维与监控指南

## 环境要求
- 建议在名为 `python311` 的 Conda 环境中运行。
- 需要可访问 `https://datachart.500.com` 与 `https://data.917500.cn` 的网络。
- 目录 `data/kl8`、`results`、`logs` 会在 `make setup` 时自动创建。

## 运行监控
- **日志**：分析脚本使用 `loguru` 输出，默认写入控制台；若需落盘，可在脚本中调用 `logger.add("logs/kl8.log")`。
- **网络请求**：`LotteryHttpClient` 已启用重试与超时（默认 20s / 3 次）。建议在外层对 `ValueError`、`requests.HTTPError` 做重试或报警。
- **数据完整性**：每次下载会生成 `data/kl8/download_meta.json`，包含期号总数与时间戳，可用于简单的健康检查。

## 故障排查
| 症状 | 可能原因 | 处理建议 |
|------|----------|----------|
| 下载报错“禁止访问域名” | 请求被重定向或被劫持到非白名单域名 | 检查网络代理，将目标域名加入白名单，再次执行 |
| 分析脚本提示找不到 `data.csv` | 未执行下载或目录未初始化 | 运行 `make setup` 后执行 `make download-data` |
| `matplotlib` 无法显示图表 | 服务器无图形后端 | 使用 `matplotlib.use("Agg")` 或只生成图片文件 |

## 常驻任务建议
- 使用 `cron` / `Windows 任务计划` 调用 `make download-data`（每日一次即可）。
- 若需批量分析，可编写 shell 脚本调用 `python src/analysis/kl8_running.py ...`，并将日志重定向到 `logs/`。
- 生产环境请结合外部监控（如 Prometheus）对任务退出码、日志关键字进行采集。
