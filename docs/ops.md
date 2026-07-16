# 运维与监控指南（多线程版本）

## 环境要求
- 建议在名为 `python311` 的 Conda 环境中运行，保持依赖一致。
- 本次Windows自动执行已验证仓库为 `D:\lottery\kl8-lottery-analyzer` 并使用其中的 `.venv`；不得在C盘创建仓库副本、虚拟环境、数据、结果或缓存。应用本身按当前仓库位置解析输出边界，不硬编码盘符。
- 网络需可访问 `https://datachart.500.com` 与 `https://data.917500.cn`。
- `make setup` 会在当前仓库父目录创建 `.tmp`、`.cache`，再安装依赖并创建仓库内 `data_cache/kl8`、`results/logs`、`reports`；Makefile按自身位置解析项目根，不受调用时当前目录影响。本次检出中这些目录分别解析为 `D:\lottery\.tmp`、`D:\lottery\.cache`。
- 建议在批量任务前执行：
  ```bash
  make download-data
  make train-graph
  python src/analysis/kl8_analysis.py ...  # 先单次验证，再批量投入
  ```

## 冻结策略前瞻监测

旧科学报告和截止2026186的final holdout已经查看并永久冻结。下载新数据后只运行：

```powershell
.venv\Scripts\python.exe scripts\get_data.py --name kl8
.venv\Scripts\python.exe scripts\prospective_evaluate.py --next-issue 2026188
```

前瞻脚本固定读取`data_cache/kl8/data.csv`，只写`results/prospective/`和
`reports/kl8_prospective_report.md`；Windows下还校验当前解释器必须是
`D:\lottery\kl8-lottery-analyzer\.venv\Scripts\python.exe`，临时文件固定写
`D:\lottery\.tmp`。禁止用新增数据重新运行旧holdout切分或选择参数。

运行先校验`config/scientific_freeze.json`中的六文件`source_manifest`，任一哈希变化即
停止并要求建立新策略版本。下一期号必须来自官方明确值；脚本绝不使用“最新期号+1”。
当前2026188候选封存为`reports/prospective_manifests/2026188.json`；已有manifest
只读且不得删除或覆盖，新一期必须创建新文件。提交manifest前不得把候选称为完整事前封存。

运行成功时应核对：每个新增期恰好500条原始记录、摘要120行、下一期候选100行；
重复运行记录文件SHA-256不变。冲突错误表示历史数据、冻结配置或已有记录不一致，
应保留现场并检查，不得删除已有记录或manifest后静默重算。

## 关键监控指标
| 项目 | 检查方式 | 目标 |
|------|----------|------|
| 下载次数 | 主线程日志 | 每次运行仅下载一次 |
| 线程数 | `threading.active_count()` | ≈ `max_workers` + 1 |
| 内存使用 | `psutil.virtual_memory()` | < 80% |
| I/O 负载 | 监控磁盘写入速率 | 避免长时间 100% |
| Copula 诊断 | 日志 `cond≈...`、`effective_draws=...` | 条件数不过高、样本量 ≥ `min_draws` |
| 前瞻记录完整性 | 每期复合主键计数 | 500且无重复 |
| 冻结产物哈希 | 前瞻运行前自动校验 | 与`scientific_freeze.json`一致 |

示例命令：
```bash
python -c "import threading; print('active threads:', threading.active_count())"
python -c "import psutil; print('memory usage:', psutil.virtual_memory().percent, '%')"
```

## 故障排查
| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 下载失败 | 网络受限或域名未放行 | 配置代理或启用离线模式（`--download 0`） |
| 线程卡住 | 锁竞争 / I/O 拥堵 | 降低 `--max_workers`，清理结果目录 |
| Copula 跳过 | 样本量不足 | 提高 `--limit_line` 或 `analysis.copula.min_draws` |
| 互信息惩罚过高 | 权重偏大 | 调整 `analysis.graph_embedding.weight` |
| 结果混淆 | 输出目录冲突 | 使用 `--path` 区分任务 |
| 前瞻记录冲突 | 数据倒退、旧行变化或冻结配置漂移 | 停止写入，核对数据快照与复合主键，不得删除旧记录规避错误 |

调试示例：
```bash
python src/analysis/kl8_analysis.py --max_workers 1 --copula_mode force --debug 1
python src/analysis/kl8_running.py --max_workers 1 --cal_nums_list "5" --total_create_list "10"
```

## 常驻任务建议
```bash
# 每日 05:00 下载数据
0 5 * * * cd /d/lottery/kl8-lottery-analyzer && make download-data

# 每日 06:00 执行全算法分析（示例参数）
0 6 * * * cd /d/lottery/kl8-lottery-analyzer && \
python src/analysis/kl8_analysis.py \
  --cal_nums 20 --total_create 240 --limit_line 240 \
  --advanced_mode 2 --feature_mode hybrid \
  --rule_filter soft --rule_support 0.08 --rule_confidence 0.7 \
  --copula_mode force --copula_samples 64 --copula_shrinkage 0.1 \
  >> results/logs/daily_analysis.log 2>&1

# 每周日 00:00 批量收益分析
0 0 * * 0 cd /d/lottery/kl8-lottery-analyzer && \
python src/analysis/kl8_cash_plus.py --path weekly_results --max_workers 6 \
  >> results/logs/weekly_cash.log 2>&1
```

## 告警建议
- 下载失败连续 ≥3 次。
- 单次任务执行超过 10 分钟未结束。
- 内存使用率 ≥ 85% 持续 5 分钟。
- Copula 拟合条件数异常升高（>1e6）或样本量低于阈值。
- 结果输出目录增长过快，需要定期清理。
- 前瞻记录不满足每新增期500条，或冻结产物哈希发生变化。
