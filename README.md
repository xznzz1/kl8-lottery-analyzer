# KL8 (快乐 8) 数据分析工具集

本仓库围绕 `src/analysis` 目录提供的脚本运行，可完成快乐 8 历史数据的下载、统计分析、组合生成与收益回测。当前版本聚焦离线分析，不再包含深度模型训练流程。

> 建议始终在 `python311` Conda 环境或等效的 Python 3.11 虚拟环境中执行命令，以保证依赖一致性。

## 功能清单
- 🔄 `scripts/get_data.py`：下载快乐 8 历史数据，支持顺序出球模式。
- 📦 `src/common.py`：统一封装数据下载、期号查询与历史数据加载。
- 🌐 `src/data_fetcher.py`：带域名白名单与重试机制的抓取器。
- 🎯 `src/analysis/feature_enhancer.py`（新增）：基于“近期动量 + 共现谱分析”的特征增强引擎，统一在 `--feature_mode` 参数下启用。
- 🧮 `src/analysis/rule_miner.py`（新增）：FP-Growth 动态挖掘频繁项集，`--rule_filter` 支持软/硬模式线上筛选组合。
- 📊 `src/analysis/*.py`：原始版 / Plus 版分析与收益脚本，支持高级算法与多线程。
- 🧪 `tests/`：覆盖配置、公共接口、特征增强及抓取模块的 Pytest 用例。

## 快速开始
```bash
conda activate python311
make setup
make run
```

若需更新数据，可执行 `make download-data` 或直接运行 `python scripts/get_data.py`。

### 特征增强示例
```bash
# 混合模式：综合考虑动量与共现谱
python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 120 --limit_line 200 --advanced_mode 2 --feature_mode hybrid

# 并行模式下强调趋势特征
python src/analysis/kl8_analysis_plus.py --cal_nums 15 --total_create 500 --max_workers 6 --advanced_mode 1 --feature_mode momentum

# 仅基于共现谱评分挑选候选
python src/analysis/kl8_analysis.py --advanced_mode 1 --feature_mode cooccurrence --limit_line 150
```

`--feature_mode` 支持以下选项：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `hybrid`（默认） | 近期频率、动量差分与共现谱得分综合排序 | 通用推荐 |
| `momentum` | 强调最近窗口与长期窗口的频次差 | 想捕捉短期趋势时 |
| `cooccurrence` | 聚焦号码共现谱的主特征向量 | 关注号码关联度时 |

### 关联规则筛选
```bash
python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 200 --limit_line 200 --advanced_mode 2 --feature_mode hybrid --rule_filter soft --rule_support 0.6 --rule_confidence 0.8
```
- `--rule_filter hard`：不满足高信心度规则的组合直接从候选集中移除。
- `--rule_filter soft`：对规则出现的组合附加罚值，保留多样性的同时拉差等级。
- 关联门值和项集范围可使用 `config.analysis.rules` 配置或 CLI 参数覆盖。
## 目录结构
```
.
├── config/                # 配置文件（config.yaml）
├── data/kl8/              # 随仓库提供的最小示例数据
├── docs/                  # 架构/API/运维文档
├── examples/              # 高频号码统计示例
├── scripts/               # 数据下载脚本
├── src/
│   ├── analysis/          # 核心分析脚本 + 特征增强工具
│   ├── common.py          # 公共接口封装
│   ├── config.py          # 快乐 8 专用配置模块
│   └── data_fetcher.py    # 历史数据抓取实现
├── tests/                 # Pytest 测试套件
└── Makefile               # 一键任务入口
```

## 配置说明
- `config/config.yaml`：仅保留数据路径与网络请求配置；`make setup` 会自动建立 `data/kl8`、`results`、`logs`。
- `src/config.py`：快乐 8 专用配置定义，包含路径常量、网络超时、彩票参数等。
- 仓库随附 `data/kl8/data.csv` 与 `download_meta.json`，方便离线演示；执行 `make download-data` 可获取最新数据。

## 常见问题
### ⚠️ 重要风险警示：关于 `src/analysis/kl8_running.py` 的内存占用
`kl8_running.py` 会按参数列表笛卡尔展开创建大量并行任务（每个期号 × cal_nums × total_create 组合将触发一个独立子进程/线程），在普通个人电脑上极易造成以下风险：
- 瞬时占用大量内存和文件句柄，出现系统卡顿甚至无响应；
- 若输出目录中存在大量结果文件，I/O 压力会进一步放大卡顿；
- 同时运行分析与现金流模式（`--running_mode 0`）会加剧资源竞争。

使用前请务必确认：
- 充分理解该脚本的工作方式与参数组合的爆炸性；
- 明确自己设备的 CPU/内存/磁盘能力是否足够支撑；
- 从安全的最小参数开始，逐步调大，观察资源监控（任务管理器/Activity Monitor）。

安全参数建议（入门稳妥值，可逐步上调）：
```bash
# 仅分析模式，单期测试
python src/analysis/kl8_running.py \
  --cal_nums_list "5" \
  --total_create_list "10" \
  --nums_range "2024268,2024268" \
  --running_mode 1 \
  --max_workers 2 \
  --download 1

# 扩容时按顺序逐步调整（建议一次只改一个维度）
# 1) 先提高 total_create 到 50，再到 100
# 2) 再增加 cal_nums_list 的取值个数
# 3) 最后再扩 nums_range 的跨度
```

**max_workers 并发度调优指南：**
并发度并非越高越好，过高反而会因资源竞争降低性能。基于测试与实践的建议：

| 设备配置 | 推荐 max_workers | 内存要求 | 说明 |
|----------|------------------|----------|------|
| 4核8GB笔记本 | 2-3 | ≥6GB可用 | 保守起步，适合日常分析 |
| 8核16GB台式机 | 4-6 | ≥12GB可用 | 平衡性能与稳定性 |
| 12核32GB工作站 | 6-10 | ≥24GB可用 | 高性能批处理 |

⚡ **性能调优技巧：**
- 先用 `--max_workers 1` 单线程测试，确保逻辑正确
- 逐步增加到 `2 → 4 → 6`，观察系统资源使用率
- 当CPU利用率超过80%或内存不足时，不要继续增加
- I/O密集型任务（大量文件读写）时适当降低并发度
- 建议通过任务管理器监控内存、CPU、磁盘使用情况

陷阱规避：
- 不要在低内存设备上直接使用大跨度 `--nums_range` 与多值的 `--cal_nums_list`/`--total_create_list`；
- 不要盲目把 `--max_workers` 调到很大；不是越大越快，易造成上下文切换和内存压力；
- 推荐先在 `--running_mode 1` 或 `2` 单独验证，再切换到 `0`。

1. **提示找不到数据文件？**  
   执行 `make setup` 创建目录，再运行 `make download-data`。如仍失败，请检查抓取域名是否可访问。
2. **如何启用高级算法与特征增强？**  
   将 `--advanced_mode` 设置为 1/2，并按需增设 `--feature_mode`。建议在 `Mode 2` 下使用 `hybrid`，获得综合得分。
3. **Plus 版本与单线程版本如何选择？**  
   - `kl8_analysis.py`：轻量、易调试，适合快速验证。  
   - `kl8_analysis_plus.py`：线程池架构，适合大批量生成，可配合 `--max_workers` 调优。
4. **如何批量调度任务？**  
   使用 `src/analysis/kl8_running.py`，可搭配参数列表自动遍历。
5. **下载脚本新增了什么？**  
   `scripts/get_data.py` 默认聚焦快乐 8，仍然支持 `--sequence` 下载顺序数据。

## 测试与质量
- `make ci`：依次执行 `fmt`、`lint`、`test`、`build`。
- `pytest --cov=src`：覆盖率统计，目前核心模块（config/common/data_fetcher/feature_enhancer）均有单元测试。
- `make clean`：清理 `__pycache__`、`.pytest_cache`、build 产物。

## 协作建议
- 新增接口或配置时同步更新 `docs/api.md`、`docs/architecture.md`，并补充测试。
- 如果未来扩展到其他彩票玩法，请在 `docs/decision_record.md` 记录设计取舍，`ASSUMPTIONS.md` 说明边界条件。
- 若需恢复模型训练能力，可在独立分支重建 pipeline，再合并到主线。

## 亮点
- 🔄 全新特征增强引擎，支持动量、共现谱、PCA主成分等多种特征融合
- 🧮 Dirichlet 平滑 + FP-Growth 关联规则：`feature_enhancer` \u63d0\u4f9b Dirichlet-Multinomial \u5f3a\u5316\u6e90\uff0c`rule_miner` \u5b9e\u65f6\u534f\u52a9 \u8fc7\u6ee4\u9ad8\u98ce\u9669\u7ec4\u5408
- 📊 原始版与 Plus 版分析脚本，支持多线程与高级算法
- 🧪 完善的测试覆盖与 CI/CD 流程
- 📚 详尽的文档与使用示例

## 更新记录
- **v1.1.0**：新增特征融合选项，优化数据下载脚本
- **v1.0.0**：初始发布，包含基本数据下载与分析功能
