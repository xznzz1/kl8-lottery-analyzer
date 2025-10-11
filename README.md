# KL8 (快乐 8) 数据分析工具集

本仓库围绕 `src/analysis` 目录提供的脚本运行，可完成快乐 8 历史数据的下载、统计分析、组合生成与收益回测。当前版本聚焦离线分析，不再包含深度模型训练流程。

> 建议始终在 `python311` Conda 环境或等效的 Python 3.11 虚拟环境中执行命令，以保证依赖一致性。

## 功能清单
- 🔄 `scripts/get_data.py`：下载快乐 8 历史数据，支持顺序出球模式。
- 📦 `src/common.py`：统一封装数据下载、期号查询与历史数据加载。
- 🌐 `src/data_fetcher.py`：带域名白名单与重试机制的抓取器。
- 🎯 `src/analysis/feature_enhancer.py`（新增）：基于“近期动量 + 共现谱分析”的特征增强引擎，统一在 `--feature_mode` 参数下启用。
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
- 📊 原始版与 Plus 版分析脚本，支持多线程与高级算法
- 🧪 完善的测试覆盖与 CI/CD 流程
- 📚 详尽的文档与使用示例

## 更新记录
- **v1.1.0**：新增特征融合选项，优化数据下载脚本
- **v1.0.0**：初始发布，包含基本数据下载与分析功能
