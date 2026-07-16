# KL8 (快乐 8) 数据分析工具集

本仓库提供快乐8历史数据下载、统计分析和严格时间滚动评估。科学评估的目标是检验策略能否在样本外优于随机选号，不宣称能够预测随机开奖；深度训练流程保持可选。

> **环境建议**：请始终在名为 `python311` 的 Conda 环境或等效的 Python 3.11 虚拟环境中执行命令，以保持依赖一致。

## 功能清单
- 🔄 `scripts/get_data.py`：下载快乐 8 历史数据，支持顺序出球模式。
- 📦 `src/common.py`：封装数据下载、期号查询与历史数据加载。
- 🌐 `src/data_fetcher.py`：带域名白名单、超时与重试机制的抓取器。
- 🔬 `scripts/backtest_baselines.py`：固定每期4元，比较选一至选十、两种出票方式和多种无泄漏策略。
- 🎯 `src/analysis/feature_enhancer.py`：整合近期动量、共现谱、Dirichlet 平滑、PCA 与图嵌入的综合特征评分。
- 🧮 `src/analysis/rule_miner.py`：基于 FP-Growth 的频繁项集与关联规则筛选，支持软/硬模式。
- 🎲 `src/analysis/copula_sampler.py`：高斯 Copula 多样性采样，结合互信息惩罚提升组合相关性建模。
- 🧠 `scripts/train_graph_embeddings.py`：Node2Vec 图嵌入训练脚本，自动识别 CPU / NVIDIA GPU / AMD ROCm 设备。
- 🧪 `tests/`：覆盖配置、特征增强、规则挖掘、Copula 采样等模块的 Pytest 套件。

## 快速开始
```bash
conda activate python311
make setup
make download-data             # 可重复执行，获取最新历史数据
make scientific-backtest       # 生成CSV与科学报告（显式封顶情景）
```

完整数据和 `results/` 默认不提交。回测产物包括 `results/scientific/*.csv` 与
`reports/kl8_scientific_report.md`。浮动奖封顶情景只用于敏感性演示，不是逐期实际
兑付；规则版本和替代输入见 [官方规则说明](docs/official_rules.md)。

PyTorch不是下载、统计和科学回测依赖。图嵌入训练仅是可选旧功能；科学holdout默认
禁用无法证明训练截止点的全样本图嵌入缓存。

## 一键启用全算法
以下命令会同时启用高级模式（遗传 + 贝叶斯 + Copula + 互信息惩罚）、特征增强、关联规则过滤以及可调的 Copula 采样参数：

```bash
python src/analysis/kl8_analysis.py \
  --cal_nums 20 \
  --total_create 240 \
  --limit_line 240 \
  --advanced_mode 2 \
  --feature_mode hybrid \
  --rule_filter soft \
  --rule_support 0.08 \
  --rule_confidence 0.7 \
  --copula_mode force \
  --copula_samples 64 \
  --copula_shrinkage 0.1
```

> 建议在执行前运行 `make train-graph` 以生成最新的图嵌入缓存，确保特征增强通道生效。

## 特征增强与模式组合示例
```bash
# 混合模式：综合近期动量与共现谱
python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 120 --limit_line 200 --advanced_mode 2 --feature_mode hybrid

# 并行模式下强调趋势：使用 Plus 版本并设定线程池
python src/analysis/kl8_analysis_plus.py --cal_nums 15 --total_create 500 --max_workers 6 --advanced_mode 1 --feature_mode momentum

# 仅基于共现谱评分挑选候选
python src/analysis/kl8_analysis.py --advanced_mode 1 --feature_mode cooccurrence --limit_line 150
```

`--feature_mode` 选项：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `hybrid`（默认） | 动量、共现谱、Dirichlet、PCA、图嵌入综合排序 | 通用推荐 |
| `momentum` | 强调近期与长期窗口频率差 | 捕捉短期热点时 |
| `cooccurrence` | 聚焦号码共现谱主特征向量 | 注重号码协同关系时 |

## 关联规则筛选
```bash
python src/analysis/kl8_analysis.py \
  --cal_nums 10 --total_create 200 --limit_line 200 \
  --advanced_mode 2 --feature_mode hybrid \
  --rule_filter soft --rule_support 0.6 --rule_confidence 0.8
```
- `--rule_filter hard`：违反高置信度规则的组合将被直接剔除。
- `--rule_filter soft`：违规组合接受罚分，但可保留以丰富多样性。
- 相关阈值可在 `config.analysis.rules` 中配置，也可通过 CLI 覆盖。

## Copula / 图嵌入 / 互信息
- Copula 采样通过 `--copula_mode auto|off|force` 控制，`--copula_samples`、`--copula_shrinkage`、`--copula_min_draws`、`--copula_multiplier` 可调精细权重。
- 图嵌入缓存默认为 `data_cache/graph_embeddings.npz`，生成命令：`make train-graph`。
- 互信息惩罚默认在高级模式开启，如需调节可调整 `analysis.graph_embedding.weight`。

## 目录结构
```
.
├── config/                # 配置文件（config.yaml）
├── data/kl8/              # 随仓库提供的最小示例数据
├── docs/                  # 架构 / API / 运维文档
├── examples/              # 高频号码统计示例
├── scripts/               # 数据下载与图嵌入训练脚本
├── src/
│   ├── analysis/          # 核心分析脚本与工具
│   ├── scientific/        # 无泄漏策略、奖金、统计检验与滚动评估
│   ├── common.py          # 公共接口
│   ├── config.py          # 快乐 8 配置入口
│   └── data_fetcher.py    # 历史数据抓取
├── tests/                 # Pytest 测试套件
└── Makefile               # 一键任务入口
```

## 常见问题 FAQ
1. **提示找不到数据文件？**  
   先执行 `make setup` 创建目录，再运行 `make download-data`。确保网络可访问 `https://datachart.500.com` 和 `https://data.917500.cn`。
2. **如何启用高级算法组合？**  
   使用前述“一键启用全算法”命令或将 `--advanced_mode 2` 与 `--copula_mode auto/force`、`--feature_mode hybrid`、`--rule_filter soft` 联合使用。
3. **Plus 版本与单线程版本如何选择？**  
   - `kl8_analysis.py`：轻量、易调试，适合快速验证。  
   - `kl8_analysis_plus.py`：线程池并行，适合批量生成，需合理设置 `--max_workers`。
4. **批量任务调度**  
   使用 `src/analysis/kl8_running.py` 可遍历参数组合，并通过 `--running_mode` 控制执行策略。
5. **Copula / 图嵌入如何维护？**  
   - 建议定期运行 `make train-graph` 更新嵌入。  
   - Copula 采样需确保 `limit_line` 不小于 `analysis.copula.min_draws`（默认 180），不足时会自动退回传统策略。
6. **旧高级脚本能直接当科学回测吗？**
   不能。审计发现部分旧流程存在倒序索引、全样本预处理或缓存截止点不明的问题。
   请使用 `scripts/backtest_baselines.py`；它按期号升序并保证预测第t期只读取t之前数据。

## `kl8_running.py` 资源提示
批量运行会按参数列表笛卡尔展开生成大量任务（每个期号 × cal_nums × total_create），容易造成内存和文件句柄压力。请从小规模参数起步，确认后再逐步放大，同时监控 `max_workers`、内存及磁盘空间。

## 测试与质量
- `make ci`：依次执行 `fmt`、`lint`、`test`、`build`。
- `pytest --cov=src`：查看覆盖率（核心模块覆盖率 ≥ 80%）。
- `make clean`：清理缓存、编译产物与 `__pycache__`。

## 协作建议
- 新增接口或配置时同步更新 `docs/api.md`、`docs/architecture.md`、`ASSUMPTIONS.md`。
- 如果扩展其他玩法，请在 `docs/decision_record.md` 记录设计取舍，并在 `config/` 提供新的配置段。
- 若需恢复深度模型训练，可在独立分支重建 pipeline 后再合入主线。

## 版本历史
- **v1.4.0**：新增 Copula 多样性采样、互信息惩罚、图嵌入训练脚本及全算法命令。
- **v1.3.0**：引入 Dirichlet 平滑与关联规则筛选。
- **v1.1.0**：新增特征融合选项、优化数据下载脚本。
- **v1.0.0**：初始发布，包含基本数据下载与分析能力。
