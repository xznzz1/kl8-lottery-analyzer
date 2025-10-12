# KL8 算法扩展研究报告（2025.10 更新）

## 1. 现状回顾
- 核心入口 `src/analysis/kl8_analysis.py`：负责参数解析、历史数据加载、候选组合生成与写盘，支持基础模式与 `--advanced_mode` 下的多策略调度。
- 特征增强 `src/analysis/feature_enhancer.py`：整合近期频率、动量、共现谱、人为先验等得分，并输出结构化 `FeatureDebugInfo`。
- 规则挖掘 `src/analysis/rule_miner.py`：利用频繁项集过滤高风险组合，提供软惩罚与硬裁剪两种模式。
- 数据与脚本体系：`config/config.yaml` 管理参数、`scripts/get_data.py`/`examples` 覆盖下载与演示，`tests/` 下的 Pytest 套件保证核心路径可回归。

## 2. 新增模型与状态概览
| 模型/机制 | 迭代编号 | 状态 | 代码入口 | 说明 |
| --- | --- | --- | --- | --- |
| Dirichlet-Multinomial 层级平滑 | 短期 | ✅ 已落地 | `feature_enhancer._compute_dirichlet_scores` | 保持窗口自适应、提供均值+方差调试信息，参数可通过 YAML 调整 |
| FP-Growth 关联规则挖掘 | 短期 | ✅ 已落地 | `src/analysis/rule_miner.py` | 缓存支持、CLI 阈值覆盖，集成 `append_result_with_rules` |
| Copula 多样性采样 | 中期 | ✅ 新增 | `src/analysis/copula_sampler.py` + `advanced_number_generation` | 估计 80 维相关结构，支持 `--copula_*` CLI 覆盖 |
| Node2Vec 图嵌入特征 | 中期 | ✅ 新增 | `scripts/train_graph_embeddings.py` + `feature_enhancer` | PyTorch Skip-gram 训练，CPU/GPU 自动切换，特征通道融入综合评分 |
| 互信息多样性惩罚（新提案） | 自主扩展 | ✅ 新增 | `src/analysis/mutual_information.py` | 计算 80×80 互信息矩阵，在高级模式评分阶段扣减高相关组合 |

## 3. 实施记录
### 3.1 短期任务：Dirichlet 平滑 + 规则挖掘
- **代码调整**：`feature_enhancer` 保持 Dirichlet 通道，并新增调试字段；`config/config.yaml` 暴露 `analysis.dirichlet` 与 `analysis.rules` 参数，默认适配 120 期窗口。
- **验证**：`tests/test_feature_enhancer.py` 增强断言、`tests/test_rule_miner.py`（原有）覆盖软/硬模式；自测中随机数据保证排序稳定。
- **文档**：`docs/decision_record.md` 更新默认先验来源，`ASSUMPTIONS.md` 记录“窗口 ≥ min_draws” 前提。

### 3.2 中期任务：Copula 采样
- **实现**：新增 `CopulaSampler`（NumPy 实现，高斯 Copula + 特征值截断）与辅助函数 `generate_copula_candidates`，并在高级策略中以 `--copula_mode` 控制是否参与。
- **集成点**：候选组合池加入 Copula 产出，自动去重，日志记录条件数与有效样本。配置支持 CLI 覆盖 `min_draws / shrinkage / samples / multiplier / seed`。
- **测试**：`tests/test_copula_sampler.py` 验证生成长度、去重、随机种子稳定性与异常分支。

### 3.3 中期任务：图嵌入特征
- **训练脚本**：`scripts/train_graph_embeddings.py` 基于随机游走 + Skip-gram（负采样）训练 80 个节点嵌入，默认使用 `auto` 设备（CUDA/ROCm/MPS/CPU）。
- **特征融合**：`feature_enhancer` 加载 `npz` 缓存并归一化为 `graph_embedding_scores`，新增 `clear_graph_embedding_cache()` 供测试复位。
- **调用方式**：新增 Makefile 目标 `make train-graph`，README/运行手册更新快速指引。
- **测试**：`tests/test_feature_enhancer.py::test_compute_enhanced_scores_with_graph_embeddings` 避免回归；脚本本身输出训练元信息（epoch 损失、耗时、设备）。

### 3.4 自主扩展：互信息多样性惩罚
- **目的**：在高级评估阶段约束高相关号码，以互信息矩阵代替启发式惩罚。
- **实现**：`src/analysis/mutual_information.py` 通过指示矩阵快速计算 `80×80` 互信息（带平滑），在 `advanced_number_generation` 中扣减 `mi_penalty`。
- **验证**：`tests/test_mutual_information.py` 检查矩阵对称性、对角线归零、空输入回退。
- **文档**：在报告与决策记录中说明新增策略及权衡（惩罚尺度随期数变化需关注）。

## 4. 自测与验证
- `pytest tests -v`（详见 `agent_report.md`）覆盖新增模块：
  - `tests/test_copula_sampler.py`、`tests/test_mutual_information.py`、更新后的 `test_feature_enhancer.py`。
  - 原有回归套件（数据下载、分析指标、共享工具）保持通过。
- 关键日志：
  - Copula 拟合输出 `cond≈xxx`、`effective_draws=yyy` 便于排查。
  - 图嵌入训练脚本逐 epoch 打印平均损失，保存 metadata（维度、步长、设备、耗时）。
- 覆盖率：维持项目默认（pytest-cov 统计 >80%），互信息/采样模块均被直接测试路径触达。

## 5. 后续建议
1. **Copula 质量评估**：可追加回测脚本，对比遗传算法初始种群在命中率与覆盖度上的变化。
2. **图嵌入调参**：当前使用简化 Node2Vec，可探索加偏置参数 `p/q` 或更长随机游走，输出多份嵌入做 A/B。
3. **互信息阈值**：现阶段线性扣分，可进一步归一化或采用自适应权重，以防样本量突变导致惩罚过大。
4. **长周期算法**：报告中 Contextual Bandit / VAE 仍待回测体系支撑，建议先补齐回测流水线与收益记录。
5. **文档与监控**：建议在 `docs/ops.md` 扩写 Copula/图嵌入训练的监控指标（耗时、设备占用、缓存更新频率）。

> 本报告同步更新 `docs/decision_record.md`、`CHANGELOG.md` 与 `agent_report.md`，所有实验假设写入 `ASSUMPTIONS.md`。
