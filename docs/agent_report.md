# 本次自动执行报告 | Automation Execution Report

## 2025-10-14 Copula 采样与图嵌入扩展
- 在高级候选生成中引入 Copula 多样性采样（`src/analysis/copula_sampler.py`），并附加互信息惩罚，强化号码相关性建模。
- 新增图嵌入训练脚本 `scripts/train_graph_embeddings.py`（PyTorch Node2Vec），支持自动检测 CPU/GPU/AMD ROCm，结果由 `feature_enhancer` 直接加载。
- `feature_enhancer` 增加 `graph_embedding_scores` 字段与缓存清理方法，综合得分权重可通过 `analysis.graph_embedding.weight` 调整。
- 重大配置扩充：`config.analysis.copula`、`config.analysis.graph_embedding`、Makefile `train-graph` 目标，以及 `--copula_*` CLI 覆盖参数。

## 需求摘记 | Requirement Summary
- 落实《docs/kl8_algorithm_extension_report.md》中的短期与中期任务：
  - 完成 Dirichlet 平滑、频繁项集挖掘的验证与文档记录。
  - 实装 Copula 蒙特卡洛采样与图嵌入特征通道。
  - 扩展互信息惩罚机制，提升候选组合多样性。
- 保持 CLI 可配置、兼容 CPU/GPU 环境，并在文档中记录实现过程。

## 方案概览 | Solution Overview
- 架构与模块：
  - `copula_sampler` 负责相关矩阵拟合和采样，`advanced_number_generation` 从配置读取并整合候选。
  - 图嵌入训练脚本独立运行，分析阶段仅依赖 NumPy，避免在生产环境安装 PyTorch。
  - 互信息模块将历史开奖转换为指示矩阵后计算 80×80 MI 矩阵，为高级模式提供多样性扣分依据。
- 选型与权衡：
  - **Copula**：自实现高斯 Copula + 特征值截断，避免额外依赖且便于调参。
  - **图嵌入**：简化 Node2Vec（随机游走 + Skip-gram）可在 CPU 环境训练，脚本支持 `--device` 手动覆盖。
  - **互信息**：基于矩阵乘法快速求解并引入平滑，默认权重保守，防止惩罚过大。

## 实现与自测 | Implementation & Self-testing
- **代码实现**：
  - `src/analysis/copula_sampler.py`（CopulaSampler + Diagnostics）
  - `src/analysis/mutual_information.py`（互信息矩阵计算）
  - `scripts/train_graph_embeddings.py`（Skip-gram 训练）
  - `feature_enhancer` 图嵌入通道、`kl8_analysis.py` Copula/互信息集成、Makefile 新增 `train-graph`
- **单元测试**：
  - `tests/test_copula_sampler.py`：生成组合长度、去重、种子稳定性、异常分支。
  - `tests/test_mutual_information.py`：矩阵对称性、对角线归零、空输入回退。
  - `tests/test_feature_enhancer.py`：缓存失效与有效嵌入场景、`FeatureDebugInfo` 新字段。
- **自测命令**：
  - `pytest tests/test_copula_sampler.py tests/test_mutual_information.py tests/test_feature_enhancer.py -v`
  - `python scripts/train_graph_embeddings.py --lottery kl8 --epochs 5 --device auto`
  - `python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 120 --limit_line 200 --advanced_mode 2 --copula_mode auto`

## 风险与后续改进 | Risks & Next Steps
- **样本量敏感**：Copula 需要足够历史期数，建议增加数据源监控与预警。
- **权重调节**：互信息惩罚与图嵌入权重仍需长期回测验证，建议加入自动化调参脚本。
- **性能扩展**：Node2Vec 训练仍为单机脚本，可探索引入更丰富的 walk 参数或批量游走。
- **运营支持**：后续可在 `docs/ops.md` 补充训练脚本运行指标、缓存更新频率与告警阈值。
