# 本次自动执行报告 | Automation Execution Report

## 2026-07-16 快乐8真正前瞻验证

- 分支目标：在不重新划分、调参或覆盖旧final holdout的前提下，冻结截止2026186的全部科学参数，并只处理之后自然到达的开奖。
- 实现：新增`config/scientific_freeze.json`、`src/scientific/prospective.py`和`scripts/prospective_evaluate.py`；每个新增期生成500条原始记录、120行无机会性排名的描述摘要，并为下一期生成100行合法冻结策略票面。
- 数据：当前`data_cache/kl8/data.csv`共1631期，新增期2026187的预测历史严格截止2026186；冻结前缀1630期的原CSV SHA-256仍为`fffab75fffd17a9a84f728a3a52c27fc8bc598a5630ffc60a43510dc8a4becb6`。
- 幂等：同一主键`issue/strategy/seed/play/ticket_mode`重复运行不追加；已有记录会被全量重算逐字段校验，冲突、缺失网格或数据倒退在写入前失败。真实数据重复运行的`prospective_records.csv` SHA-256保持一致。
- 审计链：运行在任何开奖计算/写入前校验六个指定冻结源码文件的规范化SHA-256；2026188完整100行候选内嵌于`reports/prospective_manifests/2026188.json`，已有期号manifest禁止覆盖，候选CSV跨期覆盖前必须先验封。
- 保护：旧科学报告与本地旧`final_holdout_results.csv`运行前后SHA-256分别保持`5bc8abb7...eebe9`和`210e7a8f...879c`；前瞻CLI不能把输出指向旧报告或`results/scientific/`。
- 首次结果：2026187是首个冻结后样本外观察，但具体票面未事前公开封存；2026188是首个具有完整预先封存候选集的期号。当前只有1个冻结后观察期，20个seed不是20个独立期开奖，因此不运行显著性检验或排名；不能证明任何策略优于随机，也不能证明等效。
- 质量：Black、isort、Ruff、flake8、严格mypy、compileall和`pip check`通过；完整测试`133 passed`，当前约定覆盖范围为`85.24%`。
- 复现：`.venv\Scripts\python.exe scripts\prospective_evaluate.py --next-issue 2026188`，输出位于`results/prospective/`、`reports/kl8_prospective_report.md`和版本化逐期manifest。下一期号不作整数加一推断。完整质量结果在本次Draft PR说明中记录。

## 2026-07-16 快乐8科学评估审计

- 数据：实时复核500.com全量响应，1,955个原始行中1,630个有效期开奖行、325个空白分隔行，期号2021313—2026186；CSV和HTML未提交。
- 实现：新增`src/scientific/`和`scripts/backtest_baselines.py`，固定两注4元，覆盖十种玩法、两种出票方式、六类策略、20个随机seed、区间估计、风险和多重检验。
- 防泄漏：最终326期在首次评估前为未接触holdout，当前结果已查看并永久冻结；261期滚动验证选择预注册参数。后续不得用这326期选择策略或重新调参，新主张必须使用未来数据或新的未接触holdout。
- 奖金：按2025350切换规则版本，浮动奖采用显式情景；报告明确区分名义/情景奖金与实际兑付。
- 结果：100个主检验组合中，没有任何策略在Holm校正后显著优于随机基线。
- 复现：`.venv\Scripts\python.exe scripts/get_data.py --name kl8`，随后`.venv\Scripts\python.exe scripts/backtest_baselines.py --data data_cache/kl8/data.csv --floating-prize-mode cap-scenario`。
- 存储：本次Windows执行已验证实际根目录为`D:\lottery\kl8-lottery-analyzer`；实现约束是当前仓库内`data_cache/`、`results/`、`reports/`输出边界，不是硬编码盘符。自动执行未在C盘创建仓库副本、虚拟环境、缓存或结果。
- 审计：复核`scientific-model`近期提交及`src/data_fetcher.py`、`src/common.py`、`src/common_legacy.py`、`scripts/backtest_baselines.py`；未从旧提交拣选会退化严格解析或统计设计的实现。
- 质量：完整测试`108 passed`；当前轻量下载、配置、可复用分析组件与科学评估路径覆盖率`84.66%`。核心路径启用Ruff、Black、flake8与严格mypy；旧深度训练/预测入口不计入该覆盖率，排除清单在`.coveragerc`和`ASSUMPTIONS.md`显式记录。当前系统未安装`make`，已逐项执行`fmt`、`lint`、`test`、`build`等价命令。

## 2025-10-14 Copula 采样与图嵌入扩展
- 在高级候选生成中引入 Copula 多样性采样与互信息惩罚（`src/analysis/copula_sampler.py`、`src/analysis/mutual_information.py`），提升号码相关性建模。
- 新增图嵌入训练脚本 `scripts/train_graph_embeddings.py`（PyTorch Node2Vec），自动适配 CPU / GPU / AMD ROCm，并生成 `graph_embedding_scores`。
- `feature_enhancer` 追加图嵌入调试字段与缓存清理方法，综合得分权重大幅增强。
- 配置更新：`config.analysis.copula`、`config.analysis.graph_embedding`、Makefile `train-graph` 目标，以及完整的 `--copula_*` CLI 覆盖参数。

## 需求摘要 | Requirement Summary
- 实施文档《docs/kl8_algorithm_extension_report.md》中列出的短期与中期任务，落地 Dirichlet 平滑、频繁项集挖掘、Copula 采样、图嵌入与互信息惩罚。
- 维持 CLI 配置灵活，同时兼容 CPU/GPU 环境，并将实现细节同步到 README、使用指南与运维文档。

## 方案概览 | Solution Overview
- 架构模块：
  - Copula 模块拟合 80 维相关矩阵，提供候选组合与诊断信息，最终由高级模式整合。
  - 图嵌入训练脚本独立运行，分析阶段仅依赖 NumPy，避免在生产环境安装 PyTorch。
  - 互信息矩阵在高级评分阶段扣减高相关组合，与 Copula 采样共同提升多样性。
- 选型与权衡：
  - 采用自实现高斯 Copula + 特征值截断，避免额外依赖且易于调参。
  - Node2Vec（随机游走 + Skip-gram）在 CPU 环境即可训练，可选 GPU 加速。
  - 互信息默认保守权重，防止惩罚过大，可通过 YAML 统一调整。

## 实现与自测 | Implementation & Self-testing
- 代码实现：`src/analysis/copula_sampler.py`、`src/analysis/mutual_information.py`、`scripts/train_graph_embeddings.py`、`feature_enhancer` 图嵌入通道、`kl8_analysis.py` Copula/互信息集成、Makefile `train-graph`。
- 单元测试：`tests/test_copula_sampler.py`、`tests/test_mutual_information.py`、`tests/test_feature_enhancer.py`。
- 自测命令：
  ```bash
  pytest tests/test_copula_sampler.py tests/test_mutual_information.py tests/test_feature_enhancer.py -v
  python scripts/train_graph_embeddings.py --lottery kl8 --epochs 5 --device auto
  python src/analysis/kl8_analysis.py \
    --cal_nums 20 --total_create 240 --limit_line 240 \
    --advanced_mode 2 --feature_mode hybrid \
    --rule_filter soft --rule_support 0.08 --rule_confidence 0.7 \
    --copula_mode force --copula_samples 64 --copula_shrinkage 0.1
  ```

## 风险与后续改进 | Risks & Next Steps
- 样本量敏感：Copula 需足够历史期，建议监控 `effective_draws` 与数据完整性。
- 权重调节：互信息与图嵌入权重应结合回测持续调整，可研发自动化调参脚本。
- 图嵌入优化：可探索 Node2Vec 的 `p/q` 偏置、长游走及多嵌入投票。
- 运维支持：在 `docs/ops.md` 中补充训练耗时、设备占用与缓存更新频率的监控指引。
