# 设计决策记录（Decision Record）

## 2026-07-16 冻结策略前瞻监测

- 将原科学评估永久冻结在期号2026186；旧科学报告、旧`final_holdout_results.csv`和参数文件不再重算或覆盖。`config/scientific_freeze.json`记录原1630期CSV快照、产物哈希、窗口120、衰减0.99、hybrid权重、20个seed及奖金情景。
- `repository_advanced`原先还有独立的窗口40/160、衰减0.97、特征权重、Dirichlet、PCA和图缓存开关。为避免后续静默漂移，在不改变算法输出的前提下将这些隐含常量显式化，并将科学计算库版本纳入冻结配置的运行时校验与依赖锁定。
- `scripts/prospective_evaluate.py`不导入或调用时间切分、验证调参或旧回测入口。对每个目标索引`t`，只把截至`t`的截断数组交给评估器，策略仍只能读取`t`之前的历史；2026187的策略输入严格截止2026186。
- 每个新增期固定生成500条原始记录：五个确定性策略共100条，20-seed均匀随机基线共400条。幂等主键为`issue/strategy/seed/play/ticket_mode`；每次重算全部已见前瞻期并逐字段核对，冲突或数据倒退在任何写入前失败，只追加完整新期网格。
- 前瞻摘要只保留100个确定性配置和20个随机seed ensemble配置。20个seed不是独立开奖；当前仅有一个新增期，因此不运行显著性检验、多重校正或排名，也不展示退化的单期置信区间。现阶段既不能证明策略优于随机，也不能证明等效。
- 下一期候选覆盖五个冻结确定性策略、选一至选十及`disjoint`/`independent`，共100行。`independent`仅表示允许重叠，不表示统计独立；票面被标记为冻结策略输出而非预测或投注建议。
- CLI路径锁定为仓库内`data_cache/kl8/data.csv`、`results/prospective/`与`reports/kl8_prospective_report.md`，并显式保护旧科学报告和`results/scientific/`。真实Windows运行还校验仓库根与`.venv`位于指定D盘位置，原子临时文件只写`D:\lottery\.tmp`。

## 2026-07-16 严格时间滚动科学评估

- 以行索引表示时间，CSV先按期号升序；预测第`t`期只允许访问`draws[:t]`，不再用期号数值差计算偏移。
- 最后20%固定为final holdout；此前20%的滚动验证区间只搜索预注册窗口、衰减和hybrid权重小网格。当前326期holdout的结果已经查看并永久冻结，后续不得据此选择新策略或重新调参；新的预测主张必须使用未来数据或另一个预先划定、真正未接触的holdout。
- 随机基线使用20个固定seed并在同期开奖上配对；主端点为单注平均命中数，使用单侧符号翻转检验和跨5策略×10玩法×2出票方式的Holm校正。符号翻转依赖配对差值在零假设下符号可交换（通常要求近似对称），故报告将其明确标为有假设的近似检验，并另报配对效应bootstrap区间。
- 随机ensemble的事件概率先在每个seed×期判定再平均；最大回撤、最长亏损与期末收益先按每条seed路径计算再平均，禁止对seed平均奖金路径做非线性判定。
- 单策略/单seed的二元事件使用整数成功数Wilson 95%区间；随机ensemble先聚合同一期跨seed事件率，再以期为cluster做百分位bootstrap，禁止把小数成功数代入Wilson。ensemble全零区间退化为`[0, 0]`仅代表样本期簇无观测事件，不证明真实概率为零，必须同时展示精确组合概率或更多未来数据。
- 原仓库深度特征在倒序数据上存在未来索引和全样本预处理，图嵌入/规则缓存缺训练截止点；科学路径排除这些功能，只保留每期用既往数据重算且可关闭外部缓存的高级特征。
- 玩法结构排名使用组合数学精确概率，避免用一次随机抽样决定选一至选十排序；回测仍保留多seed经验结果和风险路径。
- 奖金按第2025350期切换版本。浮动奖必须由逐期输入或显式情景提供；封顶情景只用于敏感性，不宣称是历史实付ROI。

## 2025-10-13 Copula 采样、图嵌入与互信息惩罚
- 新增 `src/analysis/copula_sampler.py`，在高级策略中默认随 `advanced_mode>=2` 启用；CLI `--copula_*` 参数支持覆盖样本量、收缩强度、候选倍率与随机种子。
- 训练脚本 `scripts/train_graph_embeddings.py` 基于 PyTorch Skip-gram，支持 `--device auto` 自动选择 CPU/GPU/AMD ROCm，结果写入 `analysis.graph_embedding.cache_file`。
- `feature_enhancer` 引入 `graph_embedding_scores` 与缓存清理函数，综合得分增加图嵌入权重，并在测试中覆盖空/有缓存场景。
- `src/analysis/mutual_information.py` 计算 80×80 互信息矩阵，在 `advanced_number_generation` 中作为多样性扣分，避免高度相关号码集中。
- `Makefile` 新增 `make train-graph`，README/运行指南同步补充；`config/config.yaml` 扩充 `analysis.copula`、`analysis.graph_embedding` 配置项。

## 2025-10-13 Dirichlet 平滑与关联规则
- 在 `src/analysis/feature_enhancer.py` 中引入 Dirichlet-Multinomial 后验得分，并通过 `config.analysis.dirichlet` 控制先验与方差惩罚。
- 新增 `src/analysis/rule_miner.py`，基于 FP-Growth 提供 `--rule_filter` 软/硬模式并缓存规则。
- `kl8_analysis*.py` 接入规则筛选并在 README/文档中暴露 CLI 参数与配置示例。
- `tests/test_rule_miner.py` 补充硬/软模式单元测试，验证违规组合与惩罚权重。

## 2025-10-12 特征增强引擎升级
- 新增 `src/analysis/feature_enhancer.py`，实现“近期动量 + 共现谱”的混合评分，支持 `--feature_mode`。
- `kl8_analysis.py` 与 `kl8_analysis_plus.py` 在高级模式下引入特征得分，于候选筛选阶段叠加权重。
- 提供 `FeatureDebugInfo` 以输出细节，便于调试与文档示例。
- 规划单元测试 `tests/test_feature_enhancer.py` 覆盖动量、共现与空数据边界（若缺失将补充）。

---

## 2025-12-19 多线程执行架构优化（Plus 版本）
### 背景与问题
历史实现中使用多进程并行：
- 存在重复数据下载，浪费网络与 IO 资源。
- 进程间共享状态困难，导致全局变量易被误用。
- 进程模型带来更高的内存开销与启动成本。
- 错误传播与统一处理复杂，定位困难。

### 决策
在 I/O 为主的批量任务场景，采用 `concurrent.futures.ThreadPoolExecutor` 替代 `multiprocessing.Process`。

### 选型对比（要点）
- 共享内存：线程天然共享，进程需要序列化与跨进程通信。
- 启动成本：线程低，进程高。
- GIL 影响：CPU 密集型线程受限，但 I/O 密集型收益明显。
- 错误处理：线程池有统一 Future/异常收集；进程需额外管道与协议。

### 架构设计要点
1. 数据下载去重：统一入口 `download_data_if_needed()`，避免并发重复拉取。
2. 线程安全：必要处使用锁保护共享资源，约束副作用边界。
3. 任务编排：主流程负责构建任务，工作线程仅处理纯函数/无副作用逻辑。
4. 错误治理：使用 `as_completed` 聚合结果，失败与重试策略解耦。

### 实现节点
- 线程池配置：`max_workers` 可配置，结合任务与机器核数动态调整。
- 进度与日志：实时输出成功/失败计数与摘要日志。
- 序列化优化：跨线程传递轻量数据结构，避免不必要的大对象复制。

### 效果（定性）
- 明显减少重复下载与请求抖动。
- 内存占用更稳定，整体吞吐提升。
- 故障可观测性与定位效率提升。

---

## 2025-10-11 架构精简与模块重组
- 精简历史脚本与模型代码，保留 `src/analysis/*` 的核心实现，移除未使用与耦合度高的遗留模块。
- 配置与公共方法下沉至统一位置，减少隐式导入与副作用，改进启动与调试体验。
- 数据抓取模块独立，统一格式校验与缓存策略，提升可维护性与可测试性。
- 工具链与测试重建：补充开发依赖与 Pytest 基础用例，使本地 `make ci` 可重复通过。
- 引入示例/样例数据用于本地演示与 CI，但不提交真实数据集。
