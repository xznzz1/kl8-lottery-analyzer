# 快乐8策略 v2：365 期未来前瞻封存与确认协议

## 当前状态

本阶段冻结第一、第二阶段已合并的五个概率模型，并建立“本地生成 manifest—通过
PR 合并远程封存—开奖后验证与评价”的未来前瞻链。确认窗口固定为 365 个未来
官方开奖期。旧 holdout 已被查看，所有历史结果仍只是
`exploratory development evidence`，不能重新称为独立 holdout 或确认性证据。

配置当前故意保持
`research_model_and_protocol_frozen_pending_code_audit`。在 pending 状态下，生产
CLI 的 `manifest`、`evaluate` 和 `summary` 三个操作都会 fail closed。本轮代码
审计不得把状态改为 active，不创建 `kl8-v2-prospective-365-v1` 标签，也不生成
正式首期 manifest 或评价。

未来只有同时满足以下条件才能运行生产操作：配置状态为 `active`；冻结标签真实
存在；标签提交是当前 HEAD 的祖先；完整配置指纹、source manifest 逐文件哈希和
总指纹均无漂移；冻结源码与配置没有未提交修改。

## 五个冻结模型

| 模型 | 冻结定义 | 目标期前信息 |
|---|---|---|
| `uniform_random` | 80 个号码概率均为 0.25；20 个预注册 seed 产生随机排名 | 不使用开奖历史 |
| `dynamic_bayesian` | 第一阶段 3×3 小网格，按目标期前内层平均 Brier 选择参数 | 至少 365 期初始历史和 120 个内层观察 |
| `fixed_normal_bayesian` | 全部历史，decay=0.995，prior_strength=80 | 全部目标期前历史 |
| `fixed_high_bayesian` | 最近 120 期，decay=0.95，prior_strength=80 | 最近 120 个目标期前开奖 |
| `changepoint_bayesian` | recent_window∈{30,60,120}，reference_window=240，目标期前变化分数 90% 分位数为阈值；normal 复用 fixed_normal，high_change 复用 fixed_high | 只用目标期前数据形成窗口、阈值和状态 |

五个模型必须全部保留，不能根据中途结果删除表现不佳的模型。changepoint 的唯一
新增机制是依据目标期前变化分数在两个固定模型之间切换；不能改阈值、扩大窗口
网格或另选参数。

## 未开奖目标期门槛

正式 target issue 必须从官方 HTTPS 来源明确确认，不能根据最新期号整数加一
推断。生产 manifest 在预测前强制验证：

- target issue 不得已经存在于输入数据；
- target issue 必须严格晚于输入数据中的最新已开奖期；
- 输入数据不得包含任何晚于 target issue 的记录；
- `history_through_issue` 必须等于通过完整校验后的输入数据最新期号；
- 目标期行或未来行不会被静默裁掉，而是直接拒绝运行。

CLI 不接受人工 `--generated-at-utc` 或 `--git-commit-sha`。本地生成时间由真实
UTC 时钟读取，提交 SHA 由 `git rev-parse HEAD` 读取。manifest 生成阶段只记录
`local_manifest_generated_at_utc` 和 `remote_preseal_verified=false`，不直接声称
已经远程封存。

## 365 期不可选择顺序链

每份 manifest 自动记录：

- `confirmation_index`，由已有链自动形成且必须完整为 1 至 365；
- 固定 `freeze_id` 和首期 `protocol_start_target_issue`；
- `previous_target_issue`；
- 前一份 manifest 原始字节的 `previous_manifest_sha256`，首期两字段均为 null。

第 2 期以后，上一份 manifest 和对应的、已经通过远程封存验证的 evaluation 必须
都存在。上一 evaluation 记录的 manifest SHA 必须与真实文件一致。缺少评价、额外
评价、索引断裂、目标期不递增、上一期号不匹配或哈希链断裂都会阻止继续。链满
365 期后拒绝生成额外记录；已有 manifest 和 evaluation 均采用独占创建，拒绝
覆盖和重写。

正式 summary 再次要求恰好 365 份 manifest 和 365 份 evaluation，并验证索引
1—365 完整、哈希链完整、target issue 一一对应、每条 evaluation 的本地及合并
提交 manifest SHA 均与真实文件相同，且不存在缺失、重复或额外 JSON。

## GitHub 远程封存验证

开奖后的 `evaluate` 必须提供 seal PR 号、官方结果来源 URL 和官方结果发布时间。
实现通过 GitHub API fail closed 验证：

- PR 属于 `xznzz1/kl8-lottery-analyzer`；
- PR 状态为 merged，base 为 `scientific-model`；
- 合并时间严格早于官方结果发布时间；
- 合并提交包含该目标期 manifest；
- 合并提交中的文件原始字节 SHA-256 等于本地 manifest 原始字节 SHA-256。

网络不可用、PR 未合并、合并过晚、base 错误、提交缺文件或哈希不符时，不会构造
或写入 evaluation。只有验证成功才记录：

- `remote_preseal_verified=true` 和 `sealed_before_official_result=true`；
- seal PR 编号与 URL；
- 合并提交 SHA 和合并 UTC 时间；
- `manifest_sha256_at_merge`。

单元测试使用可替换的 mock 客户端，不发出真实网络请求。

## Uniform 随机排序

uniform 概率始终为每个号码 0.25；1—80 升序不再称为随机排名。20 个预注册基础
seed 固定为 202601 至 202620。每个目标期与每个基础 seed 使用：

`kl8-v2-prospective-365-v1|<target_issue>|<base_seed>`

对该 ASCII 字节串计算 SHA-256，取前 8 字节按大端解释为无符号整数，传给
`numpy.random.default_rng`，再生成 1—80 的完整随机排列。manifest 保存 20 组
seed、派生哈希与整数、完整排名和 Top-1 至 Top-10。

开奖后 uniform Top-k 先在同期开奖期内对 20 个 seed 求平均，再在 365 个开奖期
之间求均值。20 个 seed 不会被当作 20 个独立开奖样本。

## 评价与描述性次要汇总

每期 evaluation 直接从已经远程验证的 manifest 概率和排名计算，不能重新预测。
记录五个模型的 80 维 Brier、Bernoulli log loss、Top-1 至 Top-10、四模型相对
uniform 的差值，以及 changepoint 相对两个固定基线的差值。

365 期 summary 会从每份 manifest 和实际 20 个号码独立复算，并要求结果与
evaluation 逐项一致。次要汇总固定包括：

- 五个模型的平均 Brier 和平均 Bernoulli log loss；
- 每个模型固定 10 个 [0,1] 等宽概率箱的 calibration 与 ECE；
- 五个模型 Top-1 至 Top-10 平均命中；
- uniform 的“期内 20-seed 均值，再跨期均值”；
- changepoint 相对 fixed_normal 和 fixed_high 的 Brier、log-loss 平均差值；
- high_change 触发次数和比例；
- 365 条目标期索引、confirmation index 与真实 manifest SHA-256。

这些次要指标明确标记为描述性，不用于提前选模、改模或改变主要终点。

## 主要检验：固定循环移动分块 bootstrap

独立统计单位是开奖期。主要指标是每期 80 维 Brier score，固定比较族为：

1. `dynamic_bayesian - uniform_random`；
2. `fixed_normal_bayesian - uniform_random`；
3. `fixed_high_bayesian - uniform_random`；
4. `changepoint_bayesian - uniform_random`。

负差值表示模型 Brier 较低。每项比较固定使用：

- `block_length=30`；
- `resample_count=20000`；
- `seed=20260717`；
- 循环连续块重采样至 365 期；
- 单侧备择 `mean(model minus uniform Brier) < 0`。

检验先把 365 个逐期差值中心化为均值 0，再从中心化序列产生 null bootstrap
均值。原始单侧 p 值固定为：

`(1 + count(null_bootstrap_mean <= observed_mean)) / (20000 + 1)`。

四个原始 p 值一次性使用 Holm 校正。每项同时报告平均 Brier 差、普通标准误、
未中心化循环分块重采样均值的 2.5%—97.5% 描述区间、原始 p 值、Holm 校正 p
值，以及固定参数。不得按结果改变分块长度、seed、重采样数或另选检验。

该推断依赖 365 期差值的时间平稳性和预注册 30 期块长能够合理保留相关结构的
假设；区间和 p 值都不能消除这些假设风险，也不能支持中途成功声明。

## 操作顺序

基础设施通过审计、配置未来改为 active 且冻结标签按协议创建后，每个目标期才
可依次执行：

1. 更新本地数据并从官方来源明确确认未开奖 target issue；
2. 运行 `manifest`，人工复核后提交并通过 PR 合并到 `scientific-model`；
3. 官方结果发布后更新数据；
4. 运行 `evaluate --seal-pr-number ... --official-result-source-url ...`；
5. 上一期 evaluation 完成后才可生成下一份 manifest；
6. 只在完整 365 期结束后运行 `summary`。

本轮 pending 状态下，以下操作模板均应被拒绝，不能用来生成正式产物。

## 解释边界

- 禁止提前停止、提前宣布成功或根据中途结果修改模型。
- 历史 v2 回测不能证明真实预测能力提高。
- changepoint 分数只表示历史频率结构偏离；彩票随机波动本身会产生假变点。
- GitHub 合并审计证明特定字节在特定合并提交与时间存在，不保证官方来源本身永远
  正确，因此每期仍需保存和复核官方结果来源。
- 若彩票公平且独立，任何历史模型都不应存在稳定预测优势。即使 365 期比较有利，
  也必须结合效应大小、完整记录、多重比较校正及时间依赖假设谨慎解释，不能声称
  保证中奖或稳定套利能力。
