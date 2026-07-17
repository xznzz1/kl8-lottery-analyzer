# 快乐8策略 v2：365 期未来前瞻封存与确认协议

## 技术摘要

本阶段冻结第一、第二阶段已经合并的五个概率模型，并建立“开奖前生成不可覆盖
manifest—提交并 push—开奖后只读评价”的未来前瞻链。确认窗口固定为 365 个
未来官方开奖期；在第 365 期完成前禁止提前停止、提前宣布成功、删除表现不佳的
模型或根据中途结果改变参数。

本协议不把已经查看过的历史回测重新包装为独立 holdout。历史结果仍只是
`exploratory development evidence`。新记录使用
`evidence_status=prospective_presealed_research`，但只有在每期 manifest 确实先于
官方结果提交并 push 的前提下，才能保留“预先封存”的证据含义。

## 冻结范围

机器可读契约位于
`config/research_v2_prospective_freeze.json`。第三阶段只编排现有 v2 公共接口，
不改写第一、第二阶段模型公式、参数选择或历史报告数值。

| 模型 | 冻结定义 | 目标期前信息 |
|---|---|---|
| `uniform_random` | 80 个号码概率均为 0.25 | 不使用历史 |
| `dynamic_bayesian` | 第一阶段 3×3 小网格；用目标期前内层平均 Brier 选择参数；同分按预注册网格顺序 | 至少 365 期初始历史，至少 120 个内层观察 |
| `fixed_normal_bayesian` | 全部历史，decay=0.995，prior_strength=80 | 全部目标期前历史 |
| `fixed_high_bayesian` | 最近 120 期，decay=0.95，prior_strength=80 | 最近 120 个目标期前开奖 |
| `changepoint_bayesian` | recent_window∈{30,60,120}，reference_window=240，目标期前变化分数的 90% 分位数为阈值；normal 复用 fixed_normal，high_change 复用 fixed_high | 窗口选择、分数、阈值和状态均只使用目标期前数据 |

五个模型必须同时保留。`changepoint_bayesian` 的新增机制只是依据目标期前变化
分数在两个固定后验间切换；它不能根据中途表现改阈值、换窗口网格或删除任一
消融基线。

## 因果时序与封存步骤

每个目标期严格按以下顺序执行：

1. 更新并校验截至最新已开奖期的本地数据。
2. 从官方来源明确确认下一目标期号；不得用最新期号整数加一代替确认。
3. 在目标期开奖前运行 `manifest` 操作。脚本只选取 `issue < target_issue` 的
   历史；目标期行和任何未来行即使意外存在，也不会进入预测。
4. 检查 manifest、提交到当前研究分支并 push。只有远程提交发生在官方结果之前，
   该期才可标记为有效预封存。
5. 官方结果发布后更新数据，再运行 `evaluate` 操作。评价直接读取 manifest 中
   已封存概率，不重新计算预测。
6. 每个目标期只允许生成一个 manifest 和一个评价 JSON；已存在即失败。

基础设施完成和代码审计期间不生成正式首期 manifest。首期目标号仍必须在审计
完成后通过官方来源重新确认。

## Manifest 契约

正式目录固定为 `reports/research_v2_prospective_manifests/`，文件名为
`<target_issue>.json`。独占创建模式拒绝覆盖、重写或删除已有文件。每份文件至少
记录：

- 目标期、UTC 生成时间、历史截止期和历史期数；
- 目标期前规范历史数据 SHA-256 与输入路径；
- 当前 Git 提交 SHA；
- 冻结源码清单、逐文件 SHA-256 和总指纹；
- 五个模型的冻结参数；
- 五组 80 维概率、完整 1—80 排名和 Top-1 至 Top-10 候选；
- dynamic 的当期目标前选参结果；
- changepoint 的窗口、变化分数、阈值、状态、实际 decay 和有效历史长度；
- 官方期号确认来源及时间；
- `evidence_status=prospective_presealed_research`。

概率必须严格位于 (0,1)，每个模型的 80 个概率之和须在数值误差内等于 20，
排名必须是 1—80 的完整排列。源码清单使用 UTF-8、统一 LF 后的 SHA-256；任一
文件漂移时，manifest、开奖后评价和正式汇总均应拒绝运行。

## 开奖后不可覆盖评价

评价目录固定为 `results/research_v2_prospective/`，采用“一期一个不可变 JSON”
而不是可被整体重写的累计文件。同一目标期第二次写入会失败。每条记录包含：

- 实际 20 个开奖号码；
- manifest 路径、原始字节 SHA-256、生成时间；
- 官方结果发布时间、评价时间和是否在官方结果前封存；
- 五个模型逐期 80 维 Brier score、Bernoulli log loss；
- 五个模型 Top-1 至 Top-10 命中、随机理论期望和 excess hits；
- 四个模型相对 uniform 的 Brier 与 log-loss 差值；
- changepoint 相对 fixed_normal、fixed_high 的对应差值；
- changepoint 当期状态。

评价时间不得早于官方结果发布时间，manifest 生成时间必须严格早于该发布时间。
这项时间校验不能代替 Git 远程提交时间审计；正式汇总前还应核对每个 manifest
对应提交确实已在开奖前推送。

## 365 期主要确认方案

独立统计单位是开奖期，不把一期内 80 个号码、多个玩法或随机 seed 当作独立
样本。主要指标是每期开奖期的 80 维均值 Brier score：

`mean_i((p_i - y_i)^2)`。

固定四项主要比较为：

1. `dynamic_bayesian - uniform_random`；
2. `fixed_normal_bayesian - uniform_random`；
3. `fixed_high_bayesian - uniform_random`；
4. `changepoint_bayesian - uniform_random`。

差值为负代表模型 Brier 较低。正式主要汇总只允许在恰好 365 个唯一且合格的
预封存目标期完成后运行。预注册检验是开奖期配对差值的单侧 t 检验，备择方向为
平均 `model - uniform < 0`；四个原始 p 值同时使用 Holm 校正。禁止在中途计算
结果后据此提前结束或修改模型。

该检验把开奖期作为观测单位，但普通配对 t 检验没有额外校正潜在时间相关性；
最终报告必须同时给出效应量、普通标准误、完整逐期结果和这一限制，不能只依据
校正 p 值作结论。

## 次要指标

次要指标包括 Bernoulli log loss、固定分箱 calibration 与 ECE、Top-1 至
Top-10 命中、changepoint 相对两个固定基线的差值，以及 high_change 触发比例。
这些指标用于描述概率质量、排序表现和切换机制，不能用于中途选择模型、改变
主要终点或扩大参数空间。

## 操作命令

以下命令只是操作模板；目标期号、官方 URL 和时间必须替换成当期已核验值：

```powershell
.\.venv\Scripts\python.exe scripts\research_v2_prospective.py manifest `
  --target-issue <官方确认期号> `
  --official-source-url <官方HTTPS来源> `
  --official-confirmed-at-utc <UTC时间>
```

开奖后：

```powershell
.\.venv\Scripts\python.exe scripts\research_v2_prospective.py evaluate `
  --target-issue <目标期号> `
  --official-result-published-at-utc <官方结果发布时间UTC>
```

仅在 365 期完成后：

```powershell
.\.venv\Scripts\python.exe scripts\research_v2_prospective.py summary
```

## 限制与解释边界

- 旧 holdout 已被查看；历史 v2 结果不能称为新的独立 holdout 或确认性证据。
- 预封存基础设施降低事后选择空间，但不能证明数据源、Git 远程时间或官方时间
  永远正确；这些仍需逐期审计。
- changepoint 变化分数只表示历史频率结构偏离，不证明开奖机制发生改变；公平
  随机波动本身会产生假变点。
- Brier、log loss、ECE 和 Top-k 在 365 期内都可能因随机波动出现暂时优势。
- 若彩票公平且独立，任何历史模型都不应存在稳定预测优势。即使正式比较有利，
  也必须结合效应大小、校准、完整记录和多重比较校正谨慎解释，不能声称保证中奖
  或建立稳定套利能力。

## 审计后下一步

代码审计应先核验冻结配置、源码哈希、路径边界、防覆盖行为、目标期前数据切片、
五模型概率与排名，以及 v1 和既有 v2 零变化。审计通过后，再更新正式数据、从
官方来源确认首期目标号、生成首份 manifest、人工复核提交内容并在开奖前 push。
