# 快乐 8 数据源实时核验

访问日期：2026-07-16（Europe/Budapest；下载元信息使用 UTC）

## 数据源

- 走势图入口：<https://datachart.500.com/kl8/>
- 实际数据端点：<https://datachart.500.com/kl8/zoushi/newinc/jbzs_redblue.php>
- 全量查询参数：`from=1&to=999999&shujcount=0&sort=0`

入口页 JavaScript 当前请求 `/kl8/zoushi/newinc/jbzs_redblue.php`。旧代码中的
`/kl8/history/newinc/jbzs_redblue.php` 在核验时返回 HTTP 404，因此抓取器已切换到
入口页实际使用的端点和参数名。

## 只读结构核验结果

对全量响应进行只读解析，未保存或提交完整 HTML：

| 项目 | 实测值 |
| --- | ---: |
| 响应字节数（抓取时） | 3,420,801 |
| `tbody#tdata` 原始 `tr` | 1,955 |
| 81 个 `td` 的有效开奖行 | 1,630 |
| 1 个 `td colspan=81` 的空白分隔行 | 325 |
| 最早期号 | 2021313 |
| 最新期号 | 2026186 |
| 重复期号 | 0 |

每个有效开奖行均满足：第 0 列为数字期号；第 1—80 列与号码 1—80 一一对应；
20 列使用 `chartBall01`，60 列使用 `yl01`；开奖号码文本与所在号码列一致；每期
20 个开奖号码互不重复且均在 1—80。325 个额外行全部为空白分隔行，不是开奖记录。

## 可复现保护

抓取器只接纳上述 81-cell 契约，并对原始、有效和拒绝行分别计数。CSV 保存前会检查
新数据最新期号不得早于现有文件；保存后在 `download_meta.json` 记录来源 URL、UTC
抓取时间、期号范围、记录数、拒绝原因及 CSV SHA-256，且只记录相对 CSV 文件名，
不持久化本机绝对路径。`data_cache/`、下载 HTML、CSV 和运行结果继续由 `.gitignore`
排除，不将本次全量响应作为 fixture 提交。
