# KL8 快乐 8 预测系统使用指南（2025.10 更新）

## 最新亮点
- ✅ 支持 `--advanced_mode` 三档算法配置（0=原始、1=遗传+贝叶斯、2=全栈增强）。
- ✅ 新增特征增强引擎（`src/analysis/feature_enhancer.py`），`--feature_mode` 可在 `hybrid` / `momentum` / `cooccurrence` 之间切换。
- ✅ Plus 版本脚本使用线程池并行生成号码，数据下载仅执行一次，避免重复 IO。

## 快速起步

### 0. 环境准备
```bash
conda activate python311
make setup
```

### 1. 原始模式（Mode 0）
```bash
python src/analysis/kl8_analysis.py --cal_nums 5 --total_create 50 --limit_line 80 --advanced_mode 0
```

### 2. 中级模式（Mode 1）
```bash
python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 100 --limit_line 200 --advanced_mode 1
```

### 3. 全栈模式 + 特征增强（Mode 2）
```bash
python src/analysis/kl8_analysis.py \
  --cal_nums 10 \
  --total_create 200 \
  --limit_line 300 \
  --advanced_mode 2 \
  --feature_mode hybrid
```

### 4. 多线程批量生成
```bash
python src/analysis/kl8_analysis_plus.py \
  --cal_nums 15 \
  --total_create 1000 \
  --limit_line 250 \
  --max_workers 8 \
  --advanced_mode 1 \
  --feature_mode momentum
```

### 5. 批量收益分析
```bash
python src/analysis/kl8_cash_plus.py --path example_results --max_workers 4
```

### 6. 自动任务调度
```bash
python src/analysis/kl8_running.py \
  --cal_nums_list "5,7,10" \
  --total_create_list "100,300" \
  --nums_range "2023200,2023210" \
  --running_mode 0
```

## 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cal_nums` | `10` | 每组号码数量（推荐 5~20） |
| `--total_create` | `50` | 生成组合数量，可与 `--multiple` 配合 |
| `--limit_line` | `50` | 分析所用历史期数（越大越平滑） |
| `--advanced_mode` | `0` | 算法档位：0=原始、1=遗传+贝叶斯、2=全栈 |
| `--feature_mode` | `hybrid` | 特征增强模式：`hybrid` / `momentum` / `cooccurrence` |
| `--download` | `1` | 是否下载最新数据（0=使用本地） |
| `--simple_mode` | `0` | 控制输出粒度：0=详细，1=简化，2=仅盈利汇总 |
| `--max_workers` | `4` | Plus 版本线程数，建议等于 CPU 核心或稍多 |

### 特征增强模式选择指南

| 模式 | 特点 | 适用情况 |
|------|------|----------|
| `hybrid` | 同时考虑近期频率、动量与共现谱 | 默认推荐，综合能力最强 |
| `momentum` | 强调近期窗口与长窗口的频率差（趋势动量） | 近期出现明显热点或趋势时 |
| `cooccurrence` | 使用共现谱主特征向量衡量号码关联度 | 希望保持组合互补与协同时 |

### 新增主成分分析（PCA）特征
- 2025.10起，系统集成PCA主成分特征：自动提取历史号码矩阵的全局主成分，进一步提升号码表达能力。
- 默认在hybrid模式中融合PCA特征（权重可调），无需额外参数。
- 依赖scikit-learn，若未安装则自动跳过。

| 特征类型 | 说明 |
|----------|------|
| `pca`    | 主成分分析，提取全局分布主轴，提升多样性与全局性 |

**建议**：PCA特征适合大样本、全局趋势分析，能有效补充局部动量与共现谱的不足。

## 约束与指标（高级模式生效）
- 重复率 / 连号数 / 奇偶比 / 分组分布 均有自适应阈值控制。
- 遗传算法支持多目标适应度（热号覆盖、冷号多样性、趋势约束等）。
- 特征增强将评分结果注入最终候选筛选阶段，显著提高高级模式的命中率与稳定性。

## 故障排查
- **下载失败**：检查网络能否访问 `https://datachart.500.com` 与 `https://data.917500.cn`，必要时配置代理。
- **特征增强报错**：确认安装了 `numpy`、`scikit-learn`；若使用 `feature_mode` 时缺依赖，会自动回退并给出日志提示。
- **线程模式卡住**：适当降低 `--max_workers`，并确认本地磁盘是否存在锁文件/大量未清理结果。

## 建议
- 常规生产推荐 `Mode 1 + hybrid`，需要更强探索能力时使用 `Mode 2`。
- 多线程运行时，在同一目录下多次执行请搭配 `--path` 区分输出，避免文件覆盖。
- 若脚本仅用于回测，可设定 `--download 0` 并指向本地数据，减少请求频次。
