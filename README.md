# KL8 (快乐 8) 数据分析工具集

本仓库提供一套围绕 `src/analysis` 目录脚本运行的最小化环境，支持下载、加载并分析快乐 8 历史数据。我们移除了与模型训练无关的模块，聚焦于概率统计、组合过滤与收益回测等离线分析场景。

> 开发/运行时请确保已激活名为 `python311` 的 Conda 环境，或使用等效的 Python 3.11 虚拟环境。

## 功能清单
- 🔄 `scripts/get_data.py`：下载快乐 8 历史数据，支持顺序出球模式。
- 📦 `src/common.py`：对外暴露数据下载、期号查询与数据加载的统一入口。
- 🌐 `src/data_fetcher.py`：封装 500.com / 917500 的抓取逻辑与解析。
- 📊 `src/analysis/`：包含多版本分析/回测脚本（原始版、Plus 版、收益版等），**支持优化的多线程处理**。
- 🧪 `tests/`：针对配置、公共接口与抓取模块的回归测试，默认覆盖率 ≥80%。
- ⚡ **多线程优化**：Plus 版本脚本已优化，数据下载与多线程处理完全分离，提升性能和稳定性。

## 快速开始（≤3 步）
```bash
conda activate python311
make setup
make run
```

若需要更新数据，可执行 `make download-data` 或直接调用 `scripts/get_data.py`。

## 目录结构
```
.
├── config/                # 配置文件（config.yaml）
├── data/kl8/              # 随仓库提供的最小数据示例
├── docs/                  # 架构/API/运维与历史文档
├── examples/              # 快速统计示例
├── scripts/               # 数据拉取脚本
├── src/
│   ├── analysis/          # 各类分析脚本（保持原始接口）
│   ├── common.py          # 公共调用入口
│   ├── config.py          # 精简配置模块（仅支持 kl8）
│   └── data_fetcher.py    # 历史数据抓取实现
├── tests/                 # Pytest 测试套件
└── Makefile               # 一键任务入口
```

## 配置说明
- `config/config.yaml` 仅保留数据路径和网络请求相关配置，运行时会自动创建 `data/kl8`、`results`、`logs`。
- `src/config.py` 固定快乐 8 为唯一支持的彩票类型，同时暴露 `PATHS`、`NETWORK_CONFIG` 等运行所需常量。
- 样例数据已内置于 `data/kl8/data.csv`，用于离线演示和测试；若使用最新数据，请执行 `make download-data`。

## 常见问题（FAQ）
1. **脚本提示找不到数据文件？**  
   请确保执行过 `make setup`（会创建目录）以及 `make download-data`（下载最新 csv），或使用仓库内置的样例数据。
2. **分析脚本需要的依赖如何安装？**  
   所有运行时依赖都已列在 `requirements.txt`，开发工具位于 `requirements-dev.txt`。`make setup` 会自动安装。
3. **是否仍然支持其他彩票玩法或模型训练？**  
   当前仓库仅聚焦快乐 8，为保证维护成本，已移除 pipeline、建模及深度学习相关模块。
4. **如何批量分析历史结果？**  
   使用 `src/analysis/kl8_running.py` 提供的批处理能力，或参考 `examples/analysis_example.py` 自行编写脚本。
5. **Plus 版本脚本的多线程优化有什么特点？**  
   Plus 版本（`kl8_analysis_plus.py`、`kl8_cash_plus.py`）已优化多线程架构：数据下载仅在主进程执行一次，然后使用线程池进行并发数据处理，避免重复下载和全局变量冲突问题。
6. **如何选择单线程版本还是 Plus 版本？**  
   - 单线程版本（`kl8_analysis.py`、`kl8_cash.py`）：适合小规模测试和学习验证
   - Plus 版本（`kl8_analysis_plus.py`、`kl8_cash_plus.py`）：适合大批量生产环境，性能更高

## 测试与质量
- 使用 `make ci` 依次执行格式化、静态检查、单元测试与构建（编译字节码）。
- 覆盖率通过 `pytest --cov=src` 收集，核心模块（config/common/data_fetcher）均有回归测试。
- `clean` 目标会移除 `__pycache__`、`.pytest_cache` 等临时文件。

## 合作建议
- 新增接口时，请同步更新 `docs/api.md` 与相应测试。
- 若扩展到其他彩票类型，请在 `docs/decision_record.md` 中记录取舍，并在 `ASSUMPTIONS.md` 明确假设。
- 如需恢复模型训练能力，可在 `feature/modeling` 分支重建 pipeline，再合并到主线。
