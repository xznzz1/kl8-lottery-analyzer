# Changelog
All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-11
### Added
- 精简后的 `src/common.py`、`src/config.py`、`src/data_fetcher.py` 以及对应单元测试。
- Makefile、示例数据 (`data/kl8/data.csv`) 与全新的 README/文档体系。
- `ASSUMPTIONS.md`、`docs/decision_record.md`、`docs/architecture.md`、`docs/api.md`、`docs/ops.md`、`agent_report.md`。

### Changed
- 只保留快乐 8 玩法，所有公共接口默认针对 `kl8`。
- 重新配置测试流程，使 `pytest --cov=src` 应用于核心模块。
- `scripts/get_data.py` 和 `examples/analysis_example.py` 与新结构对齐。

### Removed
- pipeline、modeling、preprocessing 等与模型训练相关的代码及测试。
