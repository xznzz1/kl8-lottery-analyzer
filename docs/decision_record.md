# 设计决策记录

## 2025-10-11 精简版改造

- **聚焦快乐 8 分析脚本**  
  删除 `src` 下与模型训练相关的 `analysis.py`、`bootstrap.py`、`modeling.py`、`pipeline.py`、`preprocessing.py` 等文件，仅保留 `src/analysis/*` 的原始脚本。

- **配置与公共接口瘦身**  
  `src/config.py` 与 `src/common.py` 仅暴露快乐 8 所需字段与方法；移除训练管线加载逻辑，避免无效依赖。

- **数据抓取模块改写**  
  新版 `src/data_fetcher.py` 只处理快乐 8 页面结构与顺序数据格式，同时加强域名白名单校验。

- **工具链与测试重建**  
  重新编写 `Makefile`、`requirements-dev.txt` 与 Pytest 用例，确保 `make ci` 可在无额外依赖下运行，覆盖率集中在保留下来的核心模块。

- **样例数据托管**  
  仓库内置 3 期样例数据 (`data/kl8/data.csv`)，用于离线演示与 CI；`.gitignore` 对该目录放行，避免未来被误删。
