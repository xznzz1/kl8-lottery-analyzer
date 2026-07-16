# KL8 分析工具 Makefile

.PHONY: setup fmt lint test build run train-graph download-data scientific-backtest prospective-evaluate research-v2-backtest research-v2-changepoint-backtest ci clean help
.DEFAULT_GOAL := help

PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
LOTTERY_ROOT := $(abspath $(PROJECT_ROOT)/..)
PYTHON := $(PROJECT_ROOT)/.venv/Scripts/python.exe
QUALITY_PATHS := src/scientific src/research_v2 src/data_fetcher.py src/config.py src/analysis/shared_utils.py src/analysis/kl8_running.py scripts/get_data.py scripts/backtest_baselines.py scripts/prospective_evaluate.py scripts/research_v2_backtest.py scripts/research_v2_changepoint_backtest.py scripts/train_graph_embeddings.py tests/test_data_fetcher.py tests/test_scientific_backtest.py tests/test_prospective_evaluate.py tests/test_config.py tests/test_shared_utils.py tests/test_kl8_running.py tests/test_research_v2_bayesian.py tests/test_research_v2_metrics.py tests/test_research_v2_evaluation.py tests/test_research_v2_backtest.py tests/test_research_v2_changepoint.py tests/test_research_v2_changepoint_evaluation.py tests/test_research_v2_changepoint_backtest.py
export PYTHONPYCACHEPREFIX := $(PROJECT_ROOT)/data_cache/pycache
export COVERAGE_FILE := $(PROJECT_ROOT)/data_cache/.coverage
export XDG_CACHE_HOME := $(LOTTERY_ROOT)/.cache
export BLACK_CACHE_DIR := $(LOTTERY_ROOT)/.cache/black
export RUFF_CACHE_DIR := $(LOTTERY_ROOT)/.cache/ruff
export MYPY_CACHE_DIR := $(LOTTERY_ROOT)/.cache/mypy
export PIP_CACHE_DIR := $(LOTTERY_ROOT)/.cache/pip
export TMP := $(LOTTERY_ROOT)/.tmp
export TEMP := $(LOTTERY_ROOT)/.tmp

setup: ## 安装依赖并准备目录
	@echo "使用固定 Python 3.11 虚拟环境：$(PYTHON)"
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in [r'$(LOTTERY_ROOT)/.tmp', r'$(LOTTERY_ROOT)/.cache/pip', r'$(PROJECT_ROOT)/data_cache/kl8', r'$(PROJECT_ROOT)/results/logs', r'$(PROJECT_ROOT)/reports']]"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r "$(PROJECT_ROOT)/requirements.txt"
	$(PYTHON) -m pip install -r "$(PROJECT_ROOT)/requirements-dev.txt"
	@echo "依赖安装完成，目录已就绪"

fmt: ## 代码格式化
	@echo "执行 black + isort..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m black $(QUALITY_PATHS)
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m isort $(QUALITY_PATHS) --profile black

lint: ## 静态检查
	@echo "执行 Ruff、flake8 与 mypy..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m ruff check $(QUALITY_PATHS)
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m flake8 $(QUALITY_PATHS)
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m mypy --strict src/scientific src/data_fetcher.py src/config.py scripts/backtest_baselines.py scripts/prospective_evaluate.py --ignore-missing-imports --follow-imports=skip --disable-error-code=type-arg --disable-error-code=no-any-return
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m mypy --strict src/research_v2 scripts/research_v2_backtest.py scripts/research_v2_changepoint_backtest.py --ignore-missing-imports

test: ## 运行测试与覆盖率
	@echo "运行 pytest..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m pytest tests -v -p no:cacheprovider --basetemp "$(LOTTERY_ROOT)/.tmp/pytest" --cov=src --cov-fail-under=80 --cov-report=term-missing --cov-report=xml:results/coverage.xml

build: ## 生成可分发的 pyc 产物
	@echo "编译 Python 字节码..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m compileall src

run: ## 执行示例（需要已有数据文件）
	@echo "运行快乐8高频号码统计示例..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" examples/analysis_example.py

train-graph: ## 训练共现图嵌入缓存
	@echo "训练 Node2Vec 图嵌入..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" scripts/train_graph_embeddings.py --lottery kl8

download-data: ## 下载快乐8历史数据
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" scripts/get_data.py --name kl8

scientific-backtest: ## 固定4元预算的无泄漏科学回测（封顶奖金情景）
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" scripts/backtest_baselines.py --data data_cache/kl8/data.csv --floating-prize-mode cap-scenario

prospective-evaluate: ## 冻结参数后只评估2026186之后真实到达的数据
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" scripts/prospective_evaluate.py

research-v2-backtest: ## 运行v2探索性嵌套时间回测并生成报告
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" scripts/research_v2_backtest.py

research-v2-changepoint-backtest: ## 运行v2第二阶段在线变点探索性回测
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" scripts/research_v2_changepoint_backtest.py

ci: fmt lint test build ## 本地 CI
	@echo "本地 CI 全部通过"

clean: ## 清理临时文件和缓存
	@echo "清理缓存与编译文件..."
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -c "import shutil, pathlib; [shutil.rmtree(pathlib.Path(name), ignore_errors=True) for name in ['build', 'dist', '.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov']]"
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

help: ## 查看可用命令
	@echo "可用任务："
	@echo "  make setup          - 安装依赖并初始化目录"
	@echo "  make download-data  - 下载快乐8历史数据"
	@echo "  make scientific-backtest - 生成严格时间滚动评估与报告"
	@echo "  make prospective-evaluate - 运行冻结策略前瞻监测"
	@echo "  make research-v2-backtest - 运行v2探索性嵌套时间回测"
	@echo "  make research-v2-changepoint-backtest - 运行v2第二阶段在线变点回测"
	@echo "  make fmt            - 格式化代码"
	@echo "  make lint           - 静态检查"
	@echo "  make test           - 运行测试"
	@echo "  make build          - 生成 pyc 产物"
	@echo "  make run            - 运行示例分析"
	@echo "  make train-graph    - 训练共现图嵌入缓存"
	@echo "  make ci             - 本地 CI（fmt+lint+test+build）"
	@echo "  make clean          - 清理缓存文件"
