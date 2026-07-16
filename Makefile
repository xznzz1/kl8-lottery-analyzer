# KL8 分析工具 Makefile

.PHONY: setup fmt lint test build run train-graph download-data scientific-backtest ci clean help
.DEFAULT_GOAL := help

PYTHON ?= python

setup: ## 安装依赖并准备目录
	@echo "请确认已激活 python311 (conda) 环境"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/kl8', 'results', 'logs']]"
	@echo "依赖安装完成，目录已就绪"

fmt: ## 代码格式化
	@echo "执行 black + isort..."
	$(PYTHON) -m black src tests
	$(PYTHON) -m isort src tests --profile black

lint: ## 静态检查
	@echo "执行 flake8 与 mypy..."
	$(PYTHON) -m flake8 src tests
	$(PYTHON) -m mypy src --ignore-missing-imports

test: ## 运行测试与覆盖率
	@echo "运行 pytest..."
	$(PYTHON) -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=xml

build: ## 生成可分发的 pyc 产物
	@echo "编译 Python 字节码..."
	$(PYTHON) -m compileall src

run: ## 执行示例（需要已有数据文件）
	@echo "运行快乐8高频号码统计示例..."
	$(PYTHON) examples/analysis_example.py

train-graph: ## 训练共现图嵌入缓存
	@echo "训练 Node2Vec 图嵌入..."
	$(PYTHON) scripts/train_graph_embeddings.py --lottery kl8

download-data: ## 下载快乐8历史数据
	$(PYTHON) scripts/get_data.py --name kl8

scientific-backtest: ## 固定4元预算的无泄漏科学回测（封顶奖金情景）
	$(PYTHON) scripts/backtest_baselines.py --data data/kl8/data.csv --floating-prize-mode cap-scenario

ci: fmt lint test build ## 本地 CI
	@echo "本地 CI 全部通过"

clean: ## 清理临时文件和缓存
	@echo "清理缓存与编译文件..."
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(pathlib.Path(name), ignore_errors=True) for name in ['build', 'dist', '.pytest_cache', 'htmlcov']]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

help: ## 查看可用命令
	@echo "可用任务："
	@echo "  make setup          - 安装依赖并初始化目录"
	@echo "  make download-data  - 下载快乐8历史数据"
	@echo "  make scientific-backtest - 生成严格时间滚动评估与报告"
	@echo "  make fmt            - 格式化代码"
	@echo "  make lint           - 静态检查"
	@echo "  make test           - 运行测试"
	@echo "  make build          - 生成 pyc 产物"
	@echo "  make run            - 运行示例分析"
	@echo "  make train-graph    - 训练共现图嵌入缓存"
	@echo "  make ci             - 本地 CI（fmt+lint+test+build）"
	@echo "  make clean          - 清理缓存文件"
