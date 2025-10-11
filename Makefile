# KL8彩票分析器 Makefile
# 符合AGENTS.md规范的一键任务管理

.PHONY: setup fmt lint test run build ci clean help
.DEFAULT_GOAL := help

# 环境和依赖管理
setup:  ## 安装依赖、初始化环境
	@echo "正在设置KL8分析器环境..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	@echo "创建必要的目录..."
	mkdir -p data results logs
	@echo "环境设置完成！"

# 代码格式化
fmt:  ## 代码格式化
	@echo "正在格式化代码..."
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black

# 静态检查
lint:  ## 静态检查
	@echo "正在进行静态代码检查..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/kl8_analyzer --ignore-missing-imports

# 运行测试
test:  ## 运行测试
	@echo "正在运行测试套件..."
	pytest tests/ -v --cov=src/kl8_analyzer --cov-report=term-missing --cov-report=html --cov-report=xml

# 启动应用或示例
run:  ## 启动应用或示例
	@echo "启动KL8分析器演示..."
	python -m kl8_analyzer.cli.main analysis --advanced_mode 0 --cal_nums 10 --window_size 6

# 构建产物
build:  ## 构建产物
	@echo "正在构建分发包..."
	python -m build
	@echo "构建完成，产物位于 dist/ 目录"

# 本地模拟CI
ci: fmt lint test build  ## 本地模拟 CI：lint + test + build
	@echo "✅ CI流程全部通过！"

# 清理临时文件
clean:  ## 清理构建和缓存文件
	@echo "正在清理临时文件..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 数据管理
download-data:  ## 下载历史数据
	@echo "正在下载KL8历史数据..."
	python -m kl8_analyzer.cli.main data --download

# 运行不同模式的分析
run-mode0:  ## 运行原始算法模式
	@echo "运行模式0（原始算法）..."
	python -m kl8_analyzer.cli.main analysis --advanced_mode 0 --cal_nums 20 --total_create 100

run-mode1:  ## 运行中级算法模式
	@echo "运行模式1（中级算法）..."
	python -m kl8_analyzer.cli.main analysis --advanced_mode 1 --cal_nums 20 --total_create 200

run-mode2:  ## 运行高级算法模式
	@echo "运行模式2（高级算法）..."
	python -m kl8_analyzer.cli.main analysis --advanced_mode 2 --cal_nums 20 --total_create 500

# 收益分析
cash-analysis:  ## 运行收益分析
	@echo "运行收益分析..."
	python -m kl8_analyzer.cli.main cash --start_date 2025-01-01 --end_date 2025-01-31

# 批量任务
batch-run:  ## 运行批量任务
	@echo "运行批量任务..."
	python -m kl8_analyzer.cli.main runner --mode batch --total_create 1000

# 开发环境
dev-install:  ## 安装开发依赖
	@echo "安装开发依赖..."
	pip install -r requirements-dev.txt
	pre-commit install

# 帮助信息
help:  ## 显示此帮助信息
	@echo "KL8彩票分析器 - 可用命令："
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "快速开始："
	@echo "  make setup     # 首次安装和环境设置"
	@echo "  make ci        # 运行完整的CI检查"
	@echo "  make run       # 运行基本示例"
	@echo ""
	@echo "算法模式对比："
	@echo "  make run-mode0 # 原始算法（快速）"
	@echo "  make run-mode1 # 中级算法（平衡）"
	@echo "  make run-mode2 # 高级算法（最佳效果）"