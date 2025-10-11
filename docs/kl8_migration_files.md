# KL8项目迁移文件清单

## 🎯 必须迁移的核心文件

### 1. 核心算法模块 (5个文件)
```
源路径: src/analysis/kl8_analysis.py
目标: src/kl8_analyzer/core/analysis.py
描述: 主分析引擎，包含2025年8种高级算法
依赖: pandas, numpy, scikit-learn, scipy, matplotlib
状态: 需要重构导入路径

源路径: src/analysis/kl8_analysis_plus.py
目标: src/kl8_analyzer/core/analysis_plus.py
描述: 多进程并行版本分析引擎
依赖: multiprocessing, pandas, numpy
状态: 需要重构导入路径

源路径: src/analysis/kl8_cash.py
目标: src/kl8_analyzer/core/cash_analysis.py
描述: 单文件收益分析工具
依赖: pandas, itertools
状态: 需要重构导入路径

源路径: src/analysis/kl8_cash_plus.py
目标: src/kl8_analyzer/core/cash_plus.py
描述: 批量收益分析工具
依赖: multiprocessing, pandas
状态: 需要重构导入路径

源路径: src/analysis/kl8_running.py
目标: src/kl8_analyzer/core/runner.py
描述: 任务调度和批量执行管理
依赖: subprocess, threading, argparse
状态: 需要重构导入路径
```

### 2. 支持工具模块 (需要提取和重构)
```
源路径: src/common.py
目标: src/kl8_analyzer/utils/common.py
描述: 提取KL8相关的通用函数
提取内容:
  - get_data_run() # KL8数据获取
  - get_current_number() # 当前期号获取
状态: 需要提取KL8相关部分

源路径: src/config.py
目标: src/kl8_analyzer/utils/config.py
描述: 提取KL8配置部分
提取内容:
  - LOTTERY_CONFIGS["kl8"] # KL8配置
  - PATHS相关配置
  - 工具函数
状态: 需要重构为独立配置系统

源路径: src/data_fetcher.py
目标: src/kl8_analyzer/utils/data_fetcher.py
描述: 数据获取模块（如果存在）
状态: 需要检查并适配
```

## 📚 文档文件 (直接迁移)

### 技术文档 (4个文件)
```
源路径: docs/kl8_algorithm_theory.md
目标: docs/algorithm_theory.md
描述: 完整的数学算法理论文档，24个算法章节
大小: ~50KB
状态: 直接迁移

源路径: docs/kl8_usage_guide.md
目标: docs/user_guide.md
描述: 详细使用指南，包含三种算法模式说明
大小: ~30KB
状态: 直接迁移

源路径: docs/kl8_optimization_report.md
目标: docs/performance_analysis.md
描述: 算法优化报告和性能分析
大小: ~25KB
状态: 直接迁移

源路径: README_KL8.md
目标: README.md
描述: 项目主文档
大小: ~15KB
状态: 适配为新项目主页
```

### 开发文档 (1个文件)
```
源路径: AGENTS.md
目标: docs/development_guide.md
描述: 开发规范和代码标准
状态: 适配为KL8项目开发指南
```

## ⚙️ 配置和数据文件

### 配置文件 (新建)
```
新建: config/default.yaml
内容: 基于src/config.py中的KL8配置
包含: 
  - 算法参数配置
  - 数据路径配置
  - 网络配置
  - 日志配置

新建: config/production.yaml
内容: 生产环境优化配置

新建: config/development.yaml
内容: 开发环境配置

新建: .env.example
内容: 环境变量示例
```

### 数据文件 (条件迁移)
```
源路径: data/kl8/
目标: data/raw/
描述: KL8历史数据文件
状态: 软链接或复制迁移

源路径: results/
目标: results/
描述: 分析结果输出目录
状态: 创建新目录

源路径: logs/
目标: logs/
描述: 日志文件目录
状态: 创建新目录
```

## 🧪 测试和示例文件 (新建)

### 测试文件
```
新建: tests/unit/test_analysis.py
描述: 核心分析功能单元测试

新建: tests/unit/test_algorithms.py
描述: 8种高级算法单元测试

新建: tests/integration/test_full_pipeline.py
描述: 完整流程集成测试

新建: tests/fixtures/sample_data.csv
描述: 测试用数据样本
```

### 示例文件
```
新建: examples/basic_analysis.py
描述: 基础分析使用示例

新建: examples/advanced_algorithms.py
描述: 高级算法使用示例

新建: examples/performance_comparison.py
描述: 性能对比示例
```

## 📋 项目配置文件清单

### Python包配置
```
新建: pyproject.toml
描述: 现代Python项目配置文件
内容: 包信息、依赖、工具配置

新建: setup.py
描述: 传统安装脚本（兼容性）
内容: 包安装和分发配置

新建: requirements.txt
描述: 生产环境依赖
大小: ~20个核心依赖包

新建: requirements-dev.txt
描述: 开发环境依赖
内容: 测试、代码质量、文档工具

新建: requirements-test.txt
描述: 测试环境专用依赖
```

### 开发工具配置
```
新建: .gitignore
描述: Git忽略规则文件

新建: .pre-commit-config.yaml
描述: 预提交代码检查钩子

新建: .flake8
描述: 代码风格检查配置

新建: .isort.cfg
描述: 导入语句排序配置

新建: mypy.ini
描述: 静态类型检查配置

新建: pytest.ini
描述: 测试框架配置

新建: Makefile
描述: 开发任务自动化

新建: Dockerfile
描述: 容器化配置

新建: docker-compose.yml
描述: 多服务编排配置
```

### GitHub配置
```
新建: .github/workflows/ci.yml
描述: 持续集成流水线

新建: .github/workflows/release.yml
描述: 自动发布流程

新建: .github/ISSUE_TEMPLATE/
描述: Issue模板目录

新建: .github/PULL_REQUEST_TEMPLATE.md
描述: PR模板文件
```

## 🔄 需要重构的代码部分

### 1. 导入路径修改
```python
# 原始导入 (需要修改)
from ..config import *
from ..common import get_data_run
from ..config import name_path, data_file_name

# 新项目导入 (目标格式)
from kl8_analyzer.utils.config import get_kl8_config, get_data_path
from kl8_analyzer.utils.data_fetcher import fetch_kl8_data
from kl8_analyzer.utils.common import setup_logging
```

### 2. 配置系统重构
```python
# 原始配置访问 (需要修改)
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))

# 新项目配置访问 (目标格式)
from kl8_analyzer.utils.config import Config
config = Config.load()
data_path = config.get_data_path("raw") / "kl8_data.csv"
ori_data = pd.read_csv(data_path)
```

### 3. 命令行参数重构
```python
# 原始参数解析 (需要修改)
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str)

# 新项目参数解析 (目标格式)
# KL8专用，移除name参数，专门化参数设计
parser = argparse.ArgumentParser(description="KL8 Lottery Analyzer")
parser.add_argument('--mode', choices=[0, 1, 2], default=0, 
                   help="Algorithm mode: 0=original, 1=intermediate, 2=advanced")
```

## 📊 文件统计信息

### 核心代码文件
- **总计**: 5个核心算法文件
- **代码行数**: 约8000行 (包含8种高级算法)
- **文件大小**: 约500KB
- **依赖包数**: 15个主要依赖

### 文档文件
- **总计**: 5个主要文档文件
- **内容量**: 约120KB markdown文档
- **覆盖范围**: 算法理论、使用指南、性能分析

### 配置文件
- **总计**: 20+个配置文件
- **类型**: Python包配置、开发工具配置、CI/CD配置
- **覆盖**: 开发全生命周期

## ✅ 迁移优先级

### 🔥 高优先级 (立即执行)
1. 核心算法文件 (analysis.py系列)
2. 主要技术文档 (algorithm_theory.md等)
3. 基础项目配置 (pyproject.toml, requirements.txt)

### 🟡 中优先级 (后续完善)
1. 支持工具模块重构
2. 测试文件编写
3. CI/CD流水线设置

### 🟢 低优先级 (最后完善)
1. 示例代码编写
2. 详细文档完善
3. 容器化配置

这个清单提供了KL8项目独立化所需的完整文件迁移计划，确保新项目具备完整的功能和专业的项目结构。