# KL8 Running 批量运行指南

## 概述

`kl8_running.py` 是一个批量运行脚本，用于同时执行多个 KL8 分析和现金流分析任务。该脚本优化了多线程架构，避免了数据下载冲突问题。

## 主要特性

### 🚀 优化的多线程架构
- **统一数据下载**：主线程完成数据下载后，所有工作线程共享数据
- **避免下载冲突**：多线程不再并发下载，消除竞争条件
- **路径自动解析**：使用绝对路径定位脚本文件，避免路径错误

### 📊 灵活的运行模式
- `--running_mode 0`：同时运行分析和现金流模式
- `--running_mode 1`：仅运行分析模式  
- `--running_mode 2`：仅运行现金流模式

## 使用方法

### 基本用法

```bash
# 运行分析模式，处理多个参数组合
python src/analysis/kl8_running.py \
    --cal_nums_list "5,7,10" \
    --total_create_list "50,100" \
    --nums_range "2024001,2024010" \
    --running_mode 1
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cal_nums_list` | "4,5,7,10" | 选号数量列表（逗号分隔） |
| `--total_create_list` | "50,100,1000" | 生成数量列表（逗号分隔） |
| `--nums_range` | "2023140,2023241" | 期号范围（开始,结束） |
| `--repeat` | 1 | 重复次数 |
| `--running_mode` | 0 | 运行模式（0=全部，1=分析，2=现金流） |
| `--max_workers` | 4 | 最大工作线程数 |
| `--random_mode` | 0 | 随机模式 |
| `--download` | 1 | 是否下载数据（1=是，0=否） |

### 实际使用示例

#### 示例1：快速测试
```bash
# 小规模测试：选5个号码，生成10次预测，针对最近3期
python src/analysis/kl8_running.py \
    --cal_nums_list "5" \
    --total_create_list "10" \
    --nums_range "2024268,2024270" \
    --running_mode 1 \
    --download 1
```

#### 示例2：综合分析
```bash
# 多参数组合：不同选号数量和生成次数
python src/analysis/kl8_running.py \
    --cal_nums_list "5,7,10" \
    --total_create_list "50,100" \
    --nums_range "2024200,2024250" \
    --running_mode 0 \
    --max_workers 8
```

#### 示例3：现金流分析
```bash
# 仅运行现金流分析
python src/analysis/kl8_running.py \
    --cal_nums_list "7,10" \
    --total_create_list "100" \
    --running_mode 2 \
    --download 0  # 假设数据已存在
```

## 工作流程

1. **数据下载阶段**
   - 检查 `--download` 参数
   - 如果需要，统一下载最新 KL8 数据
   - 避免多线程下载冲突

2. **任务分配阶段**
   - 根据参数列表生成任务组合
   - 分析模式：每个期号单独创建线程
   - 现金流模式：每个参数组合创建线程

3. **并行执行阶段**
   - 启动多个工作线程
   - 每个线程调用对应的 plus 版本脚本
   - 实时显示进度条

4. **结果收集阶段**
   - 等待所有线程完成
   - 结果文件保存到对应目录

## 输出文件

### 分析模式输出
- 路径格式：`results_{total_create}_{cal_nums}/`
- 文件命名：`result_{timestamp}_{cal_nums}_{period}.csv`

### 现金流模式输出  
- 路径格式：`results_{total_create}_{cal_nums}/`
- 文件命名：`kl8_runnint_results.txt`

## 性能优化建议

### 线程数配置
- CPU密集型任务：`max_workers = CPU核心数`
- I/O密集型任务：`max_workers = CPU核心数 * 2`
- 默认值4适合大多数场景

### 内存管理
- 大批量任务建议分批运行
- 监控内存使用，避免系统过载

### 存储空间
- 确保有足够磁盘空间存储结果
- 定期清理旧的结果文件

## 故障排除

### 常见问题

**问题1：文件路径错误**
```
python: can't open file 'xxx': No such file or directory
```
- 确保从项目根目录运行脚本
- 检查 plus 版本脚本文件是否存在

**问题2：目录创建失败**
```
FileExistsError: 当文件已存在时，无法创建该文件
```
- 已在新版本中修复，使用 `exist_ok=True`

**问题3：数据下载失败**
```
网络连接错误或数据源不可用
```
- 检查网络连接
- 使用 `--download 0` 跳过下载（如果数据已存在）

### 调试技巧

1. **单线程测试**
   ```bash
   # 使用最小参数测试
   --cal_nums_list "5" --total_create_list "5" --nums_range "2024001,2024001"
   ```

2. **分模式测试**  
   ```bash
   # 先测试分析模式
   --running_mode 1
   # 再测试现金流模式
   --running_mode 2
   ```

3. **详细日志**
   - 查看各个 plus 脚本的输出信息
   - 检查生成的结果文件内容

## 版本历史

- **v1.2.1**: 修复多线程下载冲突、目录创建竞争条件
- **v1.1.0**: 初始多线程架构版本
- **v1.0.0**: 基础功能版本

## 相关文档

- [KL8 使用指南](kl8_usage_guide.md) - 基础分析功能
- [算法理论文档](kl8_algorithm_theory.md) - 核心算法原理
- [API 文档](api.md) - 程序接口说明