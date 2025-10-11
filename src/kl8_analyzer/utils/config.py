# -*- coding: utf-8 -*-
"""
KL8专用配置管理模块
提供KL8算法相关的配置管理功能
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass

@dataclass
class KL8Config:
    """KL8配置类"""
    name: str = "快乐8"
    code: str = "kl8"
    sequence_len: int = 20
    num_classes: int = 80
    default_window: int = 6
    cal_nums: int = 20
    
    # 高级算法模式配置
    advanced_modes: Dict[str, Dict[str, Any]] = None
    
    # 数据路径配置
    data_path: str = "data"
    results_path: str = "results"
    logs_path: str = "logs"
    
    def __post_init__(self):
        if self.advanced_modes is None:
            self.advanced_modes = self._get_default_advanced_modes()
    
    def _get_default_advanced_modes(self) -> Dict[str, Dict[str, Any]]:
        """获取默认的高级算法模式配置"""
        return {
            "advanced_mode_0": {
                "name": "修正贝叶斯分析",
                "enabled": True,
                "weight": 0.15,
                "parameters": {}
            },
            "advanced_mode_1": {
                "name": "马尔可夫链分析", 
                "enabled": True,
                "weight": 0.12,
                "parameters": {}
            },
            "advanced_mode_2": {
                "name": "遗传算法优化",
                "enabled": True,
                "weight": 0.13,
                "parameters": {}
            },
            "advanced_mode_3": {
                "name": "深度学习特征",
                "enabled": True,
                "weight": 0.18,
                "parameters": {}
            },
            "advanced_mode_4": {
                "name": "信息熵分析",
                "enabled": True,
                "weight": 0.11,
                "parameters": {}
            },
            "advanced_mode_5": {
                "name": "自适应阈值",
                "enabled": True,
                "weight": 0.14,
                "parameters": {}
            },
            "advanced_mode_6": {
                "name": "统计检验",
                "enabled": True,
                "weight": 0.09,
                "parameters": {}
            },
            "advanced_mode_7": {
                "name": "集成策略",
                "enabled": True,
                "weight": 0.08,
                "parameters": {}
            }
        }


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config")
        self._config = None
    
    def load_config(self, env: str = "default") -> KL8Config:
        """加载配置文件"""
        config_file = self.config_dir / f"{env}.yaml"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 从YAML创建配置对象
            kl8_data = config_data.get('kl8_config', {})
            return KL8Config(**kl8_data)
        else:
            # 返回默认配置
            return KL8Config()
    
    @property 
    def config(self) -> KL8Config:
        """获取当前配置"""
        if self._config is None:
            env = os.getenv('KL8_ENV', 'default')
            self._config = self.load_config(env)
        return self._config
    
    def get_data_path(self, subdir: str = "") -> Path:
        """获取数据路径"""
        base_path = Path(self.config.data_path)
        if subdir:
            return base_path / subdir
        return base_path
    
    def get_results_path(self, subdir: str = "") -> Path:
        """获取结果路径"""
        base_path = Path(self.config.results_path)
        if subdir:
            return base_path / subdir
        return base_path
    
    def get_logs_path(self, subdir: str = "") -> Path:
        """获取日志路径"""
        base_path = Path(self.config.logs_path)
        if subdir:
            return base_path / subdir
        return base_path


# 全局配置管理器实例
_config_manager = ConfigManager()

def get_kl8_config() -> KL8Config:
    """获取KL8配置"""
    return _config_manager.config

def get_data_path(subdir: str = "") -> Path:
    """获取数据路径"""
    return _config_manager.get_data_path(subdir)

def get_results_path(subdir: str = "") -> Path:
    """获取结果路径"""
    return _config_manager.get_results_path(subdir)

def get_logs_path(subdir: str = "") -> Path:
    """获取日志路径"""
    return _config_manager.get_logs_path(subdir)

def ensure_runtime_directories():
    """确保运行时目录存在"""
    for path_getter in [get_data_path, get_results_path, get_logs_path]:
        path = path_getter()
        path.mkdir(parents=True, exist_ok=True)


# 兼容性变量 - 支持原有代码中的导入
name_path = {
    "kl8": {
        "path": "data/",
        "name": "快乐8",
        "sequence_len": 20,
        "num_classes": 80
    }
}

data_file_name = "kl8_history.csv"

# 多彩票类型配置（用于脚本兼容）
LOTTERY_CONFIGS = {
    "kl8": {
        "name": "快乐8",
        "code": "kl8", 
        "sequence_len": 20,
        "num_classes": 80,
        "default_window": 6,
        "cal_nums": 20
    },
    # 其他彩票类型可以在这里添加，但目前只支持KL8
}

# 导出兼容性变量供其他模块使用
__all__ = [
    "get_kl8_config",
    "get_data_path", 
    "get_results_path",
    "get_logs_path",
    "ensure_runtime_directories",
    "KL8Config",
    "ConfigManager",
    "name_path",
    "data_file_name",
    "LOTTERY_CONFIGS"
]