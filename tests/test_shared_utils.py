# -*- coding: utf-8 -*-
"""
测试 src.analysis.shared_utils 模块

覆盖路径计算、异步写入、奇偶检查、连号查找等工具函数
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.analysis import shared_utils
from src.analysis.shared_utils import (
    check_odd_even,
    compute_output_dir,
    ensure_dir,
    find_consecutive_number,
    write_results_async,
    write_results_core,
)


class TestComputeOutputDir:
    """测试输出目录路径计算逻辑"""

    def test_compute_output_dir_basic(self):
        """测试基本路径计算"""
        result = compute_output_dir(0, "test_label")

        # 验证路径包含必要组件
        assert "results" in result
        assert "test_label" in result

        # 验证路径格式
        assert result.endswith("test_label/")

    def test_compute_output_dir_random_mode(self):
        """测试不同模式的路径计算"""
        result1 = compute_output_dir(0, "label1")  # results mode
        result2 = compute_output_dir(1, "label1")  # random mode

        # 不同模式应产生不同路径
        assert result1 != result2
        assert "results" in result1
        assert "random" in result2

    def test_compute_output_dir_empty_label(self):
        """测试空标签的路径计算"""
        result1 = compute_output_dir(0, "")
        result2 = compute_output_dir(1, "")

        project_root = Path(__file__).resolve().parents[1]
        assert Path(result1).resolve() == project_root / "results" / "legacy"
        assert Path(result2).resolve() == project_root / "results" / "random"

    @pytest.mark.parametrize("label", ["../outside", r"..\outside", "D:outside"])
    def test_compute_output_dir_rejects_path_traversal(self, label):
        with pytest.raises(ValueError, match="输出标签"):
            compute_output_dir(0, label)


class TestEnsureDir:
    """测试目录创建功能"""

    def test_ensure_dir_creates_directory(self):
        """测试目录创建功能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test_subdir")

            # 目录不存在时应创建
            assert not os.path.exists(test_path)
            ensure_dir(test_path)
            assert os.path.exists(test_path)
            assert os.path.isdir(test_path)

    def test_ensure_dir_existing_directory(self):
        """测试对已存在目录的处理"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 对已存在目录应无副作用
            ensure_dir(tmpdir)
            assert os.path.exists(tmpdir)
            assert os.path.isdir(tmpdir)


class TestCheckOddEven:
    """测试奇偶判断功能"""

    def test_check_odd_even_basic(self):
        """测试基本奇偶判断"""
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        odd_count, even_count = check_odd_even(test_data)

        assert odd_count == 5  # 1, 3, 5, 7, 9
        assert even_count == 5  # 2, 4, 6, 8, 10

    def test_check_odd_even_all_odd(self):
        """测试全奇数情况"""
        test_data = [1, 3, 5, 7, 9]
        odd_count, even_count = check_odd_even(test_data)

        assert odd_count == 5
        assert even_count == 0

    def test_check_odd_even_all_even(self):
        """测试全偶数情况"""
        test_data = [2, 4, 6, 8, 10]
        odd_count, even_count = check_odd_even(test_data)

        assert odd_count == 0
        assert even_count == 5

    def test_check_odd_even_empty(self):
        """测试空列表"""
        odd_count, even_count = check_odd_even([])
        assert odd_count == 0
        assert even_count == 0


class TestFindConsecutiveNumber:
    """测试连号查找功能"""

    def test_find_consecutive_number_basic(self):
        """测试基本连号查找"""
        test_data = [1, 2, 3, 5, 6, 8, 9, 10]
        result = find_consecutive_number(test_data)

        # 应找到连号组合：[1,2,3], [5,6], [8,9,10]
        assert isinstance(result, list)
        assert len(result) > 0

    def test_find_consecutive_number_no_consecutive(self):
        """测试无连号情况"""
        test_data = [1, 3, 5, 7, 9]
        result = find_consecutive_number(test_data)

        # 无连号时应返回空列表或特定标识
        assert isinstance(result, list)

    def test_find_consecutive_number_all_consecutive(self):
        """测试全连号情况"""
        test_data = [1, 2, 3, 4, 5]
        result = find_consecutive_number(test_data)

        assert isinstance(result, list)
        # 全连续应被识别为一个大的连号组


class TestWriteResults:
    """测试结果写入功能"""

    def test_write_results_core_basic(self, monkeypatch, tmp_path):
        """测试核心写入功能"""
        monkeypatch.setattr(shared_utils, "RESULTS_ROOT", tmp_path)
        file_dir = f"{tmp_path.as_posix()}/"
        test_rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        write_results_core(
            rows=test_rows,
            file_dir=file_dir,
            file_prefix="test",
            cal_nums=3,
            total_create=10,
            multiple=1,
            multiple_ratio="1,0",
            period_num="123",
            current_time_str="20231012",
        )

        # 验证文件被创建
        files = os.listdir(tmp_path)
        assert len(files) > 0

        # 验证文件内容
        csv_file = [f for f in files if f.endswith(".csv")][0]
        with open(tmp_path / csv_file, "r") as f:
            content = f.read()
            assert "b1,b2,b3" in content  # 验证表头
            assert "1,2,3" in content  # 验证数据

    def test_write_results_async_thread(self, monkeypatch, tmp_path):
        """测试异步写入功能（线程模式）"""
        monkeypatch.setattr(shared_utils, "RESULTS_ROOT", tmp_path)
        file_dir = f"{tmp_path.as_posix()}/"
        test_rows = [[1, 2], [3, 4]]

        write_results_async(
            rows=test_rows,
            file_dir=file_dir,
            file_prefix="async_test",
            cal_nums=2,
            total_create=5,
            multiple=1,
            multiple_ratio="1,0",
            period_num="456",
            current_time_str="20231012",
            backend="thread",
        )

        # 等待异步操作完成
        import time

        time.sleep(0.1)

        # 验证文件被创建
        files = os.listdir(tmp_path)
        assert len(files) > 0

    def test_write_results_creates_directory(self, monkeypatch, tmp_path):
        """测试写入时自动创建目录"""
        monkeypatch.setattr(shared_utils, "RESULTS_ROOT", tmp_path)
        nested_dir = tmp_path / "subdir"
        test_rows = [[1, 2]]

        # 父目录不存在时应自动创建
        write_results_core(
            rows=test_rows,
            file_dir=nested_dir,
            file_prefix="test",
            cal_nums=2,
            total_create=1,
            multiple=1,
            multiple_ratio="1,0",
            period_num="789",
            current_time_str="20231012",
        )

        assert nested_dir.exists()
        files = os.listdir(nested_dir)
        assert len(files) > 0

    def test_write_results_rejects_outside_results(self, monkeypatch, tmp_path):
        results_root = tmp_path / "results"
        monkeypatch.setattr(shared_utils, "RESULTS_ROOT", results_root)
        with pytest.raises(ValueError, match="结果目录必须位于results内"):
            write_results_core(
                rows=[[1]],
                file_dir=tmp_path / "outside",
                file_prefix="blocked",
                cal_nums=1,
                total_create=1,
                multiple=1,
                multiple_ratio="1,0",
                period_num="0",
                current_time_str="20260716",
            )
        assert not (tmp_path / "outside").exists()

    def test_write_results_rejects_prefix_path_traversal(self, monkeypatch, tmp_path):
        results_root = tmp_path / "results"
        monkeypatch.setattr(shared_utils, "RESULTS_ROOT", results_root)
        with pytest.raises(ValueError, match="结果文件前缀"):
            write_results_core(
                rows=[[1]],
                file_dir=results_root / "legacy",
                file_prefix="../../escaped",
                cal_nums=1,
                total_create=1,
                multiple=1,
                multiple_ratio="1,0",
                period_num="0",
                current_time_str="20260716",
            )
        assert not list(tmp_path.glob("escaped*.csv"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
