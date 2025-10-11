import argparse
import subprocess
import threading
from tqdm import tqdm
import sys
from pathlib import Path

# 兼容脚本直跑：相对导入失败时，回退到把项目根加入 sys.path 并做绝对导入
try:
    from ..config import *  # type: ignore
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import *  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument('--cal_nums_list', default="4,5,7,10", type=str, help='cal_nums_list')
parser.add_argument('--total_create_list', default="50,100,1000", type=str, help='total_create_list')
parser.add_argument('--nums_range', default="2023140,2023241", type=str, help='nums_range')
parser.add_argument('--repeat', default=1, type=int, help='repeat')
parser.add_argument('--running_mode', default=0, type=int, help='running_mode')
parser.add_argument('--max_workers', default=4, type=int, help='max_workers')
parser.add_argument('--random_mode', default=0, type=int, help='random_mode')
parser.add_argument('--download', default=1, type=int, help='download data before processing')
args = parser.parse_args()

def download_data_if_needed():
    """统一下载数据，避免多线程冲突"""
    if args.download == 1:
        try:
            from src.common import get_data_run  # type: ignore
        except ImportError:
            import sys as _sys  # type: ignore
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            if str(PROJECT_ROOT) not in _sys.path:
                _sys.path.insert(0, str(PROJECT_ROOT))
            from src.common import get_data_run  # type: ignore
        
        print("正在下载数据...")
        get_data_run(name="kl8", cq=0)
        print("数据下载完成")

def _main(_total_create, _cal_nums, _current_nums, _process="./kl8_analysis.py"):
    """执行单个分析任务，不下载数据（数据已在主线程下载完成）"""
    subprocess.run(["python", _process, "--download", "0", "--total_create", str(_total_create), \
                    "--cal_nums", str(_cal_nums), "--current_nums", str(_current_nums), "--limit_line", "5", \
                    "--path", str(_total_create) + '_' + str(abs(int(_cal_nums))), "--repeat", str(args.repeat), "--simple_mode", "1", \
                    "--random_mode", str(args.random_mode), "--max_workers", str(args.max_workers)])

# 使用绝对路径来找到脚本文件
script_dir = Path(__file__).parent
kl8_analysis = str(script_dir / "kl8_analysis_plus.py")
kl8_cash = str(script_dir / "kl8_cash_plus.py")
cal_nums_list = [int(element) for element in args.cal_nums_list.split(',')]
total_create_list = [int(element) for element in args.total_create_list.split(',')]
begin, end = [int(element) for element in args.nums_range.split(',')]

# 先统一下载数据，避免多线程竞争
download_data_if_needed()

if args.running_mode in [0, 1]:
    threads = []
    for _total_create in total_create_list:
        for _cal_nums in cal_nums_list:
            for _current_nums in range(begin, end + 1):
                t = threading.Thread(target=_main, args=(_total_create, _cal_nums, _current_nums, kl8_analysis))
                threads.append(t)
                t.start()
    # for t in threads:
    for t_index in tqdm(range(len(threads)), desc='AnalysisThread', leave=True):
        t = threads[t_index]
        t.join()

if args.running_mode in [0, 2]:
    threads = []
    for _total_create in total_create_list:
        for _cal_nums in cal_nums_list:
                _current_nums = -1
                t = threading.Thread(target=_main, args=(_total_create, _cal_nums, _current_nums, kl8_cash))
                threads.append(t)
                t.start()
    for t_index in tqdm(range(len(threads)), desc='CashThread', leave=True):
        t = threads[t_index]
        t.join()
