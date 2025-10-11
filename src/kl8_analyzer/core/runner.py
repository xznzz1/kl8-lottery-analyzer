import argparse
import subprocess
import threading
from tqdm import tqdm

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal_nums_list', default="4,5,7,10", type=str, help='cal_nums_list')
    parser.add_argument('--total_create_list', default="50,100,1000", type=str, help='total_create_list')
    parser.add_argument('--nums_range', default="2023140,2023241", type=str, help='nums_range')
    parser.add_argument('--repeat', default=1, type=int, help='repeat')
    parser.add_argument('--running_mode', default=0, type=int, help='running_mode')
    parser.add_argument('--max_workers', default=4, type=int, help='max_workers')
    parser.add_argument('--random_mode', default=0, type=int, help='random_mode')
    return parser.parse_args()

# 只在作为主程序运行时解析参数
if __name__ == "__main__":
    args = get_args()
else:
    # 导入时使用默认值
    class DefaultArgs:
        def __init__(self):
            self.cal_nums_list = "4,5,7,10"
            self.total_create_list = "50,100,1000"
            self.nums_range = "2023140,2023241"
            self.repeat = 1
            self.running_mode = 0
            self.max_workers = 4
            self.random_mode = 0
    args = DefaultArgs()

# def _main(_total_create, _cal_nums, begin=2023140, end=2023241, _process="./kl8_analysis.py"):
    # for _current_nums in range(begin, end):
    #     subprocess.run(["python", _process, "--download", "0", "--total_create", str(_total_create), \
    #                     "--cal_nums", str(_cal_nums), "--current_nums", str(_current_nums), "--limit_line", "5", \
    #                     "--path", str(_total_create) + '_' + str(_cal_nums), "--repeat", str(args.repeat), "--simple_mode", "1"])
def _main(_total_create, _cal_nums, _current_nums, _process="python -m kl8_analyzer.core.analysis"):
    # 分离python命令和模块路径
    cmd_parts = _process.split()
    cmd_parts.extend(["--download", "0", "--total_create", str(_total_create), 
                     "--cal_nums", str(_cal_nums), "--current_nums", str(_current_nums), "--limit_line", "5",
                     "--path", str(_total_create) + '_' + str(abs(int(_cal_nums))), "--repeat", str(args.repeat), "--simple_mode", "1",
                     "--random_mode", str(args.random_mode), "--max_workers", str(args.max_workers)])
    subprocess.run(cmd_parts)

if __name__ == "__main__":
    # 重新解析参数用于主程序运行
    args = get_args()
    
    # 使用模块内部的分析和现金分析模块
    kl8_analysis = "python -m kl8_analyzer.core.analysis_plus"
    kl8_cash = "python -m kl8_analyzer.core.cash_plus"
    cal_nums_list = [int(element) for element in args.cal_nums_list.split(',')]
    total_create_list = [int(element) for element in args.total_create_list.split(',')]
    begin, end = [int(element) for element in args.nums_range.split(',')]

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
