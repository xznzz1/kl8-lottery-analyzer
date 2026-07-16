# -*- coding: utf-8 -*-
"""快乐8历史数据下载脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher import download_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载快乐8历史开奖数据")
    parser.add_argument(
        "--name",
        default="kl8",
        type=str,
        help="彩票类型，目前仅支持 kl8",
    )
    parser.add_argument(
        "--cq",
        default=0,
        type=int,
        choices=(0, 1),
        help="是否使用出球顺序：0=号码排序，1=出球顺序",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()

    if code != "kl8":
        raise SystemExit(f"当前仅支持 kl8，收到：{args.name}")

    download_history(
        code="kl8",
        use_sequence_order=bool(args.cq),
    )


if __name__ == "__main__":
    main()
