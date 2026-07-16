#!/usr/bin/env python3
"""运行快乐8冻结策略的真正前瞻评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scientific.prospective import (  # noqa: E402
    resolve_prospective_paths,
    run_prospective_evaluation,
    validate_runtime_storage,
)


def build_parser() -> argparse.ArgumentParser:
    """建立只允许固定路径和冻结策略的命令行接口。"""

    parser = argparse.ArgumentParser(
        description=(
            "快乐8冻结策略前瞻评估；只处理2026186之后真实到达的开奖，"
            "不重新切分holdout或调参。"
        )
    )
    parser.add_argument(
        "--data",
        default="data_cache/kl8/data.csv",
        help="固定为仓库内data_cache/kl8/data.csv",
    )
    parser.add_argument(
        "--freeze-config",
        default="config/scientific_freeze.json",
        help="固定为版本化科学冻结配置",
    )
    parser.add_argument(
        "--output-dir",
        default="results/prospective",
        help="固定为results/prospective",
    )
    parser.add_argument(
        "--report",
        default="reports/kl8_prospective_report.md",
        help="固定为前瞻专用报告，绝不覆盖旧科学报告",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="累计至少2个前瞻期后用于描述区间的按期重抽样次数",
    )
    parser.add_argument(
        "--next-issue",
        type=int,
        help="可选的下一期官方期号；默认使用最新期号加1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """验证D盘运行边界并执行前瞻评估。"""

    args = build_parser().parse_args(argv)
    try:
        validate_runtime_storage(PROJECT_ROOT)
        paths = resolve_prospective_paths(
            PROJECT_ROOT,
            data=args.data,
            freeze_config=args.freeze_config,
            output_dir=args.output_dir,
            report=args.report,
        )
        result = run_prospective_evaluation(
            paths,
            bootstrap_samples=args.bootstrap_samples,
            next_issue=args.next_issue,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"前瞻评估失败：{error}", file=sys.stderr)
        print("请检查D盘仓库、冻结配置、数据快照和已有前瞻记录。", file=sys.stderr)
        return 2
    issue_text = ",".join(map(str, result.prospective_issues))
    print(
        f"完成：前瞻期号={issue_text}，记录={result.records}，"
        f"摘要={result.summary_rows}，下一期={result.next_issue}，"
        f"候选票面={result.candidate_rows}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
