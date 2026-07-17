"""冻结、评价和最终汇总快乐8 v2 的未来前瞻记录。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_v2.prospective_monitor import (  # noqa: E402
    DATA_RELATIVE_PATH,
    FREEZE_RELATIVE_PATH,
    MANIFEST_RELATIVE_DIR,
    RESULT_RELATIVE_DIR,
    build_evaluation_record,
    build_formal_primary_summary,
    build_manifest,
    canonical_json_bytes,
    load_and_verify_freeze_config,
    load_history_csv,
    require_contract_path,
    utc_now_string,
    write_evaluation_exclusive,
    write_manifest_exclusive,
)


def parse_args() -> argparse.Namespace:
    """解析三个严格分离的前瞻阶段操作。"""

    parser = argparse.ArgumentParser(
        description="快乐8 v2 冻结模型的前瞻manifest与开奖后评价"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", help="在官方目标期开奖前独占写入一个manifest"
    )
    manifest.add_argument("--target-issue", type=int, required=True)
    manifest.add_argument("--official-source-url", required=True)
    manifest.add_argument("--official-confirmed-at-utc", required=True)
    manifest.add_argument("--generated-at-utc")
    manifest.add_argument("--git-commit-sha")
    manifest.add_argument("--data", type=Path, default=DATA_RELATIVE_PATH)
    manifest.add_argument("--freeze", type=Path, default=FREEZE_RELATIVE_PATH)
    manifest.add_argument("--manifest-dir", type=Path, default=MANIFEST_RELATIVE_DIR)

    evaluate = subparsers.add_parser(
        "evaluate", help="开奖后只使用已封存概率写入一次评价"
    )
    evaluate.add_argument("--target-issue", type=int, required=True)
    evaluate.add_argument("--official-result-published-at-utc", required=True)
    evaluate.add_argument("--evaluated-at-utc")
    evaluate.add_argument("--data", type=Path, default=DATA_RELATIVE_PATH)
    evaluate.add_argument("--freeze", type=Path, default=FREEZE_RELATIVE_PATH)
    evaluate.add_argument("--manifest-dir", type=Path, default=MANIFEST_RELATIVE_DIR)
    evaluate.add_argument("--results-dir", type=Path, default=RESULT_RELATIVE_DIR)

    summary = subparsers.add_parser(
        "summary", help="仅在365期全部完成后生成Holm校正主要指标汇总"
    )
    summary.add_argument("--freeze", type=Path, default=FREEZE_RELATIVE_PATH)
    summary.add_argument("--results-dir", type=Path, default=RESULT_RELATIVE_DIR)
    return parser.parse_args()


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def _manifest_command(args: argparse.Namespace, project_root: Path) -> Path:
    data_path = require_contract_path(
        project_root, args.data, DATA_RELATIVE_PATH, "正式数据路径"
    )
    config_path = require_contract_path(
        project_root, args.freeze, FREEZE_RELATIVE_PATH, "冻结配置路径"
    )
    manifest_dir = require_contract_path(
        project_root, args.manifest_dir, MANIFEST_RELATIVE_DIR, "manifest目录"
    )
    issues, draws = load_history_csv(data_path)
    manifest = build_manifest(
        project_root=project_root,
        data_path=data_path,
        config_path=config_path,
        issues=issues,
        draws=draws,
        target_issue=int(args.target_issue),
        generated_at_utc=args.generated_at_utc or utc_now_string(),
        git_commit_sha=args.git_commit_sha or _git_head(project_root),
        official_source_url=str(args.official_source_url),
        official_confirmed_at_utc=str(args.official_confirmed_at_utc),
    )
    return write_manifest_exclusive(manifest, manifest_dir)


def _evaluate_command(args: argparse.Namespace, project_root: Path) -> Path:
    data_path = require_contract_path(
        project_root, args.data, DATA_RELATIVE_PATH, "正式数据路径"
    )
    config_path = require_contract_path(
        project_root, args.freeze, FREEZE_RELATIVE_PATH, "冻结配置路径"
    )
    manifest_dir = require_contract_path(
        project_root, args.manifest_dir, MANIFEST_RELATIVE_DIR, "manifest目录"
    )
    results_dir = require_contract_path(
        project_root, args.results_dir, RESULT_RELATIVE_DIR, "前瞻结果目录"
    )
    issues, draws = load_history_csv(data_path)
    matches = np.flatnonzero(issues == int(args.target_issue))
    if len(matches) != 1:
        raise ValueError("正式数据中必须恰好包含一次目标期开奖")
    actual = draws[int(matches[0])]
    record = build_evaluation_record(
        project_root=project_root,
        config_path=config_path,
        manifest_path=manifest_dir / f"{int(args.target_issue)}.json",
        actual_numbers=actual,
        official_result_published_at_utc=str(args.official_result_published_at_utc),
        evaluated_at_utc=args.evaluated_at_utc or utc_now_string(),
    )
    return write_evaluation_exclusive(record, results_dir)


def _summary_command(args: argparse.Namespace, project_root: Path) -> Path:
    config_path = require_contract_path(
        project_root, args.freeze, FREEZE_RELATIVE_PATH, "冻结配置路径"
    )
    load_and_verify_freeze_config(project_root, config_path)
    results_dir = require_contract_path(
        project_root, args.results_dir, RESULT_RELATIVE_DIR, "前瞻结果目录"
    )
    summary = build_formal_primary_summary(results_dir)
    output = results_dir / "formal_primary_summary.json"
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_bytes(summary))
    except FileExistsError as exc:
        raise FileExistsError(f"正式汇总已存在，拒绝覆盖：{output}") from exc
    return output


def run() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if args.command == "manifest":
        output = _manifest_command(args, project_root)
    elif args.command == "evaluate":
        output = _evaluate_command(args, project_root)
    elif args.command == "summary":
        output = _summary_command(args, project_root)
    else:  # pragma: no cover - argparse已限制分支
        raise RuntimeError(f"未知命令：{args.command}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
