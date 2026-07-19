"""冻结、远程封存验证、评价和最终汇总快乐8 v2 的未来前瞻记录。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping
from urllib.parse import quote

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_v2.prospective_monitor import (  # noqa: E402
    DATA_RELATIVE_PATH,
    FINAL_EVALUATION_SEAL_FILENAME,
    FORMAL_SUMMARY_FILENAME,
    FREEZE_RELATIVE_PATH,
    FROZEN_SOURCE_PATHS,
    MANIFEST_RELATIVE_DIR,
    RESULT_RELATIVE_DIR,
    GitHubSealClient,
    build_evaluation_record,
    build_final_evaluation_seal,
    build_formal_summary,
    build_manifest,
    canonical_json_bytes,
    load_and_verify_freeze_config,
    load_history_csv,
    require_active_freeze,
    require_contract_path,
    utc_now_string,
    write_evaluation_exclusive,
    write_final_evaluation_seal_exclusive,
    write_manifest_exclusive,
)


class GhCliSealClient(GitHubSealClient):
    """用已登录 gh 调用 GitHub API；任何错误均向上抛出并 fail closed。"""

    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable or _find_gh_executable()

    def get_pull_request(self, repository: str, pr_number: int) -> Mapping[str, object]:
        completed = subprocess.run(
            [
                self._executable,
                "pr",
                "view",
                str(pr_number),
                "--repo",
                repository,
                "--json",
                "number,url,state,mergedAt,mergeCommit,baseRefName",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("GitHub PR响应不是JSON对象")
        payload["repository"] = repository
        return payload

    def get_file_bytes(
        self, repository: str, repository_path: str, commit_sha: str
    ) -> bytes:
        encoded_path = quote(repository_path, safe="/")
        completed = subprocess.run(
            [
                self._executable,
                "api",
                f"repos/{repository}/contents/{encoded_path}?ref={commit_sha}",
                "-H",
                "Accept: application/vnd.github.raw+json",
            ],
            check=True,
            capture_output=True,
        )
        return completed.stdout


def _find_gh_executable() -> str:
    located = shutil.which("gh.exe") or shutil.which("gh")
    if located is not None:
        return located
    standard = Path(r"C:\Program Files\GitHub CLI\gh.exe")
    if standard.is_file():
        return str(standard)
    raise FileNotFoundError("未找到已安装的GitHub CLI，远程封存验证失败")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析四个严格分离的前瞻阶段操作。"""

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
    manifest.add_argument("--data", type=Path, default=DATA_RELATIVE_PATH)
    manifest.add_argument("--freeze", type=Path, default=FREEZE_RELATIVE_PATH)
    manifest.add_argument("--manifest-dir", type=Path, default=MANIFEST_RELATIVE_DIR)
    manifest.add_argument("--results-dir", type=Path, default=RESULT_RELATIVE_DIR)

    evaluate = subparsers.add_parser(
        "evaluate", help="开奖后远程验证封存并只写入一次评价"
    )
    evaluate.add_argument("--target-issue", type=int, required=True)
    evaluate.add_argument("--seal-pr-number", type=int, required=True)
    evaluate.add_argument("--official-result-source-url", required=True)
    evaluate.add_argument("--official-result-published-at-utc", required=True)
    evaluate.add_argument("--data", type=Path, default=DATA_RELATIVE_PATH)
    evaluate.add_argument("--freeze", type=Path, default=FREEZE_RELATIVE_PATH)
    evaluate.add_argument("--manifest-dir", type=Path, default=MANIFEST_RELATIVE_DIR)
    evaluate.add_argument("--results-dir", type=Path, default=RESULT_RELATIVE_DIR)

    finalize = subparsers.add_parser(
        "finalize-evaluation", help="远程锚定没有下一份manifest的第365期evaluation"
    )
    finalize.add_argument("--target-issue", type=int, required=True)
    finalize.add_argument("--evaluation-seal-pr-number", type=int, required=True)

    summary = subparsers.add_parser(
        "summary", help="仅在365期完整链完成后生成固定分块bootstrap汇总"
    )
    summary.add_argument("--freeze", type=Path, default=FREEZE_RELATIVE_PATH)
    summary.add_argument("--manifest-dir", type=Path, default=MANIFEST_RELATIVE_DIR)
    summary.add_argument("--results-dir", type=Path, default=RESULT_RELATIVE_DIR)
    return parser.parse_args(argv)


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def _assert_frozen_worktree_clean(project_root: Path) -> None:
    paths = [FREEZE_RELATIVE_PATH.as_posix(), *FROZEN_SOURCE_PATHS]
    completed = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError("冻结源码或配置存在未提交修改，拒绝生产操作")


def _validated_active_config(
    project_root: Path, config_path: Path
) -> dict[str, object]:
    config = load_and_verify_freeze_config(project_root, config_path)
    require_active_freeze(project_root, config)
    _assert_frozen_worktree_clean(project_root)
    return config


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
    results_dir = require_contract_path(
        project_root, args.results_dir, RESULT_RELATIVE_DIR, "前瞻结果目录"
    )
    _validated_active_config(project_root, config_path)
    issues, draws = load_history_csv(data_path)
    manifest = build_manifest(
        project_root=project_root,
        data_path=data_path,
        config_path=config_path,
        manifest_dir=manifest_dir,
        results_dir=results_dir,
        issues=issues,
        draws=draws,
        target_issue=int(args.target_issue),
        local_manifest_generated_at_utc=utc_now_string(),
        git_commit_sha=_git_head(project_root),
        official_source_url=str(args.official_source_url),
        official_confirmed_at_utc=str(args.official_confirmed_at_utc),
    )
    return write_manifest_exclusive(manifest, manifest_dir)


def _evaluate_command(
    args: argparse.Namespace,
    project_root: Path,
    *,
    seal_client: GitHubSealClient | None = None,
) -> Path:
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
    _validated_active_config(project_root, config_path)
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
        seal_pr_number=int(args.seal_pr_number),
        official_result_source_url=str(args.official_result_source_url),
        official_result_published_at_utc=str(args.official_result_published_at_utc),
        evaluated_at_utc=utc_now_string(),
        seal_client=seal_client or GhCliSealClient(),
    )
    return write_evaluation_exclusive(record, results_dir)


def _summary_command(args: argparse.Namespace, project_root: Path) -> Path:
    data_path = require_contract_path(
        project_root, DATA_RELATIVE_PATH, DATA_RELATIVE_PATH, "正式数据路径"
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
    config = _validated_active_config(project_root, config_path)
    issues, draws = load_history_csv(data_path)
    summary = build_formal_summary(
        manifest_dir=manifest_dir,
        results_dir=results_dir,
        config=config,
        official_issues=issues,
        official_draws=draws,
    )
    output = results_dir / FORMAL_SUMMARY_FILENAME
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_bytes(summary))
    except FileExistsError as exc:
        raise FileExistsError(f"正式汇总已存在，拒绝覆盖：{output}") from exc
    return output


def _finalize_evaluation_command(
    args: argparse.Namespace,
    project_root: Path,
    *,
    seal_client: GitHubSealClient | None = None,
) -> Path:
    config_path = require_contract_path(
        project_root, FREEZE_RELATIVE_PATH, FREEZE_RELATIVE_PATH, "冻结配置路径"
    )
    results_dir = require_contract_path(
        project_root, RESULT_RELATIVE_DIR, RESULT_RELATIVE_DIR, "前瞻结果目录"
    )
    _validated_active_config(project_root, config_path)
    seal = build_final_evaluation_seal(
        project_root=project_root,
        config_path=config_path,
        results_dir=results_dir,
        target_issue=int(args.target_issue),
        evaluation_seal_pr_number=int(args.evaluation_seal_pr_number),
        seal_client=seal_client or GhCliSealClient(),
    )
    output = write_final_evaluation_seal_exclusive(seal, results_dir)
    if output.name != FINAL_EVALUATION_SEAL_FILENAME:
        raise RuntimeError("final evaluation seal输出路径漂移")
    return output


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if args.command == "manifest":
        output = _manifest_command(args, project_root)
    elif args.command == "evaluate":
        output = _evaluate_command(args, project_root)
    elif args.command == "finalize-evaluation":
        output = _finalize_evaluation_command(args, project_root)
    elif args.command == "summary":
        output = _summary_command(args, project_root)
    else:  # pragma: no cover - argparse已限制分支
        raise RuntimeError(f"未知命令：{args.command}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
