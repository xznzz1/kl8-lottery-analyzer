# -*- coding: utf-8 -*-
"""
快乐 8 历史数据抓取与加载工具（精简版）。

相较于原仓库，该版本仅保留快乐 8（kl8）相关逻辑，负责：
1. 带重试的 HTTP 抓取；
2. HTML / 文本解析为 pandas.DataFrame；
3. 将数据保存到 `data/kl8/data.csv` 并生成下载元信息。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .config import (
    ALLOWED_DOMAINS,
    DATA_FILE_NAME,
    LOTTERY_CONFIGS,
    NETWORK_CONFIG,
    PATHS,
    LotteryModelConfig,
    ensure_runtime_directories,
)


@dataclass(frozen=True)
class ParseStats:
    """记录一次解析的行级质量统计。"""

    raw_rows: int
    valid_rows: int
    rejected_rows: int
    earliest_issue: Optional[str]
    latest_issue: Optional[str]
    rejection_reasons: dict[str, int]


@dataclass(frozen=True)
class DownloadResult:
    """记录一次快乐 8 历史数据下载的结果。"""

    code: str
    total_issues: int
    saved_path: str
    timestamp: str
    source_url: str
    raw_rows: int
    valid_rows: int
    rejected_rows: int
    earliest_issue: str
    latest_issue: str
    csv_sha256: str
    rejection_reasons: dict[str, int]

    @property
    def fetched_at_utc(self) -> str:
        """返回兼容 ISO 8601 的 UTC 抓取时间。"""

        return self.timestamp


class DataRegressionError(ValueError):
    """下载数据的最新期号早于本地数据时抛出。"""


class LotteryHttpClient:
    """封装带重试和域名白名单校验的 HTTP 客户端。"""

    def __init__(
        self,
        timeout: float,
        retries: int,
        backoff_factor: float,
        user_agent: str,
    ) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def get_text(self, url: str) -> str:
        parsed = urlparse(url)
        domain = (parsed.hostname or "").lower()
        if not any(
            domain == allowed or domain.endswith(f".{allowed}")
            for allowed in ALLOWED_DOMAINS
        ):
            raise ValueError(f"禁止访问域名：{domain}")
        response = self._session.get(url, headers=self._headers, timeout=self._timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text


def _build_history_url(
    config: LotteryModelConfig, start: Optional[int], end: Optional[int]
) -> str:
    """构造快乐 8 历史记录页面地址。"""

    base = f"https://datachart.500.com/{config.code}/zoushi/newinc/jbzs_redblue.php"
    start_issue = start or 1
    end_issue = end or 999_999
    if start_issue > end_issue:
        raise ValueError(f"起始期号不得晚于结束期号：{start_issue} > {end_issue}")
    query = urlencode(
        {
            "from": start_issue,
            "to": end_issue,
            "shujcount": 0,
            "sort": 0,
        }
    )
    return f"{base}?{query}"


def _extract_valid_numbers(
    values: Iterable[str], expected_count: int = 20
) -> list[str]:
    """从原始字段中提取1—80的号码，并校验数量和唯一性。"""

    numbers: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value.isdigit():
            raise ValueError(f"开奖号码不是整数：{value!r}")

        number = int(value)
        if not 1 <= number <= 80:
            raise ValueError(f"开奖号码越界：{number}")
        numbers.append(str(number))

    if len(numbers) != expected_count:
        raise ValueError(
            f"开奖号码数量异常：期望{expected_count}个，实际{len(numbers)}个，内容={numbers}"
        )

    if len(set(numbers)) != expected_count:
        raise ValueError(f"开奖号码存在重复：{numbers}")

    return numbers


def _records_to_dataframe(records: list[dict[str, str]]) -> pd.DataFrame:
    """将已校验记录转为按期号倒序排列的数据表。"""

    df = pd.DataFrame(records)
    issue_sort = pd.to_numeric(df["期数"], errors="raise")
    df = df.assign(_issue_sort=issue_sort)
    df.sort_values("_issue_sort", ascending=False, inplace=True)
    return df.drop(columns="_issue_sort").reset_index(drop=True)


def _build_parse_stats(
    raw_rows: int,
    records: list[dict[str, str]],
    rejection_reasons: Counter[str],
) -> ParseStats:
    """根据解析结果构建稳定、可序列化的统计对象。"""

    issues = [int(record["期数"]) for record in records]
    return ParseStats(
        raw_rows=raw_rows,
        valid_rows=len(records),
        rejected_rows=raw_rows - len(records),
        earliest_issue=str(min(issues)) if issues else None,
        latest_issue=str(max(issues)) if issues else None,
        rejection_reasons=dict(sorted(rejection_reasons.items())),
    )


def _parse_issue_list_with_stats(
    config: LotteryModelConfig,
    html: str,
) -> tuple[pd.DataFrame, ParseStats]:
    """严格解析 500.com 快乐 8 走势表，并返回行级质量统计。

    有效开奖行必须恰好包含 81 个 ``td``：第 1 个为数字期号，
    后续 80 列逐一对应号码 1—80。开奖号码列必须标记
    ``chartBall01``，遗漏列必须标记 ``yl01``，且每期恰有 20 个
    互不重复的 1—80 号码。页面中的 ``colspan=81`` 空行会计入拒绝行，
    但不会被误当作开奖记录。
    """

    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody", attrs={"id": "tdata"})
    if not tbody:
        raise ValueError("未找到开奖号码数据表格 (id=tdata)")

    table_rows = tbody.find_all("tr", recursive=False)
    records: list[dict[str, str]] = []
    rejection_reasons: Counter[str] = Counter()
    seen_issues: set[str] = set()

    for tr in table_rows:
        tds = tr.find_all("td", recursive=False)
        if (
            len(tds) == 1
            and not tds[0].get_text(strip=True)
            and str(tds[0].get("colspan", "")) == "81"
        ):
            rejection_reasons["blank_separator"] += 1
            continue

        if len(tds) != 81:
            rejection_reasons["malformed_cell_count"] += 1
            continue

        issue = tds[0].get_text(strip=True)
        if not issue or not issue.isdigit():
            rejection_reasons["invalid_issue"] += 1
            continue
        if issue in seen_issues:
            rejection_reasons["duplicate_issue"] += 1
            continue

        selected_values: list[str] = []
        selected_positions: list[int] = []
        invalid_class = False
        invalid_omission_value = False
        for position, td in enumerate(tds[1:], start=1):
            raw_classes = td.attrs.get("class")
            if isinstance(raw_classes, str):
                classes = set(raw_classes.split())
            elif isinstance(raw_classes, list):
                classes = {str(value) for value in raw_classes}
            else:
                classes = set()
            is_selected = "chartBall01" in classes
            is_omission = "yl01" in classes
            if is_selected == is_omission:
                invalid_class = True
                break

            value = td.get_text(strip=True)
            if is_selected:
                selected_values.append(value)
                selected_positions.append(position)
            elif not value.isdigit():
                invalid_omission_value = True
                break

        if invalid_class:
            rejection_reasons["invalid_cell_class"] += 1
            continue
        if invalid_omission_value:
            rejection_reasons["invalid_omission_value"] += 1
            continue
        if len(selected_values) != config.red.sequence_len:
            rejection_reasons["invalid_selected_count"] += 1
            continue

        try:
            numbers = _extract_valid_numbers(
                selected_values,
                expected_count=config.red.sequence_len,
            )
        except ValueError as exc:
            message = str(exc)
            if "重复" in message:
                reason = "duplicate_numbers"
            elif "越界" in message:
                reason = "out_of_range_numbers"
            else:
                reason = "invalid_selected_value"
            rejection_reasons[reason] += 1
            continue

        if any(
            int(value) != position
            for value, position in zip(numbers, selected_positions)
        ):
            rejection_reasons["number_column_mismatch"] += 1
            continue

        record = {"期数": issue}
        for idx, value in enumerate(numbers, start=1):
            record[f"红球_{idx}"] = value
        records.append(record)
        seen_issues.add(issue)

    stats = _build_parse_stats(len(table_rows), records, rejection_reasons)
    if not records:
        raise ValueError(
            "解析开奖号码失败，未获取到有效数据；"
            f"原始行={stats.raw_rows}，拒绝原因={stats.rejection_reasons}"
        )

    df = _records_to_dataframe(records)
    df.attrs["parse_stats"] = asdict(stats)
    return df, stats


def _parse_issue_list(config: LotteryModelConfig, html: str) -> pd.DataFrame:
    """解析快乐 8 网页历史数据；统计信息保存在 ``DataFrame.attrs``。"""

    df, _ = _parse_issue_list_with_stats(config, html)
    return df


def _parse_kl8_sequence_with_stats(text: str) -> tuple[pd.DataFrame, ParseStats]:
    """解析顺序数据，并返回与网页解析一致的质量统计。"""

    records: list[dict[str, str]] = []
    rejection_reasons: Counter[str] = Counter()
    seen_issues: set[str] = set()
    lines = text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            rejection_reasons["blank_line"] += 1
            continue
        if "," not in line:
            rejection_reasons["malformed_line"] += 1
            continue

        first_segment = line.split(",", 1)[0]
        parts = first_segment.split()
        if len(parts) < 21:
            rejection_reasons["malformed_line"] += 1
            continue

        issue = parts[0].strip()
        if not issue.isdigit():
            rejection_reasons["invalid_issue"] += 1
            continue
        if issue in seen_issues:
            rejection_reasons["duplicate_issue"] += 1
            continue

        second = parts[1] if len(parts) > 1 else ""
        has_date = (
            len(second) == 10
            and second[4:5] == "-"
            and second[7:8] == "-"
            and second.replace("-", "").isdigit()
        )
        number_start = 2 if has_date else 1
        number_tokens = parts[number_start : number_start + 20]

        try:
            numbers = _extract_valid_numbers(number_tokens, expected_count=20)
        except ValueError as exc:
            message = str(exc)
            if "重复" in message:
                reason = "duplicate_numbers"
            elif "越界" in message:
                reason = "out_of_range_numbers"
            else:
                reason = "invalid_numbers"
            rejection_reasons[reason] += 1
            continue

        record = {"期数": issue}
        for index, value in enumerate(numbers, start=1):
            record[f"红球_{index}"] = f"{int(value):02d}"
        records.append(record)
        seen_issues.add(issue)

    stats = _build_parse_stats(len(lines), records, rejection_reasons)
    if not records:
        raise ValueError(
            "快乐8出球顺序数据解析失败；"
            f"原始行={stats.raw_rows}，拒绝原因={stats.rejection_reasons}"
        )

    df = _records_to_dataframe(records)
    df.attrs["parse_stats"] = asdict(stats)
    return df, stats


def _parse_kl8_sequence(text: str) -> pd.DataFrame:
    """解析917500顺序数据，并兼容不带日期的测试/旧数据。

    支持两种格式：
    1. 期号 日期 20个开奖号码,销售额及其他统计信息
    2. 期号 20个开奖号码,其他内容
    """

    df, _ = _parse_kl8_sequence_with_stats(text)
    return df


def get_current_issue(code: str, client: Optional[LotteryHttpClient] = None) -> str:
    """查询快乐 8 最新期号。"""

    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    url = f"https://datachart.500.com/{cfg.code}/"
    html = client.get_text(url)
    soup = BeautifulSoup(html, "lxml")
    wrap = soup.find("div", class_="wrap_datachart")
    if not wrap:
        raise ValueError("未找到数据页面主体 (div.wrap_datachart)")
    input_tag = wrap.find("input", {"id": "to"})
    if not input_tag or not input_tag.has_attr("value"):
        raise ValueError("未从页面提取到最新期号")
    value = str(input_tag["value"]).strip()
    if not value.isdigit():
        raise ValueError(f"最新期号格式异常：{value!r}")
    logger.info("【{}】最新期号：{}", cfg.name, value)
    return value


def _sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，避免将完整 CSV 读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_issue_from_csv(path: Path) -> Optional[str]:
    """读取既有 CSV 的最新期号；损坏文件会明确失败。"""

    if not path.exists():
        return None
    try:
        existing = pd.read_csv(path, encoding="utf-8", dtype={"期数": "string"})
    except Exception as exc:
        raise ValueError(f"无法读取现有数据，不能安全校验期号是否倒退：{path}") from exc
    if "期数" not in existing.columns or existing.empty:
        raise ValueError(f"现有数据为空或缺少【期数】字段，拒绝覆盖：{path}")
    issue_numbers = pd.to_numeric(existing["期数"], errors="coerce")
    if issue_numbers.isna().any():
        raise ValueError(f"现有数据含非法期号，拒绝覆盖：{path}")
    return str(int(issue_numbers.max()))


def _assert_latest_issue_not_regressed(path: Path, downloaded_latest: str) -> None:
    """阻止较旧下载静默覆盖更新的本地历史数据。"""

    existing_latest = _latest_issue_from_csv(path)
    if existing_latest is None:
        return
    if int(downloaded_latest) < int(existing_latest):
        raise DataRegressionError(
            "下载数据最新期号倒退，拒绝覆盖："
            f"本地最新={existing_latest}，下载最新={downloaded_latest}"
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """在同目录写入临时文件后原子替换 JSON 元信息。"""

    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def download_history(
    code: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    use_sequence_order: bool = False,
    client: Optional[LotteryHttpClient] = None,
) -> DownloadResult:
    """下载快乐 8 历史数据并保存到 CSV。"""

    ensure_runtime_directories()
    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    if use_sequence_order:
        source_url = "https://data.917500.cn/kl81000_cq_asc.txt"
        logger.info("下载快乐 8 出球顺序数据...")
        text = client.get_text(source_url)
        df, stats = _parse_kl8_sequence_with_stats(text)
    else:
        source_url = _build_history_url(cfg, start, end)
        logger.info("下载快乐 8 历史数据：{}", source_url)
        html = client.get_text(source_url)
        df, stats = _parse_issue_list_with_stats(cfg, html)

    if stats.latest_issue is None or stats.earliest_issue is None:
        raise ValueError("解析结果缺少期号范围，拒绝保存")

    logger.info(
        "下载解析统计：原始行={}，有效行={}，拒绝行={}，最早期号={}，最新期号={}",
        stats.raw_rows,
        stats.valid_rows,
        stats.rejected_rows,
        stats.earliest_issue,
        stats.latest_issue,
    )

    save_dir = PATHS["data"] / cfg.code
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / DATA_FILE_NAME
    _assert_latest_issue_not_regressed(output_path, stats.latest_issue)

    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    df.to_csv(temp_output_path, index=False, encoding="utf-8")
    temp_output_path.replace(output_path)
    csv_sha256 = _sha256_file(output_path)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    meta = DownloadResult(
        code=cfg.code,
        total_issues=stats.valid_rows,
        saved_path=str(output_path),
        timestamp=fetched_at_utc,
        source_url=source_url,
        raw_rows=stats.raw_rows,
        valid_rows=stats.valid_rows,
        rejected_rows=stats.rejected_rows,
        earliest_issue=stats.earliest_issue,
        latest_issue=stats.latest_issue,
        csv_sha256=csv_sha256,
        rejection_reasons=stats.rejection_reasons,
    )
    metadata: dict[str, object] = asdict(meta)
    metadata.update(
        {
            # 元信息与 CSV 同目录，避免持久化机器相关的绝对路径。
            "saved_path": DATA_FILE_NAME,
            "csv_file": DATA_FILE_NAME,
            "fetched_at_utc": fetched_at_utc,
            "record_count": stats.valid_rows,
            "issue_range": {
                "earliest": stats.earliest_issue,
                "latest": stats.latest_issue,
            },
        }
    )
    _write_json_atomic(output_path.parent / "download_meta.json", metadata)
    logger.success(
        "数据下载完成：原始 {} 行 / 有效 {} 行 / 拒绝 {} 行，期号 {}—{}，保存到 {}",
        meta.raw_rows,
        meta.valid_rows,
        meta.rejected_rows,
        meta.earliest_issue,
        meta.latest_issue,
        output_path,
    )
    return meta


def load_history(code: str) -> pd.DataFrame:
    """从本地 CSV 加载并重新验证快乐 8 历史数据。"""

    cfg = LOTTERY_CONFIGS[code]
    path = PATHS["data"] / cfg.code / DATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"未找到 {cfg.name} 历史数据文件，请先执行下载：{path}")
    df = pd.read_csv(path, encoding="utf-8")
    number_columns = [f"红球_{index}" for index in range(1, 21)]
    required = ["期数", *number_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少字段{missing}，可能是损坏文件")
    validated = df[required].apply(pd.to_numeric, errors="raise")
    if validated.isna().any().any():
        raise ValueError(f"{path} 包含缺失值，拒绝用于分析")
    if validated["期数"].duplicated().any():
        raise ValueError(f"{path} 包含重复期号，拒绝用于分析")
    for row_index, values in enumerate(validated[number_columns].to_numpy(), start=2):
        try:
            _extract_valid_numbers(values, expected_count=20)
        except ValueError as exc:
            raise ValueError(f"{path} 第{row_index}行开奖号码非法：{exc}") from exc
    return validated


__all__ = [
    "DataRegressionError",
    "DownloadResult",
    "LotteryHttpClient",
    "ParseStats",
    "download_history",
    "get_current_issue",
    "load_history",
]
