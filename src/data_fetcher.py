# -*- coding: utf-8 -*-
"""
快乐 8 历史数据抓取与加载工具（精简版）。

相较于原仓库，该版本仅保留快乐 8（kl8）相关逻辑，负责：
1. 带重试的 HTTP 抓取；
2. HTML / 文本解析为 pandas.DataFrame；
3. 将数据保存到 `data/kl8/data.csv` 并生成下载元信息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

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


@dataclass
class DownloadResult:
    """记录一次快乐 8 历史数据下载的结果。"""

    code: str
    total_issues: int
    saved_path: str
    timestamp: str


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
        domain = parsed.netloc.lower()
        if all(allowed not in domain for allowed in ALLOWED_DOMAINS):
            raise ValueError(f"禁止访问域名：{domain}")
        response = self._session.get(url, headers=self._headers, timeout=self._timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text


def _build_history_url(config: LotteryModelConfig, start: Optional[int], end: Optional[int]) -> str:
    """构造快乐 8 历史记录页面地址。"""

    base = f"https://datachart.500.com/{config.code}/history/"
    start_issue = start or 1
    end_issue = end or 999_999
    limit = end_issue - start_issue + 1
    return f"{base}newinc/jbzs_redblue.php?start={start_issue}&end={end_issue}&limit={limit}"


def _extract_valid_numbers(values: Iterable[str], expected_count: int = 20) -> list[str]:
    """从原始字段中提取1—80的号码，并校验数量和唯一性。"""

    numbers: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value.isdigit():
            continue

        number = int(value)
        if 1 <= number <= 80:
            numbers.append(str(number))

    if len(numbers) != expected_count:
        raise ValueError(
            f"开奖号码数量异常：期望{expected_count}个，实际{len(numbers)}个，内容={numbers}"
        )

    if len(set(numbers)) != expected_count:
        raise ValueError(f"开奖号码存在重复：{numbers}")

    return numbers

def _parse_issue_list(config: LotteryModelConfig, html: str) -> pd.DataFrame:
    """解析快乐8网页历史数据。"""

    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody", attrs={"id": "tdata"})
    if not tbody:
        raise ValueError("未找到开奖号码数据表格 (id=tdata)")

    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        issue = tds[0].get_text(strip=True)
        if not issue or not issue.isdigit():
            continue

        raw_values = [td.get_text(strip=True) for td in tds[1:]]
        try:
            numbers = _extract_valid_numbers(
                raw_values,
                expected_count=config.red.sequence_len,
            )
        except ValueError:
            continue

        record = {"期数": issue}
        for idx, value in enumerate(numbers, start=1):
            record[f"红球_{idx}"] = value
        rows.append(record)

    if not rows:
        raise ValueError("解析开奖号码失败，未获取到有效数据")

    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)


def _parse_kl8_sequence(text: str) -> pd.DataFrame:
    """解析917500顺序数据，并兼容不带日期的测试/旧数据。

    支持两种格式：
    1. 期号 日期 20个开奖号码,销售额及其他统计信息
    2. 期号 20个开奖号码,其他内容
    """

    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue

        first_segment = line.split(",", 1)[0]
        parts = first_segment.split()

        if len(parts) < 21:
            continue

        issue = parts[0].strip()
        if not issue.isdigit():
            continue

        # 真实数据第二项为YYYY-MM-DD；旧数据或测试样例没有日期。
        second = parts[1] if len(parts) > 1 else ""
        has_date = (
            len(second) == 10
            and second[4:5] == "-"
            and second[7:8] == "-"
            and second.replace("-", "").isdigit()
        )

        number_start = 2 if has_date else 1
        number_tokens = parts[number_start:number_start + 20]

        if len(number_tokens) != 20:
            continue

        if not all(
            token.isdigit() and 1 <= int(token) <= 80
            for token in number_tokens
        ):
            continue

        if len({int(token) for token in number_tokens}) != 20:
            continue

        # 保留两位号码格式，如01、02；读取CSV时仍可转成整数。
        numbers = [f"{int(token):02d}" for token in number_tokens]

        record = {"期数": issue}
        for index, value in enumerate(numbers, start=1):
            record[f"红球_{index}"] = value

        rows.append(record)

    if not rows:
        raise ValueError("快乐8出球顺序数据解析失败")

    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)


def get_current_issue(code: str, client: Optional[LotteryHttpClient] = None) -> str:
    """查询快乐 8 最新期号。"""

    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    url = f"https://datachart.500.com/{cfg.code}/history/newinc/jbzs_redblue.php"
    html = client.get_text(url)
    soup = BeautifulSoup(html, "lxml")
    wrap = soup.find("div", class_="wrap_datachart")
    if not wrap:
        raise ValueError("未找到数据页面主体 (div.wrap_datachart)")
    input_tag = wrap.find("input", {"id": "to"})
    if not input_tag or not input_tag.has_attr("value"):
        raise ValueError("未从页面提取到最新期号")
    value = input_tag["value"]
    logger.info("【{}】最新期号：{}", cfg.name, value)
    return value


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
        logger.info("下载快乐 8 出球顺序数据...")
        text = client.get_text("https://data.917500.cn/kl81000_cq_asc.txt")
        df = _parse_kl8_sequence(text)
    else:
        url = _build_history_url(cfg, start, end)
        logger.info("下载快乐 8 历史数据：{}", url)
        html = client.get_text(url)
        df = _parse_issue_list(cfg, html)

    save_dir = PATHS["data"] / cfg.code
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / DATA_FILE_NAME
    df.to_csv(output_path, index=False, encoding="utf-8")
    meta = DownloadResult(
        code=cfg.code,
        total_issues=len(df),
        saved_path=str(output_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    (output_path.parent / "download_meta.json").write_text(
        json.dumps(meta.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success("数据下载完成，共 {} 期，保存到 {}", meta.total_issues, output_path)
    return meta


def load_history(code: str) -> pd.DataFrame:
    """从本地 CSV 加载快乐 8 历史数据。"""

    cfg = LOTTERY_CONFIGS[code]
    path = PATHS["data"] / cfg.code / DATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"未找到 {cfg.name} 历史数据文件，请先执行下载：{path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "期数" not in df.columns:
        raise ValueError(f"{path} 缺少【期数】字段，可能是损坏文件")
    return df


__all__ = [
    "DownloadResult",
    "LotteryHttpClient",
    "download_history",
    "get_current_issue",
    "load_history",
]
