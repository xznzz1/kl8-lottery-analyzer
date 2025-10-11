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


def _parse_issue_list(config: LotteryModelConfig, html: str) -> pd.DataFrame:
    """解析快乐 8 历史页面，返回包含 20 个球位的 DataFrame。"""

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
        numbers = [
            td.get_text(strip=True)
            for td in tds
            if td.get_text(strip=True).isdigit()
        ]
        if len(numbers) < config.red.sequence_len:
            continue
        record = {"期数": issue}
        for idx, value in enumerate(numbers[: config.red.sequence_len]):
            record[f"红球_{idx + 1}"] = value
        rows.append(record)

    if not rows:
        raise ValueError("解析开奖号码失败，未获取到有效数据")
    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)


def _parse_kl8_sequence(text: str) -> pd.DataFrame:
    """解析 917500 顺序文本为 DataFrame。"""

    rows = []
    for line in sorted(text.splitlines(), reverse=True):
        if not line or "," not in line:
            continue
        first_segment = line.split(",")[0]
        parts = [item for item in first_segment.split(" ") if item]
        if len(parts) < 21:
            continue
        issue = parts[0]
        record = {"期数": issue}
        for idx in range(1, 21):
            record[f"红球_{idx}"] = parts[idx]
        rows.append(record)
    if not rows:
        raise ValueError("快乐 8 出球顺序数据解析失败")
    df = pd.DataFrame(rows)
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
