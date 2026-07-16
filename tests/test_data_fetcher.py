# -*- coding: utf-8 -*-
"""快乐 8 数据抓取、严格解析及元信息回归测试。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import requests

from src import data_fetcher


def _history_row(
    issue: str,
    selected: set[int] | None = None,
    value_overrides: dict[int, str] | None = None,
    omit_last_cell: bool = False,
) -> str:
    """构造与实时 500.com 页面一致的 81-cell 开奖行。"""

    selected = selected or set(range(1, 21))
    value_overrides = value_overrides or {}
    cells = [f"<td align='center'>{issue}</td>"]
    for number in range(1, 81):
        if omit_last_cell and number == 80:
            continue
        if number in selected:
            value = value_overrides.get(number, str(number))
            cells.append(f"<td class='chartBall01'>{value}</td>")
        else:
            cells.append("<td class='yl01'>1</td>")
    return f"<tr>{''.join(cells)}</tr>"


def _history_html(*rows: str) -> str:
    return f"<div class='wrap_datachart'><table><tbody id='tdata'>{''.join(rows)}</tbody></table></div>"


@pytest.fixture
def valid_with_separator_html() -> str:
    """包含一个有效 chartBall01 行和一个真实形态空白分隔行。"""

    return _history_html(
        _history_row("2025002"),
        "<tr><td colspan='81'></td></tr>",
    )


@pytest.fixture
def malformed_html() -> str:
    """包含有效行和缺少最后一个号码列的畸形行。"""

    return _history_html(
        _history_row("2025002"),
        _history_row("2025001", omit_last_cell=True),
    )


@pytest.fixture
def duplicate_number_html() -> str:
    """第 2 列错误地重复第 1 列开奖号码。"""

    return _history_html(_history_row("2025001", value_overrides={2: "1"}))


@pytest.fixture
def out_of_range_html() -> str:
    """第 20 个开奖号码越过快乐 8 的 1—80 边界。"""

    return _history_html(_history_row("2025001", value_overrides={20: "81"}))


@pytest.fixture
def multi_year_html() -> str:
    """最小多年份页面，模拟全量页跨年且含分隔行的结构。"""

    return _history_html(
        _history_row("2021313", set(range(1, 21))),
        "<tr><td colspan='81'></td></tr>",
        _history_row("2026186", set(range(61, 81))),
    )


@pytest.fixture
def sample_sequence_text() -> str:
    nums = " ".join(f"{i:02d}" for i in range(1, 21))
    return f"2025002 2025-01-02 {nums},其他内容\n2025001 {nums},其他内容"


def _redirect_data_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    monkeypatch.setattr(data_fetcher, "PATHS", {"data": data_root})
    monkeypatch.setattr(data_fetcher, "ensure_runtime_directories", lambda: None)
    return data_root / "kl8"


def test_http_client_rejects_unknown_or_lookalike_domain() -> None:
    client = data_fetcher.LotteryHttpClient(
        timeout=1,
        retries=1,
        backoff_factor=0.1,
        user_agent="test",
    )
    with pytest.raises(ValueError, match="禁止访问域名"):
        client.get_text("https://example.com/data")
    with pytest.raises(ValueError, match="禁止访问域名"):
        client.get_text("https://datachart.500.com.attacker.example/data")


def test_http_client_fetches_allowed_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    client = data_fetcher.LotteryHttpClient(
        timeout=1,
        retries=1,
        backoff_factor=0.1,
        user_agent="test",
    )

    class DummyResponse:
        def __init__(self) -> None:
            self.encoding = None
            self.text = "<html>ok</html>"

        def raise_for_status(self) -> None:
            return None

    dummy = DummyResponse()

    def fake_get(self, url, headers, timeout):
        assert "datachart.500.com" in url
        return dummy

    monkeypatch.setattr(requests.Session, "get", fake_get, raising=True)
    text = client.get_text("https://datachart.500.com/kl8/")
    assert dummy.encoding == "utf-8"
    assert text == dummy.text


def test_valid_chartball_row_and_blank_separator_are_counted(
    valid_with_separator_html: str,
) -> None:
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    df, stats = data_fetcher._parse_issue_list_with_stats(
        cfg, valid_with_separator_html
    )

    assert list(df.columns) == ["期数"] + [f"红球_{i}" for i in range(1, 21)]
    assert df.iloc[0]["期数"] == "2025002"
    assert list(df.iloc[0, 1:].astype(int)) == list(range(1, 21))
    assert (stats.raw_rows, stats.valid_rows, stats.rejected_rows) == (2, 1, 1)
    assert stats.rejection_reasons == {"blank_separator": 1}


def test_malformed_row_is_rejected_but_valid_row_survives(malformed_html: str) -> None:
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    df, stats = data_fetcher._parse_issue_list_with_stats(cfg, malformed_html)

    assert list(df["期数"]) == ["2025002"]
    assert stats.rejected_rows == 1
    assert stats.rejection_reasons == {"malformed_cell_count": 1}


@pytest.mark.parametrize(
    ("fixture_name", "reason"),
    [
        ("duplicate_number_html", "duplicate_numbers"),
        ("out_of_range_html", "out_of_range_numbers"),
    ],
)
def test_invalid_number_rows_fail_clearly(
    fixture_name: str,
    reason: str,
    request: pytest.FixtureRequest,
) -> None:
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    html = request.getfixturevalue(fixture_name)
    with pytest.raises(ValueError, match=reason):
        data_fetcher._parse_issue_list_with_stats(cfg, html)


def test_multi_year_page_tracks_full_issue_range(multi_year_html: str) -> None:
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    df, stats = data_fetcher._parse_issue_list_with_stats(cfg, multi_year_html)

    assert list(df["期数"]) == ["2026186", "2021313"]
    assert stats.raw_rows == 3
    assert stats.valid_rows == 2
    assert stats.rejected_rows == 1
    assert stats.earliest_issue == "2021313"
    assert stats.latest_issue == "2026186"


def test_empty_parse_result_fails_explicitly() -> None:
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    html = _history_html("<tr><td colspan='81'></td></tr>")
    with pytest.raises(ValueError, match="未获取到有效数据"):
        data_fetcher._parse_issue_list_with_stats(cfg, html)


def test_parse_sequence_supports_dated_and_legacy_rows(
    sample_sequence_text: str,
) -> None:
    df, stats = data_fetcher._parse_kl8_sequence_with_stats(sample_sequence_text)
    assert list(df["期数"]) == ["2025002", "2025001"]
    assert df.iloc[0]["红球_1"] == "01"
    assert len(df.columns) == 21
    assert stats.valid_rows == 2


def test_history_url_uses_live_endpoint_and_query_names() -> None:
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    url = data_fetcher._build_history_url(cfg, 2025001, 2025351)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.path == "/kl8/zoushi/newinc/jbzs_redblue.php"
    assert query == {
        "from": ["2025001"],
        "to": ["2025351"],
        "shujcount": ["0"],
        "sort": ["0"],
    }


def test_get_current_issue_uses_current_chart_page() -> None:
    class FakeClient:
        def get_text(self, url: str) -> str:
            assert url == "https://datachart.500.com/kl8/"
            return "<div class='wrap_datachart'><input id='to' value='2026186'></div>"

    assert data_fetcher.get_current_issue("kl8", client=FakeClient()) == "2026186"


def test_download_writes_quality_metadata_and_csv_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    multi_year_html: str,
) -> None:
    output_dir = _redirect_data_path(monkeypatch, tmp_path)

    class FakeClient:
        def get_text(self, url: str) -> str:
            assert "/kl8/zoushi/newinc/jbzs_redblue.php" in url
            return multi_year_html

    result = data_fetcher.download_history(
        "kl8",
        start=2021313,
        end=2026186,
        client=FakeClient(),
    )

    csv_path = Path(result.saved_path)
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = json.loads((output_dir / "download_meta.json").read_text(encoding="utf-8"))

    assert result.total_issues == 2
    assert (result.raw_rows, result.valid_rows, result.rejected_rows) == (3, 2, 1)
    assert result.earliest_issue == "2021313"
    assert result.latest_issue == "2026186"
    assert result.csv_sha256 == csv_digest
    assert meta["source_url"] == result.source_url
    assert meta["saved_path"] == data_fetcher.DATA_FILE_NAME
    assert meta["csv_file"] == data_fetcher.DATA_FILE_NAME
    assert meta["fetched_at_utc"] == result.timestamp
    assert (
        datetime.fromisoformat(meta["fetched_at_utc"]).utcoffset().total_seconds() == 0
    )
    assert meta["issue_range"] == {"earliest": "2021313", "latest": "2026186"}
    assert meta["record_count"] == 2
    assert meta["csv_sha256"] == csv_digest
    assert meta["rejection_reasons"] == {"blank_separator": 1}
    assert len(pd.read_csv(csv_path)) == 2


def test_latest_issue_regression_refuses_to_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = _redirect_data_path(monkeypatch, tmp_path)
    output_dir.mkdir(parents=True)
    csv_path = output_dir / data_fetcher.DATA_FILE_NAME
    csv_path.write_text("期数,红球_1\n2025002,1\n", encoding="utf-8")
    before = csv_path.read_bytes()

    class FakeClient:
        def get_text(self, url: str) -> str:
            return _history_html(_history_row("2025001"))

    with pytest.raises(data_fetcher.DataRegressionError, match="最新期号倒退"):
        data_fetcher.download_history("kl8", client=FakeClient())
    assert csv_path.read_bytes() == before
    assert not (output_dir / "download_meta.json").exists()


def test_load_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = _redirect_data_path(monkeypatch, tmp_path)
    output_dir.mkdir(parents=True)
    path = output_dir / data_fetcher.DATA_FILE_NAME
    columns = ["期数", *(f"红球_{index}" for index in range(1, 21))]
    values = ["2025001", *(str(index) for index in range(1, 21))]
    path.write_text(
        ",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8"
    )

    loaded = data_fetcher.load_history("kl8")
    assert list(loaded["期数"]) == [2025001]


def test_load_history_rejects_incomplete_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = _redirect_data_path(monkeypatch, tmp_path)
    output_dir.mkdir(parents=True)
    path = output_dir / data_fetcher.DATA_FILE_NAME
    path.write_text("期数,红球_1\n2025001,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少字段"):
        data_fetcher.load_history("kl8")
