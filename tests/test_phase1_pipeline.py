from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_pipeline import select_top_candidates  # noqa: E402
import daily_workflow  # noqa: E402


def test_select_top_candidates_orders_by_attention_only() -> None:
    rows = [
        {
            "symbol": "LH0",
            "reversal_score": 60.0,
            "trend_score": 36.0,
            "attention_score": 76.0,
        },
        {
            "symbol": "RM0",
            "reversal_score": 82.0,
            "trend_score": 55.0,
            "attention_score": 73.0,
        },
        {
            "symbol": "A0",
            "reversal_score": 70.0,
            "trend_score": 57.0,
            "attention_score": 74.0,
        },
    ]

    selected = select_top_candidates(rows, top_n=2)

    assert [row["symbol"] for row in selected] == ["LH0", "A0"]


def test_select_top_candidates_excludes_rows_below_threshold() -> None:
    rows = [
        {
            "symbol": "LOW0",
            "reversal_score": 54.0,
            "trend_score": 54.0,
            "attention_score": 99.0,
        },
        {
            "symbol": "KEEP0",
            "reversal_score": 55.0,
            "trend_score": 20.0,
            "attention_score": 10.0,
        },
    ]

    selected = select_top_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["KEEP0"]


def test_phase1_output_has_attention_not_direction() -> None:
    candidate = {
        "symbol": "LH0",
        "name": "生猪",
        "reversal_score": 88.0,
        "trend_score": 35.0,
        "attention_score": 76.0,
        "labels": ["反转候选"],
    }

    assert "direction" not in candidate
    assert candidate["attention_score"] == 76.0


def test_phase_1_screen_adds_bridge_fields_without_direction(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "load_config",
        lambda: {"fundamental_screening": {"p1_engine": "legacy", "default_threshold": 10}},
    )
    monkeypatch.setattr(
        daily_workflow,
        "get_all_symbols",
        lambda: [{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_symbol_legacy",
        lambda symbol, name, exchange, df, threshold: {
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
            "score": -66.0,
            "range_pct": 3.0,
            "fund_details": "库存下降",
        },
    )

    rows = daily_workflow.phase_1_screen(
        {"LH0": [None] * 60},
        threshold=10,
    )
    captured = capsys.readouterr()

    assert len(rows) == 1
    row = rows[0]
    assert captured.out == ""
    assert "direction" not in row
    assert row["attention_score"] == 66.0
    assert row["reversal_score"] == 66.0
    assert row["trend_score"] == 66.0
    assert row["labels"] == ["双标签候选"]
    assert row["state_labels"] == []
    assert row["data_coverage"] == 1.0
    assert row["reason_summary"] == "库存下降"


def test_phase_2_premarket_uses_resolved_direction_before_analyze(monkeypatch) -> None:
    calls: list[dict] = []
    resolve_calls: list[dict] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"trend": -18.0, "volume": 6.0, "wyckoff": -8.0},
    )

    def fake_resolve_phase2_direction(**kwargs):
        resolve_calls.append(kwargs)
        return "long"

    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", fake_resolve_phase2_direction)

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
        calls.append(
            {
                "symbol": symbol,
                "name": name,
                "direction": direction,
                "cfg": cfg,
            }
        )
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "price": 100.0,
            "score": -22.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 105.0,
            "tp1": 90.0,
            "rr": 1.2,
            "reversal_status": {
                "signal_strength": 0.4,
                "next_expected": "等待确认",
            },
        }

    monkeypatch.setattr(daily_workflow, "analyze_one", fake_analyze_one)

    actionable, watchlist = daily_workflow.phase_2_premarket(
        [
            {
                "symbol": "LH0",
                "name": "生猪",
                "score": 66.0,
                "attention_score": 66.0,
                "labels": ["趋势候选"],
                "entry_pool_reason": "配置桥接",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert resolve_calls == [{"long_score": 6.0, "short_score": 26.0, "delta": 12.0}]
    assert calls[0]["direction"] == "long"
    assert watchlist[0]["direction"] == "long"
    assert watchlist[0]["phase2_decision"] == "long"


def test_phase_2_premarket_falls_back_to_stronger_side_when_resolver_watches(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"trend": -18.0, "volume": 6.0, "wyckoff": -8.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "watch")

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
        calls.append({"symbol": symbol, "direction": direction})
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "price": 100.0,
            "score": -22.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 105.0,
            "tp1": 90.0,
            "rr": 1.2,
            "reversal_status": {
                "signal_strength": 0.4,
                "next_expected": "等待确认",
            },
        }

    monkeypatch.setattr(daily_workflow, "analyze_one", fake_analyze_one)

    actionable, watchlist = daily_workflow.phase_2_premarket(
        [
            {
                "symbol": "LH0",
                "name": "生猪",
                "score": 66.0,
                "attention_score": 66.0,
                "labels": ["趋势候选"],
                "entry_pool_reason": "配置桥接",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert calls[0]["direction"] == "short"
    assert watchlist[0]["direction"] == "short"
    assert watchlist[0]["phase2_decision"] == "watch"
