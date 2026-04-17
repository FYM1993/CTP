from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_pipeline import select_top_candidates  # noqa: E402
import phase1_pipeline  # noqa: E402
import daily_workflow  # noqa: E402
import pandas as pd  # noqa: E402


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


def test_select_top_candidates_respects_symbol_specific_threshold() -> None:
    rows = [
        {
            "symbol": "LH0",
            "reversal_score": 62.0,
            "trend_score": 15.0,
            "attention_score": 62.0,
            "entry_threshold": 65.0,
            "data_coverage": 0.8,
        },
        {
            "symbol": "M0",
            "reversal_score": 59.0,
            "trend_score": 18.0,
            "attention_score": 59.0,
            "entry_threshold": 58.0,
            "data_coverage": 0.8,
        },
    ]

    selected = select_top_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["M0"]


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
        lambda: {"fundamental_screening": {"top_n": 10, "default_threshold": 10}},
    )
    monkeypatch.setattr(
        daily_workflow,
        "get_all_symbols",
        lambda: [{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_symbol_legacy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("旧 legacy 引擎不应再被调用")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_symbol_regime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("旧 regime 引擎不应再被调用")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_phase1_pipeline",
        lambda *, all_data, symbols, threshold, config: [
            {
                "symbol": "LH0",
                "name": "生猪",
                "exchange": "dce",
                "price": 14000.0,
                "score": 66.0,
                "fund_score": 66.0,
                "fund_details": "库存下降",
                "range_pct": 3.0,
                "attention_score": 66.0,
                "reversal_score": 66.0,
                "trend_score": 31.0,
                "labels": ["反转候选"],
                "state_labels": ["低位出清"],
                "data_coverage": 0.75,
                "reason_summary": "库存下降",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        raising=False,
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
    assert row["trend_score"] == 31.0
    assert row["labels"] == ["反转候选"]
    assert row["state_labels"] == ["低位出清"]
    assert row["data_coverage"] == 0.75
    assert row["reason_summary"] == "库存下降"


def test_run_phase1_pipeline_builds_reversal_candidate_from_distress(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [20000 - i * 50 for i in range(119)] + [14100]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 100 for c in closes],
            "low": [c - 100 for c in closes],
            "close": closes,
            "volume": [1000] * 120,
            "oi": [20000] * 120,
        }
    )
    monkeypatch.setattr(phase1_pipeline, "get_inventory", lambda symbol: None)
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame: None)
    monkeypatch.setattr(
        phase1_pipeline,
        "get_hog_fundamentals",
        lambda: {
            "price": 11.8,
            "price_5d_ago": 11.2,
            "price_trend": 5.4,
            "cost": 1750.0,
            "profit_margin": -22.0,
            "data_status": "full",
        },
    )

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"LH0": df},
        symbols=[{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5}},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"] == ["反转候选"]
    assert "低位出清" in row["state_labels"]
    assert row["reversal_score"] > row["trend_score"]
    assert row["attention_score"] >= 55.0


def test_run_phase1_pipeline_builds_trend_candidate_from_tightening(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    closes = [7000 + i * 4 for i in range(180)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 40 for c in closes],
            "low": [c - 40 for c in closes],
            "close": closes,
            "volume": [1000] * 180,
            "oi": [15000] * 180,
        }
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_inventory",
        lambda symbol: {
            "inv_now": 1200.0,
            "inv_change_4wk": -18.0,
            "inv_cumulating_weeks": 1,
            "inv_percentile": 18.0,
            "inv_trend": "去库",
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_warehouse_receipt",
        lambda symbol: {
            "receipt_total": 1000.0,
            "receipt_change": -120.0,
            "exchange": "SHFE",
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_seasonality",
        lambda frame: {
            "month": 4,
            "hist_avg_return": 3.8,
            "hist_up_pct": 68.0,
            "current_month_return": 1.2,
            "seasonal_signal": 0.8,
        },
    )
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda: None)

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"M0": df},
        symbols=[{"symbol": "M0", "name": "豆粕", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5, "default_threshold": 3}},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"] == ["趋势候选"]
    assert "紧平衡强化" in row["state_labels"]
    assert row["trend_score"] > row["reversal_score"]
    assert row["attention_score"] >= 55.0


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
