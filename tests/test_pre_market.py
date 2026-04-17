from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_market  # noqa: E402


def _build_daily_frame(rows: int = 130) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    base = pd.Series(range(rows), dtype="float64")
    open_ = 100.0 + base * 0.1
    close = open_ + 0.2
    high = close + 1.0
    low = open_ - 1.0

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0 + base * 5.0,
            "oi": 5000.0 + base * 10.0,
        }
    )


def test_build_trade_plan_from_daily_df_returns_long_plan(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-05-10",
            "signal_bar": {"low": 98.0, "high": 104.0},
            "signal_detail": "synthetic long signal",
            "current_stage": "反转确认",
            "next_expected": "信号新鲜(今日)，可考虑入场",
            "confidence": 0.8,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 118.0]))
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": 25.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={"ma_windows": [5, 10, 20]},
    )

    assert plan is not None
    assert plan["direction"] == "long"
    assert plan["entry"] > 0
    assert plan["stop"] > 0
    assert plan["tp1"] > plan["entry"]
    assert plan["tp2"] >= plan["tp1"]


def test_build_trade_plan_from_daily_df_returns_none_for_empty_df() -> None:
    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=pd.DataFrame(),
        cfg={},
    )

    assert plan is None
