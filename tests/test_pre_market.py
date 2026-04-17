from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_analyze_one_restores_phase2_section_logging(monkeypatch) -> None:
    df = _build_daily_frame()
    messages: list[str] = []

    def capture_info(message, *args, **kwargs):
        if args:
            message = message % args
        messages.append(str(message))

    monkeypatch.setattr(pre_market.log, "info", capture_info)
    monkeypatch.setattr(pre_market, "fetch_data", lambda *args, **kwargs: df)
    monkeypatch.setattr(
        pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: {
            "score": 25.0,
            "actionable": True,
            "entry": float(df["close"].iloc[-1]),
            "stop": 95.0,
            "tp1": 112.0,
            "tp2": 118.0,
            "rr": 1.5,
            "support_levels": [92.0, 90.0],
            "resistance_levels": [112.0, 118.0],
            "fib_targets": {
                "回撤23.6%": 101.0,
                "回撤38.2%": 104.0,
                "回撤50.0%": 106.0,
                "回撤61.8%": 108.0,
                "回撤78.6%": 110.0,
                "扩展1.0": 112.0,
                "扩展1.272": 114.0,
                "扩展1.618": 116.0,
            },
            "scores": {
                "均线排列": 10.0,
                "MACD": 5.0,
                "RSI": 0.0,
                "布林带": 0.0,
                "动量": 2.0,
                "价格位置": 0.0,
                "Wyckoff阶段": 8.0,
                "量价关系": 5.0,
                "VSA信号": 0.0,
                "持仓信号": 0.0,
            },
            "reversal_status": {
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
        },
    )
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: SimpleNamespace(phase="markup", confidence=0.8, description="上涨阶段"))
    monkeypatch.setattr(pre_market, "analyze_volume_pattern", lambda *args, **kwargs: {"up_down_ratio": 1.4, "vol_trend": 1.1, "price_vol_corr": 0.2})
    monkeypatch.setattr(
        pre_market,
        "vsa_scan",
        lambda *args, **kwargs: [SimpleNamespace(strength=2, bias="bullish", bar_type="upthrust", description="VSA signal")],
    )
    monkeypatch.setattr(pre_market, "analyze_oi", lambda *args, **kwargs: SimpleNamespace(bias="bullish", label="新多", strength=10, description="OI signal"))
    monkeypatch.setattr(pre_market, "oi_divergence", lambda *args, **kwargs: "OI divergence")
    monkeypatch.setattr(pre_market, "detect_climax", lambda *args, **kwargs: [])
    monkeypatch.setattr(pre_market, "detect_spring_upthrust", lambda *args, **kwargs: [])
    monkeypatch.setattr(pre_market, "detect_sos_sow", lambda *args, **kwargs: [])

    result = pre_market.analyze_one("LH0", "生猪", "long", {"reason": "test", "ma_windows": [5], "sr_lookback": 120})

    assert result is not None
    assert any("经典技术指标" in msg for msg in messages)
    assert any("Wyckoff 量价分析" in msg for msg in messages)
    assert any("持仓量(OI)分析" in msg for msg in messages)
    assert any("关键价位" in msg for msg in messages)
    assert any("斐波那契目标位" in msg for msg in messages)
    assert any("综合评分" in msg for msg in messages)
