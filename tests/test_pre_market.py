from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase2 import pre_market  # noqa: E402
import wyckoff as wyckoff_module  # noqa: E402


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


def test_score_signals_keeps_finite_score_when_long_ma_window_exceeds_history() -> None:
    df = _build_daily_frame(rows=71)

    scores = pre_market.score_signals(df, direction="short", cfg={})

    assert np.isfinite(float(scores["均线排列"]))
    assert np.isfinite(sum(float(value) for value in scores.values()))


def test_build_trade_plan_from_daily_df_returns_long_plan(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-05-10",
            "signal_days_ago": 0,
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "synthetic long signal",
            "current_stage": "反转确认",
            "next_expected": "信号新鲜(今日)，可考虑入场",
            "confidence": 0.8,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
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
    assert plan["symbol"] == "LH0"
    assert plan["name"] == "生猪"
    assert plan["direction"] == "long"
    assert plan["score"] == 25.0
    assert bool(plan["actionable"]) is True
    assert plan["price"] == pytest.approx(113.1)
    assert plan["entry"] == pytest.approx(113.1)
    assert plan["stop"] == pytest.approx(109.0)
    assert plan["tp1"] == pytest.approx(118.0)
    assert plan["tp2"] == pytest.approx(120.36)
    assert plan["rr"] == pytest.approx(1.1951219512)
    assert plan["reversal_status"]["signal_type"] == "SOS"
    assert plan["wyckoff_phase"] == "markup"
    assert plan["scores"] == {"dummy": 25.0}
    assert plan["support_levels"] == [92.0]
    assert plan["resistance_levels"] == [112.0, 118.0]


def test_build_trade_plan_from_daily_df_allows_fresh_reversal_signal_with_near_neutral_score(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "Spring",
            "signal_date": "2025-05-10",
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "fresh spring",
            "current_stage": "反转信号出现",
            "next_expected": "信号新鲜(今日)，可考虑入场",
            "confidence": 0.7,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 118.0]))
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": -3.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={"ma_windows": [5, 10, 20]},
    )

    assert plan is not None
    assert plan["entry_family"] == "reversal"
    assert plan["entry_signal_type"] == "Spring"
    assert bool(plan["actionable"]) is True
    assert plan["score"] == -3.0
    assert plan["rr"] == pytest.approx(1.1951219512)


def test_build_trade_plan_from_daily_df_allows_fresh_reversal_signal_with_moderate_countertrend_score(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "Spring",
            "signal_date": "2025-05-10",
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "fresh spring but still countertrend",
            "current_stage": "反转信号出现",
            "next_expected": "信号新鲜(今日)，可考虑入场",
            "confidence": 0.7,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 118.0]))
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": -12.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={"ma_windows": [5, 10, 20]},
    )

    assert plan is not None
    assert plan["entry_family"] == "reversal"
    assert plan["entry_signal_type"] == "Spring"
    assert bool(plan["actionable"]) is True
    assert plan["score"] == -12.0
    assert plan["rr"] == pytest.approx(1.1951219512)


def test_build_trade_plan_from_daily_df_allows_reversal_entry_when_tp1_rr_is_below_one_but_farther_target_is_sufficient(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-05-10",
            "signal_days_ago": 0,
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "fresh breakout with near resistance",
            "current_stage": "反转确认",
            "next_expected": "信号新鲜(今日)，可考虑入场",
            "confidence": 0.8,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 116.0]))
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
    assert plan["entry_family"] == "reversal"
    assert bool(plan["actionable"]) is True
    assert plan["tp1"] == pytest.approx(116.0)
    assert plan["rr"] == pytest.approx(0.7073170731)
    assert plan["admission_rr"] == pytest.approx(1.2731707317)
    assert plan["admission_rr"] > plan["rr"]
    assert plan["phase2_rr_gate_passed"] is True


def test_build_trade_plan_from_daily_df_allows_fresh_reversal_signal_on_second_follow_through_day(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "Spring",
            "signal_date": "2025-05-08",
            "signal_days_ago": 2,
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "spring recovered after two days",
            "current_stage": "反转信号出现",
            "next_expected": "信号新鲜(2天前)，可考虑入场",
            "confidence": 0.7,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 118.0]))
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": 12.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={"ma_windows": [5, 10, 20]},
    )

    assert plan is not None
    assert plan["entry_family"] == "reversal"
    assert plan["entry_signal_type"] == "Spring"
    assert bool(plan["actionable"]) is True
    assert plan["score"] == 12.0
    assert plan["rr"] == pytest.approx(1.1951219512)


def test_assess_reversal_status_marks_three_day_old_entry_signal_as_stale(monkeypatch) -> None:
    df = _build_daily_frame(rows=5)
    df["date"] = pd.to_datetime(
        ["2025-05-07", "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11"]
    )

    monkeypatch.setattr(
        wyckoff_module,
        "_find_events_with_bars",
        lambda *_args, **_kwargs: [
            {
                "signal": "Spring",
                "priority": 2,
                "date": "2025-05-08",
                "bar": {"low": 98.0, "high": 102.0},
                "detail": "old spring",
            }
        ],
    )

    status = wyckoff_module.assess_reversal_status(df, "long", lookback=60)

    assert status["has_signal"] is False
    assert status["current_stage"] == "有信号但已过入场窗口"
    assert status["next_expected"] == "等新的Spring或SOS出现"


def test_build_reversal_trade_plan_from_daily_df_exposes_fresh_signal_gate_diagnostics(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "Spring",
            "signal_date": "2025-05-08",
            "signal_days_ago": 2,
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "fresh spring but score still weak",
            "current_stage": "反转信号出现",
            "next_expected": "信号新鲜(2天前)，可考虑入场",
            "confidence": 0.7,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 118.0]))
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": -22.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())

    plan = pre_market.build_reversal_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={"ma_windows": [5, 10, 20]},
        allow_watch_plan=True,
    )

    assert plan is not None
    assert bool(plan["actionable"]) is False
    assert plan["reversal_signal_fresh"] is True
    assert plan["phase2_score_gate_passed"] is False
    assert plan["phase2_rr_gate_passed"] is True


def test_build_trade_plan_from_daily_df_returns_short_plan(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOW",
            "signal_date": "2025-05-10",
            "signal_days_ago": 0,
            "signal_bar": {"low": 108.0, "high": 116.0},
            "signal_detail": "synthetic short signal",
            "current_stage": "转弱确认",
            "next_expected": "继续观察",
            "confidence": 0.7,
            "signal_strength": 0.75,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([104.0, 108.0], [118.0]))
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": -25.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="PS0",
        name="多晶硅",
        direction="short",
        df=df,
        cfg={"ma_windows": [5, 10, 20]},
    )

    assert plan is not None
    assert plan["symbol"] == "PS0"
    assert plan["name"] == "多晶硅"
    assert plan["direction"] == "short"
    assert plan["score"] == -25.0
    assert bool(plan["actionable"]) is True
    assert plan["price"] == pytest.approx(113.1)
    assert plan["entry"] == pytest.approx(113.1)
    assert plan["stop"] == pytest.approx(117.0)
    assert plan["tp1"] == pytest.approx(108.0)
    assert plan["tp2"] == pytest.approx(104.0)
    assert plan["rr"] == pytest.approx(1.3076923077)
    assert plan["reversal_status"]["signal_type"] == "SOW"
    assert plan["wyckoff_phase"] == "markdown"


def test_build_trade_plan_from_daily_df_returns_watch_plan_without_signal(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "等待信号",
            "next_expected": "继续观察",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": 5.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "range"})())
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [120.0]))

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert bool(plan["actionable"]) is False
    assert plan["price"] == pytest.approx(113.1)
    assert plan["entry"] == pytest.approx(113.1)
    assert plan["stop"] == 0.0
    assert plan["tp1"] == 0.0
    assert plan["tp2"] == 0.0
    assert plan["rr"] == 0.0
    assert plan["reversal_status"]["has_signal"] is False


def test_build_reversal_trade_plan_from_daily_df_only_returns_reversal_family(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-01-07",
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "放量突破确认",
            "current_stage": "反转确认",
            "next_expected": "继续观察",
            "confidence": 0.82,
            "signal_strength": 0.82,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": 32.0})
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([104.0], [116.0, 120.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "accumulation"})())

    plan = pre_market.build_reversal_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan["strategy_family"] == "reversal_fundamental"
    assert plan["entry_family"] == "reversal"
    assert plan["entry_signal_type"] == "SOS"


def test_build_trend_trade_plan_from_daily_df_only_returns_trend_family(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "下跌中",
            "next_expected": "等待反弹受阻",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": -12.0,
            "MACD": -6.0,
            "RSI": -4.0,
            "布林带": -3.0,
            "动量": -5.0,
            "价格位置": 0.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([104.0, 108.0], [118.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())
    monkeypatch.setattr(
        pre_market,
        "calc_ma",
        lambda series, window: pd.Series(
            [130.0 - window * 0.2 - i * (0.15 if window == 20 else 0.08) for i in range(len(series))],
            index=series.index,
        ),
    )

    plan = pre_market.build_trend_trade_plan_from_daily_df(
        symbol="PS0",
        name="多晶硅",
        direction="short",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan["strategy_family"] == "trend_following"
    assert plan["entry_family"] == "trend"
    assert plan["entry_signal_type"] == "TrendBreak"


def test_build_trade_plan_from_daily_df_returns_trend_long_plan_without_reversal_signal(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "上涨中",
            "next_expected": "等待回踩",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": 12.0,
            "MACD": 6.0,
            "RSI": 4.0,
            "布林带": 3.0,
            "动量": 5.0,
            "价格位置": 0.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [116.0, 120.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())
    monkeypatch.setattr(
        pre_market,
        "calc_ma",
        lambda series, window: pd.Series(
            [
                (
                    106.0 + i * 0.04
                    if window == 5
                    else 105.2 + i * 0.035
                    if window == 10
                    else 104.5 + i * 0.03
                )
                for i in range(len(series))
            ],
            index=series.index,
        ),
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan.get("actionable") is True
    assert plan["reversal_status"]["has_signal"] is False
    assert plan.get("entry_family") == "trend"
    assert plan.get("entry_signal_type") == "Pullback"


def test_build_trade_plan_from_daily_df_allows_trend_entry_when_tp1_rr_is_below_one_but_farther_target_is_sufficient(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "上涨中",
            "next_expected": "等待回踩",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": 12.0,
            "MACD": 6.0,
            "RSI": 4.0,
            "布林带": 3.0,
            "动量": 5.0,
            "价格位置": 0.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [112.0, 116.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())
    monkeypatch.setattr(
        pre_market,
        "calc_ma",
        lambda series, window: pd.Series(
            [
                (
                    106.0 + i * 0.04
                    if window == 5
                    else 105.2 + i * 0.035
                    if window == 10
                    else 104.5 + i * 0.03
                )
                for i in range(len(series))
            ],
            index=series.index,
        ),
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan["entry_family"] == "trend"
    assert bool(plan["actionable"]) is True
    assert plan["tp1"] == pytest.approx(116.0)
    assert plan["entry_signal_type"] == "Pullback"
    assert plan["rr"] == pytest.approx(0.9863945578)
    assert plan["admission_rr"] == pytest.approx(1.7755102041)
    assert plan["admission_rr"] > plan["rr"]
    assert plan["phase2_rr_gate_passed"] is True


def test_build_trade_plan_from_daily_df_returns_trend_short_plan_without_reversal_signal(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "下跌中",
            "next_expected": "等待反弹受阻",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": -12.0,
            "MACD": -6.0,
            "RSI": -4.0,
            "布林带": -3.0,
            "动量": -5.0,
            "价格位置": 0.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([104.0, 108.0], [118.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())
    monkeypatch.setattr(
        pre_market,
        "calc_ma",
        lambda series, window: pd.Series(
            [130.0 - window * 0.2 - i * (0.15 if window == 20 else 0.08) for i in range(len(series))],
            index=series.index,
        ),
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="short",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan.get("actionable") is True
    assert plan["reversal_status"]["has_signal"] is False
    assert plan.get("entry_family") == "trend"
    assert plan.get("entry_signal_type") == "TrendBreak"


def test_build_trade_plan_from_daily_df_prefers_higher_rr_candidate(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-05-10",
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "reversal signal",
            "current_stage": "反转确认",
            "next_expected": "继续观察",
            "confidence": 0.8,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": 12.0,
            "MACD": 5.0,
            "RSI": 4.0,
            "布林带": 3.0,
            "动量": 4.0,
            "价格位置": 2.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "calc_ma", lambda series, window: pd.Series([100.0 + i * 0.2 for i in range(len(series))], index=series.index))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [116.0, 120.0]))

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan.get("entry_family") == "trend"
    assert plan.get("entry_signal_type") == "Pullback"
    assert plan.get("rr", 0.0) >= 1.0


def test_build_trade_plan_from_daily_df_returns_none_for_empty_df() -> None:
    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=pd.DataFrame(),
        cfg={},
    )

    assert plan is None


def test_build_trade_plan_from_daily_df_keeps_signal_prices_when_score_is_severely_countertrend(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-05-10",
            "signal_bar": {"low": 110.0, "high": 114.0},
            "signal_detail": "reversal signal but weak score",
            "current_stage": "反转确认",
            "next_expected": "继续观察",
            "confidence": 0.6,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": -25.0})
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([92.0], [112.0, 118.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan["actionable"] is False
    assert plan["entry_family"] == "reversal"
    assert plan["entry_signal_type"] == "SOS"
    assert plan["score"] == -25.0
    assert plan["entry"] == pytest.approx(113.1)
    assert plan["stop"] == pytest.approx(109.0)
    assert plan["tp1"] == pytest.approx(118.0)
    assert plan["tp2"] == pytest.approx(120.36)
    assert plan["rr"] == pytest.approx(1.1951219512)


def test_build_trade_plan_from_daily_df_rejects_conflicting_trend_quality(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "上涨中",
            "next_expected": "等待回踩",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": 18.0,
            "MACD": -3.0,
            "RSI": 4.0,
            "布林带": 2.0,
            "动量": -2.0,
            "价格位置": 3.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [116.0, 120.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())
    monkeypatch.setattr(
        pre_market,
        "calc_ma",
        lambda series, window: pd.Series(
            [100.0 + window * 0.2 + i * 0.1 for i in range(len(series))],
            index=series.index,
        ),
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan["actionable"] is False
    assert plan["trend_status"]["has_signal"] is False
    assert plan["trend_status"]["trend_indicator_ok"] is False
    assert plan["entry_family"] == ""


def test_assess_active_trend_hold_from_daily_df_accepts_phase_valid_soft_countertrend(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "score_signals",
        lambda *args, **kwargs: {
            "均线排列": 0.0,
            "MACD": 0.0,
            "RSI": 0.0,
            "布林带": 0.0,
            "动量": 5.0,
            "价格位置": 0.0,
        },
    )
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())
    monkeypatch.setattr(
        pre_market,
        "_assess_trend_continuation",
        lambda **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_detail": "",
            "phase": "markdown",
            "phase_ok": True,
            "slope_ok": False,
            "trend_indicator_ok": False,
        },
    )

    hold = pre_market.assess_active_trend_hold_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="short",
        df=df,
        cfg={},
    )

    assert hold is not None
    assert hold["hold_valid"] is True
    assert hold["score"] == 5.0
    assert hold["trend_status"]["phase_ok"] is True


def test_analyze_one_reports_actionable_trend_entry(monkeypatch) -> None:
    df = _build_daily_frame()
    messages: list[str] = []

    def capture_info(message, *args, **kwargs):
        if args:
            message = message % args
        messages.append(str(message))

    monkeypatch.setattr(pre_market.log, "info", capture_info)
    monkeypatch.setattr(pre_market, "fetch_data", lambda *args, **kwargs: df)
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
    monkeypatch.setattr(
        pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: {
            "score": 25.0,
            "actionable": True,
            "entry": float(df["close"].iloc[-1]),
            "stop": 110.0,
            "tp1": 118.0,
            "tp2": 121.0,
            "rr": 1.2,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "markup 阶段顺势回踩",
            "support_levels": [100.0],
            "resistance_levels": [118.0, 121.0],
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
                "has_signal": False,
                "signal_type": "",
                "signal_date": "",
                "signal_bar": {"low": 0.0, "high": 0.0},
                "signal_detail": "",
                "current_stage": "上涨中",
                "next_expected": "等待回踩",
                "confidence": 0.0,
                "all_events": [],
                "suspect_events": [],
            },
            "trend_status": {
                "has_signal": True,
                "signal_type": "Pullback",
                "signal_detail": "markup 阶段顺势回踩",
            },
        },
    )

    result = pre_market.analyze_one("LH0", "生猪", "long", {"reason": "test", "ma_windows": [5], "sr_lookback": 120})

    assert result is not None
    assert any("顺势入场" in msg for msg in messages)
    assert any("Pullback" in msg for msg in messages)
    assert not any("尚无有效反转信号，暂不建议入场" in msg for msg in messages)


def test_analyze_one_restores_phase2_section_logging(monkeypatch) -> None:
    df = _build_daily_frame()
    messages: list[str] = []
    helper_calls: list[dict[str, object]] = []

    def capture_info(message, *args, **kwargs):
        if args:
            message = message % args
        messages.append(str(message))

    monkeypatch.setattr(pre_market.log, "info", capture_info)
    monkeypatch.setattr(pre_market, "fetch_data", lambda *args, **kwargs: df)
    sentinel_plan = {
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
    }

    def fake_build_trade_plan_from_daily_df(**kwargs):
        helper_calls.append(kwargs)
        return sentinel_plan

    monkeypatch.setattr(pre_market, "build_trade_plan_from_daily_df", fake_build_trade_plan_from_daily_df)
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

    assert result is sentinel_plan
    assert helper_calls == [
        {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "df": df,
            "cfg": {"reason": "test", "ma_windows": [5], "sr_lookback": 120},
        }
    ]
    assert any("经典技术指标" in msg for msg in messages)
    assert any("Wyckoff 量价分析" in msg for msg in messages)
    assert any("持仓量(OI)分析" in msg for msg in messages)
    assert any("关键价位" in msg for msg in messages)
    assert any("斐波那契目标位" in msg for msg in messages)
    assert any("综合评分" in msg for msg in messages)
