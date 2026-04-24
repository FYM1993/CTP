from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.models import BacktestCase, TradePlan  # noqa: E402
from backtest.phase23 import (  # noqa: E402
    _backtest_cache_paths,
    _create_backtest_api,
    _backtest_start_dt,
    _daily_warmup_bars,
    _open_backtest_anchor_api,
    _safe_generate_signals,
    aggregate_partial_5m_bars,
    load_case_frames_with_tqbacktest,
    make_trade_plan_from_phase2,
    resolve_case_tq_symbol,
    run_case_from_frames,
)
from shared.strategy import STRATEGY_REVERSAL  # noqa: E402


def _make_case(direction: str, strategy_family: str = "") -> BacktestCase:
    return BacktestCase(
        case_id=f"case_{direction}",
        symbol="LH0" if direction == "long" else "PS0",
        name="生猪" if direction == "long" else "豆粕",
        direction=direction,
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
        strategy_family=strategy_family,
    )


def _daily_df_for_backtest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                    "2025-01-07",
                ]
            ),
            "open": [95.0, 96.0, 97.0, 98.0, 99.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [94.0, 95.0, 96.0, 97.0, 98.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
            "oi": [5000, 5001, 5002, 5003, 5004],
        }
    )


def _long_plan(*, trade_id: str = "trade-long", plan_date: str = "2025-01-06") -> TradePlan:
    return TradePlan(
        trade_id=trade_id,
        symbol="LH0",
        direction="long",
        plan_date=plan_date,
        entry_ref=102.0,
        stop=98.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=30.0,
        signal_type="spring",
        meta={
            "entry_family": "reversal",
            "entry_signal_type": "spring",
            "entry_signal_detail": "反转确认",
        },
    )


def _short_plan(*, trade_id: str = "trade-short", plan_date: str = "2025-01-06") -> TradePlan:
    return TradePlan(
        trade_id=trade_id,
        symbol="PS0",
        direction="short",
        plan_date=plan_date,
        entry_ref=199.0,
        stop=205.0,
        tp1=192.0,
        tp2=189.0,
        phase2_score=-31.0,
        signal_type="upthrust",
        meta={
            "entry_family": "reversal",
            "entry_signal_type": "upthrust",
            "entry_signal_detail": "反转确认",
        },
    )


def _build_minute_df(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime([row[0] for row in rows]),
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
            "volume": [10] * len(rows),
        }
    )


def test_aggregate_partial_5m_bars_keeps_incomplete_current_bar():
    minute_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-06 09:31:00",
                    "2025-01-06 09:32:00",
                    "2025-01-06 09:34:00",
                    "2025-01-06 09:35:00",
                    "2025-01-06 09:36:00",
                ]
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 104.5, 105.5],
            "volume": [10, 20, 30, 40, 50],
        }
    )

    bars = aggregate_partial_5m_bars(minute_df)

    assert list(bars["datetime"]) == [
        pd.Timestamp("2025-01-06 09:30:00"),
        pd.Timestamp("2025-01-06 09:35:00"),
    ]
    assert float(bars.iloc[0]["open"]) == 100.0
    assert float(bars.iloc[0]["high"]) == 103.0
    assert float(bars.iloc[0]["low"]) == 99.0
    assert float(bars.iloc[0]["close"]) == 102.5
    assert float(bars.iloc[0]["volume"]) == 60.0
    assert float(bars.iloc[1]["open"]) == 103.0
    assert float(bars.iloc[1]["high"]) == 106.0
    assert float(bars.iloc[1]["low"]) == 102.0
    assert float(bars.iloc[1]["close"]) == 105.5
    assert float(bars.iloc[1]["volume"]) == 90.0


def test_make_trade_plan_from_phase2_maps_actionable_raw_plan(monkeypatch):
    raw_plan = {
        "symbol": "LH0",
        "direction": "long",
        "strategy_family": "reversal_fundamental",
        "score": 36.5,
        "actionable": True,
        "entry": 103.0,
        "stop": 99.0,
        "tp1": 106.0,
        "tp2": 109.0,
        "reversal_status": {
            "signal_date": "2025-01-06",
            "signal_type": "spring",
        },
    }

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: raw_plan,
    )

    plan = make_trade_plan_from_phase2(
        case=_make_case("long"),
        daily_df=pd.DataFrame({"date": pd.to_datetime(["2025-01-03"])}),
        pre_market_cfg={},
    )

    assert plan == TradePlan(
        trade_id="case_long_2025-01-06",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-06",
        entry_ref=103.0,
        stop=99.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=36.5,
        signal_type="spring",
        meta={
            "strategy_family": "reversal_fundamental",
            "entry_family": "",
            "entry_signal_type": "spring",
            "entry_signal_detail": "",
        },
    )


def test_make_trade_plan_from_phase2_maps_entry_family(monkeypatch):
    raw_plan = {
        "symbol": "LH0",
        "direction": "long",
        "score": 36.5,
        "actionable": True,
        "entry": 103.0,
        "stop": 99.0,
        "tp1": 106.0,
        "tp2": 109.0,
        "entry_family": "trend",
        "entry_signal_type": "Pullback",
        "entry_signal_detail": "上涨趋势回踩后转强",
        "reversal_status": {
            "signal_date": "2025-01-06",
            "signal_type": "SOS",
        },
    }

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: raw_plan,
    )

    plan = make_trade_plan_from_phase2(
        case=_make_case("long"),
        daily_df=pd.DataFrame({"date": pd.to_datetime(["2025-01-03"])}),
        pre_market_cfg={},
    )

    assert plan is not None
    assert plan.signal_type == "Pullback"
    assert plan.meta["entry_family"] == "trend"
    assert plan.meta["entry_signal_type"] == "Pullback"
    assert plan.meta["entry_signal_detail"] == "上涨趋势回踩后转强"


def test_make_trade_plan_from_phase2_preserves_unknown_entry_family(monkeypatch):
    raw_plan = {
        "symbol": "LH0",
        "direction": "long",
        "score": 36.5,
        "actionable": True,
        "entry": 103.0,
        "stop": 99.0,
        "tp1": 106.0,
        "tp2": 109.0,
        "reversal_status": {
            "signal_date": "2025-01-06",
            "signal_type": "SOS",
        },
    }

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: raw_plan,
    )

    plan = make_trade_plan_from_phase2(
        case=_make_case("long"),
        daily_df=pd.DataFrame({"date": pd.to_datetime(["2025-01-03"])}),
        pre_market_cfg={},
    )

    assert plan is not None
    assert plan.meta["entry_family"] == ""


def test_run_case_from_frames_opens_long_and_exits_on_tp2_intrabar_touch(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 101.0, 103.0, 100.0, 103.0),
            ("2025-01-06 09:33:00", 103.0, 108.0, 102.0, 107.0),
            ("2025-01-06 09:34:00", 107.0, 111.0, 106.0, 108.5),
        ]
    )
    daily_df = _daily_df_for_backtest().iloc[:3].copy()
    plan = _long_plan()

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.case_id == "case_long"
    assert result.summary == {"num_trades": 1}
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.trade_id == "trade-long"
    assert trade.direction == "long"
    assert trade.entry_time == "2025-01-06 09:31:00"
    assert trade.entry_price == 100.0
    assert trade.exit_time == "2025-01-06 09:34:00"
    assert trade.exit_price == 109.0
    assert trade.exit_reason == "tp2"
    assert trade.tp1_hit is True
    assert trade.pnl_ratio == 0.09


def test_run_case_from_frames_opens_short_and_exits_on_tp2_intrabar_touch(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 200.0, 201.0, 199.0, 200.0),
            ("2025-01-06 09:32:00", 198.0, 199.0, 197.0, 194.0),
            ("2025-01-06 09:33:00", 194.0, 195.0, 193.0, 190.0),
            ("2025-01-06 09:34:00", 190.0, 191.0, 188.0, 190.0),
        ]
    )
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-05"]),
            "open": [205.0, 204.0, 203.0],
            "high": [206.0, 205.0, 204.0],
            "low": [199.0, 198.0, 197.0],
            "close": [200.0, 199.0, 198.0],
            "volume": [1000, 1000, 1000],
            "oi": [7000, 7001, 7002],
        }
    )
    plan = _short_plan()

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开空"}],
    )

    result = run_case_from_frames(
        case=_make_case("short"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.case_id == "case_short"
    assert result.summary == {"num_trades": 1}
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.trade_id == "trade-short"
    assert trade.direction == "short"
    assert trade.entry_time == "2025-01-06 09:31:00"
    assert trade.entry_price == 200.0
    assert trade.exit_time == "2025-01-06 09:34:00"
    assert trade.exit_price == 189.0
    assert trade.exit_reason == "tp2"
    assert trade.tp1_hit is True
    assert trade.pnl_ratio == 0.055


def test_run_case_from_frames_exits_long_on_stop_intrabar_touch(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 100.0, 101.0, 97.5, 99.5),
        ]
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest().iloc[:3].copy(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: _long_plan(),
        pre_market_cfg={},
        signal_cfg={},
    )

    trade = result.trades[0]
    assert trade.exit_time == "2025-01-06 09:32:00"
    assert trade.exit_price == 98.0
    assert trade.exit_reason == "stop"
    assert trade.pnl_ratio == -0.02


def test_run_case_from_frames_exits_short_on_stop_intrabar_touch(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 200.0, 201.0, 199.0, 200.0),
            ("2025-01-06 09:32:00", 200.0, 206.0, 198.0, 204.0),
        ]
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开空"}],
    )

    result = run_case_from_frames(
        case=_make_case("short"),
        daily_df=_daily_df_for_backtest().iloc[:3].copy(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: _short_plan(),
        pre_market_cfg={},
        signal_cfg={},
    )

    trade = result.trades[0]
    assert trade.exit_time == "2025-01-06 09:32:00"
    assert trade.exit_price == 205.0
    assert trade.exit_reason == "stop"
    assert trade.pnl_ratio == -0.025


def test_run_case_from_frames_prefers_stop_when_stop_and_tp2_touch_same_bar(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 100.0, 110.0, 97.0, 107.0),
        ]
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest().iloc[:3].copy(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: _long_plan(),
        pre_market_cfg={},
        signal_cfg={},
    )

    trade = result.trades[0]
    assert trade.exit_time == "2025-01-06 09:32:00"
    assert trade.exit_price == 98.0
    assert trade.exit_reason == "stop"


def test_safe_generate_signals_returns_empty_list_when_history_is_too_short(monkeypatch):
    monkeypatch.setattr(
        "backtest.phase23.generate_signals",
        lambda df, direction, cfg: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    signals = _safe_generate_signals(
        _build_minute_df(
            [
                ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ]
        ),
        "long",
        {},
    )

    assert signals == []


def test_safe_generate_signals_propagates_index_errors_after_short_history_guard(monkeypatch):
    monkeypatch.setattr(
        "backtest.phase23.generate_signals",
        lambda df, direction, cfg: (_ for _ in ()).throw(IndexError("boom")),
    )

    with pytest.raises(IndexError, match="boom"):
        _safe_generate_signals(
            _build_minute_df(
                [
                    ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
                    ("2025-01-06 09:32:00", 101.0, 102.0, 100.0, 101.0),
                ]
            ),
            "long",
            {},
        )


def test_safe_generate_signals_propagates_runtime_errors(monkeypatch):
    monkeypatch.setattr(
        "backtest.phase23.generate_signals",
        lambda df, direction, cfg: (_ for _ in ()).throw(ValueError("boom")),
    )

    minute_df = _build_minute_df(
        [
            (f"2025-01-06 09:{minute:02d}:00", 100.0 + minute, 101.0 + minute, 99.0 + minute, 100.5 + minute)
            for minute in range(31, 51)
        ]
    )

    with pytest.raises(ValueError, match="boom"):
        _safe_generate_signals(minute_df, "long", {})


def test_run_case_from_frames_force_closes_open_position_at_end_of_data(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 100.0, 102.0, 99.0, 101.0),
            ("2025-01-06 09:33:00", 101.0, 103.0, 100.0, 102.0),
        ]
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest().iloc[:3].copy(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: _long_plan(),
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_time == "2025-01-06 09:33:00"
    assert trade.exit_price == 102.0
    assert trade.exit_reason == "end_of_data"
    assert trade.bars_held == 2


def test_run_case_from_frames_exits_active_trade_when_phase2_invalidates_next_day(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 100.0, 102.0, 99.0, 101.0),
            ("2025-01-07 09:31:00", 97.0, 98.0, 96.0, 97.5),
            ("2025-01-07 09:32:00", 97.5, 98.0, 97.0, 97.2),
        ]
    )
    plan = _long_plan()

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )
    monkeypatch.setattr(
        "backtest.phase23.pre_market.assess_active_reversal_hold_from_daily_df",
        lambda **_kwargs: {"hold_valid": False},
    )

    def plan_factory(*, daily_df, **_kwargs):
        latest_visible = daily_df["trade_date"].max() if not daily_df.empty else None
        if latest_visible == date(2025, 1, 5):
            return plan
        return None

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest(),
        minute_df=minute_df,
        plan_factory=plan_factory,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    trade = result.trades[0]
    assert trade.exit_time == "2025-01-07 09:31:00"
    assert trade.exit_price == 97.0
    assert trade.exit_reason == "phase2_invalidated"


def test_run_case_from_frames_keeps_custom_reversal_trade_when_hold_is_still_valid(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 100.0, 102.0, 99.0, 101.0),
            ("2025-01-07 09:31:00", 101.5, 102.0, 101.0, 101.8),
            ("2025-01-07 09:32:00", 101.8, 102.2, 101.5, 102.0),
        ]
    )
    daily_df = _daily_df_for_backtest().copy()
    plan = TradePlan(
        trade_id="reversal-custom-hold",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-06",
        entry_ref=102.0,
        stop=98.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=30.0,
        signal_type="Spring",
        meta={
            "strategy_family": "reversal_fundamental",
            "entry_family": "reversal",
            "entry_signal_type": "Spring",
            "entry_signal_detail": "反转确认仍然新鲜",
        },
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )
    monkeypatch.setattr(
        "backtest.phase23.pre_market.assess_active_reversal_hold_from_daily_df",
        lambda **_kwargs: {"hold_valid": True},
    )

    def plan_factory(*, daily_df, **_kwargs):
        latest_visible = daily_df["trade_date"].max() if not daily_df.empty else None
        if latest_visible == date(2025, 1, 5):
            return plan
        return None

    result = run_case_from_frames(
        case=_make_case("long", strategy_family="reversal_fundamental"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=plan_factory,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_time == "2025-01-07 09:32:00"
    assert trade.meta["strategy_family"] == "reversal_fundamental"
    assert trade.meta["entry_family"] == "reversal"


def test_run_case_from_frames_keeps_trend_trade_when_phase_still_valid_but_signal_softens(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 200.0, 201.0, 199.0, 200.0),
            ("2025-01-06 09:32:00", 199.0, 200.0, 198.0, 199.0),
            ("2025-01-07 09:31:00", 198.5, 199.0, 198.0, 198.2),
            ("2025-01-07 09:32:00", 198.0, 198.5, 197.5, 197.8),
        ]
    )
    daily_df = _daily_df_for_backtest().copy()

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开空"}],
    )
    monkeypatch.setattr(
        "backtest.phase23.pre_market.assess_active_trend_hold_from_daily_df",
        lambda **_kwargs: {"hold_valid": True},
    )

    def fake_build_trade_plan_from_daily_df(*, df, **_kwargs):
        latest_visible = pd.to_datetime(df["date"]).max().date() if not df.empty else None
        if latest_visible == date(2025, 1, 5):
            return {
                "symbol": "PS0",
                "name": "豆粕",
                "direction": "short",
                "score": -28.0,
                "actionable": True,
                "entry": 199.0,
                "stop": 205.0,
                "tp1": 192.0,
                "tp2": 189.0,
                "rr": 1.16,
                "entry_family": "trend",
                "entry_signal_type": "Pullback",
                "entry_signal_detail": "markdown 阶段顺势反弹受阻",
                "reversal_status": {"has_signal": False},
                "trend_status": {
                    "has_signal": True,
                    "signal_type": "Pullback",
                    "signal_detail": "markdown 阶段顺势反弹受阻",
                    "phase": "markdown",
                    "phase_ok": True,
                    "slope_ok": True,
                    "trend_indicator_ok": True,
                },
            }
        if latest_visible == date(2025, 1, 6):
            return {
                "symbol": "PS0",
                "name": "豆粕",
                "direction": "short",
                "score": 5.0,
                "actionable": False,
                "entry": 198.2,
                "stop": 205.0,
                "tp1": 192.0,
                "tp2": 189.0,
                "rr": 0.6,
                "entry_family": "trend",
                "entry_signal_type": "Pullback",
                "entry_signal_detail": "markdown 阶段仍在，但短期指标回摆",
                "reversal_status": {"has_signal": False},
                "trend_status": {
                    "has_signal": False,
                    "signal_type": "",
                    "signal_detail": "",
                    "phase": "markdown",
                    "phase_ok": True,
                    "slope_ok": False,
                    "trend_indicator_ok": False,
                },
            }
        return None

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    result = run_case_from_frames(
        case=_make_case("short"),
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_time == "2025-01-07 09:32:00"
    assert trade.meta["entry_family"] == "trend"


def test_run_case_from_frames_keeps_custom_trend_trade_when_hold_is_still_valid(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 200.0, 201.0, 199.0, 200.0),
            ("2025-01-06 09:32:00", 199.0, 200.0, 198.0, 199.0),
            ("2025-01-07 09:31:00", 198.5, 199.0, 198.0, 198.2),
            ("2025-01-07 09:32:00", 198.0, 198.5, 197.5, 197.8),
        ]
    )
    daily_df = _daily_df_for_backtest().copy()
    plan = TradePlan(
        trade_id="trend-custom-hold",
        symbol="LH0",
        direction="short",
        plan_date="2025-01-06",
        entry_ref=199.0,
        stop=205.0,
        tp1=192.0,
        tp2=189.0,
        phase2_score=-28.0,
        signal_type="Pullback",
        meta={
            "strategy_family": "trend_following",
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "markdown 阶段顺势反弹受阻",
        },
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开空"}],
    )
    monkeypatch.setattr(
        "backtest.phase23.pre_market.assess_active_trend_hold_from_daily_df",
        lambda **_kwargs: {"hold_valid": True},
    )

    def plan_factory(*, daily_df, **_kwargs):
        latest_visible = daily_df["trade_date"].max() if not daily_df.empty else None
        if latest_visible == date(2025, 1, 5):
            return plan
        return None

    result = run_case_from_frames(
        case=_make_case("short", strategy_family="trend_following"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=plan_factory,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_time == "2025-01-07 09:32:00"
    assert trade.meta["strategy_family"] == "trend_following"
    assert trade.meta["entry_family"] == "trend"


def test_run_case_from_frames_plan_factory_sees_only_prior_daily_bars(monkeypatch):
    seen_daily_dates: list[tuple[date, list[date]]] = []
    expected_trade_dates = [date(2025, 1, 6), date(2025, 1, 7)]

    monkeypatch.setattr("backtest.phase23._safe_generate_signals", lambda df, direction, cfg: [])

    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-07 09:31:00", 101.0, 102.0, 100.0, 101.0),
        ]
    )

    def plan_factory(*, daily_df, **kwargs):
        seen_daily_dates.append(
            (expected_trade_dates[len(seen_daily_dates)], list(daily_df["trade_date"]))
        )
        return None

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest(),
        minute_df=minute_df,
        plan_factory=plan_factory,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 0}
    assert seen_daily_dates == [
        (date(2025, 1, 6), [date(2025, 1, 3), date(2025, 1, 4), date(2025, 1, 5)]),
        (
            date(2025, 1, 7),
            [date(2025, 1, 3), date(2025, 1, 4), date(2025, 1, 5), date(2025, 1, 6)],
        ),
    ]


def test_run_case_from_frames_consumes_trade_id_after_completed_trade(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 101.0, 103.0, 100.0, 103.0),
            ("2025-01-06 09:33:00", 103.0, 108.0, 102.0, 107.0),
            ("2025-01-06 09:34:00", 107.0, 111.0, 106.0, 110.0),
            ("2025-01-07 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-07 09:32:00", 101.0, 103.0, 100.0, 103.0),
            ("2025-01-07 09:33:00", 103.0, 108.0, 102.0, 107.0),
            ("2025-01-07 09:34:00", 107.0, 111.0, 106.0, 110.0),
        ]
    )
    plan = _long_plan(trade_id="reused-trade-id")

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    assert [trade.trade_id for trade in result.trades] == ["reused-trade-id"]


def test_run_case_from_frames_reports_actionable_plan_without_entry(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 100.0, 101.0, 99.0, 100.0),
        ]
    )
    daily_df = _daily_df_for_backtest().iloc[:3].copy()
    plan = _long_plan()

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 0}
    assert result.diagnostics == {
        "daily_rows_loaded": 3,
        "daily_first_date": "2025-01-03",
        "daily_last_date": "2025-01-05",
        "first_trade_day": "2025-01-06",
        "first_trade_day_visible_daily_rows": 3,
        "first_trade_day_visible_first_date": "2025-01-03",
        "first_trade_day_visible_last_date": "2025-01-05",
        "trade_days_total": 1,
        "phase2_history_insufficient_days": 0,
        "phase2_non_actionable_days": 0,
        "phase2_actionable_days": 1,
        "phase2_actionable_reversal_days": 1,
        "phase2_actionable_trend_days": 0,
        "phase2_actionable_strategy_reversal_days": 0,
        "phase2_actionable_strategy_trend_days": 0,
        "phase2_reject_no_signal_days": 0,
        "phase2_reject_score_gate_days": 0,
        "phase2_reject_rr_gate_days": 0,
        "phase2_reject_duplicate_signal_days": 0,
        "phase3_signal_eval_bars": 2,
        "phase3_entry_signal_hits": 0,
        "trades_opened": 0,
        "trades_opened_reversal": 0,
        "trades_opened_trend": 0,
        "trades_opened_strategy_reversal": 0,
        "trades_opened_strategy_trend": 0,
    }


def test_run_case_from_frames_reports_short_history_then_entry(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-02 09:31:00", 99.0, 100.0, 98.0, 99.5),
            ("2025-01-03 09:31:00", 100.0, 101.0, 99.0, 100.0),
        ]
    )
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [95.0, 96.0, 97.0],
            "high": [101.0, 102.0, 103.0],
            "low": [94.0, 95.0, 96.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000],
            "oi": [5000, 5001, 5002],
        }
    )
    plan = _long_plan(plan_date="2025-01-02", trade_id="diag-trade")

    def fake_plan_factory(*, daily_df, **_kwargs):
        return plan if len(daily_df) >= 2 else None

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=fake_plan_factory,
        pre_market_cfg={"min_history_bars": 2},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 1}
    assert result.diagnostics == {
        "daily_rows_loaded": 3,
        "daily_first_date": "2025-01-01",
        "daily_last_date": "2025-01-03",
        "first_trade_day": "2025-01-02",
        "first_trade_day_visible_daily_rows": 1,
        "first_trade_day_visible_first_date": "2025-01-01",
        "first_trade_day_visible_last_date": "2025-01-01",
        "trade_days_total": 2,
        "phase2_history_insufficient_days": 1,
        "phase2_non_actionable_days": 0,
        "phase2_actionable_days": 1,
        "phase2_actionable_reversal_days": 1,
        "phase2_actionable_trend_days": 0,
        "phase2_actionable_strategy_reversal_days": 0,
        "phase2_actionable_strategy_trend_days": 0,
        "phase2_reject_no_signal_days": 0,
        "phase2_reject_score_gate_days": 0,
        "phase2_reject_rr_gate_days": 0,
        "phase2_reject_duplicate_signal_days": 0,
        "phase3_signal_eval_bars": 1,
        "phase3_entry_signal_hits": 1,
        "trades_opened": 1,
        "trades_opened_reversal": 1,
        "trades_opened_trend": 0,
        "trades_opened_strategy_reversal": 0,
        "trades_opened_strategy_trend": 0,
    }


def test_run_case_from_frames_tracks_trend_entry_family_diagnostics(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 101.0, 103.0, 100.0, 103.0),
        ]
    )
    plan = TradePlan(
        trade_id="trend-trade",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-05",
        entry_ref=102.0,
        stop=98.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=30.0,
        signal_type="Pullback",
        meta={
            "strategy_family": "trend_following",
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "上涨趋势回踩后转强",
        },
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long", strategy_family="trend_following"),
        daily_df=_daily_df_for_backtest().iloc[:3].copy(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.diagnostics["phase2_actionable_days"] == 1
    assert result.diagnostics["phase2_actionable_reversal_days"] == 0
    assert result.diagnostics["phase2_actionable_trend_days"] == 1
    assert result.diagnostics["phase2_actionable_strategy_trend_days"] == 1
    assert result.diagnostics["trades_opened"] == 1
    assert result.diagnostics["trades_opened_reversal"] == 0
    assert result.diagnostics["trades_opened_trend"] == 1
    assert result.diagnostics["trades_opened_strategy_trend"] == 1
    assert result.trades[0].meta["strategy_family"] == "trend_following"
    assert result.trades[0].meta["entry_family"] == "trend"
    assert result.trades[0].meta["entry_signal_type"] == "Pullback"
    assert result.trades[0].meta["entry_signal_detail"] == "上涨趋势回踩后转强"


def test_run_case_from_frames_does_not_force_unknown_entry_family_into_reversal(monkeypatch):
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 101.0, 103.0, 100.0, 103.0),
        ]
    )
    plan = TradePlan(
        trade_id="unknown-family-trade",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-05",
        entry_ref=102.0,
        stop=98.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=30.0,
        signal_type="Pullback",
        meta={
            "entry_family": "",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "未标注来源",
        },
    )

    monkeypatch.setattr(
        "backtest.phase23._safe_generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=_daily_df_for_backtest().iloc[:3].copy(),
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.diagnostics["phase2_actionable_days"] == 1
    assert result.diagnostics["phase2_actionable_reversal_days"] == 0
    assert result.diagnostics["phase2_actionable_trend_days"] == 0
    assert result.diagnostics["trades_opened"] == 1
    assert result.diagnostics["trades_opened_reversal"] == 0
    assert result.diagnostics["trades_opened_trend"] == 0
    assert result.trades[0].meta["entry_family"] == ""


def test_run_case_from_frames_reports_phase2_no_signal_rejection(monkeypatch):
    minute_df = _build_minute_df([("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0)])
    daily_df = _daily_df_for_backtest()

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 8.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "rr": 0.0,
            "reversal_status": {"has_signal": False},
        },
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 0}
    assert result.diagnostics["phase2_non_actionable_days"] == 1
    assert result.diagnostics["phase2_reject_no_signal_days"] == 1
    assert result.diagnostics["phase2_reject_score_gate_days"] == 0
    assert result.diagnostics["phase2_reject_rr_gate_days"] == 0


def test_run_case_from_frames_reports_phase2_score_and_rr_rejections(monkeypatch):
    minute_df = _build_minute_df([("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0)])
    daily_df = _daily_df_for_backtest()

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 18.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 95.0,
            "tp1": 103.0,
            "tp2": 105.0,
            "rr": 0.8,
            "reversal_status": {"has_signal": True},
        },
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 0}
    assert result.diagnostics["phase2_non_actionable_days"] == 1
    assert result.diagnostics["phase2_reject_no_signal_days"] == 0
    assert result.diagnostics["phase2_reject_score_gate_days"] == 1
    assert result.diagnostics["phase2_reject_rr_gate_days"] == 1


def test_run_case_from_frames_does_not_treat_trend_signal_as_no_signal(monkeypatch):
    minute_df = _build_minute_df([("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0)])
    daily_df = _daily_df_for_backtest()

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 18.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 95.0,
            "tp1": 103.0,
            "tp2": 105.0,
            "rr": 0.8,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "上涨趋势回踩后转强",
            "reversal_status": {"has_signal": False},
            "trend_status": {
                "has_signal": True,
                "signal_type": "Pullback",
                "signal_detail": "上涨趋势回踩后转强",
                "phase": "markup",
                "phase_ok": True,
                "slope_ok": True,
                "trend_indicator_ok": True,
            },
        },
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
    )

    assert result.summary == {"num_trades": 0}
    assert result.diagnostics["phase2_non_actionable_days"] == 1
    assert result.diagnostics["phase2_reject_no_signal_days"] == 0
    assert result.diagnostics["phase2_reject_score_gate_days"] == 1
    assert result.diagnostics["phase2_reject_rr_gate_days"] == 1


def test_run_case_from_frames_collects_phase2_debug_days_when_requested(monkeypatch):
    minute_df = _build_minute_df([("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0)])
    daily_df = _daily_df_for_backtest()

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 18.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 95.0,
            "tp1": 103.0,
            "tp2": 105.0,
            "rr": 0.8,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "上涨趋势回踩后转强",
            "reversal_status": {
                "has_signal": False,
                "current_stage": "markdown",
                "next_expected": "等待Spring",
            },
            "trend_status": {
                "has_signal": True,
                "signal_type": "Pullback",
                "signal_detail": "上涨趋势回踩后转强",
                "phase": "markup",
                "phase_ok": True,
                "slope_ok": True,
                "trend_indicator_ok": True,
            },
        },
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
        capture_debug=True,
    )

    assert result.debug["phase2_days"] == [
        {
            "trade_date": "2025-01-06",
            "visible_daily_rows": 3,
            "phase2_state": "score_gate+rr_gate",
            "strategy_family": "",
            "score": 18.0,
            "rr": 0.8,
            "phase2_score_gate_passed": False,
            "phase2_rr_gate_passed": False,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "上涨趋势回踩后转强",
            "reversal_has_signal": False,
            "reversal_signal_type": "",
            "reversal_signal_days_ago": None,
            "reversal_signal_strength": 0.0,
            "reversal_signal_fresh": False,
            "reversal_current_stage": "markdown",
            "reversal_next_expected": "等待Spring",
            "trend_has_signal": True,
            "trend_signal_type": "Pullback",
            "trend_phase": "markup",
            "trend_phase_ok": True,
            "trend_slope_ok": True,
            "trend_indicator_ok": True,
        }
    ]


def test_create_backtest_api_builds_tqsdk_backtest_session(monkeypatch):
    created: dict[str, object] = {}

    class FakeTqSim:
        pass

    class FakeTqBacktest:
        def __init__(self, start_dt, end_dt):
            self.start_dt = start_dt
            self.end_dt = end_dt

    class FakeTqAuth:
        def __init__(self, account, password):
            self.account = account
            self.password = password

    class FakeTqApi:
        def __init__(self, sim, backtest=None, auth=None):
            created["sim"] = sim
            created["backtest"] = backtest
            created["auth"] = auth

    fake_tqsdk = type(sys)("tqsdk")
    fake_tqsdk.TqApi = FakeTqApi
    fake_tqsdk.TqAuth = FakeTqAuth
    fake_tqsdk.TqBacktest = FakeTqBacktest
    fake_tqsdk.TqSim = FakeTqSim
    monkeypatch.setitem(sys.modules, "tqsdk", fake_tqsdk)

    api = _create_backtest_api(
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
        account="acct",
        password="pwd",
    )

    assert isinstance(api, FakeTqApi)
    assert isinstance(created["sim"], FakeTqSim)
    assert created["backtest"].start_dt == date(2025, 1, 1)
    assert created["backtest"].end_dt == date(2025, 1, 31)
    assert created["auth"].account == "acct"
    assert created["auth"].password == "pwd"


def test_run_case_from_frames_strategy_specific_debug_keeps_reversal_watch_snapshot(monkeypatch):
    daily_df = _daily_df_for_backtest().iloc[:4].copy()
    minute_df = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 101.0, 101.5, 100.8, 101.2),
            ("2025-01-06 09:32:00", 101.2, 101.6, 101.0, 101.3),
        ]
    )

    monkeypatch.setattr(
        "backtest.phase23.pre_market.build_reversal_trade_plan_from_daily_df",
        lambda **kwargs: {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "strategy_family": STRATEGY_REVERSAL,
            "score": 9.0,
            "actionable": False,
            "entry": 103.0,
            "stop": 98.0,
            "tp1": 107.0,
            "tp2": 110.0,
            "rr": 0.0,
            "entry_family": "",
            "entry_signal_type": "",
            "entry_signal_detail": "",
            "reversal_status": {
                "has_signal": True,
                "signal_type": "Spring",
                "signal_days_ago": 2,
                "signal_strength": 0.75,
                "current_stage": "反转确认",
                "next_expected": "确认新鲜(2天前)，可考虑入场",
            },
            "trend_status": {
                "has_signal": False,
                "signal_type": "",
                "phase": "markdown",
                "phase_ok": False,
                "slope_ok": False,
                "trend_indicator_ok": False,
            },
            "phase2_score_gate_passed": False,
            "phase2_rr_gate_passed": True,
            "reversal_signal_fresh": True,
        },
    )

    result = run_case_from_frames(
        case=_make_case("long", STRATEGY_REVERSAL),
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg={"min_history_bars": 1},
        signal_cfg={},
        plan_factory=lambda **kwargs: None,
        capture_debug=True,
    )

    assert result.diagnostics["phase2_non_actionable_days"] == 1
    assert result.diagnostics["phase2_reject_no_signal_days"] == 0
    assert result.diagnostics["phase2_reject_score_gate_days"] == 1
    assert result.diagnostics["phase2_reject_rr_gate_days"] == 0
    assert result.debug["phase2_days"] == [
        {
            "trade_date": "2025-01-06",
            "visible_daily_rows": 3,
            "phase2_state": "score_gate",
            "strategy_family": STRATEGY_REVERSAL,
            "score": 9.0,
            "rr": 0.0,
            "phase2_score_gate_passed": False,
            "phase2_rr_gate_passed": True,
            "entry_family": "",
            "entry_signal_type": "Spring",
            "entry_signal_detail": "",
            "reversal_has_signal": True,
            "reversal_signal_type": "Spring",
            "reversal_signal_days_ago": 2,
            "reversal_signal_strength": 0.75,
            "reversal_signal_fresh": True,
            "reversal_current_stage": "反转确认",
            "reversal_next_expected": "确认新鲜(2天前)，可考虑入场",
            "trend_has_signal": False,
            "trend_signal_type": "",
            "trend_phase": "markdown",
            "trend_phase_ok": False,
            "trend_slope_ok": False,
            "trend_indicator_ok": False,
        }
    ]


def test_daily_warmup_bars_defaults_to_phase2_min_history() -> None:
    assert _daily_warmup_bars({}) == 60
    assert _daily_warmup_bars({"pre_market": {"min_history_bars": 75}}) == 75


def test_backtest_start_dt_moves_earlier_for_warmup() -> None:
    case = _make_case("long")

    out = _backtest_start_dt(case, {"pre_market": {"min_history_bars": 60}})

    assert out == case.start_dt - timedelta(days=120)


def test_resolve_case_tq_symbol_prefers_dynamic_resolution(monkeypatch):
    seen: dict[str, object] = {}
    fake_api = object()

    def fake_resolve(symbol, exchange, api=None, wait_timeout=None):
        seen["symbol"] = symbol
        seen["exchange"] = exchange
        seen["api"] = api
        seen["wait_timeout"] = wait_timeout
        return "KQ.m@DCE.lh"

    monkeypatch.setattr("backtest.phase23._exchange_for_symbol", lambda symbol: "dce")
    monkeypatch.setattr("backtest.phase23.resolve_tq_continuous_symbol", fake_resolve)
    monkeypatch.setattr(
        "backtest.phase23.symbol_to_tq_main",
        lambda symbol, exchange: pytest.fail("unexpected fallback"),
    )

    out = resolve_case_tq_symbol(_make_case("long"), api=fake_api)

    assert out == "KQ.m@DCE.lh"
    assert seen == {
        "symbol": "LH0",
        "exchange": "dce",
        "api": fake_api,
        "wait_timeout": 2.0,
    }


def test_load_case_frames_with_tqbacktest_uses_case_range_and_closes_api(monkeypatch, tmp_path):
    case = BacktestCase(
        case_id="chunk_case",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 4),
    )
    created: dict[str, object] = {"calls": []}

    daily_serial = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-12-31", "2025-01-03", "2025-01-04"]),
            "open": [99.0, 100.0, 101.0],
            "high": [109.0, 110.0, 111.0],
            "low": [94.0, 95.0, 96.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [999.0, 1000.0, 1001.0],
            "close_oi": [799.0, 800.0, 801.0],
        }
    )
    minute_chunk_a = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-03 09:31:00", "2025-01-04 09:32:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 12.0],
        }
    )
    minute_chunk_b = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01 09:31:00", "2025-01-02 09:32:00"]),
            "open": [98.0, 99.0],
            "high": [99.0, 100.0],
            "low": [97.0, 98.0],
            "close": [98.5, 99.5],
            "volume": [9.0, 11.0],
        }
    )

    class FakeApi:
        def __init__(self, anchor_dt):
            self.anchor_dt = anchor_dt
            self.calls: list[tuple[str, int, int | None]] = []
            self.closed = False

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            self.calls.append((symbol, duration_seconds, data_length))
            if duration_seconds == 86400:
                return daily_serial
            if duration_seconds == 60:
                if self.anchor_dt == date(2025, 1, 5):
                    return minute_chunk_a
                if self.anchor_dt == date(2025, 1, 3):
                    return minute_chunk_b
                raise AssertionError(f"unexpected minute anchor: {self.anchor_dt}")
            raise AssertionError(f"unexpected duration: {duration_seconds}")

        def wait_update(self, deadline=None):
            raise AssertionError("chunk loader should not drive wait_update")

        def close(self):
            self.closed = True

    def fake_create_backtest_api(*, start_dt, end_dt, account, password):
        created["calls"].append((start_dt, end_dt, account, password))
        api = FakeApi(end_dt)
        created.setdefault("apis", []).append(api)
        return api

    monkeypatch.setattr("backtest.phase23.BACKTEST_CACHE_DIR", tmp_path)
    monkeypatch.setattr("backtest.phase23._create_backtest_api", fake_create_backtest_api)
    monkeypatch.setattr("backtest.phase23.BACKTEST_ANCHOR_OFFSETS", (1,))
    monkeypatch.setattr("backtest.phase23.BACKTEST_MINUTE_CHUNK_DAYS", 2)
    monkeypatch.setattr(
        "backtest.phase23.resolve_case_tq_symbol",
        lambda case, api=None: "KQ.m@DCE.lh",
    )

    daily_df, minute_df = load_case_frames_with_tqbacktest(
        case=case,
        config={"tqsdk": {"account": "acct", "password": "pwd"}},
    )

    assert created["calls"][0] == (case.start_dt - timedelta(days=120), case.end_dt, "acct", "pwd")
    assert created["calls"][1] == (date(2025, 1, 5), date(2025, 1, 5), "acct", "pwd")
    assert created["calls"][2] == (date(2025, 1, 3), date(2025, 1, 3), "acct", "pwd")
    assert created["calls"][3] == (date(2025, 1, 5), date(2025, 1, 5), "acct", "pwd")
    assert list(daily_df.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
    assert len(daily_df) == 3
    assert float(daily_df.iloc[1]["oi"]) == 800.0
    assert list(minute_df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert len(minute_df) == 4
    assert list(minute_df["datetime"].dt.date) == [
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 4),
    ]
    assert all(api.closed is True for api in created["apis"])


def test_load_case_frames_with_tqbacktest_keeps_pre_start_daily_warmup(monkeypatch, tmp_path):
    case = BacktestCase(
        case_id="warmup_case",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 6),
        end_dt=date(2025, 1, 31),
    )

    daily_serial = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "open": [98.0, 99.0, 100.0],
            "high": [108.0, 109.0, 110.0],
            "low": [93.0, 94.0, 95.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [900.0, 950.0, 1000.0],
            "close_oi": [700.0, 750.0, 800.0],
        }
    )
    minute_serial = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-06 09:31:00", "2025-01-06 09:32:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 12.0],
        }
    )

    class FakeApi:
        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            if duration_seconds == 86400:
                return daily_serial
            if duration_seconds == 60:
                return minute_serial
            raise AssertionError(f"unexpected duration: {duration_seconds}")

        def wait_update(self, deadline=None):
            return False

        def close(self):
            return None

    monkeypatch.setattr("backtest.phase23.BACKTEST_CACHE_DIR", tmp_path)
    monkeypatch.setattr("backtest.phase23._create_backtest_api", lambda **_kwargs: FakeApi())
    monkeypatch.setattr(
        "backtest.phase23.resolve_case_tq_symbol",
        lambda case, api=None: "KQ.m@DCE.lh",
    )

    daily_df, minute_df = load_case_frames_with_tqbacktest(
        case=case,
        config={"tqsdk": {"account": "acct", "password": "pwd"}},
    )

    assert list(daily_df["date"].dt.date) == [
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
    ]
    assert list(minute_df["datetime"].dt.date) == [date(2025, 1, 6), date(2025, 1, 6)]


def test_load_case_frames_with_tqbacktest_keeps_pre_start_history_stable_when_end_date_moves(monkeypatch, tmp_path):
    short_case = BacktestCase(
        case_id="ao_short_window",
        symbol="AO0",
        name="氧化铝",
        direction="long",
        start_dt=date(2024, 7, 8),
        end_dt=date(2024, 7, 19),
    )
    long_case = BacktestCase(
        case_id="ao_long_window",
        symbol="AO0",
        name="氧化铝",
        direction="long",
        start_dt=date(2024, 7, 8),
        end_dt=date(2024, 7, 31),
    )

    short_trade_days = pd.bdate_range(short_case.start_dt, short_case.end_dt)
    long_trade_days = pd.bdate_range(long_case.start_dt, long_case.end_dt)

    def _minute_serial(trade_days: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": trade_days + pd.Timedelta(hours=9, minutes=31),
                "open": [100.0 + idx for idx in range(len(trade_days))],
                "high": [101.0 + idx for idx in range(len(trade_days))],
                "low": [99.0 + idx for idx in range(len(trade_days))],
                "close": [100.5 + idx for idx in range(len(trade_days))],
                "volume": [10.0] * len(trade_days),
            }
        )

    minute_serial_by_anchor = {
        date(2024, 7, 20): _minute_serial(short_trade_days),
        date(2024, 8, 1): _minute_serial(long_trade_days),
    }

    class FakeApi:
        def __init__(self, anchor_dt):
            self.anchor_dt = anchor_dt

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            if duration_seconds == 86400:
                daily_dates = pd.bdate_range(
                    end=pd.Timestamp(self.anchor_dt) - pd.Timedelta(days=1),
                    periods=int(data_length or 0),
                )
                return pd.DataFrame(
                    {
                        "datetime": daily_dates,
                        "open": [100.0 + idx for idx in range(len(daily_dates))],
                        "high": [101.0 + idx for idx in range(len(daily_dates))],
                        "low": [99.0 + idx for idx in range(len(daily_dates))],
                        "close": [100.5 + idx for idx in range(len(daily_dates))],
                        "volume": [1000.0] * len(daily_dates),
                        "close_oi": [5000.0 + idx for idx in range(len(daily_dates))],
                    }
                )
            if duration_seconds == 60:
                return minute_serial_by_anchor[self.anchor_dt]
            raise AssertionError(f"unexpected duration: {duration_seconds}")

        def close(self):
            return None

    monkeypatch.setattr("backtest.phase23.BACKTEST_CACHE_DIR", tmp_path)
    monkeypatch.setattr("backtest.phase23._create_backtest_api", lambda **kwargs: FakeApi(kwargs["end_dt"]))
    monkeypatch.setattr("backtest.phase23.BACKTEST_ANCHOR_OFFSETS", (1,))
    monkeypatch.setattr("backtest.phase23.BACKTEST_MINUTE_CHUNK_DAYS", 40)
    monkeypatch.setattr(
        "backtest.phase23.resolve_case_tq_symbol",
        lambda case, api=None: "KQ.m@SHFE.ao",
    )

    short_daily, _ = load_case_frames_with_tqbacktest(
        case=short_case,
        config={"tqsdk": {"account": "acct", "password": "pwd"}},
    )
    long_daily, _ = load_case_frames_with_tqbacktest(
        case=long_case,
        config={"tqsdk": {"account": "acct", "password": "pwd"}},
    )

    short_visible = short_daily.loc[short_daily["date"].dt.date < short_case.start_dt].copy()
    long_visible = long_daily.loc[long_daily["date"].dt.date < long_case.start_dt].copy()

    assert len(short_visible) == len(long_visible)
    assert short_visible["date"].min() == long_visible["date"].min()
    assert short_visible["date"].max() == long_visible["date"].max()


def test_load_case_frames_with_tqbacktest_returns_cached_frames_without_tqsdk(monkeypatch, tmp_path):
    case = _make_case("long")
    warmup_bars = _daily_warmup_bars({})
    cached_daily = _daily_df_for_backtest().copy()
    cached_minute = _build_minute_df(
        [
            ("2025-01-06 09:31:00", 100.0, 101.0, 99.0, 100.0),
            ("2025-01-06 09:32:00", 101.0, 102.0, 100.0, 101.0),
        ]
    )

    monkeypatch.setattr("backtest.phase23.BACKTEST_CACHE_DIR", tmp_path)
    daily_path, minute_path = _backtest_cache_paths(case=case, warmup_bars=warmup_bars)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    cached_daily.to_parquet(daily_path, index=False)
    cached_minute.to_parquet(minute_path, index=False)
    monkeypatch.setattr(
        "backtest.phase23._create_backtest_api",
        lambda **_kwargs: pytest.fail("cache hit should not open tqbacktest"),
    )
    monkeypatch.setattr(
        "backtest.phase23.resolve_case_tq_symbol",
        lambda *args, **kwargs: pytest.fail("cache hit should not resolve tq symbol"),
    )

    daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config={})

    pd.testing.assert_frame_equal(daily_df, cached_daily)
    pd.testing.assert_frame_equal(minute_df, cached_minute)


def test_load_case_frames_with_tqbacktest_writes_cache_for_followup_runs(monkeypatch, tmp_path):
    case = BacktestCase(
        case_id="cache_case",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 4),
    )
    created: dict[str, object] = {"calls": []}

    daily_serial = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-12-31", "2025-01-03", "2025-01-04"]),
            "open": [99.0, 100.0, 101.0],
            "high": [109.0, 110.0, 111.0],
            "low": [94.0, 95.0, 96.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [999.0, 1000.0, 1001.0],
            "close_oi": [799.0, 800.0, 801.0],
        }
    )
    minute_chunk_a = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-03 09:31:00", "2025-01-04 09:32:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 12.0],
        }
    )
    minute_chunk_b = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01 09:31:00", "2025-01-02 09:32:00"]),
            "open": [98.0, 99.0],
            "high": [99.0, 100.0],
            "low": [97.0, 98.0],
            "close": [98.5, 99.5],
            "volume": [9.0, 11.0],
        }
    )

    class FakeApi:
        def __init__(self, anchor_dt):
            self.anchor_dt = anchor_dt
            self.closed = False

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            if duration_seconds == 86400:
                return daily_serial
            if duration_seconds == 60:
                if self.anchor_dt == date(2025, 1, 5):
                    return minute_chunk_a
                if self.anchor_dt == date(2025, 1, 3):
                    return minute_chunk_b
                raise AssertionError(f"unexpected minute anchor: {self.anchor_dt}")
            raise AssertionError(f"unexpected duration: {duration_seconds}")

        def close(self):
            self.closed = True

    def fake_create_backtest_api(*, start_dt, end_dt, account, password):
        created["calls"].append((start_dt, end_dt, account, password))
        api = FakeApi(end_dt)
        created.setdefault("apis", []).append(api)
        return api

    monkeypatch.setattr("backtest.phase23.BACKTEST_CACHE_DIR", tmp_path)
    monkeypatch.setattr("backtest.phase23._create_backtest_api", fake_create_backtest_api)
    monkeypatch.setattr("backtest.phase23.BACKTEST_ANCHOR_OFFSETS", (1,))
    monkeypatch.setattr("backtest.phase23.BACKTEST_MINUTE_CHUNK_DAYS", 2)
    monkeypatch.setattr(
        "backtest.phase23.resolve_case_tq_symbol",
        lambda case, api=None: "KQ.m@DCE.lh",
    )

    first_daily, first_minute = load_case_frames_with_tqbacktest(
        case=case,
        config={"tqsdk": {"account": "acct", "password": "pwd"}},
    )

    warmup_bars = _daily_warmup_bars({"tqsdk": {"account": "acct", "password": "pwd"}})
    daily_path, minute_path = _backtest_cache_paths(case=case, warmup_bars=warmup_bars)
    assert daily_path.exists() is True
    assert minute_path.exists() is True
    pd.testing.assert_frame_equal(pd.read_parquet(daily_path), first_daily)
    pd.testing.assert_frame_equal(pd.read_parquet(minute_path), first_minute)

    monkeypatch.setattr(
        "backtest.phase23._create_backtest_api",
        lambda **_kwargs: pytest.fail("cached rerun should not open tqbacktest"),
    )
    monkeypatch.setattr(
        "backtest.phase23.resolve_case_tq_symbol",
        lambda *args, **kwargs: pytest.fail("cached rerun should not resolve tq symbol"),
    )

    second_daily, second_minute = load_case_frames_with_tqbacktest(case=case, config={})

    pd.testing.assert_frame_equal(second_daily, first_daily)
    pd.testing.assert_frame_equal(second_minute, first_minute)


def test_open_backtest_anchor_api_retries_future_offsets(monkeypatch):
    attempts: list[tuple[date, date]] = []

    class FakeApi:
        def close(self):
            return None

    monkeypatch.setattr("backtest.phase23.BACKTEST_ANCHOR_OFFSETS", (1, 3))

    def fake_create_backtest_api(*, start_dt, end_dt, account, password):
        attempts.append((start_dt, end_dt))
        if start_dt == date(2025, 2, 1):
            raise IndexError("holiday calendar gap")
        return FakeApi()

    monkeypatch.setattr("backtest.phase23._create_backtest_api", fake_create_backtest_api)

    api, anchor_dt = _open_backtest_anchor_api(
        target_dt=date(2025, 1, 31),
        account="acct",
        password="pwd",
    )

    assert isinstance(api, FakeApi)
    assert anchor_dt == date(2025, 2, 3)
    assert attempts == [
        (date(2025, 2, 1), date(2025, 2, 1)),
        (date(2025, 2, 3), date(2025, 2, 3)),
    ]
