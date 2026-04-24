from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase3 import intraday as intraday_module  # noqa: E402
from cli import daily_workflow  # noqa: E402


def _build_minute_df(*, start: float = 4900.0, step: float = 2.0, wick: float = 5.0) -> pd.DataFrame:
    datetimes = pd.date_range("2026-04-23 13:30:00", periods=40, freq="5min")
    close = [start + i * step for i in range(40)]
    return pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": [value + wick for value in close],
            "low": [value - wick for value in close],
            "close": close,
            "volume": [1000 + i * 10 for i in range(40)],
        }
    )


def test_print_dashboard_labels_no_signal_as_minute_auxiliary_only(monkeypatch) -> None:
    infos: list[str] = []
    df = _build_minute_df()

    monkeypatch.setattr(intraday_module, "generate_signals", lambda *args, **kwargs: [])
    monkeypatch.setattr(intraday_module, "vsa_scan", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        intraday_module,
        "relative_volume",
        lambda *args, **kwargs: pd.Series([1.0] * len(df), index=df.index),
    )
    monkeypatch.setattr(
        intraday_module.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    intraday_module.print_dashboard("A0", "豆一", "long", df, {})

    assert any("分钟级无新增执行观察，日线计划不变" in message for message in infos)
    assert all("继续观望" not in message for message in infos)


def test_print_dashboard_labels_minute_signals_as_execution_observation_only(monkeypatch) -> None:
    infos: list[str] = []
    df = _build_minute_df()

    monkeypatch.setattr(
        intraday_module,
        "generate_signals",
        lambda *args, **kwargs: [
            {
                "type": "开多",
                "strength": "强",
                "reason": "回踩后再启动",
                "entry": 4980.0,
                "stop": 4950.0,
                "target": 5050.0,
            },
            {
                "type": "平多",
                "strength": "中",
                "reason": "短线冲高后注意保护",
            },
        ],
    )
    monkeypatch.setattr(intraday_module, "vsa_scan", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        intraday_module,
        "relative_volume",
        lambda *args, **kwargs: pd.Series([1.0] * len(df), index=df.index),
    )
    monkeypatch.setattr(
        intraday_module.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    intraday_module.print_dashboard("A0", "豆一", "long", df, {})

    assert any("分钟级执行观察" in message for message in infos)
    assert any("仅作执行辅助观察，不单独改变日线级计划" in message for message in infos)
    assert any("[执行辅助-偏多]" in message for message in infos)
    assert any("[执行辅助-保护]" in message for message in infos)
    assert all("[开多]" not in message for message in infos)
    assert all("[平多]" not in message for message in infos)


def test_phase_3_intraday_labels_actionable_entry_as_daily_plan(monkeypatch) -> None:
    infos: list[str] = []
    trading_states = iter([True, False])
    printed: list[str] = []

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_minute",
        lambda symbol, period: _build_minute_df(start=4940.0, step=2.0, wick=3.0),
    )
    monkeypatch.setattr(daily_workflow, "print_dashboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: {},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [
            {
                "symbol": "A0",
                "name": "豆一",
                "direction": "long",
                "entry": 5019.0,
                "stop": 4931.0,
                "tp1": 5250.0,
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "entry_signal_detail": "markup 阶段趋势突破",
            }
        ],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[],
    )

    assert any("│ 📌 日线级入场计划:" in message for message in infos)
    assert any("│ 🎯 顺势执行: 价格已触及入场5019附近，继续观察能否站稳且不跌破止损4931" in message for message in infos)
    assert all("│ 📌 入场信号:" not in message for message in infos)
    assert any("终端仅打印等待确认品种变化和持仓调整提醒" in line for line in printed)


def test_phase_3_intraday_orders_actionable_output_as_plan_then_execution_then_minute_observation(monkeypatch) -> None:
    infos: list[str] = []
    trading_states = iter([True, False])

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_minute",
        lambda symbol, period: _build_minute_df(start=4940.0, step=2.0, wick=3.0),
    )
    monkeypatch.setattr(
        daily_workflow,
        "print_dashboard",
        lambda *args, **kwargs: (
            daily_workflow.log.info("│ ⚡ 分钟级执行观察:"),
            daily_workflow.log.info("│    仅作执行辅助观察，不单独改变日线级计划"),
            daily_workflow.log.info("│   🔥 [执行辅助-偏多] 回踩后再启动"),
        ),
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: {},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [
            {
                "symbol": "A0",
                "name": "豆一",
                "direction": "long",
                "entry": 5019.0,
                "stop": 4931.0,
                "tp1": 5250.0,
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "entry_signal_detail": "markup 阶段趋势突破",
            }
        ],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[],
    )

    plan_idx = next(i for i, message in enumerate(infos) if "│ 📌 日线级入场计划:" in message)
    execution_idx = next(i for i, message in enumerate(infos) if "│ 🎯 顺势执行:" in message)
    minute_idx = next(i for i, message in enumerate(infos) if "│ ⚡ 分钟级执行观察:" in message)

    assert plan_idx < execution_idx < minute_idx


def test_phase_3_intraday_labels_reversal_entry_with_reversal_execution_state(monkeypatch) -> None:
    infos: list[str] = []
    trading_states = iter([True, False])

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_minute",
        lambda symbol, period: _build_minute_df(start=4960.0, step=1.0, wick=2.0),
    )
    monkeypatch.setattr(daily_workflow, "print_dashboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: {},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [
            {
                "symbol": "RM0",
                "name": "菜粕",
                "direction": "long",
                "entry": 4970.0,
                "stop": 4920.0,
                "tp1": 5050.0,
                "entry_family": "reversal",
                "entry_signal_type": "Spring",
                "entry_signal_detail": "超跌后止跌回升",
                "reversal_status": {
                    "has_signal": True,
                    "signal_type": "Spring",
                    "signal_bar": {"low": 4935.0},
                },
            }
        ],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[],
    )

    assert any("│ 📌 日线级入场计划:" in message for message in infos)
    assert any("│ 🎯 反转执行: 已守住确认位4935并站回入场4970上方，可按反转执行" in message for message in infos)
