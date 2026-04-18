from __future__ import annotations

from datetime import timedelta

import pandas as pd

from phase2 import pre_market
from backtest.models import BacktestCase, BacktestResult, TradePlan, TradeRecord
from data_cache import _exchange_for_symbol
from phase3.intraday import generate_signals
from market.tq import (
    klines_to_daily_frame,
    resolve_tq_continuous_symbol,
    symbol_to_tq_main,
)


PHASE2_REJECTION_KEYS = (
    "phase2_reject_no_signal_days",
    "phase2_reject_score_gate_days",
    "phase2_reject_rr_gate_days",
    "phase2_reject_duplicate_signal_days",
)

PHASE2_ENTRY_FAMILY_KEYS = (
    "phase2_actionable_reversal_days",
    "phase2_actionable_trend_days",
    "trades_opened_reversal",
    "trades_opened_trend",
)


def _create_backtest_api(*, start_dt, end_dt, account: str, password: str):
    from tqsdk import TqApi, TqAuth, TqBacktest, TqSim

    return TqApi(
        TqSim(),
        backtest=TqBacktest(start_dt=start_dt, end_dt=end_dt),
        auth=TqAuth(account, password),
    )


def resolve_case_tq_symbol(case: BacktestCase, api=None) -> str:
    symbol = case.symbol.strip()
    exchange = _exchange_for_symbol(symbol)
    if not exchange:
        raise ValueError(f"未知品种代码（未在内置列表）: {symbol}")

    try:
        resolved = resolve_tq_continuous_symbol(
            symbol,
            exchange,
            api=api,
            wait_timeout=2.0,
        )
        if resolved:
            return resolved
    except Exception:
        pass
    return symbol_to_tq_main(symbol, exchange)


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "oi"])


def _empty_minute_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def _daily_warmup_bars(config: dict) -> int:
    pre_market_cfg = config.get("pre_market") or {}
    return max(int(pre_market_cfg.get("min_history_bars", 60)), 1)


def _backtest_start_dt(case: BacktestCase, config: dict):
    warmup_bars = _daily_warmup_bars(config)
    return case.start_dt - timedelta(days=warmup_bars * 2)


def _klines_to_minute_frame(klines: pd.DataFrame | None) -> pd.DataFrame:
    if klines is None or len(klines) < 1:
        return _empty_minute_frame()

    df = klines.copy()
    required = ("datetime", "open", "high", "low", "close", "volume")
    for column in required:
        if column not in df.columns:
            return _empty_minute_frame()

    df["datetime"] = pd.to_datetime(df["datetime"])
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    out = df[list(required)].dropna(subset=["close"])
    if out.empty:
        return _empty_minute_frame()
    return out.sort_values("datetime", kind="stable").reset_index(drop=True)


def _drain_backtest_serial_frames(*, api, daily_klines, minute_klines) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tqsdk import BacktestFinished

    daily_df = klines_to_daily_frame(daily_klines)
    minute_df = _klines_to_minute_frame(minute_klines)
    wait_update = getattr(api, "wait_update", None)
    if not callable(wait_update):
        return daily_df, minute_df

    try:
        while True:
            updated = wait_update()
            next_daily_df = klines_to_daily_frame(daily_klines)
            next_minute_df = _klines_to_minute_frame(minute_klines)
            if not next_daily_df.empty:
                daily_df = next_daily_df
            if not next_minute_df.empty:
                minute_df = next_minute_df
            if not updated:
                break
    except BacktestFinished:
        pass
    return daily_df, minute_df


def _filter_daily_range(frame: pd.DataFrame, case: BacktestCase) -> pd.DataFrame:
    if frame.empty:
        return _empty_daily_frame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.loc[out["date"].dt.date <= case.end_dt].copy()
    return out.reset_index(drop=True)


def _filter_minute_range(frame: pd.DataFrame, case: BacktestCase) -> pd.DataFrame:
    if frame.empty:
        return _empty_minute_frame()
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.loc[out["datetime"].dt.date.between(case.start_dt, case.end_dt)].copy()
    return out.reset_index(drop=True)


def load_case_frames_with_tqbacktest(
    *,
    case: BacktestCase,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tq_cfg = config.get("tqsdk") or {}
    account = str(tq_cfg.get("account") or "").strip()
    password = str(tq_cfg.get("password") or "").strip()
    if not account or not password:
        raise ValueError("缺少 tqsdk 账号或密码配置")

    api = None
    try:
        warmup_bars = _daily_warmup_bars(config)
        backtest_start_dt = _backtest_start_dt(case, config)
        api = _create_backtest_api(
            start_dt=backtest_start_dt,
            end_dt=case.end_dt,
            account=account,
            password=password,
        )
        tq_symbol = resolve_case_tq_symbol(case, api=api)
        day_count = max((case.end_dt - case.start_dt).days + 1, 1)
        daily_length = max(day_count + warmup_bars + 5, 30)
        minute_length = max(day_count * 24 * 60, 24 * 60)

        daily_klines = api.get_kline_serial(tq_symbol, 86400, data_length=daily_length)
        minute_klines = api.get_kline_serial(tq_symbol, 60, data_length=minute_length)

        daily_df, minute_df = _drain_backtest_serial_frames(
            api=api,
            daily_klines=daily_klines,
            minute_klines=minute_klines,
        )
        return _filter_daily_range(daily_df, case), _filter_minute_range(minute_df, case)
    finally:
        try:
            if api is not None:
                api.close()
        except Exception:
            pass


def aggregate_partial_5m_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return minute_df.copy()

    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime", kind="stable")
    df["bucket"] = df["datetime"].dt.floor("5min")

    grouped = (
        df.groupby("bucket", sort=True, as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"bucket": "datetime"})
    )
    return grouped


def make_trade_plan_from_phase2(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
) -> TradePlan | None:
    raw = pre_market.build_trade_plan_from_daily_df(
        symbol=case.symbol,
        name=case.name,
        direction=case.direction,
        df=daily_df,
        cfg=pre_market_cfg,
    )
    if raw is None or not raw.get("actionable"):
        return None

    reversal_status = raw.get("reversal_status") or {}
    entry_family = str(raw.get("entry_family") or "").strip()
    entry_signal_type = str(raw.get("entry_signal_type") or reversal_status.get("signal_type") or "")
    entry_signal_detail = str(raw.get("entry_signal_detail") or "")
    signal_date = str(reversal_status.get("signal_date") or "")
    if not signal_date and not daily_df.empty:
        signal_date = str(pd.to_datetime(daily_df["date"]).max().date())
    if not signal_date:
        signal_date = case.start_dt.isoformat()
    return TradePlan(
        trade_id=f"{case.case_id}_{signal_date}",
        symbol=case.symbol,
        direction=case.direction,
        plan_date=signal_date,
        entry_ref=float(raw["entry"]),
        stop=float(raw["stop"]),
        tp1=float(raw["tp1"]),
        tp2=float(raw["tp2"]),
        phase2_score=float(raw["score"]),
        signal_type=entry_signal_type,
        meta={
            "entry_family": entry_family,
            "entry_signal_type": entry_signal_type,
            "entry_signal_detail": entry_signal_detail,
        },
    )


def _empty_phase2_rejection_counts() -> dict[str, int]:
    return {key: 0 for key in PHASE2_REJECTION_KEYS}


def _empty_phase2_entry_family_counts() -> dict[str, int]:
    return {key: 0 for key in PHASE2_ENTRY_FAMILY_KEYS}


def _score_gate_passed(direction: str, total: float) -> bool:
    return (direction == "long" and total > 20) or (direction == "short" and total < -20)


def _entry_family_for_plan(plan: TradePlan | None) -> str:
    if plan is None:
        return ""
    entry_family = str(plan.meta.get("entry_family") or "").strip()
    if entry_family in {"reversal", "trend"}:
        return entry_family
    return ""


def _evaluate_phase2_plan(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
    plan_factory,
) -> tuple[TradePlan | None, dict[str, int]]:
    if plan_factory is not make_trade_plan_from_phase2:
        return plan_factory(case=case, daily_df=daily_df, pre_market_cfg=pre_market_cfg), _empty_phase2_rejection_counts()

    raw = pre_market.build_trade_plan_from_daily_df(
        symbol=case.symbol,
        name=case.name,
        direction=case.direction,
        df=daily_df,
        cfg=pre_market_cfg,
    )
    if raw is None:
        return None, _empty_phase2_rejection_counts()

    if raw.get("actionable"):
        return make_trade_plan_from_phase2(
            case=case,
            daily_df=daily_df,
            pre_market_cfg=pre_market_cfg,
        ), _empty_phase2_rejection_counts()

    rejection_counts = _empty_phase2_rejection_counts()
    reversal = raw.get("reversal_status") or {}
    if not reversal.get("has_signal"):
        rejection_counts["phase2_reject_no_signal_days"] += 1
        return None, rejection_counts

    total = float(raw.get("score") or 0.0)
    rr = float(raw.get("rr") or 0.0)
    if not _score_gate_passed(case.direction, total):
        rejection_counts["phase2_reject_score_gate_days"] += 1
    if rr < 1.0:
        rejection_counts["phase2_reject_rr_gate_days"] += 1
    return None, rejection_counts


def _safe_generate_signals(df: pd.DataFrame, direction: str, cfg: dict) -> list[dict]:
    if len(df) < 2:
        return []
    return generate_signals(df, direction, cfg)


def _calc_pnl_ratio(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def _build_trade_record(
    *,
    case: BacktestCase,
    plan: TradePlan,
    entry_time: pd.Timestamp,
    entry_price: float,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    bars_held: int,
    tp1_hit: bool,
) -> TradeRecord:
    entry_ts = pd.Timestamp(entry_time)
    exit_ts = pd.Timestamp(exit_time)
    return TradeRecord(
        trade_id=plan.trade_id,
        symbol=case.symbol,
        direction=case.direction,
        entry_time=str(entry_ts),
        entry_price=entry_price,
        exit_time=str(exit_ts),
        exit_price=exit_price,
        exit_reason=exit_reason,
        bars_held=bars_held,
        days_held=(exit_ts.date() - entry_ts.date()).days,
        tp1_hit=tp1_hit,
        pnl_ratio=_calc_pnl_ratio(case.direction, entry_price, exit_price),
        meta={
            "plan_date": plan.plan_date,
            "entry_family": _entry_family_for_plan(plan),
            "entry_signal_type": str(plan.meta.get("entry_signal_type") or plan.signal_type or ""),
            "entry_signal_detail": str(plan.meta.get("entry_signal_detail") or ""),
            "signal_type": plan.signal_type,
            "phase2_score": plan.phase2_score,
        },
    )


def _resolve_exit_on_bar(
    *, direction: str, bar: pd.Series, plan: TradePlan
) -> tuple[str, float] | None:
    high = float(bar["high"])
    low = float(bar["low"])

    if direction == "long":
        if low <= plan.stop:
            return "stop", plan.stop
        if high >= plan.tp2:
            return "tp2", plan.tp2
        return None

    if high >= plan.stop:
        return "stop", plan.stop
    if low <= plan.tp2:
        return "tp2", plan.tp2
    return None


def _phase2_min_history_bars(pre_market_cfg: dict) -> int:
    return max(int((pre_market_cfg or {}).get("min_history_bars", 60)), 1)


def run_case_from_frames(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    plan_factory: Callable[..., TradePlan | None] = make_trade_plan_from_phase2,
    pre_market_cfg: dict,
    signal_cfg: dict,
) -> BacktestResult:
    daily_dates = pd.to_datetime(daily_df["date"]) if not daily_df.empty else pd.Series(dtype="datetime64[ns]")
    trades: list[TradeRecord] = []
    diagnostics: dict[str, int] = {
        "daily_rows_loaded": int(len(daily_df)),
        "first_trade_day_visible_daily_rows": 0,
        "trade_days_total": 0,
        "phase2_history_insufficient_days": 0,
        "phase2_non_actionable_days": 0,
        "phase2_actionable_days": 0,
        "phase3_signal_eval_bars": 0,
        "phase3_entry_signal_hits": 0,
        "trades_opened": 0,
    }
    diagnostics.update(_empty_phase2_rejection_counts())
    diagnostics.update(_empty_phase2_entry_family_counts())
    if not daily_df.empty:
        diagnostics["daily_first_date"] = str(daily_dates.min().date())
        diagnostics["daily_last_date"] = str(daily_dates.max().date())
    else:
        diagnostics["daily_first_date"] = "NONE"
        diagnostics["daily_last_date"] = "NONE"

    minute_data = minute_df.copy()
    if minute_data.empty:
        return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": 0}, diagnostics=diagnostics)

    minute_data["datetime"] = pd.to_datetime(minute_data["datetime"])
    minute_data["trade_date"] = minute_data["datetime"].dt.date
    minute_data = minute_data.sort_values("datetime", kind="stable")
    minute_data = minute_data.loc[
        minute_data["trade_date"].between(case.start_dt, case.end_dt)
    ].copy()

    if minute_data.empty:
        return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": 0}, diagnostics=diagnostics)

    visible_daily = daily_df.copy()
    visible_daily["date"] = pd.to_datetime(visible_daily["date"])
    visible_daily = visible_daily.sort_values("date", kind="stable")
    visible_daily["trade_date"] = visible_daily["date"].dt.date

    active_plan: TradePlan | None = None
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    bars_held = 0
    tp1_hit = False
    consumed_trade_ids: set[str] = set()
    min_history_bars = _phase2_min_history_bars(pre_market_cfg)

    for trade_date, day_minutes in minute_data.groupby("trade_date", sort=True):
        diagnostics["trade_days_total"] += 1
        day_minutes = day_minutes.sort_values("datetime", kind="stable")
        day_visible_daily = visible_daily.loc[visible_daily["trade_date"] < trade_date].copy()
        if diagnostics["trade_days_total"] == 1:
            diagnostics["first_trade_day"] = str(trade_date)
            diagnostics["first_trade_day_visible_daily_rows"] = int(len(day_visible_daily))
            if not day_visible_daily.empty:
                diagnostics["first_trade_day_visible_first_date"] = str(day_visible_daily["date"].min().date())
                diagnostics["first_trade_day_visible_last_date"] = str(day_visible_daily["date"].max().date())
            else:
                diagnostics["first_trade_day_visible_first_date"] = "NONE"
                diagnostics["first_trade_day_visible_last_date"] = "NONE"

        if active_plan is None:
            history_insufficient = len(day_visible_daily) < min_history_bars
            day_plan, rejection_counts = _evaluate_phase2_plan(
                case=case,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
                plan_factory=plan_factory,
            )
            if day_plan is not None and day_plan.trade_id in consumed_trade_ids:
                diagnostics["phase2_reject_duplicate_signal_days"] += 1
                day_plan = None
            if history_insufficient:
                diagnostics["phase2_history_insufficient_days"] += 1
            elif day_plan is None:
                diagnostics["phase2_non_actionable_days"] += 1
                for key, value in rejection_counts.items():
                    diagnostics[key] += value
            else:
                diagnostics["phase2_actionable_days"] += 1
                entry_family = _entry_family_for_plan(day_plan)
                if entry_family == "trend":
                    diagnostics["phase2_actionable_trend_days"] += 1
                elif entry_family == "reversal":
                    diagnostics["phase2_actionable_reversal_days"] += 1
        else:
            refreshed_plan = plan_factory(
                case=case,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
            )
            if refreshed_plan is None:
                first_bar = day_minutes.iloc[0]
                trades.append(
                    _build_trade_record(
                        case=case,
                        plan=active_plan,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=first_bar["datetime"],
                        exit_price=float(first_bar["open"]),
                        exit_reason="phase2_invalidated",
                        bars_held=bars_held,
                        tp1_hit=tp1_hit,
                    )
                )
                active_plan = None
                entry_time = None
                entry_price = None
                bars_held = 0
                tp1_hit = False
                continue
            day_plan = None

        for minute_idx in range(len(day_minutes)):
            current_minute = day_minutes.iloc[minute_idx]

            if active_plan is None:
                if day_plan is None:
                    break
                diagnostics["phase3_signal_eval_bars"] += 1
                partial_5m = aggregate_partial_5m_bars(day_minutes.iloc[: minute_idx + 1])
                signals = _safe_generate_signals(partial_5m, case.direction, signal_cfg)
                open_signal = "开多" if case.direction == "long" else "开空"
                if any(signal.get("type") == open_signal for signal in signals):
                    diagnostics["phase3_entry_signal_hits"] += 1
                    consumed_trade_ids.add(day_plan.trade_id)
                    active_plan = day_plan
                    entry_time = pd.Timestamp(current_minute["datetime"])
                    entry_price = float(current_minute["close"])
                    bars_held = 0
                    tp1_hit = False
                    diagnostics["trades_opened"] += 1
                    entry_family = _entry_family_for_plan(day_plan)
                    if entry_family == "trend":
                        diagnostics["trades_opened_trend"] += 1
                    elif entry_family == "reversal":
                        diagnostics["trades_opened_reversal"] += 1
                continue

            bars_held += 1
            last_price = float(current_minute["close"])

            if case.direction == "long":
                tp1_hit = tp1_hit or last_price >= active_plan.tp1
            else:
                tp1_hit = tp1_hit or last_price <= active_plan.tp1
            exit = _resolve_exit_on_bar(
                direction=case.direction,
                bar=current_minute,
                plan=active_plan,
            )

            if exit is None:
                continue
            exit_reason, exit_price = exit

            trades.append(
                _build_trade_record(
                    case=case,
                    plan=active_plan,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=current_minute["datetime"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    tp1_hit=tp1_hit,
                )
            )
            active_plan = None
            entry_time = None
            entry_price = None
            bars_held = 0
            tp1_hit = False
            break

    if active_plan is not None:
        final_bar = minute_data.iloc[-1]
        trades.append(
            _build_trade_record(
                case=case,
                plan=active_plan,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=final_bar["datetime"],
                exit_price=float(final_bar["close"]),
                exit_reason="end_of_data",
                bars_held=bars_held,
                tp1_hit=tp1_hit,
            )
        )

    return BacktestResult(
        case_id=case.case_id,
        trades=trades,
        summary={"num_trades": len(trades)},
        diagnostics=diagnostics,
    )
