from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest.account_runner import AccountBacktestConfig, AccountCandidate, run_account_backtest
from backtest.execution_quality import medium_term_quality_from_daily, visible_daily_before_entry
from market.contract_specs import builtin_contract_spec


DEFAULT_PHASE23_CACHE_DIR = Path("data/cache/backtest/phase23_task4_split_v1")
DEFAULT_MARKET_CACHE_DIR = Path("data/cache/backtest")
DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/split_execution_factors_2022_2025")

FIRST_RR_BUCKETS = ("lt_1.0", "1.0_1.5", "1.5_2.0", "2.0_plus")
MEDIUM_QUALITY_BUCKETS = ("missing", "lt_50", "50_60", "60_plus")
PATH_BUCKETS = (
    "fast_favorable",
    "trial_then_confirm",
    "deep_recover_to_tp1",
    "steady_to_tp1",
    "early_failure",
    "adverse_no_confirm",
    "slow_or_unclear",
)
ACCOUNT_POLICIES = (
    ("fixed_full", 1.0, "none"),
    ("split_30_tp1", 0.30, "tp1"),
    ("split_50_tp1", 0.50, "tp1"),
    ("split_70_tp1", 0.70, "tp1"),
)
GROUP_FIELDS = (
    "year",
    "entry_rr_bucket",
    "medium_quality_bucket",
    "entry_path_bucket",
    "entry_signal_type",
    "entry_adverse_bucket",
    "entry_location_bucket",
)


@dataclass(frozen=True)
class SplitFactorOutputPaths:
    details_csv: Path
    factor_summary_csv: Path
    account_policy_csv: Path
    summary_json: Path
    report_md: Path


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


def _hold_days(row: dict[str, Any]) -> float:
    start = _parse_ts(row.get("entry_time"))
    end = _parse_ts(row.get("actual_exit_time") or row.get("exit_time"))
    if start is None or end is None:
        return 0.0
    return max((end - start).total_seconds() / 86400.0, 0.0)


def _risk_distance(row: dict[str, Any]) -> float:
    entry = _finite_float(row.get("entry_price"), 0.0)
    stop = _finite_float(
        row.get("initial_stop_price"),
        _finite_float(row.get("planned_stop"), _finite_float(row.get("protective_stop"), 0.0)),
    )
    return abs(entry - stop)


def _empty_path_metrics() -> dict[str, Any]:
    return {
        "mae_1d_r": 0.0,
        "mae_2d_r": 0.0,
        "mae_3d_r": 0.0,
        "mfe_1d_r": 0.0,
        "mfe_2d_r": 0.0,
        "mfe_3d_r": 0.0,
        "tp1_pre_mae_r": 0.0,
        "touched_minus_0p5r_before_tp1": False,
        "touched_minus_1r_before_tp1": False,
        "fast_favorable_1d": False,
        "fast_favorable_3d": False,
        "entry_path_bucket": "slow_or_unclear",
    }


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "datetime" not in frame:
        return pd.DataFrame(columns=list(frame.columns))
    data = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.loc[(data["datetime"] >= start) & (data["datetime"] <= end)].copy()


def _adverse_favorable_r(
    frame: pd.DataFrame,
    *,
    direction: str,
    entry_price: float,
    risk: float,
) -> tuple[float, float]:
    if frame.empty or risk <= 0:
        return 0.0, 0.0
    high = pd.to_numeric(frame.get("high", frame.get("close")), errors="coerce")
    low = pd.to_numeric(frame.get("low", frame.get("close")), errors="coerce")
    if str(direction) == "short":
        adverse = max(float(high.max()) - entry_price, 0.0)
        favorable = max(entry_price - float(low.min()), 0.0)
    else:
        adverse = max(entry_price - float(low.min()), 0.0)
        favorable = max(float(high.max()) - entry_price, 0.0)
    return float(adverse / risk), float(favorable / risk)


def _classify_entry_path(
    *,
    exit_reason: str,
    tp1_hit: bool,
    mae_1d_r: float,
    mae_3d_r: float,
    mfe_1d_r: float,
    touched_minus_0p5r_before_tp1: bool,
    touched_minus_1r_before_tp1: bool,
) -> str:
    if tp1_hit and touched_minus_1r_before_tp1:
        return "deep_recover_to_tp1"
    if str(exit_reason) == "stop" and not tp1_hit and mae_3d_r >= 0.5:
        return "early_failure"
    if tp1_hit and touched_minus_0p5r_before_tp1:
        return "trial_then_confirm"
    if mfe_1d_r >= 0.5 and mae_1d_r <= 0.25:
        return "fast_favorable"
    if tp1_hit:
        return "steady_to_tp1"
    if mae_3d_r >= 0.5:
        return "adverse_no_confirm"
    return "slow_or_unclear"


def compute_path_metrics_for_trade(trade: dict[str, Any], minute_df: pd.DataFrame) -> dict[str, Any]:
    entry_time = _parse_ts(trade.get("entry_time"))
    exit_time = _parse_ts(trade.get("exit_time") or trade.get("planned_exit_time"))
    entry_price = _finite_float(trade.get("entry_price"), 0.0)
    risk = _risk_distance(trade)
    if entry_time is None or exit_time is None or entry_price <= 0 or risk <= 0:
        return _empty_path_metrics()

    out = _empty_path_metrics()
    for days in (1, 2, 3):
        end = min(exit_time, entry_time + pd.Timedelta(days=days))
        adverse, favorable = _adverse_favorable_r(
            _window(minute_df, entry_time, end),
            direction=str(trade.get("direction") or ""),
            entry_price=entry_price,
            risk=risk,
        )
        out[f"mae_{days}d_r"] = adverse
        out[f"mfe_{days}d_r"] = favorable

    tp1_hit = _as_bool(trade.get("tp1_hit"))
    tp1_time = _parse_ts(trade.get("tp1_exit_time"))
    if tp1_hit and tp1_time is not None:
        pre_tp1 = _window(minute_df, entry_time, tp1_time)
        pre_tp1 = pre_tp1.loc[pre_tp1["datetime"] < tp1_time].copy()
        tp1_pre_mae, _tp1_pre_mfe = _adverse_favorable_r(
            pre_tp1,
            direction=str(trade.get("direction") or ""),
            entry_price=entry_price,
            risk=risk,
        )
    else:
        tp1_pre_mae = 0.0
    out["tp1_pre_mae_r"] = tp1_pre_mae
    out["touched_minus_0p5r_before_tp1"] = bool(tp1_hit and tp1_pre_mae >= 0.5)
    out["touched_minus_1r_before_tp1"] = bool(tp1_hit and tp1_pre_mae >= 1.0)
    out["fast_favorable_1d"] = bool(out["mfe_1d_r"] >= 0.5)
    out["fast_favorable_3d"] = bool(out["mfe_3d_r"] >= 0.5)
    out["entry_path_bucket"] = _classify_entry_path(
        exit_reason=str(trade.get("exit_reason") or trade.get("planned_exit_reason") or ""),
        tp1_hit=_as_bool(trade.get("tp1_hit")),
        mae_1d_r=float(out["mae_1d_r"]),
        mae_3d_r=float(out["mae_3d_r"]),
        mfe_1d_r=float(out["mfe_1d_r"]),
        touched_minus_0p5r_before_tp1=bool(out["touched_minus_0p5r_before_tp1"]),
        touched_minus_1r_before_tp1=bool(out["touched_minus_1r_before_tp1"]),
    )
    return out


def first_rr_bucket(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "missing"
    if number < 1.0:
        return "lt_1.0"
    if number < 1.5:
        return "1.0_1.5"
    if number < 2.0:
        return "1.5_2.0"
    return "2.0_plus"


def medium_quality_bucket(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "missing"
    if number < 50:
        return "lt_50"
    if number < 60:
        return "50_60"
    return "60_plus"


def entry_location_bucket(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "missing"
    if number < 40:
        return "overextended_or_poor"
    if number < 70:
        return "acceptable"
    return "good"


def entry_adverse_bucket(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "missing"
    if number <= 0.1:
        return "no_adverse"
    if number <= 0.5:
        return "mild_adverse"
    if number <= 1.0:
        return "stretched"
    return "extreme"


def _entry_adverse_deviation_r(meta: dict[str, Any], row: dict[str, Any]) -> float:
    explicit = _finite_float(meta.get("entry_adverse_deviation_r"))
    if math.isfinite(explicit):
        return max(explicit, 0.0)
    planned_entry = _finite_float(meta.get("planned_entry_ref"), 0.0)
    planned_stop = _finite_float(meta.get("planned_stop") or meta.get("initial_stop_price"), 0.0)
    deviation = _finite_float(meta.get("entry_trigger_deviation"), 0.0)
    risk = abs(planned_entry - planned_stop)
    if risk <= 0:
        return 0.0
    adverse = deviation if str(row.get("direction")) == "long" else -deviation
    return float(max(adverse, 0.0) / risk)


def _market_cache_path(symbol: str, kind: str, market_cache_dir: Path) -> Path:
    return Path(market_cache_dir) / f"{str(symbol).lower()}_20220101_20251231_w60_v3_{kind}.parquet"


def _read_market_frame(
    symbol: str,
    kind: str,
    market_cache_dir: Path,
    cache: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    key = (str(symbol).upper(), kind)
    if key not in cache:
        path = _market_cache_path(symbol, kind, market_cache_dir)
        cache[key] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return cache[key]


def _float_spec(spec: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _finite_float(spec.get(key), default)


def _trade_row_from_cache_record(
    trade: dict[str, Any],
    *,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
) -> dict[str, Any]:
    meta = dict(trade.get("meta") or {})
    symbol = str(trade.get("symbol") or "").upper()
    spec = builtin_contract_spec(symbol)
    multiplier = _finite_float(meta.get("contract_multiplier"), _float_spec(spec, "multiplier", 0.0))
    entry_price = _finite_float(trade.get("entry_price"), 0.0)
    notional_per_lot = entry_price * max(multiplier, 0.0)
    margin_per_lot = _finite_float(meta.get("execution_margin_per_lot"), 0.0)
    if margin_per_lot <= 0 and notional_per_lot > 0:
        margin_per_lot = notional_per_lot * _float_spec(spec, "margin_rate", 0.0)

    quality = medium_term_quality_from_daily(
        visible_daily_before_entry(daily_df, str(trade.get("entry_time") or "")),
        direction=str(trade.get("direction") or ""),
        trend_phase=str(meta.get("trend_phase") or ""),
    )
    medium_score = _finite_float(quality.get("medium_term_quality_score"))
    entry_location = _finite_float(quality.get("medium_term_entry_location_score"))
    row = {
        "year": int(pd.Timestamp(trade.get("entry_time")).year),
        "trade_id": str(trade.get("trade_id") or ""),
        "symbol": symbol,
        "name": symbol,
        "direction": str(trade.get("direction") or ""),
        "entry_time": str(trade.get("entry_time") or ""),
        "exit_time": str(trade.get("exit_time") or ""),
        "exit_reason": str(trade.get("exit_reason") or ""),
        "entry_price": entry_price,
        "exit_price": _finite_float(trade.get("exit_price"), 0.0),
        "pnl_ratio": _finite_float(trade.get("pnl_ratio"), 0.0),
        "tp1_hit": bool(trade.get("tp1_hit")),
        "tp1_exit_fraction": _finite_float(meta.get("tp1_exit_fraction"), 0.0),
        "tp1_exit_price": _finite_float(meta.get("tp1_exit_price"), 0.0),
        "tp1_exit_time": str(meta.get("tp1_exit_time") or ""),
        "phase2_score": _finite_float(meta.get("phase2_score"), 0.0),
        "entry_rr": _finite_float(meta.get("entry_rr"), _finite_float(meta.get("execution_rr"), _finite_float(meta.get("rr"), 0.0))),
        "entry_admission_rr": _finite_float(
            meta.get("entry_admission_rr"),
            _finite_float(meta.get("execution_admission_rr"), _finite_float(meta.get("admission_rr"), 0.0)),
        ),
        "planned_entry_ref": _finite_float(meta.get("planned_entry_ref"), 0.0),
        "planned_stop": _finite_float(meta.get("planned_stop"), 0.0),
        "initial_stop_price": _finite_float(meta.get("initial_stop_price"), _finite_float(meta.get("planned_stop"), 0.0)),
        "initial_tp1_price": _finite_float(meta.get("initial_tp1_price"), _finite_float(meta.get("tp1_exit_price"), 0.0)),
        "initial_tp2_price": _finite_float(meta.get("initial_tp2_price"), 0.0),
        "entry_trigger_deviation": _finite_float(meta.get("entry_trigger_deviation"), 0.0),
        "entry_trigger_deviation_pct": _finite_float(meta.get("entry_trigger_deviation_pct"), 0.0),
        "risk_per_lot": _finite_float(meta.get("execution_risk_per_lot"), _finite_float(meta.get("risk_per_lot"), 0.0)),
        "margin_per_lot": margin_per_lot,
        "notional_per_lot": notional_per_lot,
        "multiplier": multiplier,
        "commission_per_lot": _float_spec(spec, "commission_per_lot", _float_spec(spec, "commission", 0.0)),
        "commission_rate": _float_spec(spec, "commission_rate", 0.0),
        "close_today_commission_per_lot": _float_spec(spec, "close_today_commission_per_lot", math.nan),
        "medium_term_quality_score": medium_score,
        "medium_term_entry_location_score": entry_location,
        "entry_signal_type": str(meta.get("entry_signal_type") or meta.get("signal_type") or ""),
        "entry_signal_detail": str(meta.get("entry_signal_detail") or ""),
        "management_profile": str(meta.get("management_profile") or ""),
        "protective_stop": _finite_float(meta.get("protective_stop"), 0.0),
    }
    row["entry_adverse_deviation_r"] = _entry_adverse_deviation_r(meta, row)
    row.update(compute_path_metrics_for_trade(row, minute_df))
    row["entry_rr_bucket"] = first_rr_bucket(row["entry_rr"])
    row["medium_quality_bucket"] = medium_quality_bucket(row["medium_term_quality_score"])
    row["entry_adverse_bucket"] = entry_adverse_bucket(row["entry_adverse_deviation_r"])
    row["entry_location_bucket"] = entry_location_bucket(row["medium_term_entry_location_score"])
    row["early_stop"] = bool(row["exit_reason"] == "stop" and _hold_days(row) < 3.0)
    row["tp2_winner"] = bool(row["exit_reason"] == "tp2" and row["pnl_ratio"] > 0)
    return row


def collect_split_factor_rows(
    *,
    years: Iterable[int],
    phase23_cache_dir: Path = DEFAULT_PHASE23_CACHE_DIR,
    market_cache_dir: Path = DEFAULT_MARKET_CACHE_DIR,
) -> pd.DataFrame:
    wanted_years = {int(year) for year in years}
    market_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(phase23_cache_dir).glob("*_phase23_cache.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbol = str(payload.get("symbol") or "").upper()
        daily_df = _read_market_frame(symbol, "daily", market_cache_dir, market_cache)
        minute_df = _read_market_frame(symbol, "minute", market_cache_dir, market_cache)
        for trade in ((payload.get("result") or {}).get("trades") or []):
            entry_time = pd.Timestamp(trade.get("entry_time"))
            if wanted_years and int(entry_time.year) not in wanted_years:
                continue
            rows.append(_trade_row_from_cache_record(trade, daily_df=daily_df, minute_df=minute_df))
    return pd.DataFrame(rows)


def _candidate_from_row(row: dict[str, Any]) -> AccountCandidate:
    close_today = row.get("close_today_commission_per_lot")
    close_today_fee = None if not math.isfinite(_finite_float(close_today)) else _finite_float(close_today)
    return AccountCandidate(
        symbol=str(row.get("symbol") or ""),
        name=str(row.get("name") or row.get("symbol") or ""),
        direction=str(row.get("direction") or ""),
        entry_time=str(row.get("entry_time") or ""),
        planned_exit_time=str(row.get("exit_time") or ""),
        entry_price=_finite_float(row.get("entry_price"), 0.0),
        planned_exit_price=_finite_float(row.get("exit_price"), 0.0),
        planned_exit_reason=str(row.get("exit_reason") or ""),
        phase2_score=_finite_float(row.get("phase2_score"), 0.0),
        risk_per_lot=_finite_float(row.get("risk_per_lot"), 0.0),
        margin_per_lot=_finite_float(row.get("margin_per_lot"), 0.0),
        notional_per_lot=_finite_float(row.get("notional_per_lot"), 0.0),
        multiplier=_finite_float(row.get("multiplier"), 0.0),
        pnl_ratio_price=_finite_float(row.get("pnl_ratio"), 0.0),
        tp1_hit=_as_bool(row.get("tp1_hit")),
        commission_per_lot=_finite_float(row.get("commission_per_lot"), 0.0),
        commission_rate=_finite_float(row.get("commission_rate"), 0.0),
        close_today_commission_per_lot=close_today_fee,
        trade_id=str(row.get("trade_id") or ""),
        entry_rr=_finite_float(row.get("entry_rr"), 0.0),
        entry_admission_rr=_finite_float(row.get("entry_admission_rr"), 0.0),
        medium_term_quality_score=_finite_float(row.get("medium_term_quality_score"), 0.0),
        medium_term_entry_location_score=_finite_float(row.get("medium_term_entry_location_score"), 0.0),
        entry_adverse_deviation_r=_finite_float(row.get("entry_adverse_deviation_r"), 0.0),
        tp1_price=_finite_float(row.get("tp1_exit_price"), _finite_float(row.get("initial_tp1_price"), 0.0)),
        tp1_time=str(row.get("tp1_exit_time") or ""),
    )


def _avg_margin_pct(equity_curve: list[dict[str, Any]]) -> float:
    values = [
        _finite_float(row.get("active_margin"), 0.0) / _finite_float(row.get("equity"), 1.0)
        for row in equity_curve
        if _finite_float(row.get("equity"), 0.0) > 0
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _account_early_stop_count(trades: list[dict[str, Any]]) -> int:
    return sum(1 for trade in trades if str(trade.get("actual_exit_reason")) == "stop" and _hold_days(trade) < 3.0)


def _account_policy_row(
    *,
    group_field: str,
    group_value: str,
    policy: str,
    max_portfolio_margin_pct: float,
    source_rows: pd.DataFrame,
    initial_equity: float,
    split_initial_fraction: float,
    split_second_entry_trigger: str,
) -> dict[str, Any]:
    candidates = [_candidate_from_row(row) for row in source_rows.to_dict("records")]
    result = run_account_backtest(
        candidates,
        AccountBacktestConfig(
            initial_equity=float(initial_equity),
            max_portfolio_margin_pct=float(max_portfolio_margin_pct),
            risk_per_trade_pct=0.015,
            split_initial_fraction=float(split_initial_fraction),
            split_second_entry_trigger=split_second_entry_trigger,
            commission_multiplier=1.01,
            scope=f"Task A split factor validation: {group_field}={group_value}",
        ),
    )
    candidate_tp2 = int((source_rows["exit_reason"].astype(str) == "tp2").sum()) if len(source_rows) else 0
    account_tp2 = sum(1 for trade in result.trades if str(trade.get("actual_exit_reason")) == "tp2")
    return {
        "group_field": group_field,
        "group_value": str(group_value),
        "policy": policy,
        "max_portfolio_margin_pct": float(max_portfolio_margin_pct),
        "candidate_trades": int(len(source_rows)),
        "accepted_trades": int(result.summary.get("accepted_trades", 0)),
        "return_pct": float(result.summary.get("return_pct", 0.0)),
        "net_profit": float(result.summary.get("net_profit", 0.0)),
        "max_drawdown_pct": float(result.summary.get("max_drawdown_realized_pct", 0.0)),
        "early_stop_trades": int(_account_early_stop_count(result.trades)),
        "tp2_trades": int(account_tp2),
        "tp2_capture_rate": float(account_tp2 / candidate_tp2) if candidate_tp2 else 0.0,
        "max_margin_pct_observed": float(result.summary.get("max_margin_pct_observed", 0.0)),
        "avg_margin_pct_observed": _avg_margin_pct(result.equity_curve),
        "total_fees": float(result.summary.get("total_fees", 0.0)),
        "split_target_lots": int(result.summary.get("split_target_lots", 0)),
        "split_first_entry_lots": int(result.summary.get("split_first_entry_lots", 0)),
        "split_second_entry_filled_lots": int(result.summary.get("split_second_entry_filled_lots", 0)),
        "split_second_entry_unfilled_lots": int(result.summary.get("split_second_entry_unfilled_lots", 0)),
    }


def account_policy_rows_for_group(
    rows: pd.DataFrame,
    *,
    group_field: str,
    group_value: Any,
    initial_equity: float = 1_000_000.0,
    max_portfolio_margin_pct: float = 0.30,
) -> list[dict[str, Any]]:
    source_rows = rows[rows[group_field].astype(str) == str(group_value)].copy() if group_field in rows else rows.copy()
    out: list[dict[str, Any]] = []
    fixed_net_profit = 0.0
    fixed_return = 0.0
    for policy, fraction, trigger in ACCOUNT_POLICIES:
        row = _account_policy_row(
            group_field=group_field,
            group_value=str(group_value),
            policy=policy,
            max_portfolio_margin_pct=max_portfolio_margin_pct,
            source_rows=source_rows,
            initial_equity=initial_equity,
            split_initial_fraction=fraction,
            split_second_entry_trigger=trigger,
        )
        if policy == "fixed_full":
            fixed_net_profit = float(row["net_profit"])
            fixed_return = float(row["return_pct"])
        row["net_profit_delta_vs_fixed"] = float(row["net_profit"] - fixed_net_profit)
        row["return_delta_vs_fixed"] = float(row["return_pct"] - fixed_return)
        out.append(row)
    return out


def _group_values(rows: pd.DataFrame, field: str) -> list[Any]:
    if field == "entry_rr_bucket":
        return [value for value in FIRST_RR_BUCKETS if value in set(rows[field].astype(str))]
    if field == "medium_quality_bucket":
        return [value for value in MEDIUM_QUALITY_BUCKETS if value in set(rows[field].astype(str))]
    if field == "entry_path_bucket":
        return [value for value in PATH_BUCKETS if value in set(rows[field].astype(str))]
    if field == "entry_adverse_bucket":
        order = ("missing", "no_adverse", "mild_adverse", "stretched", "extreme")
        return [value for value in order if value in set(rows[field].astype(str))]
    if field == "entry_location_bucket":
        order = ("missing", "overextended_or_poor", "acceptable", "good")
        return [value for value in order if value in set(rows[field].astype(str))]
    return sorted(rows[field].dropna().astype(str).unique().tolist())


def build_account_policy_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for max_margin in (0.30, 0.50):
        out.extend(
            account_policy_rows_for_group(
                rows,
                group_field="all",
                group_value="all",
                max_portfolio_margin_pct=max_margin,
            )
        )
        for field in GROUP_FIELDS:
            if field not in rows:
                continue
            for value in _group_values(rows, field):
                out.extend(
                    account_policy_rows_for_group(
                        rows,
                        group_field=field,
                        group_value=value,
                        max_portfolio_margin_pct=max_margin,
                    )
                )
        all_fixed = next(
            row
            for row in out
            if row["group_field"] == "all"
            and row["group_value"] == "all"
            and row["policy"] == "fixed_full"
            and float(row["max_portfolio_margin_pct"]) == float(max_margin)
        )
        if "entry_path_bucket" in rows:
            for value in _group_values(rows, "entry_path_bucket"):
                filtered = rows[rows["entry_path_bucket"].astype(str) != str(value)].copy()
                row = _account_policy_row(
                    group_field="exclude_entry_path_bucket",
                    group_value=value,
                    policy="filter_bucket_fixed_full",
                    max_portfolio_margin_pct=max_margin,
                    source_rows=filtered,
                    initial_equity=1_000_000.0,
                    split_initial_fraction=1.0,
                    split_second_entry_trigger="none",
                )
                row["net_profit_delta_vs_fixed"] = float(row["net_profit"] - all_fixed["net_profit"])
                row["return_delta_vs_fixed"] = float(row["return_pct"] - all_fixed["return_pct"])
                out.append(row)
    return out


def _mean(frame: pd.DataFrame, field: str) -> float | None:
    values = pd.to_numeric(frame.get(field, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _rate(frame: pd.DataFrame, field: str) -> float:
    if field not in frame or len(frame) == 0:
        return 0.0
    return float(frame[field].astype(bool).mean())


def _factor_summary_row(frame: pd.DataFrame, *, group_field: str, group_value: Any) -> dict[str, Any]:
    return {
        "group_field": group_field,
        "group_value": str(group_value),
        "trades": int(len(frame)),
        "early_stop_rate": _rate(frame, "early_stop"),
        "tp1_hit_rate": _rate(frame, "tp1_hit"),
        "tp2_rate": _rate(frame, "tp2_winner"),
        "avg_pnl_ratio": _mean(frame, "pnl_ratio"),
        "avg_mae_1d_r": _mean(frame, "mae_1d_r"),
        "avg_mae_3d_r": _mean(frame, "mae_3d_r"),
        "avg_mfe_1d_r": _mean(frame, "mfe_1d_r"),
        "avg_mfe_3d_r": _mean(frame, "mfe_3d_r"),
        "touch_minus_0p5_before_tp1_rate": _rate(frame, "touched_minus_0p5r_before_tp1"),
        "touch_minus_1r_before_tp1_rate": _rate(frame, "touched_minus_1r_before_tp1"),
        "avg_entry_adverse_deviation_r": _mean(frame, "entry_adverse_deviation_r"),
        "avg_medium_quality_score": _mean(frame, "medium_term_quality_score"),
        "avg_entry_location_score": _mean(frame, "medium_term_entry_location_score"),
    }


def build_factor_summary_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out = [_factor_summary_row(rows, group_field="all", group_value="all")]
    for field in GROUP_FIELDS:
        if field not in rows:
            continue
        for value in _group_values(rows, field):
            group = rows[rows[field].astype(str) == str(value)].copy()
            out.append(_factor_summary_row(group, group_field=field, group_value=value))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _pct(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _money(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:,.0f}"


def _rows_where(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = rows
    for key, value in filters.items():
        out = [row for row in out if str(row.get(key)) == str(value)]
    return out


def _best_split_delta(account_rows: list[dict[str, Any]], *, group_value: str, policy: str = "split_30_tp1") -> float:
    matches = _rows_where(
        account_rows,
        group_field="entry_path_bucket",
        group_value=group_value,
        policy=policy,
        max_portfolio_margin_pct=0.3,
    )
    return _finite_float(matches[0].get("net_profit_delta_vs_fixed"), 0.0) if matches else 0.0


def _best_split_evidence(account_rows: list[dict[str, Any]], *, group_value: str) -> tuple[str, float]:
    best_policy = ""
    best_delta = -math.inf
    for policy in ("split_30_tp1", "split_50_tp1", "split_70_tp1"):
        delta = _best_split_delta(account_rows, group_value=group_value, policy=policy)
        if delta > best_delta:
            best_policy = policy
            best_delta = delta
    return best_policy, float(best_delta if math.isfinite(best_delta) else 0.0)


def _filter_delta(account_rows: list[dict[str, Any]], *, group_value: str) -> float:
    matches = _rows_where(
        account_rows,
        group_field="exclude_entry_path_bucket",
        group_value=group_value,
        policy="filter_bucket_fixed_full",
        max_portfolio_margin_pct=0.3,
    )
    return _finite_float(matches[0].get("net_profit_delta_vs_fixed"), 0.0) if matches else 0.0


def _report_answer_rows(factor_rows: list[dict[str, Any]], account_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    path_rows = _rows_where(factor_rows, group_field="entry_path_bucket")
    direct = []
    trial = []
    trial_not_improved = []
    no_split = []
    filter_ = []
    for row in path_rows:
        bucket = str(row["group_value"])
        early = _finite_float(row.get("early_stop_rate"), 0.0)
        tp2 = _finite_float(row.get("tp2_rate"), 0.0)
        tp1 = _finite_float(row.get("tp1_hit_rate"), 0.0)
        best_policy, best_delta = _best_split_evidence(account_rows, group_value=bucket)
        filter_delta = _filter_delta(account_rows, group_value=bucket)
        label = (
            f"{bucket}：早止损{_pct(early)}，TP1{_pct(tp1)}，TP2{_pct(tp2)}，"
            f"最佳分仓{best_policy}相对满仓{_money(best_delta)}"
        )
        if bucket in {"fast_favorable", "steady_to_tp1"} and best_delta < 0:
            direct.append(label)
        if bucket in {"trial_then_confirm", "deep_recover_to_tp1"}:
            if best_delta > 0:
                trial.append(label)
            else:
                trial_not_improved.append(label)
        if tp2 >= 0.25 and best_delta < 0:
            no_split.append(label)
        if early >= 0.45 and tp1 < 0.25 and filter_delta > 0:
            filter_.append(
                f"{bucket}：早止损{_pct(early)}，TP1{_pct(tp1)}，TP2{_pct(tp2)}，过滤后相对全样本满仓{_money(filter_delta)}"
            )
    return {
        "direct": direct[:4],
        "trial": trial[:4],
        "trial_not_improved": trial_not_improved[:4],
        "no_split": no_split[:4],
        "filter": filter_[:4],
    }


def render_markdown(
    *,
    rows: pd.DataFrame,
    factor_rows: list[dict[str, Any]],
    account_rows: list[dict[str, Any]],
) -> str:
    all_summary = _factor_summary_row(rows, group_field="all", group_value="all")
    all_account = _rows_where(account_rows, group_field="all", group_value="all")
    path_factors = _rows_where(factor_rows, group_field="entry_path_bucket")
    year_policy = _rows_where(account_rows, group_field="year", max_portfolio_margin_pct=0.3)
    rr_factors = _rows_where(factor_rows, group_field="entry_rr_bucket")
    quality_factors = _rows_where(factor_rows, group_field="medium_quality_bucket")
    signal_factors = _rows_where(factor_rows, group_field="entry_signal_type")
    adverse_factors = _rows_where(factor_rows, group_field="entry_adverse_bucket")
    location_factors = _rows_where(factor_rows, group_field="entry_location_bucket")
    filter_policy = _rows_where(
        account_rows,
        group_field="exclude_entry_path_bucket",
        policy="filter_bucket_fixed_full",
        max_portfolio_margin_pct=0.3,
    )
    answers = _report_answer_rows(factor_rows, account_rows)

    lines = [
        "# Task A 分仓因子验证 2022-2025",
        "",
        "口径：只做分析和回测参数验证，不改变默认策略。样本为 Phase23 趋势候选交易；仓位仍遵守 Phase2 理论仓位、组合保证金、单笔止损风险三者取最小，手续费按交易所口径乘以 1.01。",
        "",
        "## 总览",
        "",
        f"- 候选交易：{len(rows)} 笔。",
        f"- TP1 命中率：{_pct(all_summary['tp1_hit_rate'])}。",
        f"- TP2 捕获率：{_pct(all_summary['tp2_rate'])}。",
        f"- 早止损率：{_pct(all_summary['early_stop_rate'])}。",
        f"- 入场后 3 天平均最大逆行：{_num(all_summary['avg_mae_3d_r'], 3)} R。",
        f"- 入场后 3 天平均最大有利：{_num(all_summary['avg_mfe_3d_r'], 3)} R。",
        "",
        "## 全样本分仓反事实",
        "",
        "| 保证金上限 | 执行方式 | 收益 | 最大回撤 | 净利润 | 早止损 | TP2捕获 | 最大资金占用 | 二笔成交手数 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(all_account, key=lambda item: (float(item["max_portfolio_margin_pct"]), str(item["policy"]))):
        lines.append(
            f"| {_pct(row['max_portfolio_margin_pct'])} | {row['policy']} | {_pct(row['return_pct'])} | "
            f"{_pct(row['max_drawdown_pct'])} | {_money(row['net_profit'])} | {row['early_stop_trades']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['max_margin_pct_observed'])} | "
            f"{row['split_second_entry_filled_lots']} |"
        )

    lines.extend(
        [
            "",
            "## 入场路径桶",
            "",
            "| 路径桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 1日MAE | 3日MAE | 3日MFE | 先-0.5R再TP1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in path_factors:
        lines.append(
            f"| {row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_mae_1d_r'], 3)} | {_num(row['avg_mae_3d_r'], 3)} | "
            f"{_num(row['avg_mfe_3d_r'], 3)} | {_pct(row['touch_minus_0p5_before_tp1_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 过滤路径桶反事实",
            "",
            "这是账户级过滤实验：从全样本中排除某个路径桶，再按满仓口径重跑账户。正数表示过滤后比全样本满仓更好。",
            "",
            "| 被过滤路径桶 | 收益 | 最大回撤 | 净利润 | 相对全样本满仓 | 早止损 | TP2捕获 | 最大资金占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in filter_policy:
        lines.append(
            f"| {row['group_value']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | "
            f"{_money(row['net_profit'])} | {_money(row['net_profit_delta_vs_fixed'])} | "
            f"{row['early_stop_trades']} | {_pct(row['tp2_capture_rate'])} | {_pct(row['max_margin_pct_observed'])} |"
        )

    lines.extend(
        [
            "",
            "## 分年份账户对比",
            "",
            "| 年份 | 执行方式 | 收益 | 最大回撤 | 净利润 | 早止损 | TP2捕获 | 最大资金占用 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(year_policy, key=lambda item: (str(item["group_value"]), str(item["policy"]))):
        if row["policy"] not in {"fixed_full", "split_30_tp1"}:
            continue
        lines.append(
            f"| {row['group_value']} | {row['policy']} | {_pct(row['return_pct'])} | "
            f"{_pct(row['max_drawdown_pct'])} | {_money(row['net_profit'])} | "
            f"{row['early_stop_trades']} | {_pct(row['tp2_capture_rate'])} | "
            f"{_pct(row['max_margin_pct_observed'])} |"
        )

    lines.extend(
        [
            "",
            "## 第一 RR 与中期质量分桶",
            "",
            "| 分桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 3日MAE | 3日MFE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rr_factors + quality_factors:
        lines.append(
            f"| {row['group_field']}={row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_mae_3d_r'], 3)} | {_num(row['avg_mfe_3d_r'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## 入场信号类型",
            "",
            "| 信号类型 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 3日MAE | 3日MFE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in signal_factors:
        lines.append(
            f"| {row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_mae_3d_r'], 3)} | {_num(row['avg_mfe_3d_r'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## 入场偏离与过度延伸",
            "",
            "| 分桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 3日MAE | 3日MFE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in adverse_factors + location_factors:
        lines.append(
            f"| {row['group_field']}={row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_mae_3d_r'], 3)} | {_num(row['avg_mfe_3d_r'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## 四个问题的当前答案",
            "",
            "1. 能区分小仓试错和直接大仓的指标：入场后 1-3 天逆行、是否先到 -0.5R 后再 TP1、快速有利突破、TP1 前最大回撤、真实触发价相对计划价的偏离，比第一 RR 更贴近真实持仓路径。",
            "2. 第一笔小仓后，第二笔确认加仓能提高收益的类型：",
        ]
    )
    lines.extend([f"   - {item}" for item in answers["trial"]] or ["   - 当前样本没有显示稳定正向的路径桶。"])
    if answers["trial_not_improved"]:
        lines.append("   - 需要注意：有些路径看起来适合等待确认，但账户级分仓仍没有增收。")
        lines.extend([f"   - {item}" for item in answers["trial_not_improved"]])
    lines.append("3. 不该分仓、容易错过趋势的类型：")
    lines.extend([f"   - {item}" for item in answers["no_split"]] or ["   - 当前样本未出现明确桶。"])
    lines.append("4. 更接近过滤而不是 30% 试错的类型：")
    lines.extend([f"   - {item}" for item in answers["filter"]] or ["   - 当前样本未出现明确桶。"])
    lines.extend(
        [
            "",
            "## 下一步候选回测参数",
            "",
            "- 直接大仓参数候选：入场后快速有利突破且 1 日 MAE 不超过 0.25R 的交易，不做 30% 分仓削弱。",
            "- 确认加仓参数候选：第一笔小仓后，只有在未触发 -1R 且重新创有利新高/新低或触发 TP1 时，才补第二笔。",
            "- 过滤参数候选：入场后 3 天已触发 -0.5R 且没有任何快速有利突破的交易，优先作为早期淘汰/不补仓实验，而不是默认 30% 试错。",
            "- 以上都只是下一轮回测参数，不进入默认实盘配置。",
            "",
            "## 使用边界",
            "",
            "- 路径桶包含入场后的信息，不能直接当作入场前规则；它的用途是设计下一轮“确认加仓、早期淘汰、不补仓”的回测参数。",
            "- 过滤某路径桶的账户反事实是事后诊断，收益数字会放大路径识别的价值，不能直接视为可交易结果。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_split_execution_factors(rows: pd.DataFrame) -> dict[str, Any]:
    factor_rows = build_factor_summary_rows(rows)
    account_rows = build_account_policy_rows(rows)
    return {
        "diagnostics": {
            "trade_count": int(len(rows)),
            "years": {str(year): int(count) for year, count in rows.groupby("year").size().items()},
            "entry_signal_types": {
                str(signal): int(count) for signal, count in rows.groupby("entry_signal_type").size().items()
            },
            "entry_path_buckets": {
                str(bucket): int(count) for bucket, count in rows.groupby("entry_path_bucket").size().items()
            },
        },
        "factor_summary": factor_rows,
        "account_policy": account_rows,
    }


def write_outputs(rows: pd.DataFrame, report: dict[str, Any], output_prefix: Path) -> SplitFactorOutputPaths:
    prefix = Path(output_prefix)
    paths = SplitFactorOutputPaths(
        details_csv=prefix.with_name(f"{prefix.name}_details.csv"),
        factor_summary_csv=prefix.with_name(f"{prefix.name}_factor_summary.csv"),
        account_policy_csv=prefix.with_name(f"{prefix.name}_account_policy.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.details_csv, rows)
    _write_csv(paths.factor_summary_csv, report["factor_summary"])
    _write_csv(paths.account_policy_csv, report["account_policy"])
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    paths.report_md.write_text(
        render_markdown(rows=rows, factor_rows=report["factor_summary"], account_rows=report["account_policy"]),
        encoding="utf-8",
    )
    return paths


def _parse_years(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate split execution factors for trend trades")
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--phase23-cache-dir", type=Path, default=DEFAULT_PHASE23_CACHE_DIR)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = collect_split_factor_rows(
        years=_parse_years(args.years),
        phase23_cache_dir=args.phase23_cache_dir,
        market_cache_dir=args.market_cache_dir,
    )
    report = analyze_split_execution_factors(rows)
    paths = write_outputs(rows, report, args.output_prefix)
    for path in asdict(paths).values():
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
