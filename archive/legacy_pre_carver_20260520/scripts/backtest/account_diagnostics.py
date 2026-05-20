from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from shared.position_sizing import calculate_position_sizing


AUDIT_COLUMNS = (
    "planned_entry_ref",
    "initial_stop_price",
    "initial_tp1_price",
    "initial_tp2_price",
    "entry_trigger_deviation",
    "entry_rr",
    "slippage",
)


SCORE_BUCKETS = (
    ("<35", 0.0, 35.0),
    ("35-45", 35.0, 45.0),
    ("45-55", 45.0, 55.0),
    ("55-65", 55.0, 65.0),
    ("65+", 65.0, math.inf),
)


HOLD_BUCKETS = (
    ("<1d", 0.0, 1.0),
    ("1-3d", 1.0, 3.0),
    ("3-7d", 3.0, 7.0),
    ("7-14d", 7.0, 14.0),
    ("14d+", 14.0, math.inf),
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _hold_days(row: dict[str, Any]) -> float:
    start = _parse_dt(row.get("entry_time"))
    end = _parse_dt(row.get("actual_exit_time") or row.get("exit_time"))
    if start is None or end is None:
        return 0.0
    return max((end - start).total_seconds() / 86400.0, 0.0)


def _score(row: dict[str, Any]) -> float:
    explicit = _as_float(row.get("phase2_abs_score"), default=math.nan)
    if math.isfinite(explicit):
        return abs(explicit)
    return abs(_as_float(row.get("phase2_score"), 0.0))


def _bucket_label(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str:
    for label, lower, upper in buckets:
        if lower <= value < upper:
            return label
    return buckets[-1][0]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _main_exit_reason(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    counts = Counter(str(row.get("actual_exit_reason") or row.get("exit_reason") or "") for row in rows)
    return counts.most_common(1)[0][0]


def load_trade_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def load_risk_sensitivity(path: str | Path, *, risk_cap: float | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        return payload
    wanted = str(risk_cap)
    for row in payload:
        cap = row.get("risk_cap")
        if isinstance(risk_cap, str):
            if str(cap) == wanted:
                return dict(row)
        elif isinstance(cap, (int, float)) and abs(float(cap) - float(risk_cap)) < 1e-12:
            return dict(row)
    raise ValueError(f"risk cap {risk_cap!r} not found in {path}")


def _portfolio_summary(rows: list[dict[str, Any]], risk_summary: dict[str, Any] | None) -> dict[str, Any]:
    net_pnls = [_as_float(row.get("net_pnl")) for row in rows]
    gross_pnls = [_as_float(row.get("gross_pnl")) for row in rows]
    fees = [_as_float(row.get("fees")) for row in rows]
    wins = sum(1 for pnl in net_pnls if pnl > 0)
    losses = sum(1 for pnl in net_pnls if pnl < 0)
    initial_equity = _as_float((risk_summary or {}).get("initial_equity"))
    if initial_equity <= 0 and rows:
        initial_equity = _as_float(rows[0].get("entry_equity"))
    if initial_equity <= 0:
        initial_equity = 1.0

    max_drawdown = _as_float((risk_summary or {}).get("max_drawdown_realized_pct"))
    return_pct = _as_float((risk_summary or {}).get("return_pct"))
    if return_pct == 0.0:
        return_pct = sum(net_pnls) / initial_equity

    return {
        "initial_equity": float(initial_equity),
        "net_pnl": float(sum(net_pnls)),
        "gross_pnl": float(sum(gross_pnls)),
        "fees": float(sum(fees)),
        "return_pct": float(return_pct),
        "max_drawdown_pct": float(max_drawdown),
        "return_drawdown_ratio": float(return_pct / max_drawdown) if max_drawdown > 0 else 0.0,
        "trades": int(len(rows)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(wins / len(rows)) if rows else 0.0,
        "candidate_trades": int(_as_float((risk_summary or {}).get("candidate_trades"), len(rows))),
        "accepted_trades": int(_as_float((risk_summary or {}).get("accepted_trades"), len(rows))),
        "skipped_trades": int(_as_float((risk_summary or {}).get("skipped_trades"), 0.0)),
        "skipped_reasons": dict((risk_summary or {}).get("skipped_reasons") or {}),
    }


def _symbol_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("symbol") or "")].append(row)

    out: list[dict[str, Any]] = []
    for symbol, symbol_rows in grouped.items():
        net_pnls = [_as_float(row.get("net_pnl")) for row in symbol_rows]
        wins = sum(1 for pnl in net_pnls if pnl > 0)
        margins_pct = []
        for row in symbol_rows:
            equity = _as_float(row.get("entry_equity"))
            margin = _as_float(row.get("margin_used"))
            if equity > 0:
                margins_pct.append(margin / equity)
        out.append(
            {
                "symbol": symbol,
                "name": str(symbol_rows[0].get("name") or symbol),
                "trades": len(symbol_rows),
                "win_rate": wins / len(symbol_rows) if symbol_rows else 0.0,
                "avg_net_pnl": _mean(net_pnls),
                "net_pnl": sum(net_pnls),
                "avg_hold_days": _mean([_hold_days(row) for row in symbol_rows]),
                "avg_margin_pct": _mean(margins_pct),
                "main_exit_reason": _main_exit_reason(symbol_rows),
            }
        )
    return sorted(out, key=lambda item: float(item["net_pnl"]), reverse=True)


def _bucket_summary(
    rows: list[dict[str, Any]],
    *,
    value_fn,
    buckets: tuple[tuple[str, float, float], ...],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in buckets}
    for row in rows:
        grouped[_bucket_label(value_fn(row), buckets)].append(row)

    out: list[dict[str, Any]] = []
    for label, _, _ in buckets:
        bucket_rows = grouped[label]
        net_pnls = [_as_float(row.get("net_pnl")) for row in bucket_rows]
        wins = sum(1 for pnl in net_pnls if pnl > 0)
        out.append(
            {
                "bucket": label,
                "trades": len(bucket_rows),
                "win_rate": wins / len(bucket_rows) if bucket_rows else 0.0,
                "avg_net_pnl": _mean(net_pnls),
                "net_pnl": sum(net_pnls),
            }
        )
    return out


def _is_monotonic_by_avg_net_pnl(score_buckets: list[dict[str, Any]]) -> bool:
    values = [float(row["avg_net_pnl"]) for row in score_buckets if int(row["trades"]) > 0]
    if len(values) < 2:
        return True
    return all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))


def _correlation(xs: list[float], ys: list[float]) -> float:
    paired = [(x, y) for x, y in zip(xs, ys, strict=False) if math.isfinite(x) and math.isfinite(y)]
    if len(paired) < 2:
        return 0.0
    clean_xs = [x for x, _ in paired]
    clean_ys = [y for _, y in paired]
    avg_x = _mean(clean_xs)
    avg_y = _mean(clean_ys)
    numerator = sum((x - avg_x) * (y - avg_y) for x, y in paired)
    denom_x = math.sqrt(sum((x - avg_x) ** 2 for x in clean_xs))
    denom_y = math.sqrt(sum((y - avg_y) ** 2 for y in clean_ys))
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _score_quality(rows: list[dict[str, Any]], score_buckets: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_score(row) for row in rows]
    net_pnls = [_as_float(row.get("net_pnl")) for row in rows]
    r_multiples = []
    for row in rows:
        risk = _as_float(row.get("initial_stop_risk"))
        r_multiples.append(_as_float(row.get("net_pnl")) / risk if risk > 0 else 0.0)
    return {
        "is_monotonic_by_avg_net_pnl": _is_monotonic_by_avg_net_pnl(score_buckets),
        "score_net_pnl_correlation": _correlation(scores, net_pnls),
        "score_r_multiple_correlation": _correlation(scores, r_multiples),
    }


def _exit_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("actual_exit_reason") or row.get("exit_reason") or "")].append(row)
    out = []
    for reason, reason_rows in grouped.items():
        net_pnls = [_as_float(row.get("net_pnl")) for row in reason_rows]
        wins = sum(1 for pnl in net_pnls if pnl > 0)
        out.append(
            {
                "exit_reason": reason,
                "trades": len(reason_rows),
                "win_rate": wins / len(reason_rows) if reason_rows else 0.0,
                "avg_net_pnl": _mean(net_pnls),
                "net_pnl": sum(net_pnls),
                "tp1_hits": sum(1 for row in reason_rows if _as_bool(row.get("tp1_hit"))),
            }
        )
    return sorted(out, key=lambda item: int(item["trades"]), reverse=True)


def _cost_summary(rows: list[dict[str, Any]], *, slippage_bps_per_side: float) -> dict[str, float]:
    fees = sum(_as_float(row.get("fees")) for row in rows)
    gross_pnl = sum(_as_float(row.get("gross_pnl")) for row in rows)
    net_pnl = sum(_as_float(row.get("net_pnl")) for row in rows)
    notional = sum(abs(_as_float(row.get("notional"))) for row in rows)
    estimated_slippage = notional * max(float(slippage_bps_per_side), 0.0) / 10000.0 * 2.0
    return {
        "gross_pnl": float(gross_pnl),
        "fees": float(fees),
        "slippage_bps_per_side": float(slippage_bps_per_side),
        "estimated_slippage": float(estimated_slippage),
        "net_pnl": float(net_pnl),
        "net_pnl_after_slippage": float(net_pnl - estimated_slippage),
    }


def _initial_risk_distance(row: dict[str, Any]) -> float:
    risk_per_lot = _as_float(row.get("risk_per_lot"))
    entry_price = _as_float(row.get("entry_price"))
    notional_per_lot = _as_float(row.get("notional_per_lot"))
    multiplier = notional_per_lot / entry_price if entry_price > 0 and notional_per_lot > 0 else 0.0
    if risk_per_lot > 0 and multiplier > 0:
        return risk_per_lot / multiplier
    return abs(entry_price - _as_float(row.get("actual_exit_price") or row.get("exit_price")))


def _trend_continued_after_exit(
    row: dict[str, Any],
    daily_bars_by_symbol: dict[str, Any],
    *,
    trend_lookahead_days: int,
) -> bool:
    symbol = str(row.get("symbol") or "")
    daily = daily_bars_by_symbol.get(symbol)
    if daily is None or len(daily) == 0:
        return False
    exit_time = _parse_dt(row.get("actual_exit_time") or row.get("exit_time"))
    if exit_time is None:
        return False
    bars = daily.copy()
    if "date" not in bars:
        return False
    bars["date"] = pd.to_datetime(bars["date"])
    start = pd.Timestamp(exit_time.date())
    end = start + pd.Timedelta(days=max(int(trend_lookahead_days), 1))
    window = bars.loc[(bars["date"] > start) & (bars["date"] <= end)]
    if window.empty:
        return False

    exit_price = _as_float(row.get("actual_exit_price") or row.get("exit_price"))
    risk_distance = _initial_risk_distance(row)
    if exit_price <= 0 or risk_distance <= 0:
        return False
    direction = str(row.get("direction") or "")
    if direction == "long":
        high = window["high"] if "high" in window else window["close"]
        return bool(float(high.max()) >= exit_price + risk_distance)
    low = window["low"] if "low" in window else window["close"]
    return bool(float(low.min()) <= exit_price - risk_distance)


def _trend_response_for_trade(
    row: dict[str, Any],
    daily_bars_by_symbol: dict[str, Any],
    *,
    trend_lookahead_days: int,
) -> str:
    exit_reason = str(row.get("actual_exit_reason") or row.get("exit_reason") or "")
    if exit_reason == "tp2":
        return "captured_trend"
    if exit_reason == "replaced_by_higher_phase2_score":
        return "portfolio_rotation_review"
    if exit_reason == "stop":
        continued = _trend_continued_after_exit(
            row,
            daily_bars_by_symbol,
            trend_lookahead_days=trend_lookahead_days,
        )
        return "entry_or_stop_problem" if continued else "false_positive_or_chop"
    return "other_exit_review"


def _trend_response_summary(
    rows: list[dict[str, Any]],
    daily_bars_by_symbol: dict[str, Any] | None,
    *,
    trend_lookahead_days: int,
) -> dict[str, Any]:
    daily_lookup = daily_bars_by_symbol or {}
    responses = [
        {
            "symbol": str(row.get("symbol") or ""),
            "name": str(row.get("name") or row.get("symbol") or ""),
            "response": _trend_response_for_trade(
                row,
                daily_lookup,
                trend_lookahead_days=trend_lookahead_days,
            ),
        }
        for row in rows
    ]
    overall_counts = Counter(item["response"] for item in responses)
    symbol_counts: dict[str, Counter] = defaultdict(Counter)
    symbol_names: dict[str, str] = {}
    for item in responses:
        symbol_counts[item["symbol"]][item["response"]] += 1
        symbol_names[item["symbol"]] = item["name"]
    response_names = sorted(overall_counts)
    return {
        "lookahead_days": int(trend_lookahead_days),
        "overall": [
            {"response": response, "trades": count}
            for response, count in overall_counts.most_common()
        ],
        "by_symbol": [
            {
                "symbol": symbol,
                "name": symbol_names.get(symbol, symbol),
                **{response: int(counts.get(response, 0)) for response in response_names},
            }
            for symbol, counts in sorted(symbol_counts.items())
        ],
    }


def _row_sizing_result(row: dict[str, Any]):
    margin_per_lot = _as_float(row.get("margin_per_lot"))
    risk_per_lot = _as_float(row.get("risk_per_lot"))
    entry_equity = _as_float(row.get("entry_equity"))
    target_margin_pct = _as_float(row.get("target_margin_pct"))
    score_margin_budget = entry_equity * target_margin_pct if entry_equity > 0 and target_margin_pct > 0 else 0.0
    score_lots = None
    if score_margin_budget <= 0 and margin_per_lot > 0 and _as_float(row.get("lots")) > 0:
        score_lots = int(_as_float(row.get("lots")))
    portfolio_lots = int(_as_float(row.get("margin_lots_cap"))) if row.get("margin_lots_cap") not in {None, ""} else None
    risk_lots = int(_as_float(row.get("risk_lots_cap"))) if row.get("risk_lots_cap") not in {None, ""} else None
    risk_budget = 0.0
    if risk_lots is not None and risk_per_lot > 0:
        risk_budget = risk_lots * risk_per_lot
    elif _as_float(row.get("initial_stop_risk")) > 0 and _as_float(row.get("lots")) > 0:
        risk_budget = _as_float(row.get("initial_stop_risk")) / _as_float(row.get("lots"))
    return calculate_position_sizing(
        score_margin_budget=score_margin_budget,
        portfolio_margin_budget=_as_float(row.get("margin_used")),
        risk_budget=risk_budget,
        margin_per_lot=margin_per_lot,
        risk_per_lot=risk_per_lot,
        score_lots=score_lots,
        portfolio_lots=portfolio_lots,
        risk_lots=risk_lots,
    )


def _sizing_limit_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        if not any(row.get(key) not in {None, ""} for key in ("target_margin_pct", "margin_lots_cap", "risk_lots_cap")):
            continue
        sizing = _row_sizing_result(row)
        grouped[sizing.sizing_limited_by].append(sizing)
    out = []
    for reason, sizings in grouped.items():
        out.append(
            {
                "limited_by": reason,
                "trades": len(sizings),
                "zero_lot_trades": sum(1 for sizing in sizings if sizing.suggested_lots < 1),
            }
        )
    return sorted(out, key=lambda item: int(item["trades"]), reverse=True)


def _audit_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set(rows[0].keys()) if rows else set()
    missing = [column for column in AUDIT_COLUMNS if column not in keys]
    return {
        "missing_trade_columns": missing,
        "has_complete_entry_rr": "entry_rr" not in missing,
        "has_trigger_deviation": "entry_trigger_deviation" not in missing,
        "has_slippage": "slippage" not in missing,
    }


def classify_trend_response(
    *,
    trend_available: bool,
    trade_opened: bool,
    suggested_lots: int | None = None,
    exit_reason: str = "",
    trend_continued_after_exit: bool = False,
) -> str:
    if not trend_available and trade_opened:
        return "false_positive_signal"
    if trend_available and not trade_opened:
        if suggested_lots == 0:
            return "account_constraint_zero_lot"
        return "missed_trend_signal"
    if trend_available and trade_opened and exit_reason == "stop" and trend_continued_after_exit:
        return "entry_or_stop_problem"
    if trend_available and trade_opened:
        return "captured_or_management_review"
    return "no_trade_no_trend"


def analyze_account_report(
    trades: list[dict[str, Any]],
    *,
    risk_summary: dict[str, Any] | None = None,
    slippage_bps_per_side: float = 0.0,
    daily_bars_by_symbol: dict[str, Any] | None = None,
    trend_lookahead_days: int = 10,
) -> dict[str, Any]:
    score_buckets = _bucket_summary(trades, value_fn=_score, buckets=SCORE_BUCKETS)
    return {
        "policy": {
            "symbol_is_diagnostic_only": True,
            "decision_rule": "Trade only when trend, reward-risk, cost, margin, and stop-risk gates pass; do not use symbol blacklists.",
        },
        "portfolio": _portfolio_summary(trades, risk_summary),
        "by_symbol": _symbol_summary(trades),
        "score_buckets": score_buckets,
        "score_quality": _score_quality(trades, score_buckets),
        "hold_buckets": _bucket_summary(trades, value_fn=_hold_days, buckets=HOLD_BUCKETS),
        "exit_reasons": _exit_summary(trades),
        "costs": _cost_summary(trades, slippage_bps_per_side=slippage_bps_per_side),
        "sizing_limits": _sizing_limit_summary(trades),
        "trend_response": _trend_response_summary(
            trades,
            daily_bars_by_symbol,
            trend_lookahead_days=trend_lookahead_days,
        ),
        "audit_gaps": _audit_gaps(trades),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(analysis: dict[str, Any]) -> str:
    portfolio = analysis["portfolio"]
    lines = [
        "# Account Backtest Diagnostics",
        "",
        "## 诊断原则",
        "",
        "- 不做品种黑名单；品种只作为诊断切片。",
        "- 统一归因为趋势机会与策略响应：无趋势不交易，有趋势但未抓住则回到信号、入场、止损、持仓和账户约束。",
        "",
        "## 账户概览",
        "",
        f"- 总收益: {portfolio['net_pnl']:.2f}",
        f"- 收益率: {_pct(float(portfolio['return_pct']))}",
        f"- 最大回撤: {_pct(float(portfolio['max_drawdown_pct']))}",
        f"- 收益回撤比: {portfolio['return_drawdown_ratio']:.2f}",
        f"- 候选/接受/跳过: {portfolio['candidate_trades']} / {portfolio['accepted_trades']} / {portfolio['skipped_trades']}",
        "",
        "## 分品种诊断切片",
        "",
        "| 品种 | 交易数 | 胜率 | 平均净收益 | 总贡献 | 平均持仓天数 | 平均保证金占用 | 主要退出 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in analysis["by_symbol"]:
        lines.append(
            "| {name} | {trades} | {win_rate} | {avg_net:.2f} | {net:.2f} | {hold:.2f} | {margin} | {exit} |".format(
                name=row["name"],
                trades=row["trades"],
                win_rate=_pct(float(row["win_rate"])),
                avg_net=float(row["avg_net_pnl"]),
                net=float(row["net_pnl"]),
                hold=float(row["avg_hold_days"]),
                margin=_pct(float(row["avg_margin_pct"])),
                exit=row["main_exit_reason"],
            )
        )
    lines.extend(
        [
            "",
            "## 得分分组",
            "",
            f"- 得分与平均净收益单调: {analysis['score_quality']['is_monotonic_by_avg_net_pnl']}",
            f"- 得分与净收益相关性: {float(analysis['score_quality']['score_net_pnl_correlation']):.3f}",
            f"- 得分与R倍数相关性: {float(analysis['score_quality']['score_r_multiple_correlation']):.3f}",
            "",
            "| 得分段 | 交易数 | 胜率 | 平均净收益 | 总贡献 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["score_buckets"]:
        lines.append(
            "| {bucket} | {trades} | {win_rate} | {avg_net:.2f} | {net:.2f} |".format(
                bucket=row["bucket"],
                trades=row["trades"],
                win_rate=_pct(float(row["win_rate"])),
                avg_net=float(row["avg_net_pnl"]),
                net=float(row["net_pnl"]),
            )
        )
    lines.extend(
        [
            "",
            "## 持仓分组",
            "",
            "| 持仓段 | 交易数 | 胜率 | 平均净收益 | 总贡献 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["hold_buckets"]:
        lines.append(
            "| {bucket} | {trades} | {win_rate} | {avg_net:.2f} | {net:.2f} |".format(
                bucket=row["bucket"],
                trades=row["trades"],
                win_rate=_pct(float(row["win_rate"])),
                avg_net=float(row["avg_net_pnl"]),
                net=float(row["net_pnl"]),
            )
        )
    lines.extend(
        [
            "",
            "## 退出原因",
            "",
            "| 退出原因 | 交易数 | 胜率 | 平均净收益 | 总贡献 | TP1触发数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["exit_reasons"]:
        lines.append(
            "| {reason} | {trades} | {win_rate} | {avg_net:.2f} | {net:.2f} | {tp1} |".format(
                reason=row["exit_reason"],
                trades=row["trades"],
                win_rate=_pct(float(row["win_rate"])),
                avg_net=float(row["avg_net_pnl"]),
                net=float(row["net_pnl"]),
                tp1=row["tp1_hits"],
            )
        )
    lines.extend(["", "## 资金管理跳过", ""])
    skipped_reasons = dict(portfolio.get("skipped_reasons") or {})
    if skipped_reasons:
        for reason, count in sorted(skipped_reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 无跳过原因记录")
    lines.extend(
        [
            "",
            "## 成交手数限制归因",
            "",
            "| 限制来源 | 交易数 | 0手数 |",
            "|---|---:|---:|",
        ]
    )
    if analysis["sizing_limits"]:
        for row in analysis["sizing_limits"]:
            lines.append(f"| {row['limited_by']} | {row['trades']} | {row['zero_lot_trades']} |")
    else:
        lines.append("| 无可复算字段 | 0 | 0 |")
    lines.extend(
        [
            "",
            "## 趋势机会与策略响应",
            "",
            f"- 止损后观察窗口: {analysis['trend_response']['lookahead_days']} 天",
            "",
            "| 响应类型 | 交易数 |",
            "|---|---:|",
        ]
    )
    for row in analysis["trend_response"]["overall"]:
        lines.append(f"| {row['response']} | {row['trades']} |")
    costs = analysis["costs"]
    lines.extend(
        [
            "",
            "## 成本侵蚀",
            "",
            f"- 手续费: {float(costs['fees']):.2f}",
            f"- 滑点假设: 每边 {float(costs['slippage_bps_per_side']):.2f} bp",
            f"- 估算滑点: {float(costs['estimated_slippage']):.2f}",
            f"- 滑点后净收益: {float(costs['net_pnl_after_slippage']):.2f}",
        ]
    )
    if analysis["audit_gaps"]["missing_trade_columns"]:
        lines.extend(
            [
                "",
                "## 审计缺口",
                "",
                "- 当前交易表缺少: " + ", ".join(analysis["audit_gaps"]["missing_trade_columns"]),
            ]
        )
    return "\n".join(lines) + "\n"
