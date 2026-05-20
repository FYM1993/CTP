from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from market.contract_specs import builtin_contract_spec
from shared.position_sizing import calculate_position_sizing


DEFAULT_STATES_CSV = Path("data/reports/backtest/market_mainline_phase1_episode_constrained_2022_2025_states.csv")
DEFAULT_REPRESENTATIVE_TRADES_CSV = Path(
    "data/reports/backtest/market_mainline_phase1_episode_constrained_2022_2025_representative_trades.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/reports/mainline_radar")

BUCKET_ORDER = {"missing": 0, "low": 1, "medium": 2, "high": 3, "crowded": 4}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _bucket(value: Any) -> str:
    out = str(value or "").strip()
    return out if out in BUCKET_ORDER else "missing"


def _pct(value: Any) -> str:
    number = _finite_float(value, math.nan)
    return "NA" if not math.isfinite(number) else f"{number * 100:.0f}%"


def _num(value: Any, digits: int = 1) -> str:
    number = _finite_float(value, math.nan)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_state_date(state_rows: Iterable[dict[str, Any]]) -> str:
    dates = [_date_str(row.get("date")) for row in state_rows if str(row.get("date") or "").strip()]
    if not dates:
        raise ValueError("no state dates available")
    return max(dates)


def _group_rows_for_date(state_rows: Iterable[dict[str, Any]], date_value: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    target = _date_str(date_value)
    for row in state_rows:
        if _date_str(row.get("date")) != target:
            continue
        group = str(row.get("group") or row.get("mainline_group") or "").strip()
        direction = str(row.get("direction") or "").strip()
        if not group or not direction:
            continue
        out.setdefault((group, direction), []).append(row)
    return out


def _snapshot_context(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_date: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    buckets: dict[tuple[str, str], dict[str, str]] = {}
    for row in state_rows:
        date = _date_str(row.get("date"))
        group = str(row.get("group") or row.get("mainline_group") or "").strip()
        direction = str(row.get("direction") or "").strip()
        if not group or not direction:
            continue
        key = (group, direction)
        by_date.setdefault(date, {}).setdefault(key, []).append(row)
    for date, grouped in by_date.items():
        for key, rows in grouped.items():
            buckets.setdefault(key, {})[date] = max(
                (_bucket(row.get("bucket") or row.get("mainline_bucket")) for row in rows),
                key=lambda item: BUCKET_ORDER[item],
            )
    previous: dict[tuple[str, str, str], str] = {}
    for (group, direction), values in buckets.items():
        latest = "missing"
        for date in sorted(values):
            previous[(date, group, direction)] = latest
            latest = values[date]
    return {"by_date": by_date, "previous": previous}


def _previous_group_bucket(
    state_rows: Iterable[dict[str, Any]],
    *,
    group: str,
    direction: str,
    date_value: str,
) -> str:
    target = _date_str(date_value)
    latest_date = ""
    latest_bucket = "missing"
    for row in state_rows:
        if str(row.get("group") or row.get("mainline_group") or "").strip() != group:
            continue
        if str(row.get("direction") or "").strip() != direction:
            continue
        row_date = _date_str(row.get("date"))
        if row_date >= target or row_date < latest_date:
            continue
        latest_date = row_date
        latest_bucket = _bucket(row.get("bucket") or row.get("mainline_bucket"))
    return latest_bucket


def _representative_symbol(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        leader = str(row.get("group_leader_symbol") or "").strip().upper()
        if leader:
            return leader
    best = max(rows, key=lambda row: _finite_float(row.get("symbol_setup_score"), _finite_float(row.get("score"), 0.0)))
    return str(best.get("symbol") or "").strip().upper()


def _lifecycle_stage(current_bucket: str, previous_bucket: str, crowding_score: float) -> str:
    if current_bucket == "crowded" or crowding_score >= 70.0:
        return "crowded"
    if current_bucket == "high" and previous_bucket == "medium":
        return "confirming"
    if current_bucket == "high":
        return "established"
    if current_bucket == "medium":
        return "early_trial"
    return "no_mainline"


def _action_hint(stage: str, capital_score: float) -> str:
    capital_text = "资金确认充分，" if capital_score >= 70.0 else "资金确认一般，"
    if stage == "crowded":
        return f"{capital_text}但主线已经拥挤，优先降仓或避免新增。"
    if stage == "confirming":
        return f"{capital_text}主线正在确认，值得重点观察短线触发。"
    if stage == "established":
        return f"{capital_text}主线已成立，持仓优先，谨慎追新仓。"
    if stage == "early_trial":
        return f"{capital_text}主线初现，只适合观察或极小仓试错。"
    return "暂无明确主线，等待更强的板块确认。"


def _stage_rank(stage: str) -> int:
    return {
        "confirming": 4,
        "established": 3,
        "early_trial": 2,
        "crowded": 1,
        "no_mainline": 0,
    }.get(stage, 0)


def build_daily_snapshot(
    state_rows: list[dict[str, Any]],
    date_value: str | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    target_date = _date_str(date_value or latest_state_date(state_rows))
    snapshot: list[dict[str, Any]] = []
    grouped_rows = (
        (context.get("by_date") or {}).get(target_date, {})
        if context is not None
        else _group_rows_for_date(state_rows, target_date)
    )
    for (group, direction), rows in grouped_rows.items():
        score = max(_finite_float(row.get("mainline_score"), _finite_float(row.get("score"), 0.0)) for row in rows)
        capital = max(_finite_float(row.get("mainline_capital_score"), 0.0) for row in rows)
        resonance = max(_finite_float(row.get("mainline_resonance_score"), 0.0) for row in rows)
        crowding = max(_finite_float(row.get("mainline_crowding_score"), 0.0) for row in rows)
        leadership = max(_finite_float(row.get("mainline_leadership_score"), 0.0) for row in rows)
        breadth = max(_finite_float(row.get("group_breadth"), 0.0) for row in rows)
        retrace = sum(_finite_float(row.get("retrace_atr"), 0.0) for row in rows) / max(len(rows), 1)
        symbol_count = max(
            int(_finite_float(row.get("group_symbol_count"), 0.0)) for row in rows
        ) or len({str(row.get("symbol") or "").upper() for row in rows})
        current_bucket = max(
            (_bucket(row.get("bucket") or row.get("mainline_bucket")) for row in rows),
            key=lambda item: BUCKET_ORDER[item],
        )
        previous_bucket = (
            (context.get("previous") or {}).get((target_date, group, direction), "missing")
            if context is not None
            else _previous_group_bucket(state_rows, group=group, direction=direction, date_value=target_date)
        )
        stage = _lifecycle_stage(current_bucket, previous_bucket, crowding)
        is_isolated_move = bool(symbol_count <= 1 or breadth < 0.35)
        action_hint = _action_hint(stage, capital)
        if stage != "no_mainline" and is_isolated_move:
            action_hint = "只有单品种强势，暂按观察对象处理，等待更多板块共振。"
        radar_score = (
            score
            + 8.0 * breadth
            + 2.0 * min(symbol_count, 4)
            + 0.08 * capital
            + 0.06 * resonance
            - 0.08 * crowding
        )
        snapshot.append(
            {
                "date": target_date,
                "board": group,
                "direction": direction,
                "mainline_score": round(score, 4),
                "radar_score": round(radar_score, 4),
                "lifecycle_stage": stage,
                "current_bucket": current_bucket,
                "previous_bucket": previous_bucket,
                "leadership_score": round(leadership, 4),
                "leadership_breadth": round(breadth, 4),
                "capital_confirmation": round(capital, 4),
                "board_resonance": round(resonance, 4),
                "crowding_warning": round(crowding, 4),
                "retrace_atr": round(retrace, 4),
                "symbol_count": int(symbol_count),
                "is_isolated_move": is_isolated_move,
                "representative_symbol": _representative_symbol(rows),
                "action_hint": action_hint,
            }
        )
    return sorted(
        snapshot,
        key=lambda row: (
            _stage_rank(str(row["lifecycle_stage"])),
            _finite_float(row["radar_score"]),
            _finite_float(row["mainline_score"]),
        ),
        reverse=True,
    )


def _window_dates(state_rows: list[dict[str, Any]], date_value: str, lookback_days: int) -> list[str]:
    end_date = _date_str(date_value)
    dates = sorted({_date_str(row.get("date")) for row in state_rows if str(row.get("date") or "").strip()})
    eligible = [date for date in dates if date <= end_date]
    return eligible[-int(lookback_days) :]


def _weekly_hint(status: str) -> str:
    if status == "late_cycle_risk":
        return "主线分高但拥挤明显上升，按后期风险处理，优先降仓或避免新增。"
    if status == "improving_consensus":
        return "分歧转共识迹象增强，适合重点等待短线触发。"
    if status == "persistent_mainline":
        return "主线连续存在，持仓观察优先，新增仓位需要更好的触发点。"
    if status == "confirming":
        return "本周进入确认阶段，值得纳入重点观察。"
    if status == "one_day_spike":
        return "更像单日尖峰，暂不按稳定主线处理。"
    if status == "weakening":
        return "主线强度或覆盖走弱，降低优先级。"
    return "主线证据不足，继续观察。"


def build_weekly_radar(
    state_rows: list[dict[str, Any]],
    date_value: str | None = None,
    *,
    lookback_days: int = 5,
) -> dict[str, Any]:
    end_date = _date_str(date_value or latest_state_date(state_rows))
    dates = _window_dates(state_rows, end_date, lookback_days)
    context = _snapshot_context(state_rows)
    daily_rows: list[dict[str, Any]] = []
    for date in dates:
        daily_rows.extend(build_daily_snapshot(state_rows, date, context=context))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in daily_rows:
        grouped.setdefault((str(row.get("board") or ""), str(row.get("direction") or "")), []).append(row)

    weekly_rows: list[dict[str, Any]] = []
    for (board, direction), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: str(row.get("date") or ""))
        first = ordered[0]
        latest = ordered[-1]
        active = [
            row
            for row in ordered
            if row.get("lifecycle_stage") != "no_mainline" and not bool(row.get("is_isolated_move"))
        ]
        active_days = len(active)
        avg_score = sum(_finite_float(row.get("mainline_score"), 0.0) for row in ordered) / max(len(ordered), 1)
        avg_crowding = sum(_finite_float(row.get("crowding_warning"), 0.0) for row in ordered) / max(len(ordered), 1)
        score_change = _finite_float(latest.get("mainline_score"), 0.0) - _finite_float(first.get("mainline_score"), 0.0)
        breadth_change = _finite_float(latest.get("leadership_breadth"), 0.0) - _finite_float(first.get("leadership_breadth"), 0.0)
        retrace_change = _finite_float(first.get("retrace_atr"), 0.0) - _finite_float(latest.get("retrace_atr"), 0.0)
        crowding_change = _finite_float(latest.get("crowding_warning"), 0.0) - _finite_float(first.get("crowding_warning"), 0.0)
        latest_stage = str(latest.get("lifecycle_stage") or "")
        is_late_risk = latest_stage == "crowded" or (
            _finite_float(latest.get("mainline_score"), 0.0) >= 75.0 and crowding_change >= 45.0
        )
        is_improving = active_days >= 2 and (
            score_change >= 8.0 or breadth_change >= 0.25 or retrace_change >= 1.0
        )
        if is_late_risk:
            status = "late_cycle_risk"
        elif is_improving:
            status = "improving_consensus"
        elif score_change <= -8.0 or breadth_change <= -0.25:
            status = "weakening"
        elif active_days <= 1 and latest_stage != "no_mainline":
            status = "one_day_spike"
        elif latest_stage == "confirming":
            status = "confirming"
        elif active_days >= max(3, min(lookback_days, len(dates)) - 1):
            status = "persistent_mainline"
        else:
            status = "watching"
        weekly_score = (
            25.0 * active_days
            + avg_score
            + 10.0 * _finite_float(latest.get("leadership_breadth"), 0.0)
            + 0.05 * _finite_float(latest.get("capital_confirmation"), 0.0)
            - 0.08 * avg_crowding
        )
        weekly_rows.append(
            {
                "date": end_date,
                "board": board,
                "direction": direction,
                "window_start": dates[0] if dates else "",
                "window_end": dates[-1] if dates else "",
                "active_days": int(active_days),
                "weekly_score": round(weekly_score, 4),
                "avg_mainline_score": round(avg_score, 4),
                "latest_mainline_score": latest.get("mainline_score"),
                "latest_stage": latest_stage,
                "weekly_status": status,
                "score_change": round(score_change, 4),
                "breadth_change": round(breadth_change, 4),
                "retrace_improvement": round(retrace_change, 4),
                "crowding_change": round(crowding_change, 4),
                "representative_symbol": latest.get("representative_symbol"),
                "weekly_hint": _weekly_hint(status),
            }
        )
    weekly_rows = sorted(
        weekly_rows,
        key=lambda row: (
            str(row.get("weekly_status")) != "one_day_spike",
            _finite_float(row.get("weekly_score"), 0.0),
        ),
        reverse=True,
    )
    top_mainlines = [
        row
        for row in weekly_rows
        if int(row.get("active_days") or 0) >= 2
        and row.get("weekly_status") in {"confirming", "improving_consensus", "persistent_mainline"}
    ][:3]
    return {
        "date": end_date,
        "window_dates": dates,
        "rows": weekly_rows,
        "top_mainlines": top_mainlines,
        "improving_boards": [row for row in weekly_rows if row.get("weekly_status") == "improving_consensus"],
        "weakening_boards": [row for row in weekly_rows if row.get("weekly_status") == "weakening"],
        "crowded_boards": [row for row in weekly_rows if row.get("weekly_status") == "late_cycle_risk"],
    }


def _year_key(value: Any) -> str:
    return str(int(pd.Timestamp(value).year))


def _blank_year() -> dict[str, Any]:
    return {
        "active_mainline_days": 0,
        "active_mainline_weeks": 0,
        "top_board_days": 0,
        "stage_days": {
            "early_trial": 0,
            "confirming": 0,
            "established": 0,
            "crowded": 0,
        },
        "representative_opportunities": 0,
        "profitable_representative_opportunities": 0,
    }


def _trade_stage(row: dict[str, Any]) -> str:
    return str(row.get("transition_stage") or row.get("mainline_stage") or "unknown").strip() or "unknown"


def build_replay_report(
    state_rows: list[dict[str, Any]],
    representative_trades: list[dict[str, Any]],
    *,
    years: list[int],
) -> dict[str, Any]:
    year_set = {int(year) for year in years}
    yearly: dict[str, dict[str, Any]] = {str(year): _blank_year() for year in sorted(year_set)}
    active_weeks: dict[str, set[str]] = {str(year): set() for year in year_set}
    dates = sorted({_date_str(row.get("date")) for row in state_rows if str(row.get("date") or "").strip()})
    context = _snapshot_context(state_rows)
    for date in dates:
        year = int(pd.Timestamp(date).year)
        if year not in year_set:
            continue
        year_text = str(year)
        snapshot = build_daily_snapshot(state_rows, date, context=context)
        active = [
            row
            for row in snapshot
            if row.get("lifecycle_stage") != "no_mainline" and not bool(row.get("is_isolated_move"))
        ]
        top = [
            row
            for row in active
            if row.get("lifecycle_stage") in {"early_trial", "confirming", "established"}
        ][:3]
        yearly[year_text]["active_mainline_days"] += len(active)
        yearly[year_text]["top_board_days"] += len(top)
        if active:
            stamp = pd.Timestamp(date)
            iso = stamp.isocalendar()
            active_weeks[year_text].add(f"{int(iso.year)}-{int(iso.week):02d}")
        for row in active:
            stage = str(row.get("lifecycle_stage") or "")
            if stage in yearly[year_text]["stage_days"]:
                yearly[year_text]["stage_days"][stage] += 1

    stage_buckets: dict[str, list[float]] = {}
    for trade in representative_trades:
        entry_time = str(trade.get("entry_time") or "")
        if not entry_time:
            continue
        year = int(pd.Timestamp(entry_time).year)
        if year not in year_set:
            continue
        year_text = str(year)
        pnl = _finite_float(trade.get("pnl_ratio"), 0.0)
        yearly[year_text]["representative_opportunities"] += 1
        if pnl > 0:
            yearly[year_text]["profitable_representative_opportunities"] += 1
        stage_buckets.setdefault(_trade_stage(trade), []).append(pnl)

    for year_text, weeks in active_weeks.items():
        yearly[year_text]["active_mainline_weeks"] = len(weeks)

    stage_outcomes: dict[str, dict[str, Any]] = {}
    for stage, values in sorted(stage_buckets.items()):
        wins = sum(1 for value in values if value > 0)
        stage_outcomes[stage] = {
            "trades": len(values),
            "avg_pnl_ratio": sum(values) / max(len(values), 1),
            "win_rate": wins / max(len(values), 1),
        }

    confirm_avg = _finite_float(stage_outcomes.get("confirm_medium_to_high", {}).get("avg_pnl_ratio"), 0.0)
    late_avg = _finite_float(stage_outcomes.get("late_high", {}).get("avg_pnl_ratio"), 0.0)
    return {
        "years": sorted(year_set),
        "yearly": yearly,
        "stage_outcomes": stage_outcomes,
        "crowded_vs_confirmation": {
            "confirm_avg_pnl_ratio": confirm_avg,
            "late_high_avg_pnl_ratio": late_avg,
            "late_underperformed_confirm": late_avg < confirm_avg,
        },
    }


def _symbol_close_index(state_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    raw: dict[str, dict[str, float]] = {}
    for row in state_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        close = _finite_float(row.get("close"), math.nan)
        if not math.isfinite(close) or close <= 0:
            continue
        raw.setdefault(symbol, {})[_date_str(row.get("date"))] = float(close)
    out: dict[str, dict[str, Any]] = {}
    for symbol, values in raw.items():
        dates = sorted(values)
        out[symbol] = {
            "values": values,
            "dates": dates,
            "positions": {date: index for index, date in enumerate(dates)},
        }
    return out


def _group_symbol_index(state_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in state_rows:
        date = _date_str(row.get("date"))
        group = str(row.get("group") or row.get("mainline_group") or "").strip()
        direction = str(row.get("direction") or "").strip()
        symbol = str(row.get("symbol") or "").upper()
        if not group or not direction or not symbol:
            continue
        grouped.setdefault((date, group, direction), set()).add(symbol)
    return {key: sorted(values) for key, values in grouped.items()}


def _future_symbol_path(
    close_index: dict[str, dict[str, Any]],
    *,
    symbol: str,
    date_value: str,
    horizon: int,
) -> list[float]:
    item = close_index.get(str(symbol).upper(), {})
    values = item.get("values") or {}
    dates = item.get("dates") or []
    positions = item.get("positions") or {}
    current_date = _date_str(date_value)
    if current_date not in values:
        return []
    start = positions.get(current_date)
    if start is None:
        return []
    end = start + int(horizon)
    if end >= len(dates):
        return []
    return [values[date] for date in dates[start : end + 1]]


def _future_group_outcome(
    state_rows: list[dict[str, Any]],
    close_index: dict[str, dict[str, Any]],
    group_symbol_index: dict[tuple[str, str, str], list[str]],
    *,
    date_value: str,
    group: str,
    direction: str,
    horizons: list[int],
) -> dict[str, float]:
    symbols = group_symbol_index.get((_date_str(date_value), str(group), str(direction)), [])
    sign = -1.0 if str(direction) == "short" else 1.0
    out: dict[str, float] = {}
    for horizon in horizons:
        future_returns: list[float] = []
        future_maes: list[float] = []
        for symbol in symbols:
            path = _future_symbol_path(close_index, symbol=symbol, date_value=date_value, horizon=int(horizon))
            if len(path) < int(horizon) + 1 or path[0] <= 0:
                continue
            directional_path = [sign * (price / path[0] - 1.0) for price in path[1:]]
            if not directional_path:
                continue
            future_returns.append(float(directional_path[-1]))
            future_maes.append(float(min(directional_path)))
        if future_returns:
            out[f"future_return_{int(horizon)}d"] = round(sum(future_returns) / len(future_returns), 10)
            out[f"future_mae_{int(horizon)}d"] = round(sum(future_maes) / len(future_maes), 10)
    return out


def _group_momentum_rows(state_rows: list[dict[str, Any]], date_value: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = _group_rows_for_date(state_rows, date_value)
    out: list[dict[str, Any]] = []
    for (group, direction), rows in grouped.items():
        symbol_count = max(
            int(_finite_float(row.get("group_symbol_count"), 0.0)) for row in rows
        ) or len({str(row.get("symbol") or "").upper() for row in rows})
        breadth = max(_finite_float(row.get("group_breadth"), 0.0) for row in rows)
        out.append(
            {
                "date": _date_str(date_value),
                "board": group,
                "direction": direction,
                "momentum_score": sum(_finite_float(row.get("directional_ret_20"), 0.0) for row in rows) / max(len(rows), 1),
                "is_isolated_move": bool(symbol_count <= 1 or breadth < 0.35),
                "lifecycle_stage": "momentum_baseline",
                "representative_symbol": _representative_symbol(rows),
            }
        )
    return sorted(out, key=lambda row: _finite_float(row.get("momentum_score"), 0.0), reverse=True)


def _summarize_labeled(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"signals": len(rows)}
    for horizon in horizons:
        field = f"future_return_{int(horizon)}d"
        mae_field = f"future_mae_{int(horizon)}d"
        returns = [_finite_float(row.get(field), math.nan) for row in rows]
        returns = [value for value in returns if math.isfinite(value)]
        maes = [_finite_float(row.get(mae_field), math.nan) for row in rows]
        maes = [value for value in maes if math.isfinite(value)]
        summary[f"avg_future_return_{int(horizon)}d"] = sum(returns) / len(returns) if returns else 0.0
        summary[f"positive_rate_{int(horizon)}d"] = sum(1 for value in returns if value > 0) / len(returns) if returns else 0.0
        summary[f"avg_future_mae_{int(horizon)}d"] = sum(maes) / len(maes) if maes else 0.0
    return summary


def _add_future_labels(
    rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    close_index: dict[str, dict[str, Any]],
    group_symbol_index: dict[tuple[str, str, str], list[str]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(
            _future_group_outcome(
                state_rows,
                close_index,
                group_symbol_index,
                date_value=str(item.get("date") or ""),
                group=str(item.get("board") or ""),
                direction=str(item.get("direction") or ""),
                horizons=horizons,
            )
        )
        if any(f"future_return_{int(horizon)}d" in item for horizon in horizons):
            labeled.append(item)
    return labeled


def build_effectiveness_validation(
    state_rows: list[dict[str, Any]],
    *,
    years: list[int],
    horizons: list[int] | None = None,
    top_n: int = 3,
) -> dict[str, Any]:
    resolved_horizons = [int(value) for value in (horizons or [5, 20, 60])]
    year_set = {int(year) for year in years}
    dates = sorted(
        {
            _date_str(row.get("date"))
            for row in state_rows
            if str(row.get("date") or "").strip() and int(pd.Timestamp(row.get("date")).year) in year_set
        }
    )
    close_index = _symbol_close_index(state_rows)
    group_symbols = _group_symbol_index(state_rows)
    context = _snapshot_context(state_rows)
    cohort_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for date in dates:
        snapshot = build_daily_snapshot(state_rows, date, context=context)
        tradable = [
            row
            for row in snapshot
            if row.get("lifecycle_stage") in {"early_trial", "confirming", "established"}
            and not bool(row.get("is_isolated_move"))
        ]
        for row in tradable[: int(top_n)]:
            cohort_rows.append({**row, "cohort": "radar_top"})
        for row in [item for item in snapshot if bool(item.get("is_isolated_move")) and item.get("lifecycle_stage") != "no_mainline"][: int(top_n)]:
            cohort_rows.append({**row, "cohort": "isolated_move"})
        for row in _group_momentum_rows(state_rows, date)[: int(top_n)]:
            cohort_rows.append({**row, "cohort": "momentum_top"})
        for row in snapshot:
            if row.get("lifecycle_stage") != "no_mainline":
                stage_rows.append({**row, "stage": str(row.get("lifecycle_stage") or "")})

    labeled_rows = _add_future_labels(cohort_rows, state_rows, close_index, group_symbols, resolved_horizons)
    stage_labeled_rows = _add_future_labels(stage_rows, state_rows, close_index, group_symbols, resolved_horizons)

    cohort_summary: dict[str, Any] = {}
    for cohort in sorted({str(row.get("cohort") or "") for row in labeled_rows}):
        cohort_summary[cohort] = _summarize_labeled(
            [row for row in labeled_rows if row.get("cohort") == cohort],
            resolved_horizons,
        )

    stage_summary: dict[str, Any] = {}
    for stage in sorted({str(row.get("stage") or "") for row in stage_labeled_rows}):
        stage_summary[stage] = _summarize_labeled(
            [row for row in stage_labeled_rows if row.get("stage") == stage],
            resolved_horizons,
        )

    yearly_summary: dict[str, dict[str, Any]] = {str(year): {} for year in sorted(year_set)}
    for year in sorted(year_set):
        year_rows = [row for row in labeled_rows if int(pd.Timestamp(row.get("date")).year) == int(year)]
        for cohort in sorted({str(row.get("cohort") or "") for row in year_rows}):
            yearly_summary[str(year)][cohort] = _summarize_labeled(
                [row for row in year_rows if row.get("cohort") == cohort],
                resolved_horizons,
            )

    return {
        "years": sorted(year_set),
        "horizons": resolved_horizons,
        "top_n": int(top_n),
        "labeled_rows": labeled_rows,
        "stage_labeled_rows": stage_labeled_rows,
        "cohort_summary": cohort_summary,
        "stage_summary": stage_summary,
        "yearly_summary": yearly_summary,
    }


def _stage_position(
    *,
    name: str,
    equity: float,
    margin_per_lot: float,
    risk_per_lot: float,
    lifecycle_budget_pct: float,
    max_margin_pct: float,
    risk_pct: float,
) -> dict[str, Any]:
    result = calculate_position_sizing(
        margin_per_lot=float(margin_per_lot),
        risk_per_lot=float(risk_per_lot),
        score_margin_budget=float(equity) * min(float(lifecycle_budget_pct), float(max_margin_pct)),
        portfolio_margin_budget=float(equity) * float(max_margin_pct),
        risk_budget=float(equity) * float(risk_pct),
    )
    limited_by = result.sizing_limited_by.replace("phase2_score_budget", "lifecycle_budget")
    return {
        "stage": name,
        "lifecycle_budget_pct": float(min(float(lifecycle_budget_pct), float(max_margin_pct))),
        "suggested_lots": int(result.suggested_lots),
        "suggested_margin": float(result.suggested_margin),
        "suggested_stop_risk": float(result.suggested_stop_risk),
        "margin_usage_pct": float(result.suggested_margin / equity) if equity > 0 else 0.0,
        "stop_risk_pct": float(result.suggested_stop_risk / equity) if equity > 0 else 0.0,
        "score_lots": int(result.score_lots),
        "portfolio_lots": int(result.portfolio_lots),
        "risk_lots": int(result.risk_lots),
        "sizing_limited_by": limited_by,
        "sizing_zero_lot_reason": result.sizing_zero_lot_reason,
    }


def build_risk_worksheet(
    *,
    symbol: str,
    direction: str,
    equity: float,
    max_margin_pct: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    multiplier: float,
    margin_rate: float,
) -> dict[str, Any]:
    entry = float(entry_price)
    stop = float(stop_price)
    resolved_multiplier = float(multiplier)
    margin_per_lot = entry * resolved_multiplier * float(margin_rate) if entry > 0 and resolved_multiplier > 0 else 0.0
    risk_per_lot = abs(entry - stop) * resolved_multiplier if entry > 0 and stop > 0 and resolved_multiplier > 0 else 0.0
    warnings: list[str] = []
    if margin_per_lot <= 0:
        warnings.append("缺少有效每手保证金，无法给出可靠手数。")
    if risk_per_lot <= 0:
        warnings.append("缺少有效止损距离，止损风险约束不可用。")
    stages = {
        "trial": 0.05,
        "confirming": min(0.30, float(max_margin_pct)),
        "high_conviction": float(max_margin_pct),
        "crowded_reduce": min(0.10, float(max_margin_pct)),
    }
    worksheet: dict[str, Any] = {
        "symbol": str(symbol).upper(),
        "direction": str(direction),
        "equity": float(equity),
        "entry_price": entry,
        "stop_price": stop,
        "multiplier": resolved_multiplier,
        "margin_rate": float(margin_rate),
        "margin_per_lot": float(margin_per_lot),
        "risk_per_lot": float(risk_per_lot),
        "max_margin_pct": float(max_margin_pct),
        "risk_pct": float(risk_pct),
        "warnings": warnings,
    }
    for stage, budget_pct in stages.items():
        worksheet[stage] = _stage_position(
            name=stage,
            equity=float(equity),
            margin_per_lot=margin_per_lot,
            risk_per_lot=risk_per_lot,
            lifecycle_budget_pct=budget_pct,
            max_margin_pct=float(max_margin_pct),
            risk_pct=float(risk_pct),
        )
    return worksheet


def render_snapshot_markdown(snapshot_rows: list[dict[str, Any]], date_value: str) -> str:
    lines = [
        f"# 主线雷达日度快照 {_date_str(date_value)}",
        "",
        "## 业务结论",
        "",
    ]
    active = [
        row
        for row in snapshot_rows
        if row.get("lifecycle_stage") != "no_mainline" and not bool(row.get("is_isolated_move"))
    ]
    if not active:
        lines.append("- 当前没有识别到明确板块主线，适合等待。")
    else:
        for row in active[:3]:
            lines.append(
                f"- {row['board']} {row['direction']}：{row['action_hint']} "
                f"代表观察品种 {row['representative_symbol']}。"
            )
    isolated = [row for row in snapshot_rows if bool(row.get("is_isolated_move"))]
    if isolated:
        lines.append(f"- {len(isolated)} 个方向更像孤立异动，暂不应按板块主线处理。")
    crowded = [row for row in snapshot_rows if row.get("lifecycle_stage") == "crowded"]
    if crowded:
        names = "、".join(str(row["board"]) for row in crowded[:3])
        lines.append(f"- 拥挤提示：{names} 已不适合作为新增仓位的优先方向。")
    lines.extend(
        [
            "",
            "## 板块明细",
            "",
            "| 板块 | 方向 | 阶段 | 主线分 | 资金确认 | 板块共振 | 覆盖 | 拥挤 | 代表品种 | 动作提示 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in snapshot_rows:
        lines.append(
            f"| {row['board']} | {row['direction']} | {row['lifecycle_stage']} | "
            f"{_num(row['mainline_score'])} | {_num(row['capital_confirmation'])} | "
            f"{_num(row['board_resonance'])} | {_pct(row['leadership_breadth'])} | "
            f"{_num(row['crowding_warning'])} | {row['representative_symbol']} | {row['action_hint']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 这是交易辅助报告，不是自动交易指令。",
            "- 期限结构、基差和外部基本面/舆论数据未纳入本版雷达。",
        ]
    )
    return "\n".join(lines)


def render_effectiveness_validation_markdown(validation: dict[str, Any]) -> str:
    years = [int(year) for year in validation.get("years", [])]
    title_years = f"{min(years)}-{max(years)}" if years else "unknown"
    horizons = [int(value) for value in validation.get("horizons", [])]
    main_horizon = 20 if 20 in horizons else horizons[0] if horizons else 0
    cohort_summary = validation.get("cohort_summary") or {}
    stage_summary = validation.get("stage_summary") or {}
    radar = cohort_summary.get("radar_top") or {}
    momentum = cohort_summary.get("momentum_top") or {}
    isolated = cohort_summary.get("isolated_move") or {}
    radar_ret = _finite_float(radar.get(f"avg_future_return_{main_horizon}d"), 0.0)
    momentum_ret = _finite_float(momentum.get(f"avg_future_return_{main_horizon}d"), 0.0)
    isolated_ret = _finite_float(isolated.get(f"avg_future_return_{main_horizon}d"), 0.0)
    confirm_ret = _finite_float(stage_summary.get("confirming", {}).get(f"avg_future_return_{main_horizon}d"), 0.0)
    crowded_ret = _finite_float(stage_summary.get("crowded", {}).get(f"avg_future_return_{main_horizon}d"), 0.0)
    lines = [
        f"# 主线雷达有效性验证 {title_years}",
        "",
        "## 业务结论",
        "",
        "- 本报告验证雷达判断的板块主线后续是否更像趋势，而不是验证某套自动交易规则。",
        f"- 主要观察窗口使用未来 {main_horizon} 个交易日；所有标签只使用信号日之后的数据。",
    ]
    if radar:
        if radar_ret > momentum_ret:
            lines.append("- 雷达 Top 方向在主要观察窗口内优于简单动量 Top，说明板块共振/资金确认有增量信息。")
        else:
            lines.append("- 雷达 Top 方向没有优于简单动量 Top，说明当前雷达复杂度还没有被充分证明。")
        if radar_ret > isolated_ret:
            lines.append("- 雷达 Top 方向优于孤立异动，说明区分板块共振和单品种尖峰是有价值的。")
        else:
            lines.append("- 雷达 Top 方向没有优于孤立异动，后续需要重新收紧板块共振口径。")
    if stage_summary:
        if confirm_ret > crowded_ret:
            lines.append("- 确认阶段后续表现优于拥挤阶段，支持“确认看机会、拥挤降优先级”。")
        else:
            lines.append("- 确认阶段没有优于拥挤阶段，生命周期判断需要复核。")
    lines.extend(
        [
            "",
            "## 候选集合对比",
            "",
        ]
    )
    header = "| 集合 | 信号数 | " + " | ".join(
        f"{h}日均收益 | {h}日胜率 | {h}日平均逆行" for h in horizons
    ) + " |"
    sep = "|---|---:" + "|---:|---:|---:" * len(horizons) + "|"
    lines.extend([header, sep])
    for cohort in ("radar_top", "momentum_top", "isolated_move"):
        row = cohort_summary.get(cohort)
        if not row:
            continue
        values = [cohort, str(row.get("signals", 0))]
        for horizon in horizons:
            values.extend(
                [
                    _pct(row.get(f"avg_future_return_{horizon}d")),
                    _pct(row.get(f"positive_rate_{horizon}d")),
                    _pct(row.get(f"avg_future_mae_{horizon}d")),
                ]
            )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 生命周期阶段对比",
            "",
        ]
    )
    lines.extend([header.replace("集合", "阶段"), sep])
    for stage in ("early_trial", "confirming", "established", "crowded"):
        row = stage_summary.get(stage)
        if not row:
            continue
        values = [stage, str(row.get("signals", 0))]
        for horizon in horizons:
            values.extend(
                [
                    _pct(row.get(f"avg_future_return_{horizon}d")),
                    _pct(row.get(f"positive_rate_{horizon}d")),
                    _pct(row.get(f"avg_future_mae_{horizon}d")),
                ]
            )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 分年份雷达 Top",
            "",
            "| 年份 | 信号数 | " + " | ".join(f"{h}日均收益 | {h}日胜率" for h in horizons) + " |",
            "|---:|---:" + "|---:|---:" * len(horizons) + "|",
        ]
    )
    for year in sorted((validation.get("yearly_summary") or {})):
        row = (validation["yearly_summary"].get(year) or {}).get("radar_top")
        if not row:
            continue
        values = [str(year), str(row.get("signals", 0))]
        for horizon in horizons:
            values.extend([_pct(row.get(f"avg_future_return_{horizon}d")), _pct(row.get(f"positive_rate_{horizon}d"))])
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 验证对象是盘面主线判断，不是完整交易策略。",
            "- 当前验证只使用连续主力缓存里的价格路径，仍未纳入期限结构、基差、基本面和舆论。",
            "- 如果雷达 Top 不能持续优于简单动量基准，就不能说雷达已经有效。",
        ]
    )
    return "\n".join(lines)


def render_replay_markdown(replay: dict[str, Any]) -> str:
    years = [int(year) for year in replay.get("years", [])]
    if years:
        title_years = f"{min(years)}-{max(years)}"
    else:
        title_years = "unknown"
    lines = [
        f"# 主线雷达历史回放 {title_years}",
        "",
        "## 业务结论",
        "",
        "- 本报告用于校准主线雷达是否能稳定识别趋势窗口，不是收益承诺。",
        "- 回放按年份拆分主线活跃度、阶段结构和代表机会，避免只看四年汇总。",
    ]
    crowded_cmp = replay.get("crowded_vs_confirmation") or {}
    if crowded_cmp:
        if crowded_cmp.get("late_underperformed_confirm"):
            lines.append("- 历史代表机会中，后期追高阶段弱于中到高确认阶段，符合“后期降仓”的交易习惯。")
        else:
            lines.append("- 当前样本里后期阶段没有明显弱于确认阶段，后续需要谨慎复核。")
    lines.extend(
        [
            "",
            "## 年份拆分",
            "",
            "| 年份 | 主线日 | 主线周 | Top方向日 | 初现 | 确认 | 已成立 | 拥挤 | 代表机会 | 盈利代表机会 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for year in sorted(replay.get("yearly", {})):
        row = replay["yearly"][year]
        stages = row.get("stage_days") or {}
        lines.append(
            f"| {year} | {row.get('active_mainline_days', 0)} | {row.get('active_mainline_weeks', 0)} | "
            f"{row.get('top_board_days', 0)} | {stages.get('early_trial', 0)} | {stages.get('confirming', 0)} | "
            f"{stages.get('established', 0)} | {stages.get('crowded', 0)} | "
            f"{row.get('representative_opportunities', 0)} | {row.get('profitable_representative_opportunities', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 代表机会阶段表现",
            "",
            "| 阶段 | 笔数 | 平均收益 | 胜率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for stage, row in replay.get("stage_outcomes", {}).items():
        lines.append(f"| {stage} | {row['trades']} | {_pct(row['avg_pnl_ratio'])} | {_pct(row['win_rate'])} |")
    lines.extend(
        [
            "",
            "## 与约束基线的关系",
            "",
            "- 最新约束趋势活动基线显示：30% 保证金上限年化约 31.44%，最大回撤约 18.65%；50% 保证金上限年化约 47.51%，最大回撤约 26.42%。",
            "- 主线雷达不直接复刻该收益曲线，而是把其中“少做、只做主线、确认后更重视、后期降优先级”的识别部分拆出来供交易员使用。",
            "",
            "## 边界",
            "",
            "- 回放依赖本地历史缓存和代表机会文件。",
            "- 期限结构、基差、基本面和舆论数据仍未纳入。",
        ]
    )
    return "\n".join(lines)


def render_worksheet_markdown(worksheet: dict[str, Any], date_value: str) -> str:
    lines = [
        f"# 主线雷达风险 Worksheet {worksheet['symbol']} {_date_str(date_value)}",
        "",
        "## 业务结论",
        "",
        f"- 合约方向：{worksheet['symbol']} {worksheet['direction']}。",
        f"- 入场价按 {worksheet['entry_price']:.4f} 估算，止损价按 {worksheet['stop_price']:.4f} 估算。",
        f"- 每手保证金约 {worksheet['margin_per_lot']:.2f}，每手止损风险约 {worksheet['risk_per_lot']:.2f}。",
    ]
    if worksheet.get("warnings"):
        for warning in worksheet["warnings"]:
            lines.append(f"- 风险提示：{warning}")
    lines.extend(
        [
            "",
            "## 阶段手数",
            "",
            "| 阶段 | 建议手数 | 阶段预算 | 保证金占用 | 止损风险 | 限制来源 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    labels = {
        "trial": "小仓试错",
        "confirming": "主线确认",
        "high_conviction": "高信心持仓",
        "crowded_reduce": "拥挤降仓",
    }
    for key in ("trial", "confirming", "high_conviction", "crowded_reduce"):
        row = worksheet[key]
        lines.append(
            f"| {labels[key]} | {row['suggested_lots']} | {_pct(row['lifecycle_budget_pct'])} | "
            f"{_pct(row['margin_usage_pct'])} | {_pct(row['stop_risk_pct'])} | {row['sizing_limited_by']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- Worksheet 只做风险和手数约束，不判断是否应该交易。",
            "- 最终手数永远取阶段预算、总保证金约束、单笔止损风险约束的最小值。",
        ]
    )
    return "\n".join(lines)


def render_weekly_markdown(weekly: dict[str, Any]) -> str:
    date_value = _date_str(weekly.get("date"))
    lines = [
        f"# 主线雷达周报 {date_value}",
        "",
        "## 业务结论",
        "",
    ]
    top = list(weekly.get("top_mainlines") or [])
    if not top:
        lines.append("- 最近 5 个交易日没有稳定的板块主线，适合等待。")
    else:
        for row in top:
            lines.append(
                f"- {row['board']} {row['direction']}：{row['weekly_hint']} "
                f"代表观察品种 {row['representative_symbol']}，活跃 {row['active_days']} 天。"
            )
    crowded = list(weekly.get("crowded_boards") or [])
    if crowded:
        names = "、".join(str(row["board"]) for row in crowded[:3])
        lines.append(f"- 后期风险：{names} 拥挤上升，新增仓位优先级下降。")
    improving = list(weekly.get("improving_boards") or [])
    if improving:
        names = "、".join(str(row["board"]) for row in improving[:3])
        lines.append(f"- 分歧转共识：{names} 的持续性或回撤质量在改善。")
    weakening = list(weekly.get("weakening_boards") or [])
    if weakening:
        names = "、".join(str(row["board"]) for row in weakening[:3])
        lines.append(f"- 走弱提示：{names} 已从主线候选降级，不应作为加仓优先方向。")
    lines.extend(
        [
            "",
            "## 周度明细",
            "",
            "| 板块 | 方向 | 周状态 | 活跃天数 | 周分 | 均分 | 分数变化 | 覆盖变化 | 回撤改善 | 拥挤变化 | 代表品种 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in weekly.get("rows") or []:
        lines.append(
            f"| {row['board']} | {row['direction']} | {row['weekly_status']} | {row['active_days']} | "
            f"{_num(row['weekly_score'])} | {_num(row['avg_mainline_score'])} | {_num(row['score_change'])} | "
            f"{_pct(row['breadth_change'])} | {_num(row['retrace_improvement'])} | {_num(row['crowding_change'])} | "
            f"{row['representative_symbol']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 这是交易辅助周报，不是自动交易指令。",
            "- 周报只说明盘面主线质量，不替代基本面和舆论判断。",
        ]
    )
    return "\n".join(lines)


def write_snapshot_outputs(snapshot_rows: list[dict[str, Any]], date_value: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    date_text = _date_str(date_value)
    output_dir = Path(output_dir)
    md_path = output_dir / f"mainline_snapshot_{date_text}.md"
    csv_path = output_dir / f"mainline_snapshot_{date_text}.csv"
    json_path = output_dir / f"mainline_snapshot_{date_text}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_snapshot_markdown(snapshot_rows, date_text), encoding="utf-8")
    _write_csv(csv_path, snapshot_rows)
    _write_json(json_path, {"date": date_text, "rows": snapshot_rows})
    return {"markdown": str(md_path), "csv": str(csv_path), "json": str(json_path)}


def write_effectiveness_validation_outputs(validation: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    years = [int(year) for year in validation.get("years", [])]
    suffix = f"{min(years)}_{max(years)}" if years else "unknown"
    output_dir = Path(output_dir)
    md_path = output_dir / f"mainline_effectiveness_{suffix}_report.md"
    json_path = output_dir / f"mainline_effectiveness_{suffix}.json"
    labeled_csv = output_dir / f"mainline_effectiveness_{suffix}_labeled.csv"
    cohort_csv = output_dir / f"mainline_effectiveness_{suffix}_cohort_summary.csv"
    stage_csv = output_dir / f"mainline_effectiveness_{suffix}_stage_summary.csv"
    yearly_csv = output_dir / f"mainline_effectiveness_{suffix}_yearly_summary.csv"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_effectiveness_validation_markdown(validation), encoding="utf-8")
    _write_json(json_path, validation)
    _write_csv(labeled_csv, list(validation.get("labeled_rows") or []))
    _write_csv(cohort_csv, [{"cohort": cohort, **row} for cohort, row in (validation.get("cohort_summary") or {}).items()])
    _write_csv(stage_csv, [{"stage": stage, **row} for stage, row in (validation.get("stage_summary") or {}).items()])
    yearly_rows: list[dict[str, Any]] = []
    for year, cohorts in (validation.get("yearly_summary") or {}).items():
        for cohort, row in cohorts.items():
            yearly_rows.append({"year": year, "cohort": cohort, **row})
    _write_csv(yearly_csv, yearly_rows)
    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "labeled_csv": str(labeled_csv),
        "cohort_csv": str(cohort_csv),
        "stage_csv": str(stage_csv),
        "yearly_csv": str(yearly_csv),
    }


def write_replay_outputs(replay: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    years = [int(year) for year in replay.get("years", [])]
    suffix = f"{min(years)}_{max(years)}" if years else "unknown"
    output_dir = Path(output_dir)
    md_path = output_dir / f"mainline_replay_{suffix}_report.md"
    json_path = output_dir / f"mainline_replay_{suffix}.json"
    yearly_csv = output_dir / f"mainline_replay_{suffix}_yearly.csv"
    stage_csv = output_dir / f"mainline_replay_{suffix}_stage_outcomes.csv"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_replay_markdown(replay), encoding="utf-8")
    _write_json(json_path, replay)
    _write_csv(yearly_csv, [{"year": year, **row, **{f"stage_{k}": v for k, v in row.get("stage_days", {}).items()}} for year, row in replay["yearly"].items()])
    _write_csv(stage_csv, [{"stage": stage, **row} for stage, row in replay.get("stage_outcomes", {}).items()])
    return {"markdown": str(md_path), "json": str(json_path), "yearly_csv": str(yearly_csv), "stage_csv": str(stage_csv)}


def write_worksheet_outputs(
    worksheet: dict[str, Any],
    *,
    date_value: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    date_text = _date_str(date_value)
    symbol = str(worksheet.get("symbol") or "UNKNOWN").upper()
    output_dir = Path(output_dir)
    md_path = output_dir / f"mainline_worksheet_{symbol}_{date_text}.md"
    json_path = output_dir / f"mainline_worksheet_{symbol}_{date_text}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_worksheet_markdown(worksheet, date_text), encoding="utf-8")
    _write_json(json_path, worksheet)
    return {"markdown": str(md_path), "json": str(json_path)}


def write_weekly_outputs(weekly: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    date_text = _date_str(weekly.get("date"))
    output_dir = Path(output_dir)
    md_path = output_dir / f"mainline_weekly_{date_text}.md"
    csv_path = output_dir / f"mainline_weekly_{date_text}.csv"
    json_path = output_dir / f"mainline_weekly_{date_text}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_weekly_markdown(weekly), encoding="utf-8")
    _write_csv(csv_path, list(weekly.get("rows") or []))
    _write_json(json_path, weekly)
    return {"markdown": str(md_path), "csv": str(csv_path), "json": str(json_path)}


def _run_snapshot(args: argparse.Namespace) -> int:
    state_rows = _read_csv(Path(args.states_csv))
    date_value = latest_state_date(state_rows) if args.latest else str(args.date)
    if not date_value or date_value == "None":
        raise ValueError("snapshot requires --date YYYY-MM-DD or --latest")
    snapshot = build_daily_snapshot(state_rows, date_value)
    paths = write_snapshot_outputs(snapshot, date_value, Path(args.output_dir))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def _run_weekly(args: argparse.Namespace) -> int:
    state_rows = _read_csv(Path(args.states_csv))
    date_value = latest_state_date(state_rows) if args.latest else str(args.date)
    if not date_value or date_value == "None":
        raise ValueError("weekly requires --date YYYY-MM-DD or --latest")
    weekly = build_weekly_radar(state_rows, date_value, lookback_days=int(args.lookback_days))
    paths = write_weekly_outputs(weekly, Path(args.output_dir))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def _latest_symbol_state(
    state_rows: list[dict[str, Any]],
    *,
    symbol: str,
    direction: str,
    date_value: str | None,
) -> dict[str, Any]:
    symbol_key = str(symbol).upper()
    direction_key = str(direction)
    target = _date_str(date_value or latest_state_date(state_rows))
    candidates = [
        row
        for row in state_rows
        if str(row.get("symbol") or "").upper() == symbol_key
        and str(row.get("direction") or "") == direction_key
        and _date_str(row.get("date")) <= target
    ]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda row: _date_str(row.get("date")))[-1]


def _run_worksheet(args: argparse.Namespace) -> int:
    state_rows = _read_csv(Path(args.states_csv))
    date_value = latest_state_date(state_rows) if args.latest or not args.date else str(args.date)
    state = _latest_symbol_state(state_rows, symbol=args.symbol, direction=args.direction, date_value=date_value)
    spec = builtin_contract_spec(str(args.symbol))
    entry = _finite_float(args.entry_price, _finite_float(state.get("close"), 0.0))
    stop = _finite_float(args.stop_price, 0.0)
    if stop <= 0 and entry > 0:
        stop_pct = float(args.stop_pct)
        stop = entry * (1.0 - stop_pct) if str(args.direction) == "long" else entry * (1.0 + stop_pct)
    multiplier = _finite_float(args.multiplier, _finite_float(spec.get("multiplier"), 0.0))
    margin_rate = _finite_float(args.margin_rate, _finite_float(spec.get("margin_rate"), 0.0))
    worksheet = build_risk_worksheet(
        symbol=str(args.symbol),
        direction=str(args.direction),
        equity=float(args.equity),
        max_margin_pct=float(args.max_margin_pct),
        risk_pct=float(args.risk_pct),
        entry_price=entry,
        stop_price=stop,
        multiplier=multiplier,
        margin_rate=margin_rate,
    )
    if not state:
        worksheet["warnings"].append("本地状态文件未找到该品种方向，入场价需要人工复核。")
    paths = write_worksheet_outputs(worksheet, date_value=date_value, output_dir=Path(args.output_dir))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    state_rows = _read_csv(Path(args.states_csv))
    representative_rows = _read_csv(Path(args.representative_trades_csv))
    years = [int(year) for year in args.years]
    replay = build_replay_report(state_rows, representative_rows, years=years)
    paths = write_replay_outputs(replay, Path(args.output_dir))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    state_rows = _read_csv(Path(args.states_csv))
    validation = build_effectiveness_validation(
        state_rows,
        years=[int(year) for year in args.years],
        horizons=[int(value) for value in args.horizons],
        top_n=int(args.top_n),
    )
    paths = write_effectiveness_validation_outputs(validation, Path(args.output_dir))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build board-level market mainline radar reports")
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot", help="Generate a daily board mainline snapshot")
    snapshot.add_argument("--date", default=None)
    snapshot.add_argument("--latest", action="store_true")
    snapshot.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    snapshot.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    snapshot.set_defaults(func=_run_snapshot)
    weekly = sub.add_parser("weekly", help="Generate a weekly board mainline radar")
    weekly.add_argument("--date", default=None)
    weekly.add_argument("--latest", action="store_true")
    weekly.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    weekly.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    weekly.add_argument("--lookback-days", type=int, default=5)
    weekly.set_defaults(func=_run_weekly)
    replay = sub.add_parser("replay", help="Replay mainline radar history by year")
    replay.add_argument("--years", type=int, nargs="+", required=True)
    replay.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    replay.add_argument("--representative-trades-csv", type=Path, default=DEFAULT_REPRESENTATIVE_TRADES_CSV)
    replay.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    replay.set_defaults(func=_run_replay)
    validate = sub.add_parser("validate", help="Validate whether radar mainline signals predict future trend behavior")
    validate.add_argument("--years", type=int, nargs="+", required=True)
    validate.add_argument("--horizons", type=int, nargs="+", default=[5, 20, 60])
    validate.add_argument("--top-n", type=int, default=3)
    validate.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    validate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate.set_defaults(func=_run_validate)
    worksheet = sub.add_parser("worksheet", help="Build a trader risk worksheet for one symbol")
    worksheet.add_argument("--symbol", required=True)
    worksheet.add_argument("--direction", choices=("long", "short"), required=True)
    worksheet.add_argument("--equity", type=float, required=True)
    worksheet.add_argument("--max-margin-pct", type=float, required=True)
    worksheet.add_argument("--risk-pct", type=float, required=True)
    worksheet.add_argument("--date", default=None)
    worksheet.add_argument("--latest", action="store_true")
    worksheet.add_argument("--entry-price", type=float, default=None)
    worksheet.add_argument("--stop-price", type=float, default=None)
    worksheet.add_argument("--stop-pct", type=float, default=0.02)
    worksheet.add_argument("--multiplier", type=float, default=None)
    worksheet.add_argument("--margin-rate", type=float, default=None)
    worksheet.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    worksheet.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    worksheet.set_defaults(func=_run_worksheet)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
