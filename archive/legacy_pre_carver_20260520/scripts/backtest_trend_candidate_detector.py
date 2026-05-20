from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_historical_trend_episodes import _zigzag_episodes_for_symbol
from data_cache import get_all_symbols
from recommend_trend_candidates import _read_prices as _read_current_prices
from recommend_trend_candidates import recommend


DEFAULT_PRICES = Path("data/reports/mainline_radar/weekly_mainline_hold_until_lost_states_2022_2026.csv")
DEFAULT_CURRENT_PRICES = Path("data/cache")
DEFAULT_EPISODES = Path("data/reports/mainline_radar/historical_trend_episodes_2022_2025.csv")
DEFAULT_OUTPUT_DIR = Path("data/reports/mainline_radar")
DEFAULT_PREFIX = "trend_candidate_detector_backtest_2022_2025"


@dataclass(frozen=True)
class DetectorParams:
    name: str
    signal_return: float
    signal_min_days: int
    max_signal_pullback: float
    reversal_threshold: float = 0.15


def _read_prices(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date", "symbol", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df[df["close"] > 0]
    return df.drop_duplicates(["date", "symbol"], keep="first").sort_values(["symbol", "date"])


def _read_episodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["direction"] = df["direction_text"].map({"做多": "long", "做空": "short"})
    df["directional_return"] = df["directional_return_pct"].astype(str).str.rstrip("%").astype(float) / 100.0
    return df.sort_values(["start_date", "symbol"]).reset_index(drop=True)


def _name_map() -> dict[str, str]:
    return {str(item["symbol"]).upper(): str(item["name"]) for item in get_all_symbols()}


def _pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _direction_text(direction: str) -> str:
    return "做多" if direction == "long" else "做空"


def _directional_return(direction: str, first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return 0.0
    raw = second / first - 1.0
    return -raw if direction == "short" else raw


def _active_signal_for_slice(
    frame: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    params: DetectorParams,
) -> dict[str, Any] | None:
    episodes = _zigzag_episodes_for_symbol(
        frame,
        symbol=symbol,
        name=name,
        reversal_threshold=params.reversal_threshold,
    )
    if not episodes:
        return None
    last = episodes[-1]
    direction = str(last["direction"])
    latest = frame.iloc[-1]
    latest_close = float(latest["close"])
    start_price = float(last["start_price"])
    extreme_price = float(last["end_price"])
    if latest_close <= 0 or start_price <= 0 or extreme_price <= 0:
        return None
    if direction == "long":
        current_return = latest_close / start_price - 1.0
        pullback = max(0.0, 1.0 - latest_close / extreme_price)
    else:
        current_return = start_price / latest_close - 1.0
        pullback = max(0.0, latest_close / extreme_price - 1.0)
    start_date = pd.Timestamp(last["start_date"])
    trading_days = int((frame["date"] >= start_date).sum())
    if current_return < params.signal_return:
        return None
    if trading_days < params.signal_min_days:
        return None
    if pullback > params.max_signal_pullback:
        return None
    return {
        "date": pd.Timestamp(latest["date"]).date().isoformat(),
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "direction_text": _direction_text(direction),
        "trend_start": pd.Timestamp(last["start_date"]).date().isoformat(),
        "close": latest_close,
        "current_return": current_return,
        "pullback": pullback,
        "trading_days": trading_days,
    }


def _daily_signals(prices: pd.DataFrame, params: DetectorParams) -> list[dict[str, Any]]:
    names = _name_map()
    rows: list[dict[str, Any]] = []
    for symbol, frame in prices.groupby("symbol", sort=True):
        ordered = frame.sort_values("date").reset_index(drop=True)
        name = names.get(symbol, symbol)
        for idx in range(max(params.signal_min_days, 2), len(ordered)):
            signal = _active_signal_for_slice(
                ordered.iloc[: idx + 1],
                symbol=symbol,
                name=name,
                params=params,
            )
            if signal is not None:
                rows.append(signal)
    return rows


def _close_index(prices: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row.symbol).upper(), pd.Timestamp(row.date).date().isoformat()): float(row.close)
        for row in prices.itertuples(index=False)
    }


def _date_index(prices: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for symbol, frame in prices.groupby("symbol", sort=True):
        out[symbol] = [pd.Timestamp(value).date().isoformat() for value in frame.sort_values("date")["date"]]
    return out


def _trading_day_distance(dates: list[str], first: str, second: str) -> int | None:
    if first not in dates or second not in dates:
        return None
    return dates.index(second) - dates.index(first)


def _is_inside_episode(signal: dict[str, Any], episodes: pd.DataFrame) -> bool:
    date = pd.Timestamp(signal["date"])
    symbol = str(signal["symbol"])
    direction = str(signal["direction"])
    matches = episodes[
        (episodes["symbol"] == symbol)
        & (episodes["direction"] == direction)
        & (episodes["start_date"] <= date)
        & (episodes["end_date"] >= date)
    ]
    return bool(len(matches))


def _evaluate_episodes(
    *,
    prices: pd.DataFrame,
    episodes: pd.DataFrame,
    signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signals_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for signal in signals:
        signals_by_key.setdefault((str(signal["symbol"]), str(signal["direction"])), []).append(signal)
    for key in list(signals_by_key):
        signals_by_key[key] = sorted(signals_by_key[key], key=lambda row: row["date"])

    closes = _close_index(prices)
    dates_by_symbol = _date_index(prices)
    rows: list[dict[str, Any]] = []
    for episode in episodes.to_dict("records"):
        symbol = str(episode["symbol"])
        direction = str(episode["direction"])
        start_date = pd.Timestamp(episode["start_date"]).date().isoformat()
        end_date = pd.Timestamp(episode["end_date"]).date().isoformat()
        start_price = closes.get((symbol, start_date), float(episode["start_price"]))
        end_price = closes.get((symbol, end_date), float(episode["end_price"]))
        total_return = _directional_return(direction, start_price, end_price)
        candidates = [
            signal
            for signal in signals_by_key.get((symbol, direction), [])
            if start_date <= str(signal["date"]) <= end_date
        ]
        first = candidates[0] if candidates else None
        dates = dates_by_symbol.get(symbol, [])
        if first is None:
            rows.append(
                {
                    "name": episode["name"],
                    "symbol": symbol,
                    "direction_text": episode["direction_text"],
                    "trend_start": start_date,
                    "trend_end": end_date,
                    "trend_return_pct": _pct(total_return),
                    "detected": False,
                    "first_signal_date": "",
                    "signal_progress_pct": "",
                    "remaining_return_after_signal_pct": "",
                    "signal_days_after_start": "",
                    "signal_days_before_end": "",
                    "exit_alert_date": "",
                    "exit_days_after_trend_end": "",
                    "giveback_at_exit_pct": "",
                }
            )
            continue

        signal_date = str(first["date"])
        signal_price = float(first["close"])
        progress = _directional_return(direction, start_price, signal_price) / total_return if total_return > 0 else 1.0
        remaining = _directional_return(direction, signal_price, end_price)
        signal_days_after_start = _trading_day_distance(dates, start_date, signal_date)
        signal_days_before_end = _trading_day_distance(dates, signal_date, end_date)

        later_signals = [
            signal
            for signal in signals_by_key.get((symbol, direction), [])
            if str(signal["date"]) > signal_date
        ]
        later_dates = [date for date in dates if date > signal_date]
        active_dates = {str(signal["date"]) for signal in later_signals}
        exit_date = ""
        for date in later_dates:
            if date not in active_dates:
                exit_date = date
                break
        exit_days_after_end: int | None = None
        giveback = None
        if exit_date:
            exit_days_after_end = _trading_day_distance(dates, end_date, exit_date)
            exit_price = closes.get((symbol, exit_date), 0.0)
            if exit_price > 0:
                giveback = max(0.0, -_directional_return(direction, end_price, exit_price))

        rows.append(
            {
                "name": episode["name"],
                "symbol": symbol,
                "direction_text": episode["direction_text"],
                "trend_start": start_date,
                "trend_end": end_date,
                "trend_return_pct": _pct(total_return),
                "detected": True,
                "first_signal_date": signal_date,
                "signal_progress_pct": _pct(progress),
                "remaining_return_after_signal_pct": _pct(remaining),
                "signal_days_after_start": signal_days_after_start,
                "signal_days_before_end": signal_days_before_end,
                "exit_alert_date": exit_date,
                "exit_days_after_trend_end": "" if exit_days_after_end is None else exit_days_after_end,
                "giveback_at_exit_pct": "" if giveback is None else _pct(giveback),
            }
        )
    return rows


def _summarize(
    *,
    params: DetectorParams,
    episodes: pd.DataFrame,
    signals: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    detected = [row for row in episode_rows if bool(row["detected"])]
    missed = len(episode_rows) - len(detected)
    signal_days = len(signals)
    true_signal_days = sum(1 for signal in signals if _is_inside_episode(signal, episodes))
    progress_values = [
        float(str(row["signal_progress_pct"]).rstrip("%")) / 100.0
        for row in detected
        if str(row["signal_progress_pct"])
    ]
    days_after = [int(row["signal_days_after_start"]) for row in detected if str(row["signal_days_after_start"]) != ""]
    days_before = [int(row["signal_days_before_end"]) for row in detected if str(row["signal_days_before_end"]) != ""]
    remaining_values = [
        float(str(row["remaining_return_after_signal_pct"]).rstrip("%")) / 100.0
        for row in detected
        if str(row["remaining_return_after_signal_pct"])
    ]
    exit_days = [
        int(row["exit_days_after_trend_end"])
        for row in detected
        if str(row["exit_days_after_trend_end"]) not in {"", "None"}
    ]
    givebacks = [
        float(str(row["giveback_at_exit_pct"]).rstrip("%")) / 100.0
        for row in detected
        if str(row["giveback_at_exit_pct"])
    ]
    return {
        "variant": params.name,
        "signal_return": params.signal_return,
        "signal_min_days": params.signal_min_days,
        "max_signal_pullback": params.max_signal_pullback,
        "episodes": len(episode_rows),
        "detected": len(detected),
        "missed": missed,
        "recall": len(detected) / max(len(episode_rows), 1),
        "early_half_recall": sum(1 for value in progress_values if value <= 0.50) / max(len(episode_rows), 1),
        "median_signal_progress": pd.Series(progress_values).median() if progress_values else 0.0,
        "median_days_after_start": pd.Series(days_after).median() if days_after else 0.0,
        "median_days_before_end": pd.Series(days_before).median() if days_before else 0.0,
        "median_remaining_return": pd.Series(remaining_values).median() if remaining_values else 0.0,
        "signal_days": signal_days,
        "signal_day_precision": true_signal_days / max(signal_days, 1),
        "premature_exit_rate": sum(1 for value in exit_days if value < 0) / max(len(detected), 1),
        "timely_exit_rate": sum(1 for value in exit_days if 0 <= value <= 20) / max(len(detected), 1),
        "median_exit_days_after_end": pd.Series(exit_days).median() if exit_days else 0.0,
        "median_exit_days_after_end_when_not_early": pd.Series([value for value in exit_days if value >= 0]).median()
        if any(value >= 0 for value in exit_days)
        else 0.0,
        "median_giveback_at_exit": pd.Series(givebacks).median() if givebacks else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "口径",
        "发现率",
        "半程前发现",
        "发现时趋势已走",
        "发现后剩余空间",
        "信号日准确率",
        "过早结束误报",
        "结束后20日内提示",
        "非过早提示滞后",
        "结束回吐",
    ]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    _pct(float(row["recall"])),
                    _pct(float(row["early_half_recall"])),
                    _pct(float(row["median_signal_progress"])),
                    _pct(float(row["median_remaining_return"])),
                    _pct(float(row["signal_day_precision"])),
                    _pct(float(row["premature_exit_rate"])),
                    _pct(float(row["timely_exit_rate"])),
                    f"{float(row['median_exit_days_after_end_when_not_early']):.0f} 个交易日",
                    _pct(float(row["median_giveback_at_exit"])),
                ]
            )
            + " |"
        )
    return out


def _episode_table(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    headers = ["品种", "方向", "趋势段", "趋势幅度", "首次发现", "发现时已走", "剩余空间", "结束提示", "结束后滞后"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows[:limit]:
        out.append(
            "| "
            + " | ".join(
                [
                    f"{row['name']}({row['symbol']})",
                    str(row["direction_text"]),
                    f"{row['trend_start']} -> {row['trend_end']}",
                    str(row["trend_return_pct"]),
                    str(row["first_signal_date"] or "未发现"),
                    str(row["signal_progress_pct"] or ""),
                    str(row["remaining_return_after_signal_pct"] or ""),
                    str(row["exit_alert_date"] or ""),
                    str(row["exit_days_after_trend_end"] or ""),
                ]
            )
            + " |"
        )
    return out


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    selected_episode_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    selected_name: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# 品种级趋势发现能力回测",
        "",
        "## 结论",
        "",
        "- 这次只验证一个问题：只看单品种价格，能不能在历史大趋势走完前发现，并在趋势明显转弱时提示结束。",
        "- 回测是因果回放：每个交易日只能使用当天及以前的收盘价；历史趋势段只作为事后答案，不参与当天判断。",
        "- 需要承认边界：只靠价格无法提前知道最高点/最低点；所谓“结束提醒”只能在离极值回撤达到阈值后出现，目标是早于 15% 级别的趋势破坏确认。",
        "- 结束提醒必须同时看“及时”和“误报”：如果在真实趋势高点/低点前就消失，说明它会过早把趋势踢掉。",
        f"- 当前推荐名单使用 `{selected_name}` 口径。",
        "",
        "## 参数口径对比",
        "",
    ]
    lines.extend(_summary_table(summary_rows))
    lines.extend(
        [
            "",
            "## 选定口径下的历史趋势发现明细",
            "",
        ]
    )
    lines.extend(_episode_table(selected_episode_rows, limit=40))
    lines.extend(["", "## 当前趋势推荐品种", ""])
    if current_rows:
        headers = ["品种", "方向", "趋势起点", "当前趋势幅度", "离极值回撤", "持续交易日"]
        lines.extend(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"])
        for row in current_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{row['name']}({row['symbol']})",
                        str(row["direction_text"]),
                        str(row["trend_start"]),
                        str(row["current_return_pct"]),
                        str(row["pullback_from_extreme_pct"]),
                        str(row["trading_days"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("当前没有满足口径的趋势候选。")
    lines.extend(
        [
            "",
            "## 文件",
            "",
            f"- 历史趋势样本：{args.episodes}",
            f"- 价格输入：{args.prices}",
            f"- 详细输出前缀：{args.output_dir / args.prefix}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _variants() -> list[DetectorParams]:
    return [
        DetectorParams("early_12pct_10d", signal_return=0.12, signal_min_days=10, max_signal_pullback=0.10),
        DetectorParams("balanced_15pct_15d", signal_return=0.15, signal_min_days=15, max_signal_pullback=0.10),
        DetectorParams("strict_20pct_20d", signal_return=0.20, signal_min_days=20, max_signal_pullback=0.10),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    prices = _read_prices(args.prices, args.start, args.end)
    episodes = _read_episodes(args.episodes)
    summaries: list[dict[str, Any]] = []
    selected_episode_rows: list[dict[str, Any]] = []
    selected_params: DetectorParams | None = None
    all_details: dict[str, list[dict[str, Any]]] = {}
    for params in _variants():
        signals = _daily_signals(prices, params)
        episode_rows = _evaluate_episodes(prices=prices, episodes=episodes, signals=signals)
        summary = _summarize(params=params, episodes=episodes, signals=signals, episode_rows=episode_rows)
        summaries.append(summary)
        all_details[params.name] = episode_rows
        _write_csv(args.output_dir / f"{args.prefix}_{params.name}_episodes.csv", episode_rows)
        _write_csv(args.output_dir / f"{args.prefix}_{params.name}_signals.csv", signals)
        if params.name == args.selected:
            selected_episode_rows = episode_rows
            selected_params = params

    if selected_params is None:
        raise ValueError(f"unknown selected variant: {args.selected}")

    current_prices = _read_current_prices(args.current_prices)
    current_prices = current_prices[current_prices["date"] <= pd.Timestamp(args.current_end)]
    current_rows = recommend(
        current_prices,
        reversal_threshold=selected_params.reversal_threshold,
        min_watch_return=selected_params.signal_return,
    )
    current_rows = [
        row
        for row in current_rows
        if int(row["trading_days"]) >= selected_params.signal_min_days
        and float(row["pullback_from_extreme"]) <= selected_params.max_signal_pullback
    ]
    _write_csv(args.output_dir / f"{args.prefix}_summary.csv", summaries)
    _write_csv(args.output_dir / f"{args.prefix}_current_candidates.csv", current_rows)
    report_path = args.output_dir / f"{args.prefix}_report.md"
    _write_report(
        report_path,
        summary_rows=summaries,
        selected_episode_rows=selected_episode_rows,
        current_rows=current_rows,
        selected_name=args.selected,
        args=args,
    )
    print(f"wrote {report_path}")
    for row in summaries:
        print(
            row["variant"],
            f"recall={row['recall']:.3f}",
            f"early_half={row['early_half_recall']:.3f}",
            f"precision={row['signal_day_precision']:.3f}",
            f"progress={row['median_signal_progress']:.3f}",
            f"exit_lag={row['median_exit_days_after_end']:.1f}",
        )
    print("current")
    for row in current_rows[: args.top]:
        print(row["name"], row["symbol"], row["direction_text"], row["trend_start"], row["current_return_pct"], row["pullback_from_extreme_pct"])
    return {"summaries": summaries, "current": current_rows, "details": all_details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--current-prices", type=Path, default=DEFAULT_CURRENT_PRICES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--current-end", default="2026-05-08")
    parser.add_argument("--selected", default="early_12pct_10d")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
