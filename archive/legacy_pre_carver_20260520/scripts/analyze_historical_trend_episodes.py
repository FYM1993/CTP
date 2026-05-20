from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd

from data_cache import get_all_symbols


DEFAULT_INPUT = Path("data/reports/mainline_radar/weekly_mainline_hold_until_lost_states_2022_2026.csv")
DEFAULT_OUTPUT_DIR = Path("data/reports/mainline_radar")
DEFAULT_PREFIX = "historical_trend_episodes_2022_2025"


def _read_price_rows(path: Path, start: str, end: str) -> pd.DataFrame:
    usecols = ["date", "symbol", "close"]
    df = pd.read_csv(path, usecols=usecols)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df.dropna(subset=["date", "symbol", "close"])
    df = df[df["close"] > 0]
    return df.drop_duplicates(["date", "symbol"], keep="first").sort_values(["symbol", "date"])


def _name_map() -> dict[str, str]:
    return {str(item["symbol"]).upper(): str(item["name"]) for item in get_all_symbols()}


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _direction_text(direction: str) -> str:
    return "做多" if direction == "long" else "做空"


def _price_move_text(direction: str, start_price: float, end_price: float) -> str:
    raw = end_price / start_price - 1.0
    if direction == "long":
        return f"上涨 {_pct(raw)}"
    return f"下跌 {_pct(-raw)}"


def _max_adverse_move(closes: list[float], direction: str) -> float:
    if not closes:
        return 0.0
    if direction == "long":
        best = closes[0]
        worst = 0.0
        for close in closes:
            best = max(best, close)
            worst = min(worst, close / best - 1.0)
        return abs(worst)
    best = closes[0]
    worst = 0.0
    for close in closes:
        best = min(best, close)
        worst = max(worst, close / best - 1.0)
    return abs(worst)


def _efficiency(closes: list[float], direction: str) -> float:
    if len(closes) < 2 or closes[0] <= 0:
        return 0.0
    signed_total = closes[-1] / closes[0] - 1.0
    if direction == "short":
        signed_total = -signed_total
    path = sum(abs(closes[index] / closes[index - 1] - 1.0) for index in range(1, len(closes)))
    return max(signed_total, 0.0) / path if path > 0 else 0.0


def _build_episode(
    *,
    symbol: str,
    name: str,
    dates: list[pd.Timestamp],
    closes: list[float],
    start_idx: int,
    end_idx: int,
) -> dict[str, Any] | None:
    if start_idx == end_idx:
        return None
    start_price = float(closes[start_idx])
    end_price = float(closes[end_idx])
    if start_price <= 0 or end_price <= 0:
        return None
    direction = "long" if end_price > start_price else "short"
    directional_return = end_price / start_price - 1.0
    if direction == "short":
        directional_return = -directional_return
    if directional_return <= 0:
        return None
    left = min(start_idx, end_idx)
    right = max(start_idx, end_idx)
    path = [float(value) for value in closes[left : right + 1]]
    duration = right - left
    return {
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "direction_text": _direction_text(direction),
        "start_date": _date(dates[start_idx]),
        "end_date": _date(dates[end_idx]),
        "start_price": round(start_price, 6),
        "end_price": round(end_price, 6),
        "price_move": _price_move_text(direction, start_price, end_price),
        "directional_return": directional_return,
        "directional_return_pct": _pct(directional_return),
        "duration_trading_days": duration,
        "max_adverse_move": _max_adverse_move(path, direction),
        "max_adverse_move_pct": _pct(_max_adverse_move(path, direction)),
        "efficiency": _efficiency(path, direction),
        "calendar_year_start": str(pd.Timestamp(dates[start_idx]).year),
        "calendar_year_end": str(pd.Timestamp(dates[end_idx]).year),
    }


def _zigzag_episodes_for_symbol(
    frame: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    reversal_threshold: float,
) -> list[dict[str, Any]]:
    dates = frame["date"].tolist()
    closes = [float(value) for value in frame["close"].tolist()]
    if len(closes) < 2:
        return []

    pivot_idx = 0
    pivot_price = closes[0]
    extreme_idx = 0
    extreme_price = closes[0]
    trend = 0
    episodes: list[dict[str, Any]] = []

    for idx in range(1, len(closes)):
        price = closes[idx]
        if trend == 0:
            up = price / pivot_price - 1.0
            down = pivot_price / price - 1.0
            if up >= reversal_threshold:
                trend = 1
                extreme_idx = idx
                extreme_price = price
            elif down >= reversal_threshold:
                trend = -1
                extreme_idx = idx
                extreme_price = price
            else:
                if price < pivot_price:
                    pivot_idx = idx
                    pivot_price = price
                    extreme_idx = idx
                    extreme_price = price
                elif price > pivot_price:
                    pivot_idx = idx
                    pivot_price = price
                    extreme_idx = idx
                    extreme_price = price
            continue

        if trend == 1:
            if price > extreme_price:
                extreme_idx = idx
                extreme_price = price
            elif extreme_price / price - 1.0 >= reversal_threshold:
                episode = _build_episode(
                    symbol=symbol,
                    name=name,
                    dates=dates,
                    closes=closes,
                    start_idx=pivot_idx,
                    end_idx=extreme_idx,
                )
                if episode:
                    episodes.append(episode)
                pivot_idx = extreme_idx
                pivot_price = extreme_price
                trend = -1
                extreme_idx = idx
                extreme_price = price
            continue

        if price < extreme_price:
            extreme_idx = idx
            extreme_price = price
        elif price / extreme_price - 1.0 >= reversal_threshold:
            episode = _build_episode(
                symbol=symbol,
                name=name,
                dates=dates,
                closes=closes,
                start_idx=pivot_idx,
                end_idx=extreme_idx,
            )
            if episode:
                episodes.append(episode)
            pivot_idx = extreme_idx
            pivot_price = extreme_price
            trend = 1
            extreme_idx = idx
            extreme_price = price

    if trend != 0 and extreme_idx != pivot_idx:
        episode = _build_episode(
            symbol=symbol,
            name=name,
            dates=dates,
            closes=closes,
            start_idx=pivot_idx,
            end_idx=extreme_idx,
        )
        if episode:
            episodes.append(episode)
    return episodes


def find_trend_episodes(
    prices: pd.DataFrame,
    *,
    reversal_threshold: float,
    min_return: float,
    min_trading_days: int,
) -> list[dict[str, Any]]:
    names = _name_map()
    episodes: list[dict[str, Any]] = []
    for symbol, frame in prices.groupby("symbol", sort=True):
        symbol_name = names.get(symbol, symbol)
        raw = _zigzag_episodes_for_symbol(
            frame.sort_values("date"),
            symbol=symbol,
            name=symbol_name,
            reversal_threshold=reversal_threshold,
        )
        for row in raw:
            if float(row["directional_return"]) < min_return:
                continue
            if int(row["duration_trading_days"]) < min_trading_days:
                continue
            episodes.append(row)
    return sorted(
        episodes,
        key=lambda row: (
            float(row["directional_return"]),
            int(row["duration_trading_days"]),
            float(row["efficiency"]),
        ),
        reverse=True,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "symbol",
        "direction_text",
        "start_date",
        "end_date",
        "start_price",
        "end_price",
        "price_move",
        "directional_return_pct",
        "duration_trading_days",
        "max_adverse_move_pct",
        "efficiency",
        "calendar_year_start",
        "calendar_year_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], limit: int) -> list[str]:
    headers = ["排名", "品种", "方向", "开始", "结束", "价格变化", "趋势收益", "交易日", "段内最大反向波动"]
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for idx, row in enumerate(rows[:limit], start=1):
        out.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"{row['name']}({row['symbol']})",
                    str(row["direction_text"]),
                    str(row["start_date"]),
                    str(row["end_date"]),
                    str(row["price_move"]),
                    str(row["directional_return_pct"]),
                    str(row["duration_trading_days"]),
                    str(row["max_adverse_move_pct"]),
                ]
            )
            + " |"
        )
    return out


def _year_tables(episodes: list[dict[str, Any]], per_year: int) -> list[str]:
    out: list[str] = []
    years = sorted({str(row["calendar_year_start"]) for row in episodes})
    for year in years:
        rows = [row for row in episodes if str(row["calendar_year_start"]) == year]
        if not rows:
            continue
        out.extend([f"### {year} 年启动的趋势段", ""])
        out.extend(_markdown_table(rows, per_year))
        out.append("")
    return out


def _write_report(
    path: Path,
    *,
    input_path: Path,
    prices: pd.DataFrame,
    episodes: list[dict[str, Any]],
    reversal_threshold: float,
    min_return: float,
    min_trading_days: int,
    top: int,
    per_year: int,
) -> None:
    dates = prices["date"]
    lines = [
        "# 2022-2025 历史大趋势段扫描",
        "",
        "## 结论先看",
        "",
        "- 这份表只回答历史事实：哪些连续主力品种在 2022-2025 年走出过明显多头或空头趋势段。",
        "- 本次没有使用主线雷达、Phase2、基本面故事或任何仓位规则；只使用本地日线收盘价。",
        f"- 趋势段定义为：收盘价出现至少 {_pct(reversal_threshold)} 级别的反向确认后，截取上一段低点到高点或高点到低点；过滤条件是趋势收益至少 {_pct(min_return)}，持续至少 {min_trading_days} 个交易日。",
        f"- 数据覆盖 {prices['symbol'].nunique()} 个品种，日期从 {_date(dates.min())} 到 {_date(dates.max())}；输入文件为 `{input_path}`。",
        "",
        "## 全市场趋势段排行",
        "",
    ]
    lines.extend(_markdown_table(episodes, top))
    lines.extend(["", "## 按启动年份拆分", ""])
    lines.extend(_year_tables(episodes, per_year))
    lines.extend(
        [
            "## 使用边界",
            "",
            "- 连续主力价格适合做趋势复盘，但不等于真实换月成交收益。",
            "- 这一步是历史趋势事实清单，不代表这些趋势在当时可被提前识别。",
            "- 下一步应该从这些趋势段里挑样本，反推启动前 5/10/20 天的共同盘面特征。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    prices = _read_price_rows(args.input, args.start, args.end)
    episodes = find_trend_episodes(
        prices,
        reversal_threshold=args.reversal_threshold,
        min_return=args.min_return,
        min_trading_days=args.min_trading_days,
    )
    csv_path = args.output_dir / f"{args.prefix}.csv"
    report_path = args.output_dir / f"{args.prefix}_report.md"
    _write_csv(csv_path, episodes)
    _write_report(
        report_path,
        input_path=args.input,
        prices=prices,
        episodes=episodes,
        reversal_threshold=args.reversal_threshold,
        min_return=args.min_return,
        min_trading_days=args.min_trading_days,
        top=args.top,
        per_year=args.per_year,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    for idx, row in enumerate(episodes[: args.top], start=1):
        print(
            idx,
            row["name"],
            row["symbol"],
            row["direction_text"],
            row["start_date"],
            row["end_date"],
            row["directional_return_pct"],
            row["price_move"],
        )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--reversal-threshold", type=float, default=0.15)
    parser.add_argument("--min-return", type=float, default=0.20)
    parser.add_argument("--min-trading-days", type=int, default=20)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--per-year", type=int, default=12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
