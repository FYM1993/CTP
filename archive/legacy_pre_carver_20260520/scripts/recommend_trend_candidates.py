from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_historical_trend_episodes import _zigzag_episodes_for_symbol
from data_cache import get_all_symbols


DEFAULT_INPUT = Path("data/cache")
DEFAULT_OUTPUT_DIR = Path("data/reports/mainline_radar")
DEFAULT_PREFIX = "current_trend_candidates"


def _read_prices(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames: list[pd.DataFrame] = []
        for parquet_path in sorted(path.glob("daily_*_live.parquet")):
            parts = parquet_path.name.split("_")
            if len(parts) < 3:
                continue
            symbol = parts[1].upper()
            frame = pd.read_parquet(parquet_path)
            if "date" not in frame or "close" not in frame:
                continue
            frame = frame[["date", "close"]].copy()
            frame["symbol"] = symbol
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"no daily_*_live.parquet files found in {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, usecols=["date", "symbol", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])
    df = df[df["close"] > 0]
    return df.drop_duplicates(["date", "symbol"], keep="first").sort_values(["symbol", "date"])


def _name_map() -> dict[str, str]:
    return {str(item["symbol"]).upper(): str(item["name"]) for item in get_all_symbols()}


def _pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _direction_text(direction: str) -> str:
    return "做多" if direction == "long" else "做空"


def _directional_recent_return(frame: pd.DataFrame, direction: str, lookback: int) -> float | None:
    if len(frame) <= lookback:
        return None
    latest = float(frame.iloc[-1]["close"])
    past = float(frame.iloc[-lookback - 1]["close"])
    if past <= 0:
        return None
    raw = latest / past - 1.0
    return -raw if direction == "short" else raw


def _active_trend_row(
    frame: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    reversal_threshold: float,
    min_watch_return: float,
) -> dict[str, Any] | None:
    episodes = _zigzag_episodes_for_symbol(
        frame.sort_values("date"),
        symbol=symbol,
        name=name,
        reversal_threshold=reversal_threshold,
    )
    if not episodes:
        return None
    last = episodes[-1]
    latest = frame.iloc[-1]
    latest_close = float(latest["close"])
    start_price = float(last["start_price"])
    extreme_price = float(last["end_price"])
    direction = str(last["direction"])
    if start_price <= 0 or latest_close <= 0 or extreme_price <= 0:
        return None
    if direction == "long":
        current_return = latest_close / start_price - 1.0
        extreme_return = extreme_price / start_price - 1.0
        pullback = max(0.0, 1.0 - latest_close / extreme_price)
    else:
        current_return = start_price / latest_close - 1.0
        extreme_return = start_price / extreme_price - 1.0
        pullback = max(0.0, latest_close / extreme_price - 1.0)
    if current_return < min_watch_return or pullback >= reversal_threshold:
        return None

    start_date = pd.Timestamp(last["start_date"])
    trading_days = int((frame["date"] >= start_date).sum())
    ret20 = _directional_recent_return(frame, direction, 20)
    ret60 = _directional_recent_return(frame, direction, 60)
    ret120 = _directional_recent_return(frame, direction, 120)

    if current_return >= 0.20 and trading_days >= 20 and pullback <= 0.10:
        tier = "核心趋势候选"
    elif current_return >= 0.35 and pullback <= 0.14:
        tier = "强趋势但偏后"
    else:
        tier = "早期观察"

    score = (
        current_return * 100.0
        + max(ret60 or 0.0, 0.0) * 25.0
        + max(ret120 or 0.0, 0.0) * 15.0
        - pullback * 50.0
        + min(trading_days, 120) / 120.0 * 5.0
    )
    return {
        "tier": tier,
        "name": name,
        "symbol": symbol,
        "direction": direction,
        "direction_text": _direction_text(direction),
        "trend_start": str(last["start_date"]),
        "latest_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "start_price": round(start_price, 6),
        "latest_close": round(latest_close, 6),
        "extreme_price": round(extreme_price, 6),
        "current_return": current_return,
        "current_return_pct": _pct(current_return),
        "extreme_return": extreme_return,
        "extreme_return_pct": _pct(extreme_return),
        "pullback_from_extreme": pullback,
        "pullback_from_extreme_pct": _pct(pullback),
        "trading_days": trading_days,
        "ret20": ret20,
        "ret20_pct": "" if ret20 is None else _pct(ret20),
        "ret60": ret60,
        "ret60_pct": "" if ret60 is None else _pct(ret60),
        "ret120": ret120,
        "ret120_pct": "" if ret120 is None else _pct(ret120),
        "score": round(score, 6),
    }


def recommend(prices: pd.DataFrame, *, reversal_threshold: float, min_watch_return: float) -> list[dict[str, Any]]:
    names = _name_map()
    rows: list[dict[str, Any]] = []
    for symbol, frame in prices.groupby("symbol", sort=True):
        row = _active_trend_row(
            frame,
            symbol=symbol,
            name=names.get(symbol, symbol),
            reversal_threshold=reversal_threshold,
            min_watch_return=min_watch_return,
        )
        if row is not None:
            rows.append(row)
    tier_order = {"核心趋势候选": 3, "强趋势但偏后": 2, "早期观察": 1}
    return sorted(rows, key=lambda row: (tier_order.get(str(row["tier"]), 0), float(row["score"])), reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tier",
        "name",
        "symbol",
        "direction_text",
        "trend_start",
        "latest_date",
        "start_price",
        "latest_close",
        "current_return_pct",
        "extreme_return_pct",
        "pullback_from_extreme_pct",
        "trading_days",
        "ret20_pct",
        "ret60_pct",
        "ret120_pct",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: list[dict[str, Any]]) -> list[str]:
    headers = ["分层", "品种", "方向", "趋势起点", "最新日期", "当前趋势幅度", "离极值回撤", "持续交易日", "20/60/120日同向收益"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row["tier"]),
                    f"{row['name']}({row['symbol']})",
                    str(row["direction_text"]),
                    str(row["trend_start"]),
                    str(row["latest_date"]),
                    str(row["current_return_pct"]),
                    str(row["pullback_from_extreme_pct"]),
                    str(row["trading_days"]),
                    f"{row['ret20_pct']}/{row['ret60_pct']}/{row['ret120_pct']}",
                ]
            )
            + " |"
        )
    return out


def _write_report(path: Path, *, input_path: Path, prices: pd.DataFrame, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    latest = pd.Timestamp(prices["date"].max()).date().isoformat()
    core = [row for row in rows if row["tier"] == "核心趋势候选"]
    watch = [row for row in rows if row["tier"] != "核心趋势候选"]
    lines = [
        "# 当前品种级趋势候选",
        "",
        "## 结论",
        "",
        "- 这份报告只看品种自己的主连价格趋势，不使用板块、主线、Phase1、Phase2、资金管理或入场规则。",
        f"- 最新本地数据日期为 {latest}，覆盖 {prices['symbol'].nunique()} 个连续主力品种。",
        f"- 候选规则：当前趋势幅度至少 {_pct(args.min_watch_return)}，且还没有出现 {_pct(args.reversal_threshold)} 级别的反向破坏；核心候选要求趋势幅度至少 20%、持续至少 20 个交易日、离趋势极值回撤不超过 10%。",
        f"- 当前核心趋势候选 {len(core)} 个，早期/偏后观察 {len(watch)} 个。",
        "",
        "## 当前候选清单",
        "",
    ]
    lines.extend(_table(rows))
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 这不是入场信号，只是把“当前已经像历史大趋势一样在走”的品种挑出来。",
            "- 趋势幅度越大不代表越适合追，可能只是说明已经走远；具体入场、止损、仓位由人工决定。",
            "- 价格源是本地主连日线缓存，不手工拼接单合约；主连本身仍有换月影响，适合做方向筛选，不等同真实合约成交收益。",
            f"- 输入文件：{input_path}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    prices = _read_prices(args.input)
    rows = recommend(prices, reversal_threshold=args.reversal_threshold, min_watch_return=args.min_watch_return)
    csv_path = args.output_dir / f"{args.prefix}.csv"
    report_path = args.output_dir / f"{args.prefix}_report.md"
    _write_csv(csv_path, rows)
    _write_report(report_path, input_path=args.input, prices=prices, rows=rows, args=args)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    for row in rows[: args.top]:
        print(
            row["tier"],
            row["name"],
            row["symbol"],
            row["direction_text"],
            row["trend_start"],
            row["latest_date"],
            row["current_return_pct"],
            row["pullback_from_extreme_pct"],
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--reversal-threshold", type=float, default=0.15)
    parser.add_argument("--min-watch-return", type=float, default=0.12)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
