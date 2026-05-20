from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "open_interest"]


def symbol_from_cache_path(path: Path) -> str:
    match = re.match(r"daily_([A-Z0-9]+)_\d{8}_live\.parquet$", path.name)
    if not match:
        raise ValueError(f"Unsupported daily cache filename: {path.name}")
    return match.group(1)


def symbol_from_market_path(path: Path) -> str:
    cache_match = re.match(r"daily_([A-Z0-9]+)_\d{8}_(?:live|final)\.parquet$", path.name)
    if cache_match:
        return cache_match.group(1)
    tq_match = re.match(r"([A-Z0-9]+)_tq_main_daily\.parquet$", path.name)
    if tq_match:
        return tq_match.group(1)
    raise ValueError(f"Unsupported market data filename: {path.name}")


def _normalize_market_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = raw.copy()
    if "date" not in frame.columns and "datetime" in frame.columns:
        frame["date"] = pd.to_datetime(frame["datetime"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["symbol"] = symbol
    if "open_interest" not in frame.columns and "oi" in frame.columns:
        frame["open_interest"] = frame["oi"]
    if "open_interest" not in frame.columns and "hold" in frame.columns:
        frame["open_interest"] = frame["hold"]
    if "open_interest" not in frame.columns:
        frame["open_interest"] = 0.0
    selected = frame[REQUIRED_COLUMNS].copy()
    selected = selected.sort_values("date").drop_duplicates(["date", "symbol"], keep="last")
    numeric_columns = ["open", "high", "low", "close", "volume", "open_interest"]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected.reset_index(drop=True)


def load_market_data_file(path: Path) -> pd.DataFrame:
    symbol = symbol_from_market_path(path)
    raw = pd.read_parquet(path)
    return _normalize_market_frame(raw, symbol)


def load_daily_cache_file(path: Path) -> pd.DataFrame:
    symbol_from_cache_path(path)
    return load_market_data_file(path)


def load_daily_cache_dir(cache_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(cache_dir.glob("daily_*_live.parquet"))
    return {symbol_from_cache_path(path): load_daily_cache_file(path) for path in files}


def load_market_data_dir(directory: Path, pattern: str = "*.parquet") -> dict[str, pd.DataFrame]:
    files = sorted(directory.glob(pattern))
    return {symbol_from_market_path(path): load_market_data_file(path) for path in files}


def audit_market_data(frames: dict[str, pd.DataFrame], min_history_days: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, frame in sorted(frames.items()):
        issues: list[str] = []
        if len(frame) < min_history_days:
            issues.append("short_history")
        if frame["close"].isna().any():
            issues.append("missing_close")
        if (frame["close"] <= 0).any():
            issues.append("non_positive_close")
        if not frame["date"].is_monotonic_increasing:
            issues.append("date_not_sorted")
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start_date": frame["date"].min().date().isoformat() if len(frame) else "",
                "end_date": frame["date"].max().date().isoformat() if len(frame) else "",
                "is_tradeable": len(issues) == 0,
                "issues": ",".join(issues),
            }
        )
    return pd.DataFrame(rows)
