from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from market.fundamental_universe import SYMBOL_GROUPS


@dataclass(frozen=True)
class MarketMainlineParams:
    start_year: int = 2022
    end_year: int = 2025
    leadership_window: int = 20
    medium_score: float = 55.0
    high_score: float = 70.0
    crowded_score: float = 72.0


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Any) -> float:
    number = _finite_float(value, 0.0)
    return float(max(0.0, min(1.0, number)))


def commodity_group(symbol: str) -> str:
    return SYMBOL_GROUPS.get(str(symbol or "").strip().upper(), "unknown")


def discover_cached_symbols(market_cache_dir: Path) -> list[str]:
    out: list[str] = []
    for path in sorted(Path(market_cache_dir).glob("*_20220101_20251231_w60_v3_daily.parquet")):
        prefix = path.name.split("_20220101_20251231_w60_v3_daily.parquet")[0]
        if prefix:
            out.append(prefix.upper())
    return sorted(set(out))


def _clean_daily(symbol: str, daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    data = daily_df.copy()
    if "date" not in data or "close" not in data:
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for field in ("open", "high", "low", "close", "volume", "oi", "open_interest"):
        if field in data:
            data[field] = pd.to_numeric(data[field], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date", kind="stable").reset_index(drop=True)
    data["symbol"] = str(symbol).upper()
    data["group"] = commodity_group(symbol)
    return data


def _atr_abs(data: pd.DataFrame, window: int = 20) -> pd.Series:
    close = pd.to_numeric(data["close"], errors="coerce")
    if {"high", "low"}.issubset(data.columns):
        high = pd.to_numeric(data["high"], errors="coerce")
        low = pd.to_numeric(data["low"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = close.diff().abs()
    atr = tr.rolling(window, min_periods=max(5, window // 2)).mean()
    fallback = (close.abs() * 0.005).where(close > 0, 1e-9)
    return atr.fillna(fallback).clip(lower=1e-9)


def _ratio_score(series: pd.Series, *, low: float, good_low: float, good_high: float, high: float) -> pd.Series:
    value = pd.to_numeric(series, errors="coerce")
    below = ((value - low) / max(good_low - low, 1e-9)).clip(0.0, 1.0)
    above = (1.0 - (value - good_high) / max(high - good_high, 1e-9)).clip(0.0, 1.0)
    score = pd.Series(1.0, index=value.index)
    score = score.where(value >= good_low, below)
    score = score.where(value <= good_high, above)
    return score.fillna(0.5).clip(0.0, 1.0)


def _base_symbol_features(symbol: str, daily_df: pd.DataFrame, params: MarketMainlineParams) -> pd.DataFrame:
    data = _clean_daily(symbol, daily_df)
    if data.empty:
        return data
    close = pd.to_numeric(data["close"], errors="coerce")
    data["ret_5"] = close.pct_change(5)
    data["ret_20"] = close.pct_change(int(params.leadership_window))
    data["ret_60"] = close.pct_change(60)
    data["ma20"] = close.rolling(20, min_periods=10).mean()
    data["ma60"] = close.rolling(60, min_periods=30).mean()
    data["ma120"] = close.rolling(120, min_periods=60).mean()
    data["ma20_slope_5"] = data["ma20"] - data["ma20"].shift(5)
    data["atr20_abs"] = _atr_abs(data, 20)
    data["extension_long_atr"] = (close - data["ma20"]) / data["atr20_abs"]
    data["extension_short_atr"] = (data["ma20"] - close) / data["atr20_abs"]
    if {"high", "low"}.issubset(data.columns):
        data["retrace_long_atr"] = (data["high"].rolling(20, min_periods=5).max() - close).clip(lower=0.0) / data["atr20_abs"]
        data["retrace_short_atr"] = (close - data["low"].rolling(20, min_periods=5).min()).clip(lower=0.0) / data["atr20_abs"]
    else:
        data["retrace_long_atr"] = 0.0
        data["retrace_short_atr"] = 0.0
    volume = pd.to_numeric(data.get("volume", pd.Series(index=data.index, dtype=float)), errors="coerce")
    data["volume_ratio_20_60"] = volume.rolling(20, min_periods=10).mean() / volume.rolling(60, min_periods=30).mean()
    oi_field = "oi" if "oi" in data else "open_interest"
    oi = pd.to_numeric(data.get(oi_field, pd.Series(index=data.index, dtype=float)), errors="coerce")
    data["oi_ratio_20"] = oi / oi.shift(20)
    return data


def _directional_rows(base: pd.DataFrame, direction: str) -> pd.DataFrame:
    data = base.copy()
    is_short = str(direction) == "short"
    data["direction"] = "short" if is_short else "long"
    sign = -1.0 if is_short else 1.0
    data["directional_ret_5"] = sign * data["ret_5"]
    data["directional_ret_20"] = sign * data["ret_20"]
    data["directional_ret_60"] = sign * data["ret_60"]
    extension_field = "extension_short_atr" if is_short else "extension_long_atr"
    retrace_field = "retrace_short_atr" if is_short else "retrace_long_atr"
    data["extension_atr"] = pd.to_numeric(data[extension_field], errors="coerce").fillna(0.0)
    data["retrace_atr"] = pd.to_numeric(data[retrace_field], errors="coerce").fillna(0.0)
    close = pd.to_numeric(data["close"], errors="coerce")
    if is_short:
        ma_checks = pd.concat(
            [
                close.lt(data["ma20"]),
                data["ma20"].lt(data["ma60"]),
                data["ma60"].lt(data["ma120"]),
                data["ma20_slope_5"].lt(0.0),
            ],
            axis=1,
        )
    else:
        ma_checks = pd.concat(
            [
                close.gt(data["ma20"]),
                data["ma20"].gt(data["ma60"]),
                data["ma60"].gt(data["ma120"]),
                data["ma20_slope_5"].gt(0.0),
            ],
            axis=1,
        )
    data["ma_alignment_score"] = ma_checks.mean(axis=1).fillna(0.0) * 100.0
    return data


def _score_extension(extension_atr: pd.Series) -> pd.Series:
    ext = pd.to_numeric(extension_atr, errors="coerce").fillna(0.0)
    score = pd.Series(100.0, index=ext.index)
    score = score.where(ext >= -0.75, ((ext + 1.50) / 0.75 * 100.0).clip(0.0, 100.0))
    score = score.where(ext <= 3.5, (100.0 - (ext - 3.5) / 2.5 * 100.0).clip(0.0, 100.0))
    return score.clip(0.0, 100.0)


def _score_retrace(retrace_atr: pd.Series) -> pd.Series:
    retrace = pd.to_numeric(retrace_atr, errors="coerce").fillna(0.0)
    return (100.0 - (retrace - 2.0).clip(lower=0.0) / 5.0 * 100.0).clip(0.0, 100.0)


def _add_cross_section_scores(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["leadership_rank_pct"] = (
        data.groupby(["date", "direction"])["directional_ret_20"].rank(method="average", pct=True).fillna(0.0)
    )
    data["leadership_top_flag"] = data["leadership_rank_pct"].ge(0.70).astype(float)
    data["leadership_persistence_5d"] = (
        data.sort_values(["symbol", "direction", "date"], kind="stable")
        .groupby(["symbol", "direction"])["leadership_top_flag"]
        .transform(lambda item: item.rolling(5, min_periods=3).mean())
        .fillna(0.0)
    )
    group_direction_date = ["date", "direction", "group"]
    data["group_breadth"] = data.groupby(group_direction_date)["directional_ret_20"].transform(lambda item: item.gt(0.0).mean())
    group_ret = data.groupby(group_direction_date)["directional_ret_20"].transform("mean")
    temp = data[["date", "direction", "group"]].copy()
    temp["group_directional_ret_20"] = group_ret
    temp = temp.drop_duplicates(["date", "direction", "group"])
    temp["group_relative_rank_pct"] = temp.groupby(["date", "direction"])["group_directional_ret_20"].rank(method="average", pct=True)
    data = data.merge(temp, on=["date", "direction", "group"], how="left")
    return data


def _capital_score(data: pd.DataFrame) -> pd.Series:
    price_confirm = pd.to_numeric(data["directional_ret_20"], errors="coerce").fillna(0.0).gt(0.0).astype(float)
    vol_score = _ratio_score(data["volume_ratio_20_60"], low=0.65, good_low=0.95, good_high=1.90, high=3.20)
    oi_ratio = pd.to_numeric(data["oi_ratio_20"], errors="coerce")
    oi_score = pd.Series(0.50, index=data.index)
    oi_score = oi_score.where(oi_ratio < 1.08, 1.0)
    oi_score = oi_score.where(~((oi_ratio >= 0.99) & (oi_ratio < 1.08)), 0.75)
    oi_score = oi_score.where(~((oi_ratio >= 0.94) & (oi_ratio < 0.99)), 0.45)
    oi_score = oi_score.where(oi_ratio >= 0.94, 0.20)
    oi_score = oi_score.fillna(0.50)
    return 100.0 * (0.30 * price_confirm + 0.35 * vol_score + 0.35 * oi_score)


def _structure_score(data: pd.DataFrame) -> pd.Series:
    extension = _score_extension(data["extension_atr"])
    retrace = _score_retrace(data["retrace_atr"])
    ma = pd.to_numeric(data["ma_alignment_score"], errors="coerce").fillna(0.0)
    return (0.40 * ma + 0.35 * extension + 0.25 * retrace).clip(0.0, 100.0)


def _crowding_score(data: pd.DataFrame) -> pd.Series:
    extension = pd.to_numeric(data["extension_atr"], errors="coerce").fillna(0.0)
    extension_score = ((extension - 4.0) / 3.0).clip(0.0, 1.0)
    rank_score = ((pd.to_numeric(data["leadership_rank_pct"], errors="coerce").fillna(0.0) - 0.88) / 0.12).clip(0.0, 1.0)
    persistence = ((pd.to_numeric(data["leadership_persistence_5d"], errors="coerce").fillna(0.0) - 0.80) / 0.20).clip(0.0, 1.0)
    volume_extreme = ((pd.to_numeric(data["volume_ratio_20_60"], errors="coerce").fillna(1.0) - 2.20) / 1.30).clip(0.0, 1.0)
    return 100.0 * (0.40 * extension_score + 0.25 * rank_score + 0.20 * persistence + 0.15 * volume_extreme)


def _add_group_mainline_scores(directional: pd.DataFrame, params: MarketMainlineParams) -> pd.DataFrame:
    data = directional.copy()
    symbol_raw = (
        0.30 * data["mainline_leadership_score"]
        + 0.25 * data["mainline_capital_score"]
        + 0.25 * data["mainline_resonance_score"]
        + 0.20 * data["mainline_structure_score"]
    )
    symbol_raw = symbol_raw.where(data["mainline_leadership_score"] >= 40.0, symbol_raw.clip(upper=52.0))
    symbol_raw = symbol_raw.where(data["mainline_resonance_score"] >= 38.0, symbol_raw.clip(upper=58.0))
    symbol_raw = symbol_raw.where(data["mainline_capital_score"] >= 35.0, symbol_raw.clip(upper=60.0))
    data["symbol_setup_score"] = symbol_raw.clip(0.0, 100.0)
    data["symbol_setup_bucket"] = [
        _bucket(float(score), float(crowding), params)
        for score, crowding in zip(data["symbol_setup_score"], data["mainline_crowding_score"])
    ]

    group_keys = ["date", "direction", "group"]
    group = (
        data.groupby(group_keys, as_index=False)
        .agg(
            group_mainline_ret_20=("directional_ret_20", "mean"),
            group_mainline_ret_60=("directional_ret_60", "mean"),
            group_relative_rank_pct=("group_relative_rank_pct", "first"),
            group_breadth=("group_breadth", "first"),
            group_capital_score=("mainline_capital_score", "mean"),
            group_structure_score=("mainline_structure_score", "mean"),
            group_crowding_score=("mainline_crowding_score", "max"),
            group_symbol_count=("symbol", "nunique"),
            group_best_symbol_score=("symbol_setup_score", "max"),
        )
        .sort_values(group_keys, kind="stable")
    )
    group["group_leadership_top_flag"] = group["group_relative_rank_pct"].ge(0.70).astype(float)
    group["group_leadership_persistence_5d"] = (
        group.groupby(["group", "direction"])["group_leadership_top_flag"]
        .transform(lambda item: item.rolling(5, min_periods=3).mean())
        .fillna(0.0)
    )
    group["group_leadership_score"] = (
        100.0 * (0.65 * group["group_relative_rank_pct"].fillna(0.0) + 0.35 * group["group_leadership_persistence_5d"])
    ).clip(0.0, 100.0)
    group["group_resonance_score"] = (
        100.0 * (0.55 * group["group_breadth"].fillna(0.0) + 0.45 * group["group_relative_rank_pct"].fillna(0.0))
    ).clip(0.0, 100.0)
    raw = (
        0.35 * group["group_leadership_score"]
        + 0.25 * group["group_capital_score"]
        + 0.25 * group["group_resonance_score"]
        + 0.15 * group["group_structure_score"]
    )
    raw = raw.where(group["group_leadership_score"] >= 45.0, raw.clip(upper=52.0))
    raw = raw.where(group["group_resonance_score"] >= 45.0, raw.clip(upper=58.0))
    raw = raw.where(group["group_capital_score"] >= 35.0, raw.clip(upper=60.0))
    group["group_mainline_score"] = raw.clip(0.0, 100.0)
    group["group_mainline_bucket"] = [
        _bucket(float(score), float(crowding), params)
        for score, crowding in zip(group["group_mainline_score"], group["group_crowding_score"])
    ]

    leader_idx = data.groupby(group_keys)["symbol_setup_score"].idxmax()
    leaders = data.loc[leader_idx, group_keys + ["symbol"]].rename(columns={"symbol": "group_leader_symbol"})
    group = group.merge(leaders, on=group_keys, how="left")

    data = data.merge(
        group[
            group_keys
            + [
                "group_mainline_ret_20",
                "group_mainline_ret_60",
                "group_leadership_score",
                "group_leadership_persistence_5d",
                "group_capital_score",
                "group_resonance_score",
                "group_structure_score",
                "group_crowding_score",
                "group_mainline_score",
                "group_mainline_bucket",
                "group_symbol_count",
                "group_best_symbol_score",
                "group_leader_symbol",
            ]
        ],
        on=group_keys,
        how="left",
    )
    data["mainline_score"] = data["group_mainline_score"]
    data["bucket"] = data["group_mainline_bucket"]
    data["mainline_leadership_score"] = data["group_leadership_score"]
    data["mainline_capital_score"] = data["group_capital_score"]
    data["mainline_resonance_score"] = data["group_resonance_score"]
    data["mainline_structure_score"] = data["group_structure_score"]
    data["mainline_crowding_score"] = data["group_crowding_score"]
    data["directional_ret_20"] = data["group_mainline_ret_20"]
    data["directional_ret_60"] = data["group_mainline_ret_60"]
    return data


def _bucket(score: float, crowding: float, params: MarketMainlineParams) -> str:
    if score >= params.crowded_score and crowding >= 55.0:
        return "crowded"
    if score >= params.high_score:
        return "high"
    if score >= params.medium_score:
        return "medium"
    return "low"


def build_market_mainline_state_rows(
    market_data: dict[str, pd.DataFrame],
    *,
    params: MarketMainlineParams | None = None,
) -> list[dict[str, Any]]:
    resolved = params or MarketMainlineParams()
    bases = [_base_symbol_features(symbol, frame, resolved) for symbol, frame in sorted(market_data.items())]
    bases = [frame for frame in bases if not frame.empty]
    if not bases:
        return []
    directional = pd.concat(
        [_directional_rows(frame, "long") for frame in bases] + [_directional_rows(frame, "short") for frame in bases],
        ignore_index=True,
    )
    directional = _add_cross_section_scores(directional)
    directional["mainline_leadership_score"] = (
        100.0 * (0.65 * directional["leadership_rank_pct"] + 0.35 * directional["leadership_persistence_5d"])
    ).clip(0.0, 100.0)
    directional["mainline_capital_score"] = _capital_score(directional).clip(0.0, 100.0)
    directional["mainline_resonance_score"] = (
        100.0 * (0.55 * directional["group_breadth"] + 0.45 * directional["group_relative_rank_pct"].fillna(0.0))
    ).clip(0.0, 100.0)
    directional["mainline_structure_score"] = _structure_score(directional)
    directional["mainline_crowding_score"] = _crowding_score(directional).clip(0.0, 100.0)
    directional = _add_group_mainline_scores(directional, resolved)
    start = pd.Timestamp(f"{int(resolved.start_year)}-01-01")
    end = pd.Timestamp(f"{int(resolved.end_year)}-12-31")
    directional = directional.loc[(directional["date"] >= start) & (directional["date"] <= end)].copy()

    rows: list[dict[str, Any]] = []
    fields = [
        "symbol",
        "direction",
        "date",
        "close",
        "group",
        "bucket",
        "mainline_score",
        "mainline_leadership_score",
        "mainline_capital_score",
        "mainline_resonance_score",
        "mainline_structure_score",
        "mainline_crowding_score",
        "symbol_setup_score",
        "symbol_setup_bucket",
        "group_symbol_count",
        "group_best_symbol_score",
        "group_leader_symbol",
        "leadership_rank_pct",
        "leadership_persistence_5d",
        "group_leadership_persistence_5d",
        "group_breadth",
        "group_relative_rank_pct",
        "directional_ret_20",
        "directional_ret_60",
        "volume_ratio_20_60",
        "oi_ratio_20",
        "extension_atr",
        "retrace_atr",
    ]
    for row in directional[fields].to_dict("records"):
        item = dict(row)
        item["date"] = pd.Timestamp(item["date"]).date().isoformat()
        item["score"] = _finite_float(item.get("mainline_score"), math.nan)
        item["term_structure_status"] = "not_available_in_continuous_cache"
        rows.append(item)
    return rows


def load_market_data(symbols: Iterable[str], market_cache_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted({str(item).upper() for item in symbols}):
        path = Path(market_cache_dir) / f"{symbol.lower()}_20220101_20251231_w60_v3_daily.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path)
    return out
