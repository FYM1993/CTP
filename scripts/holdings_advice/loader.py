from __future__ import annotations

from pathlib import Path

import pandas as pd

_LONG_ALIASES = {"做多", "多", "long", "LONG", "Long"}
_SHORT_ALIASES = {"做空", "空", "short", "SHORT", "Short"}


def normalize_direction(value: str) -> str:
    text = str(value).strip()
    if text in _LONG_ALIASES:
        return "long"
    if text in _SHORT_ALIASES:
        return "short"
    raise ValueError(f"不支持的方向值: {value}")


def load_holding_contexts(path: Path) -> list[dict]:
    holdings_df = pd.read_excel(path, sheet_name="持仓录入")
    recommendation_df = pd.read_excel(path, sheet_name="原始推荐")

    recommendation_by_id: dict[str, dict] = {}
    for _, row in recommendation_df.iterrows():
        record_value = row.get("记录ID")
        if pd.isna(record_value):
            continue
        record_id = _base_record_id(record_value)
        recommendation_by_id[record_id] = {
            "record_id": record_id,
            "date": str(row.get("推荐日期") or ""),
            "time": str(row.get("推荐时间") or ""),
            "symbol": str(row.get("合约代码") or ""),
            "name": str(row.get("品种名称") or ""),
            "direction": normalize_direction(row.get("方向")),
            "entry_family": normalize_entry_family(row.get("计划类型")),
            "entry": _as_float(row.get("原始入场价")),
            "first_stop": _as_float(row.get("原始第一止损位")),
            "stop": _as_float(row.get("原始止损位")),
            "tp1": _as_float(row.get("原始第一止盈位")),
            "tp2": _as_float(row.get("原始止盈位")),
            "rr": _as_float(row.get("原始第一止盈RR")),
            "admission_rr": _as_float(row.get("原始准入RR")),
            "note": str(row.get("备注") or ""),
        }

    contexts: list[dict] = []
    for _, row in holdings_df.iterrows():
        record_value = row.get("记录ID")
        if pd.isna(record_value):
            continue
        record_id = _base_record_id(record_value)
        contexts.append(
            {
                "record_id": record_id,
                "holding": {
                    "record_id": record_id,
                    "symbol": str(row.get("合约代码") or ""),
                    "name": str(row.get("品种名称") or ""),
                    "direction": normalize_direction(row.get("方向")),
                    "size": _as_float(row.get("手数")),
                    "entry_price": _as_float(row.get("开仓均价")),
                    "open_date": str(row.get("开仓日期") or ""),
                    "note": str(row.get("备注") or ""),
                },
                "recommendation": recommendation_by_id.get(record_id),
            }
        )
    return contexts


def normalize_entry_family(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"趋势", "trend"}:
        return "trend"
    if text in {"反转", "reversal"}:
        return "reversal"
    return text


def _base_record_id(value: object) -> str:
    return str(value).split("-")[0]


def _as_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)
