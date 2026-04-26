from __future__ import annotations

from datetime import date, datetime
import inspect
import warnings
from typing import Any, Callable

import pandas as pd

from market.fundamental_data import (
    FundamentalDataPool,
    FundamentalEvidence,
    get_spot_basis_fundamental,
)
from market.fundamental_universe import (
    DOMAIN_DEMAND,
    DOMAIN_INVENTORY,
    DOMAIN_MARGIN_COST,
    DOMAIN_SEASONALITY,
    DOMAIN_SPOT_BASIS,
    DOMAIN_SUPPLY,
    DOMAIN_WAREHOUSE_RECEIPT,
    commodity_group_for_symbol,
)


DOMAIN_LABELS = {
    DOMAIN_SPOT_BASIS: "现货/基差",
    DOMAIN_INVENTORY: "库存",
    DOMAIN_WAREHOUSE_RECEIPT: "仓单",
    DOMAIN_MARGIN_COST: "成本/利润",
    DOMAIN_SUPPLY: "供给",
    DOMAIN_DEMAND: "需求",
    DOMAIN_SEASONALITY: "季节性",
}


def _normalize_date(value: date | datetime | pd.Timestamp | None) -> str:
    if value is None:
        return ""
    return pd.Timestamp(value).date().isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _call_symbol_fetcher(
    fetcher: Callable[..., dict[str, Any] | None] | None,
    symbol: str,
    *,
    as_of_date: Any,
) -> dict[str, Any] | None:
    if fetcher is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _supports_as_of_date(fetcher):
            return fetcher(symbol, as_of_date=as_of_date)
        return fetcher(symbol)


def _supports_as_of_date(func: Any) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "as_of_date" in signature.parameters


def _call_frame_fetcher(
    fetcher: Callable[..., dict[str, Any] | None] | None,
    frame: pd.DataFrame | None,
    *,
    as_of_date: Any,
) -> dict[str, Any] | None:
    if fetcher is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _supports_as_of_date(fetcher):
            return fetcher(frame, as_of_date=as_of_date)
        return fetcher(frame)


def _evidence_from_payload(
    domain: str,
    payload: dict[str, Any] | None,
    *,
    metadata_keys: tuple[str, ...] = ("source", "data_date", "exchange"),
) -> FundamentalEvidence | None:
    if not payload:
        return None
    metadata = {
        key: payload.get(key)
        for key in metadata_keys
        if payload.get(key) is not None
    }
    values = {
        key: value
        for key, value in payload.items()
        if key not in metadata
    }
    if not values:
        return None
    return FundamentalEvidence(domain=domain, values=values, metadata=metadata)


def _seasonality_evidence(payload: dict[str, Any] | None) -> FundamentalEvidence | None:
    return _evidence_from_payload(DOMAIN_SEASONALITY, payload)


def _receipt_evidence(payload: dict[str, Any] | None) -> FundamentalEvidence | None:
    return _evidence_from_payload(DOMAIN_WAREHOUSE_RECEIPT, payload, metadata_keys=("exchange", "source", "data_date"))


def _inventory_evidence(payload: dict[str, Any] | None) -> FundamentalEvidence | None:
    return _evidence_from_payload(DOMAIN_INVENTORY, payload)


def _margin_cost_evidence(payload: dict[str, Any] | None) -> FundamentalEvidence | None:
    return _evidence_from_payload(DOMAIN_MARGIN_COST, payload)


def _first_direction(long_reasons: list[str], short_reasons: list[str]) -> tuple[str, list[str]]:
    if long_reasons and not short_reasons:
        return "long", long_reasons
    if short_reasons and not long_reasons:
        return "short", short_reasons
    if len(long_reasons) > len(short_reasons):
        return "long", long_reasons
    if len(short_reasons) > len(long_reasons):
        return "short", short_reasons
    return "", []


def _extreme_state(pool: FundamentalDataPool) -> tuple[str, list[str]]:
    inventory = pool.values(DOMAIN_INVENTORY)
    receipt = pool.values(DOMAIN_WAREHOUSE_RECEIPT)
    margin = pool.values(DOMAIN_MARGIN_COST)

    inv_percentile = _float_or_none(inventory.get("inv_percentile"))
    receipt_change = _float_or_none(receipt.get("receipt_change"))
    profit_margin = _float_or_none(margin.get("profit_margin"))

    long_reasons: list[str] = []
    short_reasons: list[str] = []

    if inv_percentile is not None and inv_percentile >= 80:
        long_reasons.append("高库存")
    if inv_percentile is not None and inv_percentile <= 20:
        short_reasons.append("低库存")

    if receipt_change is not None and receipt_change > 0 and (inv_percentile is None or inv_percentile >= 50):
        long_reasons.append("仓单增加")
    if receipt_change is not None and receipt_change < 0 and (inv_percentile is None or inv_percentile <= 50):
        short_reasons.append("仓单减少")

    if profit_margin is not None and profit_margin <= -30:
        long_reasons.append("利润深亏")
    if profit_margin is not None and profit_margin >= 30:
        short_reasons.append("利润高位")

    return _first_direction(long_reasons, short_reasons)


def _marginal_turn(pool: FundamentalDataPool) -> tuple[str, list[str]]:
    inventory = pool.values(DOMAIN_INVENTORY)
    receipt = pool.values(DOMAIN_WAREHOUSE_RECEIPT)
    margin = pool.values(DOMAIN_MARGIN_COST)

    inv_percentile = _float_or_none(inventory.get("inv_percentile"))
    inv_change = _float_or_none(inventory.get("inv_change_4wk"))
    receipt_change = _float_or_none(receipt.get("receipt_change"))
    profit_margin = _float_or_none(margin.get("profit_margin"))
    profit_change = _float_or_none(margin.get("profit_margin_change"))

    long_reasons: list[str] = []
    short_reasons: list[str] = []

    if inv_change is not None and inv_change <= 0 and (inv_percentile is None or inv_percentile >= 50):
        long_reasons.append("去库启动")
    if inv_change is not None and inv_change >= 0 and (inv_percentile is None or inv_percentile <= 50):
        short_reasons.append("库存回升")

    if receipt_change is not None and receipt_change < 0 and (inv_percentile is None or inv_percentile >= 50):
        long_reasons.append("仓单回落")
    if receipt_change is not None and receipt_change > 0 and (inv_percentile is None or inv_percentile <= 50):
        short_reasons.append("仓单回升")

    if profit_margin is not None and profit_margin <= -30 and profit_change is not None and profit_change > 0:
        long_reasons.append("利润压力缓和")
    if profit_margin is not None and profit_margin >= 30 and profit_change is not None and profit_change < 0:
        short_reasons.append("利润高位回落")

    return _first_direction(long_reasons, short_reasons)


def _coverage_score(pool: FundamentalDataPool) -> float:
    profile = commodity_group_for_symbol(pool.symbol)
    required = tuple(profile.required_domains)
    if not required:
        return 1.0
    present = set(pool.present_domains)
    return round(sum(1 for domain in required if domain in present) / len(required), 4)


def _coverage_status(score: float) -> str:
    if score >= 1.0:
        return "complete"
    if score <= 0.0:
        return "missing"
    return "partial"


def _raw_details(pool: FundamentalDataPool) -> dict[str, dict[str, Any]]:
    return {
        domain: {
            **pool.values(domain),
            **pool.metadata(domain),
        }
        for domain in pool.present_domains
    }


def _data_source_timestamps(pool: FundamentalDataPool) -> dict[str, str]:
    result: dict[str, str] = {}
    for domain in pool.present_domains:
        metadata = pool.metadata(domain)
        data_date = metadata.get("data_date")
        if data_date is not None:
            result[domain] = str(data_date)
    return result


def build_fundamental_snapshot(
    *,
    symbol: str,
    name: str,
    exchange: str,
    daily_df: pd.DataFrame | None,
    as_of_date: date | datetime | pd.Timestamp | None = None,
    inventory_fetcher: Callable[..., dict[str, Any] | None] | None = None,
    warehouse_receipt_fetcher: Callable[..., dict[str, Any] | None] | None = None,
    spot_basis_fetcher: Callable[..., dict[str, Any] | None] | None = None,
    seasonality_fetcher: Callable[..., dict[str, Any] | None] | None = None,
    margin_cost_fetcher: Callable[..., dict[str, Any] | None] | None = None,
    supply_fetcher: Callable[..., dict[str, Any] | None] | None = None,
    demand_fetcher: Callable[..., dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    if inventory_fetcher is None:
        from data_cache import get_inventory as inventory_fetcher
    if warehouse_receipt_fetcher is None:
        from data_cache import get_warehouse_receipt as warehouse_receipt_fetcher
    if seasonality_fetcher is None:
        from data_cache import get_seasonality as seasonality_fetcher

    normalized_symbol = str(symbol or "").strip().upper()
    profile = commodity_group_for_symbol(normalized_symbol)
    records: list[FundamentalEvidence | None] = [
        _inventory_evidence(_call_symbol_fetcher(inventory_fetcher, normalized_symbol, as_of_date=as_of_date)),
        _receipt_evidence(_call_symbol_fetcher(warehouse_receipt_fetcher, normalized_symbol, as_of_date=as_of_date)),
        get_spot_basis_fundamental(
            normalized_symbol,
            as_of_date=as_of_date,
            spot_basis_fetcher=spot_basis_fetcher,
        ),
    ]

    if seasonality_fetcher is not None:
        records.append(_seasonality_evidence(_call_frame_fetcher(seasonality_fetcher, daily_df, as_of_date=as_of_date)))
    records.append(_margin_cost_evidence(_call_symbol_fetcher(margin_cost_fetcher, normalized_symbol, as_of_date=as_of_date)))
    records.append(_evidence_from_payload(DOMAIN_SUPPLY, _call_symbol_fetcher(supply_fetcher, normalized_symbol, as_of_date=as_of_date)))
    records.append(_evidence_from_payload(DOMAIN_DEMAND, _call_symbol_fetcher(demand_fetcher, normalized_symbol, as_of_date=as_of_date)))

    pool = FundamentalDataPool.from_records(normalized_symbol, records)
    score = _coverage_score(pool)
    extreme_direction, extreme_reasons = _extreme_state(pool)
    turn_direction, turn_reasons = _marginal_turn(pool)
    missing_required = list(pool.missing_required_domains)
    reversal_confirmed = (
        score >= 1.0
        and bool(extreme_direction)
        and extreme_direction == turn_direction
    )

    return {
        "symbol": normalized_symbol,
        "name": name,
        "exchange": exchange,
        "as_of_date": _normalize_date(as_of_date),
        "commodity_group": profile.name,
        "commodity_group_name": profile.display_name,
        "data_source_timestamps": _data_source_timestamps(pool),
        "data_freshness_status": _coverage_status(score),
        "evidence_domains_present": list(pool.present_domains),
        "evidence_domains_missing": missing_required,
        "coverage_score": score,
        "coverage_status": _coverage_status(score),
        "missing_domain_reasons": [
            f"缺少{DOMAIN_LABELS.get(domain, domain)}数据"
            for domain in missing_required
        ],
        "extreme_state_direction": extreme_direction,
        "extreme_state_confirmed": bool(extreme_direction),
        "extreme_state_reasons": extreme_reasons,
        "marginal_turn_direction": turn_direction,
        "marginal_turn_confirmed": bool(turn_direction),
        "marginal_turn_reasons": turn_reasons,
        "fundamental_reversal_confirmed": reversal_confirmed,
        "only_proxy_evidence": False,
        "raw_details": _raw_details(pool),
    }
