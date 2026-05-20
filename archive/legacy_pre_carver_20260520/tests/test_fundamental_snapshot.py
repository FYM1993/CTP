from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_cache import BUILTIN_SYMBOLS  # noqa: E402
from market.fundamental_snapshot import build_fundamental_snapshot  # noqa: E402
from market.fundamental_universe import (  # noqa: E402
    DOMAIN_INVENTORY,
    DOMAIN_SEASONALITY,
    DOMAIN_SPOT_BASIS,
    DOMAIN_WAREHOUSE_RECEIPT,
    FUNDAMENTAL_GROUPS,
    commodity_group_for_symbol,
    evidence_domains_for_group,
)


def test_all_builtin_symbols_have_explicit_commodity_group() -> None:
    missing = [
        info["symbol"]
        for info in BUILTIN_SYMBOLS
        if commodity_group_for_symbol(info["symbol"]).name == "unknown"
    ]

    assert missing == []


def test_registered_groups_define_required_and_optional_evidence_domains() -> None:
    used_groups = {
        commodity_group_for_symbol(info["symbol"]).name
        for info in BUILTIN_SYMBOLS
    }

    expected_groups = {
        "black_chain",
        "non_ferrous",
        "precious_metals",
        "energy",
        "chemicals",
        "agriculture_oils",
        "livestock_perishables",
        "new_energy_materials",
    }
    assert expected_groups.issubset(used_groups)

    for group_name in used_groups:
        profile = evidence_domains_for_group(group_name)
        assert profile.name == group_name
        assert profile.required_domains
        assert profile.optional_domains
        assert set(profile.required_domains).isdisjoint(profile.optional_domains)


def test_unknown_symbol_resolves_to_explicit_unknown_group() -> None:
    profile = commodity_group_for_symbol("UNKNOWN0")

    assert profile.name == "unknown"
    assert profile.display_name == "未知品种"
    assert profile.required_domains == ()
    assert "seasonality" in profile.optional_domains


def test_representative_symbols_map_to_expected_groups() -> None:
    assert commodity_group_for_symbol("RB0").name == "black_chain"
    assert commodity_group_for_symbol("CU0").name == "non_ferrous"
    assert commodity_group_for_symbol("AU0").name == "precious_metals"
    assert commodity_group_for_symbol("SC0").name == "energy"
    assert commodity_group_for_symbol("TA0").name == "chemicals"
    assert commodity_group_for_symbol("M0").name == "agriculture_oils"
    assert commodity_group_for_symbol("LH0").name == "livestock_perishables"
    assert commodity_group_for_symbol("PS0").name == "new_energy_materials"


def test_group_registry_exposes_only_known_profiles() -> None:
    assert "unknown" in FUNDAMENTAL_GROUPS
    for group_name, profile in FUNDAMENTAL_GROUPS.items():
        assert group_name == profile.name
        assert profile.display_name


def test_snapshot_confirms_long_reversal_from_oversupply_and_fundamental_turn() -> None:
    snapshot = build_fundamental_snapshot(
        symbol="RB0",
        name="螺纹钢",
        exchange="shfe",
        daily_df=pd.DataFrame(
            {
                "date": pd.date_range("2025-03-28", periods=3),
                "close": [3200.0, 3180.0, 3190.0],
                "open_interest": [1000, 980, 940],
            }
        ),
        as_of_date=pd.Timestamp("2025-04-01").date(),
        inventory_fetcher=lambda symbol, as_of_date=None: {
            "inv_now": 800.0,
            "inv_percentile": 92.0,
            "inv_change_4wk": -6.0,
            "inv_trend": "去库",
        },
        warehouse_receipt_fetcher=lambda symbol, as_of_date=None: {
            "receipt_total": 220.0,
            "receipt_change": -18.0,
            "exchange": "SHFE",
        },
        spot_basis_fetcher=lambda symbol, as_of_date=None: {
            "commodity_code": "RB",
            "spot_price": 3181.33,
            "dominant_contract_price": 3170.0,
            "basis": -11.33,
            "basis_rate": -0.003561,
            "data_date": "20250401",
            "source": "akshare_futures_spot_price",
        },
        seasonality_fetcher=lambda daily_df, as_of_date=None: {
            "seasonal_signal": 0.3,
            "hist_avg_return": 1.2,
        },
    )

    assert snapshot["commodity_group"] == "black_chain"
    assert snapshot["evidence_domains_present"] == [
        DOMAIN_INVENTORY,
        DOMAIN_WAREHOUSE_RECEIPT,
        DOMAIN_SPOT_BASIS,
        DOMAIN_SEASONALITY,
    ]
    assert snapshot["evidence_domains_missing"] == []
    assert snapshot["coverage_score"] == 1.0
    assert snapshot["extreme_state_direction"] == "long"
    assert snapshot["marginal_turn_direction"] == "long"
    assert snapshot["fundamental_reversal_confirmed"] is True
    assert "高库存" in snapshot["extreme_state_reasons"]
    assert "去库启动" in snapshot["marginal_turn_reasons"]
    assert snapshot["raw_details"][DOMAIN_SPOT_BASIS]["basis"] == -11.33


def test_snapshot_confirms_short_reversal_from_shortage_and_restocking() -> None:
    snapshot = build_fundamental_snapshot(
        symbol="CU0",
        name="铜",
        exchange="shfe",
        daily_df=pd.DataFrame(),
        as_of_date=pd.Timestamp("2025-04-01").date(),
        inventory_fetcher=lambda symbol, as_of_date=None: {
            "inv_now": 12.0,
            "inv_percentile": 8.0,
            "inv_change_4wk": 7.0,
            "inv_trend": "累库",
        },
        warehouse_receipt_fetcher=lambda symbol, as_of_date=None: {
            "receipt_total": 30.0,
            "receipt_change": 4.0,
            "exchange": "SHFE",
        },
        spot_basis_fetcher=lambda symbol, as_of_date=None: {
            "commodity_code": "CU",
            "spot_price": 80023.33,
            "dominant_contract_price": 80430.0,
            "basis": 406.67,
            "basis_rate": 0.005082,
            "data_date": "20250401",
            "source": "akshare_futures_spot_price",
        },
        seasonality_fetcher=lambda daily_df, as_of_date=None: None,
    )

    assert snapshot["commodity_group"] == "non_ferrous"
    assert snapshot["coverage_score"] == 1.0
    assert snapshot["extreme_state_direction"] == "short"
    assert snapshot["marginal_turn_direction"] == "short"
    assert snapshot["fundamental_reversal_confirmed"] is True
    assert "低库存" in snapshot["extreme_state_reasons"]
    assert "库存回升" in snapshot["marginal_turn_reasons"]


def test_snapshot_does_not_treat_open_interest_as_fundamental_turn() -> None:
    snapshot = build_fundamental_snapshot(
        symbol="RB0",
        name="螺纹钢",
        exchange="shfe",
        daily_df=pd.DataFrame(
            {
                "date": pd.date_range("2025-03-28", periods=3),
                "close": [3200.0, 3150.0, 3100.0],
                "open_interest": [1000, 920, 850],
            }
        ),
        as_of_date=pd.Timestamp("2025-04-01").date(),
        inventory_fetcher=lambda symbol, as_of_date=None: {
            "inv_now": 900.0,
            "inv_percentile": 95.0,
            "inv_change_4wk": 12.0,
            "inv_trend": "累库",
        },
        warehouse_receipt_fetcher=lambda symbol, as_of_date=None: {
            "receipt_total": 260.0,
            "receipt_change": 16.0,
            "exchange": "SHFE",
        },
        spot_basis_fetcher=lambda symbol, as_of_date=None: {
            "commodity_code": "RB",
            "spot_price": 3181.33,
            "dominant_contract_price": 3170.0,
            "basis": -11.33,
            "basis_rate": -0.003561,
            "data_date": "20250401",
            "source": "akshare_futures_spot_price",
        },
        seasonality_fetcher=lambda daily_df, as_of_date=None: None,
    )

    assert snapshot["extreme_state_direction"] == "long"
    assert snapshot["marginal_turn_direction"] == ""
    assert snapshot["fundamental_reversal_confirmed"] is False
    assert snapshot["only_proxy_evidence"] is False
    assert all("持仓" not in reason for reason in snapshot["marginal_turn_reasons"])


def test_snapshot_marks_missing_required_domains_and_blocks_confirmation() -> None:
    snapshot = build_fundamental_snapshot(
        symbol="RB0",
        name="螺纹钢",
        exchange="shfe",
        daily_df=pd.DataFrame(),
        as_of_date=pd.Timestamp("2025-04-01").date(),
        inventory_fetcher=lambda symbol, as_of_date=None: {
            "inv_now": 800.0,
            "inv_percentile": 92.0,
            "inv_change_4wk": -6.0,
            "inv_trend": "去库",
        },
        warehouse_receipt_fetcher=lambda symbol, as_of_date=None: None,
        spot_basis_fetcher=lambda symbol, as_of_date=None: None,
        seasonality_fetcher=lambda daily_df, as_of_date=None: None,
    )

    assert snapshot["evidence_domains_present"] == [DOMAIN_INVENTORY]
    assert snapshot["evidence_domains_missing"] == [DOMAIN_SPOT_BASIS, DOMAIN_WAREHOUSE_RECEIPT]
    assert round(snapshot["coverage_score"], 2) == 0.33
    assert snapshot["coverage_status"] == "partial"
    assert snapshot["fundamental_reversal_confirmed"] is False
    assert snapshot["missing_domain_reasons"] == [
        "缺少现货/基差数据",
        "缺少仓单数据",
    ]
