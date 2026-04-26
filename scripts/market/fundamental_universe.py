from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FundamentalGroupProfile:
    name: str
    display_name: str
    required_domains: tuple[str, ...]
    optional_domains: tuple[str, ...]


DOMAIN_SPOT_BASIS = "spot_basis"
DOMAIN_INVENTORY = "inventory"
DOMAIN_WAREHOUSE_RECEIPT = "warehouse_receipt"
DOMAIN_MARGIN_COST = "margin_cost"
DOMAIN_SUPPLY = "supply"
DOMAIN_DEMAND = "demand"
DOMAIN_SEASONALITY = "seasonality"


FUNDAMENTAL_GROUPS: dict[str, FundamentalGroupProfile] = {
    "black_chain": FundamentalGroupProfile(
        name="black_chain",
        display_name="黑色产业链",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY, DOMAIN_WAREHOUSE_RECEIPT),
        optional_domains=(DOMAIN_MARGIN_COST, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "non_ferrous": FundamentalGroupProfile(
        name="non_ferrous",
        display_name="有色金属",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY, DOMAIN_WAREHOUSE_RECEIPT),
        optional_domains=(DOMAIN_MARGIN_COST, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "precious_metals": FundamentalGroupProfile(
        name="precious_metals",
        display_name="贵金属",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY),
        optional_domains=(DOMAIN_WAREHOUSE_RECEIPT, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "energy": FundamentalGroupProfile(
        name="energy",
        display_name="能源",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY),
        optional_domains=(DOMAIN_WAREHOUSE_RECEIPT, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "chemicals": FundamentalGroupProfile(
        name="chemicals",
        display_name="化工建材",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY, DOMAIN_WAREHOUSE_RECEIPT),
        optional_domains=(DOMAIN_MARGIN_COST, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "agriculture_oils": FundamentalGroupProfile(
        name="agriculture_oils",
        display_name="农产品油脂油料",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY),
        optional_domains=(DOMAIN_WAREHOUSE_RECEIPT, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "livestock_perishables": FundamentalGroupProfile(
        name="livestock_perishables",
        display_name="养殖与鲜品",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_SUPPLY),
        optional_domains=(DOMAIN_INVENTORY, DOMAIN_MARGIN_COST, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "new_energy_materials": FundamentalGroupProfile(
        name="new_energy_materials",
        display_name="新能源材料",
        required_domains=(DOMAIN_SPOT_BASIS, DOMAIN_INVENTORY, DOMAIN_WAREHOUSE_RECEIPT),
        optional_domains=(DOMAIN_MARGIN_COST, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "other_industrials": FundamentalGroupProfile(
        name="other_industrials",
        display_name="其他工业品",
        required_domains=(DOMAIN_SPOT_BASIS,),
        optional_domains=(DOMAIN_INVENTORY, DOMAIN_WAREHOUSE_RECEIPT, DOMAIN_SUPPLY, DOMAIN_DEMAND, DOMAIN_SEASONALITY),
    ),
    "unknown": FundamentalGroupProfile(
        name="unknown",
        display_name="未知品种",
        required_domains=(),
        optional_domains=(DOMAIN_SEASONALITY,),
    ),
}


SYMBOL_GROUPS: dict[str, str] = {
    # Black chain
    "I0": "black_chain",
    "J0": "black_chain",
    "JM0": "black_chain",
    "RB0": "black_chain",
    "HC0": "black_chain",
    "SS0": "black_chain",
    "SF0": "black_chain",
    "SM0": "black_chain",
    # Non-ferrous
    "CU0": "non_ferrous",
    "AL0": "non_ferrous",
    "ZN0": "non_ferrous",
    "PB0": "non_ferrous",
    "NI0": "non_ferrous",
    "SN0": "non_ferrous",
    "AO0": "non_ferrous",
    "BC0": "non_ferrous",
    # Precious metals
    "AU0": "precious_metals",
    "AG0": "precious_metals",
    # Energy
    "SC0": "energy",
    "FU0": "energy",
    "LU0": "energy",
    "BU0": "energy",
    "PG0": "energy",
    # Chemicals and construction materials
    "V0": "chemicals",
    "L0": "chemicals",
    "PP0": "chemicals",
    "EG0": "chemicals",
    "EB0": "chemicals",
    "TA0": "chemicals",
    "MA0": "chemicals",
    "FG0": "chemicals",
    "SA0": "chemicals",
    "UR0": "chemicals",
    "SH0": "chemicals",
    "PX0": "chemicals",
    "RU0": "chemicals",
    "NR0": "chemicals",
    "SP0": "chemicals",
    "CY0": "chemicals",
    # Agriculture and oils
    "P0": "agriculture_oils",
    "B0": "agriculture_oils",
    "M0": "agriculture_oils",
    "Y0": "agriculture_oils",
    "C0": "agriculture_oils",
    "A0": "agriculture_oils",
    "CS0": "agriculture_oils",
    "OI0": "agriculture_oils",
    "RM0": "agriculture_oils",
    "SR0": "agriculture_oils",
    "CF0": "agriculture_oils",
    "PK0": "agriculture_oils",
    # Livestock and perishables
    "JD0": "livestock_perishables",
    "LH0": "livestock_perishables",
    "AP0": "livestock_perishables",
    "CJ0": "livestock_perishables",
    # New energy materials
    "SI0": "new_energy_materials",
    "LC0": "new_energy_materials",
    "PS0": "new_energy_materials",
    # Sparse-coverage industrials
    "FB0": "other_industrials",
}


def evidence_domains_for_group(group_name: str) -> FundamentalGroupProfile:
    return FUNDAMENTAL_GROUPS.get(group_name, FUNDAMENTAL_GROUPS["unknown"])


def commodity_group_for_symbol(symbol: str) -> FundamentalGroupProfile:
    normalized = str(symbol or "").strip().upper()
    group_name = SYMBOL_GROUPS.get(normalized, "unknown")
    return evidence_domains_for_group(group_name)
