from __future__ import annotations


DEFAULT_MARGIN_RATE = 0.12


CONTRACT_MULTIPLIERS: dict[str, float] = {
    "A0": 10.0,
    "AG0": 15.0,
    "AL0": 5.0,
    "AO0": 20.0,
    "AP0": 10.0,
    "AU0": 1000.0,
    "B0": 10.0,
    "BC0": 5.0,
    "BU0": 10.0,
    "C0": 10.0,
    "CF0": 5.0,
    "CJ0": 5.0,
    "CS0": 10.0,
    "CU0": 5.0,
    "CY0": 5.0,
    "EB0": 5.0,
    "EG0": 10.0,
    "FB0": 10.0,
    "FG0": 20.0,
    "FU0": 10.0,
    "HC0": 10.0,
    "I0": 100.0,
    "J0": 100.0,
    "JD0": 10.0,
    "JM0": 60.0,
    "L0": 5.0,
    "LC0": 1.0,
    "LH0": 16.0,
    "LU0": 10.0,
    "M0": 10.0,
    "MA0": 10.0,
    "NI0": 1.0,
    "NR0": 10.0,
    "OI0": 10.0,
    "P0": 10.0,
    "PB0": 5.0,
    "PG0": 20.0,
    "PK0": 5.0,
    "PP0": 5.0,
    "PS0": 3.0,
    "PX0": 5.0,
    "RB0": 10.0,
    "RM0": 10.0,
    "RU0": 10.0,
    "SA0": 20.0,
    "SC0": 1000.0,
    "SF0": 5.0,
    "SH0": 30.0,
    "SI0": 5.0,
    "SM0": 5.0,
    "SN0": 1.0,
    "SP0": 10.0,
    "SR0": 10.0,
    "SS0": 5.0,
    "TA0": 5.0,
    "UR0": 20.0,
    "V0": 5.0,
    "Y0": 10.0,
    "ZN0": 5.0,
}


CONTRACT_MARGIN_RATES: dict[str, float] = {
    "AP0": 0.10,
    "AU0": 0.08,
    "BU0": 0.15,
    "CS0": 0.06,
    "FU0": 0.15,
    "SC0": 0.15,
    "SS0": 0.08,
}


CONTRACT_FEE_SPECS: dict[str, dict[str, float]] = {
    "AG0": {"commission_rate": 0.00005},
    "AP0": {"commission_per_lot": 5.0, "close_today_commission_per_lot": 20.0},
    "AU0": {"commission_per_lot": 10.0},
    "BU0": {"commission_rate": 0.00005},
    "CS0": {"commission_per_lot": 1.5},
    "FU0": {"commission_rate": 0.00005},
    "JM0": {"commission_rate": 0.0001},
    "SA0": {"commission_rate": 0.0002},
    "SC0": {"commission_per_lot": 40.0},
    "SS0": {"commission_per_lot": 2.0},
}


def builtin_contract_spec(symbol: str) -> dict[str, float]:
    normalized = str(symbol).strip().upper()
    multiplier = CONTRACT_MULTIPLIERS.get(normalized)
    if multiplier is None:
        return {}
    spec = {
        "multiplier": float(multiplier),
        "margin_rate": float(CONTRACT_MARGIN_RATES.get(normalized, DEFAULT_MARGIN_RATE)),
    }
    spec.update(CONTRACT_FEE_SPECS.get(normalized, {}))
    return spec
