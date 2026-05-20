from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import data_cache  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_DATA_TESTS") != "1",
    reason="set RUN_LIVE_DATA_TESTS=1 to hit live market data sources",
)


LIVE_DATE = pd.Timestamp("2025-04-01").date()


def test_live_czce_warehouse_receipt_returns_standard_receipt():
    out = data_cache.get_warehouse_receipt("TA0", as_of_date=LIVE_DATE)

    assert out is not None
    assert out["exchange"] == "CZCE"
    assert out["receipt_total"] > 0
    assert isinstance(out["receipt_change"], float)


@pytest.mark.xfail(
    strict=True,
    reason="AkShare DCE warehouse receipt endpoint currently reaches an upstream HTTP 412/no-table response.",
)
def test_live_dce_warehouse_receipt_returns_standard_receipt():
    out = data_cache.get_warehouse_receipt("M0", as_of_date=LIVE_DATE)

    assert out is not None
    assert out["exchange"] == "DCE"
    assert out["receipt_total"] >= 0
    assert isinstance(out["receipt_change"], float)


@pytest.mark.xfail(
    strict=True,
    reason="AkShare 100ppi spot/basis endpoint currently receives an upstream safety-check page/no-table response.",
)
def test_live_spot_basis_returns_standard_basis():
    out = data_cache.get_spot_basis("RB0", as_of_date=LIVE_DATE)

    assert out is not None
    assert out["commodity_code"] == "RB"
    assert out["spot_price"] > 0
    assert out["dominant_contract_price"] > 0
    assert isinstance(out["basis"], float)
