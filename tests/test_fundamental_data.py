from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fundamental_edb_map as edb_map  # noqa: E402
from fundamental_data import get_inventory_fundamental  # noqa: E402


def test_inventory_falls_back_to_akshare_when_edb_unmapped_or_empty():
    calls: list[tuple[str, object]] = []

    def fake_edb_fetcher(edb_id: int):
        calls.append(("edb", edb_id))
        return None

    def fake_akshare_fetcher(symbol: str):
        calls.append(("akshare", symbol))
        return {"inv_change_4wk": 10.0, "source": "akshare"}

    result = get_inventory_fundamental(
        "AU0",
        edb_fetcher=fake_edb_fetcher,
        akshare_fetcher=fake_akshare_fetcher,
    )
    assert result.source == "akshare"
    assert result.data == {"inv_change_4wk": 10.0, "source": "akshare"}
    assert calls == [("akshare", "AU0")]

    calls.clear()
    result = get_inventory_fundamental(
        "CU0",
        edb_fetcher=fake_edb_fetcher,
        akshare_fetcher=fake_akshare_fetcher,
    )
    assert result.source == "akshare"
    assert result.data == {"inv_change_4wk": 10.0, "source": "akshare"}
    assert calls == [("edb", 1), ("akshare", "CU0")]


def test_inventory_propagates_edb_fetcher_error():
    def fake_edb_fetcher(edb_id: int):
        raise RuntimeError(f"edb failed: {edb_id}")

    def fake_akshare_fetcher(symbol: str):
        raise AssertionError("akshare should not be called")

    try:
        get_inventory_fundamental(
            "CU0",
            edb_fetcher=fake_edb_fetcher,
            akshare_fetcher=fake_akshare_fetcher,
        )
    except RuntimeError as exc:
        assert str(exc) == "edb failed: 1"
    else:
        raise AssertionError("RuntimeError not raised")


def test_inventory_prefers_edb_when_mapped_and_data_available():
    calls: list[tuple[str, object]] = []

    def fake_edb_fetcher(edb_id: int):
        calls.append(("edb", edb_id))
        return {"inv_now": 123.0, "source": "edb"}

    def fake_akshare_fetcher(symbol: str):
        calls.append(("akshare", symbol))
        return {"inv_change_4wk": 10.0, "source": "akshare"}

    result = get_inventory_fundamental(
        "CU0",
        edb_fetcher=fake_edb_fetcher,
        akshare_fetcher=fake_akshare_fetcher,
    )
    assert result.source == "edb"
    assert result.data == {"inv_now": 123.0, "source": "edb"}
    assert result.edb_id == 1
    assert calls == [("edb", 1)]


def test_edb_registry_only_contains_confirmed_inventory_mappings():
    assert edb_map.EDB_FIELD_MAP["库存"] == {"CU0": 1, "AG0": 2}
    assert edb_map.EDB_FIELD_MAP["仓单"] == {}
    assert edb_map.EDB_FIELD_MAP["利润"] == {}
    assert edb_map.EDB_FIELD_MAP["成本"] == {}
    assert edb_map.EDB_FIELD_MAP["现货"] == {}
    assert edb_map.EDB_FIELD_MAP["生猪"] == {}
