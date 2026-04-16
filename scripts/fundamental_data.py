from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fundamental_edb_map import EDB_FIELD_MAP


@dataclass(slots=True)
class FundamentalBundle:
    """统一后的基本面数据包。"""

    symbol: str
    source: str
    data: dict[str, Any] | None
    edb_id: int | None = None


def _has_data(payload: dict[str, Any] | None) -> bool:
    return payload is not None and bool(payload)


def get_inventory_fundamental(
    symbol: str,
    *,
    edb_fetcher: Callable[[int], dict[str, Any] | None],
    akshare_fetcher: Callable[[str], dict[str, Any] | None],
) -> FundamentalBundle:
    """获取库存基本面，优先 EDB，缺失时回退 AkShare。"""

    edb_id = EDB_FIELD_MAP["库存"].get(symbol)
    edb_payload: dict[str, Any] | None = None
    if edb_id is not None:
        edb_payload = edb_fetcher(edb_id)
        if _has_data(edb_payload):
            return FundamentalBundle(
                symbol=symbol,
                source="edb",
                data=edb_payload,
                edb_id=edb_id,
            )

    akshare_payload = akshare_fetcher(symbol)

    return FundamentalBundle(
        symbol=symbol,
        source="akshare",
        data=akshare_payload,
        edb_id=edb_id,
    )
