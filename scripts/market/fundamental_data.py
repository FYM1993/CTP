from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Callable, Iterable

from market.fundamental_edb_map import EDB_FIELD_MAP
from market.fundamental_universe import DOMAIN_SPOT_BASIS, commodity_group_for_symbol


@dataclass(slots=True)
class FundamentalBundle:
    """统一后的基本面数据包。"""

    symbol: str
    source: str
    data: dict[str, Any] | None
    edb_id: int | None = None


@dataclass(frozen=True, slots=True)
class FundamentalEvidence:
    """标准化后的单个基本面指标域。"""

    domain: str
    values: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FundamentalDataPool:
    """策略消费的标准化基本面数据池。"""

    symbol: str
    group: str
    records: dict[str, FundamentalEvidence]

    @classmethod
    def from_records(
        cls,
        symbol: str,
        records: Iterable[FundamentalEvidence | None],
    ) -> "FundamentalDataPool":
        profile = commodity_group_for_symbol(symbol)
        valid_records = {
            record.domain: record
            for record in records
            if record is not None
        }
        return cls(
            symbol=str(symbol or "").strip().upper(),
            group=profile.name,
            records=valid_records,
        )

    @property
    def present_domains(self) -> tuple[str, ...]:
        return tuple(self.records.keys())

    @property
    def missing_required_domains(self) -> tuple[str, ...]:
        profile = commodity_group_for_symbol(self.symbol)
        return tuple(
            domain
            for domain in profile.required_domains
            if domain not in self.records
        )

    def values(self, domain: str) -> dict[str, Any]:
        record = self.records.get(domain)
        if record is None:
            return {}
        return dict(record.values)

    def metadata(self, domain: str) -> dict[str, Any]:
        record = self.records.get(domain)
        if record is None:
            return {}
        return dict(record.metadata)


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


def get_spot_basis_fundamental(
    symbol: str,
    *,
    as_of_date: Any = None,
    spot_basis_fetcher: Callable[..., dict[str, Any] | None] | None = None,
) -> FundamentalEvidence | None:
    """把任意现货/基差来源适配成标准化基本面指标域。"""

    if spot_basis_fetcher is None:
        from data_cache import get_spot_basis as spot_basis_fetcher

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = spot_basis_fetcher(symbol, as_of_date=as_of_date)
    if not _has_data(payload):
        return None

    value_keys = ("spot_price", "dominant_contract_price", "basis", "basis_rate")
    values = {key: payload.get(key) for key in value_keys}
    if any(value is None for value in values.values()):
        return None

    metadata = {
        key: payload.get(key)
        for key in ("commodity_code", "data_date", "source")
        if payload.get(key) is not None
    }
    return FundamentalEvidence(
        domain=DOMAIN_SPOT_BASIS,
        values=values,
        metadata=metadata,
    )
