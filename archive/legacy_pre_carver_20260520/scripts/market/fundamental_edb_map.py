from __future__ import annotations

"""基本面 EDB 映射注册表。

当前只把官方文档已经确认的库存映射写死，其余字段保持空映射，
运行时统一回退 AkShare。
"""

EDB_FIELD_MAP: dict[str, dict[str, int]] = {
    "库存": {
        "CU0": 1,
        "AG0": 2,
    },
    "仓单": {},
    "利润": {},
    "成本": {},
    "现货": {},
    "生猪": {},
}
