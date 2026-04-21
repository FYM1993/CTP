from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase3 import live as tqsdk_live  # noqa: E402


def test_wait_interval_keeps_waiting_until_deadline(monkeypatch):
    fake_tqsdk = ModuleType("tqsdk")

    class FakeTqAuth:
        def __init__(self, account, password):
            self.account = account
            self.password = password

    class FakeTqApi:
        def __init__(self, auth=None):
            self.auth = auth

    fake_tqsdk.TqAuth = FakeTqAuth
    fake_tqsdk.TqApi = FakeTqApi
    monkeypatch.setitem(sys.modules, "tqsdk", fake_tqsdk)

    current = {"value": 100.0}
    deadlines = []

    class SessionApi:
        def wait_update(self, deadline=None):
            deadlines.append(deadline)
            current["value"] += 0.2
            return True

    monkeypatch.setattr(tqsdk_live.time, "time", lambda: current["value"])

    mon = tqsdk_live.TqPhase3Monitor("acct", "pwd", ["LH0"])
    mon.api = SessionApi()
    mon.wait_interval(1.0)

    assert len(deadlines) >= 4
    assert all(deadline == 101.0 for deadline in deadlines)


def test_describe_subscription_includes_resolution_source(monkeypatch):
    fake_tqsdk = ModuleType("tqsdk")

    class FakeTqAuth:
        def __init__(self, account, password):
            self.account = account
            self.password = password

    class FakeTqApi:
        def __init__(self, auth=None):
            self.auth = auth

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            return []

    fake_tqsdk.TqAuth = FakeTqAuth
    fake_tqsdk.TqApi = FakeTqApi
    monkeypatch.setitem(sys.modules, "tqsdk", fake_tqsdk)
    monkeypatch.setattr(
        tqsdk_live,
        "internal_symbol_to_tq_continuous",
        lambda symbol, api=None: (
            "KQ.m@DCE.lh",
            {
                "continuous_symbol": "KQ.m@DCE.lh",
                "source": "query_cont_quotes",
                "underlying_symbol": "DCE.lh2509",
            },
        ),
    )

    mon = tqsdk_live.TqPhase3Monitor("acct", "pwd", ["LH0"])
    mon.connect()

    assert mon.describe_subscription() == "LH0→KQ.m@DCE.lh [query_cont_quotes:DCE.lh2509]"
