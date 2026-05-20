"""
Phase 3 盘中监控 — TqSdk 实时 K 线
=================================

使用 TqSdk 连续合约 `KQ.m@交易所.品种` 订阅分钟/日线序列，数据在 wait_update 循环中就地刷新。
观望列表的「日线反转检测」直接使用 TqSdk 日线序列（含当日未收盘 K 线），不再用分钟线拼日线。

依赖: pip install tqsdk
配置: config.yaml 中 tqsdk.account / tqsdk.password（快期账户）
"""

from __future__ import annotations

import contextlib
import io
import time
import pandas as pd

from data_cache import BUILTIN_SYMBOLS
from market.tq import resolve_tq_continuous_symbol_info


def exchange_for_symbol(symbol: str) -> str | None:
    for s in BUILTIN_SYMBOLS:
        if s["symbol"] == symbol:
            return s["exchange"]
    return None


def internal_symbol_to_tq_continuous(symbol: str, api=None) -> tuple[str, dict[str, str]]:
    """
    内部代码（如 LH0）→ TqSdk 连续合约代码 KQ.m@DCE.lh
    """
    ex = exchange_for_symbol(symbol)
    if not ex:
        raise ValueError(f"未知品种代码（未在内置列表）: {symbol}")
    try:
        info = resolve_tq_continuous_symbol_info(symbol.strip(), ex, api=api)
        return info["continuous_symbol"], info
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"未知交易所: {ex}") from exc


def _normalize_kline_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt


def klines_to_minute_df(klines: pd.DataFrame) -> pd.DataFrame | None:
    """TqSdk 分钟 K 线 → intraday.print_dashboard 所需列"""
    if klines is None or len(klines) < 1:
        return None
    df = klines.copy()
    df["datetime"] = _normalize_kline_datetime(df["datetime"])
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "close_oi" in df.columns:
        df["hold"] = pd.to_numeric(df["close_oi"], errors="coerce").fillna(0.0)
    else:
        df["hold"] = 0.0
    out = df[["datetime", "open", "high", "low", "close", "volume", "hold"]].dropna(
        subset=["close"]
    )
    return out.sort_values("datetime").reset_index(drop=True) if len(out) else None


def klines_to_daily_df(klines: pd.DataFrame) -> pd.DataFrame | None:
    """
    TqSdk 日线序列 → assess_reversal_status 所需 DataFrame（date + OHLCV + oi）
    最后一根为当前交易日（盘中为未完成 K 线）。
    """
    if klines is None or len(klines) < 2:
        return None
    df = klines.copy()
    dt = _normalize_kline_datetime(df["datetime"])
    out = pd.DataFrame(
        {
            "date": dt.dt.normalize(),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    if "close_oi" in df.columns:
        out["oi"] = pd.to_numeric(df["close_oi"], errors="coerce").fillna(0.0)
    else:
        out["oi"] = 0.0
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out if len(out) >= 30 else None


class TqPhase3Monitor:
    """
    单连接订阅多品种：分钟 K + 日线 K（连续合约）。
    须在交易时段外也能创建（仅 warmup 可能数据为空）；盘中 wait_update 后序列更新。
    """

    def __init__(
        self,
        account: str,
        password: str,
        symbols: list[str],
        period: str = "5",
        minute_bars: int = 800,
        daily_bars: int = 500,
    ):
        from tqsdk import TqApi, TqAuth

        self._TqApi = TqApi
        self._TqAuth = TqAuth
        self.account = account
        self.password = password
        self.symbols = list(dict.fromkeys(symbols))
        self.period_sec = int(period) * 60
        self.minute_bars = minute_bars
        self.daily_bars = daily_bars
        self.api: object | None = None
        self._min_serial: dict[str, object] = {}
        self._day_serial: dict[str, object] = {}
        self._tq_syms: dict[str, str] = {}
        self._resolve_info: dict[str, dict[str, str]] = {}

    def connect(self) -> None:
        self.api = self._TqApi(auth=self._TqAuth(self.account, self.password))
        for sym in self.symbols:
            tq_symbol, info = internal_symbol_to_tq_continuous(sym, api=self.api)
            self._tq_syms[sym] = tq_symbol
            self._resolve_info[sym] = info
        for sym in self.symbols:
            tq = self._tq_syms[sym]
            self._min_serial[sym] = self.api.get_kline_serial(
                tq, self.period_sec, data_length=self.minute_bars
            )
            self._day_serial[sym] = self.api.get_kline_serial(tq, 86400, data_length=self.daily_bars)

    def warmup(self, timeout: float = 45.0) -> None:
        """阻塞直至各序列至少有数据或超时。"""
        if self.api is None:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.api.wait_update(deadline=deadline)
            ok = True
            for sym in self.symbols:
                kl = self._min_serial.get(sym)
                if kl is None or len(kl) < 1:
                    ok = False
                    break
            if ok:
                return

    def wait_interval(self, seconds: float) -> None:
        """用 wait_update(deadline) 替代 sleep，保持行情推进。"""
        if self.api is None:
            time.sleep(seconds)
            return
        dl = time.time() + max(0.5, seconds)
        while time.time() < dl:
            updated = self.api.wait_update(deadline=dl)
            if not updated:
                break

    def minute_df(self, symbol: str) -> pd.DataFrame | None:
        kl = self._min_serial.get(symbol)
        if kl is None:
            return None
        return klines_to_minute_df(kl)

    def daily_df(self, symbol: str) -> pd.DataFrame | None:
        kl = self._day_serial.get(symbol)
        if kl is None:
            return None
        return klines_to_daily_df(kl)

    def close(self) -> None:
        """
        关闭 TqSdk。收盘或 Ctrl+C 退出时调用；close() 内部会取消大量 asyncio task，
        可能向 stderr 打出「Event loop is closed」等收尾信息，属库侧已知噪音，此处先
        wait_update 再关闭并暂时屏蔽 stderr，避免误以为是业务报错。
        """
        if self.api is None:
            return
        api = self.api
        self.api = None
        try:
            try:
                api.wait_update(deadline=time.time() + 1.5)
            except Exception:
                pass
            with contextlib.redirect_stderr(io.StringIO()):
                api.close()
        except Exception:
            try:
                if getattr(api, "_loop", None) and not api._loop.is_closed():
                    with contextlib.redirect_stderr(io.StringIO()):
                        api.close()
            except Exception:
                pass

    def describe_subscription(self) -> str:
        parts = []
        for sym in self.symbols:
            tq_symbol = self._tq_syms.get(sym, "?")
            info = self._resolve_info.get(sym) or {}
            source = info.get("source") or "fallback"
            underlying = info.get("underlying_symbol") or ""
            if source == "query_cont_quotes" and underlying:
                parts.append(f"{sym}→{tq_symbol} [query_cont_quotes:{underlying}]")
            elif source == "underlying_symbol" and underlying:
                parts.append(f"{sym}→{tq_symbol} [underlying_symbol:{underlying}]")
            else:
                parts.append(f"{sym}→{tq_symbol} [fallback]")
        return "; ".join(parts)


def tqsdk_configured(config: dict) -> bool:
    tq = config.get("tqsdk") or {}
    acc = (tq.get("account") or "").strip()
    pwd = (tq.get("password") or "").strip()
    if tq.get("phase3_enabled") is False:
        return False
    return bool(acc and pwd)


def try_create_phase3_monitor(
    config: dict,
    targets: list[dict],
    watchlist: list[dict] | None,
    period: str,
) -> TqPhase3Monitor | None:
    """
    若配置与依赖就绪则创建并 connect+warmup；失败返回 None。
    """
    if not tqsdk_configured(config):
        return None
    try:
        import tqsdk  # noqa: F401
    except ImportError:
        return None

    tq = config.get("tqsdk") or {}
    syms: list[str] = []
    for t in targets:
        syms.append(t["symbol"])
    for w in watchlist or []:
        syms.append(w["symbol"])
    if not syms:
        return None

    mon = TqPhase3Monitor(
        account=str(tq["account"]).strip(),
        password=str(tq["password"]).strip(),
        symbols=syms,
        period=period,
    )
    mon.connect()
    mon.warmup()
    return mon
