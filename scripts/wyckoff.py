"""
威科夫量价分析 + VSA + 持仓量分析
========================================

核心理论:

1. Wyckoff 市场阶段 (Market Phases):
   - Accumulation (吸筹): 主力在低位吸纳筹码，价格停止下跌，形成横盘区间
   - Markup (上涨): 吸筹完成后的上涨阶段
   - Distribution (派发): 主力在高位派发筹码，价格停止上涨，形成横盘区间
   - Markdown (下跌): 派发完成后的下跌阶段

2. Wyckoff 事件检测:
   - Selling Climax (SC/卖方高潮): 暴跌+天量，标志下跌可能结束
   - Buying Climax (BC/买方高潮): 暴涨+天量，标志上涨可能结束
   - Spring (弹簧): 价格短暂跌破支撑后快速收回，吸筹确认信号
   - Upthrust (上冲回落): 价格短暂突破阻力后快速回落，派发确认信号
   - Sign of Strength (SOS/强势信号): 放量突破交易区间上沿
   - Sign of Weakness (SOW/弱势信号): 放量跌破交易区间下沿

3. VSA (Volume Spread Analysis) K线分类:
   - Stopping Volume (停止量): 高量+窄幅+收盘靠近极端 → 趋势可能反转
   - No Demand (无需求): 窄幅上涨+缩量 → 上涨无力
   - No Supply (无供给): 窄幅下跌+缩量 → 下跌无力
   - Effort vs Result (努力vs结果): 量与价幅是否匹配

4. 期货持仓量(OI)分析:
   - 价涨+OI涨 = 新多入场 (强势看多)
   - 价涨+OI跌 = 空头回补 (弱势看多)
   - 价跌+OI涨 = 新空入场 (强势看空)
   - 价跌+OI跌 = 多头平仓 (弱势看空)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================
#  1. 基础量价计算
# ============================================================

def spread(df: pd.DataFrame) -> pd.Series:
    """K线实体幅度 (high - low)"""
    return df["high"] - df["low"]


def body(df: pd.DataFrame) -> pd.Series:
    """K线实体 (close - open)"""
    return df["close"] - df["open"]


def close_position(df: pd.DataFrame) -> pd.Series:
    """收盘价在K线中的相对位置 0=最低 1=最高"""
    sp = spread(df)
    return (df["close"] - df["low"]) / (sp + 1e-12)


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """相对成交量: 当前量 / N日均量"""
    vol_ma = df["volume"].rolling(window).mean()
    return df["volume"] / (vol_ma + 1e-12)


def relative_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """相对价幅: 当前价幅 / N日均价幅"""
    sp = spread(df)
    sp_ma = sp.rolling(window).mean()
    return sp / (sp_ma + 1e-12)


def effort_vs_result(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    努力vs结果指标
    = 价格变化率 / 成交量变化率
    >1 表示少量努力产生大结果（趋势健康）
    <1 表示大量努力只有小结果（趋势疲弱/有反向力量）
    """
    price_change = df["close"].pct_change(window).abs()
    vol_change = relative_volume(df, window)
    return price_change / (vol_change * df["close"].pct_change(1).abs().rolling(window).mean() + 1e-12)


# ============================================================
#  2. VSA K线分类
# ============================================================

@dataclass
class VSABar:
    """单根K线的VSA分析结果"""
    bar_type: str       # 类型名称
    bias: str           # bullish / bearish / neutral
    strength: int       # 1-3 信号强度
    description: str    # 中文描述


def classify_vsa_bar(
    o: float, h: float, l: float, c: float, vol: float,
    avg_vol: float, avg_spread: float,
    prev_c: float,
) -> VSABar:
    """
    对单根K线做VSA分类

    基于 Tom Williams 的 VSA 理论:
    - 价幅(spread)、收盘位置(close position)、成交量(volume)三要素
    """
    sp = h - l
    cp = (c - l) / (sp + 1e-12)  # 0~1
    rel_vol = vol / (avg_vol + 1e-12)
    rel_sp = sp / (avg_spread + 1e-12)
    is_up = c > prev_c
    is_down = c < prev_c

    # --- 高量信号 ---
    if rel_vol > 2.0:
        # 卖方高潮: 大跌+天量+收盘偏高(有承接)
        if is_down and rel_sp > 1.5 and cp > 0.4:
            return VSABar("卖方高潮", "bullish", 3,
                          f"暴跌天量(量比{rel_vol:.1f}), 但收盘被拉回({cp:.0%}位), 下方有强力承接")
        # 买方高潮: 大涨+天量+收盘偏低(有抛压)
        if is_up and rel_sp > 1.5 and cp < 0.6:
            return VSABar("买方高潮", "bearish", 3,
                          f"暴涨天量(量比{rel_vol:.1f}), 但收盘回落({cp:.0%}位), 上方有强力抛压")
        # 停止量(下跌中): 高量+收盘偏高
        if is_down and cp > 0.5:
            return VSABar("停止量", "bullish", 2,
                          f"下跌放量(量比{rel_vol:.1f}), 收盘在上半部({cp:.0%}), 卖盘被吸收")
        # 停止量(上涨中): 高量+收盘偏低
        if is_up and cp < 0.5:
            return VSABar("停止量", "bearish", 2,
                          f"上涨放量(量比{rel_vol:.1f}), 收盘在下半部({cp:.0%}), 买盘被消化")

    # --- 高量+窄幅: 努力无结果 ---
    if rel_vol > 1.5 and rel_sp < 0.7:
        if is_up:
            return VSABar("上涨受阻", "bearish", 2,
                          f"放量(量比{rel_vol:.1f})但价幅很窄(幅比{rel_sp:.1f}), 上方压力大")
        else:
            return VSABar("下跌受托", "bullish", 2,
                          f"放量(量比{rel_vol:.1f})但价幅很窄(幅比{rel_sp:.1f}), 下方支撑强")

    # --- 低量信号 ---
    if rel_vol < 0.6:
        # 无需求: 缩量上涨+窄幅
        if is_up and rel_sp < 0.8:
            return VSABar("无需求", "bearish", 1,
                          f"缩量上涨(量比{rel_vol:.1f}, 幅比{rel_sp:.1f}), 多方无力推升")
        # 无供给: 缩量下跌+窄幅
        if is_down and rel_sp < 0.8:
            return VSABar("无供给", "bullish", 1,
                          f"缩量下跌(量比{rel_vol:.1f}, 幅比{rel_sp:.1f}), 空方抛压枯竭")

    # --- 中等量+宽幅: 健康趋势 ---
    if rel_vol > 0.8 and rel_sp > 1.2:
        if is_up and cp > 0.6:
            return VSABar("健康上涨", "bullish", 1,
                          f"放量大阳(收盘{cp:.0%}位), 趋势健康")
        if is_down and cp < 0.4:
            return VSABar("健康下跌", "bearish", 1,
                          f"放量大阴(收盘{cp:.0%}位), 趋势健康")

    return VSABar("中性", "neutral", 0, f"量比{rel_vol:.1f}, 幅比{rel_sp:.1f}, 无明显信号")


def vsa_scan(df: pd.DataFrame, window: int = 20) -> list[VSABar]:
    """对整个DataFrame做VSA扫描，返回每根K线的分类"""
    avg_vol = df["volume"].rolling(window).mean()
    avg_sp = spread(df).rolling(window).mean()

    results = []
    for i in range(len(df)):
        if i < window or pd.isna(avg_vol.iloc[i]):
            results.append(VSABar("数据不足", "neutral", 0, ""))
            continue
        results.append(classify_vsa_bar(
            df["open"].iloc[i], df["high"].iloc[i],
            df["low"].iloc[i], df["close"].iloc[i],
            df["volume"].iloc[i], avg_vol.iloc[i], avg_sp.iloc[i],
            df["close"].iloc[i - 1],
        ))
    return results


# ============================================================
#  3. Wyckoff 阶段分析 (日线级别)
# ============================================================

@dataclass
class WyckoffPhase:
    phase: str             # accumulation / distribution / markup / markdown
    confidence: float      # 0~1
    events: list[str] = field(default_factory=list)
    description: str = ""


def detect_trading_range(df: pd.DataFrame, lookback: int = 60, threshold: float = 0.08) -> dict | None:
    """
    检测是否处于横盘交易区间 (Trading Range)

    如果近 lookback 日的价格波动幅度 < threshold (8%),
    则认为处于 TR 中。
    """
    recent = df.tail(lookback)
    high = recent["high"].max()
    low = recent["low"].min()
    mid = (high + low) / 2
    range_pct = (high - low) / (mid + 1e-12)

    if range_pct < threshold:
        return {
            "in_range": True,
            "high": high,
            "low": low,
            "range_pct": range_pct,
            "bars": lookback,
        }
    return None


def analyze_volume_pattern(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    分析量价模式：
    - 上涨日的平均成交量 vs 下跌日的平均成交量
    - 量价是否同步
    """
    recent = df.tail(lookback).copy()
    recent["ret"] = recent["close"].pct_change()
    recent["is_up"] = recent["ret"] > 0

    up_bars = recent[recent["is_up"]]
    down_bars = recent[~recent["is_up"]]

    avg_up_vol = up_bars["volume"].mean() if len(up_bars) > 0 else 0
    avg_down_vol = down_bars["volume"].mean() if len(down_bars) > 0 else 0

    # 上涨量/下跌量比值 > 1 = 多方主导
    up_down_ratio = avg_up_vol / (avg_down_vol + 1e-12)

    # 近期量能趋势：最近20根 vs 之前20根
    if lookback >= 40:
        vol_recent = recent["volume"].tail(20).mean()
        vol_earlier = recent["volume"].iloc[-40:-20].mean()
        vol_trend = vol_recent / (vol_earlier + 1e-12)
    else:
        vol_trend = 1.0

    # 量价相关性（价格变化和成交量的相关系数）
    price_vol_corr = recent["ret"].corr(recent["volume"])

    return {
        "avg_up_vol": avg_up_vol,
        "avg_down_vol": avg_down_vol,
        "up_down_ratio": up_down_ratio,
        "vol_trend": vol_trend,
        "price_vol_corr": price_vol_corr if not np.isnan(price_vol_corr) else 0,
    }


def detect_climax(df: pd.DataFrame, lookback: int = 120) -> list[dict]:
    """检测卖方/买方高潮事件"""
    events = []
    recent = df.tail(lookback)
    vol_ma = recent["volume"].rolling(20).mean()
    sp = spread(recent)
    sp_ma = sp.rolling(20).mean()

    for i in range(20, len(recent)):
        idx = recent.index[i]
        row = recent.iloc[i]
        prev = recent.iloc[i - 1]
        rv = row["volume"] / (vol_ma.iloc[i] + 1e-12)
        rs = sp.iloc[i] / (sp_ma.iloc[i] + 1e-12)
        cp = (row["close"] - row["low"]) / (sp.iloc[i] + 1e-12)

        if rv < 2.5 or rs < 1.5:
            continue

        date_str = str(recent.iloc[i].get("date", idx))[:10]

        if row["close"] < prev["close"] and cp > 0.4:
            events.append({
                "type": "卖方高潮(SC)",
                "date": date_str,
                "bias": "bullish",
                "detail": f"暴跌天量(量比{rv:.1f}), 收盘被承接回{cp:.0%}位",
            })
        elif row["close"] > prev["close"] and cp < 0.6:
            events.append({
                "type": "买方高潮(BC)",
                "date": date_str,
                "bias": "bearish",
                "detail": f"暴涨天量(量比{rv:.1f}), 收盘回落至{cp:.0%}位",
            })

    return events


def detect_spring_upthrust(df: pd.DataFrame, lookback: int = 60) -> list[dict]:
    """
    检测弹簧(Spring)和上冲回落(Upthrust)

    Spring: 价格跌破近期低点但当日收回 → 假突破做多信号
    Upthrust: 价格涨破近期高点但当日收回 → 假突破做空信号
    """
    events = []
    recent = df.tail(lookback)

    for i in range(20, len(recent)):
        row = recent.iloc[i]
        date_str = str(recent.iloc[i].get("date", recent.index[i]))[:10]

        prev_low = recent["low"].iloc[i - 20:i].min()
        prev_high = recent["high"].iloc[i - 20:i].max()

        # Spring: 当日最低跌破前低，但收盘收回到前低之上
        if row["low"] < prev_low and row["close"] > prev_low:
            depth = (prev_low - row["low"]) / (row["close"] + 1e-12) * 100
            if depth > 0.2:
                events.append({
                    "type": "弹簧(Spring)",
                    "date": date_str,
                    "bias": "bullish",
                    "detail": f"跌破前低{prev_low:.0f}后收回, 探底{depth:.1f}%, 假突破洗盘",
                })

        # Upthrust: 当日最高突破前高，但收盘回落到前高之下
        if row["high"] > prev_high and row["close"] < prev_high:
            depth = (row["high"] - prev_high) / (row["close"] + 1e-12) * 100
            if depth > 0.2:
                events.append({
                    "type": "上冲回落(UT)",
                    "date": date_str,
                    "bias": "bearish",
                    "detail": f"突破前高{prev_high:.0f}后回落, 冲高{depth:.1f}%, 假突破出货",
                })

    return events


def detect_sos_sow(df: pd.DataFrame, lookback: int = 60) -> list[dict]:
    """
    检测强势信号(SOS)和弱势信号(SOW)

    SOS: 放量突破近期高点，收盘在高点之上 → 吸筹完成开始上涨
    SOW: 放量跌破近期低点，收盘在低点之下 → 派发完成开始下跌
    """
    events = []
    recent = df.tail(lookback)
    vol_ma = recent["volume"].rolling(20).mean()

    for i in range(20, len(recent)):
        row = recent.iloc[i]
        date_str = str(recent.iloc[i].get("date", recent.index[i]))[:10]
        rv = row["volume"] / (vol_ma.iloc[i] + 1e-12)

        if rv < 1.3:
            continue

        prev_high = recent["high"].iloc[i - 20:i].max()
        prev_low = recent["low"].iloc[i - 20:i].min()

        if row["close"] > prev_high and rv > 1.5:
            events.append({
                "type": "强势信号(SOS)",
                "date": date_str,
                "bias": "bullish",
                "detail": f"放量(量比{rv:.1f})突破前高{prev_high:.0f}, 吸筹结束标志",
            })

        if row["close"] < prev_low and rv > 1.5:
            events.append({
                "type": "弱势信号(SOW)",
                "date": date_str,
                "bias": "bearish",
                "detail": f"放量(量比{rv:.1f})跌破前低{prev_low:.0f}, 派发结束标志",
            })

    return events


def wyckoff_phase(df: pd.DataFrame, lookback: int = 120) -> WyckoffPhase:
    """
    综合判断当前所处的 Wyckoff 阶段

    逻辑:
    1. 先看近60日是否在横盘区间(TR)
    2. 分析上涨日量/下跌日量的比值
    3. 检测关键事件(SC/BC/Spring/UT/SOS/SOW)
    4. 综合判断阶段
    """
    recent = df.tail(lookback)
    close = recent["close"]
    last = float(close.iloc[-1])

    # 趋势方向
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    trend_up = ma20 > ma60

    # 价格位置
    high_all = float(close.max())
    low_all = float(close.min())
    range_pos = (last - low_all) / (high_all - low_all + 1e-12)

    # 交易区间检测
    tr = detect_trading_range(df, lookback=60)

    # 量价分析
    vp = analyze_volume_pattern(df, lookback=60)

    # 事件检测
    climax_events = detect_climax(df, lookback)
    spring_events = detect_spring_upthrust(df, lookback=60)
    sos_sow_events = detect_sos_sow(df, lookback=60)
    all_events = climax_events + spring_events + sos_sow_events

    event_names = [e["type"] for e in all_events[-5:]]  # 只取最近5个

    bullish_events = sum(1 for e in all_events if e["bias"] == "bullish")
    bearish_events = sum(1 for e in all_events if e["bias"] == "bearish")

    # --- 阶段判断 ---
    if tr:
        # 在横盘区间内
        if range_pos < 0.35:
            # 价格在低位横盘
            if vp["up_down_ratio"] > 1.1:
                confidence = min(0.5 + bullish_events * 0.1, 0.95)
                return WyckoffPhase(
                    "accumulation", confidence, event_names,
                    f"低位横盘(区间{tr['range_pct']:.1%}), 上涨日成交量>下跌日({vp['up_down_ratio']:.2f}), "
                    f"主力可能在吸筹"
                )
            else:
                return WyckoffPhase(
                    "accumulation", 0.3, event_names,
                    f"低位横盘(区间{tr['range_pct']:.1%}), 但量价关系不明确, 等待确认"
                )
        elif range_pos > 0.65:
            # 价格在高位横盘
            if vp["up_down_ratio"] < 0.9:
                confidence = min(0.5 + bearish_events * 0.1, 0.95)
                return WyckoffPhase(
                    "distribution", confidence, event_names,
                    f"高位横盘(区间{tr['range_pct']:.1%}), 下跌日成交量>上涨日({1/vp['up_down_ratio']:.2f}), "
                    f"主力可能在派发"
                )
            else:
                return WyckoffPhase(
                    "distribution", 0.3, event_names,
                    f"高位横盘(区间{tr['range_pct']:.1%}), 但量价关系不明确, 等待确认"
                )
        else:
            return WyckoffPhase(
                "accumulation" if bullish_events > bearish_events else "distribution",
                0.2, event_names,
                f"中位横盘, 方向不明"
            )
    else:
        # 不在横盘区间 → 判断 markup 或 markdown
        # 综合三个维度: MA交叉(滞后)、价格动量(实时)、成交量趋势
        price_below_ma20 = last < ma20
        ret_10d = last / float(close.iloc[-11]) - 1 if len(close) > 11 else 0
        ret_20d = last / float(close.iloc[-21]) - 1 if len(close) > 21 else 0
        vol_declining = vp["vol_trend"] < 0.85

        # MA交叉说上涨，但价格已跌破MA20且近期动量为负 → 趋势已转
        if trend_up and price_below_ma20 and ret_10d < -0.03:
            phase = "markdown"
            desc = (f"MA20({ma20:.0f})仍>MA60({ma60:.0f})但价格({last:.0f})"
                    f"已跌破MA20, 10日跌幅{ret_10d:.1%}, 趋势可能反转")
            confidence = 0.5 if ret_20d < -0.05 else 0.3
            return WyckoffPhase(phase, confidence, event_names, desc)

        # MA交叉说下跌，但价格已站上MA20且近期动量为正 → 趋势已转
        if not trend_up and not price_below_ma20 and ret_10d > 0.03:
            phase = "markup"
            desc = (f"MA20({ma20:.0f})仍<MA60({ma60:.0f})但价格({last:.0f})"
                    f"已站上MA20, 10日涨幅{ret_10d:.1%}, 趋势可能反转")
            confidence = 0.5 if ret_20d > 0.05 else 0.3
            return WyckoffPhase(phase, confidence, event_names, desc)

        if trend_up:
            healthy = vp["up_down_ratio"] > 1.0 and not vol_declining
            confidence = 0.7 if healthy else 0.4
            vol_desc = "量价同步" if healthy else ("缩量上涨(动力不足)" if vol_declining else "量价背离(注意风险)")
            return WyckoffPhase(
                "markup", confidence, event_names,
                f"上涨趋势, 价格({last:.0f})>MA20({ma20:.0f})>MA60({ma60:.0f}), {vol_desc}"
            )
        else:
            healthy = vp["up_down_ratio"] < 1.0 and not vol_declining
            confidence = 0.7 if healthy else 0.4
            vol_desc = "量价同步" if healthy else ("缩量下跌(抛压减弱)" if vol_declining else "量价背离(可能见底)")
            return WyckoffPhase(
                "markdown", confidence, event_names,
                f"下跌趋势, 价格({last:.0f})<MA20({ma20:.0f}), MA60({ma60:.0f}), {vol_desc}"
            )


# ============================================================
#  4. 期货持仓量(OI)分析
# ============================================================

@dataclass
class OISignal:
    pattern: str      # new_long / short_covering / new_short / long_liquidation
    label: str        # 中文
    bias: str         # bullish / bearish
    strength: str     # 强势 / 弱势
    description: str


def analyze_oi(df: pd.DataFrame, window: int = 5) -> OISignal | None:
    """
    分析价格变化与持仓量变化的关系

    这是期货特有的分析维度:
    - 持仓量增加 = 新资金入场（做多或做空）
    - 持仓量减少 = 老仓位离场（平多或平空）
    """
    if "oi" not in df.columns:
        return None

    oi = pd.to_numeric(df["oi"], errors="coerce")
    close = df["close"]

    if oi.isna().all() or len(df) < window + 1:
        return None

    price_change = float(close.iloc[-1]) - float(close.iloc[-window - 1])
    oi_change = float(oi.iloc[-1]) - float(oi.iloc[-window - 1])
    price_pct = price_change / (float(close.iloc[-window - 1]) + 1e-12) * 100
    oi_pct = oi_change / (float(oi.iloc[-window - 1]) + 1e-12) * 100

    if price_change > 0 and oi_change > 0:
        return OISignal(
            "new_long", "新多入场", "bullish", "强势",
            f"价格涨{price_pct:+.1f}% + 持仓增{oi_pct:+.1f}%, 新资金做多入场, 上涨有持续动力"
        )
    elif price_change > 0 and oi_change < 0:
        return OISignal(
            "short_covering", "空头回补", "bullish", "弱势",
            f"价格涨{price_pct:+.1f}% + 持仓减{oi_pct:+.1f}%, 空头平仓推动上涨, 动力有限"
        )
    elif price_change < 0 and oi_change > 0:
        return OISignal(
            "new_short", "新空入场", "bearish", "强势",
            f"价格跌{price_pct:+.1f}% + 持仓增{oi_pct:+.1f}%, 新资金做空入场, 下跌有持续动力"
        )
    elif price_change < 0 and oi_change < 0:
        return OISignal(
            "long_liquidation", "多头平仓", "bearish", "弱势",
            f"价格跌{price_pct:+.1f}% + 持仓减{oi_pct:+.1f}%, 多头止损离场, 恐慌性下跌可能见底"
        )
    return None


def oi_divergence(df: pd.DataFrame, window: int = 20) -> str | None:
    """
    检测OI背离:
    - 价格创新高但OI下降 → 顶部背离
    - 价格创新低但OI下降 → 底部背离
    """
    if "oi" not in df.columns:
        return None

    oi = pd.to_numeric(df["oi"], errors="coerce")
    close = df["close"]
    recent = df.tail(window)

    if oi.isna().all():
        return None

    price_high = float(close.tail(window).max())
    price_low = float(close.tail(window).min())
    last_price = float(close.iloc[-1])
    prev_high = float(close.iloc[-window * 2:-window].max()) if len(close) > window * 2 else price_high
    prev_low = float(close.iloc[-window * 2:-window].min()) if len(close) > window * 2 else price_low

    oi_now = float(oi.iloc[-1])
    oi_prev = float(oi.iloc[-window]) if len(oi) > window else oi_now

    if last_price >= prev_high * 0.99 and oi_now < oi_prev * 0.95:
        return f"⚠️ 顶部背离: 价格接近前高但持仓量下降{(oi_now/oi_prev-1)*100:+.1f}%, 多头可能在撤退"
    if last_price <= prev_low * 1.01 and oi_now < oi_prev * 0.95:
        return f"💡 底部背离: 价格接近前低但持仓量下降{(oi_now/oi_prev-1)*100:+.1f}%, 空头可能在撤退"

    return None


# ============================================================
#  5. 综合评分 (供 screener 使用)
# ============================================================

def wyckoff_score(df: pd.DataFrame) -> dict:
    """
    输出一个综合的 Wyckoff 量价评分字典

    返回:
        phase: 当前阶段
        vsa_bias: 最近VSA信号偏向
        oi_signal: 持仓信号
        composite: 综合分 (-100 ~ +100, 负=做多, 正=做空)
    """
    result = {"phase": "", "vsa_bias": "", "oi_signal": "", "composite": 0.0}

    # 阶段
    phase = wyckoff_phase(df, lookback=120)
    result["phase"] = phase.phase
    result["phase_confidence"] = phase.confidence
    result["phase_desc"] = phase.description

    phase_score = 0.0
    if phase.phase == "accumulation":
        phase_score = -20 * phase.confidence
    elif phase.phase == "distribution":
        phase_score = 20 * phase.confidence
    elif phase.phase == "markup":
        phase_score = -10 * phase.confidence
    elif phase.phase == "markdown":
        phase_score = 10 * phase.confidence

    # VSA 最近5根K线
    vsa_bars = vsa_scan(df, window=20)
    recent_vsa = vsa_bars[-5:] if len(vsa_bars) >= 5 else vsa_bars
    bullish_vsa = sum(1 for b in recent_vsa if b.bias == "bullish")
    bearish_vsa = sum(1 for b in recent_vsa if b.bias == "bearish")

    if bullish_vsa > bearish_vsa:
        result["vsa_bias"] = "bullish"
    elif bearish_vsa > bullish_vsa:
        result["vsa_bias"] = "bearish"
    else:
        result["vsa_bias"] = "neutral"

    vsa_score = (bearish_vsa - bullish_vsa) * 5

    # OI
    oi_sig = analyze_oi(df, window=5)
    oi_score = 0.0
    if oi_sig:
        result["oi_signal"] = oi_sig.label
        if oi_sig.bias == "bearish":
            oi_score = 10 if oi_sig.strength == "强势" else 5
        else:
            oi_score = -10 if oi_sig.strength == "强势" else -5

    # 量价模式
    vp = analyze_volume_pattern(df, lookback=60)
    vp_score = 0.0
    if vp["up_down_ratio"] > 1.3:
        vp_score = -10
    elif vp["up_down_ratio"] < 0.7:
        vp_score = 10

    result["composite"] = phase_score + vsa_score + oi_score + vp_score
    return result
