"""
Phase 1 基本面评分 — Regime 引擎（实验）

- 默认仅对 LH0 使用分体制生猪专项；其余品种与 legacy 相同。
- 设计目标：在深亏/去产能阶段（distress）不把「短期现货反弹」打成利空。

启用：config.yaml → fundamental_screening.p1_engine: regime
"""

from __future__ import annotations

import pandas as pd

from data_cache import get_hog_fundamentals

from fundamental_legacy import compute_range_pct, compute_supply_layers, score_symbol_legacy


def classify_lh_regime(
    profit_margin: float | None,
    margin_missing: bool,
    range_pct: float,
) -> str:
    """
    生猪周期体制（粗分，可后续数据校准）:

    - distress: 深亏或数据缺失且期货仍在年度区间低位 → 更关注去产能/磨底
    - repair: 亏损收窄、接近盈亏平衡前
    - neutral: 微利震荡
    - boom: 高利润、扩产风险

    Parameters
    ----------
    profit_margin
        养殖利润率估算（%），来自卓创系数据在 ``data_cache.get_hog_fundamentals`` 里
        用 ``(猪价×120 / 成本 − 1) × 100`` 近似；**有有效数值时**与 ``margin_missing`` 互斥，
        用于按盈亏阶段分桶（如 pm < -8 → distress）。
    margin_missing
        为 True 表示**无法得到可靠利润率**（接口失败、或 ``profit_margin`` 键缺失/为 None）。
        此时**不看** ``profit_margin`` 数值，改用 ``range_pct`` 推断体制（期货仍在低位则偏 distress）。
    range_pct
        **期货**主力连续合约收盘价，在「近 min(300, 可用长度) 日」最高价与最低价之间的位置（0～100%），
        与 ``fundamental_legacy.compute_range_pct`` 一致。用于：① 利润缺失时的体制代理；
        ② ``_lh_regime_hog_layers`` 里利润缺失时的区间补偿。注意这是**期货年度区间位**，
        不是现货猪价的绝对水平。
    """
    if margin_missing:
        if range_pct < 18:
            return "distress"
        if range_pct < 45:
            return "repair"
        return "neutral"

    assert profit_margin is not None
    pm = profit_margin
    if pm < -8:
        return "distress"
    if pm < 3:
        return "repair"
    if pm < 12:
        return "neutral"
    return "boom"


def _lh_regime_hog_layers(
    regime: str,
    hog: dict | None,
    range_pct: float,
) -> tuple[float, list[str], float | None]:
    """
    生猪专项分数 + 说明；与 legacy 解耦的体制化版本。
    """
    score = 0.0
    details: list[str] = []
    hog_profit: float | None = None

    pm: float | None = None
    pm_missing = hog is None or hog.get("profit_margin") is None
    if not pm_missing and hog is not None:
        pm = hog.get("profit_margin")
        if pm is not None:
            hog_profit = float(pm)

    pt: float | None = None
    if hog and "price_trend" in hog:
        pt = float(hog["price_trend"])

    # ---- 利润率档位（随 regime 调整力度）----
    if not pm_missing and pm is not None:
        if regime == "distress":
            if pm < -15:
                score += 18
                details.append(f"深亏去产能(pm{pm:.0f}%)")
            elif pm < -5:
                score += 12
                details.append(f"亏损(pm{pm:.0f}%)")
            elif pm < 0:
                score += 8
                details.append(f"近盈亏平衡(pm{pm:.0f}%)")
            else:
                score += 3
                details.append(f"distress但已转正(pm{pm:.0f}%)")
        elif regime == "repair":
            if pm < -15:
                score += 15
                details.append(f"养殖亏损{pm:.0f}%")
            elif pm < -5:
                score += 10
                details.append(f"亏损收窄(pm{pm:.0f}%)")
            elif pm < 5:
                score += 5
                details.append(f"修复中(pm{pm:.0f}%)")
            else:
                score -= 3
                details.append(f"repair阶段偏高利润(pm{pm:.0f}%)")
        elif regime == "neutral":
            if pm < -15:
                score += 15
                details.append(f"养殖亏损{pm:.0f}%")
            elif pm < -5:
                score += 8
                details.append(f"养殖亏损{pm:.0f}%")
            elif pm > 20:
                score -= 10
                details.append(f"高利润(pm{pm:.0f}%)")
            elif pm > 10:
                score -= 5
                details.append(f"盈利偏高(pm{pm:.0f}%)")
        else:  # boom
            if pm > 20:
                score -= 18
                details.append(f"暴利扩产风险(pm{pm:.0f}%)")
            elif pm > 12:
                score -= 12
                details.append(f"高利润(pm{pm:.0f}%)")
            elif pm > 5:
                score -= 6
                details.append(f"盈利(pm{pm:.0f}%)")
            else:
                score += 2
                details.append(f"boom阶段偏低利润(pm{pm:.0f}%)")

    # 利润缺失：用期货区间位补偿（distress 略加强）
    if pm_missing:
        if range_pct < 5:
            score += 12
            details.append("利润数据缺失+期货极低位(补偿+12)")
        elif range_pct < 18:
            score += 8
            details.append("利润数据缺失+期货低位(补偿+8)")
        elif range_pct < 40:
            score += 4
            details.append("利润数据缺失+中低位(补偿+4)")

    # ---- 现货短期趋势：体制化，避免 distress 下「反弹=扣分」----
    if pt is not None:
        if regime == "distress":
            if pt < -3:
                score += 5
                details.append(f"现货短期走弱({pt:+.1f}%)")
            elif pt > 3:
                details.append(f"现货短期反弹({pt:+.1f}%)不扣分(distress)")
            # +3 ~ -3 忽略
        elif regime == "repair":
            if pt < -3:
                score += 4
                details.append(f"现货短期走弱({pt:+.1f}%)")
            elif pt > 3:
                score -= 3
                details.append(f"现货短期走强({pt:+.1f}%)轻罚")
        elif regime == "neutral":
            if pt < -3:
                score += 5
            elif pt > 3:
                score -= 5
                details.append(f"现货短期走强({pt:+.1f}%)")
        else:  # boom
            if pt < -3:
                score += 3
                details.append(f"现货回调({pt:+.1f}%)")
            elif pt > 3:
                score -= 8
                details.append(f"现货过热({pt:+.1f}%)")

    return score, details, hog_profit


def score_symbol_regime(
    sym: str,
    name: str,
    exchange: str,
    df: pd.DataFrame,
    threshold: float = 10.0,
) -> dict | None:
    """
    Regime 引擎入口。非 LH0 → 与 legacy 完全一致。
    LH0 → supply 层同 legacy + 体制化生猪专项。
    """
    if sym != "LH0":
        return score_symbol_legacy(sym, name, exchange, df, threshold=threshold)

    try:
        last, range_pct = compute_range_pct(df)
        layers_score, layers_details, meta = compute_supply_layers(sym, df)
        hog = get_hog_fundamentals()

        pm = hog.get("profit_margin") if hog else None
        margin_missing = hog is None or pm is None
        regime = classify_lh_regime(
            float(pm) if pm is not None else None,
            margin_missing,
            range_pct,
        )
        hog_score, hog_details, hog_profit = _lh_regime_hog_layers(regime, hog, range_pct)

        fund_score = layers_score + hog_score
        fund_details = list(layers_details) + hog_details

        score = fund_score

        return {
            "symbol": sym,
            "name": name,
            "exchange": exchange,
            "price": last,
            "range_pct": range_pct,
            "score": score,
            "fund_score": fund_score,
            "fund_details": ", ".join(fund_details) if fund_details else "",
            "inv_change_4wk": meta["inv_change_4wk"],
            "inv_percentile": meta["inv_percentile"],
            "receipt_change": meta["receipt_change"],
            "seasonal_signal": meta["seasonal_signal"],
            "hog_profit": hog_profit,
            "fund_threshold": threshold,
            "p1_engine": "regime",
            "lh_regime": regime,
        }
    except Exception:
        return None


def compare_engines_row(
    sym: str,
    name: str,
    exchange: str,
    df: pd.DataFrame,
    threshold: float,
) -> dict:
    """同一数据上并行跑两引擎，供报表/脚本使用（不写入生产逻辑）。"""
    leg = score_symbol_legacy(sym, name, exchange, df, threshold=threshold)
    reg = score_symbol_regime(sym, name, exchange, df, threshold=threshold)
    return {
        "symbol": sym,
        "legacy": leg,
        "regime": reg,
        "legacy_score": None if leg is None else leg.get("score"),
        "regime_score": None if reg is None else reg.get("score"),
        "regime_name": None if reg is None else reg.get("lh_regime"),
    }
