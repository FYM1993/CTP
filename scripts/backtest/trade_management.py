from __future__ import annotations

from dataclasses import dataclass

from backtest.models import BacktestCase, TradePlan
from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND


DEFAULT_REVERSAL_TP1_EXIT_FRACTION = 0.33
DEFAULT_TREND_TP1_EXIT_FRACTION = 0.33
DEFAULT_TREND_REVERSE_SCORE_THRESHOLD = 20.0
TREND_INVALIDATION_REVERSE_CONFIRMED = "reverse_confirmed"
TREND_INVALIDATION_STRICT = "strict"
POST_TP1_STOP_INITIAL = "initial"
POST_TP1_STOP_RISK_BUFFER = "risk_buffer"


@dataclass(frozen=True, slots=True)
class TradeManagementProfile:
    name: str
    tp1_exit_fraction: float
    move_stop_to_entry_after_tp1: bool
    daily_invalidation_policy: str = TREND_INVALIDATION_STRICT
    reverse_score_threshold: float = DEFAULT_TREND_REVERSE_SCORE_THRESHOLD
    post_tp1_stop_policy: str = POST_TP1_STOP_INITIAL
    post_tp1_stop_r_buffer: float = 0.0


def _clamp_fraction(value: object, default: float = 0.5) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(fraction, 0.0), 1.0)


def _bool_config(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _float_config(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dict_config(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _profile_group_config(config: dict, group_names: tuple[str, ...], key: str) -> dict:
    for group_name in group_names:
        group = _dict_config(config.get(group_name))
        profile = _dict_config(group.get(key))
        if profile:
            return profile
    return {}


def _apply_management_profile_config(
    profile: TradeManagementProfile,
    config: dict,
) -> TradeManagementProfile:
    if not config:
        return profile
    return TradeManagementProfile(
        name=profile.name,
        tp1_exit_fraction=_clamp_fraction(
            config.get("tp1_exit_fraction", profile.tp1_exit_fraction),
            profile.tp1_exit_fraction,
        ),
        move_stop_to_entry_after_tp1=_bool_config(
            config.get("move_stop_to_entry_after_tp1"),
            profile.move_stop_to_entry_after_tp1,
        ),
        daily_invalidation_policy=str(
            config.get("daily_invalidation_policy", profile.daily_invalidation_policy)
        ).strip()
        or profile.daily_invalidation_policy,
        reverse_score_threshold=_float_config(
            config.get("reverse_score_threshold"),
            profile.reverse_score_threshold,
        ),
        post_tp1_stop_policy=str(
            config.get("post_tp1_stop_policy", profile.post_tp1_stop_policy)
        ).strip()
        or profile.post_tp1_stop_policy,
        post_tp1_stop_r_buffer=max(
            0.0,
            _float_config(config.get("post_tp1_stop_r_buffer"), profile.post_tp1_stop_r_buffer),
        ),
    )


def _default_management_profile(pre_market_cfg: dict) -> TradeManagementProfile:
    trade_management = _dict_config(pre_market_cfg.get("trade_management"))
    profile = TradeManagementProfile(
        name="default",
        tp1_exit_fraction=_clamp_fraction(pre_market_cfg.get("tp1_exit_fraction", 0.5), 0.5),
        move_stop_to_entry_after_tp1=_bool_config(
            pre_market_cfg.get("move_stop_to_entry_after_tp1"),
            True,
        ),
    )
    return _apply_management_profile_config(
        profile,
        _dict_config(trade_management.get("default")),
    )


def _reversal_management_profile(name: str, pre_market_cfg: dict) -> TradeManagementProfile:
    tp1_fraction = DEFAULT_REVERSAL_TP1_EXIT_FRACTION
    move_stop = False
    if "tp1_exit_fraction" in pre_market_cfg:
        tp1_fraction = _clamp_fraction(
            pre_market_cfg.get("tp1_exit_fraction"),
            DEFAULT_REVERSAL_TP1_EXIT_FRACTION,
        )
    if "move_stop_to_entry_after_tp1" in pre_market_cfg:
        move_stop = _bool_config(
            pre_market_cfg.get("move_stop_to_entry_after_tp1"),
            False,
        )
    return TradeManagementProfile(
        name=name,
        tp1_exit_fraction=tp1_fraction,
        move_stop_to_entry_after_tp1=move_stop,
    )


def _trend_management_profile(name: str, pre_market_cfg: dict) -> TradeManagementProfile:
    tp1_fraction = DEFAULT_TREND_TP1_EXIT_FRACTION
    move_stop = False
    if "tp1_exit_fraction" in pre_market_cfg:
        tp1_fraction = _clamp_fraction(
            pre_market_cfg.get("tp1_exit_fraction"),
            DEFAULT_TREND_TP1_EXIT_FRACTION,
        )
    if "move_stop_to_entry_after_tp1" in pre_market_cfg:
        move_stop = _bool_config(
            pre_market_cfg.get("move_stop_to_entry_after_tp1"),
            False,
        )
    return TradeManagementProfile(
        name=name,
        tp1_exit_fraction=tp1_fraction,
        move_stop_to_entry_after_tp1=move_stop,
        daily_invalidation_policy=TREND_INVALIDATION_REVERSE_CONFIRMED,
        post_tp1_stop_policy=POST_TP1_STOP_RISK_BUFFER,
        post_tp1_stop_r_buffer=0.25,
    )


def _entry_family_for_plan(plan: TradePlan | None) -> str:
    if plan is None:
        return ""
    entry_family = str(plan.meta.get("entry_family") or "").strip()
    if entry_family in {"reversal", "trend"}:
        return entry_family
    return ""


def _strategy_family_for_plan(plan: TradePlan | None, case: BacktestCase | None = None) -> str:
    if plan is not None:
        strategy_family = str(plan.meta.get("strategy_family") or "").strip()
        if strategy_family in {STRATEGY_REVERSAL, STRATEGY_TREND}:
            return strategy_family
    if case is not None and case.strategy_family in {STRATEGY_REVERSAL, STRATEGY_TREND}:
        return case.strategy_family
    return ""


def resolve_trade_management_profile(
    *,
    plan: TradePlan,
    case: BacktestCase,
    pre_market_cfg: dict,
) -> TradeManagementProfile:
    trade_management = _dict_config(pre_market_cfg.get("trade_management"))
    default_profile = _default_management_profile(pre_market_cfg)
    strategy_family = _strategy_family_for_plan(plan, case)
    entry_family = _entry_family_for_plan(plan)

    if strategy_family == STRATEGY_REVERSAL:
        profile = _reversal_management_profile("strategy_reversal", pre_market_cfg)
        return _apply_management_profile_config(
            profile,
            _profile_group_config(
                trade_management,
                ("strategy_profiles", "strategies"),
                strategy_family,
            ),
        )
    if strategy_family == STRATEGY_TREND:
        profile = _trend_management_profile("strategy_trend", pre_market_cfg)
        return _apply_management_profile_config(
            profile,
            _profile_group_config(
                trade_management,
                ("strategy_profiles", "strategies"),
                strategy_family,
            ),
        )

    if entry_family == "reversal":
        profile = _reversal_management_profile("entry_reversal", pre_market_cfg)
        return _apply_management_profile_config(
            profile,
            _profile_group_config(
                trade_management,
                ("entry_profiles", "entry_families"),
                entry_family,
            ),
        )
    if entry_family == "trend":
        profile = _trend_management_profile("entry_trend", pre_market_cfg)
        return _apply_management_profile_config(
            profile,
            _profile_group_config(
                trade_management,
                ("entry_profiles", "entry_families"),
                entry_family,
            ),
        )

    return default_profile


def trade_management_meta(
    *,
    profile: TradeManagementProfile,
    tp1_hit: bool,
    tp1_exit_price: float | None,
    protective_stop: float,
) -> dict[str, object]:
    return {
        "management_profile": profile.name,
        "management_tp1_exit_fraction": profile.tp1_exit_fraction,
        "management_move_stop_to_entry_after_tp1": profile.move_stop_to_entry_after_tp1,
        "management_daily_invalidation_policy": profile.daily_invalidation_policy,
        "management_reverse_score_threshold": profile.reverse_score_threshold,
        "management_post_tp1_stop_policy": profile.post_tp1_stop_policy,
        "management_post_tp1_stop_r_buffer": profile.post_tp1_stop_r_buffer,
        "tp1_exit_fraction": profile.tp1_exit_fraction if tp1_hit else 0.0,
        "tp1_exit_price": tp1_exit_price or 0.0,
        "protective_stop": protective_stop,
    }


def stop_after_first_target(
    *,
    direction: str,
    entry_price: float,
    initial_stop: float,
    current_stop: float,
    profile: TradeManagementProfile,
) -> float:
    if profile.move_stop_to_entry_after_tp1:
        candidate = float(entry_price)
    elif profile.post_tp1_stop_policy == POST_TP1_STOP_RISK_BUFFER:
        risk = abs(float(entry_price) - float(initial_stop))
        buffer = risk * max(float(profile.post_tp1_stop_r_buffer), 0.0)
        if direction == "long":
            candidate = float(entry_price) - buffer
        else:
            candidate = float(entry_price) + buffer
    else:
        return float(current_stop)

    if direction == "long":
        return max(float(current_stop), candidate)
    return min(float(current_stop), candidate)


def calc_pnl_ratio(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def realized_pnl_ratio(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
    tp1_hit: bool,
    tp1_exit_price: float | None,
    tp1_exit_fraction: float,
) -> float:
    if not tp1_hit or tp1_exit_price is None or tp1_exit_fraction <= 0:
        return calc_pnl_ratio(direction, entry_price, exit_price)
    first_leg = calc_pnl_ratio(direction, entry_price, tp1_exit_price)
    final_leg = calc_pnl_ratio(direction, entry_price, exit_price)
    return tp1_exit_fraction * first_leg + (1.0 - tp1_exit_fraction) * final_leg


def trend_daily_review_keeps_position(
    *,
    direction: str,
    hold_status: dict | None,
    profile: TradeManagementProfile,
) -> bool:
    if hold_status and hold_status.get("hold_valid"):
        return True
    if profile.daily_invalidation_policy != TREND_INVALIDATION_REVERSE_CONFIRMED:
        return False
    if not hold_status:
        return False

    trend_status = _dict_config(hold_status.get("trend_status"))
    score = _float_config(
        trend_status.get("directional_score", hold_status.get("score")),
        0.0,
    )
    threshold = abs(float(profile.reverse_score_threshold))
    if direction == "long":
        return score > -threshold
    return score < threshold
