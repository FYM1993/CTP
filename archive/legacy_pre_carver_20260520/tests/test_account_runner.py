from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.account_runner import (  # noqa: E402
    AccountBacktestConfig,
    AccountCandidate,
    candidate_from_trade,
    run_account_backtest,
    write_account_outputs,
)
from backtest.models import TradeRecord  # noqa: E402


def _candidate(
    *,
    symbol: str,
    name: str,
    entry_time: str,
    exit_time: str,
    score: float,
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    direction: str = "long",
    risk_per_lot: float = 1_000.0,
    margin_per_lot: float = 10_000.0,
    multiplier: float = 10.0,
    commission_per_lot: float = 2.0,
    commission_rate: float = 0.0,
    close_today_commission_per_lot: float | None = None,
    entry_rr: float = 2.0,
    entry_admission_rr: float = 3.0,
    medium_term_quality_score: float = 55.0,
    medium_term_entry_location_score: float = 70.0,
    entry_adverse_deviation_r: float = 0.0,
    phase1_story_budget_margin_pct: float = 0.0,
    tp1_price: float = 105.0,
    tp1_time: str = "",
) -> AccountCandidate:
    return AccountCandidate(
        symbol=symbol,
        name=name,
        direction=direction,
        entry_time=entry_time,
        planned_exit_time=exit_time,
        entry_price=entry_price,
        planned_exit_price=exit_price,
        planned_exit_reason="tp2" if exit_price != entry_price else "stop",
        phase2_score=score,
        risk_per_lot=risk_per_lot,
        margin_per_lot=margin_per_lot,
        notional_per_lot=entry_price * multiplier,
        multiplier=multiplier,
        pnl_ratio_price=(exit_price - entry_price) / entry_price
        if direction == "long"
        else (entry_price - exit_price) / entry_price,
        tp1_hit=exit_price != entry_price,
        commission_per_lot=commission_per_lot,
        commission_rate=commission_rate,
        close_today_commission_per_lot=close_today_commission_per_lot,
        trade_id=f"{symbol}-{entry_time}",
        entry_rr=entry_rr,
        entry_admission_rr=entry_admission_rr,
        medium_term_quality_score=medium_term_quality_score,
        medium_term_entry_location_score=medium_term_entry_location_score,
        entry_adverse_deviation_r=entry_adverse_deviation_r,
        phase1_story_budget_margin_pct=phase1_story_budget_margin_pct,
        tp1_price=tp1_price,
        tp1_time=tp1_time,
    )


def test_account_backtest_sizes_trade_by_phase2_margin_portfolio_and_stop_risk() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="AU0",
                name="黄金",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-03 09:00:00",
                score=65.0,
            )
        ],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            commission_multiplier=1.01,
        ),
    )

    assert result.summary["candidate_trades"] == 1
    assert result.summary["accepted_trades"] == 1
    trade = result.trades[0]
    assert trade["target_margin_pct"] == pytest.approx(0.30)
    assert trade["margin_lots_cap"] == 3
    assert trade["risk_lots_cap"] == 1
    assert trade["lots"] == 1
    assert trade["initial_stop_risk_pct_equity"] == pytest.approx(0.01)
    assert trade["fees"] == pytest.approx(4.04)
    assert trade["net_pnl"] == pytest.approx(95.96)
    assert result.summary["final_equity"] == pytest.approx(100_095.96)


def test_account_backtest_can_size_by_phase1_story_budget_instead_of_phase2_score() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="HI0",
                name="高质量趋势",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-03 09:00:00",
                score=30.0,
                margin_per_lot=10_000.0,
                risk_per_lot=500.0,
                phase1_story_budget_margin_pct=0.30,
            ),
            _candidate(
                symbol="LO0",
                name="普通趋势",
                entry_time="2025-01-02 09:01:00",
                exit_time="2025-01-03 09:00:00",
                score=80.0,
                margin_per_lot=10_000.0,
                risk_per_lot=500.0,
                phase1_story_budget_margin_pct=0.03,
            ),
        ],
        AccountBacktestConfig(
            initial_equity=1_000_000.0,
            max_portfolio_margin_pct=1.0,
            risk_per_trade_pct=0.015,
            position_budget_source="phase1_story",
        ),
    )

    decisions = {row["symbol"]: row for row in result.decisions}
    assert decisions["HI0"]["target_margin_pct"] == pytest.approx(0.30)
    assert decisions["HI0"]["score_lots"] == 30
    assert decisions["HI0"]["target_lots"] == 30
    assert decisions["LO0"]["target_margin_pct"] == pytest.approx(0.03)
    assert decisions["LO0"]["score_lots"] == 3
    assert decisions["LO0"]["target_lots"] == 3


def test_account_backtest_can_reserve_planned_second_entry_budget() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="A0",
                name="第一张趋势票",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-04 09:00:00",
                score=70.0,
                margin_per_lot=10_000.0,
                risk_per_lot=100.0,
                phase1_story_budget_margin_pct=0.30,
                tp1_time="2025-01-03 09:00:00",
            ),
            _candidate(
                symbol="B0",
                name="第二张趋势票",
                entry_time="2025-01-02 09:01:00",
                exit_time="2025-01-04 09:00:00",
                score=70.0,
                margin_per_lot=10_000.0,
                risk_per_lot=100.0,
                phase1_story_budget_margin_pct=0.30,
            ),
        ],
        AccountBacktestConfig(
            initial_equity=1_000_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            position_budget_source="phase1_story",
            split_initial_fraction=0.30,
            split_second_entry_trigger="tp1",
            reserve_planned_second_entry_margin=True,
        ),
    )

    assert [row["decision"] for row in result.decisions] == ["accepted", "skipped"]
    assert result.decisions[0]["accepted_lots"] == 9
    assert result.decisions[0]["target_lots"] == 30
    assert result.decisions[0]["planned_second_entry_lots"] == 21
    assert result.decisions[1]["portfolio_lots"] == 0
    assert result.decisions[1]["skip_reason"] == "portfolio_margin_full_lower_score"
    assert result.trades[0]["second_entry_lots"] == 21
    assert result.summary["accepted_trades"] == 1
    assert result.summary["reserve_planned_second_entry_margin"] is True


def test_account_backtest_can_filter_when_story_or_trigger_is_not_strong_enough() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="LOWSTORY",
                name="趋势故事不足",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-03 09:00:00",
                score=80.0,
                margin_per_lot=10_000.0,
                risk_per_lot=100.0,
                phase1_story_budget_margin_pct=0.03,
            ),
            _candidate(
                symbol="LOWTRIGGER",
                name="短线触发不足",
                entry_time="2025-01-02 09:01:00",
                exit_time="2025-01-03 09:00:00",
                score=35.0,
                margin_per_lot=10_000.0,
                risk_per_lot=100.0,
                phase1_story_budget_margin_pct=0.30,
            ),
            _candidate(
                symbol="PASS",
                name="可交易趋势",
                entry_time="2025-01-02 09:02:00",
                exit_time="2025-01-03 09:00:00",
                score=45.0,
                margin_per_lot=10_000.0,
                risk_per_lot=100.0,
                phase1_story_budget_margin_pct=0.30,
            ),
        ],
        AccountBacktestConfig(
            initial_equity=1_000_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            position_budget_source="phase1_story",
            min_phase1_story_budget_margin_pct=0.15,
            min_phase2_abs_score=40.0,
        ),
    )

    decisions = {row["symbol"]: row for row in result.decisions}
    assert decisions["LOWSTORY"]["decision"] == "skipped"
    assert decisions["LOWSTORY"]["skip_reason"] == "phase1_story_below_trade_threshold"
    assert decisions["LOWTRIGGER"]["decision"] == "skipped"
    assert decisions["LOWTRIGGER"]["skip_reason"] == "phase2_trigger_below_trade_threshold"
    assert decisions["PASS"]["decision"] == "accepted"
    assert result.summary["accepted_trades"] == 1


def test_account_backtest_can_tier_stop_risk_by_target_strength() -> None:
    ordinary = _candidate(
        symbol="A0",
        name="普通趋势",
        entry_time="2025-01-02 09:00:00",
        exit_time="2025-01-03 09:00:00",
        score=65.0,
        risk_per_lot=250.0,
        margin_per_lot=1_000.0,
        entry_rr=1.6,
        entry_admission_rr=2.8,
    )
    strong = _candidate(
        symbol="B0",
        name="强趋势",
        entry_time="2025-01-04 09:00:00",
        exit_time="2025-01-05 09:00:00",
        score=65.0,
        risk_per_lot=250.0,
        margin_per_lot=1_000.0,
        entry_rr=2.2,
        entry_admission_rr=3.5,
    )

    result = run_account_backtest(
        [ordinary, strong],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.50,
            risk_per_trade_pct=0.015,
            ordinary_risk_per_trade_pct=0.005,
            strong_risk_per_trade_pct=0.015,
            strong_first_target_rr=2.0,
            strong_second_target_rr=3.0,
        ),
    )

    assert [row["risk_tier"] for row in result.decisions] == ["ordinary", "strong"]
    assert [row["effective_risk_per_trade_pct"] for row in result.decisions] == [pytest.approx(0.005), pytest.approx(0.015)]
    assert [row["accepted_lots"] for row in result.decisions] == [2, 6]
    assert [trade["risk_budget_pct"] for trade in result.trades] == [pytest.approx(0.005), pytest.approx(0.015)]
    assert [trade["risk_lots_cap"] for trade in result.trades] == [2, 6]


def test_account_backtest_can_split_entry_and_add_second_leg_after_tp1() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="AU0",
                name="黄金",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-05 09:00:00",
                score=65.0,
                entry_price=100.0,
                exit_price=110.0,
                risk_per_lot=150.0,
                margin_per_lot=1_000.0,
                multiplier=10.0,
                commission_per_lot=0.0,
                tp1_price=105.0,
                tp1_time="2025-01-03 09:00:00",
            )
        ],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            split_initial_fraction=0.5,
            split_second_entry_trigger="tp1",
        ),
    )

    decision = result.decisions[0]
    assert decision["target_lots"] == 10
    assert decision["accepted_lots"] == 5
    assert decision["first_entry_lots"] == 5
    assert decision["planned_second_entry_lots"] == 5

    trade = result.trades[0]
    assert trade["target_lots"] == 10
    assert trade["first_entry_lots"] == 5
    assert trade["second_entry_lots"] == 5
    assert trade["unfilled_second_entry_lots"] == 0
    assert trade["second_entry_time"] == "2025-01-03 09:00:00"
    assert trade["second_entry_price"] == pytest.approx(105.0)
    assert trade["lots"] == 10
    assert trade["gross_pnl"] == pytest.approx(750.0)
    assert result.summary["split_second_entry_filled_lots"] == 5
    assert result.summary["split_second_entry_unfilled_lots"] == 0


def test_account_backtest_keeps_unfilled_second_leg_when_tp1_not_hit() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="AU0",
                name="黄金",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-03 09:00:00",
                score=65.0,
                exit_price=95.0,
                risk_per_lot=150.0,
                margin_per_lot=1_000.0,
                multiplier=10.0,
                commission_per_lot=0.0,
                tp1_price=105.0,
                tp1_time="",
            )
        ],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            split_initial_fraction=0.5,
            split_second_entry_trigger="tp1",
        ),
    )

    trade = result.trades[0]
    assert trade["target_lots"] == 10
    assert trade["first_entry_lots"] == 5
    assert trade["second_entry_lots"] == 0
    assert trade["unfilled_second_entry_lots"] == 5
    assert trade["lots"] == 5
    assert trade["gross_pnl"] == pytest.approx(-250.0)
    assert result.summary["split_second_entry_filled_lots"] == 0
    assert result.summary["split_second_entry_unfilled_lots"] == 5


def test_account_backtest_conditional_execution_uses_trade_quality_to_choose_entry_style() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="A0",
                name="强趋势高确定性",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-10 09:00:00",
                score=65.0,
                risk_per_lot=150.0,
                margin_per_lot=1_000.0,
                commission_per_lot=0.0,
                entry_rr=1.8,
                entry_admission_rr=3.2,
                medium_term_quality_score=70.0,
                entry_adverse_deviation_r=0.1,
            ),
            _candidate(
                symbol="B0",
                name="趋势好但入场一般",
                entry_time="2025-01-02 09:01:00",
                exit_time="2025-01-10 09:00:00",
                score=65.0,
                risk_per_lot=150.0,
                margin_per_lot=1_000.0,
                commission_per_lot=0.0,
                entry_rr=1.2,
                entry_admission_rr=3.0,
                medium_term_quality_score=55.0,
                entry_adverse_deviation_r=0.4,
                tp1_time="2025-01-03 09:00:00",
            ),
            _candidate(
                symbol="C0",
                name="低第一目标高第二目标",
                entry_time="2025-01-02 09:02:00",
                exit_time="2025-01-10 09:00:00",
                score=65.0,
                risk_per_lot=150.0,
                margin_per_lot=1_000.0,
                commission_per_lot=0.0,
                entry_rr=0.8,
                entry_admission_rr=3.1,
                medium_term_quality_score=48.0,
                entry_adverse_deviation_r=0.6,
                tp1_time="2025-01-03 09:00:00",
            ),
            _candidate(
                symbol="D0",
                name="趋势质量不足",
                entry_time="2025-01-02 09:03:00",
                exit_time="2025-01-10 09:00:00",
                score=65.0,
                risk_per_lot=150.0,
                margin_per_lot=1_000.0,
                commission_per_lot=0.0,
                entry_rr=0.8,
                entry_admission_rr=2.0,
                medium_term_quality_score=35.0,
                entry_adverse_deviation_r=0.9,
            ),
        ],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=1.0,
            risk_per_trade_pct=0.015,
            execution_policy="conditional",
        ),
    )

    assert [row["decision"] for row in result.decisions] == ["accepted", "accepted", "accepted", "skipped"]
    assert [row["execution_profile"] for row in result.decisions] == [
        "direct_full",
        "confirm_scale_in_70",
        "trial_scale_in_30",
        "skip_weak_quality",
    ]
    assert [row["accepted_lots"] for row in result.decisions] == [10, 7, 3, 0]
    assert [row["planned_second_entry_lots"] for row in result.decisions] == [0, 3, 7, 0]
    assert result.decisions[3]["skip_reason"] == "conditional_execution_quality_below_trade_threshold"

    trades = {row["symbol"]: row for row in result.trades}
    assert trades["A0"]["execution_profile"] == "direct_full"
    assert trades["A0"]["second_entry_lots"] == 0
    assert trades["B0"]["first_entry_lots"] == 7
    assert trades["B0"]["second_entry_lots"] == 3
    assert trades["C0"]["first_entry_lots"] == 3
    assert trades["C0"]["second_entry_lots"] == 7
    assert result.summary["execution_profile_counts"] == {
        "direct_full": 1,
        "confirm_scale_in_70": 1,
        "trial_scale_in_30": 1,
    }


def test_candidate_from_trade_derives_entry_adverse_deviation_r_from_trigger_metadata() -> None:
    trade = TradeRecord(
        trade_id="AU0-2025-01-02",
        symbol="AU0",
        direction="long",
        entry_time="2025-01-02 09:00:00",
        entry_price=105.0,
        exit_time="2025-01-03 09:00:00",
        exit_price=110.0,
        exit_reason="tp2",
        bars_held=10,
        days_held=1,
        tp1_hit=True,
        pnl_ratio=0.05,
        meta={
            "phase2_score": 65.0,
            "contract_multiplier": 10.0,
            "planned_entry_ref": 102.0,
            "planned_stop": 98.0,
            "entry_trigger_deviation": 3.0,
            "execution_risk_per_lot": 40.0,
            "execution_margin_per_lot": 1_000.0,
            "entry_rr": 1.2,
            "entry_admission_rr": 3.0,
            "medium_term_quality_score": 62.0,
            "medium_term_entry_location_score": 70.0,
        },
    )

    candidate = candidate_from_trade(trade)

    assert candidate.entry_adverse_deviation_r == pytest.approx(0.75)
    assert candidate.medium_term_quality_score == pytest.approx(62.0)
    assert candidate.medium_term_entry_location_score == pytest.approx(70.0)


def test_account_backtest_replaces_lower_score_position_when_portfolio_is_full() -> None:
    low_score = _candidate(
        symbol="BU0",
        name="沥青",
        entry_time="2025-01-02 09:00:00",
        exit_time="2025-01-10 09:00:00",
        score=65.0,
        exit_price=100.0,
        margin_per_lot=29_000.0,
    )
    high_score = _candidate(
        symbol="AG0",
        name="白银",
        entry_time="2025-01-03 09:00:00",
        exit_time="2025-01-04 09:00:00",
        score=80.0,
        margin_per_lot=29_000.0,
    )

    result = run_account_backtest(
        [low_score, high_score],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.02,
            commission_multiplier=1.01,
            allow_replacement=True,
        ),
    )

    assert result.summary["candidate_trades"] == 2
    assert result.summary["accepted_trades"] == 2
    assert result.summary["replacements"] == 1
    assert result.trades[0]["actual_exit_reason"] == "replaced_by_higher_phase2_score"
    assert result.trades[0]["actual_exit_time"] == "2025-01-03 09:00:00"
    assert result.trades[1]["symbol"] == "AG0"
    assert [row["decision"] for row in result.decisions] == ["accepted", "replaced_then_accepted"]
    assert result.decisions[1]["replacement_count"] == 1
    assert result.decisions[1]["replaced_symbols"] == "BU0"
    assert result.decisions[1]["accepted_lots"] == 1


def test_account_backtest_replaces_multiple_lower_score_positions_until_funded() -> None:
    low_a = _candidate(
        symbol="BU0",
        name="沥青",
        entry_time="2025-01-02 09:00:00",
        exit_time="2025-01-10 09:00:00",
        score=50.0,
        exit_price=100.0,
        risk_per_lot=100.0,
        margin_per_lot=10_000.0,
    )
    low_b = _candidate(
        symbol="FU0",
        name="燃料油",
        entry_time="2025-01-02 09:01:00",
        exit_time="2025-01-10 09:00:00",
        score=50.0,
        exit_price=100.0,
        risk_per_lot=100.0,
        margin_per_lot=10_000.0,
    )
    high_score = _candidate(
        symbol="AU0",
        name="黄金",
        entry_time="2025-01-03 09:00:00",
        exit_time="2025-01-04 09:00:00",
        score=80.0,
        risk_per_lot=100.0,
        margin_per_lot=25_000.0,
    )

    result = run_account_backtest(
        [low_a, low_b, high_score],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.02,
            commission_multiplier=1.01,
            allow_replacement=True,
        ),
    )

    assert result.summary["accepted_trades"] == 3
    assert result.summary["skipped_trades"] == 0
    assert result.summary["replacements"] == 2
    assert [row["actual_exit_reason"] for row in result.trades[:2]] == [
        "replaced_by_higher_phase2_score",
        "replaced_by_higher_phase2_score",
    ]
    assert result.trades[2]["symbol"] == "AU0"


def test_account_backtest_does_not_replace_fresh_position_before_min_hold() -> None:
    fresh_low_score = _candidate(
        symbol="FU0",
        name="燃料油",
        entry_time="2025-01-02 09:00:00",
        exit_time="2025-01-10 09:00:00",
        score=55.0,
        exit_price=100.0,
        risk_per_lot=100.0,
        margin_per_lot=25_000.0,
    )
    high_score = _candidate(
        symbol="SC0",
        name="原油",
        entry_time="2025-01-02 09:05:00",
        exit_time="2025-01-03 09:00:00",
        score=65.0,
        risk_per_lot=100.0,
        margin_per_lot=25_000.0,
    )

    result = run_account_backtest(
        [fresh_low_score, high_score],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            commission_multiplier=1.01,
            allow_replacement=True,
            min_replacement_hold_hours=22.0,
        ),
    )

    assert result.summary["accepted_trades"] == 1
    assert result.summary["skipped_trades"] == 1
    assert result.summary["replacements"] == 0
    assert result.trades[0]["symbol"] == "FU0"
    assert result.trades[0]["actual_exit_reason"] == "stop"
    assert result.decisions[1]["decision"] == "skipped"
    assert result.decisions[1]["skip_reason"] == "portfolio_margin_full_lower_score"
    assert result.decisions[1]["replacement_block_reason"] == "min_replacement_hold_hours"
    assert result.decisions[1]["replacement_block_symbol"] == "FU0"


def test_account_backtest_does_not_skip_fresh_lowest_score_to_replace_older_position() -> None:
    older_low_score = _candidate(
        symbol="AP0",
        name="苹果",
        entry_time="2025-01-01 09:00:00",
        exit_time="2025-01-10 09:00:00",
        score=50.0,
        exit_price=100.0,
        risk_per_lot=100.0,
        margin_per_lot=15_000.0,
    )
    fresh_lower_score = _candidate(
        symbol="FU0",
        name="燃料油",
        entry_time="2025-01-02 09:00:00",
        exit_time="2025-01-10 09:00:00",
        score=45.0,
        exit_price=100.0,
        risk_per_lot=100.0,
        margin_per_lot=15_000.0,
    )
    high_score = _candidate(
        symbol="SC0",
        name="原油",
        entry_time="2025-01-02 09:05:00",
        exit_time="2025-01-03 09:00:00",
        score=65.0,
        risk_per_lot=100.0,
        margin_per_lot=25_000.0,
    )

    result = run_account_backtest(
        [older_low_score, fresh_lower_score, high_score],
        AccountBacktestConfig(
            initial_equity=100_000.0,
            max_portfolio_margin_pct=0.30,
            risk_per_trade_pct=0.015,
            commission_multiplier=1.01,
            allow_replacement=True,
            min_replacement_hold_hours=22.0,
        ),
    )

    assert result.summary["accepted_trades"] == 2
    assert result.summary["skipped_trades"] == 1
    assert result.summary["replacements"] == 0
    assert [row["symbol"] for row in result.trades] == ["AP0", "FU0"]


def test_account_backtest_charges_notional_rate_fees_on_entry_and_exit() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="BU0",
                name="沥青",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-03 09:00:00",
                score=65.0,
                entry_price=3704.0,
                exit_price=3625.5857142857144,
                risk_per_lot=100.0,
                margin_per_lot=5_556.0,
                multiplier=10.0,
                commission_per_lot=0.0,
                commission_rate=0.00005,
            )
        ],
        AccountBacktestConfig(initial_equity=100_000.0, risk_per_trade_pct=0.015),
    )

    expected = (3704.0 + 3625.5857142857144) * 10.0 * 0.00005 * 5 * 1.01
    assert result.trades[0]["lots"] == 5
    assert result.trades[0]["fees"] == pytest.approx(expected)


def test_account_backtest_uses_close_today_fee_for_same_day_exit() -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="AP0",
                name="苹果",
                entry_time="2025-01-13 09:00:00",
                exit_time="2025-01-13 10:00:00",
                score=65.0,
                risk_per_lot=100.0,
                margin_per_lot=5_000.0,
                commission_per_lot=5.0,
                close_today_commission_per_lot=20.0,
            )
        ],
        AccountBacktestConfig(initial_equity=100_000.0, risk_per_trade_pct=0.015),
    )

    assert result.trades[0]["fees"] == pytest.approx((5.0 + 20.0) * 6 * 1.01)


def test_write_account_outputs_creates_reproducible_report_files(tmp_path: Path) -> None:
    result = run_account_backtest(
        [
            _candidate(
                symbol="AU0",
                name="黄金",
                entry_time="2025-01-02 09:00:00",
                exit_time="2025-01-03 09:00:00",
                score=65.0,
            )
        ],
        AccountBacktestConfig(initial_equity=100_000.0, risk_per_trade_pct=0.015),
    )
    prefix = tmp_path / "account_smoke"

    written = write_account_outputs(result, prefix)

    assert {path.name for path in written} == {
        "account_smoke_summary.json",
        "account_smoke_trades.csv",
        "account_smoke_equity.csv",
        "account_smoke_by_symbol.csv",
        "account_smoke_decisions.csv",
    }
    summary = json.loads((tmp_path / "account_smoke_summary.json").read_text())
    assert summary["accepted_trades"] == 1
    assert "symbol,name,direction,entry_time" in (tmp_path / "account_smoke_trades.csv").read_text()
    assert "sequence,candidate_trade_id,symbol" in (tmp_path / "account_smoke_decisions.csv").read_text()
