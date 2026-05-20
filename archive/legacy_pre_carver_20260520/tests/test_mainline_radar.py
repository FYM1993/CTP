from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mainline_radar import (  # noqa: E402
    build_daily_snapshot,
    build_effectiveness_validation,
    build_replay_report,
    build_risk_worksheet,
    build_weekly_radar,
    render_effectiveness_validation_markdown,
    render_snapshot_markdown,
)


def _row(
    *,
    date: str,
    group: str,
    symbol: str,
    bucket: str = "medium",
    direction: str = "long",
    score: float = 60.0,
    capital: float = 60.0,
    resonance: float = 60.0,
    breadth: float = 0.5,
    crowding: float = 0.0,
    count: int = 1,
    leader: str | None = None,
    retrace_atr: float = 1.0,
) -> dict:
    return {
        "date": date,
        "group": group,
        "symbol": symbol,
        "direction": direction,
        "bucket": bucket,
        "mainline_score": score,
        "score": score,
        "mainline_capital_score": capital,
        "mainline_resonance_score": resonance,
        "mainline_crowding_score": crowding,
        "mainline_leadership_score": score,
        "group_breadth": breadth,
        "group_symbol_count": count,
        "group_leader_symbol": leader or symbol,
        "group_best_symbol_score": score,
        "symbol_setup_score": score,
        "retrace_atr": retrace_atr,
    }


def test_daily_snapshot_ranks_board_resonance_above_isolated_single_symbol_move() -> None:
    rows = [
        _row(date="2025-01-02", group="metals", symbol="AU0", score=62.0, breadth=1.0, count=3),
        _row(date="2025-01-02", group="metals", symbol="AG0", score=62.0, breadth=1.0, count=3),
        _row(date="2025-01-02", group="single_move", symbol="AP0", score=62.0, breadth=0.25, count=1),
    ]

    snapshot = build_daily_snapshot(rows, "2025-01-02")

    assert snapshot[0]["board"] == "metals"
    assert snapshot[0]["representative_symbol"] == "AU0"
    assert snapshot[0]["leadership_breadth"] == 1.0


def test_daily_snapshot_ranks_capital_confirmed_board_above_price_only_strength() -> None:
    rows = [
        _row(date="2025-01-02", group="capital_confirmed", symbol="CU0", score=61.0, capital=88.0, resonance=72.0, count=2),
        _row(date="2025-01-02", group="price_only", symbol="RB0", score=61.0, capital=25.0, resonance=72.0, count=2),
    ]

    snapshot = build_daily_snapshot(rows, "2025-01-02")

    assert snapshot[0]["board"] == "capital_confirmed"
    assert "资金确认" in snapshot[0]["action_hint"]


def test_daily_snapshot_marks_crowded_board_as_reduce_not_buy_more() -> None:
    rows = [
        _row(
            date="2025-01-02",
            group="energy",
            symbol="SC0",
            bucket="crowded",
            score=85.0,
            crowding=82.0,
            breadth=1.0,
            count=2,
        )
    ]

    snapshot = build_daily_snapshot(rows, "2025-01-02")

    assert snapshot[0]["lifecycle_stage"] == "crowded"
    assert "避免新增" in snapshot[0]["action_hint"]


def test_daily_snapshot_labels_single_symbol_strength_as_isolated_move() -> None:
    rows = [
        _row(date="2025-01-02", group="livestock_perishables", symbol="AP0", bucket="high", score=92.0, count=1)
    ]

    snapshot = build_daily_snapshot(rows, "2025-01-02")

    assert snapshot[0]["is_isolated_move"] is True
    assert "单品种" in snapshot[0]["action_hint"]


def test_daily_snapshot_marks_medium_to_high_transition_as_confirmation() -> None:
    rows = [
        _row(date="2025-01-01", group="metals", symbol="AU0", bucket="medium", score=61.0, count=2),
        _row(date="2025-01-02", group="metals", symbol="AU0", bucket="high", score=73.0, count=2),
    ]

    snapshot = build_daily_snapshot(rows, "2025-01-02")

    assert snapshot[0]["lifecycle_stage"] == "confirming"
    assert "确认" in snapshot[0]["action_hint"]


def test_snapshot_markdown_starts_with_business_conclusion() -> None:
    rows = [_row(date="2025-01-02", group="metals", symbol="AU0", bucket="high", score=75.0)]
    snapshot = build_daily_snapshot(rows, "2025-01-02")

    markdown = render_snapshot_markdown(snapshot, "2025-01-02")

    assert markdown.startswith("# 主线雷达日度快照 2025-01-02\n\n## 业务结论")


def test_weekly_radar_ranks_persistent_board_above_one_day_spike() -> None:
    rows: list[dict] = []
    for day in range(1, 6):
        rows.append(
            _row(
                date=f"2025-01-0{day}",
                group="persistent_board",
                symbol="AU0",
                bucket="high",
                score=68.0,
                breadth=1.0,
                count=2,
            )
        )
        rows.append(
            _row(
                date=f"2025-01-0{day}",
                group="one_day_spike",
                symbol="RB0",
                bucket="high" if day == 5 else "low",
                score=92.0 if day == 5 else 40.0,
                breadth=1.0 if day == 5 else 0.0,
                count=2,
            )
        )

    weekly = build_weekly_radar(rows, "2025-01-05")

    assert weekly["top_mainlines"][0]["board"] == "persistent_board"
    assert weekly["top_mainlines"][0]["active_days"] == 5


def test_weekly_radar_marks_shallowing_pullbacks_as_improving_consensus() -> None:
    rows = [
        _row(date="2025-01-01", group="metals", symbol="AU0", bucket="medium", score=58.0, breadth=0.5, count=2, retrace_atr=2.8),
        _row(date="2025-01-02", group="metals", symbol="AU0", bucket="medium", score=60.0, breadth=0.6, count=2, retrace_atr=2.0),
        _row(date="2025-01-03", group="metals", symbol="AU0", bucket="medium", score=63.0, breadth=0.7, count=2, retrace_atr=1.6),
        _row(date="2025-01-04", group="metals", symbol="AU0", bucket="high", score=70.0, breadth=0.9, count=2, retrace_atr=1.0),
        _row(date="2025-01-05", group="metals", symbol="AU0", bucket="high", score=75.0, breadth=1.0, count=2, retrace_atr=0.5),
    ]

    weekly = build_weekly_radar(rows, "2025-01-05")

    assert weekly["improving_boards"][0]["board"] == "metals"
    assert "分歧转共识" in weekly["improving_boards"][0]["weekly_hint"]


def test_weekly_radar_marks_high_score_rising_crowding_as_late_cycle_risk() -> None:
    rows = [
        _row(date="2025-01-01", group="energy", symbol="SC0", bucket="high", score=75.0, crowding=10.0, breadth=1.0, count=2),
        _row(date="2025-01-02", group="energy", symbol="SC0", bucket="high", score=78.0, crowding=25.0, breadth=1.0, count=2),
        _row(date="2025-01-03", group="energy", symbol="SC0", bucket="high", score=82.0, crowding=45.0, breadth=1.0, count=2),
        _row(date="2025-01-04", group="energy", symbol="SC0", bucket="high", score=86.0, crowding=65.0, breadth=1.0, count=2),
        _row(date="2025-01-05", group="energy", symbol="SC0", bucket="crowded", score=88.0, crowding=85.0, breadth=1.0, count=2),
    ]

    weekly = build_weekly_radar(rows, "2025-01-05")

    assert weekly["crowded_boards"][0]["board"] == "energy"
    assert weekly["crowded_boards"][0]["weekly_status"] == "late_cycle_risk"


def test_weekly_radar_keeps_weakening_board_out_of_top_mainlines() -> None:
    rows = [
        _row(date="2025-01-01", group="weakening", symbol="SC0", bucket="high", score=78.0, breadth=1.0, count=2),
        _row(date="2025-01-02", group="weakening", symbol="SC0", bucket="high", score=72.0, breadth=0.8, count=2),
        _row(date="2025-01-03", group="weakening", symbol="SC0", bucket="medium", score=64.0, breadth=0.6, count=2),
        _row(date="2025-01-04", group="weakening", symbol="SC0", bucket="medium", score=58.0, breadth=0.4, count=2),
        _row(date="2025-01-05", group="weakening", symbol="SC0", bucket="low", score=50.0, breadth=0.2, count=2),
        _row(date="2025-01-01", group="persistent", symbol="AU0", bucket="high", score=70.0, breadth=1.0, count=2),
        _row(date="2025-01-02", group="persistent", symbol="AU0", bucket="high", score=70.0, breadth=1.0, count=2),
        _row(date="2025-01-03", group="persistent", symbol="AU0", bucket="high", score=70.0, breadth=1.0, count=2),
        _row(date="2025-01-04", group="persistent", symbol="AU0", bucket="high", score=70.0, breadth=1.0, count=2),
        _row(date="2025-01-05", group="persistent", symbol="AU0", bucket="high", score=70.0, breadth=1.0, count=2),
    ]

    weekly = build_weekly_radar(rows, "2025-01-05")

    assert "weakening" not in {row["board"] for row in weekly["top_mainlines"]}
    assert weekly["weakening_boards"][0]["board"] == "weakening"


def test_replay_counts_board_mainline_days_by_year() -> None:
    rows = [
        _row(date="2024-01-01", group="metals", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2024-01-02", group="metals", symbol="AU0", bucket="high", score=76.0, count=2),
        _row(date="2025-01-01", group="energy", symbol="SC0", bucket="medium", score=60.0, count=2),
    ]

    replay = build_replay_report(rows, [], years=[2024, 2025])

    assert replay["yearly"]["2024"]["active_mainline_days"] == 2
    assert replay["yearly"]["2025"]["active_mainline_days"] == 1


def test_replay_separates_lifecycle_stages_by_year() -> None:
    rows = [
        _row(date="2025-01-01", group="metals", symbol="AU0", bucket="medium", score=60.0, count=2),
        _row(date="2025-01-02", group="metals", symbol="AU0", bucket="high", score=72.0, count=2),
        _row(date="2025-01-03", group="metals", symbol="AU0", bucket="crowded", score=86.0, crowding=88.0, count=2),
    ]

    replay = build_replay_report(rows, [], years=[2025])

    assert replay["yearly"]["2025"]["stage_days"]["early_trial"] == 1
    assert replay["yearly"]["2025"]["stage_days"]["confirming"] == 1
    assert replay["yearly"]["2025"]["stage_days"]["crowded"] == 1


def test_replay_reports_representative_opportunities_by_year() -> None:
    rows = [_row(date="2025-01-01", group="metals", symbol="AU0", bucket="high", score=75.0, count=2)]
    trades = [
        {"entry_time": "2025-01-02 09:00:00", "transition_stage": "confirm_medium_to_high", "pnl_ratio": "0.05"},
        {"entry_time": "2025-03-02 09:00:00", "transition_stage": "late_high", "pnl_ratio": "-0.02"},
    ]

    replay = build_replay_report(rows, trades, years=[2025])

    assert replay["yearly"]["2025"]["representative_opportunities"] == 2
    assert replay["yearly"]["2025"]["profitable_representative_opportunities"] == 1
    assert replay["stage_outcomes"]["confirm_medium_to_high"]["avg_pnl_ratio"] == 0.05


def test_risk_worksheet_uses_minimum_of_margin_risk_and_lifecycle_budget() -> None:
    worksheet = build_risk_worksheet(
        symbol="AU0",
        direction="long",
        equity=1_000_000.0,
        max_margin_pct=0.30,
        risk_pct=0.02,
        entry_price=100.0,
        stop_price=98.0,
        multiplier=10.0,
        margin_rate=0.10,
    )

    assert worksheet["high_conviction"]["suggested_lots"] == 1000
    assert worksheet["high_conviction"]["sizing_limited_by"] == "stop_risk"


def test_risk_worksheet_trial_stage_cannot_exceed_trial_budget() -> None:
    worksheet = build_risk_worksheet(
        symbol="AU0",
        direction="long",
        equity=1_000_000.0,
        max_margin_pct=0.30,
        risk_pct=0.20,
        entry_price=100.0,
        stop_price=99.0,
        multiplier=10.0,
        margin_rate=0.10,
    )

    assert worksheet["trial"]["suggested_lots"] == 500
    assert worksheet["trial"]["margin_usage_pct"] == 0.05


def test_risk_worksheet_confirming_can_use_more_budget_than_trial() -> None:
    worksheet = build_risk_worksheet(
        symbol="AU0",
        direction="long",
        equity=1_000_000.0,
        max_margin_pct=0.30,
        risk_pct=0.20,
        entry_price=100.0,
        stop_price=99.0,
        multiplier=10.0,
        margin_rate=0.10,
    )

    assert worksheet["confirming"]["suggested_lots"] > worksheet["trial"]["suggested_lots"]


def test_risk_worksheet_crowded_stage_reduces_allowed_lots() -> None:
    worksheet = build_risk_worksheet(
        symbol="AU0",
        direction="long",
        equity=1_000_000.0,
        max_margin_pct=0.50,
        risk_pct=0.20,
        entry_price=100.0,
        stop_price=99.0,
        multiplier=10.0,
        margin_rate=0.10,
    )

    assert worksheet["crowded_reduce"]["suggested_lots"] < worksheet["high_conviction"]["suggested_lots"]


def test_validation_labels_future_directional_returns_without_lookahead() -> None:
    rows = [
        _row(date="2025-01-01", group="metals", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2025-01-02", group="metals", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2025-01-03", group="metals", symbol="AU0", bucket="high", score=75.0, count=2),
    ]
    for close, row in zip((100.0, 105.0, 110.0), rows):
        row["close"] = close

    validation = build_effectiveness_validation(rows, years=[2025], horizons=[2], top_n=1)

    labeled = [row for row in validation["labeled_rows"] if row["date"] == "2025-01-01" and row["cohort"] == "radar_top"]
    assert labeled[0]["future_return_2d"] == 0.10
    assert labeled[0]["future_mae_2d"] == 0.05


def test_validation_compares_radar_with_simple_momentum_baseline() -> None:
    rows = [
        _row(date="2025-01-01", group="radar", symbol="AU0", bucket="high", score=80.0, count=2),
        _row(date="2025-01-02", group="radar", symbol="AU0", bucket="high", score=80.0, count=2),
        _row(date="2025-01-03", group="radar", symbol="AU0", bucket="high", score=80.0, count=2),
        _row(date="2025-01-01", group="momentum", symbol="RB0", bucket="low", score=50.0, count=2),
        _row(date="2025-01-02", group="momentum", symbol="RB0", bucket="low", score=50.0, count=2),
        _row(date="2025-01-03", group="momentum", symbol="RB0", bucket="low", score=50.0, count=2),
    ]
    for row in rows:
        if row["group"] == "radar":
            row["close"] = {"2025-01-01": 100.0, "2025-01-02": 108.0, "2025-01-03": 112.0}[row["date"]]
            row["directional_ret_20"] = 0.03
        else:
            row["close"] = {"2025-01-01": 100.0, "2025-01-02": 95.0, "2025-01-03": 94.0}[row["date"]]
            row["directional_ret_20"] = 0.20

    validation = build_effectiveness_validation(rows, years=[2025], horizons=[2], top_n=1)

    assert validation["cohort_summary"]["radar_top"]["avg_future_return_2d"] > 0
    assert validation["cohort_summary"]["momentum_top"]["avg_future_return_2d"] < 0


def test_validation_stage_summary_separates_confirmation_and_crowding() -> None:
    rows = [
        _row(date="2025-01-01", group="confirm", symbol="AU0", bucket="medium", score=60.0, count=2),
        _row(date="2025-01-02", group="confirm", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2025-01-03", group="confirm", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2025-01-01", group="crowded", symbol="SC0", bucket="high", score=78.0, count=2),
        _row(date="2025-01-02", group="crowded", symbol="SC0", bucket="crowded", score=90.0, crowding=90.0, count=2),
        _row(date="2025-01-03", group="crowded", symbol="SC0", bucket="crowded", score=90.0, crowding=90.0, count=2),
    ]
    for row in rows:
        if row["group"] == "confirm":
            row["close"] = {"2025-01-01": 100.0, "2025-01-02": 100.0, "2025-01-03": 110.0}[row["date"]]
        else:
            row["close"] = {"2025-01-01": 100.0, "2025-01-02": 100.0, "2025-01-03": 90.0}[row["date"]]

    validation = build_effectiveness_validation(rows, years=[2025], horizons=[1], top_n=2)

    assert validation["stage_summary"]["confirming"]["avg_future_return_1d"] > 0
    assert validation["stage_summary"]["crowded"]["avg_future_return_1d"] < 0


def test_validation_splits_results_by_year() -> None:
    rows = [
        _row(date="2024-01-01", group="metals", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2024-01-02", group="metals", symbol="AU0", bucket="high", score=75.0, count=2),
        _row(date="2025-01-01", group="energy", symbol="SC0", bucket="high", score=75.0, count=2),
        _row(date="2025-01-02", group="energy", symbol="SC0", bucket="high", score=75.0, count=2),
    ]
    for row in rows:
        row["close"] = 100.0 if row["date"].endswith("01") else 101.0

    validation = build_effectiveness_validation(rows, years=[2024, 2025], horizons=[1], top_n=1)

    assert validation["yearly_summary"]["2024"]["radar_top"]["signals"] == 1
    assert validation["yearly_summary"]["2025"]["radar_top"]["signals"] == 1


def test_validation_markdown_starts_with_business_conclusion() -> None:
    validation = {
        "years": [2025],
        "horizons": [5],
        "cohort_summary": {"radar_top": {"signals": 1, "avg_future_return_5d": 0.02, "positive_rate_5d": 1.0}},
        "stage_summary": {},
        "yearly_summary": {},
    }

    markdown = render_effectiveness_validation_markdown(validation)

    assert markdown.startswith("# 主线雷达有效性验证 2025-2025\n\n## 业务结论")
