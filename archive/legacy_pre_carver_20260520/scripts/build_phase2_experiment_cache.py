from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from backtest.cases import get_case_group
from backtest.experiment_cache import (
    build_phase2_experiment_cache,
    compare_phase2_cache_to_fresh_build,
    read_phase2_experiment_cache,
    write_phase2_experiment_cache,
)
from backtest.phase23 import load_case_frames_with_tqbacktest
from run_account_backtest import _config_with_fundamental_mode, _resolved_fundamental_mode, load_config
from run_backtest import _plan_factory_for_case


SUMMARY_FIELDS = [
    "case_id",
    "symbol",
    "direction",
    "entries",
    "actionable_plans",
    "mismatches",
    "cache_path",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build backtest-only Phase2 experiment cache")
    parser.add_argument("--case-group", default="long_trend_core")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--fundamental-mode", choices=("strict", "proxy"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _case_for_year(case, year: int):
    return replace(
        case,
        case_id=f"{case.case_id}_{year}",
        start_dt=datetime(int(year), 1, 1).date(),
        end_dt=datetime(int(year), 12, 31).date(),
    )


def _default_output_dir(year: int) -> Path:
    return Path("data/reports/backtest") / f"phase2_experiment_cache_{year}"


def _case_output_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / f"{case_id}_phase2_cache.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select_cases(case_group: str, case_limit: int | None):
    cases = get_case_group(case_group)
    if case_limit is not None:
        cases = cases[: max(int(case_limit), 0)]
    return cases


def _summary_row(*, run_case, cache_path: Path, loaded_cache, mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": run_case.case_id,
        "symbol": run_case.symbol,
        "direction": run_case.direction,
        "entries": len(loaded_cache.entries),
        "actionable_plans": sum(1 for entry in loaded_cache.entries if entry.plan is not None),
        "mismatches": len(mismatches),
        "cache_path": str(cache_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    loaded_config = load_config()
    config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )
    pre_market_cfg = config.get("pre_market") or {}
    output_dir = args.output_dir or _default_output_dir(args.year)
    rows: list[dict[str, Any]] = []

    cases = _select_cases(args.case_group, args.case_limit)
    if not args.quiet:
        print(
            f"building Phase2 backtest cache: case_group={args.case_group} "
            f"year={args.year} cases={len(cases)}",
            flush=True,
        )
    for case in cases:
        run_case = _case_for_year(case, args.year)
        if not args.quiet:
            print(f"case {case.case_id}: load frames", flush=True)
        daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
        plan_factory = _plan_factory_for_case(run_case)
        cache = build_phase2_experiment_cache(
            case=run_case,
            daily_df=daily_df,
            minute_df=minute_df,
            pre_market_cfg=pre_market_cfg,
            plan_factory=plan_factory,
        )
        cache_path = _case_output_path(output_dir, run_case.case_id)
        write_phase2_experiment_cache(cache, cache_path)
        loaded_cache = read_phase2_experiment_cache(cache_path)
        mismatches = compare_phase2_cache_to_fresh_build(
            loaded_cache,
            case=run_case,
            daily_df=daily_df,
            minute_df=minute_df,
            pre_market_cfg=pre_market_cfg,
            plan_factory=plan_factory,
        )
        rows.append(
            _summary_row(
                run_case=run_case,
                cache_path=cache_path,
                loaded_cache=loaded_cache,
                mismatches=mismatches,
            )
        )
        if not args.quiet:
            print(
                f"case {run_case.case_id}: entries={len(loaded_cache.entries)} "
                f"actionable={rows[-1]['actionable_plans']} mismatches={len(mismatches)}",
                flush=True,
            )

    summary_csv = output_dir / "phase2_cache_summary.csv"
    summary_json = output_dir / "phase2_cache_summary.json"
    _write_csv(summary_csv, rows)
    _write_json(summary_json, rows)
    print(f"wrote: {summary_csv}", flush=True)
    print(f"wrote: {summary_json}", flush=True)
    return 1 if any(int(row["mismatches"]) > 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
