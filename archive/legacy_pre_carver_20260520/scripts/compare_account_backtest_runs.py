from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "trade_key",
    "symbol",
    "direction",
    "entry_time",
    "comparison",
    "difference_type",
    "old_present",
    "new_present",
    "old_lots",
    "new_lots",
    "old_exit_time",
    "new_exit_time",
    "old_exit_reason",
    "new_exit_reason",
    "old_net_pnl",
    "new_net_pnl",
    "net_pnl_diff",
    "new_decision_sequence",
    "new_decision",
    "new_skip_reason",
    "new_replacement_count",
    "new_replaced_symbols",
    "new_replacement_block_reason",
    "new_replacement_block_symbol",
    "new_score_lots",
    "new_portfolio_lots",
    "new_risk_lots",
    "new_accepted_lots",
]


def _trade_key(row: dict[str, Any]) -> str:
    return f"{row.get('symbol', '')}|{row.get('direction', '')}|{row.get('entry_time', '')}"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text()
    if not text.strip():
        return []
    with path.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _same_float(first: Any, second: Any, *, tolerance: float = 1e-6) -> bool:
    return abs(_as_float(first) - _as_float(second)) <= tolerance


def _index_by_trade_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _trade_key(row)
        if key not in indexed:
            indexed[key] = row
    return indexed


def _common_difference_type(old: dict[str, Any], new: dict[str, Any]) -> str:
    changes: list[str] = []
    if not _same_float(old.get("lots"), new.get("lots")):
        changes.append("lots_changed")
    exit_changed = (
        str(old.get("actual_exit_time", "")) != str(new.get("actual_exit_time", ""))
        or str(old.get("actual_exit_reason", "")) != str(new.get("actual_exit_reason", ""))
    )
    if exit_changed:
        changes.append("exit_changed")
    if not _same_float(old.get("net_pnl"), new.get("net_pnl")):
        changes.append("pnl_changed")
    return ";".join(changes) if changes else "matched"


def _old_only_difference_type(decision: dict[str, Any] | None) -> str:
    if decision is None:
        return "old_only_no_new_candidate"
    new_decision = str(decision.get("decision", ""))
    if new_decision == "skipped":
        return "old_only_new_skipped"
    if new_decision:
        return f"old_only_new_decision_{new_decision}"
    return "old_only_new_decision_unknown"


def _sort_key(key: str) -> tuple[str, str, str]:
    symbol, direction, entry_time = (key.split("|", 2) + ["", "", ""])[:3]
    return entry_time, symbol, direction


def _build_row(
    *,
    key: str,
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    source = old or new or decision or {}
    if old and new:
        comparison = "common"
        difference_type = _common_difference_type(old, new)
    elif old:
        comparison = "old_only"
        difference_type = _old_only_difference_type(decision)
    else:
        comparison = "new_only"
        difference_type = "new_only"

    return {
        "trade_key": key,
        "symbol": source.get("symbol", ""),
        "direction": source.get("direction", ""),
        "entry_time": source.get("entry_time", ""),
        "comparison": comparison,
        "difference_type": difference_type,
        "old_present": old is not None,
        "new_present": new is not None,
        "old_lots": old.get("lots", "") if old else "",
        "new_lots": new.get("lots", "") if new else "",
        "old_exit_time": old.get("actual_exit_time", "") if old else "",
        "new_exit_time": new.get("actual_exit_time", "") if new else "",
        "old_exit_reason": old.get("actual_exit_reason", "") if old else "",
        "new_exit_reason": new.get("actual_exit_reason", "") if new else "",
        "old_net_pnl": old.get("net_pnl", "") if old else "",
        "new_net_pnl": new.get("net_pnl", "") if new else "",
        "net_pnl_diff": (
            _as_float(new.get("net_pnl")) - _as_float(old.get("net_pnl"))
            if old and new
            else ""
        ),
        "new_decision_sequence": decision.get("sequence", "") if decision else "",
        "new_decision": decision.get("decision", "") if decision else "",
        "new_skip_reason": decision.get("skip_reason", "") if decision else "",
        "new_replacement_count": decision.get("replacement_count", "") if decision else "",
        "new_replaced_symbols": decision.get("replaced_symbols", "") if decision else "",
        "new_replacement_block_reason": decision.get("replacement_block_reason", "") if decision else "",
        "new_replacement_block_symbol": decision.get("replacement_block_symbol", "") if decision else "",
        "new_score_lots": decision.get("score_lots", "") if decision else "",
        "new_portfolio_lots": decision.get("portfolio_lots", "") if decision else "",
        "new_risk_lots": decision.get("risk_lots", "") if decision else "",
        "new_accepted_lots": decision.get("accepted_lots", "") if decision else "",
    }


def reconcile_account_runs(
    *,
    old_trades: list[dict[str, Any]],
    new_trades: list[dict[str, Any]],
    new_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_by_key = _index_by_trade_key(old_trades)
    new_by_key = _index_by_trade_key(new_trades)
    decisions_by_key = _index_by_trade_key(new_decisions)
    keys = sorted(set(old_by_key) | set(new_by_key), key=_sort_key)

    rows = [
        _build_row(
            key=key,
            old=old_by_key.get(key),
            new=new_by_key.get(key),
            decision=decisions_by_key.get(key),
        )
        for key in keys
    ]
    counts = Counter(str(row["difference_type"]) for row in rows)
    first_difference = next((row for row in rows if row["difference_type"] != "matched"), None)
    summary = {
        "old_trades": len(old_trades),
        "new_trades": len(new_trades),
        "new_decisions": len(new_decisions),
        "compared_rows": len(rows),
        "difference_type_counts": dict(sorted(counts.items())),
        "first_difference": first_difference or {},
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, *, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two account-level backtest trade outputs")
    parser.add_argument("--old-trades", type=Path, required=True)
    parser.add_argument("--new-trades", type=Path, required=True)
    parser.add_argument("--new-decisions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows, summary = reconcile_account_runs(
        old_trades=_load_csv_rows(args.old_trades),
        new_trades=_load_csv_rows(args.new_trades),
        new_decisions=_load_csv_rows(args.new_decisions),
    )
    _write_csv(args.output_csv, rows)
    _write_json(args.output_json, rows=rows, summary=summary)
    print(
        f"compared={summary['compared_rows']} "
        f"old={summary['old_trades']} new={summary['new_trades']} "
        f"differences={summary['difference_type_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
