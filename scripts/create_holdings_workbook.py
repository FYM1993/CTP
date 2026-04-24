#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from holdings_advice.workbook import ensure_daily_holdings_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description="创建或续写每日持仓 workbook")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--root-dir",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "holdings"),
    )
    args = parser.parse_args()

    output = ensure_daily_holdings_workbook(
        root_dir=Path(args.root_dir),
        trade_date=args.date,
    )
    print(output)


if __name__ == "__main__":
    main()
