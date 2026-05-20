from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.tq_source import (  # noqa: E402
    discover_main_continuous_instruments,
    fetch_tq_daily_with_api,
    filter_universe,
    load_tq_credentials,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download TqSdk main-continuous daily futures data for systematic trading")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "systematic" / "tq_main_daily"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--static-universe", action="store_true", help="Use the built-in fallback universe instead of TqSdk discovery")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--days", type=int, default=1400)
    parser.add_argument("--wait-timeout", type=float, default=8.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    account, password = load_tq_credentials(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)

    from tqsdk import TqApi, TqAuth

    api = TqApi(auth=TqAuth(account, password))
    try:
        if args.static_universe:
            instruments = filter_universe(args.symbols)
        else:
            discovered = discover_main_continuous_instruments(api)
            if args.symbols:
                wanted = {symbol.upper() for symbol in args.symbols}
                instruments = [item for item in discovered if item["symbol"].upper() in wanted]
            else:
                instruments = discovered
        pd.DataFrame(instruments).to_csv(output_dir / "tq_main_daily_universe.csv", index=False)
        print(f"universe={len(instruments)} source={'static' if args.static_universe else 'tqsdk_query_cont_quotes'}")
        for index, instrument in enumerate(instruments, start=1):
            symbol = instrument["symbol"]
            try:
                frame = fetch_tq_daily_with_api(api, instrument, days=args.days, wait_timeout=args.wait_timeout)
                if not frame.empty:
                    frame = frame.loc[frame["date"] >= start].reset_index(drop=True)
                out_path = output_dir / f"{symbol}_tq_main_daily.parquet"
                frame.to_parquet(out_path, index=False)
                if frame.empty:
                    print(f"[{index}/{len(instruments)}] {symbol} no data")
                else:
                    print(
                        f"[{index}/{len(instruments)}] {symbol} {frame['date'].min().date()} -> {frame['date'].max().date()} rows={len(frame)}"
                    )
            except Exception as exc:
                print(f"[{index}/{len(instruments)}] {symbol} failed: {exc}")
    finally:
        api.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
