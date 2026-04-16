#!/usr/bin/env python3
"""
并行对比 Phase 1：legacy vs regime（不改变 config）

用法（在 scripts 目录）:
  python compare_p1_engines.py
  python compare_p1_engines.py --symbols LH0,M0

依赖与 daily_workflow 相同：需已缓存日线或能拉取。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from data_cache import get_all_symbols, get_daily, prefetch_all
from fundamental_legacy import score_symbol_legacy
from fundamental_regime import score_symbol_regime


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="对比 P1 legacy vs regime")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="逗号分隔，如 LH0,M0；默认全市场有缓存品种",
    )
    args = parser.parse_args()

    cfg = load_config()
    fs = cfg.get("fundamental_screening") or {}
    default_thr = float(fs.get("default_threshold") or 10)
    sym_to_threshold: dict[str, float] = {}
    for cat_cfg in (fs.get("categories") or {}).values():
        if not isinstance(cat_cfg, dict):
            continue
        t = cat_cfg.get("threshold")
        if t is None:
            continue
        for s in cat_cfg.get("symbols") or []:
            sym_to_threshold[str(s)] = float(t)

    if args.symbols.strip():
        want = {x.strip() for x in args.symbols.split(",") if x.strip()}
        symbols = [s for s in get_all_symbols() if s["symbol"] in want]
        if not symbols:
            print("未找到指定品种", file=sys.stderr)
            sys.exit(1)
    else:
        symbols = get_all_symbols()

    print("预加载日线缓存…")
    all_data = prefetch_all(symbols)

    print("\n" + "=" * 100)
    print(f"{'品种':^8} {'thr':^5} | {'legacy':^8} {'regime':^8} {'Δ':^6} | {'区间%':^6} | regime标签")
    print("=" * 100)

    for info in symbols:
        sym = info["symbol"]
        df = all_data.get(sym)
        if df is None or len(df) < 60:
            print(f"{sym:8} 数据不足(<60日)，跳过")
            continue
        thr = sym_to_threshold.get(sym, default_thr)
        leg = score_symbol_legacy(sym, info["name"], info["exchange"], df, threshold=thr)
        reg = score_symbol_regime(sym, info["name"], info["exchange"], df, threshold=thr)
        ls = leg.get("score") if leg else None
        rs = reg.get("score") if reg else None
        delta = (rs - ls) if (ls is not None and rs is not None) else None
        rp = (leg or reg or {}).get("range_pct")
        lr = reg.get("lh_regime") if reg and sym == "LH0" else "-"
        dstr = f"{delta:+.0f}" if delta is not None else "  -  "
        print(
            f"{sym:^8} {thr:^5.0f} | {ls:^8.0f} {rs:^8.0f} {dstr:^6} | "
            f"{rp:^6.0f} | {lr}"
        )

    print("=" * 100)
    print("说明: 生产环境仍由 config fundamental_screening.p1_engine 控制；本脚本仅对照。")


if __name__ == "__main__":
    main()
