from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from data_cache import get_all_symbols
from mainline_event_strategy import MainlineEventParams, run_event_strategy
from market.fundamental_universe import FUNDAMENTAL_GROUPS


DEFAULT_STATES = Path("data/reports/mainline_radar/weekly_mainline_hold_until_lost_states_2022_2026.csv")
DEFAULT_OUTPUT_DIR = Path("data/reports/mainline_radar")
DEFAULT_PREFIX = "mainline_event_strategy_2022_2026"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pct(value: Any, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _year(value: Any) -> str:
    return str(pd.Timestamp(value).year)


def _name_map() -> dict[str, str]:
    return {str(item["symbol"]).upper(): str(item["name"]) for item in get_all_symbols()}


def _group_name(group: str) -> str:
    profile = FUNDAMENTAL_GROUPS.get(str(group), None)
    return profile.display_name if profile else str(group)


def _symbol_label(symbol: str, names: dict[str, str]) -> str:
    normalized = str(symbol or "").upper()
    name = names.get(normalized, normalized)
    return f"{name}({normalized})"


def _direction_text(direction: str) -> str:
    return "多头" if str(direction) == "long" else "空头"


def _state_text(state: str) -> str:
    return {
        "sprouting": "主线萌芽/小仓试错",
        "confirmed": "主线确认/加仓",
        "markup": "主升/集中持有",
        "crowded": "拥挤/降仓",
        "ended": "主线失效",
        "none": "无主线",
    }.get(str(state), str(state))


def _exit_text(reason: str) -> str:
    return {
        "mainline_lost": "主线失效退出",
        "mainline_weakening": "主线转弱退出",
        "mainline_crowded": "拥挤后退出",
        "rotated_to_stronger_mainline": "切换到更强主线",
        "trial_not_confirmed": "试错未确认退出",
        "end_of_data": "样本结束",
    }.get(str(reason), str(reason))


def _compound(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= max(0.0, 1.0 + float(value))
    return equity - 1.0


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        equity *= max(0.0, 1.0 + float(value))
        peak = max(peak, equity)
        dd = equity / peak - 1.0 if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    return abs(max_dd)


def _yearly_rows(daily_rows: list[dict[str, Any]], trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trade_counter = Counter(_year(trade.get("entry_date")) for trade in trades if str(trade.get("entry_date") or ""))
    out: list[dict[str, Any]] = []
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        by_year[_year(row.get("date"))].append(row)
    for year, rows in sorted(by_year.items()):
        returns = [float(row.get("net_return") or 0.0) for row in rows]
        active = [row for row in rows if int(row.get("active_positions") or 0) > 0]
        out.append(
            {
                "year": year,
                "return": _compound(returns),
                "max_drawdown": _max_drawdown(returns),
                "active_day_ratio": len(active) / max(len(rows), 1),
                "avg_exposure": sum(float(row.get("exposure") or 0.0) for row in rows) / max(len(rows), 1),
                "max_exposure": max((float(row.get("exposure") or 0.0) for row in rows), default=0.0),
                "trades": trade_counter.get(year, 0),
                "trading_days": len(rows),
            }
        )
    return out


def _state_mix(daily_rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("state") or "cash") for row in daily_rows))


def _exit_mix(trades: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(trade.get("exit_reason") or "") for trade in trades))


def _entry_mix(trades: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(trade.get("entry_state") or "") for trade in trades))


def _symbol_contribution(daily_rows: list[dict[str, Any]], names: dict[str, str]) -> list[dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = defaultdict(lambda: {"days": 0, "net_return_sum": 0.0, "max_exposure": 0.0})
    for row in daily_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        item = by_symbol[symbol]
        item["days"] += 1
        item["net_return_sum"] += float(row.get("net_return") or 0.0)
        item["max_exposure"] = max(float(item["max_exposure"]), float(row.get("exposure") or 0.0))
    rows = [
        {
            "symbol": symbol,
            "name": names.get(symbol, symbol),
            "label": _symbol_label(symbol, names),
            **values,
        }
        for symbol, values in by_symbol.items()
    ]
    return sorted(rows, key=lambda row: abs(float(row["net_return_sum"])), reverse=True)


def _trade_rows_with_names(trades: list[dict[str, Any]], names: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in trades:
        symbol = str(trade.get("entry_symbol") or "").upper()
        out.append(
            {
                **trade,
                "name": names.get(symbol, symbol),
                "entry_state_text": _state_text(str(trade.get("entry_state") or "")),
                "exit_reason_text": _exit_text(str(trade.get("exit_reason") or "")),
                "direction_text": _direction_text(str(trade.get("direction") or "")),
                "board_text": _group_name(str(trade.get("board") or "")),
            }
        )
    return out


def _variant_params() -> dict[str, MainlineEventParams]:
    return {
        "trial_then_concentrate": MainlineEventParams(
            sprouting_exposure=0.50,
            confirmed_exposure=1.50,
            markup_exposure=3.00,
            crowded_exposure=0.50,
        ),
        "wait_confirmed": MainlineEventParams(
            sprouting_exposure=0.00,
            confirmed_exposure=1.50,
            markup_exposure=3.00,
            crowded_exposure=0.50,
        ),
        "conservative_trial": MainlineEventParams(
            sprouting_exposure=0.25,
            confirmed_exposure=1.00,
            markup_exposure=2.00,
            crowded_exposure=0.25,
        ),
        "ultra_selective_1x": MainlineEventParams(
            sprouting_exposure=0.00,
            confirmed_exposure=0.50,
            markup_exposure=1.00,
            crowded_exposure=0.00,
            min_confirmed_score=72.0,
            min_markup_score=85.0,
            cooldown_days=15,
        ),
        "aggressive_mainline": MainlineEventParams(
            sprouting_exposure=0.50,
            confirmed_exposure=2.00,
            markup_exposure=5.00,
            crowded_exposure=0.50,
        ),
    }


def _summary_row(variant: str, params: MainlineEventParams, result: Any) -> dict[str, Any]:
    row = {
        "variant": variant,
        **result.summary,
        "params": json.dumps(asdict(params), ensure_ascii=False, sort_keys=True),
        "entry_mix": json.dumps(_entry_mix(result.trades), ensure_ascii=False, sort_keys=True),
        "exit_mix": json.dumps(_exit_mix(result.trades), ensure_ascii=False, sort_keys=True),
        "state_mix": json.dumps(_state_mix(result.daily_rows), ensure_ascii=False, sort_keys=True),
    }
    return row


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _load_baseline(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "weekly_mainline_hold_until_lost_backtest_2022_2026_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results") or {}


def _build_report(
    *,
    states_path: Path,
    output_dir: Path,
    prefix: str,
    names: dict[str, str],
    variants: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# 主线事件策略回测 2022-2026",
        "",
        "## 业务结论",
        "",
        "- 这次验证的是“盘面主线事件策略”，不是实盘默认策略：确认前可以小仓试错，主线确认后加仓，主升阶段集中持有，拥挤后降仓或退出，同时最多只持有一条主线。",
        "- 回测使用上一轮已生成的 59 个连续主力品种主线状态文件；本脚本不重新联网抓数据，也不修改 Phase2、Phase4 或实盘配置。",
        "- 关键验收不是看它是否年化最高，而是看它有没有做到更少交易、更集中、更符合主线生命周期，并且账户收益/回撤是否比“周度主线见到就买”更合理。",
        "",
    ]
    summary_rows: list[list[str]] = []
    for variant, payload in variants.items():
        summary = payload["summary"]
        entry = payload["entry_mix"]
        exit_mix = payload["exit_mix"]
        state_mix = payload["state_mix"]
        summary_rows.append(
            [
                variant,
                _pct(summary["total_return"]),
                _pct(summary["annualized_return"]),
                _pct(summary["max_drawdown"]),
                str(summary["trades"]),
                _pct(summary["active_day_ratio"]),
                _num(summary["avg_exposure"], 2),
                _num(summary["max_exposure"], 2),
                str(summary["max_simultaneous_positions"]),
                str(entry),
                str(exit_mix),
                str(state_mix),
            ]
        )
    lines.extend(
        _markdown_table(
            ["口径", "总收益", "年化", "最大回撤", "交易数", "有仓天数", "平均仓位", "最高仓位", "同时持仓", "入场状态", "退出原因", "持仓状态天数"],
            summary_rows,
        )
    )
    lines.extend(["", "## 年度拆分", ""])
    for variant, payload in variants.items():
        lines.append(f"### {variant}")
        yearly = payload["yearly"]
        lines.extend(
            _markdown_table(
                ["年份", "收益", "年内最大回撤", "有仓天数", "平均仓位", "最高仓位", "开仓数"],
                [
                    [
                        str(row["year"]),
                        _pct(row["return"]),
                        _pct(row["max_drawdown"]),
                        _pct(row["active_day_ratio"]),
                        _num(row["avg_exposure"], 2),
                        _num(row["max_exposure"], 2),
                        str(row["trades"]),
                    ]
                    for row in yearly
                ],
            )
        )
        lines.append("")
    best_variant = max(variants, key=lambda key: float(variants[key]["summary"]["annualized_return"]))
    base_variant = "trial_then_concentrate"
    base = variants[base_variant]
    lines.extend(
        [
            "## 是否实现了交易习惯",
            "",
            f"- 同时只抓一条主线：`{base_variant}` 的最大同时持仓为 {base['summary']['max_simultaneous_positions']}。",
            f"- 交易频率明显下降：`{base_variant}` 有仓天数 {_pct(base['summary']['active_day_ratio'])}，交易 {base['summary']['trades']} 笔；此前严格 Top3 周度持有有仓天数约 98%，开仓 1208 次。",
            f"- 仓位生命周期已经落地：入场状态分布为 {base['entry_mix']}；持仓状态天数为 {base['state_mix']}；退出原因分布为 {base['exit_mix']}。",
            f"- 当前参数里收益最高的是 `{best_variant}`，但它可能只是放大杠杆后的结果，需要结合年内最大回撤和最高仓位一起看。",
            "",
        ]
    )

    baseline = _load_baseline(output_dir)
    if baseline:
        strict = baseline.get("strict_top3_stacked_100pct_each") or {}
        equal = baseline.get("strict_top3_equal_weight") or {}
        if strict and equal:
            lines.extend(
                [
                    "## 与旧周度持有口径对照",
                    "",
                    "- 旧口径回答“周度主线雷达有没有方向性”；新口径回答“只抓一条主线并按生命周期加减仓，能不能像真实交易”。",
                    f"- 旧严格 Top3 等权：总收益 {_pct(equal.get('total_return', 0.0))}，年化 {_pct(equal.get('annualized_return', 0.0))}，最大回撤 {_pct(abs(float(equal.get('max_drawdown', 0.0))))}。",
                    f"- 旧严格 Top3 每条 100% 叠加：总收益 {_pct(strict.get('total_return', 0.0))}，年化 {_pct(strict.get('annualized_return', 0.0))}，最大回撤 {_pct(abs(float(strict.get('max_drawdown', 0.0))))}。",
                    f"- 新 `{base_variant}`：总收益 {_pct(base['summary']['total_return'])}，年化 {_pct(base['summary']['annualized_return'])}，最大回撤 {_pct(base['summary']['max_drawdown'])}。",
                    "",
                ]
            )

    lines.extend(["## 主要持仓贡献", ""])
    for variant, payload in variants.items():
        lines.append(f"### {variant}")
        top_symbols = payload["symbol_contribution"][:8]
        lines.extend(
            _markdown_table(
                ["品种", "持仓天数", "收益贡献近似", "最高仓位"],
                [
                    [
                        str(row["label"]),
                        str(row["days"]),
                        _pct(row["net_return_sum"]),
                        _num(row["max_exposure"], 2),
                    ]
                    for row in top_symbols
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 风险和边界",
            "",
            "- 这个版本是研究用账户回测，仓位是名义风险倍数，不代表真实保证金、止损和滑点后的可下单手数。",
            "- 手续费按本地合约费率 * 1.01 扣除；本地缺失费率的品种在明细里标记，缺失部分仍可能高估收益。",
            "- 主线识别只用价格、成交量、持仓、板块共振等盘面数据，没有接入真实基本面和舆论数据；它只能验证“市场是否正在定价主线”，不能解释主线原因。",
            "- 当前策略没有 Phase2 具体入场点，等于主线状态确认后用收盘价持有；如果后续要进入实盘候选，必须再叠加短线触发、止损和保证金预算。",
            "",
            "## 输出文件",
            "",
            f"- 汇总 CSV：{output_dir / (prefix + '_summary.csv')}",
            f"- 汇总 JSON：{output_dir / (prefix + '_summary.json')}",
            f"- 报告：{output_dir / (prefix + '_report.md')}",
            f"- 每个口径的日收益和交易明细：`{output_dir / (prefix + '_<variant>_daily.csv')}` / `..._trades.csv`",
            f"- 输入状态文件：{states_path}",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(states_path: Path, output_dir: Path, prefix: str) -> dict[str, Any]:
    state_rows = _read_csv(states_path)
    names = _name_map()
    variants: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for variant, params in _variant_params().items():
        result = run_event_strategy(state_rows, params)
        daily_path = output_dir / f"{prefix}_{variant}_daily.csv"
        trades_path = output_dir / f"{prefix}_{variant}_trades.csv"
        _write_csv(daily_path, result.daily_rows)
        trades_with_names = _trade_rows_with_names(result.trades, names)
        _write_csv(trades_path, trades_with_names)
        yearly = _yearly_rows(result.daily_rows, result.trades)
        symbol_contribution = _symbol_contribution(result.daily_rows, names)
        entry_mix = _entry_mix(result.trades)
        exit_mix = _exit_mix(result.trades)
        state_mix = _state_mix(result.daily_rows)
        variants[variant] = {
            "params": asdict(params),
            "summary": result.summary,
            "yearly": yearly,
            "entry_mix": entry_mix,
            "exit_mix": exit_mix,
            "state_mix": state_mix,
            "symbol_contribution": symbol_contribution,
            "daily_csv": str(daily_path),
            "trades_csv": str(trades_path),
        }
        summary_rows.append(_summary_row(variant, params, result))

    _write_csv(output_dir / f"{prefix}_summary.csv", summary_rows)
    payload = {
        "input_states": str(states_path),
        "variants": variants,
    }
    _write_json(output_dir / f"{prefix}_summary.json", payload)
    report = _build_report(
        states_path=states_path,
        output_dir=output_dir,
        prefix=prefix,
        names=names,
        variants=variants,
    )
    (output_dir / f"{prefix}_report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    payload = run_analysis(args.states, args.output_dir, args.prefix)
    report_path = args.output_dir / f"{args.prefix}_report.md"
    print(f"wrote {report_path}")
    for variant, data in payload["variants"].items():
        summary = data["summary"]
        print(
            variant,
            f"annualized={summary['annualized_return']:.4f}",
            f"total={summary['total_return']:.4f}",
            f"max_dd={summary['max_drawdown']:.4f}",
            f"trades={summary['trades']}",
            f"active={summary['active_day_ratio']:.4f}",
        )


if __name__ == "__main__":
    main()
