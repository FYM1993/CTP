#!/usr/bin/env python3
"""
期货量化研究流程
================

基于 vnpy Alpha 模块的期货量化研究和回测系统。

使用方法：
    # 使用默认配置
    python scripts/research_workflow.py

    # 使用自定义配置
    python scripts/research_workflow.py my_config.yaml

流程：
    1. 准备连续合约数据
    2. 计算因子特征
    3. 训练 LightGBM 模型
    4. 生成预测信号
    5. 因子分析（Alphalens）
    6. 回测
    7. 输出结果
"""

import sys
from pathlib import Path
from datetime import datetime
from functools import partial

import yaml
import polars as pl
import numpy as np

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from vnpy.alpha import AlphaLab, AlphaDataset
from vnpy.alpha.dataset import (
    process_drop_na,
    process_cs_norm,
    Segment,
)

from src.continuous import load_continuous, prepare_all_continuous
from src.alpha_futures import FuturesAlpha158
from src.model_lgb import LGBModel


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_config(config: dict) -> None:
    """打印配置摘要"""
    seg = config["segments"]
    model_cfg = config["model"]["lightgbm"]

    print("=" * 60)
    print("📈 期货量化研究")
    print("=" * 60)
    print(f"品种:     {', '.join(config['symbols'])}")
    print(f"连续方式: {config.get('continuous_method', 'volume')}")
    print(f"因子集:   {config['factors']['factor_set']}")
    print(f"标签:     {config['factors']['label']}")
    print()
    print("📅 时间配置:")
    print(f"  训练集: {seg['train'][0]} ~ {seg['train'][1]}")
    print(f"  验证集: {seg['valid'][0]} ~ {seg['valid'][1]}")
    print(f"  测试集: {seg['test'][0]} ~ {seg['test'][1]}")
    print()
    print("🤖 模型参数:")
    print(f"  类型:     {config['model']['type']}")
    print(f"  学习率:   {model_cfg['learning_rate']}")
    print(f"  迭代次数: {model_cfg['n_estimators']}")
    print(f"  最大深度: {model_cfg['max_depth']}")
    print()
    print("💰 回测配置:")
    bt = config["backtest"]
    print(f"  初始资金: {bt['capital']:,.0f}")
    print(f"  方向:     {bt['direction']}")
    print(f"  手续费:   {bt['commission_rate']:.4%}")
    print("=" * 60)


def prepare_data(config: dict) -> dict[str, pl.DataFrame]:
    """
    步骤1: 准备连续合约数据

    如果连续合约文件不存在，自动从原始数据构建。
    """
    print("\n🔧 步骤1: 准备连续合约数据")
    print("-" * 40)

    data_dir = config["data"]["data_dir"]
    symbols = config["symbols"]

    continuous_data = {}

    for symbol in symbols:
        try:
            df = load_continuous(data_dir, symbol)
            continuous_data[symbol] = df
            print(f"  ✅ {symbol}: {len(df)} 天")
        except FileNotFoundError:
            print(f"  ⚠️ {symbol} 连续合约文件不存在，尝试构建...")

    # 如果有品种缺失，尝试构建
    missing = [s for s in symbols if s not in continuous_data]
    if missing:
        print(f"\n  构建缺失品种: {missing}")
        new_data = prepare_all_continuous(
            data_dir=data_dir,
            symbols=missing,
            method=config.get("continuous_method", "volume"),
            switch_threshold=config.get("switch_threshold", 0.8),
        )
        continuous_data.update(new_data)

    return continuous_data


def load_data_to_lab(
    continuous_data: dict[str, pl.DataFrame],
    config: dict,
) -> tuple[AlphaLab, list[str]]:
    """
    步骤2: 将连续合约数据加载到 AlphaLab

    将 DataFrame 格式的数据转换为 vnpy AlphaLab 所需的 parquet 文件格式。
    """
    print("\n📊 步骤2: 加载数据到 AlphaLab")
    print("-" * 40)

    lab_path = config["data"].get("lab_path", "./data/lab/futures")
    lab = AlphaLab(lab_path)

    symbols = []
    for symbol, df in continuous_data.items():
        vt_symbol = f"{symbol}00"  # 连续合约代号

        # 转换为 BarData 并保存
        from vnpy.trader.object import BarData
        from vnpy.trader.constant import Exchange, Interval

        # 识别交易所
        exchange_map = {
            "RB": Exchange.SHFE, "CU": Exchange.SHFE, "AU": Exchange.SHFE,
            "IF": Exchange.CFFEX, "IC": Exchange.CFFEX, "IH": Exchange.CFFEX,
            "M": Exchange.DCE, "P": Exchange.DCE, "I": Exchange.DCE,
            "SR": Exchange.CZCE, "CF": Exchange.CZCE, "TA": Exchange.CZCE,
            "SC": Exchange.INE, "NR": Exchange.INE,
        }
        exchange = exchange_map.get(symbol, Exchange.SHFE)

        bars = []
        for row in df.iter_rows(named=True):
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=Interval.DAILY,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row.get("volume", 0)),
                turnover=float(row.get("turnover", 0)),
                open_interest=float(row.get("open_interest", 0)),
                gateway_name="DB",
            )
            bars.append(bar)

        if bars:
            lab.save_bar_data(bars)
            symbols.append(vt_symbol)
            print(f"  ✅ {vt_symbol}: {len(bars)} 条数据")

        # 保存合约规格
        contract = config["contracts"].get(symbol, {})
        lab.add_contract_setting(
            vt_symbol=vt_symbol,
            long_rate=contract.get("long_rate", 0.0001),
            short_rate=contract.get("short_rate", 0.0001),
            size=contract.get("multiplier", 1),
            pricetick=contract.get("pricetick", 1),
        )

    return lab, symbols


def build_dataset(
    lab: AlphaLab,
    vt_symbols: list[str],
    config: dict,
) -> AlphaDataset:
    """
    步骤3: 构建因子数据集

    使用 FuturesAlpha158 计算所有因子。
    """
    print("\n📐 步骤3: 构建因子数据集")
    print("-" * 40)

    seg = config["segments"]
    extended_days = config.get("extended_days", 100)

    # 加载所有品种数据合并为一个 DataFrame
    print("  加载行情数据...")
    all_dfs = []

    for vt_symbol in vt_symbols:
        df = lab.load_bar_df(
            [vt_symbol],
            "d",
            seg["train"][0],
            seg["test"][1],
            extended_days=extended_days,
        )
        if df is not None and not df.is_empty():
            all_dfs.append(df)

    merged_df = pl.concat(all_dfs) if all_dfs else None

    if merged_df is None or merged_df.is_empty():
        raise RuntimeError("无法加载任何数据，请检查数据目录")

    print(f"  数据量: {len(merged_df)} 行, {merged_df['vt_symbol'].n_unique()} 个品种")

    # 创建因子数据集
    factor_set = config["factors"]["factor_set"]
    futures_windows = config["factors"].get("futures_windows", [5, 10, 20])
    label_type = config["factors"].get("label", "ret_3")

    print(f"  因子集: {factor_set}")
    print(f"  标签: {label_type}")

    if factor_set == "alpha158_futures":
        dataset = FuturesAlpha158(
            df=merged_df,
            train_period=tuple(seg["train"]),
            valid_period=tuple(seg["valid"]),
            test_period=tuple(seg["test"]),
            futures_windows=futures_windows,
            label_type=label_type,
        )
    else:
        # 使用标准 Alpha158
        from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
        dataset = Alpha158(
            df=merged_df,
            train_period=tuple(seg["train"]),
            valid_period=tuple(seg["valid"]),
            test_period=tuple(seg["test"]),
        )

    # 添加预处理器
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

    # 计算因子
    print("  计算因子特征...")
    max_workers = config.get("parallel", {}).get("max_workers", 4)
    dataset.prepare_data(max_workers=max_workers)

    print(f"  ✅ 特征维度: {len(dataset.raw_df.columns) - 3}")  # 减去 datetime, vt_symbol, label

    return dataset


def train_model(dataset: AlphaDataset, config: dict) -> LGBModel:
    """
    步骤4: 训练模型
    """
    print("\n🤖 步骤4: 训练 LightGBM 模型")
    print("-" * 40)

    model_cfg = config["model"]["lightgbm"]
    model = LGBModel(**model_cfg)

    # 数据处理
    dataset.process_data()

    # 训练
    model.fit(dataset)

    # 详情
    detail = model.detail()
    if detail:
        print(f"  最佳迭代轮次: {detail.get('best_iteration', 'N/A')}")

    return model


def generate_signals(
    dataset: AlphaDataset,
    model: LGBModel,
    config: dict,
) -> pl.DataFrame:
    """
    步骤5: 生成预测信号
    """
    print("\n📡 步骤5: 生成预测信号")
    print("-" * 40)

    # 在测试集上预测
    predictions = model.predict(dataset, Segment.TEST)

    # 获取测试集的 datetime 和 vt_symbol
    test_df = dataset.fetch_learn(Segment.TEST)

    signal_df = test_df.select(["datetime", "vt_symbol"]).clone()
    signal_df = signal_df.with_columns(pl.Series("signal", predictions))

    # 去除 NaN
    signal_df = signal_df.filter(pl.col("signal").is_not_nan())

    print(f"  信号数量: {len(signal_df)}")
    print(f"  信号均值: {signal_df['signal'].mean():.6f}")
    print(f"  信号标准差: {signal_df['signal'].std():.6f}")

    return signal_df


def run_backtest(
    lab: AlphaLab,
    vt_symbols: list[str],
    signal_df: pl.DataFrame,
    config: dict,
) -> dict:
    """
    步骤6: 回测
    """
    print("\n💰 步骤6: 回测")
    print("-" * 40)

    from vnpy.alpha.strategy import BacktestingEngine
    from vnpy.trader.constant import Interval

    seg = config["segments"]
    bt_cfg = config["backtest"]

    # 创建回测引擎
    engine = BacktestingEngine(lab)

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(seg["test"][0], "%Y-%m-%d"),
        end=datetime.strptime(seg["test"][1], "%Y-%m-%d"),
        capital=bt_cfg["capital"],
        risk_free=bt_cfg.get("risk_free", 0.03),
        annual_days=bt_cfg.get("annual_days", 240),
    )

    # 定义期货信号策略
    class FuturesSignalStrategy:
        """基于信号的期货策略"""

        def __init__(
            self,
            strategy_engine,
            strategy_name: str,
            vt_symbols: list[str],
            setting: dict,
        ):
            self.strategy_engine = strategy_engine
            self.strategy_name = strategy_name
            self.vt_symbols = vt_symbols
            self.pos_data = {}
            self.target_data = {}
            self.orders = {}
            self.active_orderids = set()

            for k, v in setting.items():
                if hasattr(self, k):
                    setattr(self, k, v)

            self.direction = setting.get("direction", "both")
            self.max_position = setting.get("max_position", 5)
            self.long_threshold = setting.get("long_threshold", 0.005)
            self.short_threshold = setting.get("short_threshold", -0.005)

        def on_init(self) -> None:
            pass

        def on_bars(self, bars: dict) -> None:
            signal_df = self.get_signal()

            for vt_symbol, bar in bars.items():
                # 获取当前信号
                day_signals = signal_df.filter(
                    (pl.col("datetime") == bar.datetime)
                    & (pl.col("vt_symbol") == vt_symbol)
                )

                if day_signals.is_empty():
                    continue

                signal = day_signals["signal"][0]
                pos = self.pos_data.get(vt_symbol, 0)

                # 根据信号调整目标仓位
                if self.direction in ("long", "both") and signal > self.long_threshold:
                    self.set_target(vt_symbol, self.max_position)
                elif self.direction in ("short", "both") and signal < self.short_threshold:
                    self.set_target(vt_symbol, -self.max_position)
                else:
                    self.set_target(vt_symbol, 0)

                # 执行交易
                target = self.target_data.get(vt_symbol, 0)
                diff = target - pos

                if diff > 0:
                    price = bar.close_price * (1 + bt_cfg.get("slippage", 0.0005))
                    if pos < 0:
                        cover_vol = min(diff, abs(pos))
                        self.cover(vt_symbol, price, cover_vol)
                        buy_vol = diff - cover_vol
                        if buy_vol > 0:
                            self.buy(vt_symbol, price, buy_vol)
                    else:
                        self.buy(vt_symbol, price, diff)
                elif diff < 0:
                    price = bar.close_price * (1 - bt_cfg.get("slippage", 0.0005))
                    if pos > 0:
                        sell_vol = min(abs(diff), pos)
                        self.sell(vt_symbol, price, sell_vol)
                        short_vol = abs(diff) - sell_vol
                        if short_vol > 0:
                            self.short(vt_symbol, price, short_vol)
                    else:
                        self.short(vt_symbol, price, abs(diff))

        def on_trade(self, trade) -> None:
            pass

        def buy(self, vt_symbol, price, volume):
            return self.send_order(vt_symbol, "LONG", "OPEN", price, volume)

        def sell(self, vt_symbol, price, volume):
            return self.send_order(vt_symbol, "SHORT", "CLOSE", price, volume)

        def short(self, vt_symbol, price, volume):
            return self.send_order(vt_symbol, "SHORT", "OPEN", price, volume)

        def cover(self, vt_symbol, price, volume):
            return self.send_order(vt_symbol, "LONG", "CLOSE", price, volume)

        def send_order(self, vt_symbol, direction, offset, price, volume):
            return self.strategy_engine.send_order(self, vt_symbol, direction, offset, price, volume)

        def get_signal(self):
            return self.strategy_engine.get_signal()

        def get_pos(self, vt_symbol):
            return self.pos_data.get(vt_symbol, 0)

        def set_target(self, vt_symbol, target):
            self.target_data[vt_symbol] = target

        def get_target(self, vt_symbol):
            return self.target_data.get(vt_symbol, 0)

        def execute_trading(self, bars, price_add):
            pass  # 在 on_bars 中直接处理

        def cancel_all(self):
            for vt_orderid in list(self.active_orderids):
                self.cancel_order(vt_orderid)

        def cancel_order(self, vt_orderid):
            self.strategy_engine.cancel_order(self, vt_orderid)

        def update_trade(self, trade):
            if trade.direction.value == "LONG":
                self.pos_data[trade.vt_symbol] = self.pos_data.get(trade.vt_symbol, 0) + trade.volume
            else:
                self.pos_data[trade.vt_symbol] = self.pos_data.get(trade.vt_symbol, 0) - trade.volume
            self.on_trade(trade)

        def update_order(self, order):
            self.orders[order.vt_orderid] = order
            if not order.is_active() and order.vt_orderid in self.active_orderids:
                self.active_orderids.remove(order.vt_orderid)

        def write_log(self, msg):
            self.strategy_engine.write_log(msg, self)

    # 添加策略
    strategy_setting = {
        "direction": bt_cfg.get("direction", "both"),
        "max_position": bt_cfg.get("max_position", 5),
        "long_threshold": bt_cfg.get("long_threshold", 0.005),
        "short_threshold": bt_cfg.get("short_threshold", -0.005),
    }

    engine.add_strategy(FuturesSignalStrategy, strategy_setting, signal_df)

    # 运行回测
    engine.load_data()
    engine.run_backtesting()

    # 计算结果
    daily_df = engine.calculate_result()
    if daily_df is not None:
        statistics = engine.calculate_statistics()
    else:
        print("  ⚠️ 无回测结果")
        statistics = {}

    return statistics


def save_results(
    statistics: dict,
    signal_df: pl.DataFrame,
    dataset: AlphaDataset,
    model: LGBModel,
    config: dict,
) -> None:
    """
    步骤7: 保存结果
    """
    print("\n💾 步骤7: 保存结果")
    print("-" * 40)

    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存回测统计
    if statistics:
        stats_df = pl.DataFrame([statistics])
        stats_path = results_dir / f"backtest_stats_{timestamp}.csv"
        stats_df.write_csv(stats_path)
        print(f"  📊 回测统计: {stats_path}")

    # 保存信号
    signal_path = results_dir / f"signals_{timestamp}.parquet"
    signal_df.write_parquet(signal_path)
    print(f"  📡 预测信号: {signal_path}")

    # 打印关键指标
    if statistics:
        print("\n" + "=" * 50)
        print("📊 回测结果摘要")
        print("=" * 50)
        print(f"  总收益率:     {statistics.get('total_return', 0):.2f}%")
        print(f"  年化收益:     {statistics.get('annual_return', 0):.2f}%")
        print(f"  Sharpe Ratio: {statistics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤:     {statistics.get('max_drawdown', 0):,.2f}")
        print(f"  回撤百分比:   {statistics.get('max_ddpercent', 0):.2f}%")
        print(f"  总交易笔数:   {statistics.get('total_trade_count', 0)}")
        print(f"  盈利天数:     {statistics.get('profit_days', 0)}")
        print(f"  亏损天数:     {statistics.get('loss_days', 0)}")
        print("=" * 50)


def main():
    """主流程"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    print(f"📄 配置文件: {config_path}")
    config = load_config(config_path)
    print_config(config)

    # 步骤1: 准备连续合约数据
    continuous_data = prepare_data(config)
    if not continuous_data:
        print("❌ 无可用数据，退出")
        return

    # 步骤2: 加载数据到 AlphaLab
    lab, vt_symbols = load_data_to_lab(continuous_data, config)

    # 步骤3: 构建因子数据集
    dataset = build_dataset(lab, vt_symbols, config)

    # 步骤4: 训练模型
    model = train_model(dataset, config)

    # 步骤5: 生成信号
    signal_df = generate_signals(dataset, model, config)

    # 步骤6: 回测
    statistics = run_backtest(lab, vt_symbols, signal_df, config)

    # 步骤7: 保存结果
    save_results(statistics, signal_df, dataset, model, config)

    print("\n✅ 研究流程完成！")


if __name__ == "__main__":
    main()
