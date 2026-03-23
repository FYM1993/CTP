"""
期货Alpha因子集
================

在 vnpy Alpha158 的基础上，扩展期货特有因子：
- 持仓量变化率（OI delta）
- 成交量变化率
- 量价背离
- 波动率因子
- 收益分布因子

所有因子使用 vnpy alpha 的表达式语法。
"""

import polars as pl

from vnpy.alpha import AlphaDataset


class FuturesAlpha158(AlphaDataset):
    """
    期货专用 Alpha158 因子集

    包含：
    1. 标准 Alpha158 因子（K线形态 + 时序量价）
    2. 期货特有因子（持仓量、波动率、收益分布等）
    """

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        futures_windows: list[int] | None = None,
        label_type: str = "ret_3",
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )

        if futures_windows is None:
            futures_windows = [5, 10, 20]

        self._add_standard_alpha158()
        self._add_futures_factors(futures_windows)
        self._set_label(label_type)

    def _add_standard_alpha158(self) -> None:
        """标准 Alpha158 因子（与 vnpy alpha 一致）"""

        # ====== K线形态特征（9个）======
        self.add_feature("kmid", "(close - open) / open")
        self.add_feature("klen", "(high - low) / open")
        self.add_feature("kmid_2", "(close - open) / (high - low + 1e-12)")
        self.add_feature("kup", "(high - ts_greater(open, close)) / open")
        self.add_feature("kup_2", "(high - ts_greater(open, close)) / (high - low + 1e-12)")
        self.add_feature("klow", "(ts_less(open, close) - low) / open")
        self.add_feature("klow_2", "((ts_less(open, close) - low) / (high - low + 1e-12))")
        self.add_feature("ksft", "(close * 2 - high - low) / open")
        self.add_feature("ksft_2", "(close * 2 - high - low) / (high - low + 1e-12)")

        # ====== 价格相对位置（4个）======
        for field in ["open", "high", "low", "vwap"]:
            self.add_feature(f"{field}_0", f"{field} / close")

        # ====== 时序因子（各5个窗口 × 12种 = 60个）======
        windows: list[int] = [5, 10, 20, 30, 60]

        for w in windows:
            self.add_feature(f"roc_{w}", f"ts_delay(close, {w}) / close")

        for w in windows:
            self.add_feature(f"ma_{w}", f"ts_mean(close, {w}) / close")

        for w in windows:
            self.add_feature(f"std_{w}", f"ts_std(close, {w}) / close")

        for w in windows:
            self.add_feature(f"beta_{w}", f"ts_slope(close, {w}) / close")

        for w in windows:
            self.add_feature(f"rsqr_{w}", f"ts_rsquare(close, {w})")

        for w in windows:
            self.add_feature(f"resi_{w}", f"ts_resi(close, {w}) / close")

        for w in windows:
            self.add_feature(f"max_{w}", f"ts_max(high, {w}) / close")

        for w in windows:
            self.add_feature(f"min_{w}", f"ts_min(low, {w}) / close")

        for w in windows:
            self.add_feature(f"qtlu_{w}", f"ts_quantile(close, {w}, 0.8) / close")

        for w in windows:
            self.add_feature(f"qtld_{w}", f"ts_quantile(close, {w}, 0.2) / close")

        for w in windows:
            self.add_feature(f"rank_{w}", f"ts_rank(close, {w})")

        for w in windows:
            self.add_feature(
                f"rsv_{w}",
                f"(close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(f"imax_{w}", f"ts_argmax(high, {w}) / {w}")

        for w in windows:
            self.add_feature(f"imin_{w}", f"ts_argmin(low, {w}) / {w}")

        for w in windows:
            self.add_feature(f"imxd_{w}", f"(ts_argmax(high, {w}) - ts_argmin(low, {w})) / {w}")

        # ====== 量价关联因子（12种 × 5窗口 = 60个）======
        for w in windows:
            self.add_feature(f"corr_{w}", f"ts_corr(close, ts_log(volume + 1), {w})")

        for w in windows:
            self.add_feature(
                f"cord_{w}",
                f"ts_corr(close / ts_delay(close, 1), ts_log(volume / ts_delay(volume, 1) + 1), {w})"
            )

        for w in windows:
            self.add_feature(f"cntp_{w}", f"ts_mean(close > ts_delay(close, 1), {w})")

        for w in windows:
            self.add_feature(f"cntn_{w}", f"ts_mean(close < ts_delay(close, 1), {w})")

        for w in windows:
            self.add_feature(
                f"cntd_{w}",
                f"ts_mean(close > ts_delay(close, 1), {w}) - ts_mean(close < ts_delay(close, 1), {w})"
            )

        for w in windows:
            self.add_feature(
                f"sump_{w}",
                f"ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) / "
                f"(ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(
                f"sumn_{w}",
                f"ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w}) / "
                f"(ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(
                f"sumd_{w}",
                f"(ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) - "
                f"ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w})) / "
                f"(ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(f"vma_{w}", f"ts_mean(volume, {w}) / (volume + 1e-12)")

        for w in windows:
            self.add_feature(f"vstd_{w}", f"ts_std(volume, {w}) / (volume + 1e-12)")

        for w in windows:
            self.add_feature(
                f"wvma_{w}",
                f"ts_std(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) / "
                f"(ts_mean(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(
                f"vsump_{w}",
                f"ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) / "
                f"(ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(
                f"vsumn_{w}",
                f"ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w}) / "
                f"(ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)"
            )

        for w in windows:
            self.add_feature(
                f"vsumd_{w}",
                f"(ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) - "
                f"ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w})) / "
                f"(ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)"
            )

    def _add_futures_factors(self, windows: list[int]) -> None:
        """期货特有因子"""

        for w in windows:
            # ====== 持仓量因子 ======
            # 持仓量变化率
            self.add_feature(
                f"oi_chg_{w}",
                f"(open_interest - ts_delay(open_interest, {w})) / (ts_delay(open_interest, {w}) + 1)"
            )

            # 持仓量均值比
            self.add_feature(
                f"oi_ma_{w}",
                f"ts_mean(open_interest, {w}) / (open_interest + 1)"
            )

            # 持仓量趋势（斜率/均值）
            self.add_feature(
                f"oi_beta_{w}",
                f"ts_slope(open_interest, {w}) / (ts_mean(open_interest, {w}) + 1)"
            )

            # ====== 波动率因子 ======
            # 真实波幅
            self.add_feature(
                f"atr_{w}",
                f"ts_mean(ts_greater(high - low, ts_abs(high - ts_delay(close, 1))), {w}) / close"
            )

            # 收益率波动率
            self.add_feature(
                f"ret_vol_{w}",
                f"ts_std(close / ts_delay(close, 1) - 1, {w})"
            )

            # 收益偏度
            self.add_feature(
                f"ret_skew_{w}",
                f"ts_mean((close / ts_delay(close, 1) - 1) ** 3, {w}) / "
                f"(ts_std(close / ts_delay(close, 1) - 1, {w}) ** 3 + 1e-12)"
            )

            # ====== 量能因子 ======
            # 成交额变化率
            self.add_feature(
                f"turnover_chg_{w}",
                f"(turnover - ts_delay(turnover, {w})) / (ts_delay(turnover, {w}) + 1)"
            )

            # 量价同步性（量涨价涨、量跌价跌的比例）
            self.add_feature(
                f"vol_price_sync_{w}",
                f"ts_mean((close > ts_delay(close, 1)) == (volume > ts_delay(volume, 1)), {w})"
            )

            # ====== 收益分布因子 ======
            # 连涨天数占比
            self.add_feature(
                f"up_ratio_{w}",
                f"ts_sum(close > ts_delay(close, 1), {w}) / {w}"
            )

            # 最大连续涨幅
            self.add_feature(
                f"max_ret_{w}",
                f"ts_max(close / ts_delay(close, 1) - 1, {w})"
            )

            # 最大连续跌幅
            self.add_feature(
                f"min_ret_{w}",
                f"ts_min(close / ts_delay(close, 1) - 1, {w})"
            )

    def _set_label(self, label_type: str) -> None:
        """设置标签"""
        label_map = {
            "ret_1": "ts_delay(close, -1) / close - 1",
            "ret_3": "ts_delay(close, -3) / ts_delay(close, -1) - 1",
            "ret_5": "ts_delay(close, -5) / ts_delay(close, -1) - 1",
        }

        expression = label_map.get(label_type, label_map["ret_3"])
        self.set_label(expression)
