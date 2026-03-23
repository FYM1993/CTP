"""
LightGBM 模型封装
=================

实现 vnpy AlphaModel 接口，封装 LightGBM 回归模型。
"""

import numpy as np
import lightgbm as lgb

from vnpy.alpha import AlphaModel
from vnpy.alpha.dataset import AlphaDataset, Segment


class LGBModel(AlphaModel):
    """LightGBM 回归模型"""

    def __init__(
        self,
        max_depth: int = 6,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        min_child_samples: int = 50,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.model: lgb.LGBMRegressor | None = None

        self.params = {
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
            **kwargs,
        }

    def fit(self, dataset: AlphaDataset) -> None:
        """训练模型"""
        # 获取训练数据
        learn_df = dataset.fetch_learn(Segment.TRAIN)

        # 分离特征和标签
        feature_cols = [c for c in learn_df.columns if c not in ("datetime", "vt_symbol", "label")]

        X = learn_df.select(feature_cols).to_numpy()
        y = learn_df.select("label").to_numpy().ravel()

        # 处理 NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        print(f"  训练样本数: {len(X)}, 特征数: {len(feature_cols)}")

        # 创建模型
        self.model = lgb.LGBMRegressor(**self.params)

        # 验证集回调
        valid_df = dataset.fetch_learn(Segment.VALID)
        X_val = valid_df.select(feature_cols).to_numpy()
        y_val = valid_df.select("label").to_numpy().ravel()
        val_mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        # 训练
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(50),
            ],
        )

        print(f"  最佳迭代: {self.model.best_iteration_}")

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        df = dataset.fetch_learn(segment)
        feature_cols = [c for c in df.columns if c not in ("datetime", "vt_symbol", "label")]

        X = df.select(feature_cols).to_numpy()

        predictions = self.model.predict(X)

        # 将 NaN 特征的预测值设为 NaN
        nan_mask = np.isnan(X).any(axis=1)
        predictions[nan_mask] = np.nan

        return predictions

    def detail(self) -> dict:
        """输出模型详情"""
        if self.model is None:
            return {}

        return {
            "best_iteration": self.model.best_iteration_,
            "n_features": self.model.n_features_in_,
            "params": self.params,
        }

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """获取特征重要性"""
        if self.model is None:
            return {}

        importance = self.model.feature_importances_

        if feature_names:
            return dict(zip(feature_names, importance))

        return {f"feature_{i}": v for i, v in enumerate(importance)}
