"""Global LightGBM point and quantile models, one set per horizon bucket.

For each bucket of horizons the wrapper trains a Tweedie point model (non-negative, suited to
intermittent counts) and one pinball-loss model per quantile level. Rows whose target day is
censored by a stockout are excluded from training: a censored sale is only a lower bound on
demand and would bias every quantile downwards. Predicted quantiles are sorted per row so they
never cross, and clipped at zero.

Models serialize to LightGBM's text format (``model_to_string``), never pickle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["GbmConfig", "QuantileGbm", "QuantilePrediction"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GbmConfig:
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.5, 0.9, 0.95)
    horizon_buckets: tuple[tuple[int, int], ...] = ((1, 7), (8, 14), (15, 28))
    num_iterations: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 40
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_threads: int = 4
    seed: int = 7
    tweedie_variance_power: float = 1.3

    def __post_init__(self) -> None:
        if not self.quantiles or any(not 0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles must lie strictly inside (0, 1)")
        if list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError("quantiles must be strictly increasing")
        if not self.horizon_buckets:
            raise ValueError("at least one horizon bucket is required")
        expected = 1
        for low, high in self.horizon_buckets:
            if low != expected or high < low:
                raise ValueError("horizon buckets must be contiguous and start at 1")
            expected = high + 1

    @property
    def max_horizon(self) -> int:
        return self.horizon_buckets[-1][1]

    def bucket_of(self, horizon: int) -> int:
        for index, (low, high) in enumerate(self.horizon_buckets):
            if low <= horizon <= high:
                return index
        raise ValueError(f"horizon {horizon} is outside the configured buckets")

    def base_params(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }


@dataclass(frozen=True, slots=True)
class QuantilePrediction:
    point: FloatArray  # (n,)
    quantiles: FloatArray  # (n, n_levels), sorted per row, >= 0
    levels: tuple[float, ...]

    def column(self, level: float) -> FloatArray:
        """The predicted quantile column for ``level`` (which must be one the model was fit on)."""
        if level not in self.levels:
            raise ValueError(f"quantile level {level} is not one of {self.levels}")
        return self.quantiles[:, self.levels.index(level)]


def _model_key(bucket: int, name: str) -> str:
    return f"bucket{bucket}_{name}"


class QuantileGbm:
    def __init__(self, config: GbmConfig, feature_names: Sequence[str]) -> None:
        self.config = config
        self.feature_names = tuple(feature_names)
        self._boosters: dict[str, lgb.Booster] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._boosters)

    def _bucket_index(self, horizon: IntArray) -> IntArray:
        edges = np.array([high for _, high in self.config.horizon_buckets], dtype=np.int64)
        if horizon.size and (horizon.min() < 1 or horizon.max() > edges[-1]):
            raise ValueError("horizon outside the configured buckets")
        return np.searchsorted(edges, horizon, side="left").astype(np.int64)

    def _check_features(self, frame: pd.DataFrame) -> FloatArray:
        if tuple(frame.columns) != self.feature_names:
            raise ValueError("feature columns do not match the model's feature names")
        return frame.to_numpy(dtype=np.float64)

    def fit(
        self,
        frame: pd.DataFrame,
        y: FloatArray,
        horizon: IntArray,
        *,
        censored: FloatArray | None = None,
    ) -> QuantileGbm:
        features = self._check_features(frame)
        target = np.asarray(y, dtype=np.float64)
        keep = ~np.isnan(target)
        if censored is not None:
            keep &= np.asarray(censored, dtype=np.float64) < 0.5
        buckets = self._bucket_index(np.asarray(horizon, dtype=np.int64))
        self._boosters = {}
        for bucket in range(len(self.config.horizon_buckets)):
            mask = keep & (buckets == bucket)
            if mask.sum() < max(2 * self.config.min_data_in_leaf, 10):
                raise ValueError(f"not enough training rows for horizon bucket {bucket}")
            dataset = lgb.Dataset(
                features[mask],
                label=target[mask],
                feature_name=list(self.feature_names),
                free_raw_data=False,
                params={"verbosity": -1},
            )
            point_params = {
                **self.config.base_params(),
                "objective": "tweedie",
                "tweedie_variance_power": self.config.tweedie_variance_power,
            }
            self._boosters[_model_key(bucket, "point")] = lgb.train(
                point_params, dataset, num_boost_round=self.config.num_iterations
            )
            for level in self.config.quantiles:
                params = {**self.config.base_params(), "objective": "quantile", "alpha": level}
                self._boosters[_model_key(bucket, f"q{level:.3f}")] = lgb.train(
                    params, dataset, num_boost_round=self.config.num_iterations
                )
        return self

    def predict(self, frame: pd.DataFrame, horizon: IntArray) -> QuantilePrediction:
        if not self.fitted:
            raise ValueError("model is not fitted")
        features = self._check_features(frame)
        buckets = self._bucket_index(np.asarray(horizon, dtype=np.int64))
        n = features.shape[0]
        point = np.zeros(n)
        quantiles = np.zeros((n, len(self.config.quantiles)))
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            rows = features[mask]
            point[mask] = self._predict_one(_model_key(int(bucket), "point"), rows)
            for j, level in enumerate(self.config.quantiles):
                quantiles[mask, j] = self._predict_one(
                    _model_key(int(bucket), f"q{level:.3f}"), rows
                )
        quantiles = np.sort(np.maximum(quantiles, 0.0), axis=1)
        return QuantilePrediction(
            point=np.maximum(point, 0.0), quantiles=quantiles, levels=self.config.quantiles
        )

    def _predict_one(self, key: str, rows: FloatArray) -> FloatArray:
        if rows.shape[0] == 0:
            return np.zeros(0)
        return np.asarray(self._boosters[key].predict(rows), dtype=np.float64)

    def to_text(self) -> dict[str, str]:
        """LightGBM text models keyed by ``bucket<i>_<point|q0.xxx>``."""
        if not self.fitted:
            raise ValueError("model is not fitted")
        return {key: booster.model_to_string() for key, booster in sorted(self._boosters.items())}

    @classmethod
    def from_text(
        cls, texts: Mapping[str, str], config: GbmConfig, feature_names: Sequence[str]
    ) -> QuantileGbm:
        model = cls(config, feature_names)
        expected = {
            _model_key(bucket, name)
            for bucket in range(len(config.horizon_buckets))
            for name in ("point", *(f"q{level:.3f}" for level in config.quantiles))
        }
        if set(texts) != expected:
            raise ValueError("model text files do not match the configuration")
        model._boosters = {key: lgb.Booster(model_str=text) for key, text in texts.items()}
        for key, booster in model._boosters.items():
            if tuple(booster.feature_name()) != model.feature_names:
                raise ValueError(f"model {key} was trained on different features")
        return model
