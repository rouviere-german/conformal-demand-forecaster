"""Lag, rolling, calendar and price features with a strict no-lookahead rule.

A feature row describes one (series, origin ``t``, horizon ``h``) triple and predicts sales on
day ``t + h``. Two kinds of inputs are allowed:

- history features: any function of the panel restricted to days ``<= t`` (lags, rolling
  statistics, days since the last sale, stockout share, current price and promotion);
- known-in-advance covariates of the target day only: the planned price and promotion flag on
  day ``t + h`` and its calendar position.

Nothing else may enter. ``tests/unit/test_features.py`` perturbs sales, stockouts and demand on
days ``> t`` and prices/promotions on days other than ``t + h`` and asserts that the row is
unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from conformal_demand_forecaster.data.panel import Panel

__all__ = ["FEATURE_NAMES", "HISTORY_FEATURES", "TARGET_FEATURES", "FeatureBuilder", "FeatureRows"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

HISTORY_FEATURES: tuple[str, ...] = (
    "lag0",
    "lag1",
    "lag6",
    "lag7",
    "lag13",
    "lag27",
    "rm7",
    "rm28",
    "rm91",
    "rs28",
    "zero28",
    "days_since_sale",
    "stockout28",
    "stockout0",
    "price_ratio0",
    "promo0",
)
TARGET_FEATURES: tuple[str, ...] = (
    "horizon",
    "dow",
    "dom",
    "month",
    "doy_sin",
    "doy_cos",
    "price_ratio_target",
    "promo_target",
    "same_dow_last",
    "same_dow_mean4",
)
STATIC_FEATURES: tuple[str, ...] = ("segment_code",)
FEATURE_NAMES: tuple[str, ...] = HISTORY_FEATURES + TARGET_FEATURES + STATIC_FEATURES
_DAYS_SINCE_CAP = 91.0


@dataclass(frozen=True, slots=True)
class FeatureRows:
    """Feature matrix plus the bookkeeping needed to score and calibrate its rows."""

    X: pd.DataFrame
    series_index: IntArray
    origin: IntArray
    horizon: IntArray
    target_day: IntArray
    y_sales: FloatArray
    censored: FloatArray
    y_demand: FloatArray | None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def observed(self) -> npt.NDArray[np.bool_]:
        """Rows whose target day has an observed value."""
        return ~np.isnan(self.y_sales)

    def subset(self, mask: npt.NDArray[np.bool_]) -> FeatureRows:
        return FeatureRows(
            X=self.X.loc[mask].reset_index(drop=True),
            series_index=self.series_index[mask],
            origin=self.origin[mask],
            horizon=self.horizon[mask],
            target_day=self.target_day[mask],
            y_sales=self.y_sales[mask],
            censored=self.censored[mask],
            y_demand=None if self.y_demand is None else self.y_demand[mask],
        )


def _rolling(wide: pd.DataFrame, window: int, kind: str) -> FloatArray:
    roller = wide.rolling(window, min_periods=1)
    if kind == "mean":
        result = roller.mean()
    elif kind == "std":
        result = wide.rolling(window, min_periods=2).std().fillna(0.0)
    else:
        result = roller.median()
    return result.to_numpy(dtype=np.float64).T


def _days_since_sale(sales: FloatArray) -> FloatArray:
    n_days = sales.shape[1]
    day = np.broadcast_to(np.arange(n_days, dtype=np.int64), sales.shape)
    last = np.where(sales > 0, day, -1)
    last = np.maximum.accumulate(last, axis=1)
    since = np.where(last >= 0, day - last, day + 1).astype(np.float64)
    return np.minimum(since, _DAYS_SINCE_CAP)


class FeatureBuilder:
    """Precomputes history features once per panel, then assembles rows for any origins."""

    def __init__(self, panel: Panel) -> None:
        self.panel = panel
        sales_wide = pd.DataFrame(panel.sales.T)
        price_wide = pd.DataFrame(panel.price.T)
        stockout_wide = pd.DataFrame(panel.stockout.T)
        zero_wide = pd.DataFrame(np.where(np.isnan(panel.sales), np.nan, panel.sales == 0).T)
        self._price_median91 = _rolling(price_wide, 91, "median")
        self._history: dict[str, FloatArray] = {
            "lag0": panel.sales,
            "lag1": self._shift(panel.sales, 1),
            "lag6": self._shift(panel.sales, 6),
            "lag7": self._shift(panel.sales, 7),
            "lag13": self._shift(panel.sales, 13),
            "lag27": self._shift(panel.sales, 27),
            "rm7": _rolling(sales_wide, 7, "mean"),
            "rm28": _rolling(sales_wide, 28, "mean"),
            "rm91": _rolling(sales_wide, 91, "mean"),
            "rs28": _rolling(sales_wide, 28, "std"),
            "zero28": _rolling(zero_wide, 28, "mean"),
            "days_since_sale": _days_since_sale(np.nan_to_num(panel.sales, nan=0.0)),
            "stockout28": _rolling(stockout_wide, 28, "mean"),
            "stockout0": panel.stockout,
            "price_ratio0": panel.price / self._price_median91,
            "promo0": panel.promo,
        }
        dates = panel.dates
        self._dow = dates.dayofweek.to_numpy().astype(np.float64)
        self._dom = dates.day.to_numpy().astype(np.float64)
        self._month = dates.month.to_numpy().astype(np.float64)
        doy = dates.dayofyear.to_numpy().astype(np.float64)
        self._doy_sin = np.sin(2.0 * math.pi * doy / 365.25)
        self._doy_cos = np.cos(2.0 * math.pi * doy / 365.25)

    @staticmethod
    def _shift(values: FloatArray, lag: int) -> FloatArray:
        shifted = np.full_like(values, np.nan)
        shifted[:, lag:] = values[:, :-lag]
        return shifted

    def rows(
        self,
        origins: Sequence[int] | IntArray,
        horizons: Sequence[int] | IntArray,
        *,
        series: Sequence[int] | IntArray | None = None,
    ) -> FeatureRows:
        """Rows for the cartesian product of ``series`` x ``origins`` x ``horizons``.

        Every ``origin + horizon`` must lie inside the panel (extend it first for future targets).
        """
        panel = self.panel
        series_arr = (
            np.arange(panel.n_series, dtype=np.int64)
            if series is None
            else np.asarray(series, dtype=np.int64)
        )
        origin_arr = np.asarray(origins, dtype=np.int64)
        horizon_arr = np.asarray(horizons, dtype=np.int64)
        if horizon_arr.size and horizon_arr.min() < 1:
            raise ValueError("horizons must be >= 1")
        if origin_arr.size and (origin_arr.min() < 0 or origin_arr.max() >= panel.n_days):
            raise ValueError("origins must lie inside the panel")
        s, t, h = (
            item.ravel() for item in np.meshgrid(series_arr, origin_arr, horizon_arr, indexing="ij")
        )
        target = t + h
        if target.size and target.max() >= panel.n_days:
            raise ValueError("origin + horizon exceeds the panel; extend it first")

        columns: dict[str, FloatArray] = {
            name: values[s, t] for name, values in self._history.items()
        }
        k = np.ceil(h / 7.0).astype(np.int64)
        same_dow_index = target - 7 * k
        columns["horizon"] = h.astype(np.float64)
        columns["dow"] = self._dow[target]
        columns["dom"] = self._dom[target]
        columns["month"] = self._month[target]
        columns["doy_sin"] = self._doy_sin[target]
        columns["doy_cos"] = self._doy_cos[target]
        columns["price_ratio_target"] = panel.price[s, target] / self._price_median91[s, t]
        columns["promo_target"] = panel.promo[s, target]
        columns["same_dow_last"] = self._lookup(s, same_dow_index)
        stacked = np.stack([self._lookup(s, same_dow_index - 7 * j) for j in range(4)], axis=1)
        # nanmean over an all-NaN row (a target in the first weeks of the panel) warns; compute
        # the mean explicitly so an empty history simply yields NaN.
        present = ~np.isnan(stacked)
        counts = present.sum(axis=1)
        totals = np.where(present, stacked, 0.0).sum(axis=1)
        columns["same_dow_mean4"] = np.divide(
            totals, counts, out=np.full(counts.shape, np.nan), where=counts > 0
        )
        columns["segment_code"] = panel.segment_codes[s].astype(np.float64)

        frame = pd.DataFrame({name: columns[name] for name in FEATURE_NAMES})
        return FeatureRows(
            X=frame,
            series_index=s,
            origin=t,
            horizon=h,
            target_day=target,
            y_sales=panel.sales[s, target],
            censored=panel.stockout[s, target],
            y_demand=None if panel.demand is None else panel.demand[s, target],
        )

    def _lookup(self, series: IntArray, day: IntArray) -> FloatArray:
        """``sales[series, day]`` with NaN for days before the panel starts."""
        valid = day >= 0
        out = np.full(day.shape, np.nan)
        out[valid] = self.panel.sales[series[valid], day[valid]]
        return out
