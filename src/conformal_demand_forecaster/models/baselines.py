"""Seasonal-naive and ETS baselines with empirical residual quantiles.

Both baselines forecast from the history of one series up to the origin. Their quantiles add
the empirical quantiles of recent in-sample one-step residuals to the point forecast: a common
non-conformal way to build bands, with no coverage guarantee, which is exactly what the conformal
layer is compared against.

ETS is statsmodels' Holt-Winters ``ExponentialSmoothing`` with additive weekly seasonality. Its
smoothing parameters are estimated once per series and reused at later origins (the filter is
re-run on the new history), which keeps a 200-series backtest inside minutes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["BaselineForecast", "EtsBaseline", "SeasonalNaiveBaseline", "mase_scale"]

FloatArray = npt.NDArray[np.float64]
Horizons = Sequence[int] | npt.NDArray[np.int64]
PERIOD = 7
_MIN_ETS_HISTORY = 3 * PERIOD


@dataclass(frozen=True, slots=True)
class BaselineForecast:
    point: FloatArray  # (n_horizons,)
    quantiles: FloatArray  # (n_horizons, n_levels)
    levels: tuple[float, ...]


def mase_scale(history: FloatArray, *, lookback: int = 365, period: int = PERIOD) -> float:
    """In-sample seasonal-naive MAE over the last ``lookback`` days (the MASE denominator)."""
    values = np.asarray(history, dtype=np.float64)[-lookback:]
    if values.size <= period:
        return float("nan")
    return float(np.mean(np.abs(values[period:] - values[:-period])))


def seasonal_naive(history: FloatArray, horizons: Horizons) -> FloatArray:
    """``y[T + h] = y[T + h - 7k]`` for the smallest ``k`` with ``T + h - 7k <= T``."""
    values = np.asarray(history, dtype=np.float64)
    last = values.size - 1
    out = np.empty(len(horizons))
    for i, h in enumerate(horizons):
        k = math.ceil(h / PERIOD)
        index = last + int(h) - PERIOD * k
        out[i] = values[index] if index >= 0 else values[-1]
    return out


def residual_quantiles(
    residuals: FloatArray, levels: Sequence[float], *, lookback: int = 91
) -> FloatArray:
    """Empirical quantiles of the most recent residuals (zeros when there are none)."""
    recent = np.asarray(residuals, dtype=np.float64)[-lookback:]
    recent = recent[~np.isnan(recent)]
    if recent.size == 0:
        return np.zeros(len(levels))
    return np.quantile(recent, np.asarray(levels, dtype=np.float64))


def _bands(point: FloatArray, offsets: FloatArray, levels: Sequence[float]) -> BaselineForecast:
    quantiles = np.maximum(point[:, None] + offsets[None, :], 0.0)
    quantiles = np.sort(quantiles, axis=1)
    return BaselineForecast(point=np.maximum(point, 0.0), quantiles=quantiles, levels=tuple(levels))


class SeasonalNaiveBaseline:
    def __init__(self, levels: Sequence[float], *, residual_lookback: int = 91) -> None:
        self.levels = tuple(levels)
        self.residual_lookback = residual_lookback

    def forecast(self, history: FloatArray, horizons: Horizons) -> BaselineForecast:
        values = np.asarray(history, dtype=np.float64)
        point = seasonal_naive(values, horizons)
        residuals = values[PERIOD:] - values[:-PERIOD] if values.size > PERIOD else np.zeros(0)
        offsets = residual_quantiles(residuals, self.levels, lookback=self.residual_lookback)
        return _bands(point, offsets, self.levels)


class EtsBaseline:
    """Holt-Winters ETS (additive weekly seasonality, no trend) via statsmodels."""

    def __init__(
        self,
        levels: Sequence[float],
        *,
        fit_window: int = 365,
        residual_lookback: int = 91,
    ) -> None:
        self.levels = tuple(levels)
        self.fit_window = fit_window
        self.residual_lookback = residual_lookback
        self._params: dict[str, dict[str, float]] = {}

    def forecast(
        self, history: FloatArray, horizons: Horizons, *, series_key: str = ""
    ) -> BaselineForecast:
        values = np.asarray(history, dtype=np.float64)[-self.fit_window :]
        if values.size < _MIN_ETS_HISTORY or np.allclose(values, values[0]):
            return SeasonalNaiveBaseline(self.levels).forecast(values, horizons)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: PLC0415

        model = ExponentialSmoothing(
            values,
            trend=None,
            seasonal="add",
            seasonal_periods=PERIOD,
            initialization_method="estimated",
        )
        cached = self._params.get(series_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cached is None:
                fitted = model.fit(optimized=True, use_brute=False)
                self._params[series_key] = {
                    "smoothing_level": float(fitted.params["smoothing_level"]),
                    "smoothing_seasonal": float(fitted.params["smoothing_seasonal"]),
                }
            else:
                fitted = model.fit(optimized=False, **cached)
        max_h = int(max(horizons))
        path = np.asarray(fitted.forecast(max_h), dtype=np.float64)
        point = np.array([path[int(h) - 1] for h in horizons])
        residuals = values - np.asarray(fitted.fittedvalues, dtype=np.float64)
        offsets = residual_quantiles(residuals, self.levels, lookback=self.residual_lookback)
        return _bands(np.nan_to_num(point, nan=0.0), offsets, self.levels)
