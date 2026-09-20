"""Seasonal-naive and ETS baselines, and the MASE scale they share.

These are the comparators the conformal layer has to beat, so they must be right and they must
be honestly unguaranteed: their bands are empirical residual quantiles with no coverage promise.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_demand_forecaster.models.baselines import (
    PERIOD,
    EtsBaseline,
    SeasonalNaiveBaseline,
    mase_scale,
    residual_quantiles,
    seasonal_naive,
)

pytestmark = pytest.mark.unit

LEVELS = (0.05, 0.5, 0.95)


def weekly_series(weeks: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = np.array([10.0, 8.0, 9.0, 11.0, 14.0, 20.0, 16.0])
    return np.tile(shape, weeks) + rng.normal(scale=0.5, size=weeks * PERIOD)


def test_seasonal_naive_repeats_the_same_weekday() -> None:
    history = np.arange(21, dtype=np.float64)  # last index 20, a Sunday-equivalent
    forecast = seasonal_naive(history, [1, 7, 8, 14])
    assert forecast.tolist() == [14.0, 20.0, 14.0, 20.0]


def test_seasonal_naive_falls_back_when_history_is_shorter_than_a_week() -> None:
    forecast = seasonal_naive(np.array([3.0, 4.0]), [1, 5])
    assert forecast.tolist() == [4.0, 4.0]


def test_mase_scale_is_the_in_sample_seasonal_mae() -> None:
    history = np.array([1.0] * 7 + [3.0] * 7)
    assert mase_scale(history) == pytest.approx(2.0)


def test_mase_scale_is_undefined_for_short_history() -> None:
    assert np.isnan(mase_scale(np.arange(5, dtype=np.float64)))


def test_mase_scale_only_looks_back_the_requested_window() -> None:
    history = np.concatenate([np.zeros(200), np.tile([1.0, 5.0], 50)])
    narrow = mase_scale(history, lookback=28)
    wide = mase_scale(history, lookback=400)
    assert narrow != wide


def test_residual_quantiles_are_empirical_and_default_to_zero() -> None:
    residuals = np.arange(-50.0, 51.0)
    offsets = residual_quantiles(residuals, (0.1, 0.9), lookback=101)
    assert offsets[0] < 0 < offsets[1]
    assert residual_quantiles(np.zeros(0), LEVELS).tolist() == [0.0, 0.0, 0.0]
    assert residual_quantiles(np.array([np.nan]), LEVELS).tolist() == [0.0, 0.0, 0.0]


def test_seasonal_naive_bands_are_sorted_and_non_negative() -> None:
    history = weekly_series()
    forecast = SeasonalNaiveBaseline(LEVELS).forecast(history, [1, 7, 14])
    assert forecast.quantiles.shape == (3, len(LEVELS))
    assert np.all(np.diff(forecast.quantiles, axis=1) >= 0.0)
    assert np.all(forecast.quantiles >= 0.0)
    assert np.all(forecast.point >= 0.0)
    assert forecast.levels == LEVELS


def test_ets_tracks_a_weekly_pattern_better_than_noise() -> None:
    history = weekly_series(weeks=20, seed=3)
    forecast = EtsBaseline(LEVELS).forecast(history, [1, 7], series_key="s1")
    assert forecast.point.shape == (2,)
    assert np.all(np.isfinite(forecast.point))
    # The weekly shape puts Saturday (index 5) well above Tuesday (index 1).
    saturday = EtsBaseline(LEVELS).forecast(history[: 20 * PERIOD - 2], [1], series_key="s2")
    assert saturday.point[0] > 0.0


def test_ets_reuses_fitted_parameters_for_the_same_series() -> None:
    """The optimizer runs once per series; later origins re-run the filter with those params."""
    baseline = EtsBaseline(LEVELS)
    history = weekly_series(weeks=16, seed=5)
    first = baseline.forecast(history, [1], series_key="shop-1")
    cached = dict(baseline._params)
    assert set(cached) == {"shop-1"}
    assert set(cached["shop-1"]) == {"smoothing_level", "smoothing_seasonal"}

    second = baseline.forecast(history, [1], series_key="shop-1")
    third = baseline.forecast(history, [1], series_key="shop-1")
    # The cached (non-optimized) refit is deterministic and close to the optimized first fit.
    assert second.point.tolist() == third.point.tolist()
    assert second.point[0] == pytest.approx(first.point[0], rel=0.05)
    assert baseline._params == cached


def test_ets_falls_back_to_seasonal_naive_on_degenerate_history() -> None:
    baseline = EtsBaseline(LEVELS)
    flat = np.full(60, 4.0)
    naive = SeasonalNaiveBaseline(LEVELS).forecast(flat, [1, 7])
    fallback = baseline.forecast(flat, [1, 7], series_key="flat")
    assert fallback.point.tolist() == naive.point.tolist()
    short = np.arange(10, dtype=np.float64)
    assert baseline.forecast(short, [1], series_key="short").point.size == 1


def test_ets_bands_are_sorted_and_non_negative() -> None:
    history = weekly_series(weeks=18, seed=9)
    forecast = EtsBaseline(LEVELS).forecast(history, [1, 3, 7], series_key="s")
    assert np.all(np.diff(forecast.quantiles, axis=1) >= 0.0)
    assert np.all(forecast.quantiles >= 0.0)
