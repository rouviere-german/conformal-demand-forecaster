"""Accuracy, calibration and interval-quality metrics, with their defining properties."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conformal_demand_forecaster.evaluation.metrics import (
    coverage,
    mase,
    mean_width,
    newsvendor_cost,
    pinball,
    scaled_pinball,
    winkler,
)

pytestmark = pytest.mark.unit

values = st.lists(st.floats(0.0, 1e3, allow_nan=False), min_size=1, max_size=60)


# --- coverage and width ---------------------------------------------------------------------


def test_coverage_counts_inclusive_containment() -> None:
    assert coverage([1.0, 5.0, 9.0], [2.0, 2.0, 2.0], [8.0, 8.0, 8.0]) == pytest.approx(1 / 3)
    assert coverage([2.0, 8.0], [2.0, 2.0], [8.0, 8.0]) == 1.0
    assert np.isnan(coverage([], [], []))


@given(y=values, data=st.data())
def test_a_wider_band_never_covers_less(y: list[float], data: st.DataObject) -> None:
    n = len(y)
    lower = np.array(data.draw(st.lists(st.floats(0.0, 1e3), min_size=n, max_size=n)))
    upper = lower + np.array(data.draw(st.lists(st.floats(0.0, 1e3), min_size=n, max_size=n)))
    assert coverage(y, lower - 1.0, upper + 1.0) >= coverage(y, lower, upper)


def test_mean_width_is_the_average_band_size() -> None:
    assert mean_width([1.0, 2.0], [4.0, 10.0]) == pytest.approx(5.5)
    assert np.isnan(mean_width([], []))


# --- Winkler --------------------------------------------------------------------------------


def test_winkler_reduces_to_the_width_when_everything_is_covered() -> None:
    assert winkler([5.0], [2.0], [8.0], 0.2) == pytest.approx(6.0)


def test_winkler_penalises_misses_by_two_over_alpha() -> None:
    alpha = 0.2
    inside = winkler([5.0], [2.0], [8.0], alpha)
    above = winkler([9.0], [2.0], [8.0], alpha)
    assert above == pytest.approx(inside + (2.0 / alpha) * 1.0)
    below = winkler([1.0], [2.0], [8.0], alpha)
    assert below == pytest.approx(inside + (2.0 / alpha) * 1.0)


@given(y=values, data=st.data(), alpha=st.floats(0.02, 0.5))
def test_winkler_is_never_below_the_mean_width(
    y: list[float], data: st.DataObject, alpha: float
) -> None:
    n = len(y)
    lower = np.array(data.draw(st.lists(st.floats(0.0, 1e3), min_size=n, max_size=n)))
    upper = lower + np.array(data.draw(st.lists(st.floats(0.0, 1e3), min_size=n, max_size=n)))
    assert winkler(y, lower, upper, alpha) >= mean_width(lower, upper) - 1e-9


def test_winkler_validates_alpha_and_empty_input() -> None:
    with pytest.raises(ValueError, match="alpha must lie strictly inside"):
        winkler([1.0], [0.0], [2.0], 0.0)
    assert np.isnan(winkler([], [], [], 0.2))


# --- pinball --------------------------------------------------------------------------------


def test_pinball_is_asymmetric_around_the_quantile_level() -> None:
    # q = 0.9 punishes under-forecasting nine times as hard as over-forecasting.
    under = pinball([10.0], [9.0], 0.9)
    over = pinball([10.0], [11.0], 0.9)
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


@given(y=values, q=st.floats(0.05, 0.95))
def test_pinball_is_minimised_at_the_empirical_quantile(y: list[float], q: float) -> None:
    """The minimiser is the empirical q-quantile taken as an order statistic (inverted CDF)."""
    best = float(np.quantile(y, q, method="inverted_cdf"))
    loss = pinball(y, [best] * len(y), q)
    candidates = [*y, best - 5.0, best + 5.0, float(np.mean(y))]
    for candidate in candidates:
        assert pinball(y, [candidate] * len(y), q) >= loss - 1e-9


def test_pinball_validates_its_level() -> None:
    with pytest.raises(ValueError, match="q must lie strictly inside"):
        pinball([1.0], [1.0], 1.0)
    assert np.isnan(pinball([], [], 0.5))


def test_scaled_pinball_averages_over_levels_and_divides_by_the_scale() -> None:
    quantiles = np.array([[8.0, 10.0, 12.0]])
    levels = (0.1, 0.5, 0.9)
    raw = np.mean([pinball([10.0], quantiles[:, j], q) for j, q in enumerate(levels)])
    assert scaled_pinball([10.0], quantiles, levels, 2.0) == pytest.approx(raw / 2.0)
    assert np.isnan(scaled_pinball([10.0], quantiles, levels, 0.0))
    with pytest.raises(ValueError, match=r"quantiles must be \(n, len\(levels\)\)"):
        scaled_pinball([10.0], quantiles, (0.1, 0.9), 1.0)


# --- MASE -----------------------------------------------------------------------------------


def test_mase_divides_the_mean_absolute_error_by_the_seasonal_scale() -> None:
    assert mase([10.0, 12.0], [8.0, 12.0], 2.0) == pytest.approx(0.5)


def test_mase_is_undefined_for_a_degenerate_scale() -> None:
    assert np.isnan(mase([1.0], [1.0], 0.0))
    assert np.isnan(mase([1.0], [1.0], float("nan")))
    assert np.isnan(mase([], [], 1.0))


# --- newsvendor cost ------------------------------------------------------------------------


def test_newsvendor_cost_charges_holding_and_shortage_separately() -> None:
    assert newsvendor_cost([5.0], [8.0], 1.0, 4.0) == pytest.approx(3.0)
    assert newsvendor_cost([5.0], [3.0], 1.0, 4.0) == pytest.approx(8.0)
    assert newsvendor_cost([5.0], [5.0], 1.0, 4.0) == 0.0


@given(y=values, order=values, holding=st.floats(0.1, 10.0), stockout=st.floats(0.1, 10.0))
def test_newsvendor_cost_is_non_negative_and_zero_only_on_an_exact_order(
    y: list[float], order: list[float], holding: float, stockout: float
) -> None:
    n = min(len(y), len(order))
    cost = newsvendor_cost(y[:n], order[:n], holding, stockout)
    assert cost >= 0.0
    assert newsvendor_cost(y, y, holding, stockout) == 0.0


def test_newsvendor_cost_validates_costs_and_empty_input() -> None:
    with pytest.raises(ValueError, match="costs must be positive"):
        newsvendor_cost([1.0], [1.0], 0.0, 1.0)
    assert np.isnan(newsvendor_cost([], [], 1.0, 1.0))
