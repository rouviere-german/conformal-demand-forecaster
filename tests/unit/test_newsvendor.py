"""The newsvendor decision layer: turning conformalized quantiles into order quantities.

The classical result (Arrow, Harris and Marschak 1951) is that the cost-minimising order is the
``c_s / (c_s + c_h)`` fractile of the demand distribution. Here that distribution is only known
at a handful of points -- the median and the conformalized interval bounds -- so the order comes
from a monotone interpolation of those points. The properties below pin down what that must
satisfy: monotone in the stockout cost, bounded by the outermost known quantile, never negative.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conformal_demand_forecaster.decisions.newsvendor import (
    critical_fractile,
    interpolate_quantile,
    order_quantities,
    order_quantity,
)

pytestmark = pytest.mark.unit

costs = st.floats(min_value=0.01, max_value=100.0)
INTERVALS = {0.2: (4.0, 12.0), 0.05: (2.0, 18.0)}


# --- the fractile ---------------------------------------------------------------------------


def test_equal_costs_give_the_median() -> None:
    assert critical_fractile(1.0, 1.0) == pytest.approx(0.5)


def test_expensive_stockouts_push_the_fractile_up() -> None:
    assert critical_fractile(1.0, 9.0) == pytest.approx(0.9)
    assert critical_fractile(9.0, 1.0) == pytest.approx(0.1)


@given(holding=costs, stockout=costs)
def test_the_fractile_always_lies_inside_the_unit_interval(holding: float, stockout: float) -> None:
    assert 0.0 < critical_fractile(holding, stockout) < 1.0


@pytest.mark.parametrize(("holding", "stockout"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_non_positive_costs_are_refused(holding: float, stockout: float) -> None:
    with pytest.raises(ValueError, match="costs must be positive"):
        critical_fractile(holding, stockout)


# --- interpolation --------------------------------------------------------------------------


def test_interpolation_hits_the_known_points() -> None:
    assert interpolate_quantile([0.1, 0.5, 0.9], [2.0, 5.0, 9.0], 0.5) == pytest.approx(5.0)
    assert interpolate_quantile([0.1, 0.5, 0.9], [2.0, 5.0, 9.0], 0.3) == pytest.approx(3.5)


def test_interpolation_clips_outside_the_known_support() -> None:
    assert interpolate_quantile([0.1, 0.9], [2.0, 9.0], 0.99) == pytest.approx(9.0)
    assert interpolate_quantile([0.1, 0.9], [2.0, 9.0], 0.01) == pytest.approx(2.0)


def test_interpolation_enforces_monotonicity_of_the_quantile_function() -> None:
    """Crossing inputs are repaired by a running maximum rather than producing nonsense."""
    assert interpolate_quantile([0.1, 0.5, 0.9], [5.0, 2.0, 9.0], 0.5) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("levels", "values", "fractile"),
    [([], [], 0.5), ([0.1], [1.0, 2.0], 0.5), ([0.1, 0.9], [1.0, 2.0], 0.0)],
)
def test_interpolation_rejects_malformed_input(
    levels: list[float], values: list[float], fractile: float
) -> None:
    with pytest.raises(ValueError, match=r"non-empty|strictly inside"):
        interpolate_quantile(levels, values, fractile)


# --- single orders --------------------------------------------------------------------------


def test_equal_costs_order_the_median() -> None:
    decision = order_quantity(7.0, INTERVALS, 1.0, 1.0)
    assert decision.fractile == pytest.approx(0.5)
    assert decision.order == pytest.approx(7.0)
    assert decision.order_units == 7


def test_costly_stockouts_order_above_the_median() -> None:
    cheap = order_quantity(7.0, INTERVALS, 1.0, 1.0).order
    expensive = order_quantity(7.0, INTERVALS, 1.0, 20.0).order
    assert expensive > cheap
    assert expensive <= 18.0  # clipped at the widest conformalized upper bound


def test_costly_holding_orders_below_the_median() -> None:
    assert order_quantity(7.0, INTERVALS, 20.0, 1.0).order < 7.0


def test_order_units_round_up() -> None:
    decision = order_quantity(7.2, {0.2: (7.0, 7.5)}, 1.0, 1.0)
    assert decision.order == pytest.approx(7.2)
    assert decision.order_units == 8


def test_orders_are_never_negative() -> None:
    assert order_quantity(0.0, {0.2: (0.0, 0.0)}, 5.0, 1.0).order == 0.0


def test_invalid_alphas_are_refused() -> None:
    with pytest.raises(ValueError, match="alpha must lie strictly inside"):
        order_quantity(5.0, {1.0: (1.0, 9.0)}, 1.0, 1.0)


@given(
    median=st.floats(0.0, 500.0),
    half=st.floats(0.0, 500.0),
    holding=costs,
    stockout=costs,
)
def test_the_order_stays_between_the_outermost_known_quantiles(
    median: float, half: float, holding: float, stockout: float
) -> None:
    intervals = {0.05: (max(0.0, median - half), median + half)}
    decision = order_quantity(median, intervals, holding, stockout)
    assert max(0.0, median - half) - 1e-9 <= decision.order <= median + half + 1e-9


@given(median=st.floats(1.0, 200.0), stockout=costs, extra=st.floats(0.01, 100.0))
def test_the_order_is_monotone_in_the_stockout_cost(
    median: float, stockout: float, extra: float
) -> None:
    low = order_quantity(median, INTERVALS, 1.0, stockout).order
    high = order_quantity(median, INTERVALS, 1.0, stockout + extra).order
    assert high >= low - 1e-9


# --- vectorised orders ----------------------------------------------------------------------


def test_the_vectorised_form_matches_the_scalar_one() -> None:
    median = np.array([7.0, 3.0, 20.0])
    intervals = {
        0.2: (np.array([4.0, 1.0, 15.0]), np.array([12.0, 6.0, 26.0])),
        0.05: (np.array([2.0, 0.0, 11.0]), np.array([18.0, 9.0, 31.0])),
    }
    for holding, stockout in ((1.0, 4.0), (3.0, 1.0), (1.0, 1.0)):
        batch = order_quantities(median, intervals, holding, stockout)
        expected = [
            order_quantity(
                float(median[i]),
                {a: (float(lo[i]), float(hi[i])) for a, (lo, hi) in intervals.items()},
                holding,
                stockout,
            ).order
            for i in range(3)
        ]
        assert batch == pytest.approx(expected)


def test_the_vectorised_form_clips_at_the_outer_columns() -> None:
    median = np.array([5.0])
    intervals = {0.2: (np.array([3.0]), np.array([8.0]))}
    assert order_quantities(median, intervals, 1.0, 1e6)[0] == pytest.approx(8.0)
    assert order_quantities(median, intervals, 1e6, 1.0)[0] == pytest.approx(3.0)


def test_the_vectorised_form_handles_the_median_only_case() -> None:
    orders = order_quantities(np.array([4.0, 9.0]), {}, 1.0, 4.0)
    assert orders.tolist() == [4.0, 9.0]
