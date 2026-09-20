"""The wide-array panel container: round-trips, extension for future targets and truncation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_demand_forecaster.data.generator import GeneratorConfig, generate_panel
from conformal_demand_forecaster.data.panel import Panel

pytestmark = pytest.mark.unit


@pytest.fixture
def frame() -> pd.DataFrame:
    return generate_panel(GeneratorConfig(n_series=4, n_days=90, seed=77, shifts=()))


@pytest.fixture
def panel(frame: pd.DataFrame) -> Panel:
    return Panel.from_frame(frame)


def test_round_trip_through_a_long_frame_preserves_the_data(
    frame: pd.DataFrame, panel: Panel
) -> None:
    restored = panel.to_frame()
    left = frame.sort_values(["series_id", "date"]).reset_index(drop=True)
    right = restored.sort_values(["series_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[right.columns], right, check_dtype=False)


def test_shape_and_segment_bookkeeping(panel: Panel) -> None:
    assert panel.n_series == 4
    assert panel.n_days == 90
    assert panel.sales.shape == (4, 90)
    assert panel.segment_codes.shape == (4,)
    assert set(panel.segment_codes.tolist()) <= {0, 1}
    assert panel.last_observed_day == 89


def test_day_index_maps_dates_and_rejects_outsiders(panel: Panel) -> None:
    assert panel.day_index(panel.dates[10]) == 10
    with pytest.raises(KeyError, match="outside the panel"):
        panel.day_index(pd.Timestamp("1999-01-01"))


def test_extend_appends_unknown_days_and_carries_prices_forward(panel: Panel) -> None:
    extended = panel.extend(14)
    assert extended.n_days == panel.n_days + 14
    assert np.all(np.isnan(extended.sales[:, -14:]))
    assert np.all(extended.price[:, -1] == panel.price[:, -1])
    assert np.all(extended.promo[:, -14:] == 0.0)
    assert extended.last_observed_day == panel.last_observed_day
    assert extended.demand is not None
    assert np.all(np.isnan(extended.demand[:, -14:]))
    # to_frame drops the unknown days again.
    assert len(extended.to_frame()) == len(panel.to_frame())


def test_extend_by_zero_days_is_a_no_op(panel: Panel) -> None:
    assert panel.extend(0) is panel


def test_truncate_keeps_a_prefix(panel: Panel) -> None:
    short = panel.truncate(30)
    assert short.n_days == 30
    assert np.array_equal(short.sales, panel.sales[:, :30])
    with pytest.raises(ValueError, match="must lie within the panel"):
        panel.truncate(0)
    with pytest.raises(ValueError, match="must lie within the panel"):
        panel.truncate(panel.n_days + 1)


def test_with_sales_replaces_only_the_arrays_asked_for(panel: Panel) -> None:
    zeros = np.zeros_like(panel.sales)
    replaced = panel.with_sales(zeros)
    assert np.array_equal(replaced.sales, zeros)
    assert np.array_equal(replaced.price, panel.price)
    flagged = panel.with_sales(zeros, stockout=np.ones_like(panel.stockout))
    assert np.all(flagged.stockout == 1.0)


def test_shape_mismatches_are_refused(panel: Panel) -> None:
    with pytest.raises(ValueError, match="sales must have shape"):
        Panel(
            series_ids=panel.series_ids,
            segments=panel.segments,
            dates=panel.dates,
            sales=panel.sales[:, :10],
            price=panel.price,
            promo=panel.promo,
            stockout=panel.stockout,
        )
    with pytest.raises(ValueError, match="one segment per series"):
        Panel(
            series_ids=panel.series_ids,
            segments=("smooth",),
            dates=panel.dates,
            sales=panel.sales,
            price=panel.price,
            promo=panel.promo,
            stockout=panel.stockout,
        )
    with pytest.raises(ValueError, match="segments must be one of"):
        Panel(
            series_ids=panel.series_ids,
            segments=("bogus",) * panel.n_series,
            dates=panel.dates,
            sales=panel.sales,
            price=panel.price,
            promo=panel.promo,
            stockout=panel.stockout,
        )


def test_demand_shape_is_checked(panel: Panel) -> None:
    with pytest.raises(ValueError, match="demand must have shape"):
        Panel(
            series_ids=panel.series_ids,
            segments=panel.segments,
            dates=panel.dates,
            sales=panel.sales,
            price=panel.price,
            promo=panel.promo,
            stockout=panel.stockout,
            demand=panel.sales[:, :5],
        )


def test_ragged_series_are_refused(frame: pd.DataFrame) -> None:
    ragged = frame.drop(frame.index[-1])
    with pytest.raises(ValueError, match="same contiguous date range"):
        Panel.from_frame(ragged, validate=False)


def test_series_starting_on_different_days_are_refused(frame: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    first = frame["series_id"].iloc[0]
    shifted = frame.copy()
    mask = shifted["series_id"] == first
    shifted.loc[mask, "date"] = shifted.loc[mask, "date"] + pd.Timedelta(days=1)
    # Same number of rows, same per-series length, but a different start day.
    shifted = shifted[shifted["date"] <= dates[-1]]
    extra = shifted[shifted["series_id"] == first].head(1).copy()
    extra["date"] = dates[-1]
    with pytest.raises(ValueError, match="same"):
        Panel.from_frame(pd.concat([shifted, extra], ignore_index=True), validate=False)


def test_a_panel_without_observations_reports_it() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    empty = Panel(
        series_ids=("S1",),
        segments=("smooth",),
        dates=dates,
        sales=np.full((1, 3), np.nan),
        price=np.ones((1, 3)),
        promo=np.zeros((1, 3)),
        stockout=np.zeros((1, 3)),
    )
    with pytest.raises(ValueError, match="no observed sales"):
        _ = empty.last_observed_day
