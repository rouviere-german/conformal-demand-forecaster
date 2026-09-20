"""No-lookahead guard for the feature builder.

A feature row for (series ``s``, origin ``t``, horizon ``h``) may read the panel only up to day
``t``, plus the known-in-advance covariates of the target day ``t + h`` (planned price, promotion
flag, calendar position). The tests below *perturb the future* and assert the row is byte-for-byte
unchanged; a leak of any kind makes them fail. This is the data-integrity control the whole
calibration story rests on: a leaking feature inflates in-sample accuracy and silently destroys
the conformal coverage guarantee on live data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conformal_demand_forecaster.data.generator import GeneratorConfig, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.features.builder import (
    FEATURE_NAMES,
    HISTORY_FEATURES,
    TARGET_FEATURES,
    FeatureBuilder,
)

pytestmark = pytest.mark.unit

ORIGIN = 200
HORIZONS = (1, 7, 14, 28)


@pytest.fixture(scope="module")
def panel() -> Panel:
    frame = generate_panel(GeneratorConfig(n_series=6, n_days=260, seed=99, shifts=()))
    return Panel.from_frame(frame, validate=False)


def _rows(panel: Panel) -> pd.DataFrame:
    return FeatureBuilder(panel).rows([ORIGIN], list(HORIZONS)).X


def _perturbed(panel: Panel, **arrays: np.ndarray) -> Panel:
    return Panel(
        series_ids=panel.series_ids,
        segments=panel.segments,
        dates=panel.dates,
        sales=arrays.get("sales", panel.sales),
        price=arrays.get("price", panel.price),
        promo=arrays.get("promo", panel.promo),
        stockout=arrays.get("stockout", panel.stockout),
        demand=panel.demand,
    )


def test_future_sales_never_reach_a_feature_row(panel: Panel) -> None:
    """Rewriting every sale after the origin leaves all features identical."""
    sales = panel.sales.copy()
    sales[:, ORIGIN + 1 :] = 9_999.0
    pd.testing.assert_frame_equal(_rows(panel), _rows(_perturbed(panel, sales=sales)))


def test_future_stockouts_never_reach_a_feature_row(panel: Panel) -> None:
    stockout = panel.stockout.copy()
    stockout[:, ORIGIN + 1 :] = 1.0
    pd.testing.assert_frame_equal(_rows(panel), _rows(_perturbed(panel, stockout=stockout)))


def test_prices_and_promotions_of_other_days_do_not_leak(panel: Panel) -> None:
    """Only the target days themselves may contribute their planned price and promotion."""
    price = panel.price.copy()
    promo = panel.promo.copy()
    targets = {ORIGIN + h for h in HORIZONS}
    for day in range(ORIGIN + 1, panel.n_days):
        if day not in targets:
            price[:, day] = 123.45
            promo[:, day] = 1.0
    pd.testing.assert_frame_equal(_rows(panel), _rows(_perturbed(panel, price=price, promo=promo)))


def test_target_day_price_does_change_the_row(panel: Panel) -> None:
    """Sanity check on the guard: a change that *is* allowed must actually move the features."""
    price = panel.price.copy()
    price[:, ORIGIN + 7] *= 0.5
    changed = _rows(_perturbed(panel, price=price))
    base = _rows(panel)
    assert not changed["price_ratio_target"].equals(base["price_ratio_target"])
    # history features are untouched by the future price change
    pd.testing.assert_frame_equal(changed[list(HISTORY_FEATURES)], base[list(HISTORY_FEATURES)])


@given(
    origin=st.integers(min_value=120, max_value=230),
    horizon=st.integers(min_value=1, max_value=28),
    fill=st.floats(min_value=0.0, max_value=5_000.0),
)
def test_no_lookahead_for_any_origin_and_horizon(
    panel: Panel, origin: int, horizon: int, fill: float
) -> None:
    """Property form of the guard over arbitrary origins, horizons and replacement values."""
    builder = FeatureBuilder(panel)
    base = builder.rows([origin], [horizon]).X
    sales = panel.sales.copy()
    sales[:, origin + 1 :] = fill
    stockout = panel.stockout.copy()
    stockout[:, origin + 1 :] = 1.0 - stockout[:, origin + 1 :]
    poisoned = FeatureBuilder(_perturbed(panel, sales=sales, stockout=stockout))
    pd.testing.assert_frame_equal(base, poisoned.rows([origin], [horizon]).X)


def test_history_features_only_use_days_up_to_the_origin(panel: Panel) -> None:
    """Changing one day at or before the origin must change at least one history feature."""
    sales = panel.sales.copy()
    sales[:, ORIGIN] += 50.0
    changed = _rows(_perturbed(panel, sales=sales))
    assert not changed["lag0"].equals(_rows(panel)["lag0"])


def test_feature_names_are_the_declared_union(panel: Panel) -> None:
    frame = _rows(panel)
    assert tuple(frame.columns) == FEATURE_NAMES
    assert set(HISTORY_FEATURES) | set(TARGET_FEATURES) <= set(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_rows_carry_targets_and_bookkeeping(panel: Panel) -> None:
    rows = FeatureBuilder(panel).rows([ORIGIN], [1, 2])
    assert len(rows) == panel.n_series * 2
    assert np.all(rows.target_day == rows.origin + rows.horizon)
    assert rows.y_demand is not None
    assert np.all(rows.y_sales <= rows.y_demand)
    subset = rows.subset(rows.horizon == 1)
    assert len(subset) == panel.n_series
    assert np.all(subset.horizon == 1)
    assert subset.y_demand is not None


def test_rows_reject_impossible_requests(panel: Panel) -> None:
    builder = FeatureBuilder(panel)
    with pytest.raises(ValueError, match="horizons must be >= 1"):
        builder.rows([ORIGIN], [0])
    with pytest.raises(ValueError, match="origins must lie inside the panel"):
        builder.rows([panel.n_days], [1])
    with pytest.raises(ValueError, match="exceeds the panel"):
        builder.rows([panel.n_days - 1], [5])


def test_future_targets_are_available_after_extending_the_panel(panel: Panel) -> None:
    extended = panel.extend(30)
    rows = FeatureBuilder(extended).rows([panel.n_days - 1], [1, 28])
    assert np.all(np.isnan(rows.y_sales))
    assert rows.observed.sum() == 0
    assert not rows.X[list(HISTORY_FEATURES)].isna().all(axis=None)


def test_days_since_sale_is_capped_and_counts_from_the_last_sale() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    sales = np.zeros((1, 10))
    sales[0, 2] = 5.0
    panel = Panel(
        series_ids=("S1",),
        segments=("smooth",),
        dates=dates,
        sales=sales,
        price=np.ones((1, 10)),
        promo=np.zeros((1, 10)),
        stockout=np.zeros((1, 10)),
    )
    builder = FeatureBuilder(panel)
    rows = builder.rows([2, 5], [1])
    assert rows.X["days_since_sale"].tolist() == [0.0, 3.0]
