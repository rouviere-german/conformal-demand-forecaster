"""The DuckDB forecast store.

Every statement is parameterised, so the tests below feed hostile-looking identifiers through the
same paths a caller would and assert they are treated as data, not SQL.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformal_demand_forecaster.serving.store import FORECAST_COLUMNS, ForecastStore

pytestmark = pytest.mark.unit


def panel_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "series_id": np.repeat(["S0001_ST00", "S0002_ST00"], 10),
            "date": np.tile(dates.to_numpy(), 2),
            "sales": np.tile(np.arange(10), 2).astype(np.int64),
            "demand": np.tile(np.arange(10), 2).astype(np.int64),
            "price": np.full(20, 3.5),
            "promo": np.zeros(20, dtype=np.int64),
            "stockout": np.zeros(20, dtype=np.int64),
            "segment": np.repeat(["smooth", "intermittent"], 10),
        }
    )


def forecast_frame(origin: str = "2024-01-10", n: int = 2) -> pd.DataFrame:
    origin_date = pd.Timestamp(origin)
    rows = {
        "series_id": ["S0001_ST00"] * n,
        "origin_date": [origin_date] * n,
        "target_date": [origin_date + pd.Timedelta(days=h) for h in range(1, n + 1)],
        "horizon": list(range(1, n + 1)),
        "segment": ["smooth"] * n,
        "point": [5.0] * n,
        "q0.050": [1.0] * n,
        "q0.100": [2.0] * n,
        "q0.500": [5.0] * n,
        "q0.900": [8.0] * n,
        "q0.950": [9.0] * n,
        "lower_80": [1.5] * n,
        "upper_80": [8.5] * n,
        "lower_95": [0.5] * n,
        "upper_95": [9.5] * n,
        "alpha_eff_80": [0.2] * n,
        "alpha_eff_95": [0.05] * n,
        "saturated_80": [False] * n,
        "saturated_95": [False] * n,
        "calibration_version": ["c" * 64] * n,
        "model_version": ["m" * 64] * n,
        "created_at": [datetime(2024, 1, 10, 12, 0, tzinfo=UTC).replace(tzinfo=None)] * n,
    }
    return pd.DataFrame(rows)


@pytest.fixture
def store() -> Iterator[ForecastStore]:
    instance = ForecastStore()
    yield instance
    instance.close()


def test_panel_round_trips_and_drops_the_latent_demand_column(store: ForecastStore) -> None:
    assert store.write_panel(panel_frame()) == 20
    loaded = store.load_panel()
    assert len(loaded) == 20
    assert "demand" not in loaded.columns
    assert loaded["sales"].dtype == np.int64
    assert store.series_segments() == {"S0001_ST00": "smooth", "S0002_ST00": "intermittent"}
    assert store.last_sales_date() == date(2024, 1, 10)


def test_rewriting_the_panel_replaces_rows_instead_of_duplicating(store: ForecastStore) -> None:
    store.write_panel(panel_frame())
    store.write_panel(panel_frame())
    assert store.counts()["sales"] == 20


def test_recent_sales_returns_the_tail_in_chronological_order(store: ForecastStore) -> None:
    store.write_panel(panel_frame())
    recent = store.recent_sales("S0001_ST00", 3)
    assert recent.tolist() == [7.0, 8.0, 9.0]
    assert store.recent_sales("unknown", 5).size == 0


def test_appending_actuals_carries_the_last_price_forward(store: ForecastStore) -> None:
    store.write_panel(panel_frame())
    appended = store.append_sales(
        [
            {
                "series_id": "S0001_ST00",
                "date": date(2024, 1, 11),
                "sales": 42,
                "stockout": 0,
                "segment": "smooth",
            }
        ]
    )
    assert appended == 1
    loaded = store.load_panel()
    new_row = loaded[loaded["date"] == pd.Timestamp("2024-01-11")]
    assert new_row["sales"].iloc[0] == 42
    assert new_row["price"].iloc[0] == pytest.approx(3.5)
    assert store.append_sales([]) == 0


def test_appending_an_unknown_series_uses_the_supplied_price(store: ForecastStore) -> None:
    store.append_sales(
        [
            {
                "series_id": "S9999_ST00",
                "date": date(2024, 2, 1),
                "sales": 1,
                "segment": "smooth",
                "price": 7.25,
            }
        ]
    )
    assert store.load_panel()["price"].iloc[0] == pytest.approx(7.25)


def test_forecasts_round_trip_and_normalize_dotted_column_names(store: ForecastStore) -> None:
    assert store.write_forecasts(forecast_frame()) == 2
    assert store.latest_origin() == date(2024, 1, 10)
    record = store.latest_forecast("S0001_ST00", 1)
    assert record is not None
    assert set(FORECAST_COLUMNS) <= set(record)
    assert record["q0_500"] == pytest.approx(5.0)
    assert store.latest_forecast("S0001_ST00", 99) is None
    assert store.latest_forecast("unknown", 1) is None


def test_a_forecast_frame_missing_columns_is_refused(store: ForecastStore) -> None:
    incomplete = forecast_frame().drop(columns=["lower_95"])
    with pytest.raises(ValueError, match="missing columns"):
        store.write_forecasts(incomplete)


def test_latest_forecast_prefers_the_most_recent_origin(store: ForecastStore) -> None:
    store.write_forecasts(forecast_frame("2024-01-10"))
    later = forecast_frame("2024-01-11")
    later["point"] = 99.0
    store.write_forecasts(later)
    record = store.latest_forecast("S0001_ST00", 1)
    assert record is not None
    assert record["point"] == pytest.approx(99.0)


def test_forecasts_for_targets_picks_one_row_per_pair(store: ForecastStore) -> None:
    store.write_forecasts(forecast_frame("2024-01-09"))
    store.write_forecasts(forecast_frame("2024-01-10"))
    pairs = [("S0001_ST00", date(2024, 1, 11))]
    frame = store.forecasts_for_targets(pairs)
    assert len(frame) == 1
    assert pd.Timestamp(frame["origin_date"].iloc[0]) == pd.Timestamp("2024-01-10")
    assert store.forecasts_for_targets([]).empty


def test_actuals_are_stored_with_their_guard_verdict(store: ForecastStore) -> None:
    store.write_actuals(
        [
            {
                "series_id": "S0001_ST00",
                "date": date(2024, 1, 11),
                "sales": 7,
                "status": "accepted",
                "z_score": 0.5,
                "actor": "key:abc123",
            },
            {
                "series_id": "S0002_ST00",
                "date": date(2024, 1, 11),
                "sales": 9_000,
                "status": "quarantined",
                "z_score": 88.0,
                "actor": "key:abc123",
            },
        ]
    )
    assert len(store.actuals()) == 2
    accepted = store.actuals(status="accepted")
    assert len(accepted) == 1
    assert accepted["series_id"].iloc[0] == "S0001_ST00"
    assert store.counts()["actuals"] == 2


def test_identifiers_that_look_like_sql_are_treated_as_data(store: ForecastStore) -> None:
    store.write_panel(panel_frame())
    hostile = "'; DROP TABLE sales; --"
    assert store.recent_sales(hostile, 5).size == 0
    assert store.latest_forecast(hostile, 1) is None
    assert store.forecasts_for_targets([(hostile, date(2024, 1, 11))]).empty
    assert store.actuals(status=hostile).empty
    assert store.counts()["sales"] == 20  # the table is still there


def test_counts_start_at_zero(store: ForecastStore) -> None:
    assert store.counts() == {"sales": 0, "forecasts": 0, "actuals": 0}
    assert store.latest_origin() is None
    assert store.last_sales_date() is None


def test_a_file_backed_store_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "store.duckdb"
    first = ForecastStore(path)
    first.write_panel(panel_frame())
    first.close()
    second = ForecastStore(path)
    try:
        assert second.counts()["sales"] == 20
    finally:
        second.close()
