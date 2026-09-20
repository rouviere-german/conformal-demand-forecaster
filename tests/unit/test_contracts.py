"""Data-contract tests: what the pandera schemas accept, reject and reveal.

Two properties matter for security as well as correctness: the contract catches malformed or
hostile input before it reaches the model, and the error it raises summarizes *what* failed
without copying the rejected values, so a contract violation can be logged safely.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from conformal_demand_forecaster.ml.contracts import (
    MAX_PRICE,
    MAX_UNITS,
    SEGMENTS,
    DataContractError,
    validate_actuals,
    validate_panel,
)

pytestmark = pytest.mark.unit


def _with_value(column: pd.Series[Any], position: int, value: object) -> pd.Series[Any]:
    """Replace one entry with an arbitrary value, keeping the natural dtype where possible."""
    values = column.tolist()
    values[position] = value
    return pd.Series(values, index=column.index, name=column.name).infer_objects()


def make_panel(n_series: int = 2, n_days: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = {
        "series_id": np.repeat([f"S{i:04d}_ST00" for i in range(n_series)], n_days),
        "date": np.tile(dates.to_numpy(), n_series),
        "sales": np.tile(np.arange(n_days), n_series).astype(np.int64),
        "demand": np.tile(np.arange(n_days), n_series).astype(np.int64),
        "price": np.full(n_series * n_days, 4.5),
        "promo": np.zeros(n_series * n_days, dtype=np.int64),
        "stockout": np.zeros(n_series * n_days, dtype=np.int64),
        "segment": np.repeat(["smooth", "intermittent"][:n_series], n_days),
    }
    return pd.DataFrame(rows)


def test_a_clean_panel_validates_unchanged() -> None:
    frame = make_panel()
    validated = validate_panel(frame)
    pd.testing.assert_frame_equal(validated, frame)


def test_demand_is_optional_so_real_data_without_it_still_validates() -> None:
    frame = make_panel().drop(columns=["demand"])
    assert "demand" not in validate_panel(frame).columns


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("sales", -1, "greater_than_or_equal_to"),
        ("sales", MAX_UNITS + 1, "less_than_or_equal_to"),
        ("price", 0.0, "greater_than"),
        ("price", MAX_PRICE + 1.0, "less_than_or_equal_to"),
        ("promo", 2, "isin"),
        ("stockout", 7, "isin"),
        ("segment", "unknown", "isin"),
    ],
)
def test_out_of_range_values_are_rejected(column: str, value: object, check: str) -> None:
    frame = make_panel()
    frame[column] = _with_value(frame[column], 0, value)
    with pytest.raises(DataContractError) as info:
        validate_panel(frame)
    failures = info.value.failures
    assert any(item["column"] == column and check in str(item["check"]) for item in failures)


def test_series_ids_must_match_the_identifier_pattern() -> None:
    frame = make_panel()
    frame.at[0, "series_id"] = "../../etc/passwd"
    with pytest.raises(DataContractError, match="data contract violated"):
        validate_panel(frame)


def test_extra_columns_are_refused() -> None:
    frame = make_panel()
    frame["injected"] = 1
    with pytest.raises(DataContractError):
        validate_panel(frame)


def test_duplicate_series_day_rows_are_refused() -> None:
    frame = pd.concat([make_panel(), make_panel().head(1)], ignore_index=True)
    with pytest.raises(DataContractError):
        validate_panel(frame)


def test_a_missing_day_breaks_date_continuity() -> None:
    frame = make_panel(n_series=1, n_days=6).drop(index=3).reset_index(drop=True)
    with pytest.raises(DataContractError) as info:
        validate_panel(frame)
    assert any("consecutive" in str(item["check"]) for item in info.value.failures)


def test_sales_may_not_exceed_latent_demand() -> None:
    frame = make_panel()
    frame.at[1, "sales"] = int(frame["demand"].iloc[1]) + 5
    with pytest.raises(DataContractError) as info:
        validate_panel(frame)
    assert any("sales_never_exceed_demand" in str(item["check"]) for item in info.value.failures)


def test_uncensored_days_must_report_sales_equal_to_demand() -> None:
    frame = make_panel()
    frame.at[2, "sales"] = int(frame["demand"].iloc[2]) - 1
    with pytest.raises(DataContractError) as info:
        validate_panel(frame)
    assert any(
        "uncensored_sales_equal_demand" in str(item["check"]) for item in info.value.failures
    )
    frame.at[2, "stockout"] = 1
    validate_panel(frame)  # marking it censored makes the same row legal


def test_failures_summarize_without_leaking_the_rejected_values() -> None:
    frame = make_panel()
    secret = 4_242_424
    frame.at[0, "sales"] = secret
    frame.at[1, "price"] = -1.0
    with pytest.raises(DataContractError) as info:
        validate_panel(frame)
    error = info.value
    assert str(secret) not in str(error)
    assert all(set(item) == {"column", "check", "count"} for item in error.failures)
    assert sum(int(item["count"]) for item in error.failures) >= 2


def test_lazy_validation_reports_every_failing_check_at_once() -> None:
    frame = make_panel()
    frame.at[0, "sales"] = -1
    frame.at[1, "promo"] = 5
    with pytest.raises(DataContractError) as info:
        validate_panel(frame)
    columns = {item["column"] for item in info.value.failures}
    assert {"sales", "promo"} <= columns


def test_wrong_dtypes_are_rejected_without_coercion() -> None:
    frame = make_panel()
    frame["sales"] = frame["sales"].astype(float)
    with pytest.raises(DataContractError):
        validate_panel(frame)


# --- actuals --------------------------------------------------------------------------------


def make_actuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["S0001_ST00", "S0002_ST00"],
            "date": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "sales": np.array([3, 0], dtype=np.int64),
        }
    )


def test_actuals_accept_a_clean_batch_including_zeroes() -> None:
    validate_actuals(make_actuals())


@pytest.mark.parametrize(
    ("column", "value"),
    [("sales", -2), ("sales", MAX_UNITS + 1), ("series_id", "bad id!")],
)
def test_actuals_reject_bad_rows(column: str, value: object) -> None:
    frame = make_actuals()
    frame[column] = _with_value(frame[column], 0, value)
    with pytest.raises(DataContractError):
        validate_actuals(frame)


def test_actuals_reject_duplicates_and_extra_columns() -> None:
    duplicated = pd.concat([make_actuals(), make_actuals().head(1)], ignore_index=True)
    with pytest.raises(DataContractError):
        validate_actuals(duplicated)
    extra = make_actuals()
    extra["note"] = "x"
    with pytest.raises(DataContractError):
        validate_actuals(extra)


def test_segments_are_the_documented_pair() -> None:
    assert SEGMENTS == ("smooth", "intermittent")
