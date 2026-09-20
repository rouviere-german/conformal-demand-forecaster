"""pandera data contracts for the sales panel and for ingested actuals.

Failure summaries report column, check and count only. The failing values themselves are never
copied into the exception, so a contract error can be logged without leaking the data it
rejected.

``validate_actuals`` also takes the day window the serving state accepts, because an ingested
date is attacker-controlled and the panel is a dense (series x day) grid.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing.pandas import Series

__all__ = [
    "MAX_PRICE",
    "MAX_UNITS",
    "SEGMENTS",
    "SERIES_ID_PATTERN",
    "ActualsSchema",
    "DataContractError",
    "SalesPanelSchema",
    "validate_actuals",
    "validate_panel",
]

MAX_UNITS = 100_000
MAX_PRICE = 10_000.0
SEGMENTS = ("smooth", "intermittent")
SERIES_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"
_ONE_DAY = pd.Timedelta(days=1)


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


def _consecutive_days(frame: pd.DataFrame) -> Series[bool]:
    """True for every row whose date follows the previous date of the same series by one day."""
    ordered = frame.sort_values(["series_id", "date"])
    gaps = ordered.groupby("series_id", sort=False)["date"].diff()
    ok = gaps.isna() | (gaps == _ONE_DAY)
    return Series[bool](ok.reindex(frame.index))


class SalesPanelSchema(pa.DataFrameModel):
    """Daily sales panel: one row per series and day, days contiguous within a series.

    ``demand`` is the latent (uncensored) demand kept by the generator for scoring; real data has
    ``sales`` only, so the column is optional. Types are checked without coercion.
    """

    series_id: Series[str] = pa.Field(str_matches=SERIES_ID_PATTERN)
    date: Series[pa.DateTime]
    sales: Series[int] = pa.Field(ge=0, le=MAX_UNITS)
    demand: Series[int] | None = pa.Field(ge=0, le=MAX_UNITS, nullable=False)
    price: Series[float] = pa.Field(gt=0.0, le=MAX_PRICE)
    promo: Series[int] = pa.Field(isin=[0, 1])
    stockout: Series[int] = pa.Field(isin=[0, 1])
    segment: Series[str] = pa.Field(isin=list(SEGMENTS))

    class Config:
        strict = True
        name = "sales_panel"
        unique: ClassVar[list[str]] = ["series_id", "date"]

    @pa.dataframe_check
    @classmethod
    def dates_are_consecutive_within_series(cls, frame: pd.DataFrame) -> Series[bool]:
        return _consecutive_days(frame)

    @pa.dataframe_check
    @classmethod
    def sales_never_exceed_demand(cls, frame: pd.DataFrame) -> Series[bool]:
        if "demand" not in frame.columns:
            return Series[bool](pd.Series(True, index=frame.index))
        return Series[bool](frame["sales"] <= frame["demand"])

    @pa.dataframe_check
    @classmethod
    def uncensored_sales_equal_demand(cls, frame: pd.DataFrame) -> Series[bool]:
        if "demand" not in frame.columns:
            return Series[bool](pd.Series(True, index=frame.index))
        return Series[bool]((frame["stockout"] == 1) | (frame["sales"] == frame["demand"]))


class ActualsSchema(pa.DataFrameModel):
    """Actual sales ingested after the fact: series, day and non-negative units."""

    series_id: Series[str] = pa.Field(str_matches=SERIES_ID_PATTERN)
    date: Series[pa.DateTime]
    sales: Series[int] = pa.Field(ge=0, le=MAX_UNITS)

    class Config:
        strict = True
        name = "actuals"
        unique: ClassVar[list[str]] = ["series_id", "date"]


def _summarize(exc: SchemaErrors | SchemaError) -> DataContractError:
    if isinstance(exc, SchemaError):
        column = str(exc.schema.name if exc.schema is not None else "<frame>")
        return DataContractError([{"column": column, "check": str(exc.check)[:80], "count": 1}])
    cases = exc.failure_cases[["column", "check"]].astype(str)
    counts = cases.value_counts().sort_index()
    failures: list[dict[str, str | int]] = [
        {"column": column, "check": check, "count": int(count)}
        for (column, check), count in counts.items()
    ]
    return DataContractError(failures)


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized DataContractError."""
    try:
        return SalesPanelSchema.validate(frame, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        raise _summarize(exc) from None


def validate_actuals(
    frame: pd.DataFrame,
    *,
    min_date: pd.Timestamp | date | None = None,
    max_date: pd.Timestamp | date | None = None,
) -> pd.DataFrame:
    """Validate an actuals frame the same way, optionally inside an accepted day window.

    The schema alone cannot bound the day, because the window is the serving state: the panel
    starts somewhere and forecasts reach a horizon past its last observed day. The caller that
    knows the window passes it, and dates outside it are a contract violation rather than a
    quarantine -- they are rejected before anything downstream sees them.
    """
    try:
        validated = ActualsSchema.validate(frame, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        raise _summarize(exc) from None
    if min_date is None and max_date is None:
        return validated
    days = pd.to_datetime(validated["date"])
    outside = pd.Series(False, index=days.index)
    if min_date is not None:
        outside |= days < pd.Timestamp(min_date)
    if max_date is not None:
        outside |= days > pd.Timestamp(max_date)
    count = int(outside.sum())
    if count:
        raise DataContractError(
            [{"column": "date", "check": "within_ingest_window", "count": count}]
        )
    return validated
