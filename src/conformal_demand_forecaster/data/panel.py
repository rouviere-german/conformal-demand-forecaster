"""Wide-array container for a contract-valid sales panel.

All series share one contiguous daily date range (the contract guarantees continuity within a
series; ``from_frame`` requires that every series covers the same days). Days after the last
observed day can be appended with :meth:`Panel.extend` so features for future targets exist:
sales and demand become NaN, price is carried forward and promotions default to none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from conformal_demand_forecaster.ml.contracts import SEGMENTS, validate_panel

__all__ = ["Panel"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _fill_missing_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto the full (series x day) grid, leaving unobserved sales as NaN."""
    if frame.empty:
        return frame
    series_ids = sorted(str(item) for item in frame["series_id"].unique())
    dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    index = pd.MultiIndex.from_product([series_ids, dates], names=["series_id", "date"])
    indexed = frame.set_index(["series_id", "date"])
    if len(indexed) == len(index):
        return frame
    filled = indexed.reindex(index)
    grouped = filled.groupby(level=0, sort=False)
    filled["price"] = grouped["price"].ffill().bfill()
    filled["segment"] = grouped["segment"].ffill().bfill()
    for column in ("promo", "stockout"):
        filled[column] = filled[column].fillna(0.0)
    return filled.reset_index()


@dataclass(frozen=True, slots=True)
class Panel:
    series_ids: tuple[str, ...]
    segments: tuple[str, ...]
    dates: pd.DatetimeIndex
    sales: FloatArray
    price: FloatArray
    promo: FloatArray
    stockout: FloatArray
    demand: FloatArray | None = None

    def __post_init__(self) -> None:
        shape = (len(self.series_ids), len(self.dates))
        for name in ("sales", "price", "promo", "stockout"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.demand is not None and self.demand.shape != shape:
            raise ValueError(f"demand must have shape {shape}")
        if len(self.segments) != shape[0]:
            raise ValueError("one segment per series is required")
        if any(segment not in SEGMENTS for segment in self.segments):
            raise ValueError(f"segments must be one of {SEGMENTS}")

    @property
    def n_series(self) -> int:
        return len(self.series_ids)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def last_observed_day(self) -> int:
        """Index of the last day with observed sales (NaN days added by ``extend`` excluded)."""
        observed = np.flatnonzero(~np.all(np.isnan(self.sales), axis=0))
        if observed.size == 0:
            raise ValueError("panel has no observed sales")
        return int(observed[-1])

    @property
    def segment_codes(self) -> IntArray:
        return np.array([SEGMENTS.index(segment) for segment in self.segments], dtype=np.int64)

    def day_index(self, day: pd.Timestamp) -> int:
        position = self.dates.get_indexer(pd.DatetimeIndex([day]))[0]
        if position < 0:
            raise KeyError(f"{day.date()} is outside the panel")
        return int(position)

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, *, validate: bool = True, fill_missing_days: bool = False
    ) -> Panel:
        """Build a panel from a long frame.

        ``fill_missing_days`` reindexes every series onto the full daily range and leaves the
        sales of days it has no row for as NaN. The serving store needs this: when the outlier
        guard quarantines one series' day, that series simply has no observation for it, while
        every other series moves on. Prices are carried forward, promotions and stockout flags
        default to none, and the NaN sales flow through the feature builder as missing history.
        """
        data = validate_panel(frame) if validate else frame
        if fill_missing_days:
            data = _fill_missing_days(data)
        data = data.sort_values(["series_id", "date"], kind="stable")
        series_ids = tuple(str(item) for item in data["series_id"].unique())
        dates = pd.DatetimeIndex(sorted(data["date"].unique()))
        n_series, n_days = len(series_ids), len(dates)
        if len(data) != n_series * n_days:
            raise ValueError("every series must cover the same contiguous date range")
        starts = data.groupby("series_id", sort=True)["date"].min()
        if not bool((starts == dates[0]).all()):
            raise ValueError("every series must start on the same day")

        def wide(column: str) -> FloatArray:
            return data[column].to_numpy(dtype=np.float64).reshape(n_series, n_days)

        segments = tuple(
            str(item) for item in data.groupby("series_id", sort=True)["segment"].first()
        )
        return cls(
            series_ids=series_ids,
            segments=segments,
            dates=dates,
            sales=wide("sales"),
            price=wide("price"),
            promo=wide("promo"),
            stockout=wide("stockout"),
            demand=wide("demand") if "demand" in data.columns else None,
        )

    def to_frame(self) -> pd.DataFrame:
        """Long frame of the observed days (NaN-sales days added by ``extend`` are dropped)."""
        n = self.n_series
        frame = pd.DataFrame(
            {
                "series_id": np.repeat(np.array(self.series_ids), self.n_days),
                "date": np.tile(self.dates.to_numpy(), n),
                "sales": self.sales.ravel(),
                "price": self.price.ravel(),
                "promo": self.promo.ravel(),
                "stockout": self.stockout.ravel(),
                "segment": np.repeat(np.array(self.segments), self.n_days),
            }
        )
        if self.demand is not None:
            frame.insert(3, "demand", self.demand.ravel())
        frame = frame[~frame["sales"].isna()].copy()
        for column in (
            "sales",
            "promo",
            "stockout",
            *(["demand"] if self.demand is not None else []),
        ):
            frame[column] = frame[column].astype(np.int64)
        return frame.reset_index(drop=True)

    def extend(self, days: int) -> Panel:
        """Append ``days`` future days with unknown sales and carried-forward prices."""
        if days < 1:
            return self
        n = self.n_series
        future = np.full((n, days), np.nan)
        return Panel(
            series_ids=self.series_ids,
            segments=self.segments,
            dates=pd.DatetimeIndex(
                self.dates.append(
                    pd.date_range(self.dates[-1] + pd.Timedelta(days=1), periods=days, freq="D")
                )
            ),
            sales=np.concatenate([self.sales, future], axis=1),
            price=np.concatenate([self.price, np.repeat(self.price[:, -1:], days, axis=1)], axis=1),
            promo=np.concatenate([self.promo, np.zeros((n, days))], axis=1),
            stockout=np.concatenate([self.stockout, np.zeros((n, days))], axis=1),
            demand=None if self.demand is None else np.concatenate([self.demand, future], axis=1),
        )

    def truncate(self, n_days: int) -> Panel:
        """Keep only the first ``n_days`` days."""
        if not 1 <= n_days <= self.n_days:
            raise ValueError("n_days must lie within the panel")
        return Panel(
            series_ids=self.series_ids,
            segments=self.segments,
            dates=self.dates[:n_days],
            sales=self.sales[:, :n_days],
            price=self.price[:, :n_days],
            promo=self.promo[:, :n_days],
            stockout=self.stockout[:, :n_days],
            demand=None if self.demand is None else self.demand[:, :n_days],
        )

    def with_sales(self, sales: FloatArray, stockout: FloatArray | None = None) -> Panel:
        """Return a copy with replaced sales (and optionally stockout) arrays."""
        return Panel(
            series_ids=self.series_ids,
            segments=self.segments,
            dates=self.dates,
            sales=np.asarray(sales, dtype=np.float64),
            price=self.price,
            promo=self.promo,
            stockout=self.stockout if stockout is None else np.asarray(stockout, dtype=np.float64),
            demand=self.demand,
        )
