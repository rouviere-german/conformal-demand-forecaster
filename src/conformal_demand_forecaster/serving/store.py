"""DuckDB forecast store: the sales panel, issued forecasts and ingested actuals.

One connection guarded by a lock (DuckDB connections are not thread-safe and the API serves
sync endpoints from a thread pool). All statements are parameterised; identifiers are fixed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["ForecastStore"]

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS sales (
        series_id VARCHAR NOT NULL,
        date DATE NOT NULL,
        sales INTEGER NOT NULL,
        price DOUBLE NOT NULL,
        promo TINYINT NOT NULL,
        stockout TINYINT NOT NULL,
        segment VARCHAR NOT NULL,
        PRIMARY KEY (series_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS forecasts (
        series_id VARCHAR NOT NULL,
        origin_date DATE NOT NULL,
        target_date DATE NOT NULL,
        horizon INTEGER NOT NULL,
        segment VARCHAR NOT NULL,
        point DOUBLE NOT NULL,
        q0_050 DOUBLE NOT NULL,
        q0_100 DOUBLE NOT NULL,
        q0_500 DOUBLE NOT NULL,
        q0_900 DOUBLE NOT NULL,
        q0_950 DOUBLE NOT NULL,
        lower_80 DOUBLE NOT NULL,
        upper_80 DOUBLE NOT NULL,
        lower_95 DOUBLE NOT NULL,
        upper_95 DOUBLE NOT NULL,
        alpha_eff_80 DOUBLE NOT NULL,
        alpha_eff_95 DOUBLE NOT NULL,
        saturated_80 BOOLEAN NOT NULL,
        saturated_95 BOOLEAN NOT NULL,
        calibration_version VARCHAR NOT NULL,
        model_version VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL,
        PRIMARY KEY (series_id, origin_date, target_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS actuals (
        series_id VARCHAR NOT NULL,
        date DATE NOT NULL,
        sales INTEGER NOT NULL,
        status VARCHAR NOT NULL,
        z_score DOUBLE,
        actor VARCHAR NOT NULL,
        ingested_at TIMESTAMP NOT NULL,
        PRIMARY KEY (series_id, date)
    )
    """,
)
FORECAST_COLUMNS = (
    "series_id",
    "origin_date",
    "target_date",
    "horizon",
    "segment",
    "point",
    "q0_050",
    "q0_100",
    "q0_500",
    "q0_900",
    "q0_950",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
    "alpha_eff_80",
    "alpha_eff_95",
    "saturated_80",
    "saturated_95",
    "calibration_version",
    "model_version",
    "created_at",
)


class ForecastStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = duckdb.connect(str(path))
        with self._lock:
            for statement in _SCHEMA:
                self._conn.execute(statement)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- sales panel -----------------------------------------------------------------------

    def write_panel(self, frame: pd.DataFrame) -> int:
        """Insert or replace panel rows (columns of the sales contract; ``demand`` is dropped)."""
        data = frame[["series_id", "date", "sales", "price", "promo", "stockout", "segment"]].copy()
        data["date"] = pd.to_datetime(data["date"]).dt.date
        with self._lock:
            self._conn.register("panel_df", data)
            self._conn.execute(
                "INSERT OR REPLACE INTO sales SELECT series_id, date, sales, price,"
                " promo, stockout, segment FROM panel_df"
            )
            self._conn.unregister("panel_df")
        return len(data)

    def load_panel(self) -> pd.DataFrame:
        with self._lock:
            frame = self._conn.execute(
                "SELECT series_id, date, sales, price, promo, stockout, segment FROM sales"
                " ORDER BY series_id, date"
            ).df()
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ("sales", "promo", "stockout"):
            frame[column] = frame[column].astype(np.int64)
        frame["series_id"] = frame["series_id"].astype(str)
        frame["segment"] = frame["segment"].astype(str)
        return frame

    def series_segments(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT series_id, any_value(segment) FROM sales"
                " GROUP BY series_id ORDER BY series_id"
            ).fetchall()
        return {str(series): str(segment) for series, segment in rows}

    def last_sales_date(self) -> date | None:
        with self._lock:
            row = self._conn.execute("SELECT max(date) FROM sales").fetchone()
        value = row[0] if row else None
        return value if isinstance(value, date) else None

    def recent_sales(self, series_id: str, days: int) -> npt.NDArray[np.float64]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT sales FROM sales WHERE series_id = ? ORDER BY date DESC LIMIT ?",
                [series_id, int(days)],
            ).fetchall()
        return np.asarray([float(r[0]) for r in reversed(rows)], dtype=np.float64)

    def append_sales(self, rows: Iterable[dict[str, Any]]) -> int:
        """Append accepted actuals as new panel days (price carried forward per series)."""
        payload = list(rows)
        if not payload:
            return 0
        with self._lock:
            for row in payload:
                price = self._conn.execute(
                    "SELECT price FROM sales WHERE series_id = ? ORDER BY date DESC LIMIT 1",
                    [row["series_id"]],
                ).fetchone()
                self._conn.execute(
                    "INSERT OR REPLACE INTO sales VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        row["series_id"],
                        row["date"],
                        int(row["sales"]),
                        float(price[0]) if price else float(row.get("price", 1.0)),
                        int(row.get("promo", 0)),
                        int(row.get("stockout", 0)),
                        row["segment"],
                    ],
                )
        return len(payload)

    # --- forecasts -------------------------------------------------------------------------

    def write_forecasts(self, frame: pd.DataFrame) -> int:
        data = frame.copy()
        data.columns = [str(c).replace(".", "_") for c in data.columns]
        missing = [c for c in FORECAST_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"forecast frame is missing columns: {missing}")
        data = data[list(FORECAST_COLUMNS)]
        data["origin_date"] = pd.to_datetime(data["origin_date"]).dt.date
        data["target_date"] = pd.to_datetime(data["target_date"]).dt.date
        with self._lock:
            self._conn.register("forecast_df", data)
            self._conn.execute("INSERT OR REPLACE INTO forecasts SELECT * FROM forecast_df")
            self._conn.unregister("forecast_df")
        return len(data)

    def latest_origin(self) -> date | None:
        with self._lock:
            row = self._conn.execute("SELECT max(origin_date) FROM forecasts").fetchone()
        value = row[0] if row else None
        return value if isinstance(value, date) else None

    def latest_forecast(self, series_id: str, horizon: int) -> dict[str, Any] | None:
        """The forecast for ``horizon`` days after the most recent origin of ``series_id``."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM forecasts WHERE series_id = ? AND horizon = ?"
                " ORDER BY origin_date DESC LIMIT 1",
                [series_id, int(horizon)],
            )
            row = cursor.fetchone()
            names = [item[0] for item in cursor.description or []]
        if row is None:
            return None
        return dict(zip(names, row, strict=True))

    def forecasts_for_targets(self, pairs: Sequence[tuple[str, date]]) -> pd.DataFrame:
        """Most recently issued forecast for each (series, target date) pair."""
        if not pairs:
            return pd.DataFrame(columns=list(FORECAST_COLUMNS))
        keys = pd.DataFrame(
            {"series_id": [p[0] for p in pairs], "target_date": [p[1] for p in pairs]}
        )
        with self._lock:
            self._conn.register("keys_df", keys)
            frame = self._conn.execute(
                "SELECT f.* FROM forecasts f JOIN keys_df k"
                " ON f.series_id = k.series_id AND f.target_date = k.target_date"
                " QUALIFY row_number() OVER (PARTITION BY f.series_id, f.target_date"
                " ORDER BY f.origin_date DESC) = 1"
            ).df()
            self._conn.unregister("keys_df")
        frame["series_id"] = frame["series_id"].astype(str)
        return frame

    # --- actuals ---------------------------------------------------------------------------

    def write_actuals(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = list(rows)
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            for row in payload:
                self._conn.execute(
                    "INSERT OR REPLACE INTO actuals VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        row["series_id"],
                        row["date"],
                        int(row["sales"]),
                        row["status"],
                        row.get("z_score"),
                        row["actor"],
                        now,
                    ],
                )
        return len(payload)

    def actuals(self, status: str | None = None) -> pd.DataFrame:
        query = "SELECT series_id, date, sales, status, z_score, actor FROM actuals"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        with self._lock:
            frame = self._conn.execute(query + " ORDER BY series_id, date", params).df()
        frame["series_id"] = frame["series_id"].astype(str)
        return frame

    def counts(self) -> dict[str, int]:
        with self._lock:
            sales = self._conn.execute("SELECT count(*) FROM sales").fetchone()
            forecasts = self._conn.execute("SELECT count(*) FROM forecasts").fetchone()
            actuals = self._conn.execute("SELECT count(*) FROM actuals").fetchone()
        return {
            "sales": int(sales[0]) if sales else 0,
            "forecasts": int(forecasts[0]) if forecasts else 0,
            "actuals": int(actuals[0]) if actuals else 0,
        }
