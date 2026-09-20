"""Seeded M5-like generator: daily SKU-store demand with the pathologies forecasters meet.

Each series gets a base level, weekly and yearly seasonality, a mild trend, price elasticity with
promotion episodes, an intermittent or smooth demand process, scheduled regime shifts that hit a
subset of series, and stockout episodes that censor sales. Latent demand is kept so intervals can
be scored against the truth as well as against censored sales.

The process is deterministic for a given :class:`GeneratorConfig` (one ``numpy`` Generator seeded
from ``seed``; the draw order never depends on data), so the dataset hash in the run manifest is
reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from conformal_demand_forecaster.ml.contracts import MAX_UNITS, validate_panel

__all__ = ["GeneratorConfig", "RegimeShift", "generate_panel"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RegimeShift:
    """From ``day`` on, a random ``fraction`` of the series has its level multiplied."""

    day: int
    fraction: float
    multiplier: float

    def __post_init__(self) -> None:
        if self.day < 0:
            raise ValueError("shift day must be >= 0")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("shift fraction must lie in [0, 1]")
        if self.multiplier <= 0.0:
            raise ValueError("shift multiplier must be positive")


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    n_series: int = 200
    n_days: int = 1096
    start: date = date(2023, 1, 1)
    seed: int = 20260914
    intermittent_fraction: float = 0.4
    promo_start_probability: float = 0.02
    stockout_start_probability: float = 0.006
    shifts: tuple[RegimeShift, ...] = field(
        default_factory=lambda: (
            RegimeShift(day=730, fraction=0.5, multiplier=1.6),
            RegimeShift(day=900, fraction=0.3, multiplier=0.55),
        )
    )

    def __post_init__(self) -> None:
        if self.n_series < 1 or self.n_days < 60:
            raise ValueError("need at least 1 series and 60 days")
        if not 0.0 <= self.intermittent_fraction <= 1.0:
            raise ValueError("intermittent_fraction must lie in [0, 1]")
        if not 0.0 <= self.promo_start_probability <= 1.0:
            raise ValueError("promo_start_probability must lie in [0, 1]")
        if not 0.0 <= self.stockout_start_probability <= 1.0:
            raise ValueError("stockout_start_probability must lie in [0, 1]")


def _episodes(
    rng: np.random.Generator, n_days: int, start_probability: float, min_len: int, max_len: int
) -> IntArray:
    """0/1 indicator of non-overlapping episodes with geometric-like starts and uniform lengths."""
    active = np.zeros(n_days, dtype=np.int64)
    starts = rng.random(n_days) < start_probability
    lengths = rng.integers(min_len, max_len + 1, size=n_days)
    day = 0
    while day < n_days:
        if starts[day]:
            end = min(n_days, day + int(lengths[day]))
            active[day:end] = 1
            day = end
        else:
            day += 1
    return active


def _series_draws(rng: np.random.Generator, config: GeneratorConfig) -> dict[str, FloatArray]:
    """Per-series parameters, drawn in a fixed order."""
    n = config.n_series
    return {
        "intermittent": (rng.random(n) < config.intermittent_fraction).astype(np.float64),
        "level_smooth": np.exp(rng.normal(2.4, 0.6, size=n)),
        "level_intermittent": np.exp(rng.normal(0.9, 0.4, size=n)),
        "zero_probability": rng.uniform(0.45, 0.85, size=n),
        "dispersion": rng.uniform(3.0, 12.0, size=n),
        "yearly_amplitude": rng.uniform(0.0, 0.3, size=n),
        "yearly_phase": rng.uniform(0.0, 2.0 * math.pi, size=n),
        "trend_per_year": rng.normal(0.0, 0.08, size=n),
        "base_price": np.round(np.exp(rng.normal(2.3, 0.5, size=n)), 2),
        "elasticity": rng.uniform(0.5, 2.5, size=n),
    }


@dataclass(frozen=True, slots=True)
class _Calendar:
    """Calendar arrays shared by every series of one generator run."""

    dates: pd.DatetimeIndex
    day_index: FloatArray
    dow: IntArray
    doy: FloatArray


@dataclass(frozen=True, slots=True)
class _SeriesDraw:
    """One simulated series: latent demand, censored sales and its covariates."""

    demand: IntArray
    sales: IntArray
    price: FloatArray
    promo: IntArray
    stockout: IntArray


def _simulate_series(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    calendar: _Calendar,
    params: dict[str, FloatArray],
    *,
    index: int,
    dow_effect: FloatArray,
    shifted: Sequence[bool],
) -> _SeriesDraw:
    """Simulate one series; the draw order must not depend on any previously drawn value."""
    days = cfg.n_days
    intermittent = params["intermittent"][index] > 0.5
    level = params["level_intermittent" if intermittent else "level_smooth"][index]
    seasonal = 1.0 + params["yearly_amplitude"][index] * np.sin(
        2.0 * math.pi * calendar.doy / 365.25 + params["yearly_phase"][index]
    )
    trend = np.maximum(0.3, 1.0 + params["trend_per_year"][index] * calendar.day_index / 365.0)

    promo_days = _episodes(rng, days, cfg.promo_start_probability, 3, 10)
    discount = rng.uniform(0.10, 0.40) * promo_days
    series_price = np.round(params["base_price"][index] * (1.0 - discount), 2)
    price_effect = (series_price / params["base_price"][index]) ** (-params["elasticity"][index])

    shift_effect = np.ones(days)
    for shift, hit in zip(cfg.shifts, shifted, strict=True):
        if hit and shift.day < days:
            shift_effect[shift.day :] *= shift.multiplier
    mean = level * dow_effect[calendar.dow] * seasonal * trend * price_effect * shift_effect

    if intermittent:
        occurs = rng.random(days) >= params["zero_probability"][index]
        size = 1 + rng.poisson(mean / (1.0 - params["zero_probability"][index]), size=days)
        latent = np.where(occurs, size, 0)
    else:
        k = params["dispersion"][index]
        rate = rng.gamma(shape=k, scale=mean / k, size=days)
        latent = rng.poisson(rate, size=days)
    latent = np.minimum(latent, MAX_UNITS).astype(np.int64)

    out_days = _episodes(rng, days, cfg.stockout_start_probability, 2, 7)
    caps = rng.integers(0, 3, size=days)
    observed = np.where(out_days == 1, np.minimum(latent, caps), latent).astype(np.int64)
    # A stockout day only censors when it actually withheld units; otherwise it is ordinary.
    censored = ((out_days == 1) & (observed < latent)).astype(np.int64)
    return _SeriesDraw(
        demand=latent,
        sales=observed,
        price=series_price,
        promo=promo_days,
        stockout=censored,
    )


def generate_panel(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Return a contract-valid long panel.

    Columns: ``series_id, date, sales, demand, price, promo, stockout, segment``.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n, days = cfg.n_series, cfg.n_days
    params = _series_draws(rng, cfg)
    dow_effects = np.exp(rng.normal(0.0, 0.15, size=(n, 7)))
    dow_effects /= dow_effects.mean(axis=1, keepdims=True)
    shift_masks = [rng.random(n) < shift.fraction for shift in cfg.shifts]

    dates = pd.date_range(cfg.start, periods=days, freq="D")
    calendar = _Calendar(
        dates=dates,
        day_index=np.arange(days, dtype=np.float64),
        dow=dates.dayofweek.to_numpy().astype(np.int64),
        doy=dates.dayofyear.to_numpy().astype(np.float64),
    )

    sales = np.zeros((n, days), dtype=np.int64)
    demand = np.zeros((n, days), dtype=np.int64)
    price = np.zeros((n, days), dtype=np.float64)
    promo = np.zeros((n, days), dtype=np.int64)
    stockout = np.zeros((n, days), dtype=np.int64)
    for i in range(n):
        draw = _simulate_series(
            rng,
            cfg,
            calendar,
            params,
            index=i,
            dow_effect=dow_effects[i],
            shifted=[bool(mask[i]) for mask in shift_masks],
        )
        demand[i], sales[i] = draw.demand, draw.sales
        price[i], promo[i], stockout[i] = draw.price, draw.promo, draw.stockout

    width = max(4, len(str(n)))
    series_ids = np.array([f"S{i:0{width}d}_ST{(i % 10):02d}" for i in range(n)])
    segments = np.where(params["intermittent"] > 0.5, "intermittent", "smooth")
    frame = pd.DataFrame(
        {
            "series_id": np.repeat(series_ids, days),
            "date": np.tile(dates.to_numpy(), n),
            "sales": sales.ravel(),
            "demand": demand.ravel(),
            "price": price.ravel(),
            "promo": promo.ravel(),
            "stockout": stockout.ravel(),
            "segment": np.repeat(segments, days),
        }
    )
    return validate_panel(frame)
