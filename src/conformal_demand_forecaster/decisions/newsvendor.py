"""Newsvendor order quantities from conformalized quantiles.

With holding cost ``c_h`` per unit left over and stockout cost ``c_s`` per unit short, the
expected-cost-minimising order is the critical fractile ``F^{-1}(c_s / (c_s + c_h))`` of the
demand distribution (Arrow, Harris and Marschak 1951). The conformalized intervals give a few
points of that distribution: the median and, for each coverage level, a lower and an upper bound
interpreted as the ``alpha/2`` and ``1 - alpha/2`` quantiles. The order is read off the piecewise
linear interpolation of these points; fractiles beyond the outermost point are clipped to it,
which the model card lists as a known approximation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "NewsvendorDecision",
    "critical_fractile",
    "interpolate_quantile",
    "order_quantities",
    "order_quantity",
]

FloatArray = npt.NDArray[np.float64]


def critical_fractile(holding_cost: float, stockout_cost: float) -> float:
    if holding_cost <= 0.0 or stockout_cost <= 0.0:
        raise ValueError("costs must be positive")
    return stockout_cost / (stockout_cost + holding_cost)


def interpolate_quantile(
    levels: Sequence[float], values: Sequence[float], fractile: float
) -> float:
    """Linear interpolation of a monotone quantile function; clipped outside its support."""
    if len(levels) != len(values) or not levels:
        raise ValueError("levels and values must be non-empty and of equal length")
    if not 0.0 < fractile < 1.0:
        raise ValueError("fractile must lie strictly inside (0, 1)")
    order = np.argsort(np.asarray(levels, dtype=np.float64))
    xs = np.asarray(levels, dtype=np.float64)[order]
    ys = np.maximum.accumulate(np.asarray(values, dtype=np.float64)[order])
    return float(np.interp(fractile, xs, ys))


@dataclass(frozen=True, slots=True)
class NewsvendorDecision:
    fractile: float
    order: float
    order_units: int


def order_quantity(
    median: float,
    intervals: Mapping[float, tuple[float, float]],
    holding_cost: float,
    stockout_cost: float,
) -> NewsvendorDecision:
    """Order from the median plus ``{alpha: (lower, upper)}`` conformalized bounds."""
    fractile = critical_fractile(holding_cost, stockout_cost)
    levels: list[float] = [0.5]
    values: list[float] = [median]
    for alpha, (lower, upper) in intervals.items():
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        levels.extend([alpha / 2.0, 1.0 - alpha / 2.0])
        values.extend([lower, upper])
    order = max(0.0, interpolate_quantile(levels, values, fractile))
    return NewsvendorDecision(fractile=fractile, order=order, order_units=int(np.ceil(order)))


def order_quantities(
    median: FloatArray,
    intervals: Mapping[float, tuple[FloatArray, FloatArray]],
    holding_cost: float,
    stockout_cost: float,
) -> FloatArray:
    """Vectorised :func:`order_quantity` (continuous orders) for a batch of rows."""
    fractile = critical_fractile(holding_cost, stockout_cost)
    levels = [0.5]
    columns = [np.asarray(median, dtype=np.float64)]
    for alpha, (lower, upper) in intervals.items():
        levels.extend([alpha / 2.0, 1.0 - alpha / 2.0])
        columns.extend([np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)])
    order = np.argsort(levels)
    xs = np.asarray(levels, dtype=np.float64)[order]
    matrix = np.maximum.accumulate(np.stack(columns, axis=1)[:, order], axis=1)
    position = int(np.searchsorted(xs, fractile, side="right"))
    if position <= 0:
        values = matrix[:, 0]
    elif position >= xs.size:
        values = matrix[:, -1]
    else:
        x0, x1 = xs[position - 1], xs[position]
        weight = (fractile - x0) / (x1 - x0)
        values = matrix[:, position - 1] * (1.0 - weight) + matrix[:, position] * weight
    return np.asarray(np.maximum(values, 0.0), dtype=np.float64)
