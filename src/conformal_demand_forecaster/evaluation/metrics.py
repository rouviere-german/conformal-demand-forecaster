"""Accuracy, calibration and interval-quality metrics.

- MASE (Hyndman and Koehler 2006): MAE divided by the in-sample seasonal-naive MAE of the series.
- Pinball loss for quantile ``q``: ``mean(max(q (y - p), (q - 1)(y - p)))``; the scaled variant
  divides by the same seasonal-naive scale, and the weighted aggregate weights series by their
  recent sales volume (the M5 competition's SPL/WSPL construction).
- Empirical coverage and mean width of an interval; Winkler score (Winkler 1972) penalises
  misses by ``2/alpha`` times the distance outside the band.
- Newsvendor cost: holding cost per unit over-ordered plus stockout cost per unit short.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "coverage",
    "mase",
    "mean_width",
    "newsvendor_cost",
    "pinball",
    "scaled_pinball",
    "winkler",
]

FloatArray = npt.NDArray[np.float64]


def _arr(values: npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64).ravel()


def coverage(y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    y_arr, lo, hi = _arr(y), _arr(lower), _arr(upper)
    if y_arr.size == 0:
        return float("nan")
    return float(np.mean((y_arr >= lo) & (y_arr <= hi)))


def mean_width(lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    lo, hi = _arr(lower), _arr(upper)
    return float(np.mean(hi - lo)) if lo.size else float("nan")


def winkler(y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, alpha: float) -> float:
    y_arr, lo, hi = _arr(y), _arr(lower), _arr(upper)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    if y_arr.size == 0:
        return float("nan")
    width = hi - lo
    below = np.maximum(lo - y_arr, 0.0)
    above = np.maximum(y_arr - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def pinball(y: npt.ArrayLike, prediction: npt.ArrayLike, q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly inside (0, 1)")
    diff = _arr(y) - _arr(prediction)
    if diff.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def mase(y: npt.ArrayLike, prediction: npt.ArrayLike, scale: float) -> float:
    """MAE over the seasonal-naive scale; NaN when the scale is zero or undefined."""
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    y_arr, p = _arr(y), _arr(prediction)
    if y_arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_arr - p)) / scale)


def scaled_pinball(
    y: npt.ArrayLike, quantiles: npt.ArrayLike, levels: Sequence[float], scale: float
) -> float:
    """Mean pinball loss over ``levels`` divided by the seasonal-naive scale (NaN if undefined)."""
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    matrix = np.asarray(quantiles, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(levels):
        raise ValueError("quantiles must be (n, len(levels))")
    losses = [pinball(y, matrix[:, j], q) for j, q in enumerate(levels)]
    return float(np.mean(losses) / scale)


def newsvendor_cost(
    y: npt.ArrayLike, order: npt.ArrayLike, holding_cost: float, stockout_cost: float
) -> float:
    """Mean realized cost of ordering ``order`` when demand turns out to be ``y``."""
    if holding_cost <= 0.0 or stockout_cost <= 0.0:
        raise ValueError("costs must be positive")
    y_arr, q = _arr(y), _arr(order)
    if y_arr.size == 0:
        return float("nan")
    over = np.maximum(q - y_arr, 0.0)
    short = np.maximum(y_arr - q, 0.0)
    return float(np.mean(holding_cost * over + stockout_cost * short))
