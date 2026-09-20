"""Robust z-score guard for ingested actuals.

An ingestor key can feed the online updates (ACI alpha, calibration scores, coverage monitor).
A poisoned spike would drag the alpha state down and widen every interval for that group, so
each incoming value is compared with the series' recent history: ``z = (y - median) / scale``
with ``scale = max(1.4826 * MAD, sqrt(max(median, p90) + 1))``.

The floor is a Poisson-like spread for count data. It is anchored on the 90th percentile when
that is above the median, which is what makes the guard usable on intermittent series: a series
that sells on one day in four has a median of 0 and a MAD of 0, so a median-only floor would
score an ordinary sale of ten units at z = 10 and quarantine it. A percentile is used rather
than the mean because the mean is itself movable by the spikes this guard exists to catch; the
90th tolerates roughly a tenth of the window being poisoned before it starts to drift, which is
the guard's documented contamination limit. Only the upper tail is tested: a sudden zero is a
real event
(closure, stockout) that must reach the monitor. Values above the contract maximum are rejected
outright. Quarantined values are stored for review and never touch the online state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from conformal_demand_forecaster.ml.contracts import MAX_UNITS

__all__ = ["GuardDecision", "RobustZScoreGuard", "robust_z"]

_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    z: float
    reason: str


def robust_z(value: float, history: npt.ArrayLike) -> float:
    """Upper-tail robust z-score of ``value`` against ``history`` (NaN with no history)."""
    values = np.asarray(history, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        return math.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    centre = max(median, float(np.percentile(values, 90)))
    scale = max(_MAD_TO_SIGMA * mad, math.sqrt(centre + 1.0))
    return (value - median) / scale


class RobustZScoreGuard:
    def __init__(
        self, *, threshold: float = 10.0, min_history: int = 14, hard_max: int = MAX_UNITS
    ) -> None:
        if threshold <= 0.0 or min_history < 1 or hard_max < 1:
            raise ValueError("threshold, min_history and hard_max must be positive")
        self.threshold = threshold
        self.min_history = min_history
        self.hard_max = hard_max

    def check(self, value: float, history: npt.ArrayLike) -> GuardDecision:
        if not math.isfinite(value) or value < 0.0 or value > self.hard_max:
            return GuardDecision(False, math.nan, "out_of_range")
        values = np.asarray(history, dtype=np.float64).ravel()
        if values.size < self.min_history:
            return GuardDecision(True, math.nan, "insufficient_history")
        z = robust_z(value, values)
        if z > self.threshold:
            return GuardDecision(False, z, "spike")
        return GuardDecision(True, z, "ok")
