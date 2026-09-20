"""Rolling-window coverage monitor with one-sided binomial alarms.

The service level objective is "empirical coverage within ``tolerance`` of nominal over the last
``window_days``". Two arms are tested. The breach arm is the SLO itself; the over-coverage arm
tests the other tail, because intervals wider than the level asks for are not free either --
they are holding cost, which the newsvendor layer prices. Only a breach opens a recalibration
proposal; over-coverage is reported and counted.

For each (segment, alpha) the monitor keeps daily counts of realized targets and misses. The
alarm rule is a one-sided binomial test of the null hypothesis that the miss probability is at
most ``alpha + tolerance``: with ``n`` observations and ``k`` misses in the window,
``p = P(K >= k | n, alpha + tolerance)``, and the alarm fires when ``p < significance`` and
``n >= min_observations``. The over-coverage arm is the same test on the lower tail against
``max(alpha - over_tolerance, alpha / 2)``. The window for day ``d`` is bounded on both sides --
``d - window_days < day <= d`` -- so a day recorded out of order is counted on its own day and
nowhere else. The tolerance keeps the test conservative under stationarity, and
the significance level is chosen so that the family-wise false-alarm rate over many overlapping
daily tests stays low (``bench/bench_monitor.py`` measures it over 500 stationary simulations).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from scipy.stats import binom

__all__ = [
    "CoverageMonitor",
    "MonitorConfig",
    "MonitorStatus",
    "binomial_miss_pvalue",
    "binomial_over_coverage_pvalue",
]


def _check_arguments(misses: int, n: int, miss_probability: float) -> None:
    if n < 1 or not 0 <= misses <= n:
        raise ValueError("need 0 <= misses <= n and n >= 1")
    if not 0.0 < miss_probability < 1.0:
        raise ValueError("miss_probability must lie strictly inside (0, 1)")


def binomial_miss_pvalue(misses: int, n: int, miss_probability: float) -> float:
    """``P(K >= misses)`` for ``K ~ Binomial(n, miss_probability)``."""
    _check_arguments(misses, n, miss_probability)
    return float(binom.sf(misses - 1, n, miss_probability))


def binomial_over_coverage_pvalue(misses: int, n: int, miss_probability: float) -> float:
    """``P(K <= misses)``: the other tail, for bands wider than the level asks for."""
    _check_arguments(misses, n, miss_probability)
    return float(binom.cdf(misses, n, miss_probability))


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    window_days: int = 28
    tolerance: float = 0.02
    significance: float = 0.01
    min_observations: int = 30
    # Over-coverage is not an SLO breach, so its arm is deliberately looser than the breach
    # tolerance: conformal bands are conservative by construction (the finite-sample
    # 1/(n+1) slack alone over-covers), and only a sizeable surplus is worth reading.
    over_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.window_days < 1 or self.min_observations < 1:
            raise ValueError("window_days and min_observations must be >= 1")
        if not 0.0 <= self.tolerance < 0.5:
            raise ValueError("tolerance must lie in [0, 0.5)")
        if not 0.0 < self.significance < 0.5:
            raise ValueError("significance must lie in (0, 0.5)")
        if not 0.0 < self.over_tolerance < 0.5:
            raise ValueError("over_tolerance must lie in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class MonitorStatus:
    segment: str
    alpha: float
    day: int
    n: int
    misses: int
    p_value: float
    alarm: bool
    over_p_value: float = 1.0
    over_alarm: bool = False

    @property
    def coverage(self) -> float:
        return 1.0 - self.misses / self.n if self.n else float("nan")


class CoverageMonitor:
    """Daily (n, misses) counts per (segment, alpha) over a rolling window of days."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._counts: dict[tuple[str, float], deque[tuple[int, int, int]]] = {}
        self.alarms_total = 0
        self.over_alarms_total = 0

    def record(self, segment: str, alpha: float, day: int, n: int, misses: int) -> None:
        if n < 0 or not 0 <= misses <= n:
            raise ValueError("need 0 <= misses <= n")
        if n == 0:
            return
        window = self._counts.setdefault((segment, alpha), deque())
        window.append((day, n, misses))
        self._trim(window)

    def _trim(self, window: deque[tuple[int, int, int]]) -> None:
        """Drop days outside the retention span of the newest day this key has ever seen.

        Days do not have to arrive in order: an actual can be backfilled behind the window's
        leading edge, and an actual for a target date inside the forecast horizon arrives ahead
        of it. Trimming is therefore relative to the largest day held rather than to the incoming
        one, and keeps a whole extra window of slack, so that neither kind of out-of-order record
        can evict days the live window still needs. ``status`` bounds the window on both sides,
        so the retained surplus is never counted into an earlier day's test.
        """
        if not window:
            return
        cutoff = max(item[0] for item in window) - self.retention_days
        if any(item[0] <= cutoff for item in window):
            kept = [item for item in window if item[0] > cutoff]
            window.clear()
            window.extend(kept)

    @property
    def retention_days(self) -> int:
        """How far back days are kept: the alarm window plus one window of out-of-order slack."""
        return 2 * self.config.window_days

    def status(self, segment: str, alpha: float, day: int) -> MonitorStatus:
        cutoff = day - self.config.window_days
        entries = [
            item for item in self._counts.get((segment, alpha), ()) if cutoff < item[0] <= day
        ]
        n = sum(item[1] for item in entries)
        misses = sum(item[2] for item in entries)
        if n < self.config.min_observations:
            return MonitorStatus(segment, alpha, day, n, misses, 1.0, False)
        p_value = binomial_miss_pvalue(misses, n, min(alpha + self.config.tolerance, 0.999))
        # The over-coverage arm tests the opposite null: that the miss probability is at
        # least ``alpha - over_tolerance``. The floor at ``alpha / 2`` keeps the reference
        # probability inside (0, 1) at the tight levels, where a fixed tolerance runs past 0.
        floor = max(alpha - self.config.over_tolerance, alpha / 2.0)
        over_p_value = binomial_over_coverage_pvalue(misses, n, floor)
        return MonitorStatus(
            segment,
            alpha,
            day,
            n,
            misses,
            p_value,
            p_value < self.config.significance,
            over_p_value,
            over_p_value < self.config.significance,
        )

    def evaluate(
        self, day: int, keys: Iterable[tuple[str, float]] | None = None
    ) -> list[MonitorStatus]:
        """Status for every tracked key (or ``keys``); counts alarms as they fire."""
        selected = list(keys) if keys is not None else sorted(self._counts)
        statuses = [self.status(segment, alpha, day) for segment, alpha in selected]
        self.alarms_total += sum(1 for item in statuses if item.alarm)
        self.over_alarms_total += sum(1 for item in statuses if item.over_alarm)
        return statuses

    def to_dict(self) -> dict[str, list[list[int]]]:
        return {
            f"{segment}|{alpha:.4f}": [list(item) for item in window]
            for (segment, alpha), window in self._counts.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[list[int]]], config: MonitorConfig) -> CoverageMonitor:
        monitor = cls(config)
        for key, entries in data.items():
            segment, _, alpha = key.rpartition("|")
            window: deque[tuple[int, int, int]] = deque(
                (int(d), int(n), int(m)) for d, n, m in entries
            )
            monitor._trim(window)
            monitor._counts[(segment, float(alpha))] = window
        return monitor
