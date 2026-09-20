"""Rolling-window coverage monitor with one-sided binomial alarms.

The service level objective is "empirical coverage within ``tolerance`` of nominal over the last
``window_days``". For each (segment, alpha) the monitor keeps daily counts of realized targets
and misses. The alarm rule is a one-sided binomial test of the null hypothesis that the miss
probability is at most ``alpha + tolerance``: with ``n`` observations and ``k`` misses in the
window, ``p = P(K >= k | n, alpha + tolerance)``, and the alarm fires when ``p < significance``
and ``n >= min_observations``. The window for day ``d`` is bounded on both sides --
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

__all__ = ["CoverageMonitor", "MonitorConfig", "MonitorStatus", "binomial_miss_pvalue"]


def binomial_miss_pvalue(misses: int, n: int, miss_probability: float) -> float:
    """``P(K >= misses)`` for ``K ~ Binomial(n, miss_probability)``."""
    if n < 1 or not 0 <= misses <= n:
        raise ValueError("need 0 <= misses <= n and n >= 1")
    if not 0.0 < miss_probability < 1.0:
        raise ValueError("miss_probability must lie strictly inside (0, 1)")
    return float(binom.sf(misses - 1, n, miss_probability))


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    window_days: int = 28
    tolerance: float = 0.02
    significance: float = 0.01
    min_observations: int = 30

    def __post_init__(self) -> None:
        if self.window_days < 1 or self.min_observations < 1:
            raise ValueError("window_days and min_observations must be >= 1")
        if not 0.0 <= self.tolerance < 0.5:
            raise ValueError("tolerance must lie in [0, 0.5)")
        if not 0.0 < self.significance < 0.5:
            raise ValueError("significance must lie in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class MonitorStatus:
    segment: str
    alpha: float
    day: int
    n: int
    misses: int
    p_value: float
    alarm: bool

    @property
    def coverage(self) -> float:
        return 1.0 - self.misses / self.n if self.n else float("nan")


class CoverageMonitor:
    """Daily (n, misses) counts per (segment, alpha) over a rolling window of days."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._counts: dict[tuple[str, float], deque[tuple[int, int, int]]] = {}
        self.alarms_total = 0

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
        alarm = p_value < self.config.significance
        return MonitorStatus(segment, alpha, day, n, misses, p_value, alarm)

    def evaluate(
        self, day: int, keys: Iterable[tuple[str, float]] | None = None
    ) -> list[MonitorStatus]:
        """Status for every tracked key (or ``keys``); counts alarms as they fire."""
        selected = list(keys) if keys is not None else sorted(self._counts)
        statuses = [self.status(segment, alpha, day) for segment, alpha in selected]
        self.alarms_total += sum(1 for item in statuses if item.alarm)
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
