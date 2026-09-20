"""The coverage SLO and its binomial alarm rule.

The SLO is "empirical coverage stays within ``tolerance`` of nominal over the last
``window_days``". The monitor implements it as a one-sided binomial test against a miss
probability of ``alpha + tolerance``. Two behaviours must hold together: a genuinely calibrated
stream must rarely alarm (false alarms are measured over many simulations in
``bench/bench_monitor.py`` and bounded here on a smaller sample), and a real breach must be
detected within a useful number of days.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import binom

from conformal_demand_forecaster.monitoring.coverage import (
    CoverageMonitor,
    MonitorConfig,
    binomial_miss_pvalue,
    binomial_over_coverage_pvalue,
)

pytestmark = pytest.mark.unit

ALPHA = 0.2
CONFIG = MonitorConfig(window_days=28, tolerance=0.02, significance=0.01, min_observations=30)


def feed(monitor: CoverageMonitor, *, days: int, n: int, miss_rate: float, start: int = 0) -> int:
    rng = np.random.default_rng(0)
    day = start
    for day in range(start, start + days):
        misses = int(rng.binomial(n, miss_rate))
        monitor.record("smooth", ALPHA, day, n, misses)
    return day


# --- the test statistic ---------------------------------------------------------------------


def test_p_value_matches_the_binomial_survival_function() -> None:
    assert binomial_miss_pvalue(5, 20, 0.2) == pytest.approx(float(binom.sf(4, 20, 0.2)))
    assert binomial_miss_pvalue(0, 20, 0.2) == pytest.approx(1.0)
    assert binomial_miss_pvalue(20, 20, 0.2) == pytest.approx(0.2**20)


def test_p_value_is_monotone_in_the_observed_misses() -> None:
    values = [binomial_miss_pvalue(k, 50, 0.2) for k in range(51)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(values))


@pytest.mark.parametrize(
    ("misses", "n", "probability"),
    [(-1, 10, 0.2), (11, 10, 0.2), (0, 0, 0.2), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_p_value_rejects_impossible_arguments(misses: int, n: int, probability: float) -> None:
    with pytest.raises(ValueError, match=r"need|miss_probability"):
        binomial_miss_pvalue(misses, n, probability)


# --- the rolling window ---------------------------------------------------------------------


def test_status_waits_for_enough_observations() -> None:
    monitor = CoverageMonitor(CONFIG)
    monitor.record("smooth", ALPHA, 1, n=5, misses=5)
    status = monitor.status("smooth", ALPHA, 1)
    assert status.n == 5
    assert status.alarm is False
    assert status.p_value == 1.0


def test_old_days_drop_out_of_the_window() -> None:
    monitor = CoverageMonitor(CONFIG)
    monitor.record("smooth", ALPHA, 1, n=100, misses=100)
    monitor.record("smooth", ALPHA, 100, n=100, misses=0)
    status = monitor.status("smooth", ALPHA, 100)
    assert status.n == 100
    assert status.misses == 0
    assert status.coverage == pytest.approx(1.0)


def test_empty_batches_are_ignored_and_bad_counts_refused() -> None:
    monitor = CoverageMonitor()
    monitor.record("smooth", ALPHA, 1, n=0, misses=0)
    assert monitor.status("smooth", ALPHA, 1).n == 0
    with pytest.raises(ValueError, match="0 <= misses <= n"):
        monitor.record("smooth", ALPHA, 1, n=5, misses=6)


def test_status_of_an_untracked_key_is_quiet() -> None:
    status = CoverageMonitor(CONFIG).status("intermittent", 0.05, 10)
    assert status.n == 0
    assert status.alarm is False
    assert np.isnan(status.coverage)


# --- the alarm rule -------------------------------------------------------------------------


def test_a_calibrated_stream_does_not_alarm() -> None:
    monitor = CoverageMonitor(CONFIG)
    day = feed(monitor, days=40, n=40, miss_rate=ALPHA)
    status = monitor.status("smooth", ALPHA, day)
    assert status.n >= CONFIG.min_observations
    assert status.coverage == pytest.approx(1 - ALPHA, abs=0.05)
    assert status.alarm is False


def test_a_clear_breach_raises_an_alarm() -> None:
    monitor = CoverageMonitor(CONFIG)
    day = feed(monitor, days=40, n=40, miss_rate=0.45)
    status = monitor.status("smooth", ALPHA, day)
    assert status.alarm is True
    assert status.p_value < CONFIG.significance
    assert status.coverage < 1 - ALPHA - CONFIG.tolerance


def test_over_coverage_never_alarms() -> None:
    """The test is one-sided: intervals that are too wide are a cost problem, not an SLO breach."""
    monitor = CoverageMonitor(CONFIG)
    day = feed(monitor, days=40, n=40, miss_rate=0.0)
    assert monitor.status("smooth", ALPHA, day).alarm is False


def test_the_slo_tolerance_makes_the_rule_conservative() -> None:
    """A stream missing by exactly the tolerance is not flagged; a much worse one is."""
    monitor = CoverageMonitor(CONFIG)
    for day in range(30):
        monitor.record("smooth", ALPHA, day, n=40, misses=round(40 * (ALPHA + 0.02)))
    assert monitor.status("smooth", ALPHA, 29).alarm is False


def test_false_alarm_rate_of_a_stationary_stream_is_small() -> None:
    """A smaller, faster version of bench/bench_monitor.py's 500-run false-alarm measurement."""
    rng = np.random.default_rng(20260920)
    alarms = 0
    runs = 120
    for _ in range(runs):
        monitor = CoverageMonitor(CONFIG)
        fired = False
        for day in range(60):
            monitor.record("smooth", ALPHA, day, 40, int(rng.binomial(40, ALPHA)))
            if monitor.status("smooth", ALPHA, day).alarm:
                fired = True
        alarms += int(fired)
    assert alarms / runs <= 0.05


def test_detection_delay_after_a_shift_is_within_the_window() -> None:
    rng = np.random.default_rng(3)
    monitor = CoverageMonitor(CONFIG)
    for day in range(40):
        monitor.record("smooth", ALPHA, day, 40, int(rng.binomial(40, ALPHA)))
    delay: int | None = None
    for offset in range(CONFIG.window_days):
        day = 40 + offset
        monitor.record("smooth", ALPHA, day, 40, int(rng.binomial(40, 0.5)))
        if monitor.status("smooth", ALPHA, day).alarm:
            delay = offset
            break
    assert delay is not None
    assert delay <= 10


def test_evaluate_counts_alarms_and_honours_an_explicit_key_list() -> None:
    monitor = CoverageMonitor(CONFIG)
    day = feed(monitor, days=40, n=40, miss_rate=0.45)
    statuses = monitor.evaluate(day)
    assert [s.segment for s in statuses] == ["smooth"]
    assert monitor.alarms_total == 1
    explicit = monitor.evaluate(day, [("intermittent", ALPHA)])
    assert explicit[0].n == 0
    assert monitor.alarms_total == 1


def test_monitor_state_round_trips() -> None:
    monitor = CoverageMonitor(CONFIG)
    feed(monitor, days=10, n=20, miss_rate=0.2)
    restored = CoverageMonitor.from_dict(monitor.to_dict(), CONFIG)
    assert restored.to_dict() == monitor.to_dict()
    assert restored.status("smooth", ALPHA, 9).n == monitor.status("smooth", ALPHA, 9).n


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_days": 0}, "window_days"),
        ({"min_observations": 0}, "window_days"),
        ({"tolerance": 0.6}, "tolerance"),
        ({"significance": 0.9}, "significance"),
    ],
)
def test_invalid_monitor_configurations_are_refused(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MonitorConfig(**kwargs)  # type: ignore[arg-type]


# --- out-of-order days ------------------------------------------------------------------------
# An actual can arrive for a day that is not the newest one: a backfill lands behind the window's
# leading edge, and an ingest for a target date inside the forecast horizon lands ahead of it.
# Neither may disturb the window the alarm rule is computed over.


def test_a_forward_dated_record_does_not_evict_the_window() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=20, n=10, miss_rate=0.2)
    before = monitor.status("smooth", ALPHA, last)
    assert before.n == 200

    monitor.record("smooth", ALPHA, last + 7, 20, 4)

    after = monitor.status("smooth", ALPHA, last)
    assert (after.n, after.misses) == (before.n, before.misses)


def test_a_forward_dated_record_is_not_counted_before_its_day() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=20, n=10, miss_rate=0.2)
    before = monitor.status("smooth", ALPHA, last)
    monitor.record("smooth", ALPHA, last + 7, 20, 20)
    assert monitor.status("smooth", ALPHA, last).misses == before.misses
    assert monitor.status("smooth", ALPHA, last + 7).misses == before.misses + 20


def test_a_backfilled_earlier_day_is_still_counted() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=20, n=10, miss_rate=0.2)
    before = monitor.status("smooth", ALPHA, last)
    monitor.record("smooth", ALPHA, last - 3, 10, 5)
    after = monitor.status("smooth", ALPHA, last)
    assert after.n == before.n + 10
    assert after.misses == before.misses + 5


def test_days_outside_the_retention_span_are_dropped() -> None:
    monitor = CoverageMonitor(CONFIG)
    feed(monitor, days=200, n=10, miss_rate=0.2)
    entries = monitor.to_dict()[f"smooth|{ALPHA:.4f}"]
    assert len(entries) <= 2 * CONFIG.window_days


# --- the over-coverage arm --------------------------------------------------------------------
# Systematically wider intervals than the level asks for are not a broken promise, but they are
# holding cost: the newsvendor layer orders off the same quantiles. The second arm reports them
# without opening a recalibration proposal.


def test_a_calibrated_stream_raises_neither_arm() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=40, n=40, miss_rate=ALPHA)
    status = monitor.status("smooth", ALPHA, last)
    assert status.alarm is False
    assert status.over_alarm is False


def test_intervals_that_never_miss_raise_the_over_coverage_arm() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=40, n=40, miss_rate=0.0)
    status = monitor.status("smooth", ALPHA, last)
    assert status.alarm is False
    assert status.over_alarm is True
    assert status.coverage == pytest.approx(1.0)


def test_a_breach_does_not_raise_the_over_coverage_arm() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=40, n=40, miss_rate=0.45)
    status = monitor.status("smooth", ALPHA, last)
    assert status.alarm is True
    assert status.over_alarm is False


def test_the_over_coverage_arm_also_waits_for_enough_observations() -> None:
    monitor = CoverageMonitor(CONFIG)
    monitor.record("smooth", ALPHA, 0, CONFIG.min_observations - 1, 0)
    assert monitor.status("smooth", ALPHA, 0).over_alarm is False


def test_the_over_coverage_arm_stays_valid_at_the_tight_level() -> None:
    """At alpha = 0.05 a fixed 5-point tolerance would test against a zero miss probability."""
    monitor = CoverageMonitor(CONFIG)
    for day in range(40):
        monitor.record("smooth", 0.05, day, 40, 2)
    status = monitor.status("smooth", 0.05, 39)
    assert 0.0 < status.over_p_value <= 1.0
    assert status.over_alarm is False


def test_evaluate_counts_both_arms() -> None:
    monitor = CoverageMonitor(CONFIG)
    last = feed(monitor, days=40, n=40, miss_rate=0.0)
    monitor.evaluate(last, [("smooth", ALPHA)])
    assert monitor.alarms_total == 0
    assert monitor.over_alarms_total == 1


def test_the_lower_tail_p_value_matches_the_binomial_cdf() -> None:
    assert binomial_over_coverage_pvalue(2, 20, 0.2) == pytest.approx(float(binom.cdf(2, 20, 0.2)))
    assert binomial_over_coverage_pvalue(20, 20, 0.2) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="need"):
        binomial_over_coverage_pvalue(3, 2, 0.2)


def test_an_invalid_over_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="over_tolerance"):
        MonitorConfig(over_tolerance=0.0)
