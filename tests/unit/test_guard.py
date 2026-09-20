"""The robust z-score guard that screens ingested actuals before they touch online state.

Threat: an ingestor key (or a compromised upstream feed) sends fabricated sales. Those values
would be scored against the issued forecasts, drive the ACI alpha recursion and the coverage
monitor, and end up widening every interval for that group. The guard compares each value with
the series' own recent history using a median/MAD statistic, which a single spike cannot move.

Two design choices are asserted here as much as the maths: only the *upper* tail is screened, so
a genuine collapse to zero demand still reaches the monitor; and intermittent series (whose MAD
is often exactly zero) get a Poisson-like scale floor instead of an infinite z-score.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conformal_demand_forecaster.ml.contracts import MAX_UNITS
from conformal_demand_forecaster.serving.guard import RobustZScoreGuard, robust_z

pytestmark = pytest.mark.unit

HISTORY = [10.0, 12.0, 9.0, 11.0, 10.0, 13.0, 8.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 10.0]


def test_a_normal_value_passes() -> None:
    decision = RobustZScoreGuard().check(11.0, HISTORY)
    assert decision.accepted is True
    assert decision.reason == "ok"
    assert abs(decision.z) < 2.0


def test_an_adversarial_spike_is_quarantined() -> None:
    decision = RobustZScoreGuard().check(5_000.0, HISTORY)
    assert decision.accepted is False
    assert decision.reason == "spike"
    assert decision.z > 10.0


@pytest.mark.parametrize("value", [-1.0, MAX_UNITS + 1.0, math.inf, math.nan])
def test_values_outside_the_contract_range_are_refused_outright(value: float) -> None:
    decision = RobustZScoreGuard().check(value, HISTORY)
    assert decision.accepted is False
    assert decision.reason == "out_of_range"
    assert math.isnan(decision.z)


def test_a_sudden_zero_is_accepted_because_the_test_is_one_sided() -> None:
    """A closure or stockout is a real event the coverage monitor must see."""
    decision = RobustZScoreGuard().check(0.0, HISTORY)
    assert decision.accepted is True
    assert decision.z < 0


def test_an_intermittent_series_with_zero_mad_does_not_flag_every_sale() -> None:
    """Median and MAD are both 0 here, so only the Poisson-like floor keeps z finite."""
    history = [0.0] * 20
    guard = RobustZScoreGuard()
    assert guard.check(3.0, history).accepted is True
    assert guard.check(2_000.0, history).accepted is False


def test_an_intermittent_series_that_sells_in_bursts_is_not_flagged_for_ordinary_sales() -> None:
    """A median-only floor would score a routine sale of 12 at z = 12 and quarantine it."""
    history = [0.0, 0.0, 0.0, 12.0] * 10  # sells one day in four, about a dozen units
    guard = RobustZScoreGuard()
    assert guard.check(12.0, history).accepted is True
    assert guard.check(25.0, history).accepted is True
    assert guard.check(3_000.0, history).accepted is False


def test_short_histories_are_passed_through_with_a_reason() -> None:
    decision = RobustZScoreGuard(min_history=14).check(9_999.0, [1.0, 2.0])
    assert decision.accepted is True
    assert decision.reason == "insufficient_history"
    assert math.isnan(decision.z)


def test_the_threshold_is_configurable() -> None:
    value = 40.0
    assert RobustZScoreGuard(threshold=10.0).check(value, HISTORY).accepted is True
    assert RobustZScoreGuard(threshold=2.0).check(value, HISTORY).accepted is False


@pytest.mark.parametrize("kwargs", [{"threshold": 0.0}, {"min_history": 0}, {"hard_max": 0}])
def test_invalid_guard_settings_are_refused(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RobustZScoreGuard(**kwargs)  # type: ignore[arg-type]


# --- the statistic --------------------------------------------------------------------------


def test_robust_z_is_undefined_without_history() -> None:
    assert math.isnan(robust_z(5.0, []))
    assert math.isnan(robust_z(5.0, [float("nan")]))


@given(
    history=st.lists(st.floats(0.0, 500.0), min_size=20, max_size=60),
    outlier=st.floats(1e4, 1e6),
)
def test_one_poisoned_point_cannot_hide_the_next_spike(
    history: list[float], outlier: float
) -> None:
    """The property that matters: contaminating the window must not disarm the guard."""
    guard = RobustZScoreGuard()
    spike = 50_000.0
    assert guard.check(spike, history).accepted is False
    assert guard.check(spike, [*history[:-1], outlier]).accepted is False


@given(
    history=st.lists(st.floats(0.0, 500.0), min_size=20, max_size=60),
    outlier=st.floats(1e4, 1e6),
)
def test_a_single_outlier_keeps_the_statistic_in_the_same_order_of_magnitude(
    history: list[float], outlier: float
) -> None:
    """One poisoned point may move the percentile anchor by a few order statistics, so the
    statistic can shrink somewhat -- but not by the three orders of magnitude a mean/standard
    deviation screen loses on the same input (see the contrast test below)."""
    clean = robust_z(1_000.0, history)
    poisoned = robust_z(1_000.0, [*history[:-1], outlier])
    if math.isnan(clean) or math.isnan(poisoned) or clean <= 0.0:
        return
    assert poisoned >= clean / 10.0


def test_a_mean_and_standard_deviation_statistic_would_be_disarmed() -> None:
    """Why median/MAD: one spike in the window destroys a mean/std screen."""
    history = np.full(40, 20.0)
    poisoned = np.concatenate([history[:-1], [1e6]])
    naive = (5_000.0 - poisoned.mean()) / poisoned.std()
    assert abs(naive) < 1.0  # a mean/std guard would wave the next spike straight through
    assert RobustZScoreGuard().check(5_000.0, poisoned).accepted is False


@given(
    history=st.lists(st.floats(1.0, 200.0), min_size=20, max_size=40),
    value=st.floats(0.0, 1e4),
)
def test_robust_z_is_monotone_in_the_observed_value(history: list[float], value: float) -> None:
    assert robust_z(value + 1.0, history) >= robust_z(value, history)


@given(history=st.lists(st.floats(1.0, 200.0), min_size=20, max_size=40))
def test_the_median_itself_scores_zero(history: list[float]) -> None:
    median = float(np.median(history))
    assert robust_z(median, history) == pytest.approx(0.0, abs=1e-9)


def test_the_guard_accepts_decisions_it_reports_z_for() -> None:
    guard = RobustZScoreGuard(threshold=10.0)
    decision = guard.check(12.0, HISTORY)
    assert decision.z == pytest.approx(robust_z(12.0, np.asarray(HISTORY)))
