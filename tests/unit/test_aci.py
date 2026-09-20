"""Property tests for Adaptive Conformal Inference.

The Gibbs-Candes guarantee is distribution-free: for any sequence of miscoverage indicators that
respects the recursion's premise, ``alpha_{t+1} = alpha_t + gamma (alpha - err_t)`` keeps the
realized long-run error within ``(max(alpha_1, 1-alpha_1) + gamma) / (gamma T)``. The property
tests assert exactly that, over adversarial sequences chosen by hypothesis rather than a nice
simulation -- and one test pins down where the premise fails in this implementation, because the
calibrator saturates instead of returning an infinite interval.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conformal_demand_forecaster.conformal.aci import (
    AciController,
    AlphaState,
    aci_long_run_bound,
)

pytestmark = pytest.mark.unit

targets = st.floats(min_value=0.02, max_value=0.4)
gammas = st.floats(min_value=0.005, max_value=0.2)
error_sequences = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=400)


def _premise(state: AlphaState, proposed: float) -> float:
    """The error the theorem assumes: an interval at ``alpha_t <= 0`` covers, at ``>= 1`` misses.

    Hypothesis is free to choose anything in between; those are the cases the bound really
    constrains.
    """
    if state.alpha <= 0.0:
        return 0.0
    if state.alpha >= 1.0:
        return 1.0
    return proposed


@given(target=targets, gamma=gammas, errors=error_sequences)
def test_long_run_error_respects_the_gibbs_candes_bound(
    target: float, gamma: float, errors: list[float]
) -> None:
    """Holds for any adversarial error sequence consistent with the recursion's premise."""
    state = AlphaState.initial(target, gamma)
    for error in errors:
        state.update(_premise(state, error))
    bound = aci_long_run_bound(target, gamma, len(errors))
    assert abs(state.long_run_error) <= bound + 1e-9


def test_the_bound_needs_its_premise_and_saturation_can_break_it() -> None:
    """An honest limitation, not a bug.

    The calibrator saturates at the widest observed score rather than returning an infinite
    interval, so a run of misses is possible even once alpha has been driven to its floor. When
    that happens the realized error can exceed the theoretical bound; the affected rows carry a
    ``saturated`` flag so an operator sees that the calibration set was exhausted.
    """
    state = AlphaState.initial(0.25, 0.1875)
    for _ in range(7):
        state.update(1.0)  # every interval missed, including those built at alpha_t <= 0
    assert state.alpha <= 0.0
    assert state.long_run_error > aci_long_run_bound(0.25, 0.1875, 7)


@given(target=targets, gamma=gammas, errors=error_sequences)
def test_alpha_stays_inside_the_clipping_range(
    target: float, gamma: float, errors: list[float]
) -> None:
    """The bound's proof needs ``alpha_t`` in ``[-gamma, 1+gamma]``; the state enforces it.

    This one holds for *every* sequence, premise or not, because the clipping is unconditional.
    """
    state = AlphaState.initial(target, gamma)
    for error in errors:
        state.update(error)
        assert -gamma - 1e-12 <= state.alpha <= 1.0 + gamma + 1e-12
        assert 0.0 <= state.effective <= 1.0


@given(target=targets, gamma=gammas, steps=st.integers(min_value=1, max_value=300))
def test_persistent_undercoverage_drives_alpha_down_monotonically(
    target: float, gamma: float, steps: int
) -> None:
    """Every interval missing (err = 1) shrinks alpha, which widens the next interval."""
    state = AlphaState.initial(target, gamma)
    previous = state.alpha
    for _ in range(steps):
        state.update(1.0)
        assert state.alpha <= previous + 1e-12
        previous = state.alpha
    assert state.alpha <= target


@given(target=targets, gamma=gammas, steps=st.integers(min_value=1, max_value=300))
def test_persistent_overcoverage_drives_alpha_up(target: float, gamma: float, steps: int) -> None:
    state = AlphaState.initial(target, gamma)
    for _ in range(steps):
        state.update(0.0)
    assert state.alpha >= target
    assert state.effective <= 1.0


def test_alpha_recovers_after_a_burst_of_misses() -> None:
    """A run of misses widens intervals; once covered again, alpha returns towards target."""
    state = AlphaState.initial(0.1, 0.05)
    for _ in range(20):
        state.update(1.0)
    widened = state.alpha
    assert widened < 0.1
    for _ in range(100):
        state.update(0.0)
    assert state.alpha > widened
    assert state.alpha == pytest.approx(0.1, abs=0.05) or state.alpha > 0.1


def test_bound_shrinks_with_more_steps_and_rejects_zero() -> None:
    assert aci_long_run_bound(0.1, 0.03, 1000) < aci_long_run_bound(0.1, 0.03, 100)
    with pytest.raises(ValueError, match="steps must be"):
        aci_long_run_bound(0.1, 0.03, 0)


def test_alpha_state_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="target alpha"):
        AlphaState.initial(0.0, 0.1)
    with pytest.raises(ValueError, match="gamma must be positive"):
        AlphaState.initial(0.1, 0.0)
    state = AlphaState.initial(0.1, 0.05)
    with pytest.raises(ValueError, match=r"error_rate must lie in \[0, 1\]"):
        state.update(1.5)


def test_alpha_state_reset_and_round_trip() -> None:
    state = AlphaState.initial(0.1, 0.05)
    for value in (1.0, 0.0, 0.5):
        state.update(value)
    restored = AlphaState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    assert restored.long_run_error == pytest.approx(state.long_run_error)
    state.reset()
    assert state.alpha == 0.1
    assert state.updates == 0
    assert state.long_run_error == 0.0


# --- controller -----------------------------------------------------------------------------


def _controller() -> AciController:
    return AciController(["smooth/b0", "intermittent/b0"], (0.2, 0.05), gamma=0.05)


def test_controller_tracks_one_state_per_group_and_level() -> None:
    controller = _controller()
    assert len(controller.states) == 4
    assert controller.effective_alphas()[("smooth/b0", 0.2)] == pytest.approx(0.2)
    controller.update("smooth/b0", 0.2, misses=4, n=4)
    assert controller.effective_alphas()[("smooth/b0", 0.2)] < 0.2
    # The other group is untouched: Mondrian groups adapt independently.
    assert controller.effective_alphas()[("intermittent/b0", 0.2)] == pytest.approx(0.2)


def test_controller_rejects_impossible_counts() -> None:
    controller = _controller()
    with pytest.raises(ValueError, match="n must be positive"):
        controller.update("smooth/b0", 0.2, misses=0, n=0)
    with pytest.raises(ValueError, match=r"misses must lie in \[0, n\]"):
        controller.update("smooth/b0", 0.2, misses=5, n=4)


def test_controller_round_trips_and_resets() -> None:
    controller = _controller()
    controller.update("smooth/b0", 0.05, misses=1, n=2)
    restored = AciController.from_dict(controller.to_dict(), gamma=0.05)
    assert restored.to_dict() == controller.to_dict()
    assert restored.alphas == (0.2, 0.05)
    controller.reset()
    assert all(state.alpha == state.target for state in controller.states.values())


def test_daily_updates_keep_coverage_near_target_under_a_shift() -> None:
    """Simulated shift: the miss rate jumps, ACI widens, and long-run coverage returns to target."""
    rng = np.random.default_rng(7)
    state = AlphaState.initial(0.1, 0.05)
    realized: list[float] = []
    for day in range(400):
        # Before day 150 the model is calibrated; after it, the nominal band misses far more.
        base = 0.1 if day < 150 else 0.45
        # Widening (a smaller effective alpha) reduces the realized miss rate proportionally.
        rate = float(np.clip(base * state.effective / 0.1, 0.0, 1.0))
        error = float(rng.binomial(20, rate) / 20)
        realized.append(error)
        state.update(error)
    tail = float(np.mean(realized[-100:]))
    assert tail < 0.25  # recovered well below the un-adapted 0.45 miss rate
    assert abs(state.long_run_error) <= aci_long_run_bound(0.1, 0.05, len(realized))
