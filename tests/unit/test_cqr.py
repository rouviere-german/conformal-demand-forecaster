"""Property and unit tests for the conformal layer: scores, quantiles and Mondrian calibration.

The headline property is the split-conformal coverage guarantee. For exchangeable scores the
interval built at the ``ceil((n+1)(1-alpha))``-th smallest score covers a fresh point with
probability in ``[1-alpha, 1-alpha + 1/(n+1)]``. The test below checks the *exact* finite-sample
statement on random score vectors by counting, rather than estimating it by simulation:
with distinct scores, the rank of a fresh exchangeable score is uniform over the ``n+1`` gaps.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from conformal_demand_forecaster.conformal.cqr import (
    GroupKey,
    MondrianCalibrator,
    ScoreBuffer,
    conformal_quantile,
    conformal_rank,
    coverage_indicator,
    cqr_scores,
    quantile_pair,
)
from conformal_demand_forecaster.models.gbm import QuantilePrediction

pytestmark = pytest.mark.unit

LEVELS = (0.05, 0.1, 0.5, 0.9, 0.95)
BUCKETS = ((1, 7), (8, 14), (15, 28))

finite_scores = st.lists(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=200,
)
alphas = st.floats(min_value=0.01, max_value=0.5)


def _calibrator(min_group_size: int = 5) -> MondrianCalibrator:
    return MondrianCalibrator((0.2, 0.05), BUCKETS, LEVELS, min_group_size=min_group_size)


# --- the guarantee --------------------------------------------------------------------------


@given(
    n=st.integers(min_value=5, max_value=300),
    alpha=st.floats(min_value=0.02, max_value=0.4),
)
def test_finite_sample_coverage_bracket_holds_exactly(n: int, alpha: float) -> None:
    """Exchangeable distinct scores: coverage is in ``[1-a, 1-a+1/(n+1)]`` by rank counting.

    With ``n`` calibration scores and a fresh exchangeable score, every one of the ``n+1``
    orderings is equally likely, so ``P(new <= k-th smallest) = k / (n+1)`` exactly.
    """
    rank = conformal_rank(n, alpha)
    assume(rank <= n)
    probability = rank / (n + 1)
    assert probability >= 1.0 - alpha - 1e-12
    assert probability <= 1.0 - alpha + 1.0 / (n + 1) + 1e-12


@given(scores=finite_scores, alpha=alphas)
def test_conformal_quantile_is_an_order_statistic(scores: list[float], alpha: float) -> None:
    value = conformal_quantile(scores, alpha)
    rank = conformal_rank(len(scores), alpha)
    if rank > len(scores):
        assert value == math.inf
    else:
        assert value in set(scores)
        assert sum(1 for s in scores if s <= value) >= rank


@given(scores=finite_scores, alpha=alphas, other=alphas)
def test_conformal_quantile_is_monotone_in_alpha(
    scores: list[float], alpha: float, other: float
) -> None:
    """A smaller miscoverage level never yields a smaller (narrower) conformal quantile."""
    low, high = min(alpha, other), max(alpha, other)
    assert conformal_quantile(scores, low) >= conformal_quantile(scores, high)


@given(scores=finite_scores, shift=st.floats(min_value=-1e3, max_value=1e3))
def test_conformal_quantile_is_shift_equivariant(scores: list[float], shift: float) -> None:
    shifted = [s + shift for s in scores]
    base = conformal_quantile(scores, 0.1)
    assert conformal_quantile(shifted, 0.1) == pytest.approx(base + shift, abs=1e-6)


@given(
    y=st.lists(st.floats(0, 1e4, allow_nan=False), min_size=1, max_size=50),
    data=st.data(),
)
def test_cqr_score_sign_matches_containment(y: list[float], data: st.DataObject) -> None:
    """A score is <= 0 exactly when the raw band already contains the target."""
    n = len(y)
    lower = np.array(
        data.draw(st.lists(st.floats(0, 1e4, allow_nan=False), min_size=n, max_size=n))
    )
    width = np.array(
        data.draw(st.lists(st.floats(0, 1e4, allow_nan=False), min_size=n, max_size=n))
    )
    upper = lower + width
    scores = cqr_scores(y, lower, upper)
    inside = coverage_indicator(y, lower, upper)
    assert np.array_equal(scores <= 0.0, inside > 0.5)


@given(
    y=st.lists(st.floats(0, 1e3, allow_nan=False), min_size=2, max_size=60),
    data=st.data(),
    alpha=st.floats(min_value=0.05, max_value=0.4),
)
def test_conformalized_band_covers_at_least_the_nominal_share_in_sample(
    y: list[float], data: st.DataObject, alpha: float
) -> None:
    """Calibrating on a sample and scoring the same sample covers at least ``1-alpha`` of it."""
    n = len(y)
    lower = np.array(
        data.draw(st.lists(st.floats(0, 1e3, allow_nan=False), min_size=n, max_size=n))
    )
    upper = lower + np.array(
        data.draw(st.lists(st.floats(0, 1e3, allow_nan=False), min_size=n, max_size=n))
    )
    scores = cqr_scores(y, lower, upper)
    q = conformal_quantile(scores, alpha)
    assume(math.isfinite(q))
    # The band edges are sums of large floats, so allow a relative epsilon on containment.
    tol = 1e-9 * max(1.0, float(np.max(np.abs(upper))))
    covered = coverage_indicator(y, lower - q - tol, upper + q + tol).mean()
    assert covered >= conformal_rank(n, alpha) / n - 1e-12


# --- scores and helpers ---------------------------------------------------------------------


def test_conformal_rank_matches_the_ceiling_formula() -> None:
    assert conformal_rank(9, 0.1) == 9
    assert conformal_rank(19, 0.1) == 18
    assert conformal_rank(100, 0.05) == 96
    with pytest.raises(ValueError, match="at least one"):
        conformal_rank(0, 0.1)


def test_conformal_quantile_saturates_out_of_range() -> None:
    assert conformal_quantile([1.0, 2.0, 3.0], 0.01) == math.inf
    # rank ceil((n+1)(1-alpha)) only drops below 1 once alpha reaches 1 (an empty prediction).
    assert conformal_quantile([1.0, 2.0, 3.0], 1.0) == -math.inf
    assert conformal_quantile([1.0, 2.0, 3.0], 0.999) == 1.0


def test_conformal_quantile_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        conformal_quantile([1.0, float("nan")], 0.1)
    with pytest.raises(ValueError, match="at least one"):
        conformal_quantile([], 0.1)


def test_cqr_scores_reject_crossing_quantiles() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        cqr_scores([1.0], [5.0], [2.0])


def test_quantile_pair_picks_the_closest_symmetric_pair() -> None:
    assert quantile_pair(0.1, LEVELS) == (0.05, 0.95)
    assert quantile_pair(0.2, LEVELS) == (0.1, 0.9)
    with pytest.raises(ValueError, match="no lower quantile"):
        quantile_pair(0.1, (0.6, 0.9))
    with pytest.raises(ValueError, match="symmetric upper"):
        quantile_pair(0.1, (0.05, 0.5, 0.9))


def test_group_key_round_trips_and_rejects_junk() -> None:
    key = GroupKey("smooth", 2)
    assert GroupKey.parse(key.label) == key
    for bad in ("nonsense/b1", "smooth/bX", "smooth"):
        with pytest.raises(ValueError, match="invalid group label"):
            GroupKey.parse(bad)


# --- score buffer ---------------------------------------------------------------------------


def test_score_buffer_keeps_only_the_window() -> None:
    buffer = ScoreBuffer(window_days=10)
    key = ("smooth/b0", 0.2)
    buffer.append(key, [1, 2, 3], [0.1, 0.2, 0.3])
    buffer.append(key, [20, 21], [1.0, 2.0])
    assert buffer.count(key) == 2
    assert buffer.scores(key).tolist() == [1.0, 2.0]
    assert buffer.days(key).tolist() == [20, 21]
    assert buffer.latest_day() == 21


def test_score_buffer_validates_and_copies() -> None:
    buffer = ScoreBuffer(window_days=5)
    with pytest.raises(ValueError, match="same length"):
        buffer.append(("g", 0.1), [1, 2], [0.5])
    with pytest.raises(ValueError, match="NaN"):
        buffer.append(("g", 0.1), [1], [float("nan")])
    with pytest.raises(ValueError, match="window_days"):
        ScoreBuffer(window_days=0)
    buffer.append(("g", 0.1), [1], [0.5])
    clone = buffer.copy()
    clone.append(("g", 0.1), [2], [9.0])
    assert buffer.count(("g", 0.1)) == 1
    assert clone.count(("g", 0.1)) == 2
    assert buffer.latest_day() == 1
    assert ScoreBuffer(3).latest_day() is None


def test_score_buffer_round_trips_through_arrays() -> None:
    buffer = ScoreBuffer(window_days=30)
    buffer.append(("smooth/b0", 0.2), [1, 2], [0.5, 0.75])
    buffer.append(("intermittent/b1", 0.05), [3], [1.25])
    restored = ScoreBuffer.from_arrays(buffer.to_arrays(), buffer.keys, 30)
    assert restored.keys == buffer.keys
    for key in buffer.keys:
        assert restored.scores(key).tolist() == buffer.scores(key).tolist()
        assert restored.days(key).tolist() == buffer.days(key).tolist()


def test_score_buffer_merge_combines_entries() -> None:
    left = ScoreBuffer(window_days=50)
    left.append(("smooth/b0", 0.2), [1], [0.5])
    right = ScoreBuffer(window_days=50)
    right.append(("smooth/b0", 0.2), [2], [0.75])
    right.append(("smooth/b1", 0.2), [2], [1.5])
    left.merge(right)
    assert left.scores(("smooth/b0", 0.2)).tolist() == [0.5, 0.75]
    assert left.count(("smooth/b1", 0.2)) == 1


def test_pooled_scores_gather_every_group_at_one_alpha() -> None:
    buffer = ScoreBuffer(window_days=50)
    buffer.append(("smooth/b0", 0.2), [1, 2], [0.5, 0.75])
    buffer.append(("intermittent/b0", 0.2), [1], [3.0])
    buffer.append(("smooth/b0", 0.05), [1], [9.0])
    assert sorted(buffer.pooled(0.2).tolist()) == [0.5, 0.75, 3.0]
    assert buffer.pooled(0.99).size == 0


# --- Mondrian calibration -------------------------------------------------------------------


def test_bucket_assignment_and_group_labels() -> None:
    calibrator = _calibrator()
    assert calibrator.bucket_of([1, 7, 8, 14, 15, 28]).tolist() == [0, 0, 1, 1, 2, 2]
    labels = calibrator.group_labels(["smooth", "intermittent"], [3, 20])
    assert labels == ["smooth/b0", "intermittent/b2"]
    with pytest.raises(ValueError, match="outside the configured buckets"):
        calibrator.bucket_of([29])
    assert len(calibrator.all_groups()) == 6


def test_calibrator_rejects_invalid_alphas() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        MondrianCalibrator((0.0,), BUCKETS, LEVELS)


def test_small_groups_fall_back_to_the_pooled_scores() -> None:
    calibrator = _calibrator(min_group_size=10)
    buffer = ScoreBuffer(window_days=100)
    buffer.append(("smooth/b0", 0.2), list(range(3)), [1.0, 1.0, 1.0])
    buffer.append(("intermittent/b0", 0.2), list(range(20)), [5.0] * 20)
    result = calibrator.quantile(buffer, "smooth/b0", 0.2)
    assert result.pooled is True
    assert result.n == 23
    big = calibrator.quantile(buffer, "intermittent/b0", 0.2)
    assert big.pooled is False
    assert big.n == 20


def test_quantile_reports_saturation_when_calibration_is_exhausted() -> None:
    calibrator = _calibrator(min_group_size=1)
    buffer = ScoreBuffer(window_days=100)
    buffer.append(("smooth/b0", 0.05), [1, 2, 3], [1.0, 2.0, 3.0])
    result = calibrator.quantile(buffer, "smooth/b0", 0.05)
    assert result.saturated is True
    assert result.value == 3.0
    with pytest.raises(ValueError, match="no calibration scores"):
        calibrator.quantile(ScoreBuffer(10), "smooth/b0", 0.05)


def test_intervals_widen_per_group_and_stay_non_negative() -> None:
    calibrator = _calibrator(min_group_size=1)
    buffer = ScoreBuffer(window_days=100)
    buffer.append(("smooth/b0", 0.2), list(range(30)), [1.0] * 30)
    buffer.append(("intermittent/b0", 0.2), list(range(30)), [40.0] * 30)
    buffer.append(("smooth/b0", 0.05), list(range(30)), [2.0] * 30)
    buffer.append(("intermittent/b0", 0.05), list(range(30)), [50.0] * 30)
    quantiles = np.tile(np.array([2.0, 3.0, 5.0, 7.0, 8.0]), (2, 1))
    prediction = QuantilePrediction(point=np.array([5.0, 5.0]), quantiles=quantiles, levels=LEVELS)

    intervals = calibrator.intervals(prediction, ["smooth", "intermittent"], [1, 1], buffer)
    # alpha = 0.2 corrects the raw (0.1, 0.9) pair, here (3.0, 7.0).
    assert intervals.lower[0.2].tolist() == [2.0, 0.0]  # clipped at zero demand
    assert intervals.upper[0.2].tolist() == [8.0, 47.0]
    assert intervals.width(0.05)[1] > intervals.width(0.05)[0]
    assert np.all(intervals.upper[0.2] >= intervals.lower[0.2])

    pooled = calibrator.intervals(
        prediction, ["smooth", "intermittent"], [1, 1], buffer, mondrian=False
    )
    assert pooled.width(0.2)[0] == pooled.width(0.2)[1]


def test_intervals_honour_an_effective_alpha_per_group() -> None:
    calibrator = _calibrator(min_group_size=1)
    buffer = ScoreBuffer(window_days=100)
    buffer.append(("smooth/b0", 0.2), list(range(40)), np.linspace(0.0, 10.0, 40))
    buffer.append(("smooth/b0", 0.05), list(range(40)), np.linspace(0.0, 20.0, 40))
    quantiles = np.array([[2.0, 3.0, 5.0, 7.0, 8.0]])
    prediction = QuantilePrediction(point=np.array([5.0]), quantiles=quantiles, levels=LEVELS)
    base = calibrator.intervals(prediction, ["smooth"], [1], buffer)
    tight = calibrator.intervals(
        prediction, ["smooth"], [1], buffer, alpha_effective={("smooth/b0", 0.2): 0.45}
    )
    assert tight.width(0.2)[0] < base.width(0.2)[0]
    assert tight.alpha_effective[0.2][0] == pytest.approx(0.45)
    assert base.alpha_effective[0.2][0] == pytest.approx(0.2)


def test_calibrator_scores_use_the_mapped_quantile_pair() -> None:
    calibrator = _calibrator()
    quantiles = np.array([[1.0, 2.0, 5.0, 8.0, 9.0]])
    prediction = QuantilePrediction(point=np.array([5.0]), quantiles=quantiles, levels=LEVELS)
    scores = calibrator.scores(prediction, [12.0])
    assert scores[0.05][0] == pytest.approx(12.0 - 9.0)  # (0.05, 0.95) pair
    assert scores[0.2][0] == pytest.approx(12.0 - 8.0)  # (0.1, 0.9) pair


def test_score_buffer_merge_is_idempotent() -> None:
    """Merging the same live buffer twice must not double-weight its calibration scores."""
    base = ScoreBuffer(window_days=50)
    base.append(("smooth/b0", 0.2), [1], [0.5])
    live = ScoreBuffer(window_days=50)
    live.append(("smooth/b0", 0.2), [2, 3], [0.75, 1.25])

    base.merge(live)
    once = base.scores(("smooth/b0", 0.2)).tolist()
    base.merge(live)

    assert base.scores(("smooth/b0", 0.2)).tolist() == once
    assert base.count(("smooth/b0", 0.2)) == 3


def test_score_buffer_merge_keeps_days_the_target_has_not_seen() -> None:
    base = ScoreBuffer(window_days=50)
    base.append(("smooth/b0", 0.2), [1, 2], [0.5, 0.6])
    live = ScoreBuffer(window_days=50)
    live.append(("smooth/b0", 0.2), [2, 4], [9.0, 1.5])  # day 2 is already held
    base.merge(live)
    assert base.days(("smooth/b0", 0.2)).tolist() == [1, 2, 4]
    assert base.scores(("smooth/b0", 0.2)).tolist() == [0.5, 0.6, 1.5]
