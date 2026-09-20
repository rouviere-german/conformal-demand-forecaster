"""Conformalized quantile regression with Mondrian (segment x horizon-bucket) calibration.

CQR (Romano, Patterson and Candes, "Conformalized Quantile Regression", NeurIPS 2019): given a
lower and an upper quantile prediction, the nonconformity score of a calibration point is
``max(lower - y, y - upper)``. With ``n`` calibration scores and target miscoverage ``alpha``, the
conformal quantile ``q`` is the ``ceil((n + 1)(1 - alpha))``-th smallest score, and the interval
``[lower - q, upper + q]`` covers a new exchangeable point with probability at least
``1 - alpha`` (and at most ``1 - alpha + 1/(n + 1)`` when scores are almost surely distinct).

Mondrian conformal prediction (Vovk et al.) calibrates each group with its own scores, so the
guarantee holds per group (here: demand segment x horizon bucket) instead of only on average.
Groups with fewer than ``min_group_size`` scores fall back to the pooled scores.

The quantile can be requested at an *effective* alpha supplied by Adaptive Conformal Inference.
When that alpha is so small that the rank exceeds ``n`` (or so large that it drops below 1) the
result saturates at the widest (narrowest) observed score; the flag is reported so operators see
when the calibration set is exhausted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from conformal_demand_forecaster.ml.contracts import SEGMENTS
from conformal_demand_forecaster.models.gbm import QuantilePrediction

__all__ = [
    "GroupKey",
    "IntervalSet",
    "MondrianCalibrator",
    "QuantileResult",
    "ScoreBuffer",
    "conformal_quantile",
    "conformal_rank",
    "coverage_indicator",
    "cqr_scores",
    "quantile_pair",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def conformal_rank(n: int, alpha: float) -> int:
    """Rank ``ceil((n + 1)(1 - alpha))`` of the conformal quantile among ``n`` scores."""
    if n < 1:
        raise ValueError("at least one calibration score is required")
    return math.ceil((n + 1) * (1.0 - alpha))


def conformal_quantile(scores: npt.ArrayLike, alpha: float) -> float:
    """Finite-sample corrected empirical quantile.

    Returns ``inf`` (``-inf``) when the rank falls above (below) the available scores.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if np.any(np.isnan(values)):
        raise ValueError("scores must not contain NaN")
    rank = conformal_rank(values.size, alpha)
    if rank > values.size:
        return math.inf
    if rank < 1:
        return -math.inf
    return float(np.partition(values, rank - 1)[rank - 1])


def cqr_scores(y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike) -> FloatArray:
    """CQR nonconformity scores ``max(lower - y, y - upper)`` (negative inside the band)."""
    y_arr = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if np.any(lo > hi):
        raise ValueError("lower quantile must not exceed upper quantile")
    return np.maximum(lo - y_arr, y_arr - hi)


def coverage_indicator(y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike) -> FloatArray:
    """1.0 where ``lower <= y <= upper``, else 0.0."""
    y_arr = np.asarray(y, dtype=np.float64)
    inside = (y_arr >= np.asarray(lower, dtype=np.float64)) & (
        y_arr <= np.asarray(upper, dtype=np.float64)
    )
    return inside.astype(np.float64)


def quantile_pair(alpha: float, available: Sequence[float]) -> tuple[float, float]:
    """The raw quantile pair used for miscoverage ``alpha``: the closest available symmetric pair.

    CQR corrects whatever the raw pair covers, so ``(0.05, 0.95)`` is a fine base for a 95%
    interval; the conformal quantile absorbs the difference.
    """
    lower_target = alpha / 2.0
    lowers = sorted(q for q in available if q < 0.5)
    if not lowers:
        raise ValueError("no lower quantile available")
    lower = min(lowers, key=lambda q: (abs(q - lower_target), q))
    upper = round(1.0 - lower, 6)
    if upper not in available:
        raise ValueError(f"symmetric upper quantile {upper} is not available")
    return lower, upper


@dataclass(frozen=True, slots=True, order=True)
class GroupKey:
    segment: str
    bucket: int

    @property
    def label(self) -> str:
        return f"{self.segment}/b{self.bucket}"

    @classmethod
    def parse(cls, label: str) -> GroupKey:
        segment, _, bucket = label.partition("/b")
        if segment not in SEGMENTS or not bucket.isdigit():
            raise ValueError(f"invalid group label {label!r}")
        return cls(segment, int(bucket))


BufferKey = tuple[str, float]  # (group label, alpha)


@dataclass
class ScoreBuffer:
    """Rolling store of (target day, score) per group and alpha, trimmed to a window of days."""

    window_days: int
    _days: dict[BufferKey, IntArray] = field(default_factory=dict)
    _scores: dict[BufferKey, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")

    @property
    def keys(self) -> list[BufferKey]:
        return sorted(self._days)

    def append(self, key: BufferKey, days: npt.ArrayLike, scores: npt.ArrayLike) -> None:
        new_days = np.asarray(days, dtype=np.int64).ravel()
        new_scores = np.asarray(scores, dtype=np.float64).ravel()
        if new_days.shape != new_scores.shape:
            raise ValueError("days and scores must have the same length")
        if np.any(np.isnan(new_scores)):
            raise ValueError("scores must not contain NaN")
        if key in self._days:
            new_days = np.concatenate([self._days[key], new_days])
            new_scores = np.concatenate([self._scores[key], new_scores])
        keep = new_days > new_days.max() - self.window_days if new_days.size else np.zeros(0, bool)
        self._days[key] = new_days[keep]
        self._scores[key] = new_scores[keep]

    def scores(self, key: BufferKey) -> FloatArray:
        return self._scores.get(key, np.zeros(0))

    def days(self, key: BufferKey) -> IntArray:
        """Target days of the scores stored under ``key`` (empty when the key is unknown)."""
        return self._days.get(key, np.zeros(0, dtype=np.int64))

    def merge(self, other: ScoreBuffer) -> None:
        """Append every entry of ``other`` (its own window trimming still applies per key)."""
        for key in other.keys:
            self.append(key, other.days(key), other.scores(key))

    def count(self, key: BufferKey) -> int:
        return int(self._scores.get(key, np.zeros(0)).size)

    def latest_day(self) -> int | None:
        days = [int(item.max()) for item in self._days.values() if item.size]
        return max(days) if days else None

    def pooled(self, alpha: float) -> FloatArray:
        parts = [scores for (_, level), scores in self._scores.items() if level == alpha]
        return np.concatenate(parts) if parts else np.zeros(0)

    def to_arrays(self) -> dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]]:
        arrays: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]] = {}
        for (label, alpha), days in self._days.items():
            name = f"{label.replace('/', '__')}__a{alpha:.4f}".replace(".", "_")
            arrays[f"days__{name}"] = days
            arrays[f"scores__{name}"] = self._scores[(label, alpha)]
        return arrays

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]],
        keys: Iterable[BufferKey],
        window_days: int,
    ) -> ScoreBuffer:
        buffer = cls(window_days)
        for label, alpha in keys:
            name = f"{label.replace('/', '__')}__a{alpha:.4f}".replace(".", "_")
            days = np.asarray(arrays[f"days__{name}"], dtype=np.int64)
            scores = np.asarray(arrays[f"scores__{name}"], dtype=np.float64)
            buffer._days[(label, alpha)] = days
            buffer._scores[(label, alpha)] = scores
        return buffer

    def copy(self) -> ScoreBuffer:
        clone = ScoreBuffer(self.window_days)
        clone._days = {key: value.copy() for key, value in self._days.items()}
        clone._scores = {key: value.copy() for key, value in self._scores.items()}
        return clone


@dataclass(frozen=True, slots=True)
class QuantileResult:
    value: float
    n: int
    saturated: bool
    pooled: bool


@dataclass(frozen=True, slots=True)
class IntervalSet:
    """Intervals per alpha for a batch of rows: ``lower[alpha]``, ``upper[alpha]`` arrays."""

    lower: dict[float, FloatArray]
    upper: dict[float, FloatArray]
    saturated: dict[float, npt.NDArray[np.bool_]]
    alpha_effective: dict[float, FloatArray]

    def width(self, alpha: float) -> FloatArray:
        return self.upper[alpha] - self.lower[alpha]


class MondrianCalibrator:
    def __init__(
        self,
        alphas: Sequence[float],
        horizon_buckets: Sequence[tuple[int, int]],
        available_quantiles: Sequence[float],
        *,
        min_group_size: int = 50,
    ) -> None:
        if not alphas or any(not 0.0 < a < 1.0 for a in alphas):
            raise ValueError("alphas must lie strictly inside (0, 1)")
        self.alphas = tuple(sorted(set(alphas), reverse=True))
        self.horizon_buckets = tuple(horizon_buckets)
        self.pairs = {alpha: quantile_pair(alpha, available_quantiles) for alpha in self.alphas}
        self.min_group_size = min_group_size
        self._edges = np.array([high for _, high in self.horizon_buckets], dtype=np.int64)

    def bucket_of(self, horizon: npt.ArrayLike) -> IntArray:
        h = np.asarray(horizon, dtype=np.int64)
        if h.size and (h.min() < 1 or h.max() > self._edges[-1]):
            raise ValueError("horizon outside the configured buckets")
        return np.searchsorted(self._edges, h, side="left").astype(np.int64)

    def group_labels(self, segments: Sequence[str], horizon: npt.ArrayLike) -> list[str]:
        buckets = self.bucket_of(horizon)
        return [GroupKey(seg, int(b)).label for seg, b in zip(segments, buckets, strict=True)]

    def all_groups(self) -> list[GroupKey]:
        return [
            GroupKey(segment, bucket)
            for segment in SEGMENTS
            for bucket in range(len(self.horizon_buckets))
        ]

    def scores(self, prediction: QuantilePrediction, y: npt.ArrayLike) -> dict[float, FloatArray]:
        """CQR scores per alpha for realized targets ``y``."""
        return {
            alpha: cqr_scores(y, prediction.column(lower), prediction.column(upper))
            for alpha, (lower, upper) in self.pairs.items()
        }

    def quantile(
        self,
        buffer: ScoreBuffer,
        group: str,
        alpha: float,
        *,
        alpha_effective: float | None = None,
        mondrian: bool = True,
    ) -> QuantileResult:
        """Conformal quantile for one group at the nominal alpha (or an ACI-adjusted one)."""
        scores = buffer.scores((group, alpha)) if mondrian else np.zeros(0)
        pooled = False
        if scores.size < self.min_group_size:
            scores = buffer.pooled(alpha)
            pooled = True
        if scores.size == 0:
            raise ValueError(f"no calibration scores for alpha={alpha}")
        level = alpha if alpha_effective is None else alpha_effective
        value = conformal_quantile(scores, min(max(level, 0.0), 1.0))
        saturated = not math.isfinite(value)
        if value == math.inf:
            value = float(scores.max())
        elif value == -math.inf:
            value = float(scores.min())
        return QuantileResult(value=value, n=int(scores.size), saturated=saturated, pooled=pooled)

    def intervals(
        self,
        prediction: QuantilePrediction,
        segments: Sequence[str],
        horizon: npt.ArrayLike,
        buffer: ScoreBuffer,
        *,
        alpha_effective: Mapping[tuple[str, float], float] | None = None,
        mondrian: bool = True,
    ) -> IntervalSet:
        """Conformalized intervals for every alpha; lower bounds are clipped at zero demand."""
        labels = np.array(self.group_labels(segments, horizon))
        n = len(labels)
        lower: dict[float, FloatArray] = {}
        upper: dict[float, FloatArray] = {}
        saturated: dict[float, npt.NDArray[np.bool_]] = {}
        effective: dict[float, FloatArray] = {}
        for alpha, (lo_level, hi_level) in self.pairs.items():
            lo = prediction.column(lo_level).copy()
            hi = prediction.column(hi_level).copy()
            flags = np.zeros(n, dtype=bool)
            used = np.full(n, alpha)
            for label in np.unique(labels):
                mask = labels == label
                level = (
                    None if alpha_effective is None else alpha_effective.get((str(label), alpha))
                )
                result = self.quantile(
                    buffer, str(label), alpha, alpha_effective=level, mondrian=mondrian
                )
                lo[mask] -= result.value
                hi[mask] += result.value
                flags[mask] = result.saturated
                if level is not None:
                    used[mask] = level
            lower[alpha] = np.maximum(lo, 0.0)
            upper[alpha] = np.maximum(hi, lower[alpha])
            saturated[alpha] = flags
            effective[alpha] = used
        return IntervalSet(lower=lower, upper=upper, saturated=saturated, alpha_effective=effective)
