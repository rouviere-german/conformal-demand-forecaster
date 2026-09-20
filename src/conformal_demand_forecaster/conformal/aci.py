"""Adaptive Conformal Inference: online updates of the effective miscoverage level.

Gibbs and Candes ("Adaptive conformal inference under distribution shift", NeurIPS 2021) update
``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`` where ``err_t`` is 1 when the interval built
at level ``alpha_t`` missed. Because the recursion telescopes,

    | (1/T) sum_t err_t - alpha |  <=  (max(alpha_1, 1 - alpha_1) + gamma) / (gamma * T)

for every sequence of errors, without any assumption on the data: coverage is guaranteed in the
long run even under arbitrary distribution shift. The bound relies on ``alpha_t`` staying inside
``[-gamma, 1 + gamma]``, which holds when ``alpha_t <= 0`` yields the whole space (``err = 0``)
and ``alpha_t >= 1`` the empty set (``err = 1``); the calibrator saturates instead of returning
infinite intervals, so :meth:`AlphaState.update` also clips ``alpha_t`` to that range explicitly.

Here one update per day uses the day's miscoverage *rate* across the series of a group (a value
in ``[0, 1]``); the same telescoping argument applies unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["AciController", "AlphaState", "aci_long_run_bound"]


def aci_long_run_bound(alpha_initial: float, gamma: float, steps: int) -> float:
    """Gibbs-Candes bound on ``|mean(err) - alpha|`` after ``steps`` updates.

    The bound holds for *any* error sequence that respects the recursion's premise: an interval
    built at ``alpha_t <= 0`` covers by construction (``err_t = 0``) and one built at
    ``alpha_t >= 1`` is empty (``err_t = 1``). This implementation saturates at the widest
    observed calibration score instead of returning an infinite interval, so the premise can be
    violated once a group's calibration set is exhausted; those rows are flagged ``saturated``
    (and counted by ``app_calibration_saturated_total``) precisely because the guarantee is
    approximate there. See ``tests/unit/test_aci.py`` and docs/adr/0003-aci-over-static-split.md.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return (max(alpha_initial, 1.0 - alpha_initial) + gamma) / (gamma * steps)


@dataclass(slots=True)
class AlphaState:
    target: float
    gamma: float
    alpha: float
    updates: int = 0
    error_sum: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target < 1.0:
            raise ValueError("target alpha must lie strictly inside (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")

    @classmethod
    def initial(cls, target: float, gamma: float) -> AlphaState:
        return cls(target=target, gamma=gamma, alpha=target)

    @property
    def effective(self) -> float:
        """``alpha_t`` clipped to ``[0, 1]`` for the quantile lookup."""
        return min(max(self.alpha, 0.0), 1.0)

    @property
    def long_run_error(self) -> float:
        """``mean(err) - target`` over every update so far (0 before any update)."""
        if self.updates == 0:
            return 0.0
        return self.error_sum / self.updates - self.target

    def update(self, error_rate: float) -> float:
        """Apply one ACI step with the observed miscoverage rate in ``[0, 1]``; return new alpha."""
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must lie in [0, 1]")
        self.alpha += self.gamma * (self.target - error_rate)
        self.alpha = min(max(self.alpha, -self.gamma), 1.0 + self.gamma)
        self.updates += 1
        self.error_sum += error_rate
        return self.alpha

    def reset(self) -> None:
        self.alpha = self.target
        self.updates = 0
        self.error_sum = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "target": self.target,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "updates": self.updates,
            "error_sum": self.error_sum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AlphaState:
        return cls(
            target=float(data["target"]),
            gamma=float(data["gamma"]),
            alpha=float(data["alpha"]),
            updates=int(data["updates"]),
            error_sum=float(data["error_sum"]),
        )


class AciController:
    """One :class:`AlphaState` per (group label, nominal alpha)."""

    def __init__(self, groups: Iterable[str], alphas: Iterable[float], gamma: float) -> None:
        self.gamma = gamma
        self.alphas = tuple(alphas)
        self.states: dict[tuple[str, float], AlphaState] = {
            (group, alpha): AlphaState.initial(alpha, gamma)
            for group in groups
            for alpha in self.alphas
        }

    def effective_alphas(self) -> dict[tuple[str, float], float]:
        return {key: state.effective for key, state in self.states.items()}

    def update(self, group: str, alpha: float, misses: int, n: int) -> float:
        if n <= 0:
            raise ValueError("n must be positive")
        if not 0 <= misses <= n:
            raise ValueError("misses must lie in [0, n]")
        return self.states[(group, alpha)].update(misses / n)

    def reset(self) -> None:
        for state in self.states.values():
            state.reset()

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            f"{group}|{alpha:.4f}": state.to_dict() for (group, alpha), state in self.states.items()
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Any]], gamma: float) -> AciController:
        controller = cls([], [], gamma)
        for key, value in data.items():
            group, _, alpha = key.rpartition("|")
            controller.states[(group, float(alpha))] = AlphaState.from_dict(value)
        controller.alphas = tuple(sorted({alpha for _, alpha in controller.states}, reverse=True))
        return controller
