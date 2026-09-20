"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import structlog
from hypothesis import HealthCheck, settings

from conformal_demand_forecaster.data.generator import GeneratorConfig, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.training import TrainingConfig

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

# A panel small enough to train on in a couple of seconds but long enough for a 28-day horizon
# on top of a 56-day calibration window and a training window before that.
SMALL_PANEL = GeneratorConfig(n_series=8, n_days=400, seed=1234, shifts=())


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def small_frame() -> pd.DataFrame:
    """A seeded, contract-valid panel shared by the slower tests (generated once)."""
    return generate_panel(SMALL_PANEL)


@pytest.fixture
def small_panel(small_frame: pd.DataFrame) -> Panel:
    return Panel.from_frame(small_frame, validate=False)


@pytest.fixture
def fast_training() -> TrainingConfig:
    """Training settings that keep a single fit under a second on one CPU core."""
    return TrainingConfig(
        gbm=GbmConfig(
            horizon_buckets=((1, 7), (8, 14)),
            num_iterations=25,
            num_leaves=7,
            min_data_in_leaf=10,
            num_threads=2,
        ),
        train_days=180,
        train_stride=7,
        calibration_days=42,
        min_group_size=20,
    )
