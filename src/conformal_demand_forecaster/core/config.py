"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. API keys are configured as SHA-256 digests grouped by role
(reader, ingestor, operator), so the process never holds raw keys and every key carries exactly
the roles it was listed under.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["ROLES", "Role", "Settings", "get_settings"]

Role = Literal["reader", "ingestor", "operator"]
ROLES: tuple[Role, ...] = ("reader", "ingestor", "operator")

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _check_digests(value: list[str]) -> list[str]:
    for digest in value:
        if not _SHA256_HEX.fullmatch(digest):
            raise ValueError("API key digests must be lowercase hex SHA-256 digests")
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="conformal-demand-forecaster", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security: one digest list per role ---
    reader_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    ingestor_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    operator_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- storage ---
    data_dir: Path = Path("var")

    # --- forecasting and calibration ---
    forecast_horizon: int = Field(default=28, ge=1, le=90)
    coverage_levels: Annotated[list[float], NoDecode] = Field(default_factory=lambda: [0.8, 0.95])
    # 0.08 comes from bench/bench_shift.py --sweep: it holds post-shift coverage error near
    # one point at both levels, for about 3% more interval width than 0.03.
    aci_gamma: float = Field(default=0.08, gt=0, le=0.5)
    calibration_window_days: int = Field(default=56, ge=7, le=365)

    # --- coverage monitor (SLO: within tolerance of nominal over the window) ---
    coverage_window_days: int = Field(default=28, ge=7, le=365)
    coverage_tolerance: float = Field(default=0.02, ge=0, le=0.2)
    coverage_significance: float = Field(default=0.01, gt=0, lt=0.5)
    coverage_min_observations: int = Field(default=30, ge=1)

    # --- actuals ingest ---
    ingest_max_records: int = Field(default=1_000, ge=1, le=100_000)
    guard_z_threshold: float = Field(default=10.0, gt=0)
    freshness_max_days: int = Field(default=2, ge=1, le=365)

    # --- newsvendor defaults (per unit) ---
    holding_cost: float = Field(default=1.0, gt=0)
    stockout_cost: float = Field(default=4.0, gt=0)

    @field_validator(
        "reader_key_sha256",
        "ingestor_key_sha256",
        "operator_key_sha256",
        "cors_allow_origins",
        "coverage_levels",
        mode="before",
    )
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("reader_key_sha256", "ingestor_key_sha256", "operator_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        return _check_digests(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @field_validator("coverage_levels")
    @classmethod
    def _check_levels(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("at least one coverage level is required")
        for level in value:
            if not 0.5 <= level < 1.0:
                raise ValueError("coverage levels must lie in [0.5, 1)")
        if len(set(value)) != len(value):
            raise ValueError("coverage levels must be distinct")
        return sorted(value)

    @property
    def api_key_digests(self) -> list[str]:
        """Every configured key digest, whatever its roles (used for authentication)."""
        return sorted(
            set(self.reader_key_sha256)
            | set(self.ingestor_key_sha256)
            | set(self.operator_key_sha256)
        )

    @property
    def key_roles(self) -> dict[str, frozenset[Role]]:
        """Map from key id (first 12 hex characters of the digest) to the roles it holds."""
        roles: dict[str, set[Role]] = {}
        assignments: tuple[tuple[Role, list[str]], ...] = (
            ("reader", self.reader_key_sha256),
            ("ingestor", self.ingestor_key_sha256),
            ("operator", self.operator_key_sha256),
        )
        for role, digests in assignments:
            for digest in digests:
                roles.setdefault(digest[:12], set()).add(role)
        return {key_id: frozenset(items) for key_id, items in roles.items()}

    @property
    def alphas(self) -> tuple[float, ...]:
        """Nominal miscoverage levels (1 - coverage level), largest first."""
        return tuple(round(1.0 - level, 6) for level in self.coverage_levels)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
