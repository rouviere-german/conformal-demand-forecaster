"""Typed settings: fail-closed defaults, role-scoped API keys and validated forecasting knobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conformal_demand_forecaster.core.config import ROLES, Settings, get_settings

pytestmark = pytest.mark.unit

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.reader_key_sha256 == []
    assert settings.ingestor_key_sha256 == []
    assert settings.operator_key_sha256 == []
    assert settings.api_key_digests == []
    assert settings.key_roles == {}
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"


def test_role_lists_are_parsed_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_READER_KEY_SHA256", f"{DIGEST_A}, {DIGEST_B}")
    monkeypatch.setenv("APP_OPERATOR_KEY_SHA256", DIGEST_B)
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    settings = Settings(_env_file=None)
    assert settings.reader_key_sha256 == [DIGEST_A, DIGEST_B]
    assert settings.api_key_digests == [DIGEST_A, DIGEST_B]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]


def test_a_key_holds_exactly_the_roles_it_was_listed_under() -> None:
    settings = Settings(
        _env_file=None,
        reader_key_sha256=[DIGEST_A, DIGEST_B],
        ingestor_key_sha256=[DIGEST_A],
        operator_key_sha256=[DIGEST_B],
    )
    roles = settings.key_roles
    assert roles[DIGEST_A[:12]] == frozenset({"reader", "ingestor"})
    assert roles[DIGEST_B[:12]] == frozenset({"reader", "operator"})
    assert set(ROLES) == {"reader", "ingestor", "operator"}


def test_the_key_id_is_a_non_secret_prefix() -> None:
    settings = Settings(_env_file=None, reader_key_sha256=[DIGEST_A])
    key_id = next(iter(settings.key_roles))
    assert len(key_id) == 12
    assert DIGEST_A.startswith(key_id)


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
@pytest.mark.parametrize(
    "field", ["reader_key_sha256", "ingestor_key_sha256", "operator_key_sha256"]
)
def test_non_sha256_key_hashes_are_rejected(field: str, digest: str) -> None:
    values: dict[str, Any] = {field: [digest]}
    with pytest.raises(ValidationError, match="SHA-256"):
        Settings(_env_file=None, **values)


def test_coverage_levels_become_sorted_alphas() -> None:
    settings = Settings(_env_file=None, coverage_levels=[0.95, 0.8])
    assert settings.coverage_levels == [0.8, 0.95]
    assert settings.alphas == (0.2, 0.05)


@pytest.mark.parametrize("levels", [[], [0.4], [1.0], [0.8, 0.8]])
def test_invalid_coverage_levels_are_rejected(levels: list[float]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, coverage_levels=levels)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forecast_horizon", 0),
        ("forecast_horizon", 200),
        ("aci_gamma", 0.0),
        ("aci_gamma", 0.9),
        ("calibration_window_days", 3),
        ("coverage_window_days", 1),
        ("coverage_tolerance", 0.5),
        ("coverage_significance", 0.9),
        ("ingest_max_records", 0),
        ("guard_z_threshold", 0.0),
        ("freshness_max_days", 0),
        ("holding_cost", 0.0),
        ("stockout_cost", -1.0),
        ("service_name", "Bad Name"),
        ("port", 0),
    ],
)
def test_out_of_range_settings_are_rejected(field: str, value: object) -> None:
    values: dict[str, Any] = {field: value}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_settings_are_frozen() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.port = 9000  # type: ignore[misc]


def test_data_dir_is_a_path() -> None:
    settings = Settings(_env_file=None, data_dir="var/run")
    assert isinstance(settings.data_dir, Path)


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {f"APP_{name.upper()}" for name in Settings.model_fields}
    assert documented == expected


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
