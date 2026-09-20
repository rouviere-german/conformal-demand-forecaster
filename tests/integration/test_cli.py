"""The command line: generate, train, forecast, backtest, key generation, audit verify, serve."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from conformal_demand_forecaster import cli
from conformal_demand_forecaster.core import tracing
from conformal_demand_forecaster.core.security import hash_api_key
from conformal_demand_forecaster.monitoring.audit import AuditChain

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_SERIES = 6
N_DAYS = 200
SEED = 31337
BACKTEST_DAYS = 560


def read_json(capsys: pytest.CaptureFixture[str]) -> Any:
    """Every subcommand writes exactly one JSON document to stdout; logs go to stderr."""
    return json.loads(capsys.readouterr().out)


def generate_panel_csv(path: Path, days: int, capsys: pytest.CaptureFixture[str]) -> Path:
    args = [
        "generate",
        "--out",
        str(path),
        "--n-series",
        str(N_SERIES),
        "--n-days",
        str(days),
        "--seed",
        str(SEED),
    ]
    assert cli.main(args) == 0
    capsys.readouterr()
    return path


@pytest.fixture
def panel_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    return generate_panel_csv(tmp_path / "panel.csv", N_DAYS, capsys)


@pytest.fixture
def long_panel_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """The default backtest window needs roughly a year and a half of history."""
    return generate_panel_csv(tmp_path / "long-panel.csv", BACKTEST_DAYS, capsys)


def test_generate_writes_a_reproducible_panel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "panel.csv"
    args = ["generate", "--out", str(out), "--n-series", "4", "--n-days", "120", "--seed", "7"]
    assert cli.main(args) == 0
    payload = read_json(capsys)
    assert payload == {"rows": 480, "series": 4, "days": 120, "path": str(out)}
    first = out.read_bytes()

    assert cli.main(args) == 0
    capsys.readouterr()
    assert out.read_bytes() == first


@pytest.mark.usefixtures("restore_logging")
def test_train_forecast_and_audit_verify_work_together(
    panel_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "var"
    assert (
        cli.main(
            ["train", "--panel", str(panel_csv), "--data-dir", str(data_dir), "--seed", str(SEED)]
        )
        == 0
    )
    install = read_json(capsys)
    assert len(install["model_version"]) == 64
    assert install["forecast_rows"] == N_SERIES * 28

    assert cli.main(["forecast", "--data-dir", str(data_dir)]) == 0
    batch = read_json(capsys)
    assert batch["rows"] == N_SERIES * 28
    assert batch["origin"] == install["origin"]

    assert cli.main(["audit", "verify", "--data-dir", str(data_dir)]) == 0
    audit = read_json(capsys)
    assert audit["ok"] is True
    assert audit["entries"] == 1

    head = AuditChain(data_dir / "audit.jsonl").head
    assert cli.main(["audit", "verify", "--data-dir", str(data_dir), "--expect-head", head]) == 0
    capsys.readouterr()
    assert (
        cli.main(["audit", "verify", "--data-dir", str(data_dir), "--expect-head", "0" * 64]) == 1
    )
    assert read_json(capsys)["ok"] is False


def test_audit_verify_reports_a_broken_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "var"
    chain = AuditChain(data_dir / "audit.jsonl")
    chain.append("trainer", "model_installed", {"model_version": "a" * 64})
    chain.append("key:operator", "recalibration_approved", {"proposal_id": "b" * 32})
    path = data_dir / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n" + lines[0] + "\n", encoding="utf-8")
    assert cli.main(["audit", "verify", "--data-dir", str(data_dir)]) == 1
    payload = read_json(capsys)
    assert payload["ok"] is False
    assert payload["problem"]


def test_backtest_writes_metrics_and_a_digest(
    long_panel_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "backtest.json"
    assert (
        cli.main(
            [
                "backtest",
                "--panel",
                str(long_panel_csv),
                "--n-origins",
                "2",
                "--out",
                str(out),
                "--ets-series-limit",
                "2",
            ]
        )
        == 0
    )
    printed = read_json(capsys)
    assert printed["coverage_80_mondrian"] >= 0.0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved["digest"]) == 64
    assert len(saved["per_origin"]) == 2
    assert saved["metrics"]["origins"] == 2


def test_generate_api_key_prints_a_key_and_its_digest(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])
    assert len(payload["api_key_sha256"]) == 64
    # The guidance to store only the digest goes to stderr so piping stdout stays clean JSON.
    assert "APP_<ROLE>_KEY_SHA256" in captured.err


@pytest.mark.usefixtures("restore_logging")
def test_serve_configures_uvicorn_safely(
    panel_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import uvicorn

    data_dir = tmp_path / "var"
    assert cli.main(["train", "--panel", str(panel_csv), "--data-dir", str(data_dir)]) == 0
    capsys.readouterr()

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)

    from conformal_demand_forecaster.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        get_settings.cache_clear()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_an_unknown_command_exits_with_usage() -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["nonsense"])
    assert info.value.code == 2


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["conformal_demand_forecaster", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("conformal_demand_forecaster", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
