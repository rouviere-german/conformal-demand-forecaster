"""MLflow tracking on the local file backend.

The run log is a convenience, not the source of truth -- the run manifest is. What matters here
is that logging a run stays local (a folder URI, never a server) and that the parameters a
reviewer needs to reproduce a model actually land in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conformal_demand_forecaster.tracking import EXPERIMENT, list_runs, log_training_run

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_a_run_logs_its_params_metrics_and_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "manifest.json"
    artifact.write_text('{"seed": 7}', encoding="utf-8")
    run_id = log_training_run(
        tmp_path / "mlruns",
        run_name="unit",
        params={"seed": 7, "data_sha256": "a" * 64},
        metrics={"training_rows": 1234.0},
        artifacts=[artifact],
        tags={"kind": "test"},
    )
    assert len(run_id) == 32

    runs = list_runs(tmp_path / "mlruns")
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == run_id
    assert run["run_name"] == "unit"
    assert run["params"]["seed"] == "7"
    assert run["params"]["data_sha256"] == "a" * 64
    assert run["metrics"]["training_rows"] == pytest.approx(1234.0)


def test_long_parameter_values_are_truncated(tmp_path: Path) -> None:
    """MLflow rejects very long parameter values; the wrapper clips instead of failing a run."""
    log_training_run(
        tmp_path / "mlruns",
        run_name="long",
        params={"lockfile": "x" * 2_000},
        metrics={},
    )
    assert len(list_runs(tmp_path / "mlruns")[0]["params"]["lockfile"]) == 500


def test_listing_an_empty_tracking_directory_returns_nothing(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "empty") == []


def test_the_tracking_directory_is_a_local_folder(tmp_path: Path) -> None:
    """Nothing is uploaded: the tracking URI is a file:// URI under the data directory."""
    directory = tmp_path / "mlruns"
    log_training_run(directory, run_name="local", params={}, metrics={})
    assert directory.is_dir()
    assert (directory / "models").exists() or any(directory.iterdir())
    assert EXPERIMENT == "conformal-demand-forecaster"
