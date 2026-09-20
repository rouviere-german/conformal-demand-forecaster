"""MLflow tracking on the local file backend (mlflow-skinny, no server).

Every training run logs the generator seed, data hash, model and calibration parameters, the
lock-file hash and the headline metrics, plus the run manifest as an artifact, so a run can be
reproduced and compared. The tracking directory is a plain folder; nothing is uploaded.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
# MLflow >= 3.4 refuses the filesystem tracking backend unless this opt-out is set. A local
# folder is deliberate here: the whole repository runs offline with no tracking server and no
# database, and the run manifest (not MLflow) is the authoritative reproducibility record.
# Non-goals in the README documents that the file store is in upstream maintenance mode.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

__all__ = ["EXPERIMENT", "list_runs", "log_training_run"]

EXPERIMENT = "conformal-demand-forecaster"


def _tracking_uri(directory: Path) -> str:
    """The file URI of ``directory``, with the layout the file store expects already in place.

    MLflow keeps one store object per tracking URI for the life of the process, and that object
    assumes its own folders still exist. A tracking directory that is recreated between two runs
    in one process -- which is what the demo does to start from a clean slate -- would otherwise
    fail on the missing ``.trash`` folder the cached store never recreates.
    """
    (directory / ".trash").mkdir(parents=True, exist_ok=True)
    return directory.resolve().as_uri()


def log_training_run(
    tracking_dir: Path,
    *,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    artifacts: Iterable[Path] = (),
    tags: Mapping[str, str] | None = None,
) -> str:
    """Log one run and return its id."""
    import mlflow  # noqa: PLC0415 - slow import, only needed when tracking

    mlflow.set_tracking_uri(_tracking_uri(tracking_dir))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return str(run.info.run_id)


def list_runs(tracking_dir: Path) -> list[dict[str, Any]]:
    """Runs of the experiment as plain dicts (run id, name, params, metrics)."""
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(_tracking_uri(tracking_dir))
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])
    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        }
        for run in runs
    ]
