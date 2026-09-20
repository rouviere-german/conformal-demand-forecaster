"""Calibration artifacts: frozen scores, segment map, ACI state and their integrity checks.

A calibration version is the SHA-256 of the canonical record (scores digest, segment map, data
hash, model version, alphas, buckets). The frozen scores live in an ``.npz`` written and read
with ``allow_pickle=False``; ``calibration.json`` records the npz digest and is itself hashed,
so any edit to either file is detected on load. The ACI alpha state changes at every ingest and
is kept in a separate ``alpha_state.json`` outside the version hash.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from conformal_demand_forecaster.conformal.aci import AciController
from conformal_demand_forecaster.conformal.cqr import ScoreBuffer
from conformal_demand_forecaster.core.canonical import canonical_json, sha256_hex
from conformal_demand_forecaster.core.fileio import atomic_write_text
from conformal_demand_forecaster.ml.serialization import (
    dumps_npz,
    load_json,
    load_npz,
    loads_npz,
    save_json,
    save_npz,
)

__all__ = ["CalibrationArtifact", "CalibrationIntegrityError", "CalibrationRecord"]

CALIBRATION_JSON = "calibration.json"
SCORES_NPZ = "scores.npz"
ALPHA_JSON = "alpha_state.json"
REFERENCE_NPZ = "reference.npz"


class CalibrationIntegrityError(ValueError):
    """A calibration file does not match the digest recorded for it."""


class CalibrationRecord(BaseModel):
    """The versioned, immutable part of a calibration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    created_at: str
    model_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    alphas: tuple[float, ...]
    horizon_buckets: tuple[tuple[int, int], ...]
    quantile_pairs: dict[str, tuple[float, float]]
    window_days: int
    calibrated_through_day: str
    segment_map: dict[str, str]
    group_counts: dict[str, int]
    scores_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_version: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def version_of(record: dict[str, Any]) -> str:
        return sha256_hex(canonical_json(record))


class CalibrationArtifact:
    """Record + frozen score buffer + reference feature samples + mutable ACI state."""

    def __init__(
        self,
        record: CalibrationRecord,
        scores: ScoreBuffer,
        reference: dict[str, npt.NDArray[np.float64]],
        aci: AciController,
    ) -> None:
        self.record = record
        self.scores = scores
        self.reference = reference
        self.aci = aci

    @property
    def version(self) -> str:
        return self.record.calibration_version

    @classmethod
    def build(
        cls,
        *,
        scores: ScoreBuffer,
        segment_map: dict[str, str],
        model_version: str,
        data_sha256: str,
        alphas: tuple[float, ...],
        horizon_buckets: tuple[tuple[int, int], ...],
        quantile_pairs: dict[float, tuple[float, float]],
        reference: dict[str, npt.NDArray[np.float64]],
        calibrated_through_day: str,
        gamma: float,
        aci: AciController | None = None,
        clock: datetime | None = None,
    ) -> CalibrationArtifact:
        scores_bytes = dumps_npz(scores.to_arrays()) if scores.keys else b""
        reference_bytes = dumps_npz(reference) if reference else b""
        groups = sorted({label for label, _ in scores.keys})
        body: dict[str, Any] = {
            "schema_version": 1,
            "created_at": (clock or datetime.now(UTC)).isoformat(),
            "model_version": model_version,
            "data_sha256": data_sha256,
            "alphas": list(alphas),
            "horizon_buckets": [list(item) for item in horizon_buckets],
            "quantile_pairs": {f"{a:.4f}": list(pair) for a, pair in quantile_pairs.items()},
            "window_days": scores.window_days,
            "calibrated_through_day": calibrated_through_day,
            "segment_map": dict(sorted(segment_map.items())),
            "group_counts": {
                f"{label}|{alpha:.4f}": scores.count((label, alpha)) for label, alpha in scores.keys
            },
            "scores_sha256": sha256_hex(scores_bytes),
            "reference_sha256": sha256_hex(reference_bytes),
        }
        version = CalibrationRecord.version_of(body)
        record = CalibrationRecord.model_validate({**body, "calibration_version": version})
        controller = aci if aci is not None else AciController(groups, alphas, gamma)
        return cls(record, scores, reference, controller)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / CALIBRATION_JSON, self.record.model_dump(mode="json"))
        if self.scores.keys:
            save_npz(directory / SCORES_NPZ, self.scores.to_arrays())
        if self.reference:
            save_npz(directory / REFERENCE_NPZ, self.reference)
        self.save_alpha_state(directory)

    def save_alpha_state(self, directory: Path) -> None:
        payload = {"calibration_version": self.version, "states": self.aci.to_dict()}
        atomic_write_text(directory / ALPHA_JSON, canonical_json(payload) + "\n")

    @classmethod
    def load(cls, directory: Path, *, gamma: float) -> CalibrationArtifact:
        """Load and verify every file against the digests in ``calibration.json``."""
        raw = load_json(directory / CALIBRATION_JSON)
        if not isinstance(raw, dict):
            raise CalibrationIntegrityError("calibration.json must hold an object")
        record = CalibrationRecord.model_validate(raw)
        body = record.model_dump(mode="json")
        body.pop("calibration_version")
        if CalibrationRecord.version_of(body) != record.calibration_version:
            raise CalibrationIntegrityError("calibration.json does not match its version digest")

        scores_path = directory / SCORES_NPZ
        scores_bytes = scores_path.read_bytes() if scores_path.exists() else b""
        if sha256_hex(scores_bytes) != record.scores_sha256:
            raise CalibrationIntegrityError("scores.npz does not match the recorded digest")
        keys = [
            (label, float(alpha))
            for label, _, alpha in (key.rpartition("|") for key in record.group_counts)
        ]
        scores = (
            ScoreBuffer.from_arrays(loads_npz(scores_bytes), keys, record.window_days)
            if scores_bytes
            else ScoreBuffer(record.window_days)
        )

        reference_path = directory / REFERENCE_NPZ
        reference_bytes = reference_path.read_bytes() if reference_path.exists() else b""
        if sha256_hex(reference_bytes) != record.reference_sha256:
            raise CalibrationIntegrityError("reference.npz does not match the recorded digest")
        reference = (
            {k: np.asarray(v, dtype=np.float64) for k, v in load_npz(reference_path).items()}
            if reference_bytes
            else {}
        )

        alpha_path = directory / ALPHA_JSON
        groups = sorted({label for label, _ in keys})
        if alpha_path.exists():
            state = load_json(alpha_path)
            if (
                not isinstance(state, dict)
                or state.get("calibration_version") != record.calibration_version
            ):
                raise CalibrationIntegrityError("alpha_state.json belongs to another calibration")
            aci = AciController.from_dict(state["states"], gamma)
        else:
            aci = AciController(groups, record.alphas, gamma)
        return cls(record, scores, reference, aci)
