"""Calibration-artifact integrity.

A calibration version is the SHA-256 of its canonical record, and the record carries the digests
of the score and reference files. The threat is an attacker (or a corrupt disk) editing the
frozen scores so that every future interval is wrong while the version string still looks right.
Every test below tampers with one file and asserts the load refuses it.

The artifacts are JSON and npz written with ``allow_pickle=False``: loading one must never be
able to execute code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conformal_demand_forecaster.conformal.aci import AciController
from conformal_demand_forecaster.conformal.artifacts import (
    ALPHA_JSON,
    CALIBRATION_JSON,
    REFERENCE_NPZ,
    SCORES_NPZ,
    CalibrationArtifact,
    CalibrationIntegrityError,
    CalibrationRecord,
)
from conformal_demand_forecaster.conformal.cqr import ScoreBuffer

pytestmark = pytest.mark.unit

ALPHAS = (0.2, 0.05)
BUCKETS = ((1, 7), (8, 14))
PAIRS = {0.2: (0.1, 0.9), 0.05: (0.05, 0.95)}
MODEL_VERSION = "a" * 64
DATA_SHA = "b" * 64


def make_artifact(**overrides: object) -> CalibrationArtifact:
    scores = ScoreBuffer(window_days=56)
    rng = np.random.default_rng(0)
    for label in ("smooth/b0", "intermittent/b0"):
        for alpha in ALPHAS:
            scores.append((label, alpha), np.arange(30), rng.normal(size=30))
    kwargs: dict[str, object] = {
        "scores": scores,
        "segment_map": {"S0001_ST00": "smooth", "S0002_ST00": "intermittent"},
        "model_version": MODEL_VERSION,
        "data_sha256": DATA_SHA,
        "alphas": ALPHAS,
        "horizon_buckets": BUCKETS,
        "quantile_pairs": PAIRS,
        "reference": {"sales__smooth": rng.normal(size=100)},
        "calibrated_through_day": "2024-06-01",
        "gamma": 0.03,
    }
    kwargs.update(overrides)
    return CalibrationArtifact.build(**kwargs)  # type: ignore[arg-type]


def test_a_saved_artifact_loads_back_identically(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    loaded = CalibrationArtifact.load(tmp_path, gamma=0.03)
    assert loaded.version == artifact.version
    assert loaded.record == artifact.record
    assert sorted(loaded.scores.keys) == sorted(artifact.scores.keys)
    for key in artifact.scores.keys:
        assert loaded.scores.scores(key).tolist() == artifact.scores.scores(key).tolist()
    assert set(loaded.reference) == set(artifact.reference)


def test_the_version_is_a_deterministic_hash_of_the_record() -> None:
    clock = __import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC)
    first = make_artifact(clock=clock)
    second = make_artifact(clock=clock)
    assert first.version == second.version
    changed = make_artifact(clock=clock, calibrated_through_day="2024-06-02")
    assert changed.version != first.version


def test_tampering_with_the_scores_file_is_detected(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    poisoned = {name: values * 10.0 for name, values in np.load(tmp_path / SCORES_NPZ).items()}
    np.savez(tmp_path / SCORES_NPZ, **poisoned)
    with pytest.raises(CalibrationIntegrityError, match=r"scores\.npz"):
        CalibrationArtifact.load(tmp_path, gamma=0.03)


def test_deleting_the_scores_file_is_detected(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    (tmp_path / SCORES_NPZ).unlink()
    with pytest.raises(CalibrationIntegrityError, match=r"scores\.npz"):
        CalibrationArtifact.load(tmp_path, gamma=0.03)


def test_tampering_with_the_reference_file_is_detected(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    np.savez(tmp_path / REFERENCE_NPZ, sales__smooth=np.zeros(100))
    with pytest.raises(CalibrationIntegrityError, match=r"reference\.npz"):
        CalibrationArtifact.load(tmp_path, gamma=0.03)


def test_editing_the_record_without_rehashing_is_detected(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    path = tmp_path / CALIBRATION_JSON
    record = json.loads(path.read_text(encoding="utf-8"))
    record["calibrated_through_day"] = "2030-01-01"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CalibrationIntegrityError, match="version digest"):
        CalibrationArtifact.load(tmp_path, gamma=0.03)


def test_an_alpha_state_from_another_calibration_is_refused(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    state = json.loads((tmp_path / ALPHA_JSON).read_text(encoding="utf-8"))
    state["calibration_version"] = "c" * 64
    (tmp_path / ALPHA_JSON).write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(CalibrationIntegrityError, match="another calibration"):
        CalibrationArtifact.load(tmp_path, gamma=0.03)


def test_a_non_object_calibration_file_is_refused(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / CALIBRATION_JSON).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CalibrationIntegrityError, match="must hold an object"):
        CalibrationArtifact.load(tmp_path, gamma=0.03)


def test_a_missing_alpha_state_falls_back_to_a_fresh_controller(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    (tmp_path / ALPHA_JSON).unlink()
    loaded = CalibrationArtifact.load(tmp_path, gamma=0.03)
    assert all(state.alpha == state.target for state in loaded.aci.states.values())


def test_alpha_state_survives_a_save_and_load_round_trip(tmp_path: Path) -> None:
    artifact = make_artifact()
    artifact.save(tmp_path)
    artifact.aci.update("smooth/b0", 0.2, misses=4, n=4)
    artifact.save_alpha_state(tmp_path)
    loaded = CalibrationArtifact.load(tmp_path, gamma=0.03)
    assert loaded.aci.states[("smooth/b0", 0.2)].alpha < 0.2
    assert loaded.version == artifact.version  # alpha state is outside the version hash


def test_an_empty_calibration_still_round_trips(tmp_path: Path) -> None:
    artifact = make_artifact(scores=ScoreBuffer(window_days=56), reference={})
    artifact.save(tmp_path)
    loaded = CalibrationArtifact.load(tmp_path, gamma=0.03)
    assert loaded.scores.keys == []
    assert loaded.reference == {}
    assert not (tmp_path / SCORES_NPZ).exists()


def test_the_record_rejects_malformed_digests() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalibrationRecord.model_validate(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "model_version": "not-a-digest",
                "data_sha256": DATA_SHA,
                "alphas": ALPHAS,
                "horizon_buckets": BUCKETS,
                "quantile_pairs": {"0.2000": (0.1, 0.9)},
                "window_days": 56,
                "calibrated_through_day": "2024-06-01",
                "segment_map": {},
                "group_counts": {},
                "scores_sha256": DATA_SHA,
                "reference_sha256": DATA_SHA,
                "calibration_version": DATA_SHA,
            }
        )


def test_a_supplied_aci_controller_is_kept() -> None:
    controller = AciController(["smooth/b0"], ALPHAS, 0.03)
    controller.update("smooth/b0", 0.2, misses=1, n=1)
    artifact = make_artifact(aci=controller)
    assert artifact.aci.states[("smooth/b0", 0.2)].updates == 1


def test_no_pickle_is_used_by_the_artifact_files(tmp_path: Path) -> None:
    """``np.load`` without ``allow_pickle`` must be enough to read every array back."""
    artifact = make_artifact()
    artifact.save(tmp_path)
    for name in (SCORES_NPZ, REFERENCE_NPZ):
        with np.load(tmp_path / name, allow_pickle=False) as data:
            assert len(data.files) > 0


def test_an_alpha_state_written_under_another_gamma_is_refused(tmp_path: Path) -> None:
    """The step size is not covered by the record digest, so the load fails closed on it."""
    artifact = make_artifact()
    artifact.save(tmp_path)
    with pytest.raises(CalibrationIntegrityError, match="gamma"):
        CalibrationArtifact.load(tmp_path, gamma=0.09)
    assert CalibrationArtifact.load(tmp_path, gamma=artifact.aci.gamma).version == artifact.version
