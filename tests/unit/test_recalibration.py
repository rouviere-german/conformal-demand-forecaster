"""The approval gate on recalibration, and the hash chain that records every decision.

Promoting a new calibration changes the intervals every planner sees, so it is the one action in
this system that needs a human. The workflow enforces a state machine -- pending -> approved ->
applied, or pending -> rejected -- and appends each transition to the audit chain. The tests
assert that no shortcut exists: an unapproved or already-decided proposal cannot be applied, and
the chain notices any later edit of the log.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from conformal_demand_forecaster.monitoring.audit import AuditChain, verify_chain
from conformal_demand_forecaster.monitoring.coverage import MonitorStatus
from conformal_demand_forecaster.monitoring.recalibration import (
    RecalibrationWorkflow,
    WorkflowError,
)

pytestmark = pytest.mark.unit


def alarm(segment: str = "smooth", alpha: float = 0.2, day: int = 100) -> MonitorStatus:
    return MonitorStatus(
        segment=segment, alpha=alpha, day=day, n=200, misses=70, p_value=1e-6, alarm=True
    )


def quiet() -> MonitorStatus:
    return MonitorStatus(
        segment="smooth", alpha=0.2, day=100, n=200, misses=40, p_value=0.4, alarm=False
    )


@pytest.fixture
def workflow(tmp_path: Path) -> RecalibrationWorkflow:
    return RecalibrationWorkflow(tmp_path / "proposals.json", AuditChain(tmp_path / "audit.jsonl"))


def test_an_alarm_opens_exactly_one_pending_proposal(workflow: RecalibrationWorkflow) -> None:
    proposal = workflow.open(alarm())
    assert proposal is not None
    assert proposal.status == "pending"
    assert proposal.segment == "smooth"
    assert proposal.coverage == pytest.approx(0.65)
    # A second alarm for the same key does not pile up proposals.
    assert workflow.open(alarm(day=101)) is None
    assert len(workflow.list("pending")) == 1


def test_a_quiet_status_opens_nothing(workflow: RecalibrationWorkflow) -> None:
    assert workflow.open(quiet()) is None
    assert workflow.list() == []


def test_different_segments_and_levels_get_their_own_proposals(
    workflow: RecalibrationWorkflow,
) -> None:
    workflow.open(alarm(segment="smooth", alpha=0.2))
    workflow.open(alarm(segment="intermittent", alpha=0.2))
    workflow.open(alarm(segment="smooth", alpha=0.05))
    assert len(workflow.list("pending")) == 3


def test_an_unapproved_proposal_cannot_be_applied(workflow: RecalibrationWorkflow) -> None:
    proposal = workflow.open(alarm())
    assert proposal is not None
    with pytest.raises(WorkflowError, match="only an approved proposal"):
        workflow.mark_applied(proposal.id, "key:abc", from_version="a" * 64, to_version="b" * 64)


def test_a_rejected_proposal_cannot_be_approved_afterwards(
    workflow: RecalibrationWorkflow,
) -> None:
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.reject(proposal.id, "key:operator", "expected seasonal dip")
    with pytest.raises(WorkflowError, match="already rejected"):
        workflow.approve(proposal.id, "key:operator")
    with pytest.raises(WorkflowError, match="only an approved proposal"):
        workflow.mark_applied(
            proposal.id, "key:operator", from_version="a" * 64, to_version="b" * 64
        )


def test_an_approved_proposal_cannot_be_approved_twice(workflow: RecalibrationWorkflow) -> None:
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.approve(proposal.id, "key:operator")
    with pytest.raises(WorkflowError, match="already approved"):
        workflow.approve(proposal.id, "key:other")


def test_unknown_proposals_are_refused(workflow: RecalibrationWorkflow) -> None:
    missing = "f" * 32
    calls: list[Callable[[], object]] = [
        lambda: workflow.get(missing),
        lambda: workflow.approve(missing, "key:operator"),
        lambda: workflow.reject(missing, "key:operator"),
        lambda: workflow.mark_applied(
            missing, "key:operator", from_version="a" * 64, to_version="b" * 64
        ),
    ]
    for call in calls:
        with pytest.raises(WorkflowError, match="unknown proposal"):
            call()


def test_the_happy_path_records_who_decided_and_what_was_promoted(
    workflow: RecalibrationWorkflow,
) -> None:
    proposal = workflow.open(alarm())
    assert proposal is not None
    approved = workflow.approve(proposal.id, "key:operator1", "regime shift confirmed")
    assert approved.status == "approved"
    assert approved.decided_by == "key:operator1"
    assert approved.decided_at is not None
    applied = workflow.mark_applied(
        proposal.id, "key:operator1", from_version="a" * 64, to_version="b" * 64
    )
    assert applied.status == "applied"
    assert applied.applied_version == "b" * 64
    assert workflow.get(proposal.id).status == "applied"
    assert workflow.list("pending") == []


def test_every_transition_lands_in_the_audit_chain(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    workflow = RecalibrationWorkflow(tmp_path / "proposals.json", AuditChain(audit_path))
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.approve(proposal.id, "key:operator1")
    workflow.mark_applied(proposal.id, "key:operator1", from_version="a" * 64, to_version="b" * 64)
    rejected = workflow.open(alarm(segment="intermittent"))
    assert rejected is not None
    workflow.reject(rejected.id, "key:operator2", "known promotion")

    actions = [entry.action for entry in AuditChain(audit_path).entries()]
    assert actions == [
        "recalibration_proposed",
        "recalibration_approved",
        "recalibration_applied",
        "recalibration_proposed",
        "recalibration_rejected",
    ]
    verification = verify_chain(audit_path)
    assert verification.ok is True
    assert verification.entries == 5


def test_the_applied_entry_names_the_approver_and_both_versions(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    workflow = RecalibrationWorkflow(tmp_path / "proposals.json", AuditChain(audit_path))
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.approve(proposal.id, "key:operator1")
    workflow.mark_applied(proposal.id, "key:runtime", from_version="a" * 64, to_version="b" * 64)
    applied = AuditChain(audit_path).entries()[-1]
    assert applied.data["approved_by"] == "key:operator1"
    assert applied.data["from_version"] == "a" * 64
    assert applied.data["to_version"] == "b" * 64


def test_editing_an_audit_entry_breaks_the_chain(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    workflow = RecalibrationWorkflow(tmp_path / "proposals.json", AuditChain(audit_path))
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.approve(proposal.id, "key:operator1")
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["actor"] = "key:attacker"
    lines[1] = json.dumps(record, separators=(",", ":"), sort_keys=True)
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verification = verify_chain(audit_path)
    assert verification.ok is False
    assert verification.problem is not None
    with pytest.raises(ValueError, match="broken audit chain"):
        AuditChain(audit_path).append("key:operator1", "recalibration_proposed", {})


def test_removing_an_audit_entry_breaks_the_chain(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    workflow = RecalibrationWorkflow(tmp_path / "proposals.json", AuditChain(audit_path))
    first = workflow.open(alarm())
    assert first is not None
    workflow.approve(first.id, "key:operator1")
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    audit_path.write_text(lines[1] + "\n", encoding="utf-8")
    assert verify_chain(audit_path).ok is False


def test_listing_filters_by_status_and_is_stable(workflow: RecalibrationWorkflow) -> None:
    first = workflow.open(alarm(segment="smooth"))
    second = workflow.open(alarm(segment="intermittent"))
    assert first is not None
    assert second is not None
    workflow.reject(second.id, "key:operator")
    assert [p.id for p in workflow.list("pending")] == [first.id]
    assert [p.id for p in workflow.list("rejected")] == [second.id]
    assert len(workflow.list()) == 2


def test_the_proposal_file_is_valid_json_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "proposals.json"
    workflow = RecalibrationWorkflow(path, AuditChain(tmp_path / "audit.jsonl"))
    proposal = workflow.open(alarm())
    assert proposal is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {proposal.id}
    assert payload[proposal.id]["status"] == "pending"


# --- the file is not the authority -------------------------------------------------------------
# Every other artifact in the data directory is digest-verified on load. The proposal file is
# plain JSON, so the gate takes the audit chain as the record of what was decided and refuses any
# transition the chain does not support.


def _rewrite(path: Path, proposal_id: str, **fields: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[proposal_id].update(fields)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_rejected_proposal_cannot_be_resurrected_by_editing_the_file(tmp_path: Path) -> None:
    path = tmp_path / "proposals.json"
    workflow = RecalibrationWorkflow(path, AuditChain(tmp_path / "audit.jsonl"))
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.reject(proposal.id, "key:operator")

    _rewrite(path, proposal.id, status="approved", decided_by="attacker")

    with pytest.raises(WorkflowError, match="audit chain"):
        workflow.mark_applied(proposal.id, "attacker", from_version="a" * 64, to_version="b" * 64)
    assert workflow.get(proposal.id).status == "approved"  # the file still lies
    assert [entry.action for entry in workflow.audit.entries()] == [
        "recalibration_proposed",
        "recalibration_rejected",
    ]


def test_a_decided_proposal_cannot_be_reset_to_pending_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "proposals.json"
    workflow = RecalibrationWorkflow(path, AuditChain(tmp_path / "audit.jsonl"))
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.reject(proposal.id, "key:operator")

    _rewrite(path, proposal.id, status="pending", decided_by=None, decided_at=None)

    with pytest.raises(WorkflowError, match="audit chain"):
        workflow.approve(proposal.id, "attacker")


def test_a_proposal_that_was_never_opened_cannot_be_decided(tmp_path: Path) -> None:
    path = tmp_path / "proposals.json"
    workflow = RecalibrationWorkflow(path, AuditChain(tmp_path / "audit.jsonl"))
    proposal = workflow.open(alarm())
    assert proposal is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    forged = dict(payload[proposal.id])
    forged_id = "f" * 32
    forged["id"] = forged_id
    payload[forged_id] = forged
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="audit chain"):
        workflow.approve(forged_id, "attacker")


def test_the_applied_entry_names_the_approver_from_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "proposals.json"
    workflow = RecalibrationWorkflow(path, AuditChain(tmp_path / "audit.jsonl"))
    proposal = workflow.open(alarm())
    assert proposal is not None
    workflow.approve(proposal.id, "key:operator")
    _rewrite(path, proposal.id, decided_by="attacker")

    applied = workflow.mark_applied(
        proposal.id, "key:operator", from_version="a" * 64, to_version="b" * 64
    )

    assert applied.status == "applied"
    entry = workflow.audit.entries()[-1]
    assert entry.action == "recalibration_applied"
    assert entry.data["approved_by"] == "key:operator"


def test_a_broken_audit_chain_blocks_every_decision(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    path = tmp_path / "proposals.json"
    workflow = RecalibrationWorkflow(path, AuditChain(audit_path))
    proposal = workflow.open(alarm())
    assert proposal is not None
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8").replace('"pending"', '"approved"'),
        encoding="utf-8",
    )
    with pytest.raises(WorkflowError, match="audit chain"):
        workflow.approve(proposal.id, "key:operator")
