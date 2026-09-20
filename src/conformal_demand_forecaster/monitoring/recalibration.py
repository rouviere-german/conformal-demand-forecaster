"""Approval-gated recalibration proposals.

A coverage alarm opens a proposal (one pending proposal per segment and level). Only an operator
can approve or reject it, and only an approved proposal can be applied, which promotes a new
calibration version. Every transition is appended to the hash-chained audit log. Proposals live
in a JSON file written atomically under a file lock.

The JSON file is a projection, not the authority. It carries no integrity tag, so before any
transition the workflow replays the audit chain for that proposal and requires the chain to agree
with the state on disk. An edited file cannot resurrect a rejected proposal, reopen a decided one
or introduce a proposal that was never proposed, and the applied entry names the approver the
chain recorded rather than the one the file claims.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from conformal_demand_forecaster.core.fileio import atomic_write_text, file_lock
from conformal_demand_forecaster.monitoring.audit import AuditChain
from conformal_demand_forecaster.monitoring.coverage import MonitorStatus

__all__ = ["Proposal", "ProposalStatus", "RecalibrationWorkflow", "WorkflowError"]

ProposalStatus = Literal["pending", "approved", "rejected", "applied"]

_AUDITED_TRANSITIONS: dict[str, ProposalStatus] = {
    "recalibration_proposed": "pending",
    "recalibration_approved": "approved",
    "recalibration_rejected": "rejected",
    "recalibration_applied": "applied",
}


class WorkflowError(ValueError):
    """Invalid transition or unknown proposal."""


class Proposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: str
    day: int
    segment: str
    alpha: float
    n: int
    misses: int
    coverage: float
    p_value: float
    status: ProposalStatus = "pending"
    decided_by: str | None = None
    decided_at: str | None = None
    reason: str | None = None
    applied_version: str | None = None


class RecalibrationWorkflow:
    def __init__(self, path: Path, audit: AuditChain) -> None:
        self.path = path
        self.audit = audit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Proposal]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {str(key): Proposal.model_validate(value) for key, value in raw.items()}

    def _write(self, proposals: dict[str, Proposal]) -> None:
        payload = {key: value.model_dump(mode="json") for key, value in sorted(proposals.items())}
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def audited_state(self, proposal_id: str) -> tuple[ProposalStatus | None, str | None]:
        """The status and deciding actor the audit chain records for ``proposal_id``.

        Returns ``(None, None)`` when the chain never saw the proposal. A chain that does not
        verify is refused outright: without it there is nothing to check the file against.
        """
        try:
            entries = self.audit.entries()
        except (ValueError, TypeError, KeyError) as exc:
            raise WorkflowError(f"audit chain is unreadable: {exc}") from None
        status: ProposalStatus | None = None
        actor: str | None = None
        for entry in entries:
            action = _AUDITED_TRANSITIONS.get(entry.action)
            if action is None:
                continue
            target = entry.data.get("proposal_id", entry.data.get("id"))
            if target != proposal_id:
                continue
            status = action
            if action in ("approved", "rejected"):
                actor = entry.actor
        return status, actor

    def _require_agreement(self, proposal_id: str, current: Proposal) -> str | None:
        """Refuse a transition when the stored status is not the one the chain last recorded."""
        audited, actor = self.audited_state(proposal_id)
        if audited != current.status:
            raise WorkflowError(
                f"proposal state {current.status!r} does not match the audit chain "
                f"({audited or 'never proposed'})"
            )
        return actor

    def list(self, status: ProposalStatus | None = None) -> list[Proposal]:
        with file_lock(self.path):
            items = list(self._read().values())
        if status is not None:
            items = [item for item in items if item.status == status]
        return sorted(items, key=lambda item: (item.created_at, item.id))

    def get(self, proposal_id: str) -> Proposal:
        with file_lock(self.path):
            proposals = self._read()
        if proposal_id not in proposals:
            raise WorkflowError("unknown proposal")
        return proposals[proposal_id]

    def open(self, status: MonitorStatus, *, actor: str = "coverage-monitor") -> Proposal | None:
        """Open a proposal for an alarm unless one is already pending for the same key."""
        if not status.alarm:
            return None
        with file_lock(self.path):
            proposals = self._read()
            for item in proposals.values():
                if (
                    item.status == "pending"
                    and item.segment == status.segment
                    and item.alpha == status.alpha
                ):
                    return None
            proposal = Proposal(
                id=uuid.uuid4().hex,
                created_at=datetime.now(UTC).isoformat(),
                day=status.day,
                segment=status.segment,
                alpha=status.alpha,
                n=status.n,
                misses=status.misses,
                coverage=round(status.coverage, 6),
                p_value=status.p_value,
            )
            proposals[proposal.id] = proposal
            self._write(proposals)
        self.audit.append(actor, "recalibration_proposed", proposal.model_dump(mode="json"))
        return proposal

    def _decide(
        self, proposal_id: str, actor: str, status: ProposalStatus, reason: str | None
    ) -> Proposal:
        with file_lock(self.path):
            proposals = self._read()
            current = proposals.get(proposal_id)
            if current is None:
                raise WorkflowError("unknown proposal")
            self._require_agreement(proposal_id, current)
            if current.status != "pending":
                raise WorkflowError(f"proposal is already {current.status}")
            decided = current.model_copy(
                update={
                    "status": status,
                    "decided_by": actor,
                    "decided_at": datetime.now(UTC).isoformat(),
                    "reason": reason,
                }
            )
            proposals[proposal_id] = decided
            self._write(proposals)
        self.audit.append(
            actor, f"recalibration_{status}", {"proposal_id": proposal_id, "reason": reason}
        )
        return decided

    def approve(self, proposal_id: str, operator: str, reason: str | None = None) -> Proposal:
        return self._decide(proposal_id, operator, "approved", reason)

    def reject(self, proposal_id: str, operator: str, reason: str | None = None) -> Proposal:
        return self._decide(proposal_id, operator, "rejected", reason)

    def mark_applied(
        self, proposal_id: str, actor: str, *, from_version: str, to_version: str
    ) -> Proposal:
        with file_lock(self.path):
            proposals = self._read()
            current = proposals.get(proposal_id)
            if current is None:
                raise WorkflowError("unknown proposal")
            approved_by = self._require_agreement(proposal_id, current)
            if current.status != "approved":
                raise WorkflowError("only an approved proposal can be applied")
            applied = current.model_copy(
                update={"status": "applied", "applied_version": to_version}
            )
            proposals[proposal_id] = applied
            self._write(proposals)
        self.audit.append(
            actor,
            "recalibration_applied",
            {
                # The approver comes from the chain, not from the file the chain is checking.
                "proposal_id": proposal_id,
                "approved_by": approved_by,
                "from_version": from_version,
                "to_version": to_version,
            },
        )
        return applied
