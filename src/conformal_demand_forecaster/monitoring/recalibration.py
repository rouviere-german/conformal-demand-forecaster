"""Approval-gated recalibration proposals.

A coverage alarm opens a proposal (one pending proposal per segment and level). Only an operator
can approve or reject it, and only an approved proposal can be applied, which promotes a new
calibration version. Every transition is appended to the hash-chained audit log. Proposals live
in a JSON file written atomically under a file lock.
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
                "proposal_id": proposal_id,
                "approved_by": current.decided_by,
                "from_version": from_version,
                "to_version": to_version,
            },
        )
        return applied
