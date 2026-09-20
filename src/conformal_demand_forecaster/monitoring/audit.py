"""Hash-chained audit log for calibration decisions.

Every entry is one JSON line ``{seq, at, actor, action, data, prev, hash}`` where ``hash`` is the
SHA-256 of the canonical JSON of the entry without ``hash`` and ``prev`` is the previous entry's
hash (the genesis entry links to 64 zeros). Editing, deleting, reordering or re-encoding any
entry breaks the chain from that point on, which :func:`verify_chain` reports. The log is
tamper-evident, not tamper-proof: whoever can rewrite the whole file can rebuild the chain, so the
head hash should be published elsewhere for a strong guarantee (a documented non-goal here).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conformal_demand_forecaster.core.canonical import canonical_json, sha256_hex
from conformal_demand_forecaster.core.fileio import file_lock
from conformal_demand_forecaster.core.redaction import redact_value

__all__ = ["AuditChain", "AuditEntry", "ChainVerification", "verify_chain"]

GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class AuditEntry:
    seq: int
    at: str
    actor: str
    action: str
    data: dict[str, Any]
    prev: str
    hash: str

    def body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.at,
            "actor": self.actor,
            "action": self.action,
            "data": self.data,
            "prev": self.prev,
        }


@dataclass(frozen=True, slots=True)
class ChainVerification:
    ok: bool
    entries: int
    head: str
    problem: str | None = None


def _entry_hash(body: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(body)))


def _parse(line: str, expected_seq: int, expected_prev: str) -> AuditEntry:
    raw = json.loads(line)
    if not isinstance(raw, dict) or set(raw) != {
        "seq",
        "at",
        "actor",
        "action",
        "data",
        "prev",
        "hash",
    }:
        raise ValueError(f"entry {expected_seq}: malformed record")
    entry = AuditEntry(
        seq=raw["seq"],
        at=raw["at"],
        actor=raw["actor"],
        action=raw["action"],
        data=raw["data"],
        prev=raw["prev"],
        hash=raw["hash"],
    )
    if not (isinstance(entry.seq, int) and isinstance(entry.data, dict)):
        raise TypeError(f"entry {expected_seq}: wrong field types")
    if entry.seq != expected_seq:
        raise ValueError(f"entry {expected_seq}: sequence number is {entry.seq}")
    if entry.prev != expected_prev:
        raise ValueError(f"entry {expected_seq}: previous hash does not match")
    if _entry_hash(entry.body()) != entry.hash or canonical_json(raw) != canonical_json(
        {**entry.body(), "hash": entry.hash}
    ):
        raise ValueError(f"entry {expected_seq}: hash does not match its content")
    return entry


def verify_chain(path: Path) -> ChainVerification:
    """Walk the file and check every link; the head is the last valid hash."""
    if not path.exists():
        return ChainVerification(ok=True, entries=0, head=GENESIS)
    prev, count = GENESIS, 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = _parse(line, count, prev)
            except (ValueError, KeyError, TypeError) as exc:
                return ChainVerification(ok=False, entries=count, head=prev, problem=str(exc))
            prev, count = entry.hash, count + 1
    return ChainVerification(ok=True, entries=count, head=prev)


class AuditChain:
    """Append-only writer that re-verifies the tail before every append."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def head(self) -> str:
        return verify_chain(self.path).head

    def append(self, actor: str, action: str, data: Mapping[str, Any]) -> AuditEntry:
        with file_lock(self.path):
            state = verify_chain(self.path)
            if not state.ok:
                raise ValueError(f"refusing to append to a broken audit chain: {state.problem}")
            body = {
                "seq": state.entries,
                "at": datetime.now(UTC).isoformat(),
                "actor": actor,
                "action": action,
                "data": redact_value(dict(data)),
                "prev": state.head,
            }
            entry = AuditEntry(**body, hash=_entry_hash(body))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json({**body, "hash": entry.hash}) + "\n")
                handle.flush()
            return entry

    def entries(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        result: list[AuditEntry] = []
        prev = GENESIS
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = _parse(line, len(result), prev)
                    result.append(entry)
                    prev = entry.hash
        return result
