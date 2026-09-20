# 5. Recalibration is proposed automatically and promoted only by an operator

- Status: accepted
- Date: 2026-09-20

## Context

The coverage monitor fires when a segment's realized coverage breaches the SLO. The obvious next
step is to recalibrate automatically: merge the live scores into the calibration set, promote a
new version, carry on. It is also the step that turns a data problem into an outage.

A breach has several possible causes, and they call for opposite responses:

- a genuine regime shift -- recalibrating is right;
- a known seasonal event (a holiday, a planned promotion) -- recalibrating bakes a temporary
  regime into the permanent calibration;
- an upstream data fault, or poisoned actuals that slipped past the outlier guard --
  recalibrating on them is exactly how an attacker would widen or narrow every interval for a
  segment, and the audit trail would show the system doing it to itself.

Automatic promotion also removes the moment where somebody looks at the numbers. ACI already
provides the fast, bounded, reversible response to a shift (ADR 0003); recalibration is the slow,
permanent one, so it does not need to be automatic to keep the service calibrated.

## Decision

Split the response in two.

**Automatic, bounded:** ACI adjusts the effective alpha every day. This absorbs a shift within
days, is self-correcting if the shift reverses, and changes no stored artifact.

**Gated, permanent:** an alarm opens a `Proposal` in state `pending`, carrying the day, segment,
level, observation count, miss count, realized coverage and p-value. Only a key holding the
`operator` role can approve or reject it, and only an approved proposal can be applied. Applying
merges the live scores into the frozen ones and promotes a new calibration version. Every
transition -- proposed, approved, rejected, applied -- is appended to the hash-chained audit log
with the actor, and the applied entry names both the approver and the from/to versions.

At most one pending proposal exists per (segment, level), so a sustained breach produces one item
to decide rather than a daily stream.

## Consequences

- No single API key can change the calibration. The ingestor key that feeds actuals cannot
  approve, and the operator key that approves cannot ingest; `tests/integration/test_service_api.py`
  asserts both directions.
- An unapproved or already-decided proposal cannot be applied, which is asserted directly in
  `tests/unit/test_recalibration.py` and again through the runtime in
  `tests/integration/test_runtime.py`.
- Recalibration waits for a human, so a real shift stays on ACI-widened intervals until someone
  acts. That is the intended trade: wider intervals are a cost, a wrongly promoted calibration is
  a silent error. The runbook in MODEL_CARD.md gives the operator the checklist.
- The audit log is tamper-**evident**, not tamper-proof: whoever can rewrite the whole file can
  rebuild the chain. Publishing the head hash elsewhere would close that gap and is listed as a
  non-goal.
- Applying a proposal resets the monitor window and the live score buffer, so the new version is
  judged on fresh evidence rather than on the breach that triggered it.
