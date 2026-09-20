# Threat model

Scope: the forecasting service in this repository — the batch trainer, the DuckDB forecast store,
the role-based HTTP API, the actuals ingest path, the calibration artifacts and the
approval-gated recalibration workflow. Out of scope: the deployment platform, the network between
clients and the service, and the planners' own systems.

The system has one asset that is unusual for a forecasting service: **the calibration**. A model
that is quietly wrong produces bad point forecasts, which planners notice. A *calibration* that is
quietly wrong produces intervals that still look authoritative, and the whole point of the
repository is that those intervals are trusted enough to drive order quantities. Most of what
follows is about protecting the calibration and the evidence trail around it.

## Assets

| Asset | Why it matters |
|---|---|
| Calibration artifact (frozen scores, segment map, quantile pairs) | Determines every interval width; a silent change mis-sizes every order |
| ACI alpha state | Online control variable; poisoning it widens or narrows a whole segment |
| Registered LightGBM models | The point and quantile forecasts themselves |
| Sales panel and ingested actuals | Training and calibration input; also the coverage evidence |
| Audit chain | The only record of who approved which calibration |
| API keys (as SHA-256 digests) | Authentication for reader, ingestor and operator roles |

## Trust boundaries

1. **Network to API.** Untrusted callers; authenticated by API key, authorized by role.
2. **Ingestor to online state.** A valid ingestor key is semi-trusted: it can write actuals, but
   every value is contract-checked and screened before it can touch the ACI state or the monitor.
3. **Disk to process.** Artifacts are re-read on every load and verified against recorded digests.
4. **Operator to calibration.** The only path that can promote a calibration version, gated and
   audited.

## STRIDE

### Spoofing

| Threat | Control | Test |
|---|---|---|
| Caller impersonates a reader, ingestor or operator | API-key authn; keys stored as SHA-256 digests, compared in constant time against every configured digest with no early exit; fails closed when no keys are configured | `tests/unit/test_security.py`, `tests/integration/test_service_api.py::test_an_unknown_key_is_rejected_without_revealing_anything` |
| A key with one role acts in another | Each endpoint demands exactly one role; roles come from the settings list the digest was configured under | `tests/integration/test_service_api.py::test_a_reader_key_cannot_ingest_or_decide`, `tests/integration/test_service_api.py::test_an_ingestor_key_cannot_read_forecasts_or_decide` |
| Recalibration attributed to the wrong actor | Approver's key id is recorded on the proposal and in the audit entry that applies it | `tests/unit/test_recalibration.py::test_the_applied_entry_names_the_approver_and_both_versions` |

### Tampering

| Threat | Control | Test |
|---|---|---|
| Calibration scores edited on disk so intervals are silently wrong | `calibration.json` records the SHA-256 of `scores.npz` and is itself hashed into the calibration version; both are verified on load | `tests/unit/test_calibration_artifacts.py::test_tampering_with_the_scores_file_is_detected`, `tests/integration/test_runtime.py::test_a_tampered_calibration_file_refuses_to_load` |
| Calibration record edited without rehashing | The version digest is recomputed from the canonical record and compared | `tests/unit/test_calibration_artifacts.py::test_editing_the_record_without_rehashing_is_detected` |
| An alpha state file from another calibration is swapped in | `alpha_state.json` carries its calibration version and is refused if it does not match | `tests/unit/test_calibration_artifacts.py::test_an_alpha_state_from_another_calibration_is_refused` |
| Model artifact replaced | Models are registered content-addressed; the registry verifies digests on load | `tests/unit/test_ml_registry.py` |
| Poisoned actuals shift the calibration and the ACI state | Contract validation, then a robust median/MAD guard with a Poisson-like floor; quarantined values are stored for review and never reach ACI, the monitor or the panel | `tests/unit/test_guard.py`, `tests/integration/test_runtime.py::test_quarantined_values_never_enter_the_panel` |
| Poisoned history disarms the guard | The statistic is anchored on the median and the 90th percentile, so one spike cannot move it far enough to hide the next | `tests/unit/test_guard.py::test_one_poisoned_point_cannot_hide_the_next_spike`, `tests/unit/test_robustness.py::test_the_guard_rejects_a_sustained_poisoning_campaign` |
| SQL injection through series identifiers | Every statement is parameterised; identifiers are fixed; ids are pattern-validated at the contract and the route | `tests/unit/test_store.py::test_identifiers_that_look_like_sql_are_treated_as_data` |
| Code execution through a deserialized artifact | No pickle anywhere: LightGBM text models, JSON and npz with `allow_pickle=False` | `tests/unit/test_no_pickle.py`, `tests/unit/test_calibration_artifacts.py::test_no_pickle_is_used_by_the_artifact_files` |
| Future information leaks into features, invalidating calibration | Feature rows may read only days `<= t` plus the target day's known covariates; tests perturb the future and assert the rows are unchanged | `tests/unit/test_features.py::test_no_lookahead_for_any_origin_and_horizon` |

### Repudiation

| Threat | Control | Test |
|---|---|---|
| An operator denies approving a calibration | Every transition is appended to a hash-chained JSONL log with actor, action and payload | `tests/unit/test_recalibration.py::test_every_transition_lands_in_the_audit_chain` |
| History is rewritten to hide a bad promotion | Each entry hashes its predecessor; editing, deleting or reordering breaks the chain from that point, and the writer refuses to append to a broken chain | `tests/unit/test_recalibration.py::test_editing_an_audit_entry_breaks_the_chain`, `tests/integration/test_cli.py::test_audit_verify_reports_a_broken_chain` |

The log is tamper-**evident**, not tamper-proof: an attacker who can rewrite the whole file can
rebuild a consistent chain. Publishing the head hash elsewhere would close this; it is a non-goal
here and `conformal-demand-forecaster audit verify --expect-head` exists for that workflow.

### Information disclosure

| Threat | Control | Test |
|---|---|---|
| Errors leak internals, paths or submitted values | Fixed error codes and messages; exception text and tracebacks never reach responses | `tests/integration/test_app.py::test_unhandled_exceptions_become_generic_500s`, `tests/integration/test_app.py::test_validation_errors_never_echo_submitted_values` |
| A contract violation logs the data it rejected | `DataContractError` carries only (column, check, count) summaries | `tests/unit/test_contracts.py::test_failures_summarize_without_leaking_the_rejected_values` |
| Logs leak secrets or PII | Redaction processor over structlog and stdlib records, including tracebacks and audit payloads | `tests/unit/test_redaction.py`, `tests/unit/test_logging.py` |
| Metrics expose per-request identifiers | Metrics are labelled by route template and segment, never by series id or path; `/metrics` requires a key by default | `tests/integration/test_app.py::test_metrics_require_auth_and_label_route_templates` |
| Interactive docs expose the schema publicly | `enable_docs` defaults to false | `tests/integration/test_service_api.py::test_docs_are_disabled_by_default` |

The panel contains no PII by construction: series are SKU-store identifiers and the only measures
are units, prices and flags.

### Denial of service

| Threat | Control | Test |
|---|---|---|
| Request flood | Token-bucket rate limiting per client, with a bounded client table | `tests/integration/test_service_api.py::test_rate_limiting_applies_to_the_forecast_route`, `tests/unit/test_ratelimit.py` |
| Invalid keys mint fresh rate-limit buckets | The limiter verifies the key first, so invalid keys share the caller's address bucket | `tests/integration/test_app.py::test_invalid_keys_cannot_mint_fresh_rate_limit_buckets` |
| Oversized ingest batch exhausts memory | Body-size limit on declared and streamed bodies; `ingest_max_records` cap returning 413; per-record field bounds | `tests/integration/test_app.py::test_body_size_limit_applies_to_declared_and_streamed_bodies`, `tests/integration/test_service_api.py::test_an_empty_or_oversized_batch_is_rejected` |
| Unbounded horizon or cost parameters | Query bounds on horizon and costs, plus a settings-level horizon check | `tests/integration/test_service_api.py::test_out_of_range_horizons_are_rejected_by_validation` |
| Serving stale forecasts after a failed batch | `/readyz` fails when the latest origin is older than `freshness_max_days` | `tests/integration/test_service_api.py::test_readiness_fails_when_forecasts_go_stale` |

### Elevation of privilege

| Threat | Control | Test |
|---|---|---|
| An ingestor promotes a calibration | Only the `operator` role reaches the decision routes | `tests/integration/test_service_api.py::test_an_ingestor_key_cannot_read_forecasts_or_decide` |
| A proposal is applied without approval | The workflow state machine refuses to apply anything not in `approved` | `tests/unit/test_recalibration.py::test_an_unapproved_proposal_cannot_be_applied`, `tests/integration/test_runtime.py::test_a_regime_shift_eventually_opens_an_approval_gated_proposal` |
| A rejected proposal is resurrected | Only `pending` proposals can be decided | `tests/unit/test_recalibration.py::test_a_rejected_proposal_cannot_be_approved_afterwards` |
| Container escape widens the blast radius | Non-root final user, read-only root filesystem, dropped capabilities, `no-new-privileges` | `.github/workflows/ci.yml#docker`, `scripts/check_baseline.py` |

## Residual risks

- **Per-series miscoverage.** The guarantee is segment-marginal (ADR 0004). A single SKU can be
  systematically mis-covered while its segment looks healthy.
- **Slow poisoning.** The guard screens each value against its own series' recent history. An
  attacker who moves a series gradually, staying inside the threshold, can drift the calibration.
  The drift monitor (PSI against the training reference) is the second line of defence, and the
  coverage monitor is the third.
- **Saturation.** When ACI drives alpha to its floor the calibrator saturates at the widest
  observed score, so the long-run bound's premise fails; the rows are flagged, not hidden
  (ADR 0003).
- **Local file locks.** The audit chain, proposal store and artifacts coordinate with local file
  locks, which assume a local filesystem rather than a network share or multiple hosts.
- **No key rotation or identity system.** One static API-key scheme per role.
