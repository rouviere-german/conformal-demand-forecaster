# 4. Mondrian calibration by demand segment and horizon bucket

- Status: accepted
- Date: 2026-09-20

## Context

A single pooled conformal quantile gives one guarantee: coverage is right *on average over the
whole panel*. That average hides the two things a planner cares about.

Smooth fast movers and intermittent slow movers have nonconformity scores on completely
different scales. Pooling them produces a band calibrated for neither: in
`bench/bench_conformal.py`, with two groups whose noise differs by 6x, the pooled 80% band
covers the tight group 0.878 of the time and the wide group 0.760 -- 7.8 points of spread around
a nominal 0.80 that the pooled average reports as fine. The same effect appears on real
forecasts: the 200-series backtest shows a worst-segment gap of 8.5 points pooled.

Horizon has the same problem for a different reason: a one-day-ahead forecast is far more
certain than a 28-day-ahead one, so one width across all horizons is simultaneously too wide at
the short end and too narrow at the long end.

Mondrian conformal prediction (Vovk et al.) is the standard answer: calibrate each group with
its own scores, and the guarantee holds per group instead of only on average.

## Decision

Group calibration scores by **(demand segment, horizon bucket)**, where segment is
`smooth | intermittent` and the buckets are 1-7, 8-14 and 15-28 days. That is 6 groups per
nominal level. A group with fewer than `min_group_size` (50) scores falls back to the pooled
scores, and the fallback is reported on the result so an operator can see it.

Per-series calibration was rejected. With 28 target days per origin and a 56-day calibration
window, a single series contributes far too few scores for the rank `ceil((n+1)(1-alpha))` to be
meaningful at the 95% level -- `n` has to reach 19 before a 95% interval is anything other than
"the widest score seen", and below that the quantile saturates. Segments are the coarsest
grouping that still separates the populations that actually differ.

## Consequences

- Measured benefit, from `bench/bench_backtest.py` (20 series, 12 origins): the worst-segment
  coverage gap at 80% falls from 0.0854 pooled to 0.0052 Mondrian, at the cost of about 3% more
  average width. `bench/bench_conformal.py` reproduces the same effect in isolation.
- The guarantee is now **segment-marginal**, not per-series. An individual SKU inside a segment
  can still be systematically mis-covered. MODEL_CARD.md states this; closing it would need
  either per-series calibration (not enough data) or a conditional method such as localized
  conformal prediction (out of scope, recorded in Non-goals).
- Segments come from the data contract, so a third segment cannot appear without a schema
  change. That is deliberate: the segment map is part of the hashed calibration record, and
  silently re-segmenting would change every interval without changing the version.
- Sparse groups degrade gracefully rather than failing: the pooled fallback keeps serving, and
  `QuantileResult.pooled` records that it happened.
- ACI runs per group too (ADR 0003), so a shift that only hits intermittent series widens only
  those intervals.
