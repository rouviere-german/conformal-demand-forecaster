# 3. Adaptive Conformal Inference on top of split conformal, with gamma = 0.08

- Status: accepted
- Date: 2026-09-20

## Context

The CQR guarantee of ADR 0002 holds under exchangeability. Retail demand is not exchangeable
across a regime shift: a promotion calendar changes, a competitor closes, a category is
re-priced. When that happens the frozen calibration scores describe a world that no longer
exists, and the intervals silently stop covering.

`bench/bench_shift.py` measures exactly this. A model trained and frozen before a 2.2x level
shift keeps printing "80%" and "95%" intervals whose realized coverage over the next 200 days is
0.685 and 0.804 -- eleven and fifteen points short.

The options were:

1. **Retrain and recalibrate more often.** Helps, but the gap is between scheduled runs, which
   is precisely when nobody is watching. It also does not tell anyone that coverage broke.
2. **Weight recent calibration scores more heavily.** A reasonable heuristic, with no guarantee
   attached and another window-length parameter to tune.
3. **Adaptive Conformal Inference** (Gibbs and Candes, NeurIPS 2021): treat the miscoverage
   level itself as a control variable and update it online from realized errors,
   `alpha_{t+1} = alpha_t + gamma (alpha - err_t)`.

## Decision

Run ACI on top of the Mondrian split-conformal layer: one `AlphaState` per (segment, horizon
bucket, nominal level), updated once per day from that group's realized miss rate as actuals
arrive. Forecasts are served at the ACI-adjusted alpha; the frozen calibration scores are still
what the quantile is read from.

Ship `gamma = 0.08`, chosen from the measured trade-off rather than by feel
(`uv run python bench/bench_shift.py --sweep`, 25 series, 200 post-shift days):

| gamma | 80% coverage error | 95% coverage error | 80% width | 95% width |
|---|---|---|---|---|
| 0.03 | 0.0260 | 0.0115 | 29.40 | 50.21 |
| 0.05 | 0.0160 | 0.0084 | 29.86 | 51.21 |
| **0.08** | **0.0103** | **0.0070** | **30.34** | **51.55** |
| 0.12 | 0.0066 | 0.0056 | 30.74 | 51.69 |

Below 0.08 the recursion is too slow to recover inside a season; above it the extra accuracy is
small and the alpha state gets noticeably noisier day to day, which shows up as intervals that
visibly breathe. 0.08 sits where the error curve flattens, at about 3% more width than 0.03.

## Consequences

- The long-run coverage bound `|mean(err) - alpha| <= (max(alpha_1, 1-alpha_1) + gamma) / (gamma T)`
  is distribution-free and holds for adversarial sequences, which is why
  `tests/unit/test_aci.py` can assert it with hypothesis rather than simulate it.
- **The bound has a premise this implementation does not fully satisfy.** The proof needs an
  interval built at `alpha_t <= 0` to cover certainly. Here the calibrator saturates at the
  widest observed calibration score instead of returning an infinite interval, so once a group's
  calibration set is exhausted a miss is still possible and the realized error can exceed the
  bound. Those rows carry a `saturated` flag and increment
  `app_calibration_saturated_total`; `tests/unit/test_aci.py::test_the_bound_needs_its_premise_and_saturation_can_break_it`
  pins the behaviour down rather than hiding it.
- ACI restores coverage by *widening*, and the width is the cost: after the shift the adaptive
  bands are about 1.4x (80%) and 1.7x (95%) the frozen ones. Wider intervals mean larger
  newsvendor orders, so the decision layer pays for the calibration.
- ACI is a long-run guarantee, not a finite-sample one. It says nothing about coverage on any
  particular day, which is why the binomial monitor (ADR 0005) exists as a separate alarm rather
  than being folded into the same mechanism.
- The alpha state is mutable and lives outside the calibration version hash, in
  `alpha_state.json` bound to its calibration version. A state file from another calibration is
  refused on load.
