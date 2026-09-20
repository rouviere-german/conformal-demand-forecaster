# Model card: conformal demand forecaster

Format based on Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Every number
below was produced by a command in this repository, named next to it.

## Model details

- **Name and version:** `demand-gbm`, registered content-addressed in `var/registry`; the
  registry digest is printed by `uv run conformal-demand-forecaster train` and recorded in
  `install.json`. The *calibration* carries its own version (the SHA-256 of its canonical
  record), and a forecast is only meaningful as the pair `(model_version, calibration_version)`.
- **Type:** a global LightGBM model per horizon bucket — one Tweedie point model plus one
  pinball-loss model at each of q = 0.05, 0.1, 0.5, 0.9, 0.95 — wrapped in a conformal
  prediction layer (conformalized quantile regression with Mondrian grouping and Adaptive
  Conformal Inference).
- **Owner:** Rouviere German
- **Date:** 2026-09-20
- **License:** MIT (code). Data is generated locally; no third-party dataset is bundled.
- **Artifact format:** LightGBM text (`model_to_string`), JSON, and npz written and read with
  `allow_pickle=False`. Never pickle. Artifacts are SHA-256 verified on load.
- **Run manifest:** `var/manifest.json` — generator seed, dataset SHA-256, training parameters,
  package versions and the `uv.lock` digest.

## Intended use

- **Primary use:** short-horizon (1–28 day) SKU-store demand forecasting where the *interval*,
  not the point forecast, is the product: replenishment planning, safety-stock sizing, and the
  newsvendor order quantity derived from the conformalized quantiles.
- **Intended users:** demand planners and the systems that order on their behalf, reading through
  the role-based API.
- **Out-of-scope uses:** anything that needs a per-series guarantee (see Caveats); financial or
  medical decisions; horizons beyond 28 days; series with less than roughly a year of history;
  new SKUs with no history at all (the model has no cold-start path).

## Training data

- **Source:** a seeded, synthetic M5-like generator
  (`src/conformal_demand_forecaster/data/generator.py`). It produces weekly and yearly
  seasonality, a mild trend, price elasticity with promotion episodes, intermittent and smooth
  demand processes, scheduled regime shifts and stockout censoring. Latent demand is retained
  alongside censored sales so evaluation can score against the truth.
- **Fingerprint:** the default configuration is 200 series x 1096 days = 219,200 rows, seed
  20260914. The exact CSV digest is recorded in the run manifest; the same seed reproduces the
  same bytes (`tests/unit/test_generator.py::test_the_same_seed_reproduces_the_same_bytes`).
- **Contract:** `SalesPanelSchema` (pandera). Non-negative units under a hard cap, positive
  prices, binary promo/stockout flags, contiguous dates within a series, sales never above latent
  demand, and sales equal to demand on uncensored days. Rows failing the contract are **rejected**
  — the whole frame is refused, with a summary that never copies the offending values.
- **Known gaps:** no cold-start series, no cannibalisation or cross-SKU substitution, no supply
  constraints beyond the stockout censoring, no calendar of real holidays, and no returns.
- **PII:** none by construction. Series are SKU-store identifiers; measures are units, prices and
  flags.

## Evaluation

- **Data:** rolling-origin backtest over the generated panel — 12 origins, 28-day stride, 28-day
  horizon — scored against latent demand. Coverage against censored sales is reported separately
  because a censored day makes sales a lower bound on demand.
- **Metrics and why:**
  - *MASE* (Hyndman & Koehler 2006): scale-free accuracy, comparable across series.
  - *Weighted scaled pinball loss* (M5 construction): quantile accuracy across all five levels,
    weighted by recent sales volume.
  - *Empirical coverage and mean width*: the promise on the tin, and what it costs.
  - *Winkler score* (Winkler 1972): a single number that trades width against misses.
  - *Worst-segment coverage gap*: what a pooled average hides.
  - *Realized newsvendor cost*: whether the uncertainty actually improves the decision.
- **Results** — `uv run python bench/bench_backtest.py --series 200`, 200 series, 12 origins,
  67,200 scored rows, Windows 10, 6 logical CPUs, Python 3.12.14:

  | Metric | Conformal GBM | Seasonal naive | ETS |
  |---|---|---|---|
  | MASE | 0.8099 | 1.0487 | 0.8891 |
  | Weighted scaled pinball | 0.1901 | 0.2722 | 0.2185 |
  | Coverage 80% | 0.7999 | 0.8055 | 0.7768 |
  | Coverage 95% | 0.9532 | 0.8882 | 0.8762 |

  The baselines' 80% bands happen to land near nominal here; their 95% bands are six to seven
  points short, which is the usual behaviour of empirical residual quantiles with no coverage
  guarantee attached. The run took 3m39s on the reference machine.

- **Slices:** worst-segment (smooth vs intermittent) coverage gap at 80%: **0.0082** with
  Mondrian calibration against **0.0249** pooled, on the same raw quantiles. In isolation
  (`bench/bench_conformal.py`, two groups whose noise differs by 6x) the gap falls from 0.078 to
  0.022.

- **Shift behaviour** — `uv run python bench/bench_shift.py`, a frozen model across a 2.2x level
  shift, 200 post-shift days:

  | Level | Method | Coverage before | Coverage after | Width after |
  |---|---|---|---|---|
  | 80% | frozen split conformal | 0.8411 | 0.6847 | 21.63 |
  | 80% | ACI | 0.8214 | 0.7897 | 30.34 |
  | 95% | frozen split conformal | 0.9409 | 0.8038 | 29.75 |
  | 95% | ACI | 0.9381 | 0.9430 | 51.55 |

- **Decisions:** ordering from the conformalized quantiles at holding cost 1 and stockout cost 4
  gives a realized cost of **11.77** against **14.43** for ordering the point forecast — about
  18% lower.

## Robustness, drift and monitoring

- **Robustness and adversarial tests:** `tests/unit/test_robustness.py` (missing days, a series
  collapsing to zero, 99%-discount promotions, injected spikes), `tests/unit/test_guard.py`
  (poisoning the guard's own history), `tests/unit/test_features.py` (no-lookahead under
  arbitrary perturbation of the future).
- **Input screening:** each ingested actual is compared with its series' recent history using
  `z = (y - median) / max(1.4826 * MAD, sqrt(max(median, p90) + 1))`, rejected above z = 10.
  Only the upper tail is screened, so a genuine collapse to zero still reaches the monitor.
- **Drift monitors:** PSI of recent accepted actuals against the per-segment training reference,
  exported as `app_feature_psi`. PSI above 0.2 is the conventional "investigate" line.
- **Performance monitor (the SLO):** empirical coverage within 2 points of nominal over a rolling
  28-day window, per segment and level. Implemented as a one-sided binomial test against a miss
  probability of `alpha + tolerance` at significance 0.01 with at least 30 observations.
  Measured operating characteristic (`bench/bench_monitor.py`, 500 simulations each): false-alarm
  rate **0.4%** of stationary runs, median detection delay **9 days**, p90 **12 days**, and no
  undetected breach.
- **Retraining trigger:** a coverage alarm opens a recalibration proposal. ACI absorbs the shift
  automatically in the meantime; promoting a new calibration requires an operator approval and is
  hash-chain audited (ADR 0005).

### Coverage-alert runbook

1. **Read the proposal.** `GET /v1/recalibration/proposals?status=pending` gives the segment,
   level, window size, miss count, realized coverage and p-value.
2. **Check whether the input is real.** `GET /v1/calibration` shows the live score counts and the
   effective alphas. Look at `app_actuals_quarantined_total` and the `actuals` table: a spike in
   quarantined rows alongside the alarm suggests a data fault, not a regime shift.
3. **Check whether the inputs drifted.** `app_feature_psi` above 0.2 for that segment points at a
   genuine change in the demand distribution.
4. **Check whether it is expected.** A holiday, a planned promotion or a range change is a
   temporary regime; recalibrating bakes it in permanently. Reject with a reason instead.
5. **Check saturation.** A rising `app_calibration_saturated_total` means ACI has pushed alpha to
   the floor and the calibration set is exhausted — recalibration (or retraining) is needed, not
   more adaptation.
6. **Decide.** Approve (`POST /v1/recalibration/proposals/{id}/approve`) to merge the live scores
   and promote a new calibration version, or reject with a reason. Both are audited with your key
   id; verify the chain afterwards with `conformal-demand-forecaster audit verify`.
7. **Watch.** Applying resets the monitor window and the live scores, so the new version is judged
   on fresh evidence. If the alarm returns within a window, the model — not the calibration — is
   probably the problem; retrain.

## Ethical considerations

The data is synthetic and contains no personal information, so the direct privacy risk is
negligible. The realistic harm is economic and indirect: intervals that are trusted more than
they deserve. A segment-marginal guarantee presented as a per-SKU promise would lead a planner to
under-stock specific items while the aggregate dashboard stays green. The Caveats below, the
`saturated` flag on every interval, and the separate coverage monitor exist to keep that gap
visible rather than papered over.

## Caveats and recommendations

- **The guarantee is marginal, per calibration group** — (segment x horizon bucket), not per
  series and not conditional on covariates. Aggregate coverage can be exactly right while a
  particular SKU is not.
- **ACI is a long-run guarantee, not a finite-sample one.** It says nothing about coverage on any
  given day, and its distribution-free bound assumes an interval built at `alpha_t <= 0` covers
  certainly — which this implementation violates when the calibration set saturates (ADR 0003).
- **Exchangeability within a calibration window** is still assumed by the underlying CQR step.
  ACI handles slow or stepwise shift, not adversarial reordering.
- **The newsvendor layer interpolates** the demand quantile function from five known points and
  clips outside them, so an extreme cost ratio saturates at the outermost conformalized bound
  rather than extrapolating.
- **Censoring is handled by exclusion, not by a censored-likelihood model.** Stockout days are
  dropped from training and calibration; a panel where most days are censored would leave too
  little to calibrate on.
- **Results are from synthetic data.** The generator was written to contain the pathologies this
  design targets, so the measured improvements are evidence that the mechanism works as intended,
  not a forecast of the numbers on any real retailer's panel.
- **MLflow's file backend is in upstream maintenance mode**; it is used here as a local run log
  only, and the run manifest, not MLflow, is the authoritative reproducibility record.
