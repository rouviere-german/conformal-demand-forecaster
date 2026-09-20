# 2. Conformalized quantile regression as the interval layer

- Status: accepted
- Date: 2026-09-20

## Context

Planners need intervals they can size inventory with, which means intervals whose *empirical*
coverage is close to the level printed on them. Three families were available:

1. **Raw quantile regression.** Train LightGBM at q = 0.05 and 0.95 and call the pair a 90%
   interval. Cheap, but the coverage is whatever the optimizer happened to achieve on the
   training distribution. In the backtest the ungapped ETS residual bands, built the same way,
   land at 0.785 and 0.873 against nominal 0.80 and 0.95 -- off by up to seven points.
2. **Distributional / parametric intervals.** Fit a negative binomial or Tweedie and read
   quantiles off the fitted law. This adds a distributional assumption that intermittent retail
   demand violates badly (mass at zero, promotion spikes, censoring).
3. **Conformal prediction.** Wrap any model with a calibration step that has a finite-sample,
   distribution-free coverage guarantee under exchangeability.

Within conformal prediction the choice is between plain split conformal on absolute residuals
(one symmetric width for everything) and conformalized quantile regression (CQR; Romano,
Patterson and Candes, NeurIPS 2019), which conformalizes an existing quantile pair.

## Decision

Use CQR on a LightGBM quantile pair, with the nonconformity score
`s = max(lower - y, y - upper)` and the conformal quantile at rank `ceil((n+1)(1-alpha))`.

Rationale:

- The guarantee is finite-sample and distribution-free: coverage lies in
  `[1-alpha, 1-alpha + 1/(n+1)]` for exchangeable scores. `bench/bench_conformal.py` measures it
  over 500 independent splits and `tests/unit/test_cqr.py` proves the rank arithmetic.
- Unlike absolute-residual split conformal, CQR keeps the *shape* the quantile model learned:
  the band is asymmetric where demand is skewed and narrow where the model is confident. It only
  shifts both edges by a constant, so a heteroscedastic panel does not collapse to one width.
- It composes with everything else here: the same score feeds Mondrian grouping (ADR 0004) and
  the ACI alpha update (ADR 0003), and the calibration state is a small array of scores that can
  be hashed, versioned and shipped (`conformal/artifacts.py`).

Lower bounds are clipped at zero, because negative demand is not a thing; that clipping can only
increase coverage, never reduce it.

## Consequences

- Calibration costs a held-out window (56 days by default) that cannot be trained on. With a
  three-year panel that is acceptable; on very short series it is a real cost.
- The guarantee is **marginal**, over the exchangeable population inside a calibration group --
  not per series and not conditional on the covariates. A specific slow-moving SKU can be badly
  covered while the group is fine. Mondrian grouping narrows this (ADR 0004) but does not remove
  it, and MODEL_CARD.md states it as a limitation.
- Exchangeability is the assumption that actually breaks in production, because demand regimes
  shift. That is the whole motivation for ADR 0003.
- Censored (stockout) days are excluded from calibration: a censored sale is a lower bound on
  demand, so scoring it would bias the conformal quantile downwards. This is enforced in
  `training.train_at` and tested in `tests/unit/test_training.py`.
