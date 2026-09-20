"""Conformal demand forecaster.

Retail demand forecasts whose prediction intervals keep their promised coverage through regime
shifts: conformalized quantile regression (CQR) with Mondrian (segment-conditional) calibration,
Adaptive Conformal Inference for online alpha updates, a binomial coverage monitor with
approval-gated recalibration, and a newsvendor layer that turns intervals into order quantities.

Sub-packages:
- ``core``: settings, logging with PII redaction, tracing, metrics, hardened FastAPI app factory.
- ``ml``: data contracts, run manifests, content-addressed registry, drift monitors and
  pickle-free serialization.
- ``data``, ``features``, ``models``: seeded M5-like generator, no-lookahead feature builder,
  seasonal-naive/ETS baselines and LightGBM quantile models.
- ``conformal``: CQR scores, Mondrian calibration, ACI alpha state, calibration artifacts.
- ``evaluation``, ``decisions``, ``monitoring``, ``serving``: backtest, newsvendor, coverage
  monitor with recalibration workflow, DuckDB store and role-based API.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
