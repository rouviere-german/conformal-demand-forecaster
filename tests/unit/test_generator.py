"""The seeded generator: reproducibility, the pathologies it injects, and its self-consistency.

Reproducibility is a security property here as much as an ML one: the run manifest records a
seed and a dataset digest, and a promoted calibration is only auditable if the same seed really
does rebuild the same bytes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_demand_forecaster.core.canonical import sha256_hex
from conformal_demand_forecaster.data.generator import (
    GeneratorConfig,
    RegimeShift,
    generate_panel,
)
from conformal_demand_forecaster.ml.contracts import MAX_UNITS, SEGMENTS, validate_panel

pytestmark = pytest.mark.unit

SMALL = GeneratorConfig(n_series=6, n_days=200, seed=42, shifts=())


def digest(frame: pd.DataFrame) -> str:
    return sha256_hex(frame.to_csv(index=False, lineterminator="\n"))


def test_the_same_seed_reproduces_the_same_bytes() -> None:
    assert digest(generate_panel(SMALL)) == digest(generate_panel(SMALL))


def test_a_different_seed_produces_different_data() -> None:
    other = GeneratorConfig(n_series=6, n_days=200, seed=43, shifts=())
    assert digest(generate_panel(SMALL)) != digest(generate_panel(other))


def test_the_panel_satisfies_its_own_contract() -> None:
    frame = generate_panel(SMALL)
    validate_panel(frame)
    assert len(frame) == SMALL.n_series * SMALL.n_days
    assert frame["series_id"].nunique() == SMALL.n_series
    assert set(frame["segment"].unique()) <= set(SEGMENTS)


def test_series_are_shaped_like_a_balanced_panel() -> None:
    frame = generate_panel(SMALL)
    per_series = frame.groupby("series_id")["date"].agg(["count", "min", "max"])
    assert set(per_series["count"]) == {SMALL.n_days}
    assert per_series["min"].nunique() == 1
    assert per_series["max"].nunique() == 1


def test_censoring_only_happens_on_stockout_days() -> None:
    frame = generate_panel(GeneratorConfig(n_series=20, n_days=400, seed=5, shifts=()))
    censored = frame[frame["stockout"] == 1]
    assert len(censored) > 0, "the generator must produce some censored days"
    assert (censored["sales"] < censored["demand"]).all()
    clean = frame[frame["stockout"] == 0]
    assert (clean["sales"] == clean["demand"]).all()


def test_both_segments_appear_and_intermittent_series_have_more_zero_days() -> None:
    frame = generate_panel(GeneratorConfig(n_series=40, n_days=300, seed=11, shifts=()))
    zero_share = frame.assign(zero=frame["sales"] == 0).groupby("segment")["zero"].mean()
    assert set(zero_share.index) == set(SEGMENTS)
    assert zero_share["intermittent"] > zero_share["smooth"]


def test_promotions_lower_the_price_and_lift_demand() -> None:
    frame = generate_panel(GeneratorConfig(n_series=30, n_days=400, seed=3, shifts=()))
    promo_days = frame[frame["promo"] == 1]
    assert len(promo_days) > 0
    base = frame.groupby("series_id")["price"].max()
    assert (promo_days["price"] <= promo_days["series_id"].map(base)).all()
    means = frame.groupby("promo")["demand"].mean()
    assert means[1] > means[0]


def test_a_regime_shift_moves_the_level_on_the_scheduled_day() -> None:
    day, multiplier = 200, 2.5
    config = GeneratorConfig(
        n_series=40,
        n_days=300,
        seed=17,
        shifts=(RegimeShift(day=day, fraction=1.0, multiplier=multiplier),),
    )
    frame = generate_panel(config)
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    before = frame[(frame["date"] >= dates[day - 40]) & (frame["date"] < dates[day])]
    after = frame[(frame["date"] >= dates[day]) & (frame["date"] < dates[day + 40])]
    ratio = after["demand"].mean() / before["demand"].mean()
    assert 1.5 < ratio < 4.0


def test_a_shift_that_hits_nobody_leaves_the_data_alone() -> None:
    base = GeneratorConfig(n_series=6, n_days=200, seed=42, shifts=())
    untouched = GeneratorConfig(
        n_series=6,
        n_days=200,
        seed=42,
        shifts=(RegimeShift(day=100, fraction=0.0, multiplier=9.0),),
    )
    # The shift mask still consumes draws, so only the *shape* is compared.
    frame = generate_panel(untouched)
    assert len(frame) == len(generate_panel(base))
    assert frame["demand"].max() <= MAX_UNITS


def test_demand_is_never_negative_or_above_the_contract_cap() -> None:
    frame = generate_panel(GeneratorConfig(n_series=10, n_days=200, seed=8))
    assert frame["demand"].min() >= 0
    assert frame["demand"].max() <= MAX_UNITS
    assert np.isfinite(frame["price"]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_series": 0}, "at least 1 series"),
        ({"n_days": 10}, "at least 1 series"),
        ({"intermittent_fraction": 1.5}, "intermittent_fraction"),
        ({"promo_start_probability": -0.1}, "promo_start_probability"),
        ({"stockout_start_probability": 2.0}, "stockout_start_probability"),
    ],
)
def test_invalid_configurations_are_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GeneratorConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"day": -1}, "shift day"),
        ({"fraction": 1.2}, "shift fraction"),
        ({"multiplier": 0.0}, "shift multiplier"),
    ],
)
def test_invalid_shifts_are_refused(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"day": 10, "fraction": 0.5, "multiplier": 1.5}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RegimeShift(**values)  # type: ignore[arg-type]


def test_a_shift_scheduled_past_the_horizon_is_ignored() -> None:
    config = GeneratorConfig(
        n_series=4, n_days=100, seed=2, shifts=(RegimeShift(day=500, fraction=1.0, multiplier=5.0),)
    )
    frame = generate_panel(config)
    assert len(frame) == 400
