"""The LightGBM quantile model wrapper.

Two properties carry real weight downstream. Quantiles must not cross, because CQR turns the
(0.1, 0.9) pair into a band and a crossed pair would produce a negative-width interval. And
stockout-censored rows must be excluded from training: a censored sale is a *lower bound* on
demand, so training on it biases every quantile downwards and the conformal correction would
then have to absorb a bias it was never meant to.

Models serialize to LightGBM's text format, never pickle, and the reload checks the feature
names so a model trained on a different feature set cannot be silently served.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_demand_forecaster.models.gbm import GbmConfig, QuantileGbm, QuantilePrediction

pytestmark = pytest.mark.unit

FEATURES = ("a", "b", "horizon")
CONFIG = GbmConfig(
    quantiles=(0.1, 0.5, 0.9),
    horizon_buckets=((1, 7), (8, 14)),
    num_iterations=30,
    num_leaves=7,
    min_data_in_leaf=5,
    num_threads=2,
)


def make_training_set(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    horizon = rng.integers(1, 15, size=n).astype(np.int64)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = np.maximum(0.0, 10.0 + 3.0 * a - 2.0 * b + rng.normal(scale=2.0, size=n))
    frame = pd.DataFrame({"a": a, "b": b, "horizon": horizon.astype(float)})
    return frame, y, horizon


@pytest.fixture(scope="module")
def fitted() -> tuple[QuantileGbm, pd.DataFrame, np.ndarray]:
    frame, y, horizon = make_training_set()
    model = QuantileGbm(CONFIG, FEATURES).fit(frame, y, horizon)
    return model, frame, horizon


def test_predicted_quantiles_never_cross_and_stay_non_negative(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, frame, horizon = fitted
    prediction = model.predict(frame, horizon)
    assert prediction.quantiles.shape == (len(frame), len(CONFIG.quantiles))
    assert np.all(np.diff(prediction.quantiles, axis=1) >= 0.0)
    assert np.all(prediction.quantiles >= 0.0)
    assert np.all(prediction.point >= 0.0)


def test_the_quantile_models_are_ordered_on_average(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, frame, horizon = fitted
    prediction = model.predict(frame, horizon)
    means = prediction.quantiles.mean(axis=0)
    assert means[0] < means[1] < means[2]


def test_censored_rows_are_excluded_from_training() -> None:
    """Censoring the largest targets must not drag the upper quantile down as much as keeping
    them as if they were real demand would."""
    frame, y, horizon = make_training_set(seed=1)
    censored = (y > np.quantile(y, 0.8)).astype(np.float64)
    truncated = np.where(censored > 0.5, 1.0, y)  # what a naive pipeline would train on

    excluded = QuantileGbm(CONFIG, FEATURES).fit(frame, truncated, horizon, censored=censored)
    naive = QuantileGbm(CONFIG, FEATURES).fit(frame, truncated, horizon)
    upper_excluded = excluded.predict(frame, horizon).column(0.9).mean()
    upper_naive = naive.predict(frame, horizon).column(0.9).mean()
    assert upper_excluded > upper_naive


def test_nan_targets_are_dropped(fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray]) -> None:
    frame, y, horizon = make_training_set(seed=2)
    y_with_gaps = y.copy()
    y_with_gaps[::3] = np.nan
    model = QuantileGbm(CONFIG, FEATURES).fit(frame, y_with_gaps, horizon)
    assert model.fitted is True


def test_each_horizon_bucket_gets_its_own_models(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, _, _ = fitted
    texts = model.to_text()
    expected = {
        f"bucket{bucket}_{name}"
        for bucket in range(2)
        for name in ("point", "q0.100", "q0.500", "q0.900")
    }
    assert set(texts) == expected


def test_text_serialization_round_trips_predictions(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, frame, horizon = fitted
    restored = QuantileGbm.from_text(model.to_text(), CONFIG, FEATURES)
    original = model.predict(frame, horizon)
    copy = restored.predict(frame, horizon)
    assert np.allclose(original.point, copy.point)
    assert np.allclose(original.quantiles, copy.quantiles)


def test_the_serialized_form_is_plain_text_not_pickle(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, _, _ = fitted
    for text in model.to_text().values():
        assert text.startswith("tree")
        assert "\x80" not in text  # a pickle protocol marker would show up here


def test_reloading_with_a_different_feature_set_is_refused(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, _, _ = fitted
    with pytest.raises(ValueError, match="trained on different features"):
        QuantileGbm.from_text(model.to_text(), CONFIG, ("a", "b", "other"))


def test_reloading_with_missing_models_is_refused(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, _, _ = fitted
    texts = dict(model.to_text())
    texts.pop("bucket0_point")
    with pytest.raises(ValueError, match="do not match the configuration"):
        QuantileGbm.from_text(texts, CONFIG, FEATURES)


def test_predicting_with_mismatched_columns_is_refused(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, frame, horizon = fitted
    renamed = frame.rename(columns={"a": "z"})
    with pytest.raises(ValueError, match="feature columns do not match"):
        model.predict(renamed, horizon)


def test_an_unfitted_model_refuses_to_predict_or_serialize() -> None:
    model = QuantileGbm(CONFIG, FEATURES)
    assert model.fitted is False
    frame, _, horizon = make_training_set(n=10)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(frame, horizon)
    with pytest.raises(ValueError, match="not fitted"):
        model.to_text()


def test_horizons_outside_the_buckets_are_refused(
    fitted: tuple[QuantileGbm, pd.DataFrame, np.ndarray],
) -> None:
    model, frame, _ = fitted
    with pytest.raises(ValueError, match="outside the configured buckets"):
        model.predict(frame, np.full(len(frame), 99, dtype=np.int64))


def test_too_few_rows_for_a_bucket_is_an_explicit_error() -> None:
    frame, y, horizon = make_training_set(n=40)
    with pytest.raises(ValueError, match="not enough training rows"):
        QuantileGbm(GbmConfig(min_data_in_leaf=100), FEATURES).fit(frame, y, horizon)


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    frame, y, horizon = make_training_set(seed=3)
    first = QuantileGbm(CONFIG, FEATURES).fit(frame, y, horizon).predict(frame, horizon)
    second = QuantileGbm(CONFIG, FEATURES).fit(frame, y, horizon).predict(frame, horizon)
    assert np.array_equal(first.quantiles, second.quantiles)


# --- configuration --------------------------------------------------------------------------


def test_bucket_lookup_and_max_horizon() -> None:
    assert CONFIG.max_horizon == 14
    assert [CONFIG.bucket_of(h) for h in (1, 7, 8, 14)] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="outside the configured buckets"):
        CONFIG.bucket_of(15)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantiles": ()}, "strictly inside"),
        ({"quantiles": (0.0, 0.5)}, "strictly inside"),
        ({"quantiles": (0.9, 0.1)}, "strictly increasing"),
        ({"horizon_buckets": ()}, "at least one horizon bucket"),
        ({"horizon_buckets": ((2, 7),)}, "contiguous and start at 1"),
        ({"horizon_buckets": ((1, 7), (9, 14))}, "contiguous and start at 1"),
    ],
)
def test_invalid_configurations_are_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GbmConfig(**kwargs)  # type: ignore[arg-type]


def test_prediction_column_lookup() -> None:
    prediction = QuantilePrediction(
        point=np.array([1.0]), quantiles=np.array([[1.0, 2.0, 3.0]]), levels=(0.1, 0.5, 0.9)
    )
    assert prediction.column(0.9).tolist() == [3.0]
    with pytest.raises(ValueError, match="is not one of"):
        prediction.column(0.42)
