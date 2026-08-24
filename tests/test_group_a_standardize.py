"""Group A Task 2: train-only standardisation with logged, never-imputed drops."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from polyt5.data.standardize import Standardizer, fit_target_standardizer

TRAIN = np.array(
    [
        [1.0, 5.0, 7.0, np.nan],
        [2.0, 5.0, 9.0, 1.0],
        [3.0, 5.0, 11.0, 2.0],
        [4.0, 5.0, 13.0, 3.0],
    ]
)
COLUMNS = ["varying", "constant", "spread", "has_nan"]


def test_fit_keeps_only_the_informative_columns():
    std = Standardizer.fit(TRAIN, COLUMNS)
    assert std.columns == ("varying", "spread")
    assert std.dropped == ("constant", "has_nan")
    assert std.keep_index == (0, 2)
    assert std.n_features == 2


def test_dropped_columns_are_logged_by_name(caplog):
    """'the drop is logged, never silently imputed' -- the names must appear."""
    with caplog.at_level(logging.INFO, logger="polyt5.data.standardize"):
        Standardizer.fit(TRAIN, COLUMNS)
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "constant" in message
    assert "has_nan" in message


def test_transform_uses_train_statistics_on_unseen_rows():
    std = Standardizer.fit(TRAIN, COLUMNS)
    test = np.array([[2.5, 5.0, 10.0, 99.0]])
    out = std.transform(test)
    assert out.shape == (1, 2)
    # train mean of "varying" is 2.5, so it standardises to exactly 0.
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(0.0)


def test_inverse_transform_round_trips():
    std = Standardizer.fit(TRAIN, COLUMNS)
    forward = std.transform(TRAIN)
    back = std.inverse_transform(forward)
    assert np.allclose(back, TRAIN[:, [0, 2]])


def test_a_nan_column_is_dropped_not_imputed():
    std = Standardizer.fit(TRAIN, COLUMNS)
    assert "has_nan" not in std.columns
    assert not np.isnan(std.transform(TRAIN)).any()


def test_fit_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="columns"):
        Standardizer.fit(TRAIN, ["a", "b"])
    with pytest.raises(ValueError, match="2-D"):
        Standardizer.fit(np.array([1.0, 2.0]), ["a"])


def test_round_trips_through_dict():
    std = Standardizer.fit(TRAIN, COLUMNS)
    assert Standardizer.from_dict(std.to_dict()) == std


def test_target_standardizer_scales_a_scalar_property():
    values = [200.0, 300.0, 400.0, 500.0]
    std = fit_target_standardizer(values)
    assert std.columns == ("Tg",)
    scaled = std.transform(np.asarray(values, dtype=float)[:, None])
    assert scaled.mean() == pytest.approx(0.0)
    assert scaled.std() == pytest.approx(1.0)
    kelvin = std.inverse_transform(scaled)
    assert np.allclose(kelvin.ravel(), values)
    assert std.mean[0] == pytest.approx(350.0)


def test_target_standardizer_refuses_a_constant_property():
    """A zero-variance target would divide by zero at inference."""
    with pytest.raises(ValueError, match="no usable"):
        fit_target_standardizer([310.0, 310.0, 310.0])
