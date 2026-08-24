"""Group A Task 3: reliability weighting, and the floor that forbids infinity."""

from __future__ import annotations

import pytest

from polyt5.data.tg_metadata import TgExample, TgRow
from polyt5.data.weighting import (
    DEFAULT_STD_FLOOR,
    RED_RELIABILITY,
    drop_red_reliability,
    reliability_weights,
)


def make(std: float, reliability: str = "black") -> TgExample:
    return TgExample(
        pselfies="[At][C][C][At]",
        row=TgRow(
            psmiles="[At]CC[At]",
            tg=300.0,
            std=std,
            num_of_points=1 if std == 0.0 else 3,
            reliability=reliability,
            polymer_class="Polyolefins",
            descriptors=(1.0, 2.0),
        ),
    )


def test_red_rows_are_dropped_and_returned_for_the_log():
    examples = [make(0.0), make(4.0, RED_RELIABILITY), make(9.0)]
    kept, dropped = drop_red_reliability(examples)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0].row.reliability == RED_RELIABILITY
    assert all(e.row.reliability != RED_RELIABILITY for e in kept)


def test_gold_and_yellow_are_kept():
    kept, dropped = drop_red_reliability([make(1.0, "gold"), make(1.0, "yellow")])
    assert len(kept) == 2
    assert dropped == []


def test_a_single_measurement_polymer_does_not_get_infinite_weight():
    """std == 0 for 7088 of 7367 rows; without the floor every one is infinite."""
    weights = reliability_weights([make(0.0)], normalize=False)
    assert weights == [pytest.approx(1.0 / DEFAULT_STD_FLOOR)]


def test_a_noisy_label_is_downweighted_relative_to_a_precise_one():
    weights = reliability_weights([make(0.0), make(145.0629562)], normalize=False)
    assert weights[0] > weights[1]
    assert weights[1] == pytest.approx(1.0 / 145.0629562)


def test_std_below_the_floor_is_clamped_to_the_floor():
    weights = reliability_weights([make(1.0), make(0.0)], normalize=False)
    assert weights[0] == pytest.approx(weights[1])


def test_normalisation_keeps_the_mean_weight_at_one():
    """A4 must not smuggle in an effective learning-rate change."""
    weights = reliability_weights([make(0.0), make(20.0), make(145.0)])
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert all(w > 0.0 for w in weights)


def test_a_nonpositive_floor_is_refused():
    with pytest.raises(ValueError, match="floor"):
        reliability_weights([make(0.0)], floor=0.0)
    with pytest.raises(ValueError, match="floor"):
        reliability_weights([make(0.0)], floor=-1.0)


def test_empty_input_is_empty_output_not_a_division_by_zero():
    assert reliability_weights([]) == []
    assert drop_red_reliability([]) == ([], [])
