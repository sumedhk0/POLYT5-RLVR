# tests/test_group_a_arms.py
"""Group A Task 8: the seven-arm table, and the invariants it must satisfy."""

from __future__ import annotations

import pytest

from polyt5.training.group_a import ARM_IDS, SWITCH_NAMES, GroupAConfig, arm_config


def test_there_are_exactly_seven_arms_in_spec_order():
    assert ARM_IDS == ("B0", "A1", "A2", "A3", "A4", "A5", "A6")


def test_b0_turns_nothing_on():
    """B0 is the baseline: current text head, single task."""
    assert arm_config("B0").switches() == dict.fromkeys(SWITCH_NAMES, False)


def test_each_single_change_arm_turns_on_exactly_one_switch():
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        on = [name for name, value in arm_config(arm).switches().items() if value]
        assert len(on) == 1, f"{arm} flips {on}; individual ablations must be individual"


def test_the_five_single_change_arms_cover_all_five_switches():
    covered = {
        name
        for arm in ("A1", "A2", "A3", "A4", "A5")
        for name, value in arm_config(arm).switches().items()
        if value
    }
    assert covered == set(SWITCH_NAMES)


def test_a6_is_exactly_the_union_of_a1_through_a5():
    """'all five combined' is a property, not a convention someone maintains."""
    union = {name: False for name in SWITCH_NAMES}
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        for name, value in arm_config(arm).switches().items():
            union[name] = union[name] or value
    assert arm_config("A6").switches() == union
    assert all(union.values())


def test_no_arm_enables_cycle_consistency():
    """Spec 4.5: behind a flag, default OFF, never a primary objective."""
    assert all(not arm_config(arm).cycle_consistency for arm in ARM_IDS)


def test_overrides_reach_the_hyperparameters_but_not_the_switches():
    config = arm_config("A6", descriptor_lambda=0.5, n_writings=8, std_floor=10.0)
    assert config.descriptor_lambda == 0.5
    assert config.n_writings == 8
    assert config.std_floor == 10.0
    assert config.switches() == arm_config("A6").switches()


def test_overriding_a_switch_is_refused():
    with pytest.raises(ValueError, match="switch"):
        arm_config("A1", regression_head=False)


def test_unknown_arm_is_refused_with_the_valid_list():
    with pytest.raises(ValueError, match="B0"):
        arm_config("A9")


def test_effective_n_writings_is_one_unless_augmentation_is_on():
    assert arm_config("A1", n_writings=6).effective_n_writings() == 1
    assert arm_config("A3", n_writings=6).effective_n_writings() == 6


def test_degenerate_hyperparameters_are_refused():
    with pytest.raises(ValueError, match="descriptor_lambda"):
        arm_config("A2", descriptor_lambda=-1.0)
    with pytest.raises(ValueError, match="n_writings"):
        arm_config("A3", n_writings=0)
    with pytest.raises(ValueError, match="std_floor"):
        arm_config("A4", std_floor=0.0)


def test_cycle_consistency_requires_a_regression_head_to_score_with():
    with pytest.raises(ValueError, match="regression_head"):
        GroupAConfig(arm="X", cycle_consistency=True, regression_head=False)


def test_config_round_trips_to_a_dict_for_the_run_manifest():
    payload = arm_config("A6").to_dict()
    assert payload["arm"] == "A6"
    assert payload["regression_head"] is True
    assert payload["cycle_consistency"] is False
