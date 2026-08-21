# tests/test_rewards.py
from __future__ import annotations

import pytest

from polyt5.rewards import RewardResult, TgRewardConfig, tg_reward, validity_gate


def test_valid_polymer_passes_the_gate():
    result = validity_gate("[At][C][C][O][At]")
    assert result.gated is False
    assert result.value == 1.0
    assert result.reason is None


def test_unparseable_string_is_gated_to_zero():
    result = validity_gate("[Zz][Qq][not-a-token]")
    assert result.gated is True
    assert result.value == 0.0
    assert result.reason is not None


def test_wrong_terminus_count_is_gated():
    # one [At] only - not a polymer repeat unit
    result = validity_gate("[At][C][C][O]")
    assert result.gated is True
    assert "termin" in result.reason.lower()


def test_double_bonded_terminus_fails_despite_correct_count():
    result = validity_gate("[At][=C][At]")  # decodes to [At]C=[At]
    assert result.gated is True
    assert result.value == 0.0
    assert result.reason == "terminus_valency"


def test_gate_never_raises_on_adversarial_input():
    for junk in ["", "   ", "((((", "[At]" * 500, "\x00", "[C][Ring9][C]"]:
        result = validity_gate(junk)
        assert isinstance(result, RewardResult)
        assert result.value == 0.0 or result.gated is False


def test_tg_reward_worked_example_from_the_spec():
    """The five candidates tabulated in the design doc."""
    cfg = TgRewardConfig(tolerance=100.0, sigma0=17.0)
    # (|err|, std) -> expected final value, from the spec's table
    cases = [(10, 5, 0.695), (10, 17, 0.450), (10, 45, 0.247),
             (80, 5, 0.155), (80, 45, 0.055)]
    for err, std, expected in cases:
        got = tg_reward(500.0 + err, std, 500.0, config=cfg).value
        assert got == pytest.approx(expected, abs=0.002), f"err={err} std={std}"


def test_confidence_weighting_penalises_disagreement_not_novelty():
    cfg = TgRewardConfig()
    agree = tg_reward(505.0, 2.0, 500.0, config=cfg).value
    disagree = tg_reward(505.0, 45.0, 500.0, config=cfg).value
    assert agree > disagree
    assert disagree > 0.0, "soft weighting must never zero out exploration"


def test_tg_reward_is_zero_beyond_tolerance():
    assert tg_reward(700.0, 1.0, 500.0, config=TgRewardConfig(tolerance=100.0)).value == 0.0


def test_tg_reward_records_unweighted_value_for_logging():
    r = tg_reward(510.0, 45.0, 500.0, config=TgRewardConfig())
    assert "closeness" in r.components and "confidence" in r.components
    assert r.components["closeness"] == pytest.approx(0.9, abs=1e-6)


def test_non_finite_prediction_is_gated():
    r = tg_reward(float("nan"), 5.0, 500.0, config=TgRewardConfig())
    assert r.gated is True and r.value == 0.0
