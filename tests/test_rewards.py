# tests/test_rewards.py
from __future__ import annotations

from polyt5.rewards import RewardResult, validity_gate


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


def test_gate_never_raises_on_adversarial_input():
    for junk in ["", "   ", "((((", "[At]" * 500, "\x00", "[C][Ring9][C]"]:
        result = validity_gate(junk)
        assert isinstance(result, RewardResult)
        assert result.value == 0.0 or result.gated is False
