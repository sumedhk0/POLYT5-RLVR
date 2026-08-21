# tests/test_rewards.py
from __future__ import annotations

import pytest

from polyt5.rewards import RewardResult, TgRewardConfig, build_arm, tg_reward, validity_gate


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


VALID = "[At][C][C][O][At]"
INVALID = "[Zz][Qq]"


class _FakeIndex:
    """Stands in for ScalableNoveltyIndex; 'known' PSELFIES are not novel."""

    def __init__(self, known):
        self._known = set(known)

    def is_novel(self, psmiles):
        return psmiles not in self._known


def test_accuracy_arm_uses_only_the_tg_term():
    arm = build_arm("accuracy")
    out = arm([VALID], [500.0], [(505.0, 5.0, 4)])
    assert 0.0 < out[0].value <= 1.0
    assert "closeness" in out[0].components


def test_every_arm_zeroes_an_invalid_candidate():
    for name in ("accuracy", "validity", "composite", "constraint"):
        arm = build_arm(name, novelty_index=_FakeIndex([]))
        out = arm([INVALID], [500.0], [(500.0, 1.0, 4)])
        assert out[0].value == 0.0, name
        assert out[0].gated is True, name


def test_validity_arm_is_binary():
    arm = build_arm("validity", novelty_index=_FakeIndex([]))
    out = arm([VALID], [500.0], [(999.0, 99.0, 4)])
    assert out[0].value == 1.0, "validity arm must ignore how wrong Tg is"


def test_constraint_arm_requires_every_condition():
    arm = build_arm("constraint", novelty_index=_FakeIndex([]),
                    tolerance=50.0, sa_max=6.0)
    on_target = arm([VALID], [500.0], [(510.0, 2.0, 4)])[0].value
    off_target = arm([VALID], [500.0], [(700.0, 2.0, 4)])[0].value
    assert on_target == 1.0
    assert off_target == 0.0, "conjunction: missing one condition scores zero"


def test_composite_arm_weights_are_configurable_not_hardcoded():
    idx = _FakeIndex([])
    a = build_arm("composite", novelty_index=idx, weights={"tg": 1.0, "pv": 0.0, "novelty": 0.0})
    b = build_arm("composite", novelty_index=idx, weights={"tg": 0.0, "pv": 1.0, "novelty": 0.0})
    va = a([VALID], [500.0], [(600.0, 2.0, 4)])[0].value
    vb = b([VALID], [500.0], [(600.0, 2.0, 4)])[0].value
    assert va != vb


def test_rewards_package_imports_without_torch():
    """Reward workers must run CPU-only, with no model in the process."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import polyt5.rewards; "
         "sys.exit(1 if 'torch' in sys.modules else 0)"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[:500]
