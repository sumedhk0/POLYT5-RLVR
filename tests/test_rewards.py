# tests/test_rewards.py
from __future__ import annotations

import pytest

from polyt5.chemistry.canonicalization import canonical_psmiles
from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.chemistry.metrics import synthetic_accessibility
from polyt5.rewards import (
    RewardResult,
    TgRewardConfig,
    build_arm,
    sa_reward,
    tg_reward,
    validity_gate,
)


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

# The canonical PSMILES an arm actually looks up for VALID - computed the same
# way _BaseArm._prepare / ValidityArm do, so a fake novelty index seeded with
# this string is seeded with what the code under test really queries with,
# not with the raw PSELFIES.
VALID_CANONICAL = canonical_psmiles(pselfies_to_psmiles(VALID))


class _FakeIndex:
    """Stands in for ScalableNoveltyIndex; 'known' PSELFIES are not novel."""

    def __init__(self, known):
        self._known = set(known)

    def is_novel(self, psmiles):
        return psmiles not in self._known


def test_sa_reward_normalises_and_fails_closed():
    # SA=1 (easiest possible) -> full credit.
    assert sa_reward(1.0, sa_max=6.0).value == pytest.approx(1.0)
    # SA at the threshold -> zero credit, not partial credit.
    assert sa_reward(6.0, sa_max=6.0).value == pytest.approx(0.0)
    # SA past the threshold -> clamped to zero, not negative.
    assert sa_reward(8.0, sa_max=6.0).value == 0.0
    # No scorer available -> 0.0, NOT a pass: a missing capability must never
    # inflate a reward.
    r = sa_reward(None, sa_max=6.0)
    assert r.value == 0.0
    assert r.components == {"sa": 0.0}
    # Hand-computed midpoint: (sa_max - score) / (sa_max - 1) = (6 - 3.5) / 5 = 0.5.
    mid = sa_reward(3.5, sa_max=6.0)
    assert mid.value == pytest.approx(0.5)
    assert mid.components["sa_score"] == 3.5


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


def test_validity_arm_tsd_stage_rejects_training_set_members():
    """A training-set member is a well-formed, non-duplicate polymer that
    must still score zero - TSD is a real stage of the cascade, not a
    diagnostic-only component. It is a reward miss, not structural
    invalidity, so gated must stay False - Task 7's gated_fraction diagnostic
    would otherwise misreport a healthy, chemically-valid batch as broken."""
    arm = build_arm("validity", novelty_index=_FakeIndex([VALID_CANONICAL]))
    out = arm([VALID], [500.0], [(500.0, 1.0, 4)])[0]
    assert out.value == 0.0
    assert out.gated is False, "TSD failure is valid chemistry, not a structural gate"
    assert out.reason is None
    assert out.components["sv"] == 1.0
    assert out.components["tsd"] == 0.0
    assert out.components["dd"] == 0.0, "DD is never reached once TSD has failed"
    assert out.components["pv"] == 0.0, "PV is never reached once TSD has failed"


def test_validity_arm_dd_stage_rejects_within_batch_duplicates():
    """The first occurrence of a candidate passes; a later occurrence of the
    same polymer within the same batch must fail DD even though it is
    individually valid and novel. Like TSD, this is a reward miss on valid
    chemistry, not structural invalidity, so gated must stay False."""
    arm = build_arm("validity", novelty_index=_FakeIndex([]))
    out = arm([VALID, VALID], [500.0, 500.0], [(500.0, 1.0, 4), (500.0, 1.0, 4)])
    assert out[0].value == 1.0, "first occurrence clears the full cascade"
    assert out[1].value == 0.0, "second occurrence of the same polymer fails DD"
    assert out[1].gated is False, "DD failure is valid chemistry, not a structural gate"
    assert out[1].reason is None
    assert out[1].components["sv"] == 1.0
    assert out[1].components["tsd"] == 1.0, "still absent from the reference index"
    assert out[1].components["dd"] == 0.0
    assert out[1].components["pv"] == 0.0, "PV is never reached once DD has failed"


def test_validity_arm_sv_stage_gates_invalid_structures():
    """An unparseable structure fails at SV, before TSD/DD/PV ever run. This
    IS structural invalidity, so gated stays True - the same meaning
    validity_gate and every other arm give it."""
    arm = build_arm("validity", novelty_index=_FakeIndex([]))
    out = arm([INVALID], [500.0], [(500.0, 1.0, 4)])[0]
    assert out.value == 0.0
    assert out.gated is True
    assert out.components["sv"] == 0.0
    assert out.components["tsd"] == 0.0
    assert out.components["dd"] == 0.0
    assert out.components["pv"] == 0.0


def test_validity_arm_requires_a_novelty_index_by_default():
    """Without an index, TSD would fail closed for every candidate, silently
    collapsing every GRPO group to zero reward variance and zero gradient
    with no error raised. Fail loudly at construction instead."""
    with pytest.raises(ValueError, match="novelty_index"):
        build_arm("validity")


def test_validity_arm_opt_out_still_enforces_sv_dd_pv():
    """A caller who explicitly opts out of requiring a novelty index gets a
    working arm: TSD becomes a no-op pass, but SV, DD, and PV are still
    enforced - the opt-out must not silently degenerate into "always 1.0"
    either."""
    arm = build_arm("validity", require_novelty_index=False)
    out = arm([VALID, VALID], [500.0, 500.0], [(500.0, 1.0, 4), (500.0, 1.0, 4)])
    assert out[0].value == 1.0
    assert out[0].components["tsd"] == 1.0, "TSD is a no-op without an index, not a failure"
    assert out[1].value == 0.0, "DD still applies under the opt-out"
    assert out[1].components["dd"] == 0.0


def test_constraint_arm_requires_every_condition():
    arm = build_arm("constraint", novelty_index=_FakeIndex([]),
                    tolerance=50.0, sa_max=6.0)
    on_target = arm([VALID], [500.0], [(510.0, 2.0, 4)])[0].value
    off_target = arm([VALID], [500.0], [(700.0, 2.0, 4)])[0].value
    assert on_target == 1.0
    assert off_target == 0.0, "conjunction: missing one condition scores zero"


def test_constraint_arm_sa_leg_can_fail_alone():
    """Tg on-target and novel, but not synthesisable, must still score zero -
    otherwise an implementation could drop the SA leg and this suite would not
    notice."""
    actual_sa = synthetic_accessibility(VALID_CANONICAL)
    assert actual_sa is not None, "SA scorer must be available for this test to mean anything"
    arm = build_arm("constraint", novelty_index=_FakeIndex([]),
                    tolerance=50.0, sa_max=actual_sa - 0.5)
    out = arm([VALID], [500.0], [(510.0, 2.0, 4)])[0]
    assert out.value == 0.0
    assert out.components["in_window"] == 1.0
    assert out.components["novel"] == 1.0
    assert out.components["synthesisable"] == 0.0


def test_constraint_arm_novelty_leg_can_fail_alone():
    """Tg on-target and synthesisable, but already in the reference set, must
    still score zero - otherwise an implementation could drop the novelty leg
    and this suite would not notice."""
    arm = build_arm("constraint", novelty_index=_FakeIndex([VALID_CANONICAL]),
                    tolerance=50.0, sa_max=6.0)
    out = arm([VALID], [500.0], [(510.0, 2.0, 4)])[0]
    assert out.value == 0.0
    assert out.components["in_window"] == 1.0
    assert out.components["synthesisable"] == 1.0
    assert out.components["novel"] == 0.0


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
