# tests/test_rewards.py
from __future__ import annotations

import pytest

from polyt5.chemistry.canonicalization import canonical_psmiles
from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.chemistry.metrics import synthetic_accessibility
from polyt5.rewards import (
    DEFAULT_SIGMA_UNKNOWN,
    RewardResult,
    TgRewardConfig,
    build_arm,
    effective_sigma,
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
        got = tg_reward(500.0 + err, std, 500.0, n_contributing=1, n_total=1, config=cfg).value
        assert got == pytest.approx(expected, abs=0.002), f"err={err} std={std}"


def test_confidence_weighting_penalises_disagreement_not_novelty():
    cfg = TgRewardConfig()
    agree = tg_reward(505.0, 2.0, 500.0, n_contributing=1, n_total=1, config=cfg).value
    disagree = tg_reward(505.0, 45.0, 500.0, n_contributing=1, n_total=1, config=cfg).value
    assert agree > disagree
    assert disagree > 0.0, "soft weighting must never zero out exploration"


def test_tg_reward_is_zero_beyond_tolerance():
    assert tg_reward(700.0, 1.0, 500.0, n_contributing=1, n_total=1,
                     config=TgRewardConfig(tolerance=100.0)).value == 0.0


def test_tg_reward_records_unweighted_value_for_logging():
    r = tg_reward(510.0, 45.0, 500.0, n_contributing=1, n_total=1, config=TgRewardConfig())
    assert "closeness" in r.components and "confidence" in r.components
    assert r.components["closeness"] == pytest.approx(0.9, abs=1e-6)


def test_non_finite_prediction_is_gated():
    r = tg_reward(float("nan"), 5.0, 500.0, n_contributing=1, n_total=1,
                   config=TgRewardConfig())
    assert r.gated is True and r.value == 0.0


# ---------------------------------------------------------------------------
# Ruling F / review finding B1: the confidence weight must not INVERT on
# candidates the reward ensemble cannot score.
#
# `EnsemblePropertyPredictor.predict_with_uncertainty` drops members that
# returned no number, so `std == 0.0` means EITHER "every member agreed" OR
# "only one member answered, and a spread over one number is undefined". The
# old weight `1/(1 + std/sigma0)` read both as maximal confidence, so a
# molecule three of four reward models could not parse scored 1.0000 while one
# all four agreed on scored 0.8095 -- an ASCENDING gradient toward chemistry
# that breaks the reward models' decoders, which is precisely what the weight
# exists to prevent.
# ---------------------------------------------------------------------------


def test_one_of_four_scores_far_below_four_of_four_at_equal_closeness():
    """The whole finding, as one numeric assertion.

    Both candidates are EXACTLY on target (closeness 1.0) and both report
    ``std`` values that the old formula treated as high confidence. The only
    difference is how much of the ensemble could actually read them.
    """
    cfg = TgRewardConfig()
    all_four = tg_reward(400.0, 4.0, 400.0, n_contributing=4, n_total=4, config=cfg)
    one_of_four = tg_reward(400.0, 0.0, 400.0, n_contributing=1, n_total=4, config=cfg)

    assert all_four.components["closeness"] == pytest.approx(1.0)
    assert one_of_four.components["closeness"] == pytest.approx(1.0)

    # The exact numbers from the review's reproduction, now the right way up.
    assert all_four.value == pytest.approx(0.8095, abs=1e-4)
    assert one_of_four.value == pytest.approx(0.0683, abs=1e-4)
    assert one_of_four.value < all_four.value, (
        "a candidate three quarters of the reward ensemble could not parse must NEVER "
        "outscore one all four agreed on"
    )


def test_confidence_is_monotone_in_how_much_of_the_ensemble_answered():
    """Mutant (a): a confidence weight that ignores ``n_contributing``.

    Holding the reported spread fixed, more contributing members must never
    score lower. Dropping the coverage factor makes 1-of-4, 2-of-4, 3-of-4 and
    4-of-4 all identical and this fails.
    """
    cfg = TgRewardConfig()
    values = [
        tg_reward(400.0, 4.0, 400.0, n_contributing=n, n_total=4, config=cfg).value
        for n in (2, 3, 4)
    ]
    assert values == sorted(values), values
    assert len(set(values)) == 3, "coverage is not affecting the reward at all"


def test_zero_contributing_members_scores_zero_and_is_gated():
    result = tg_reward(float("nan"), float("nan"), 400.0, n_contributing=0, n_total=4)
    assert result.value == 0.0
    assert result.gated is True
    assert result.reason == "no_contributing_members"


def test_a_single_member_of_a_real_ensemble_uses_the_pessimistic_sigma():
    """``std`` is UNDEFINED there, not zero, so the maximum observed
    disagreement is substituted -- never the reported ``0.0``.
    """
    assert effective_sigma(0.0, 1, 4) == pytest.approx(DEFAULT_SIGMA_UNKNOWN)
    assert effective_sigma(4.0, 4, 4) == pytest.approx(4.0)


def test_a_single_model_predictor_is_not_the_undefined_case():
    """Ruling C must survive Ruling F.

    ``compare_arms``'s auditor is ONE model reporting ``(mean, 0.0, 1)``. It
    has no members that could have failed, so its coverage is 1/1 and its
    spread really is 0.0: the auditor-side score stays unweighted closeness.
    Conflating it with "1 of 4" would silently multiply every auditor column
    by 0.068.
    """
    auditor = tg_reward(400.0, 0.0, 400.0, n_contributing=1, n_total=1)
    assert auditor.value == pytest.approx(1.0)
    assert auditor.components["confidence"] == pytest.approx(1.0)


def test_declaring_the_wrong_ensemble_size_raises_rather_than_understating():
    with pytest.raises(ValueError, match="n_contributing"):
        tg_reward(400.0, 0.0, 400.0, n_contributing=4, n_total=1)


def test_partial_ensemble_is_visible_in_the_components_for_step_logging():
    result = tg_reward(400.0, 0.0, 400.0, n_contributing=1, n_total=4)
    assert result.components["n_contributing"] == 1.0
    assert result.components["coverage"] == pytest.approx(0.25)
    assert result.components["sigma_effective"] == pytest.approx(DEFAULT_SIGMA_UNKNOWN)


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


def test_accuracy_arm_uses_only_the_tg_term():
    arm = build_arm("accuracy", ensemble_size=4)
    out = arm([VALID], [500.0], [(505.0, 5.0, 4)])
    assert 0.0 < out[0].value <= 1.0
    assert "closeness" in out[0].components


def test_every_arm_zeroes_an_invalid_candidate():
    for name in ("accuracy", "validity", "composite", "constraint"):
        arm = build_arm(name, novelty_index=_FakeIndex([]), ensemble_size=4)
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
                    tolerance=50.0, sa_max=6.0, ensemble_size=4)
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
                    tolerance=50.0, sa_max=actual_sa - 0.5, ensemble_size=4)
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
                    tolerance=50.0, sa_max=6.0, ensemble_size=4)
    out = arm([VALID], [500.0], [(510.0, 2.0, 4)])[0]
    assert out.value == 0.0
    assert out.components["in_window"] == 1.0
    assert out.components["synthesisable"] == 1.0
    assert out.components["novel"] == 0.0


def test_composite_arm_weights_are_configurable_not_hardcoded():
    idx = _FakeIndex([])
    a = build_arm("composite", novelty_index=idx, ensemble_size=4,
                  weights={"tg": 1.0, "pv": 0.0, "novelty": 0.0})
    b = build_arm("composite", novelty_index=idx, ensemble_size=4,
                  weights={"tg": 0.0, "pv": 1.0, "novelty": 0.0})
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


# ------------------------------------------------------- arms thread n_contributing


def test_every_tg_reading_arm_penalises_a_partial_ensemble():
    """Ruling F: ``n_contributing`` must reach C1, C3 AND C4, not just
    ``tg_reward``. Each arm is scored twice on the SAME candidate at the SAME
    target -- once with all four members agreeing, once with a single member's
    unopposed guess.
    """
    idx = _FakeIndex([])
    cases = {
        "accuracy": build_arm("accuracy", ensemble_size=4),
        "composite": build_arm("composite", novelty_index=idx, ensemble_size=4),
        "constraint": build_arm("constraint", novelty_index=idx, ensemble_size=4,
                                tolerance=50.0, sa_max=6.0),
    }
    for name, arm in cases.items():
        consensus = arm([VALID], [400.0], [(400.0, 4.0, 4)])[0].value
        partial = arm([VALID], [400.0], [(400.0, 0.0, 1)])[0].value
        assert partial < consensus, (
            f"{name}: a one-of-four prediction scored {partial} against a full-consensus "
            f"{consensus} -- this arm is still reading a single member's guess as a consensus"
        )


def test_constraint_arm_requires_the_ensemble_to_have_actually_scored_it():
    """C4 carries no continuous confidence weight by spec, so coverage enters
    as a fourth conjunct instead. One member of four is below ``min_coverage``
    and the conjunction fails even though Tg, SA and novelty all pass.
    """
    arm = build_arm("constraint", novelty_index=_FakeIndex([]), ensemble_size=4,
                    tolerance=50.0, sa_max=6.0)
    out = arm([VALID], [400.0], [(400.0, 0.0, 1)])[0]
    assert out.value == 0.0
    assert out.components["in_window"] == 1.0
    assert out.components["synthesisable"] == 1.0
    assert out.components["novel"] == 1.0
    assert out.components["ensemble_backed"] == 0.0
    assert out.components["coverage"] == pytest.approx(0.25)


def test_constraint_arm_accepts_a_single_model_predictor():
    """Ruling C again: the auditor is one model, so 1 of 1 is full coverage."""
    arm = build_arm("constraint", novelty_index=_FakeIndex([]), ensemble_size=1,
                    tolerance=50.0, sa_max=6.0)
    out = arm([VALID], [400.0], [(400.0, 0.0, 1)])[0]
    assert out.value == 1.0
    assert out.components["ensemble_backed"] == 1.0


def test_validity_arm_rejects_misaligned_inputs_like_every_other_arm():
    """Minor 12: ``ValidityArm`` reads neither targets nor predictions, but it
    must still refuse three sequences of different lengths rather than
    silently scoring the shortest.
    """
    arm = build_arm("validity", novelty_index=_FakeIndex([]))
    with pytest.raises(ValueError):
        arm([VALID, VALID], [500.0], [(500.0, 1.0, 4), (500.0, 1.0, 4)])


def test_sa_reward_is_no_longer_a_public_symbol():
    """Minor 13: the branch's one unwired public symbol. No arm called it --
    ``ConstraintArm`` reads the raw SA score and ``constraint_reward`` applies
    the threshold -- and wiring it would have changed C4's reward at the
    boundary, which is a spec change, not a review fix.
    """
    import polyt5.rewards as rewards

    assert not hasattr(rewards, "sa_reward")
    assert "sa_reward" not in rewards.__all__
