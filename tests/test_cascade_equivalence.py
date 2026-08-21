"""The SV -> TSD -> DD -> PV cascade exists twice. Pin them together.

Minor 17 of the final review. :class:`polyt5.rewards.composite.ValidityArm` is
what C2 *trains* on; :func:`polyt5.evaluation.apply_filter_cascade` is what C2
is *judged* by, since ``ARM_METRIC["validity"] = "pv_rate"`` and ``pv_rate``
comes from ``screen_candidates``, which calls the evaluation cascade. The
ledger deferred reconciling the two as "acceptable duplication; reconcile only
if a third caller appears" -- ``scripts/compare_arms.py`` is now effectively
that third caller, and the equivalence is load-bearing for C2's entire matrix
column.

They agree today. Nothing held them there. This does.
"""

from __future__ import annotations

import pytest

from polyt5.chemistry.canonicalization import canonical_psmiles
from polyt5.chemistry.conversion import pselfies_to_psmiles
from polyt5.evaluation import apply_filter_cascade
from polyt5.rewards import build_arm

#: One batch spanning every stage of the cascade plus a within-batch duplicate.
BATCH: tuple[str, ...] = (
    "[At][C][C][O][At]",          # full pass
    "[At][C][C][O][At]",          # DD: exact repeat of the line above
    "[At][C][C][C][At]",          # full pass, different polymer
    "[Zz][Qq][not-a-token]",      # SV: unparseable
    "",                           # SV: empty
    "((((",                       # SV: junk
    "[At][C][C][O]",              # PV: one terminus only
    "[At][=C][At]",               # PV: double-bonded terminus
    "[C][C][O]",                  # PV: no termini at all
    "[At][C][O][C][At]",          # full pass
    "[At][C][O][C][At]",          # DD: repeat
    "[At][C][C][N][At]",          # full pass
    "[At][C][C][C][At]",          # DD: repeat of index 2
    "[At][O][C][C][O][At]",       # full pass
    "\x00",                       # SV: control character
    "[At][C][Ring9][C][At]",      # SV: unresolvable ring
)


class _FakeIndex:
    """TSD reference set, in the injected-index protocol the reward side uses."""

    def __init__(self, known):
        self._known = set(known)

    # ValidityArm's protocol.
    def is_novel(self, psmiles):
        return psmiles not in self._known

    # apply_filter_cascade's NoveltyIndex protocol.
    def __contains__(self, psmiles):
        return psmiles in self._known


def _known_canonical(pselfies: str) -> str:
    return canonical_psmiles(pselfies_to_psmiles(pselfies))


@pytest.mark.parametrize("training_set", [
    (),                                 # nothing is a training duplicate
    ("[At][C][C][O][At]",),             # the first full-pass polymer is
    ("[At][C][C][N][At]", "[At][O][C][C][O][At]"),
])
def test_validity_arm_and_apply_filter_cascade_agree_candidate_by_candidate(training_set):
    """Not just the same rate -- the same VERDICT on every candidate.

    A rate-only comparison would pass if the two implementations disagreed on
    two candidates in opposite directions, which is exactly the drift this is
    supposed to catch.
    """
    index = _FakeIndex(_known_canonical(entry) for entry in training_set)
    arm = build_arm("validity", novelty_index=index)

    rewards = arm(list(BATCH), [400.0] * len(BATCH), [(400.0, 3.0, 4)] * len(BATCH))
    records, counts = apply_filter_cascade(list(BATCH), training_index=index)

    mismatches = [
        (i, candidate, reward.value, record.passed_pv)
        for i, (candidate, reward, record) in enumerate(zip(BATCH, rewards, records, strict=True))
        if bool(reward.value) != bool(record.passed_pv)
    ]
    assert not mismatches, f"the two cascade implementations disagree: {mismatches}"

    arm_rate = sum(reward.value for reward in rewards) / len(BATCH)
    cascade_rate = counts.n_pv / counts.n_input
    assert arm_rate == pytest.approx(cascade_rate), (
        "C2 trains on ValidityArm's cascade and is judged on apply_filter_cascade's "
        "pv_rate; if these diverge, C2's entire matrix column means something other "
        "than what it was trained to maximise"
    )


def test_the_batch_actually_exercises_every_stage():
    """A pinning test on two implementations is worthless if the batch is
    trivial -- this fails if the fixture stops covering some stage.
    """
    index = _FakeIndex([_known_canonical("[At][C][C][O][At]")])
    records, counts = apply_filter_cascade(list(BATCH), training_index=index)
    stages = {record.failure_stage for record in records}
    assert {"SV", "TSD", "DD", "PV"} <= stages, stages
    assert counts.n_pv > 0, "and at least one candidate must clear the whole cascade"
