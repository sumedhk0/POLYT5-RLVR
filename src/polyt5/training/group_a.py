"""The seven Group A ablation configurations.

Spec section 5 runs seven configurations over the SAME five splits the frozen
baseline used, so every number is directly comparable to 28.6733 +/- 0.7591 K::

    B0  baseline -- current text head, single task
    A1  + regression head
    A2  + descriptor auxiliaries
    A3  + invariance augmentation
    A4  + label weighting
    A5  + multi-task shared encoder
    A6  all five combined

The table lives here rather than in YAML because "A6 is all five combined" then
becomes a property a test asserts. A6's switches are computed as the union of
A1 through A5's switches below, not written out by hand: if a future edit
changes which switch an individual arm flips, A6 follows it automatically
instead of silently drifting out of sync with the arms it is supposed to
combine. Individual ablations run AS WELL AS the combination: a combined gain
with no per-change attribution cannot tell you which idea to keep.

Cycle consistency (spec 4.5) is a field here but is OFF on every arm. A model
can satisfy it by being consistently wrong -- generate something odd,
confidently mispredict it, incur zero loss -- so it is a regulariser anchored by
real labels, never a primary objective, and it ships behind a flag that no
ablation arm sets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ARM_IDS", "SWITCH_NAMES", "GroupAConfig", "arm_config"]

#: The seven configurations, in spec section 5's order.
ARM_IDS: tuple[str, ...] = ("B0", "A1", "A2", "A3", "A4", "A5", "A6")

#: The five independently switchable changes.
SWITCH_NAMES: tuple[str, ...] = (
    "regression_head",
    "descriptors",
    "augment",
    "reliability_weighting",
    "multitask",
)

#: The single-change arms: each names the one switch it flips on.
_SINGLE_ARM_SWITCHES: dict[str, tuple[str, ...]] = {
    "B0": (),
    "A1": ("regression_head",),
    "A2": ("descriptors",),
    "A3": ("augment",),
    "A4": ("reliability_weighting",),
    "A5": ("multitask",),
}

#: A6 is DERIVED as the union of A1..A5's switches, never typed out by hand --
#: an edit to any single-change arm above propagates here without a second edit.
_ARM_SWITCHES: dict[str, tuple[str, ...]] = {
    **_SINGLE_ARM_SWITCHES,
    "A6": tuple(
        name
        for name in SWITCH_NAMES
        if any(name in _SINGLE_ARM_SWITCHES[arm] for arm in ("A1", "A2", "A3", "A4", "A5"))
    ),
}


@dataclass(frozen=True)
class GroupAConfig:
    """One ablation configuration: five switches plus their hyperparameters.

    Attributes:
        arm: The configuration id, e.g. ``"A3"``.
        regression_head: Predict Tg with the pooled scalar head instead of
            decoding it as text.
        descriptors: Add the auxiliary descriptor heads,
            ``L = L_Tg + descriptor_lambda * L_descriptors``.
        augment: Train on several PSELFIES writings per polymer.
        reliability_weighting: Weight examples by ``1 / max(std, std_floor)``.
            The ``reliability == red`` drop this switch was once documented
            as also controlling runs UNCONDITIONALLY for every arm, including
            B0 -- ``polyt5.data.multitask._drop_red_for_split`` is called
            regardless of this flag, and
            ``test_red_rows_leave_train_but_the_test_split_is_untouched``
            pins that as deliberate. So A4 measures only the weighting half
            of spec Sec 4.4's "weight by 1/max(std,floor) AND drop red" change;
            the drop itself is common to every arm's train/val pool and is
            not part of what distinguishes A4 from B0. Whole-branch review
            finding 6: impact is 2-3 train rows and 1 val row per split out
            of ~5,295, so this is a documentation correction, not a behaviour
            change -- gating the drop on this switch was considered and
            rejected, since it would perturb B0's in-harness rerun (and every
            other arm) for a negligible-magnitude, already-tested design.
        multitask: Train prediction and generation together on the shared
            encoder, alternating batches.
        cycle_consistency: OFF on every arm; see the module docstring.
        descriptor_lambda: Weight of the descriptor term. Configurable because
            100 auxiliary targets against one Tg target risk swamping the
            objective we care about (spec section 8).
        n_writings: Writings per polymer when ``augment`` is on. Configurable
            because a large N means fewer distinct chemistries per epoch.
        std_floor: Weight floor in Kelvin; must be positive.
        huber_delta: Huber transition point, in standardised units.
        cycle_weight: Weight of the cycle term when it is enabled at all.
    """

    arm: str
    regression_head: bool = False
    descriptors: bool = False
    augment: bool = False
    reliability_weighting: bool = False
    multitask: bool = False
    cycle_consistency: bool = False
    descriptor_lambda: float = 0.1
    n_writings: int = 4
    std_floor: float = 5.6
    huber_delta: float = 1.0
    cycle_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.descriptor_lambda < 0.0:
            raise ValueError(f"descriptor_lambda must be >= 0, got {self.descriptor_lambda}")
        if self.n_writings < 1:
            raise ValueError(f"n_writings must be >= 1, got {self.n_writings}")
        if self.std_floor <= 0.0:
            raise ValueError(f"std_floor must be > 0, got {self.std_floor}")
        if self.huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")
        if self.cycle_consistency and not self.regression_head:
            raise ValueError(
                "cycle_consistency needs regression_head=True: the cycle scores its own "
                "generations with the regression head, and without one there is nothing "
                "to close the loop with"
            )
        if self.cycle_consistency and self.cycle_weight <= 0.0:
            raise ValueError(
                f"cycle_weight must be > 0 when cycle_consistency is on, got "
                f"{self.cycle_weight}"
            )

    def switches(self) -> dict[str, bool]:
        """The five switches as a plain dict, in :data:`SWITCH_NAMES` order."""
        return {name: bool(getattr(self, name)) for name in SWITCH_NAMES}

    def effective_n_writings(self) -> int:
        """Writings per polymer actually used: 1 unless ``augment`` is on."""
        return self.n_writings if self.augment else 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the run manifest."""
        return asdict(self)


def arm_config(arm: str, **overrides: Any) -> GroupAConfig:
    """Build the configuration for one of the seven arms.

    Args:
        arm: One of :data:`ARM_IDS`.
        **overrides: Hyperparameter overrides (``descriptor_lambda``,
            ``n_writings``, ``std_floor``, ``huber_delta``, ``cycle_weight``,
            ``cycle_consistency``). The five switches cannot be overridden --
            they define the arm.

    Returns:
        The configuration.

    Raises:
        ValueError: If ``arm`` is unknown, or an override names a switch.
    """
    if arm not in _ARM_SWITCHES:
        raise ValueError(f"unknown arm {arm!r}; valid arms are {list(ARM_IDS)}")
    clashing = sorted(set(overrides) & set(SWITCH_NAMES))
    if clashing:
        raise ValueError(
            f"cannot override the switch(es) {clashing} on arm {arm!r}: the switches define "
            "the arm, and changing one would make the ablation row unattributable"
        )
    enabled = dict.fromkeys(_ARM_SWITCHES[arm], True)
    return GroupAConfig(arm=arm, **enabled, **overrides)
