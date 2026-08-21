"""Spec section 4.4 drift monitoring: watch the policy leave the data, do not stop it.

Spec section 4.4 ("Drift is monitored, not prevented") promises two logged
quantities that nothing in the branch actually computed:

* **the auditor gap** -- the held-out split-4 predictor's Tg estimate minus the
  reward ensemble's, per candidate. A policy farming the reward ensemble drives
  this apart; a policy genuinely getting closer to the requested Tg does not.
* **max-Tanimoto to the labelled Tg set** -- how close each generated candidate
  sits to its nearest known polymer, using the paper's loop-closed ECFP6
  fingerprints (:mod:`polyt5.evaluation.similarity`).

The second matters more than it looks. Novelty as this branch rewards it is
*absence of the exact canonical form* from the index
(:meth:`~polyt5.chemistry.scalable_novelty.ScalableNoveltyIndex.is_novel`), so
a one-atom edit of a memorised training polymer scores ``novel = 1.0`` -- full
credit in C3's novelty term and a satisfied conjunct in C4's. Nothing in the
reward can tell that apart from genuinely new chemistry. ``max_tanimoto_mean``
climbing toward 1.0 is the only in-flight signal that it is happening. This
module REPORTS that; it does not change any reward, and no reward term reads
its output.

Auditor containment
-------------------
Loading the auditor inside the training process is a real containment risk: the
whole study rests on split 4 never entering a reward path. Three things keep it
out, structurally rather than by convention:

1. :class:`DriftMonitor` accepts the auditor as a plain
   ``Callable[[Sequence[str]], Sequence[float]]`` and stores it privately. It
   exposes exactly one method, :meth:`DriftMonitor.observe`, which returns a
   dict of floats. There is no accessor that hands the predictor back out.
2. :class:`~polyt5.rl.trainer.GRPOTrainer` holds the monitor and the reward arm
   in separate attributes and never passes one to the other; ``observe`` is
   called AFTER the optimizer step, and its return value only ever reaches the
   stats dict.
3. ``tests/test_rl_drift.py`` pins that the rewards, advantages and loss of a
   step are bit-identical with and without a monitor attached.

What the auditor gap can and cannot establish is a separate question -- see
:func:`DriftMonitor.observe`'s note and ``artifacts/baseline/
frozen_baseline.json``'s ``auditor_note``: split 4 is an independent random 80%
sibling of the same corpus, so it is held out of the reward PATH, not
statistically independent of the reward models.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from polyt5.chemistry import pselfies_to_psmiles
from polyt5.evaluation.similarity import (
    FINGERPRINTS_AVAILABLE,
    build_reference_fingerprints,
    ecfp6,
)

__all__ = ["DEFAULT_MAX_REFERENCE", "DriftMonitor"]

#: Reference polymers fingerprinted for the max-Tanimoto half. The labelled Tg
#: generation training split is ~6.6k rows, so this is "all of it" today; the
#: cap exists so pointing the monitor at a 100M-row corpus degrades to a
#: subsample rather than to an out-of-memory error.
DEFAULT_MAX_REFERENCE = 20_000

#: Similarity at or above which a candidate is counted a near-copy of something
#: already in the labelled set. Exact-canonical-match novelty scores such a
#: candidate 1.0; this counter is what makes that visible.
NEAR_COPY_THRESHOLD = 0.9


def _to_psmiles(candidate: str) -> str | None:
    """Best-effort PSELFIES -> PSMILES, returning ``None`` on any failure."""
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    try:
        psmiles = pselfies_to_psmiles(candidate)
    except Exception:
        return None
    return psmiles or None


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


class DriftMonitor:
    """Computes spec section 4.4's two drift quantities for one rollout batch.

    Both halves are optional and independently degradable: a monitor built with
    no auditor reports the auditor keys as ``None``, and one whose reference set
    could not be fingerprinted reports the Tanimoto keys as ``None``. A missing
    capability is reported, never substituted -- the same rule the rest of this
    codebase follows.
    """

    def __init__(
        self,
        *,
        auditor: Callable[[Sequence[str]], Sequence[float]] | None = None,
        reference_psmiles: Sequence[str] | None = None,
        max_reference: int = DEFAULT_MAX_REFERENCE,
        seed: int = 0,
    ) -> None:
        """Build a monitor.

        Args:
            auditor: The held-out confirmation predictor, as a callable
                returning one float per candidate (NaN where it produced no
                number). Stored privately; never handed back out. ``None``
                disables the auditor-gap half.
            reference_psmiles: Known polymers -- the labelled Tg training set --
                for the max-Tanimoto half. Fingerprinted once, here.
            max_reference: Cap on how many reference polymers are fingerprinted;
                a larger set is randomly subsampled, so the reported max is a
                lower bound on the true max over the full set.
            seed: Seed for that subsample, so a reported number is reproducible.
        """
        self._auditor = auditor
        self.max_reference = max_reference
        self.reference_seed = seed

        self._reference_fps: list[Any] = []
        self.n_reference = 0
        if reference_psmiles and FINGERPRINTS_AVAILABLE:
            unique = list(dict.fromkeys(reference_psmiles))
            if len(unique) > max_reference:
                unique = random.Random(seed).sample(unique, max_reference)
            self._reference_fps = build_reference_fingerprints(unique)
            self.n_reference = len(self._reference_fps)

    @property
    def has_auditor(self) -> bool:
        """Whether the auditor-gap half is available."""
        return self._auditor is not None

    @property
    def has_similarity(self) -> bool:
        """Whether the max-Tanimoto half is available."""
        return bool(self._reference_fps)

    def observe(
        self,
        candidates: Sequence[str],
        ensemble_predictions: Sequence[tuple[float, float, int]],
    ) -> dict[str, float | None]:
        """Measure drift for one rollout batch.

        Args:
            candidates: The batch's generated PSELFIES strings.
            ensemble_predictions: The reward ensemble's own
                ``(mean, std, n_contributing_members)`` triples for those same
                candidates -- reused rather than recomputed, so the monitor
                costs one auditor pass and nothing else.

        Returns:
            A stats dict. Keys are always present; a value is ``None`` when the
            quantity could not be measured at all (no auditor, no usable
            reference set, or nothing comparable in this batch).

            ``auditor_gap_mean``
                Mean ``|auditor - ensemble_mean|`` over candidates both could
                score, in Kelvin.
            ``auditor_gap_signed_mean``
                Mean ``auditor - ensemble_mean``. A persistent sign says the
                ensemble is systematically optimistic or pessimistic relative to
                the held-out split, which a magnitude-only gap hides.
            ``auditor_gap_n``
                How many candidates that mean is over.
            ``max_tanimoto_mean`` / ``max_tanimoto_p90``
                Mean and 90th-percentile similarity to the nearest labelled
                polymer.
            ``near_copy_fraction``
                Fraction of fingerprintable candidates whose nearest labelled
                neighbour is at or above :data:`NEAR_COPY_THRESHOLD`. This is
                the counter that exact-canonical-match novelty cannot provide.
            ``max_tanimoto_n``
                How many candidates were fingerprintable.

        Note:
            The auditor gap answers "do these four particular reward models
            agree with a fifth model trained the same way", not "is the Tg claim
            true". Split 4 shares ~80% of its training data with each reward
            model in expectation, so a reward hack exploiting a genuinely
            data-sparse region fools all five identically. It is a real check
            and a weak one; see ``frozen_baseline.json``'s ``auditor_note``.
        """
        stats: dict[str, float | None] = {
            "auditor_gap_mean": None,
            "auditor_gap_signed_mean": None,
            "auditor_gap_n": None,
            "max_tanimoto_mean": None,
            "max_tanimoto_p90": None,
            "near_copy_fraction": None,
            "max_tanimoto_n": None,
        }

        if self._auditor is not None:
            try:
                auditor_values = list(self._auditor(candidates))
            except Exception:
                auditor_values = []
            gaps: list[float] = []
            for auditor_value, prediction in zip(auditor_values, ensemble_predictions,
                                                 strict=False):
                ensemble_mean = prediction[0]
                if math.isfinite(auditor_value) and math.isfinite(ensemble_mean):
                    gaps.append(float(auditor_value) - float(ensemble_mean))
            stats["auditor_gap_n"] = float(len(gaps))
            if gaps:
                stats["auditor_gap_mean"] = sum(abs(gap) for gap in gaps) / len(gaps)
                stats["auditor_gap_signed_mean"] = sum(gaps) / len(gaps)

        if self._reference_fps:
            from rdkit import DataStructs

            similarities: list[float] = []
            for candidate in candidates:
                psmiles = _to_psmiles(candidate)
                if psmiles is None:
                    continue
                fingerprint = ecfp6(psmiles)
                if fingerprint is None:
                    continue
                try:
                    bulk = DataStructs.BulkTanimotoSimilarity(fingerprint, self._reference_fps)
                except Exception:
                    continue
                if bulk:
                    similarities.append(float(max(bulk)))
            stats["max_tanimoto_n"] = float(len(similarities))
            if similarities:
                stats["max_tanimoto_mean"] = sum(similarities) / len(similarities)
                stats["max_tanimoto_p90"] = _percentile(sorted(similarities), 0.9)
                stats["near_copy_fraction"] = sum(
                    1.0 for value in similarities if value >= NEAR_COPY_THRESHOLD
                ) / len(similarities)

        return stats
