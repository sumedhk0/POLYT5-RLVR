# src/polyt5/data/multitask.py
"""Group A batch construction: one item shape for all seven ablation arms.

Every arm -- text head or regression head, augmented or not, weighted or not,
single-task or multi-task -- produces batches with the same keys, so one
collator and one trainer serve all seven and the five switches stay genuinely
independent. Unused slots are empty tensors rather than absent keys, because
``Trainer._to_device`` moves every value and a missing key would branch the
training loop.

:func:`assemble_split` owns the ordering rules, and they are load-bearing:

1. ``reliability == red`` rows leave TRAIN and VAL. **Test is never filtered** --
   changing the evaluation set would make MAE incomparable to the frozen
   28.6733 K, which is the only number every arm is measured against.
   ``n_red_in_test`` reports the residual instead of hiding it. This is
   enforced by a runtime guard, :func:`_drop_red_for_split`, which is the
   single call site every reliability-based row drop goes through and which
   refuses outright to run for ``split="test"`` -- see its docstring.
2. Standardizers are fitted on TRAIN rows only, after the red drop and
   **before** augmentation -- augmenting first lets a polymer with many
   writings drag the mean.
3. Reliability weights are computed on train rows before augmentation; each
   writing inherits its source polymer's weight.
4. Only TRAIN is augmented. Val selects checkpoints; test is the measurement.
5. Generation items come from train polymers only, and only when asked, and
   from the SAME augmented writings as the prediction items -- so when
   augmentation and multi-task are both on (arm A6), the two streams' sizes
   scale together instead of one being replayed via ``InterleavedLoader``
   cycling to match the other's inflated length.

This module imports torch and is therefore NOT re-exported from
``polyt5.data.__init__`` -- same rule as ``polyt5.data.datasets``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.utils.data

from polyt5.data.augment import augment_indices
from polyt5.data.collate import LABEL_IGNORE_ID, pad_sequences
from polyt5.data.prepare import format_property_value
from polyt5.data.standardize import Standardizer, fit_target_standardizer
from polyt5.data.tg_metadata import TgExample, descriptor_matrix
from polyt5.data.weighting import drop_red_reliability, reliability_weights

__all__ = [
    "GENERATION_TASK",
    "PREDICTION_TASK",
    "SplitTensors",
    "TaskCollator",
    "TaskDataset",
    "TaskItem",
    "assemble_split",
]

#: Task markers carried in every batch as a ``(batch,)`` long tensor.
PREDICTION_TASK = 0
GENERATION_TASK = 1


@dataclass(frozen=True)
class TaskItem:
    """One training example, whichever arm produced it.

    Attributes:
        input_ids: Encoder input ids. PSELFIES for prediction, the formatted
            conditioning number for generation.
        label_ids: Decoder target ids, or ``()`` when the regression head owns
            the Tg objective and no text is decoded.
        tg_standardised: The Tg target in standardised units, always populated
            on both paths (the same standardizer runs regardless of the
            regression/text switch). The switch decides which field a head
            CONSUMES -- this one vs ``label_ids`` -- not which one exists;
            the text-head trainer simply does not read this field, the same
            way the regression head does not read ``label_ids``.
        descriptors: Standardised descriptor targets, or ``()`` when the
            descriptor switch is off.
        weight: Per-example loss weight; ``1.0`` when weighting is off.
        task_id: :data:`PREDICTION_TASK` or :data:`GENERATION_TASK`.
    """

    input_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    tg_standardised: float
    descriptors: tuple[float, ...]
    weight: float
    task_id: int


class TaskDataset(torch.utils.data.Dataset):
    """A plain, picklable list of :class:`TaskItem` for ``DataLoader``."""

    def __init__(self, items: Sequence[TaskItem]) -> None:
        """Wrap already-tokenized items.

        Args:
            items: Output of :func:`assemble_split`.
        """
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> TaskItem:
        """Return one item, already tokenized."""
        return self._items[index]

    @property
    def stats(self) -> dict[str, Any]:
        """Cheap descriptive statistics for run manifests."""
        return {
            "n_items": len(self._items),
            "n_prediction": sum(1 for i in self._items if i.task_id == PREDICTION_TASK),
            "n_generation": sum(1 for i in self._items if i.task_id == GENERATION_TASK),
        }


class TaskCollator:
    """Pad and tensorize a batch of :class:`TaskItem`.

    Every batch is single-task by construction (the loaders never mix), so
    ``task_id`` is uniform and the trainer can read element 0.
    """

    def __init__(
        self, pad_id: int, *, max_source_length: int = 200, max_target_length: int = 200
    ) -> None:
        """Initialize the collator.

        Args:
            pad_id: Encoder padding token id; labels always pad with ``-100``.
            max_source_length: Source truncation length.
            max_target_length: Target truncation length.
        """
        self.pad_id = int(pad_id)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)

    def __call__(self, batch: Sequence[TaskItem]) -> dict[str, torch.Tensor]:
        """Tensorize ``batch``.

        Returns:
            ``{"input_ids", "attention_mask", "labels", "tg_targets",
            "descriptor_targets", "weights", "task_id"}``. ``labels`` has width
            0 when no item decodes text; ``descriptor_targets`` has width 0
            when the descriptor switch is off.
        """
        inputs, mask = pad_sequences(
            [list(item.input_ids) for item in batch],
            pad_id=self.pad_id,
            max_length=self.max_source_length,
        )
        labels, _ = pad_sequences(
            [list(item.label_ids) for item in batch],
            pad_id=LABEL_IGNORE_ID,
            max_length=self.max_target_length,
        )
        n_descriptors = len(batch[0].descriptors) if batch else 0
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long).reshape(len(batch), -1),
            "tg_targets": torch.tensor(
                [item.tg_standardised for item in batch], dtype=torch.float32
            ),
            "descriptor_targets": torch.tensor(
                [list(item.descriptors) for item in batch], dtype=torch.float32
            ).reshape(len(batch), n_descriptors),
            "weights": torch.tensor([item.weight for item in batch], dtype=torch.float32),
            "task_id": torch.tensor([item.task_id for item in batch], dtype=torch.long),
        }


@dataclass(frozen=True)
class SplitTensors:
    """Everything one split needs, plus what was dropped getting there."""

    train: list[TaskItem]
    train_generation: list[TaskItem]
    val: list[TaskItem]
    test_pselfies: list[str]
    test_tg: list[float]
    target_standardizer: Standardizer
    descriptor_standardizer: Standardizer | None
    dropped_descriptor_columns: tuple[str, ...]
    n_train_polymers: int
    n_train_writings: int
    n_dropped_red: int
    n_red_in_test: int

    def to_manifest(self) -> dict[str, Any]:
        """Return the attrition and scaling record for the run manifest."""
        return {
            "n_train_polymers": self.n_train_polymers,
            "n_train_writings": self.n_train_writings,
            "n_train_generation": len(self.train_generation),
            "n_val": len(self.val),
            "n_test": len(self.test_pselfies),
            "n_dropped_red": self.n_dropped_red,
            "n_red_in_test": self.n_red_in_test,
            "dropped_descriptor_columns": list(self.dropped_descriptor_columns),
            "n_descriptors_kept": (
                0 if self.descriptor_standardizer is None
                else self.descriptor_standardizer.n_features
            ),
            "target_mean": self.target_standardizer.mean[0],
            "target_std": self.target_standardizer.std[0],
        }


def _drop_red_for_split(
    examples: Sequence[TgExample], split: str
) -> tuple[list[TgExample], list[TgExample]]:
    """Apply :func:`drop_red_reliability`, but refuse outright on the test split.

    ``drop_red_reliability`` is a leaf primitive and correctly takes no split
    parameter -- it does not know what a split is. The "never filter test"
    rule therefore has to be enforced here, at the composition entry point,
    which is the first place that does know. Every reliability-based row drop
    in :func:`assemble_split` is routed through this one function -- train's
    and val's calls pass, and there is no call for test at all -- so a future
    refactor that reused this call site for the test pool would fail loudly
    instead of silently shrinking the evaluation set.

    Args:
        examples: The pool for one split.
        split: ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        ``(kept, dropped)`` as :func:`drop_red_reliability` returns.

    Raises:
        ValueError: If ``split == "test"``. Dropping rows from test would
            change the evaluation set and make MAE incomparable to the frozen
            28.6733 K -- the one number every Group A arm is measured
            against. A shrunken test set would look exactly like a real
            improvement; nothing downstream could tell the difference.
    """
    if split == "test":
        raise ValueError(
            "reliability filtering must never run on split='test': dropping rows "
            "from test would change the evaluation set and make MAE incomparable "
            "to the frozen baseline"
        )
    return drop_red_reliability(examples)


def _encode(tokenizer, text: str, max_length: int) -> tuple[int, ...]:
    return tuple(
        tokenizer.encode(text, add_eos=True, max_length=max_length, truncation=True)
    )


def assemble_split(
    examples: Sequence[TgExample],
    descriptor_names: Sequence[str],
    *,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    tokenizer,
    use_regression_head: bool,
    use_descriptors: bool,
    n_writings: int,
    use_reliability_weighting: bool,
    std_floor: float,
    build_generation: bool,
    seed: int,
    max_source_length: int = 200,
    max_target_length: int = 200,
) -> SplitTensors:
    """Turn one split's indices into train/val items and a raw test set.

    Args:
        examples: The full prepared corpus, in frozen-splits order.
        descriptor_names: The 100 descriptor column names, positionally aligned
            with ``TgRow.descriptors``.
        train_indices: This split's train positions.
        val_indices: This split's validation positions (may be empty).
        test_indices: This split's held-out positions.
        tokenizer: Duck-typed tokenizer with ``encode``.
        use_regression_head: When True the Tg objective is the scalar head and
            items carry no text labels; when False they carry the numeric string.
        use_descriptors: Attach standardised descriptor targets.
        n_writings: Writings per train polymer (1 disables augmentation).
        use_reliability_weighting: Weight train examples by measurement spread.
        std_floor: Floor passed to :func:`reliability_weights`.
        build_generation: Also build Tg-conditioned generation items from train.
        seed: Augmentation seed.
        max_source_length: Encoder truncation length.
        max_target_length: Decoder truncation length.

    Returns:
        A :class:`SplitTensors`. ``test_pselfies``/``test_tg`` always have the
        same length as ``test_indices`` -- the test split is never filtered.
    """
    train_pool = [examples[i] for i in train_indices]
    val_pool = [examples[i] for i in val_indices]

    kept_train, dropped_train = _drop_red_for_split(train_pool, "train")
    kept_val, dropped_val = _drop_red_for_split(val_pool, "val")
    n_dropped_red = len(dropped_train) + len(dropped_val)
    n_red_in_test = sum(1 for i in test_indices if examples[i].row.reliability == "red")

    target_standardizer = fit_target_standardizer([e.row.tg for e in kept_train])

    descriptor_standardizer: Standardizer | None = None
    train_descriptors: np.ndarray | None = None
    if use_descriptors:
        matrix = descriptor_matrix(kept_train)
        descriptor_standardizer = Standardizer.fit(matrix, descriptor_names)
        train_descriptors = descriptor_standardizer.transform(matrix)

    weights = (
        reliability_weights(kept_train, floor=std_floor)
        if use_reliability_weighting
        else [1.0] * len(kept_train)
    )

    def scalar(example: TgExample) -> float:
        return float(
            target_standardizer.transform(np.asarray([[example.row.tg]], dtype=float))[0, 0]
        )

    def build_item(
        pselfies: str, example: TgExample, descriptors: tuple[float, ...], weight: float
    ) -> TaskItem:
        return TaskItem(
            input_ids=_encode(tokenizer, pselfies, max_source_length),
            label_ids=(
                ()
                if use_regression_head
                else _encode(
                    tokenizer, format_property_value(example.row.tg), max_target_length
                )
            ),
            tg_standardised=scalar(example),
            descriptors=descriptors,
            weight=weight,
            task_id=PREDICTION_TASK,
        )

    writings = augment_indices(
        kept_train,
        range(len(kept_train)),
        n_writings=n_writings,
        seed=seed,
        max_tokens=max_source_length,
        tokenizer=tokenizer,
    )
    train_items = [
        build_item(
            writing.pselfies,
            kept_train[writing.source_index],
            (
                tuple(float(v) for v in train_descriptors[writing.source_index])
                if train_descriptors is not None
                else ()
            ),
            weights[writing.source_index],
        )
        for writing in writings
    ]

    val_descriptors = (
        descriptor_standardizer.transform(descriptor_matrix(kept_val))
        if descriptor_standardizer is not None and kept_val
        else None
    )
    val_items = [
        build_item(
            example.pselfies,
            example,
            (
                tuple(float(v) for v in val_descriptors[position])
                if val_descriptors is not None
                else ()
            ),
            1.0,
        )
        for position, example in enumerate(kept_val)
    ]

    generation_items: list[TaskItem] = []
    if build_generation:
        # Built from the SAME `writings` as train_items, not from kept_train
        # directly. When augmentation is ALSO on (arm A6), this keeps the two
        # streams' sizes scaling together: with augmentation off (n_writings
        # == 1) `writings` reproduces kept_train verbatim (see
        # augment_indices), so this is a no-op for every arm but A6. With
        # augmentation on, generation now gets all n_writings per polymer
        # too, so InterleavedLoader never has to CYCLE the generation side
        # back up to match an augmented (larger) prediction side -- A6 gets
        # one natural pass over generation per epoch, not the same handful
        # of batches replayed ~n_writings times. That is what makes A6 a
        # genuine union of A3 and A5 in TRAINING BEHAVIOUR, not only in the
        # switches recorded in its GroupAConfig.
        generation_items = [
            TaskItem(
                input_ids=_encode(
                    tokenizer,
                    format_property_value(kept_train[writing.source_index].row.tg),
                    max_source_length,
                ),
                label_ids=_encode(tokenizer, writing.pselfies, max_target_length),
                tg_standardised=scalar(kept_train[writing.source_index]),
                descriptors=(),
                weight=1.0,
                task_id=GENERATION_TASK,
            )
            for writing in writings
        ]

    return SplitTensors(
        train=train_items,
        train_generation=generation_items,
        val=val_items,
        test_pselfies=[examples[i].pselfies for i in test_indices],
        test_tg=[examples[i].row.tg for i in test_indices],
        target_standardizer=target_standardizer,
        descriptor_standardizer=descriptor_standardizer,
        dropped_descriptor_columns=(
            () if descriptor_standardizer is None else descriptor_standardizer.dropped
        ),
        n_train_polymers=len(kept_train),
        n_train_writings=len(train_items),
        n_dropped_red=n_dropped_red,
        n_red_in_test=n_red_in_test,
    )
