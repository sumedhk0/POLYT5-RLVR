# src/polyt5/training/multitask_trainer.py
"""Alternating-batch training for the Group A arms.

Spec section 4.5 asks for prediction and generation trained together on the
shared encoder, "alternating batches". That is expressed here as a LOADER --
:class:`InterleavedLoader` yields one prediction batch, then one generation
batch -- so :meth:`polyt5.training.Trainer.train_epoch` runs completely
unchanged and every gradient-accumulation, AMP, clipping and checkpoint
behaviour is shared with the baseline trainer rather than re-implemented.

:class:`GroupATrainer` overrides exactly three hooks:

* ``_to_device`` -- tolerates non-tensor values in a batch dict.
* ``_forward_loss`` -- routes a batch to one of the model's three forward paths
  by its ``task_id``.
* ``_batch_weight`` -- weights the epoch mean by EXAMPLES, not label tokens,
  because a regression batch has no label tokens at all. The reported
  ``train_loss`` for a Group A run is therefore an example-weighted mean and is
  not comparable to the baseline trainer's token-weighted one. MAE on the held
  out split is the comparable number, and that is what the ablation reports.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import cycle
from typing import Any

import torch
from torch import Tensor

from polyt5.data.multitask import GENERATION_TASK, PREDICTION_TASK
from polyt5.model.multitask import PolyT5MultiTask
from polyt5.training.group_a import GroupAConfig
from polyt5.training.trainer import Batch, Trainer, TrainerConfig

__all__ = ["GroupATrainer", "InterleavedLoader"]


class InterleavedLoader:
    """Alternate batches from two loaders, one from each, until the longer ends.

    The shorter side is cycled, so a small generation set is reused rather than
    silently truncating the prediction set. With no generation loader this is a
    transparent pass-through, which is what the six single-task arms use.
    """

    def __init__(
        self, prediction: Iterable[Batch], generation: Iterable[Batch] | None = None
    ) -> None:
        """Initialize the loader.

        Args:
            prediction: Loader of prediction batches.
            generation: Optional loader of generation batches.
        """
        self.prediction = prediction
        self.generation = generation

    def _lengths(self) -> tuple[int, int]:
        n_prediction = len(self.prediction)  # type: ignore[arg-type]
        n_generation = (
            0 if self.generation is None else len(self.generation)  # type: ignore[arg-type]
        )
        return n_prediction, n_generation

    def __len__(self) -> int:
        """Total batches yielded per epoch."""
        n_prediction, n_generation = self._lengths()
        if n_generation == 0:
            return n_prediction
        return 2 * max(n_prediction, n_generation)

    def __iter__(self) -> Iterator[Batch]:
        """Yield prediction, generation, prediction, generation, ...

        Mirrors :meth:`__len__`: an EMPTY generation loader (``[]``) degrades
        to the same pass-through as ``generation=None``, rather than reaching
        ``next()`` on an exhausted ``cycle([])`` and raising ``RuntimeError``.
        """
        n_prediction, n_generation = self._lengths()
        if n_generation == 0:
            yield from self.prediction
            return
        rounds = max(n_prediction, n_generation)
        predictions = cycle(self.prediction) if n_prediction < rounds else iter(self.prediction)
        generations = cycle(self.generation) if n_generation < rounds else iter(self.generation)
        for _ in range(rounds):
            yield next(predictions)
            yield next(generations)


class GroupATrainer(Trainer):
    """The baseline trainer, with a task-routing loss and example weighting."""

    def __init__(
        self,
        model: PolyT5MultiTask,
        train_loader: Iterable[Batch],
        config: TrainerConfig,
        *,
        group_a: GroupAConfig,
        cycle_loss: Callable[[Tensor], Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the trainer.

        Args:
            model: The wrapped multi-task model.
            train_loader: Usually an :class:`InterleavedLoader`.
            config: Standard trainer configuration.
            group_a: Which of the five switches this arm turns on.
            cycle_loss: Optional callable taking the batch's standardised Tg
                targets and returning a scalar cycle-consistency loss. ``None``
                on every ablation arm; see :mod:`polyt5.training.cycle`.
            **kwargs: Forwarded to :class:`~polyt5.training.Trainer`
                (``val_loader``, ``run_dir``, ``tokenizer_path``,
                ``tokenizer_sha256``, ``run_config``, ``logger``).

        Raises:
            ValueError: If the arm enables cycle consistency but no
                ``cycle_loss`` was supplied, or supplies one for an arm that
                does not enable it.
        """
        if group_a.cycle_consistency and cycle_loss is None:
            raise ValueError(
                f"arm {group_a.arm!r} enables cycle_consistency but no cycle_loss was given"
            )
        if cycle_loss is not None and not group_a.cycle_consistency:
            raise ValueError(
                f"a cycle_loss was given for arm {group_a.arm!r}, which has "
                "cycle_consistency=False; an objective that is off must not be trained"
            )
        super().__init__(model, train_loader, config, **kwargs)
        self.group_a = group_a
        self.cycle_loss = cycle_loss

    def _to_device(self, batch: Batch) -> Batch:
        """Move tensor values to the device, passing anything else through."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _batch_weight(self, batch: Batch) -> int:
        """Weight the epoch mean by EXAMPLES; a regression batch has no tokens."""
        return int(batch["input_ids"].shape[0])

    def _task_of(self, batch: Batch) -> int:
        task_ids = batch["task_id"]
        unique = torch.unique(task_ids)
        if unique.numel() != 1:
            raise ValueError(
                f"every batch must be single-task, got task ids {unique.tolist()}; the "
                "loaders are built per task so a mixed batch is a plumbing bug"
            )
        return int(unique.item())

    def _forward_loss(self, batch: Batch) -> Tensor:
        """Route the batch to one of the model's three forward paths.

        Args:
            batch: A batch from :class:`polyt5.data.multitask.TaskCollator`.

        Returns:
            The scalar loss for this batch.
        """
        task = self._task_of(batch)
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        weights = batch["weights"] if self.group_a.reliability_weighting else None
        descriptors = batch["descriptor_targets"] if self.group_a.descriptors else None

        with torch.amp.autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            if task == GENERATION_TASK:
                generation_output = self.model.forward_generation(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                if generation_output.loss is None:
                    raise RuntimeError(
                        "forward_generation returned no loss; labels were not supplied"
                    )
                return generation_output.loss
            if task != PREDICTION_TASK:
                raise ValueError(f"unknown task id {task}")

            if self.group_a.regression_head:
                output = self.model.forward_regression(
                    batch["input_ids"],
                    batch["attention_mask"],
                    tg_targets=batch["tg_targets"],
                    descriptor_targets=descriptors,
                    weights=weights,
                )
            else:
                output = self.model.forward_text(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                    descriptor_targets=descriptors,
                    weights=weights,
                )
            if output.loss is None:
                raise RuntimeError(
                    "the prediction-path forward returned no loss; targets were not supplied"
                )
            loss = output.loss
            if self.cycle_loss is not None:
                loss = loss + self.group_a.cycle_weight * self.cycle_loss(batch["tg_targets"])
            return loss
