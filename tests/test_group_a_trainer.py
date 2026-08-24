# tests/test_group_a_trainer.py
"""Group A Task 9: alternating batches, and one loss router for seven arms."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polyt5.data.multitask import GENERATION_TASK, PREDICTION_TASK
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.training import Trainer, TrainerConfig
from polyt5.training.group_a import arm_config
from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"


def build_model(**kwargs) -> PolyT5MultiTask:
    seed_everything(0)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    return PolyT5MultiTask(
        PolyT5ForConditionalGeneration(config), MultiTaskConfig(head_dropout=0.0, **kwargs)
    )


def make_batch(
    task_id: int, *, n: int = 2, n_descriptors: int = 0, text: bool = True, tag: int = 0
):
    """Build a batch.

    ``tag`` is a plain int carried through, unused by any forward path -- it
    exists so loader-ordering tests can tell WHICH batch was yielded, not just
    how many: every batch of a given ``task_id`` is otherwise byte-identical
    (the generator is seeded from ``task_id`` alone).
    """
    generator = torch.Generator().manual_seed(17 + task_id)
    return {
        "input_ids": torch.randint(2, 458, (n, 5), generator=generator),
        "attention_mask": torch.ones(n, 5, dtype=torch.long),
        "labels": (
            torch.randint(2, 458, (n, 3), generator=generator)
            if text
            else torch.zeros(n, 0, dtype=torch.long)
        ),
        "tg_targets": torch.zeros(n, dtype=torch.float32),
        "descriptor_targets": torch.zeros(n, n_descriptors, dtype=torch.float32),
        "weights": torch.ones(n, dtype=torch.float32),
        "task_id": torch.full((n,), task_id, dtype=torch.long),
        "tag": tag,
    }


def trainer_config(**kwargs) -> TrainerConfig:
    defaults = dict(
        max_epochs=1, physical_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=3e-4, weight_decay=0.01, scheduler="constant", amp=False,
        device="cpu", log_every=1000,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


# ------------------------------------------------------- the base-class refactor
def test_base_trainer_batch_weight_still_counts_label_tokens():
    """The extraction must not change what the base trainer measures."""
    model = PolyT5ForConditionalGeneration(PolyT5Config.from_yaml(TINY_YAML))
    trainer = Trainer(model, [], trainer_config())
    batch = {"labels": torch.tensor([[1, 2, -100], [3, -100, -100]])}
    assert trainer._batch_weight(batch) == 3


# --------------------------------------------------------------- interleaving
def test_interleaved_loader_alternates_prediction_and_generation():
    prediction = [make_batch(PREDICTION_TASK, tag=i) for i in range(3)]
    generation = [make_batch(GENERATION_TASK, tag=i) for i in range(3)]
    emitted = [
        (int(batch["task_id"][0]), int(batch["tag"]))
        for batch in InterleavedLoader(prediction, generation)
    ]
    # Not just the task-id sequence: WHICH prediction/generation batch lands
    # at each position, so a loader that repeats or drops an item is caught.
    assert emitted == [
        (PREDICTION_TASK, 0), (GENERATION_TASK, 0),
        (PREDICTION_TASK, 1), (GENERATION_TASK, 1),
        (PREDICTION_TASK, 2), (GENERATION_TASK, 2),
    ]
    assert len(InterleavedLoader(prediction, generation)) == 6


def test_interleaved_loader_without_generation_is_a_passthrough():
    prediction = [make_batch(PREDICTION_TASK, tag=i) for i in range(4)]
    loader = InterleavedLoader(prediction, None)
    assert len(loader) == 4
    assert [int(b["tag"]) for b in loader] == [0, 1, 2, 3]


def test_interleaved_loader_empty_generation_list_is_also_a_passthrough():
    """An empty (not None) generation loader must degrade the same way.

    Regression for a real bug: __len__ special-cased ``n_generation == 0``
    but __iter__ did not, so InterleavedLoader(pred, []) reported a length of
    ``2 * len(pred)`` and then raised RuntimeError on the first ``next()`` of
    an exhausted ``cycle([])``.
    """
    prediction = [make_batch(PREDICTION_TASK, tag=i) for i in range(3)]
    loader = InterleavedLoader(prediction, [])
    assert len(loader) == 3
    assert [int(b["tag"]) for b in loader] == [0, 1, 2]


def test_interleaved_loader_cycles_the_shorter_side():
    prediction = [make_batch(PREDICTION_TASK, tag=i) for i in range(4)]
    generation = [make_batch(GENERATION_TASK, tag=i) for i in range(2)]
    batches = list(InterleavedLoader(prediction, generation))
    assert len(batches) == 8
    prediction_tags = [int(b["tag"]) for b in batches if int(b["task_id"][0]) == PREDICTION_TASK]
    generation_tags = [int(b["tag"]) for b in batches if int(b["task_id"][0]) == GENERATION_TASK]
    # The longer side (4 items) is consumed once, in order; the shorter side
    # (2 items) genuinely CYCLES -- [0, 1, 0, 1] -- rather than truncating to
    # its own length or repeating a single item four times.
    assert prediction_tags == [0, 1, 2, 3]
    assert generation_tags == [0, 1, 0, 1]


def test_interleaved_loader_is_reiterable():
    loader = InterleavedLoader(
        [make_batch(PREDICTION_TASK, tag=0)], [make_batch(GENERATION_TASK, tag=0)]
    )
    expected = [PREDICTION_TASK, GENERATION_TASK]
    assert [int(b["task_id"][0]) for b in loader] == expected
    assert [int(b["task_id"][0]) for b in loader] == expected


# ------------------------------------------------------------- the loss router
def test_regression_arm_uses_the_scalar_head():
    """A1 must route through forward_regression, not merely produce a finite loss.

    ``requires_grad``/``isfinite`` alone would also hold if the switch were
    ignored and the text head ran instead -- pin against the scalar head's own
    loss instead, the way the B0 and generation tests pin against the backbone.
    """
    model = build_model(use_regression_head=True)
    model.eval()
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    batch = make_batch(PREDICTION_TASK, text=False)
    reference = model.forward_regression(
        batch["input_ids"], batch["attention_mask"], tg_targets=batch["tg_targets"]
    ).loss
    loss = trainer._forward_loss(batch)
    assert loss.requires_grad
    assert float(loss) == pytest.approx(float(reference), abs=1e-6)


def test_text_arm_matches_the_backbone_loss_exactly():
    """B0 reproduces the baseline objective, it does not approximate it."""
    model = build_model()
    model.eval()
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("B0"))
    batch = make_batch(PREDICTION_TASK)
    reference = model.backbone(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss
    assert float(trainer._forward_loss(batch)) == pytest.approx(float(reference), abs=1e-6)


def test_generation_batches_route_to_the_decoder():
    model = build_model(use_regression_head=True)
    model.eval()
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A6"))
    batch = make_batch(GENERATION_TASK)
    reference = model.backbone(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss
    assert float(trainer._forward_loss(batch)) == pytest.approx(float(reference), abs=1e-6)


def test_descriptor_arm_adds_a_term_to_the_text_loss():
    plain = build_model()
    plain.eval()
    with_descriptors = build_model(n_descriptors=3, descriptor_lambda=1.0)
    with_descriptors.eval()
    batch = make_batch(PREDICTION_TASK, n_descriptors=3)
    base = GroupATrainer(plain, [], trainer_config(), group_a=arm_config("B0"))
    aux = GroupATrainer(
        with_descriptors, [], trainer_config(), group_a=arm_config("A2")
    )
    assert float(aux._forward_loss(batch)) != pytest.approx(float(base._forward_loss(batch)))


def test_batch_weight_counts_examples_not_tokens():
    """Must count the batch axis, not the sequence axis.

    ``n`` is deliberately different from ``make_batch``'s hard-coded source
    length (5): with ``n=5`` a ``shape[0]`` vs ``shape[1]`` bug is invisible
    because both axes are 5. Pinning ``n=3`` against seq-len 5 means a
    ``shape[1]`` mutant returns 5, not 3, and the assertion below catches it.
    """
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    batch = make_batch(PREDICTION_TASK, n=3, text=False)
    assert batch["input_ids"].shape[0] != batch["input_ids"].shape[1]  # axes must differ
    assert trainer._batch_weight(batch) == 3


def test_to_device_tolerates_a_non_tensor_value():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    moved = trainer._to_device({**make_batch(PREDICTION_TASK, text=False), "arm": "A1"})
    assert moved["arm"] == "A1"
    assert moved["input_ids"].device.type == "cpu"


def test_a_mixed_task_batch_is_refused():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A6"))
    batch = make_batch(PREDICTION_TASK, n=2)
    batch["task_id"] = torch.tensor([PREDICTION_TASK, GENERATION_TASK])
    with pytest.raises(ValueError, match="single-task"):
        trainer._forward_loss(batch)


def test_one_epoch_runs_end_to_end_and_moves_the_weights():
    model = build_model(use_regression_head=True, n_descriptors=3)
    before = model.tg_head.projection.weight.detach().clone()
    loader = InterleavedLoader(
        [make_batch(PREDICTION_TASK, n_descriptors=3, text=False) for _ in range(4)],
        [make_batch(GENERATION_TASK) for _ in range(4)],
    )
    trainer = GroupATrainer(model, loader, trainer_config(), group_a=arm_config("A6"))
    metrics = trainer.train()
    assert metrics["global_step"] == 8
    assert not torch.allclose(before, model.tg_head.projection.weight)
