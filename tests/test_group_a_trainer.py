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


def make_batch(task_id: int, *, n: int = 2, n_descriptors: int = 0, text: bool = True):
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
    prediction = [make_batch(PREDICTION_TASK) for _ in range(3)]
    generation = [make_batch(GENERATION_TASK) for _ in range(3)]
    tasks = [
        int(batch["task_id"][0]) for batch in InterleavedLoader(prediction, generation)
    ]
    assert tasks == [PREDICTION_TASK, GENERATION_TASK] * 3
    assert len(InterleavedLoader(prediction, generation)) == 6


def test_interleaved_loader_without_generation_is_a_passthrough():
    prediction = [make_batch(PREDICTION_TASK) for _ in range(4)]
    loader = InterleavedLoader(prediction, None)
    assert len(loader) == 4
    assert all(int(b["task_id"][0]) == PREDICTION_TASK for b in loader)


def test_interleaved_loader_cycles_the_shorter_side():
    prediction = [make_batch(PREDICTION_TASK) for _ in range(4)]
    generation = [make_batch(GENERATION_TASK)]
    assert len(list(InterleavedLoader(prediction, generation))) == 8


def test_interleaved_loader_is_reiterable():
    loader = InterleavedLoader([make_batch(PREDICTION_TASK)], [make_batch(GENERATION_TASK)])
    assert [int(b["task_id"][0]) for b in loader] == [int(b["task_id"][0]) for b in loader]


# ------------------------------------------------------------- the loss router
def test_regression_arm_uses_the_scalar_head():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    loss = trainer._forward_loss(make_batch(PREDICTION_TASK, text=False))
    assert loss.requires_grad
    assert torch.isfinite(loss)


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
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    assert trainer._batch_weight(make_batch(PREDICTION_TASK, n=5, text=False)) == 5


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
