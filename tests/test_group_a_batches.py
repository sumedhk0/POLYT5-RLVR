# tests/test_group_a_batches.py
"""Group A Task 7: one batch shape for seven arms, and the ordering rules."""

from __future__ import annotations

import pytest
import torch

from polyt5.data.multitask import (
    GENERATION_TASK,
    PREDICTION_TASK,
    SplitTensors,
    TaskCollator,
    TaskDataset,
    TaskItem,
    assemble_split,
)
from polyt5.data.tg_metadata import TgExample, TgRow


class FakeTokenizer:
    """Character-level stand-in; ``polyt5.data`` never constructs a tokenizer."""

    pad_id = 0
    eos_id = 1

    def encode(self, text, *, add_eos=True, max_length=None, truncation=True):
        ids = [2 + (ord(ch) % 60) for ch in str(text)]
        if add_eos:
            ids.append(self.eos_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


def make(index: int, *, std: float = 0.0, reliability: str = "black") -> TgExample:
    return TgExample(
        pselfies=f"[At][C]{'[C]' * index}[At]",
        row=TgRow(
            psmiles="[*]CCO[*]" if index % 2 else "[*]CC(C)[*]",
            tg=300.0 + 10.0 * index,
            std=std,
            num_of_points=1,
            reliability=reliability,
            polymer_class="test",
            descriptors=(float(index), 5.0, float(index) * 2.0),
        ),
    )


NAMES = ["varies", "constant", "doubles"]


def build(**kwargs) -> SplitTensors:
    examples = [make(i) for i in range(10)]
    defaults = dict(
        train_indices=[0, 1, 2, 3, 4, 5],
        val_indices=[6, 7],
        test_indices=[8, 9],
        tokenizer=FakeTokenizer(),
        use_regression_head=True,
        use_descriptors=False,
        n_writings=1,
        use_reliability_weighting=False,
        std_floor=5.6,
        build_generation=False,
        seed=0,
    )
    defaults.update(kwargs)
    return assemble_split(examples, NAMES, **defaults)


def test_test_split_is_never_augmented_weighted_or_filtered():
    out = build(n_writings=4, use_reliability_weighting=True)
    assert out.test_pselfies == [make(8).pselfies, make(9).pselfies]
    assert out.test_tg == [380.0, 390.0]


def test_red_rows_leave_train_but_the_test_split_is_untouched():
    """Filtering test would break comparability with the frozen 28.67 K."""
    examples = [make(i) for i in range(10)]
    examples[2] = make(2, reliability="red")
    examples[9] = make(9, reliability="red")
    out = assemble_split(
        examples, NAMES, train_indices=[0, 1, 2, 3], val_indices=[4],
        test_indices=[8, 9], tokenizer=FakeTokenizer(), use_regression_head=True,
        use_descriptors=False, n_writings=1, use_reliability_weighting=False,
        std_floor=5.6, build_generation=False, seed=0,
    )
    assert out.n_dropped_red == 1
    assert out.n_train_polymers == 3
    assert len(out.test_pselfies) == 2, "test keeps its red row"
    assert out.n_red_in_test == 1


def test_target_standardizer_is_fitted_on_train_only():
    out = build()
    train_tg = [300.0 + 10.0 * i for i in range(6)]
    assert out.target_standardizer.mean[0] == pytest.approx(
        sum(train_tg) / len(train_tg)
    )


def test_standardised_targets_have_zero_mean_on_train():
    out = build()
    values = [item.tg_standardised for item in out.train]
    assert sum(values) / len(values) == pytest.approx(0.0, abs=1e-9)


def test_descriptor_columns_that_are_constant_on_train_are_dropped_and_named():
    out = build(use_descriptors=True)
    assert out.descriptor_standardizer is not None
    assert out.dropped_descriptor_columns == ("constant",)
    assert out.descriptor_standardizer.columns == ("varies", "doubles")
    assert all(len(item.descriptors) == 2 for item in out.train)


def test_descriptors_are_empty_when_the_switch_is_off():
    out = build(use_descriptors=False)
    assert out.descriptor_standardizer is None
    assert out.dropped_descriptor_columns == ()
    assert all(item.descriptors == () for item in out.train)


def test_weights_are_all_one_when_weighting_is_off():
    out = build(use_reliability_weighting=False)
    assert {item.weight for item in out.train} == {1.0}


def test_weights_are_inherited_by_every_writing_of_a_polymer():
    examples = [make(i, std=float(i) * 20.0) for i in range(6)]
    out = assemble_split(
        examples, NAMES, train_indices=[0, 1, 2, 3], val_indices=[4], test_indices=[5],
        tokenizer=FakeTokenizer(), use_regression_head=True, use_descriptors=False,
        n_writings=3, use_reliability_weighting=True, std_floor=5.6,
        build_generation=False, seed=0,
    )
    by_target = {}
    for item in out.train:
        by_target.setdefault(item.tg_standardised, set()).add(item.weight)
    assert all(len(weights) == 1 for weights in by_target.values())


def test_augmentation_grows_train_but_not_val():
    plain = build(n_writings=1)
    grown = build(n_writings=4)
    assert grown.n_train_writings >= plain.n_train_writings
    assert grown.n_train_polymers == plain.n_train_polymers == 6
    assert len(grown.val) == len(plain.val) == 2


def test_regression_arms_carry_no_text_labels_and_text_arms_carry_no_scalar_target():
    regression = build(use_regression_head=True)
    assert all(item.label_ids == () for item in regression.train)
    text = build(use_regression_head=False)
    assert all(item.label_ids for item in text.train)


def test_generation_items_are_built_only_when_asked_and_only_from_train():
    off = build(build_generation=False)
    assert off.train_generation == []
    on = build(build_generation=True)
    assert len(on.train_generation) == 6
    assert {item.task_id for item in on.train_generation} == {GENERATION_TASK}
    assert {item.task_id for item in on.train} == {PREDICTION_TASK}


def test_collator_emits_only_tensors_with_the_expected_keys():
    out = build(use_descriptors=True, n_writings=2, use_reliability_weighting=True)
    collator = TaskCollator(pad_id=0, max_source_length=200, max_target_length=200)
    batch = collator(out.train[:4])
    assert set(batch) == {
        "input_ids", "attention_mask", "labels", "tg_targets",
        "descriptor_targets", "weights", "task_id",
    }
    assert all(isinstance(value, torch.Tensor) for value in batch.values())
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["tg_targets"].dtype == torch.float32
    assert batch["descriptor_targets"].shape == (4, 2)
    assert batch["weights"].shape == (4,)
    assert batch["task_id"].tolist() == [PREDICTION_TASK] * 4


def test_collator_pads_labels_with_the_ignore_id():
    from polyt5.data.collate import LABEL_IGNORE_ID

    items = [
        TaskItem((5, 6, 1), (7, 1), 0.0, (), 1.0, PREDICTION_TASK),
        TaskItem((5, 1), (7, 8, 9, 1), 0.0, (), 1.0, PREDICTION_TASK),
    ]
    batch = TaskCollator(pad_id=0)(items)
    assert batch["labels"][0].tolist() == [7, 1, LABEL_IGNORE_ID, LABEL_IGNORE_ID]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]


def test_collator_handles_items_with_no_text_labels():
    items = [TaskItem((5, 6, 1), (), 0.5, (), 1.0, PREDICTION_TASK)]
    batch = TaskCollator(pad_id=0)(items)
    assert batch["labels"].shape == (1, 0)


def test_dataset_is_picklable_for_dataloader_workers():
    import pickle

    dataset = TaskDataset(build().train)
    clone = pickle.loads(pickle.dumps(dataset))
    assert len(clone) == len(dataset)
    assert clone[0] == dataset[0]
    assert dataset.stats["n_items"] == len(dataset)


def test_manifest_records_what_was_dropped():
    out = build(use_descriptors=True)
    manifest = out.to_manifest()
    assert manifest["dropped_descriptor_columns"] == ["constant"]
    assert manifest["n_train_polymers"] == 6
    assert "n_red_in_test" in manifest


# --- ADDENDUM: the test split must never be reliability-filtered ---------


def test_reliability_drop_guard_raises_when_asked_to_filter_the_test_split():
    """The guard prevents the call outright; it does not merely detect it
    after the fact. A future refactor that reuses this call site for test
    must fail loudly, naming the split, rather than silently shrinking it."""
    from polyt5.data.multitask import _drop_red_for_split

    examples = [make(i, reliability="red") for i in range(3)]
    with pytest.raises(ValueError, match="test"):
        _drop_red_for_split(examples, "test")


def test_test_split_length_is_unchanged_by_reliability_weighting_end_to_end():
    """Second, independent check that survives a future refactor of the
    guard itself: even with weighting on and red rows sitting in test, the
    assembled test split must keep every row it was given."""
    examples = [make(i) for i in range(10)]
    examples[8] = make(8, reliability="red")
    examples[9] = make(9, reliability="red")
    out = assemble_split(
        examples, NAMES, train_indices=[0, 1, 2, 3, 4, 5], val_indices=[6, 7],
        test_indices=[8, 9], tokenizer=FakeTokenizer(), use_regression_head=True,
        use_descriptors=False, n_writings=1, use_reliability_weighting=True,
        std_floor=5.6, build_generation=False, seed=0,
    )
    assert len(out.test_pselfies) == 2
    assert len(out.test_tg) == 2
    assert out.n_red_in_test == 2
