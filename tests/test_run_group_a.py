# tests/test_run_group_a.py
"""Group A Task 13: the runner's guards. Nothing here trains anything."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from polyt5.data.multitask import assemble_split
from polyt5.data.tg_metadata import TgExample, TgRow
from polyt5.evaluation import ArmResult
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import save_checkpoint
from polyt5.training.group_a import ARM_IDS
from polyt5.utils import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_group_a  # noqa: E402


def _tiny_model_config(tokenizer: PolyT5Tokenizer) -> PolyT5Config:
    """A deliberately tiny model, wired to the real tokenizer's ids.

    Mirrors the pattern in ``tests/test_predictor.py``: CPU-fast, no training,
    just enough of a real ``PolyT5ForConditionalGeneration`` to exercise
    ``build_arm_model`` / ``_score_split`` without a paper-sized model.
    """
    return PolyT5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        d_kv=8,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        n_positions=32,
        dropout_rate=0.0,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
    )


def _tiny_model_yaml(path: Path) -> Path:
    """The subset of ``_tiny_model_config`` that a YAML-driven build needs.

    ``build_arm_model``'s no-checkpoint path overwrites vocab/pad/eos/decoder
    ids from the tokenizer regardless, so those are omitted here.
    """
    payload = {
        "d_model": 16, "d_kv": 8, "num_heads": 2, "d_ff": 32,
        "num_layers": 1, "n_positions": 32, "dropout_rate": 0.0,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class _FakeTokenizer:
    """Character-level stand-in for ``assemble_split``, not a real tokenizer.

    Mirrors ``tests/test_group_a_batches.py``'s fixture: ``polyt5.data``
    never constructs one itself.
    """

    pad_id = 0
    eos_id = 1

    def encode(self, text, *, add_eos=True, max_length=None, truncation=True):
        ids = [2 + (ord(ch) % 60) for ch in str(text)]
        if add_eos:
            ids.append(self.eos_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


_DESCRIPTOR_NAMES = ["varies", "doubles"]


def _synthetic_example(index: int) -> TgExample:
    return TgExample(
        pselfies=f"[At][C]{'[C]' * index}[At]",
        row=TgRow(
            psmiles="[*]CCO[*]" if index % 2 else "[*]CC(C)[*]",
            tg=300.0 + 10.0 * index,
            std=0.0,
            num_of_points=1,
            reliability="black",
            polymer_class="test",
            descriptors=(float(index), float(index) * 2.0),
        ),
    )


def write_splits(path: Path, *, n: int = 10, n_splits: int = 2) -> Path:
    splits = []
    for index in range(n_splits):
        order = [(i + index) % n for i in range(n)]
        splits.append(
            {"index": index, "train": order[:6], "val": order[6:8], "test": order[8:]}
        )
    path.write_text(
        json.dumps({"task": "tg_prediction", "n": n, "base_seed": 0,
                    "n_splits": n_splits, "splits": splits}),
        encoding="utf-8",
    )
    return path


def test_frozen_splits_load_and_validate(tmp_path):
    path = write_splits(tmp_path / "splits.json")
    splits = run_group_a.load_frozen_splits(path, n_examples=10)
    assert [s.index for s in splits] == [0, 1]
    # Contents, not just shape: "loaded verbatim" means these exact indices,
    # not merely a train/val/test list of the right lengths. Split 0's train
    # ([0..5]) happens to coincide with range(6), so split 1 -- whose train is
    # [1..6], NOT range(6) -- is what actually proves the indices survived.
    assert splits[0].train == [0, 1, 2, 3, 4, 5]
    assert splits[0].val == [6, 7]
    assert splits[0].test == [8, 9]
    assert splits[1].train == [1, 2, 3, 4, 5, 6]
    assert splits[1].val == [7, 8]
    assert splits[1].test == [9, 0]


def test_a_corpus_of_the_wrong_size_is_refused(tmp_path):
    """The one guard that keeps every Group A number comparable to 28.67 K."""
    path = write_splits(tmp_path / "splits.json", n=10)
    with pytest.raises(ValueError, match="was built over"):
        run_group_a.load_frozen_splits(path, n_examples=9)


def test_a_recorded_size_mismatch_is_refused_even_when_coverage_would_pass(tmp_path):
    """Isolates the SIZE guard from the COVERAGE guard.

    This split's train/val/test union is exactly range(9) -- coverage holds
    for n_examples=9 -- so only the recorded-``n`` check can catch that the
    file itself declares it was built over 10 examples. Without this test,
    deleting the size guard entirely is invisible: the sibling test above
    still fails for a DIFFERENT reason (coverage), so it cannot tell the two
    guards apart.
    """
    path = tmp_path / "mismatched_but_covering.json"
    path.write_text(
        json.dumps({
            "n": 10, "n_splits": 1,
            "splits": [{"index": 0, "train": [0, 1, 2, 3, 4, 5, 6], "val": [7], "test": [8]}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="was built over"):
        run_group_a.load_frozen_splits(path, n_examples=9)


def test_overlapping_train_and_test_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"n": 4, "n_splits": 1,
                    "splits": [{"index": 0, "train": [0, 1, 2], "val": [], "test": [2, 3]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_a_split_that_does_not_cover_every_index_is_refused(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(
        json.dumps({"n": 4, "n_splits": 1,
                    "splits": [{"index": 0, "train": [0, 1], "val": [], "test": [2]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cover"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_a_file_with_no_splits_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"n": 4, "splits": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no splits"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_config_fingerprint_differs_for_different_hyperparameters():
    """Finding 4: a resume cache keyed only on arm/split cannot tell apart

    two runs of the same arm with different hyperparameters (e.g. the
    brief's own ``--arm A3 --set group_a.n_writings=8`` example vs the
    default ``n_writings=4``) -- one would silently reuse or overwrite the
    other's ``results.json``. The fingerprint folded into the run-directory
    name must differ whenever the effective config differs, and must be
    stable for the identical config so a genuine resume still finds its own
    cache.
    """
    default_a3 = run_group_a.resolve_arms(["A3"])[0]
    overridden_a3 = run_group_a.resolve_arms(["A3"], n_writings=8)[0]
    assert default_a3.n_writings != overridden_a3.n_writings  # fixture sanity

    default_fp = run_group_a._config_fingerprint(default_a3)
    overridden_fp = run_group_a._config_fingerprint(overridden_a3)
    assert default_fp != overridden_fp

    # Determinism: re-resolving the SAME config reproduces the SAME
    # fingerprint, so a resumed run actually finds the cache it wrote.
    again = run_group_a.resolve_arms(["A3"])[0]
    assert run_group_a._config_fingerprint(again) == default_fp

    # Different ARMS with otherwise-identical hyperparameters must not
    # collide either -- the cache key is arm-scoped by directory already,
    # but the fingerprint itself should not conflate two switch tables.
    default_a4 = run_group_a.resolve_arms(["A4"])[0]
    assert run_group_a._config_fingerprint(default_a4) != default_fp


def test_resolve_arms_defaults_to_all_seven_in_order():
    assert [c.arm for c in run_group_a.resolve_arms(None)] == list(ARM_IDS)


def test_resolve_arms_honours_a_subset_and_deduplicates():
    assert [c.arm for c in run_group_a.resolve_arms(["A3", "B0", "A3"])] == ["B0", "A3"]


def test_resolve_arms_passes_hyperparameter_overrides_through():
    configs = run_group_a.resolve_arms(["A2", "A3"], descriptor_lambda=0.4, n_writings=6)
    assert configs[0].descriptor_lambda == 0.4
    assert configs[1].n_writings == 6


def test_resolve_arms_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="A9"):
        run_group_a.resolve_arms(["A9"])


def test_baseline_reference_comes_from_the_frozen_artifact(tmp_path):
    # Sentinel values, deliberately NOT 28.6733/0.7591: a stub implementation
    # that just returns the real frozen numbers verbatim (never reading the
    # file at all) would otherwise pass this test by coincidence.
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps({"tg_prediction_5split": {"mae": {"mean": 11.5, "std": 2.25}}}),
        encoding="utf-8",
    )
    assert run_group_a.load_baseline_reference(path) == (11.5, 2.25)


def test_baseline_reference_refuses_an_artifact_without_the_key(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"tg_prediction_5split": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mae"):
        run_group_a.load_baseline_reference(path)


def test_the_real_frozen_artifact_still_says_what_the_plan_assumes():
    repo = Path(__file__).resolve().parents[1]
    artifact = repo / "artifacts" / "baseline" / "frozen_baseline.json"
    if not artifact.is_file():
        pytest.skip("frozen baseline artifact missing")
    mean, std = run_group_a.load_baseline_reference(artifact)
    assert mean == pytest.approx(28.6733, abs=1e-4)
    assert std == pytest.approx(0.7591, abs=1e-4)


def test_default_output_root_is_not_a_live_run_directory():
    args = run_group_a.parse_args([])
    assert Path(args.out).as_posix().endswith("results/group_a")
    assert "grpo" not in Path(args.out).as_posix()


def test_arms_can_be_selected_on_the_command_line():
    args = run_group_a.parse_args(["--arm", "A1", "--arm", "A4"])
    assert args.arm == ["A1", "A4"]


# --------------------------------------------------------------------------
# Coverage for build_arm_model / _score_split: real model construction and
# real forward passes on a tiny (CPU-fast) polyT5, no training loop, no
# ablation run. This is the review's coverage gap -- everything below is
# still zero training steps and writes nothing under results/.
# --------------------------------------------------------------------------


def test_build_arm_model_wires_the_descriptor_head_width_to_the_switch(tmp_path):
    tokenizer = PolyT5Tokenizer.default()
    model_config_path = _tiny_model_yaml(tmp_path / "model.yaml")
    logger = get_logger("test.run_group_a.build_arm_model")

    descriptors_on = run_group_a.arm_config("A2")   # descriptors on, no regression head
    model_on, head_on, pretrained_on = run_group_a.build_arm_model(
        descriptors_on, 37, model_config_path=model_config_path, init_checkpoint=None,
        tokenizer=tokenizer, logger=logger, device="cpu",
    )
    descriptors_off = run_group_a.arm_config("A1")  # regression head, descriptors off
    model_off, head_off, pretrained_off = run_group_a.build_arm_model(
        descriptors_off, 37, model_config_path=model_config_path, init_checkpoint=None,
        tokenizer=tokenizer, logger=logger, device="cpu",
    )

    assert pretrained_on is False
    assert pretrained_off is False
    # The descriptor head's width comes from n_descriptors ONLY when the
    # switch is on -- a mutation that always passed n_descriptors through (or
    # always passed 0) would be caught by one of these two arms.
    assert head_on.n_descriptors == 37
    assert model_on.descriptor_head is not None
    assert head_off.n_descriptors == 0
    assert model_off.descriptor_head is None
    assert head_off.use_regression_head is True
    assert model_off.tg_head is not None


def test_build_arm_model_refuses_a_tokenizer_mismatched_checkpoint(tmp_path):
    tokenizer = PolyT5Tokenizer.default()
    model_config_path = _tiny_model_yaml(tmp_path / "model.yaml")
    backbone = PolyT5ForConditionalGeneration(_tiny_model_config(tokenizer))
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        checkpoint_path, model=backbone, epoch=0, global_step=0,
        config={}, model_config=backbone.config.to_dict(),
        tokenizer_sha256="0" * 64,  # not this tokenizer's real hash
    )
    logger = get_logger("test.run_group_a.build_arm_model_mismatch")
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        run_group_a.build_arm_model(
            run_group_a.arm_config("B0"), 0, model_config_path=model_config_path,
            init_checkpoint=checkpoint_path, tokenizer=tokenizer, logger=logger,
            device="cpu",
        )


def test_build_arm_model_warm_starts_from_the_checkpoints_actual_weights(tmp_path):
    tokenizer = PolyT5Tokenizer.default()
    model_config_path = _tiny_model_yaml(tmp_path / "model.yaml")
    source = PolyT5ForConditionalGeneration(_tiny_model_config(tokenizer))
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        checkpoint_path, model=source, epoch=0, global_step=0,
        config={}, model_config=source.config.to_dict(),
        tokenizer_sha256=tokenizer.sha256,
    )
    logger = get_logger("test.run_group_a.build_arm_model_warm_start")
    model, _, pretrained = run_group_a.build_arm_model(
        run_group_a.arm_config("B0"), 0, model_config_path=model_config_path,
        init_checkpoint=checkpoint_path, tokenizer=tokenizer, logger=logger,
        device="cpu",
    )
    assert pretrained is True
    # The loaded backbone must carry the CHECKPOINT's weights, not a fresh
    # random init from model_config_path -- a mutation that ignored
    # init_checkpoint and always built from YAML would give a different
    # (freshly initialised) embedding table here.
    assert torch.equal(model.backbone.shared.weight.detach(), source.shared.weight.detach())


def test_score_split_regression_head_arm_scores_via_the_regression_predictor(tmp_path):
    tokenizer = PolyT5Tokenizer.default()
    model_config_path = _tiny_model_yaml(tmp_path / "model.yaml")
    logger = get_logger("test.run_group_a.score_split_regression")

    group_a = run_group_a.arm_config("A1")  # regression head, no descriptors
    model, _, _ = run_group_a.build_arm_model(
        group_a, 0, model_config_path=model_config_path, init_checkpoint=None,
        tokenizer=tokenizer, logger=logger, device="cpu",
    )
    # Pin the head to emit a KNOWN Kelvin value regardless of input: zero the
    # projection weight (a constant function of the pooled state), fix the
    # bias, and record a known (mean, std) scaling. If _score_split ever took
    # the WRONG path for this arm -- beam-decoding a random untrained model
    # instead of calling the regression head -- the resulting MAE would not
    # land on exactly 0.0.
    with torch.no_grad():
        model.tg_head.projection.weight.zero_()
        model.tg_head.projection.bias.fill_(0.5)
    model.set_target_scaling(mean=100.0, std=50.0)
    expected_kelvin = 0.5 * 50.0 + 100.0  # == 125.0

    tensors = SimpleNamespace(
        test_pselfies=["[At][C][C][At]", "[At][C][O][C][At]", "[At][C][C][C][At]"],
        test_tg=[expected_kelvin, expected_kelvin, expected_kelvin],
    )
    cfg = {"data": {"max_length": 32}, "evaluation": {"batch_size": 2}}
    report, predictions = run_group_a._score_split(
        model, tokenizer, group_a, tensors, cfg=cfg, device="cpu", logger=logger,
    )
    assert predictions == ["125.0", "125.0", "125.0"]
    assert report.n_total == 3
    assert report.n_valid_numeric == 3
    assert report.mae == pytest.approx(0.0, abs=1e-6)


def test_score_split_text_head_arm_scores_via_beam_search(tmp_path):
    """B0 has no regression head, so routing it through
    RegressionPropertyPredictor would raise before producing any report --
    exactly what would happen if a mutation swapped the branch condition.
    """
    tokenizer = PolyT5Tokenizer.default()
    model_config_path = _tiny_model_yaml(tmp_path / "model.yaml")
    logger = get_logger("test.run_group_a.score_split_text")

    group_a = run_group_a.arm_config("B0")  # every switch off: pure text decode
    model, _, _ = run_group_a.build_arm_model(
        group_a, 0, model_config_path=model_config_path, init_checkpoint=None,
        tokenizer=tokenizer, logger=logger, device="cpu",
    )
    assert model.tg_head is None  # this arm structurally cannot use the regression path

    tensors = SimpleNamespace(
        test_pselfies=["[At][C][C][At]", "[At][C][O][C][At]"],
        test_tg=[236.0, 180.0],
    )
    cfg = {
        "data": {"max_length": 32},
        "evaluation": {
            "batch_size": 2, "beam_width": 2, "max_target_length": 8, "length_penalty": 1.0,
        },
    }
    report, predictions = run_group_a._score_split(
        model, tokenizer, group_a, tensors, cfg=cfg, device="cpu", logger=logger,
    )
    assert len(predictions) == 2
    assert report.n_total == 2


# --------------------------------------------------------------------------
# Whole-branch review round: findings 5 and 2. Neither runs any training --
# _reference_step_budget is a pure function, and n_train_polymers-invariance
# is checked via real (fast, tiny) assemble_split calls, not a live run.
# --------------------------------------------------------------------------


def test_reference_step_budget_is_a_pure_function_of_its_inputs():
    """Direct unit test of the formula: ceil(polymers/batch) steps/epoch,
    times the epoch count -- no arm identity enters the computation at all,
    which is exactly why it produces the same number for every arm sharing
    a split's n_train_polymers.
    """
    # 5293 polymers / batch 16 = 330.8125 -> 331 micro-batches/epoch (ceil),
    # grad_accum 1 -> 331 steps/epoch, x 30 epochs.
    assert run_group_a._reference_step_budget(
        5293, batch_size=16, gradient_accumulation_steps=1, epochs=30
    ) == 331 * 30
    # gradient accumulation divides the micro-batch count again.
    assert run_group_a._reference_step_budget(
        5293, batch_size=16, gradient_accumulation_steps=2, epochs=30
    ) == 166 * 30


def test_step_budget_is_equal_across_b0_a3_a5_a6_on_the_same_split():
    """Finding 5 verification, end to end from the same building blocks
    main() uses -- no training, no model, just assemble_split + the pure
    budget formula.

    tensors.n_train_polymers is the ONLY per-arm input to the step budget
    besides the shared batch_size/grad_accum/epochs, and it is arm-invariant
    (the red drop in assemble_split runs unconditionally for every arm). So
    if n_train_polymers matches across B0/A3/A5/A6, their step budgets are
    equal BY CONSTRUCTION -- this test demonstrates that invariant on real
    (tiny, synthetic) data rather than asserting it as a given.
    """
    examples = [_synthetic_example(i) for i in range(8)]
    tokenizer = _FakeTokenizer()
    arms = {arm: run_group_a.arm_config(arm) for arm in ("B0", "A3", "A5", "A6")}

    n_train_polymers: dict[str, int] = {}
    for name, group_a in arms.items():
        tensors = assemble_split(
            examples, _DESCRIPTOR_NAMES,
            train_indices=[0, 1, 2, 3, 4, 5], val_indices=[6], test_indices=[7],
            tokenizer=tokenizer,
            use_regression_head=group_a.regression_head,
            use_descriptors=group_a.descriptors,
            n_writings=group_a.effective_n_writings(),
            use_reliability_weighting=group_a.reliability_weighting,
            std_floor=group_a.std_floor,
            build_generation=group_a.multitask,
            seed=0,
        )
        n_train_polymers[name] = tensors.n_train_polymers

    assert len(set(n_train_polymers.values())) == 1, (
        f"n_train_polymers must be arm-invariant per split, got {n_train_polymers}"
    )

    budgets = {
        name: run_group_a._reference_step_budget(
            count, batch_size=2, gradient_accumulation_steps=1, epochs=30
        )
        for name, count in n_train_polymers.items()
    }
    assert len(set(budgets.values())) == 1, f"step budgets must be equal, got {budgets}"
    assert budgets["B0"] == budgets["A3"] == budgets["A5"] == budgets["A6"]


def test_config_fingerprint_folds_in_max_steps_and_only_split():
    """Finding 2: a --max-steps smoke run must not share a cache key with a
    real run of the same arm. Both --max-steps and --only-split are OUTSIDE
    GroupAConfig, so they cannot be caught by the round-1 fix that folded
    group_a.to_dict() into the fingerprint -- they need their own inputs.
    """
    group_a = run_group_a.arm_config("B0")
    real_run = run_group_a._config_fingerprint(group_a, max_steps=None, only_split=None)
    smoke_run = run_group_a._config_fingerprint(group_a, max_steps=20, only_split=0)
    different_smoke_cap = run_group_a._config_fingerprint(
        group_a, max_steps=50, only_split=0
    )
    only_split_alone = run_group_a._config_fingerprint(
        group_a, max_steps=None, only_split=0
    )

    assert real_run != smoke_run
    assert smoke_run != different_smoke_cap
    assert real_run != only_split_alone
    # Determinism: identical arguments still reproduce the identical
    # fingerprint, or a genuine resume of a real run could never find its
    # own cache.
    assert real_run == run_group_a._config_fingerprint(
        group_a, max_steps=None, only_split=None
    )


# --------------------------------------------------------------------------
# Task 13 finding 9 (routed to the whole-branch review): compare B0's
# in-harness rerun to the frozen number, instead of only labelling the
# frozen number "baseline B0" and never checking it against the real rerun.
# --------------------------------------------------------------------------


def _arm_result(arm: str, mae_mean: float | None, *, n_splits: int = 5) -> ArmResult:
    return ArmResult(
        arm=arm, switches={}, n_splits=n_splits, mae_mean=mae_mean, mae_std=0.1,
        rmse_mean=None, rmse_std=None, r2_mean=None, r2_std=None,
        non_numeric_rate_mean=0.0,
    )


def test_b0_vs_frozen_reports_not_run_when_b0_is_excluded():
    results = [_arm_result("A3", 27.0)]
    comparison = run_group_a._b0_vs_frozen(results, baseline_mean=28.6733, baseline_std=0.7591)
    assert comparison["status"] == "not_run"
    assert comparison["delta"] is None


def test_b0_vs_frozen_reports_no_evaluable_split_when_b0_never_scored():
    results = [_arm_result("B0", None, n_splits=0)]
    comparison = run_group_a._b0_vs_frozen(results, baseline_mean=28.6733, baseline_std=0.7591)
    assert comparison["status"] == "no_evaluable_split"


def test_b0_vs_frozen_flags_a_harness_divergence():
    """A B0 rerun landing outside the frozen std must be flagged, not merged
    silently into "the baseline" the way format_ablation_matrix's header
    text alone would read.
    """
    within = run_group_a._b0_vs_frozen(
        [_arm_result("B0", 29.0)], baseline_mean=28.6733, baseline_std=0.7591
    )
    assert within["status"] == "compared"
    assert within["diverges"] is False

    outside = run_group_a._b0_vs_frozen(
        [_arm_result("B0", 35.0)], baseline_mean=28.6733, baseline_std=0.7591
    )
    assert outside["status"] == "compared"
    assert outside["diverges"] is True
    assert outside["delta"] == pytest.approx(35.0 - 28.6733)
