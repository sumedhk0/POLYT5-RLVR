"""Tests for the sampling-hyperparameter sweep (Arm B of the RLVR comparison).

Everything here is CPU-only and either uses the deliberately tiny debug model
from ``configs/model/polyt5_tiny.yaml`` or a hand-built fake model whose logits
the test controls, so the whole file runs in seconds and never needs a
checkpoint.

Ground truth for the paper's axes (Sahu et al., npj Artificial Intelligence
2026): ``top_p`` in {0.75, 0.95}, temperature 0.1 -> 2.0 in increments of 0.1,
fine-tuning epochs 1 -> 15, 10,000 polymers per configuration -- 3 model sizes
x 15 epochs x 20 temperatures x 2 top_p = 1,800 configurations. The paper
publishes those per-configuration counts only as unlabeled heatmaps, so the
numeric table this module produces has no published counterpart to diff
against; what *is* checkable is that our axes are the paper's axes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from polyt5.evaluation.sweep import (  # noqa: E402
    DEFAULT_GRID,
    PAPER_GRID,
    PAPER_SAMPLES_PER_CONFIG,
    PAPER_TEMPERATURES,
    PAPER_TOP_PS,
    SweepPoint,
    SweepResult,
    append_result_jsonl,
    assign_targets,
    generate_candidates,
    read_results_jsonl,
    run_sweep_point,
    select_best,
    sweep_grid,
    sweep_to_dataframe,
    sweep_to_markdown,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "model"


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> PolyT5Tokenizer:
    """The built-in 458-token vocabulary (no artifact file required)."""
    return PolyT5Tokenizer.default()


@pytest.fixture()
def tiny_model(tokenizer: PolyT5Tokenizer) -> PolyT5ForConditionalGeneration:
    config = PolyT5Config.from_yaml(CONFIG_DIR / "polyt5_tiny.yaml")
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_id
    config.eos_token_id = tokenizer.eos_id
    config.decoder_start_token_id = tokenizer.decoder_start_token_id
    torch.manual_seed(0)
    model = PolyT5ForConditionalGeneration(config)
    model.eval()
    return model


class _FakeEncoderDecoder(torch.nn.Module):
    """Base for fake models that satisfy the contract :func:`generate` needs.

    :func:`polyt5.generation.generate` only ever calls ``encode`` and
    ``decode_step`` on the KV-cached path, so a fake needs nothing else. The
    encoder output is a dummy ``(batch, 1, 1)`` tensor whose only job is to
    carry the batch size.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return torch.zeros(input_ids.shape[0], 1, 1)

    def _step_logits(self, batch: int, step: int) -> torch.Tensor:
        raise NotImplementedError

    def decode_step(
        self,
        step_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
    ):
        step = 0 if past_key_values is None else int(past_key_values) + 1
        batch = encoder_hidden_states.shape[0]
        return self._step_logits(batch, step)[:, None, :], step


class ConstantLogitModel(_FakeEncoderDecoder):
    """A model whose next-token distribution is a fixed, controlled vector.

    The support is restricted to a handful of SELFIES tokens so that *every*
    sampled string decodes to some molecule; what temperature changes is how
    often the same molecule comes back. That is exactly the axis the
    duplication half of the paper's trade-off lives on.
    """

    def __init__(self, vocab_size: int, logits: Mapping[int, float], floor: float = -30.0) -> None:
        super().__init__(vocab_size)
        vector = torch.full((vocab_size,), floor)
        for token_id, value in logits.items():
            vector[token_id] = value
        self._vector = vector

    def _step_logits(self, batch: int, step: int) -> torch.Tensor:
        return self._vector[None, :].expand(batch, self.vocab_size)


class ScriptedModel(_FakeEncoderDecoder):
    """A model that emits one fixed token sequence, whatever the temperature.

    The scripted token gets a logit of ``+50`` against a floor of ``-50``, so
    even at the sweep's hottest temperature (2.0) the softmax mass on the
    scripted token is ``1 - O(1e-21)``. That makes the whole cascade -- and
    therefore the property columns -- exactly predictable.
    """

    def __init__(self, vocab_size: int, script: Sequence[int], eos_id: int) -> None:
        super().__init__(vocab_size)
        self._script = list(script)
        self._eos_id = eos_id

    def _step_logits(self, batch: int, step: int) -> torch.Tensor:
        token = self._script[step] if step < len(self._script) else self._eos_id
        vector = torch.full((self.vocab_size,), -50.0)
        vector[token] = 50.0
        return vector[None, :].expand(batch, self.vocab_size)


def make_result(
    *,
    temperature: float = 1.0,
    top_p: float = 0.95,
    n_pv: int = 10,
    n_input: int = 100,
    duplicate_rate: float = 0.0,
    tp_rate: float | None = None,
    seconds: float = 1.0,
) -> SweepResult:
    """Build a synthetic :class:`SweepResult` for the pure-function tests."""
    counts = {
        "n_input": n_input,
        "n_sv": n_input,
        "n_tsd": n_input,
        "n_dd": n_input,
        "n_pv": n_pv,
        "sv_rate": 1.0,
        "tsd_rate": 1.0,
        "dd_rate": 1.0,
        "pv_rate": n_pv / n_input,
    }
    return SweepResult(
        point=SweepPoint(temperature=temperature, top_p=top_p),
        n_requested=n_input,
        counts=counts,
        sr_rate=0.5,
        duplicate_rate=duplicate_rate,
        mean_length=12.0,
        tp_rate=tp_rate,
        property_mean=None,
        property_mae=None,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------


class TestSweepGrid:
    def test_cardinality_and_uniqueness(self) -> None:
        temperatures = [0.5, 1.0, 1.5]
        top_ps = [0.75, 0.95]
        epochs = [1, 2, 3, 4]
        points = sweep_grid(temperatures=temperatures, top_ps=top_ps, epochs=epochs)

        assert len(points) == len(temperatures) * len(top_ps) * len(epochs)
        assert len(set(points)) == len(points)
        assert {p.temperature for p in points} == set(temperatures)
        assert {p.top_p for p in points} == set(top_ps)
        assert {p.epoch for p in points} == set(epochs)

    def test_without_epochs_the_epoch_axis_is_absent(self) -> None:
        points = sweep_grid(temperatures=[0.9, 1.1], top_ps=[0.75])
        assert len(points) == 2
        assert all(p.epoch is None and p.checkpoint is None for p in points)

    def test_checkpoints_are_attached_per_epoch(self) -> None:
        points = sweep_grid(
            temperatures=[1.0],
            top_ps=[0.75, 0.95],
            epochs=[6, 8],
            checkpoints={6: "epoch_0006.pt", 8: "epoch_0008.pt"},
        )
        assert {(p.epoch, p.checkpoint) for p in points} == {
            (6, "epoch_0006.pt"),
            (8, "epoch_0008.pt"),
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperatures": [], "top_ps": [0.75]},
            {"temperatures": [1.0], "top_ps": []},
            {"temperatures": [1.0, 1.0], "top_ps": [0.75]},
            {"temperatures": [-0.5], "top_ps": [0.75]},
            {"temperatures": [1.0], "top_ps": [1.5]},
            {"temperatures": [1.0], "top_ps": [0.0]},
        ],
    )
    def test_rejects_malformed_axes(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            sweep_grid(**kwargs)

    def test_points_are_hashable_and_frozen(self) -> None:
        point = SweepPoint(temperature=1.1, top_p=0.75)
        assert hash(point) == hash(SweepPoint(temperature=1.1, top_p=0.75))
        with pytest.raises(dataclasses.FrozenInstanceError):
            point.temperature = 2.0  # type: ignore[misc]


class TestPaperGrid:
    """Direct claims about the paper's stated sweep axes."""

    def test_forty_temperature_top_p_combinations(self) -> None:
        assert len(PAPER_GRID) == 40
        combinations = {(p.temperature, p.top_p) for p in PAPER_GRID}
        assert len(combinations) == 40

    def test_temperature_axis_is_0_1_to_2_0_in_steps_of_0_1(self) -> None:
        temperatures = sorted({p.temperature for p in PAPER_GRID})
        assert len(temperatures) == 20
        assert temperatures[0] == pytest.approx(0.1)
        assert temperatures[-1] == pytest.approx(2.0)
        for lower, upper in zip(temperatures, temperatures[1:], strict=False):
            assert upper - lower == pytest.approx(0.1)
        assert list(PAPER_TEMPERATURES) == temperatures

    def test_top_p_axis_is_the_papers_two_values(self) -> None:
        assert sorted({p.top_p for p in PAPER_GRID}) == [0.75, 0.95]
        assert sorted(PAPER_TOP_PS) == [0.75, 0.95]

    def test_default_grid_is_an_affordable_subset_of_the_paper_axes(self) -> None:
        assert 0 < len(DEFAULT_GRID) < len(PAPER_GRID)
        assert {p.top_p for p in DEFAULT_GRID} <= set(PAPER_TOP_PS)
        assert {p.temperature for p in DEFAULT_GRID} <= set(PAPER_TEMPERATURES)

    def test_paper_sample_budget_is_recorded(self) -> None:
        assert PAPER_SAMPLES_PER_CONFIG == 10_000


class TestAssignTargets:
    def test_round_robin_over_targets(self) -> None:
        assert assign_targets([300.0, 500.0], 5) == [300.0, 500.0, 300.0, 500.0, 300.0]

    def test_single_target(self) -> None:
        assert assign_targets([500.0], 3) == [500.0, 500.0, 500.0]

    @pytest.mark.parametrize("targets,n", [([], 3), ([500.0], 0), ([500.0], -1)])
    def test_rejects_empty_inputs(self, targets: list[float], n: int) -> None:
        with pytest.raises(ValueError):
            assign_targets(targets, n)


# ---------------------------------------------------------------------------
# running a point
# ---------------------------------------------------------------------------


class TestRunSweepPoint:
    def test_well_formed_result_on_an_untrained_model(self, tiny_model, tokenizer) -> None:
        point = SweepPoint(temperature=1.0, top_p=0.95)
        result = run_sweep_point(
            tiny_model,
            tokenizer,
            point=point,
            targets=[300.0, 500.0],
            n_samples=8,
            training_index=None,
            device="cpu",
            batch_size=4,
            seed=0,
            max_length=12,
        )

        assert isinstance(result, SweepResult)
        assert result.point == point
        assert result.n_requested == 8
        counts = result.counts
        assert counts["n_input"] == 8
        # The cascade is nested: SV > TSD > DD > PV.
        assert counts["n_input"] >= counts["n_sv"] >= counts["n_tsd"] >= counts["n_dd"]
        assert counts["n_dd"] >= counts["n_pv"] >= 0
        for key in ("sv_rate", "tsd_rate", "dd_rate", "pv_rate"):
            assert 0.0 <= counts[key] <= 1.0
        assert 0.0 <= result.sr_rate <= 1.0
        assert 0.0 <= result.duplicate_rate <= 1.0
        assert result.mean_length >= 0.0
        assert result.seconds >= 0.0
        assert math.isfinite(result.seconds)

    def test_never_raises_on_adversarial_output(self, tokenizer) -> None:
        """A model that only ever emits a nonsense token must still be scored."""
        junk = tokenizer.token_to_id("[C]")
        model = ConstantLogitModel(tokenizer.vocab_size, {junk: 50.0})
        result = run_sweep_point(
            model,
            tokenizer,
            point=SweepPoint(temperature=1.0, top_p=0.95),
            targets=[500.0],
            n_samples=4,
            training_index=None,
            device="cpu",
            batch_size=4,
            seed=0,
            max_length=6,
        )
        assert result.counts["n_input"] == 4

    def test_to_dict_is_json_serialisable_and_flat(self, tiny_model, tokenizer) -> None:
        result = run_sweep_point(
            tiny_model,
            tokenizer,
            point=SweepPoint(temperature=1.0, top_p=0.75, epoch=3, checkpoint="best.pt"),
            targets=[500.0],
            n_samples=4,
            training_index=None,
            device="cpu",
            batch_size=4,
            seed=0,
            max_length=8,
        )
        payload = result.to_dict()
        json.dumps(payload)  # must not raise
        assert payload["temperature"] == 1.0
        assert payload["top_p"] == 0.75
        assert payload["epoch"] == 3
        assert payload["checkpoint"] == "best.pt"
        assert payload["n_pv"] == result.counts["n_pv"]


class TestDeterminism:
    def test_same_seed_gives_identical_counts(self, tiny_model, tokenizer) -> None:
        kwargs = {
            "point": SweepPoint(temperature=1.2, top_p=0.95),
            "targets": [400.0],
            "n_samples": 8,
            "training_index": None,
            "device": "cpu",
            "batch_size": 4,
            "max_length": 12,
        }
        first = run_sweep_point(tiny_model, tokenizer, seed=7, **kwargs)
        second = run_sweep_point(tiny_model, tokenizer, seed=7, **kwargs)
        assert first.counts == second.counts
        assert first.sr_rate == second.sr_rate
        assert first.duplicate_rate == second.duplicate_rate
        assert first.mean_length == second.mean_length

    def test_same_seed_gives_identical_candidates(self, tiny_model, tokenizer) -> None:
        kwargs = {
            "point": SweepPoint(temperature=1.2, top_p=0.95),
            "targets": [400.0],
            "n_samples": 8,
            "device": "cpu",
            "batch_size": 4,
            "max_length": 12,
        }
        first = generate_candidates(tiny_model, tokenizer, seed=7, **kwargs)
        second = generate_candidates(tiny_model, tokenizer, seed=7, **kwargs)
        assert first == second
        assert len(first) == 8

    def test_different_seeds_give_different_candidates(self, tiny_model, tokenizer) -> None:
        kwargs = {
            "point": SweepPoint(temperature=1.2, top_p=0.95),
            "targets": [400.0],
            "n_samples": 8,
            "device": "cpu",
            "batch_size": 4,
            "max_length": 12,
        }
        first = generate_candidates(tiny_model, tokenizer, seed=7, **kwargs)
        second = generate_candidates(tiny_model, tokenizer, seed=8, **kwargs)
        assert first != second

    def test_batch_size_does_not_change_the_sample_count(self, tiny_model, tokenizer) -> None:
        kwargs = {
            "point": SweepPoint(temperature=1.0, top_p=0.95),
            "targets": [300.0, 500.0],
            "n_samples": 7,
            "device": "cpu",
            "seed": 0,
            "max_length": 8,
        }
        assert len(generate_candidates(tiny_model, tokenizer, batch_size=3, **kwargs)) == 7
        assert len(generate_candidates(tiny_model, tokenizer, batch_size=64, **kwargs)) == 7


class TestTemperatureTradeOff:
    """The paper's stated trade-off, on a model whose logits we control.

    An untrained model has a nearly uniform output distribution, so its
    duplicate rate is dominated by noise rather than by temperature -- see
    :meth:`test_untrained_model_monotonicity_is_not_guaranteed`, which is
    marked xfail for exactly that reason. The property is therefore asserted
    on :class:`ConstantLogitModel`, whose next-token distribution is a fixed
    vector: at a very low temperature the softmax collapses onto its argmax and
    every sample is the same string, while at a high temperature it spreads
    over the whole support.
    """

    def _duplicate_rate(self, model, tokenizer, temperature: float) -> float:
        result = run_sweep_point(
            model,
            tokenizer,
            point=SweepPoint(temperature=temperature, top_p=0.95),
            targets=[500.0],
            n_samples=24,
            training_index=None,
            device="cpu",
            batch_size=24,
            seed=0,
            max_length=8,
        )
        return result.duplicate_rate

    @pytest.fixture()
    def spread_model(self, tokenizer: PolyT5Tokenizer) -> ConstantLogitModel:
        # A small support of chain-building SELFIES tokens: every sampled
        # string decodes to *some* molecule, so SV survivors are plentiful and
        # the duplicate rate is a real measurement rather than 0/0.
        logits = {
            tokenizer.token_to_id("[C]"): 6.0,
            tokenizer.token_to_id("[O]"): 0.0,
            tokenizer.token_to_id("[N]"): 0.0,
            tokenizer.token_to_id("[S]"): 0.0,
        }
        return ConstantLogitModel(tokenizer.vocab_size, logits)

    def test_low_temperature_duplicates_at_least_as_much_as_high(
        self, spread_model, tokenizer
    ) -> None:
        cold = self._duplicate_rate(spread_model, tokenizer, 0.1)
        hot = self._duplicate_rate(spread_model, tokenizer, 2.0)
        assert cold >= hot
        # Not vacuous: the cold run really does collapse onto one molecule.
        assert cold > 0.5

    @pytest.mark.xfail(
        reason=(
            "An untrained model's output distribution is nearly uniform, so temperature "
            "barely moves it and the duplicate rate is noise. The real property is "
            "asserted on ConstantLogitModel above."
        ),
        strict=False,
    )
    def test_untrained_model_monotonicity_is_not_guaranteed(self, tiny_model, tokenizer) -> None:
        cold = self._duplicate_rate(tiny_model, tokenizer, 0.1)
        hot = self._duplicate_rate(tiny_model, tokenizer, 2.0)
        assert cold > hot


# ---------------------------------------------------------------------------
# property columns
# ---------------------------------------------------------------------------


class TestPropertyColumns:
    @pytest.fixture()
    def scripted(self, tokenizer: PolyT5Tokenizer) -> ScriptedModel:
        """A model that always writes ``[At][C][C][At]`` -- a valid polyethylene."""
        script = [tokenizer.token_to_id(t) for t in ("[At]", "[C]", "[C]", "[At]")]
        return ScriptedModel(tokenizer.vocab_size, script, tokenizer.eos_id)

    def _run(self, model, tokenizer, **overrides):
        kwargs = {
            "point": SweepPoint(temperature=1.0, top_p=0.95),
            "targets": [500.0],
            "n_samples": 6,
            "training_index": None,
            "device": "cpu",
            "batch_size": 6,
            "seed": 0,
            "max_length": 10,
        }
        kwargs.update(overrides)
        return run_sweep_point(model, tokenizer, **kwargs)

    def test_scripted_model_produces_the_expected_cascade(self, scripted, tokenizer) -> None:
        result = self._run(scripted, tokenizer)
        assert result.counts["n_input"] == 6
        assert result.counts["n_sv"] == 6
        assert result.counts["n_tsd"] == 6
        assert result.counts["n_dd"] == 1  # all six are the same polymer
        assert result.counts["n_pv"] == 1
        assert result.duplicate_rate == pytest.approx(1.0 - 1.0 / 6.0)

    def test_none_without_a_predictor(self, scripted, tokenizer) -> None:
        result = self._run(scripted, tokenizer)
        assert result.tp_rate is None
        assert result.property_mean is None
        assert result.property_mae is None

    def test_populated_with_a_fake_predictor(self, scripted, tokenizer) -> None:
        result = self._run(
            scripted,
            tokenizer,
            property_predictor=lambda psmiles: [480.0] * len(psmiles),
            target_property=500.0,
            tolerance=50.0,
        )
        assert result.property_mean == pytest.approx(480.0)
        assert result.tp_rate == pytest.approx(1.0)  # |480 - 500| <= 50
        assert result.property_mae == pytest.approx(20.0)

    def test_tp_rate_falls_to_zero_outside_the_window(self, scripted, tokenizer) -> None:
        result = self._run(
            scripted,
            tokenizer,
            property_predictor=lambda psmiles: [100.0] * len(psmiles),
            target_property=500.0,
            tolerance=50.0,
        )
        assert result.tp_rate == pytest.approx(0.0)
        assert result.property_mae == pytest.approx(400.0)

    def test_mae_uses_each_candidates_own_conditioning_target(self, scripted, tokenizer) -> None:
        """With no fixed ``target_property``, MAE is against the prompt target."""
        result = self._run(
            scripted,
            tokenizer,
            targets=[400.0],
            property_predictor=lambda psmiles: [450.0] * len(psmiles),
        )
        assert result.property_mae == pytest.approx(50.0)

    def test_a_failing_predictor_yields_none_not_a_crash(self, scripted, tokenizer) -> None:
        def exploding(_psmiles: Sequence[str]) -> Sequence[float]:
            raise RuntimeError("predictor blew up")

        result = self._run(scripted, tokenizer, property_predictor=exploding)
        assert result.property_mean is None
        assert result.property_mae is None
        assert result.tp_rate is None


# ---------------------------------------------------------------------------
# objective selection
# ---------------------------------------------------------------------------


class TestSelectBest:
    def test_pv_rate_objective(self) -> None:
        results = [
            make_result(temperature=1.5, n_pv=10),
            make_result(temperature=0.5, n_pv=90),
            make_result(temperature=1.0, n_pv=50),
        ]
        assert select_best(results, objective="pv_rate").point.temperature == 0.5

    def test_tp_rate_objective(self) -> None:
        results = [
            make_result(temperature=0.5, n_pv=90, tp_rate=0.1),
            make_result(temperature=1.1, n_pv=40, tp_rate=0.8),
        ]
        assert select_best(results, objective="tp_rate").point.temperature == 1.1

    def test_composite_penalises_duplication(self) -> None:
        # A cold point with a great PV rate but a batch that is 90% repeats
        # loses to a warmer point that yields fewer but more varied polymers.
        cold = make_result(temperature=0.2, n_pv=90, duplicate_rate=0.9)
        warm = make_result(temperature=1.1, n_pv=50, duplicate_rate=0.1)
        assert select_best([cold, warm], objective="pv_rate") is cold
        assert select_best([cold, warm], objective="composite") is warm

    def test_tie_break_is_the_first_result_in_input_order(self) -> None:
        first = make_result(temperature=0.5, n_pv=50)
        second = make_result(temperature=1.5, n_pv=50)
        assert select_best([first, second], objective="pv_rate") is first
        assert select_best([second, first], objective="pv_rate") is second

    def test_rejects_unknown_objective(self) -> None:
        with pytest.raises(ValueError):
            select_best([make_result()], objective="nonsense")

    def test_rejects_empty_results(self) -> None:
        with pytest.raises(ValueError):
            select_best([], objective="pv_rate")

    def test_tp_objective_without_any_predictions_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            select_best([make_result(tp_rate=None)], objective="tp_rate")

    def test_tp_objective_skips_results_without_predictions(self) -> None:
        missing = make_result(temperature=0.5, tp_rate=None)
        present = make_result(temperature=1.1, tp_rate=0.3)
        assert select_best([missing, present], objective="tp_rate") is present


# ---------------------------------------------------------------------------
# tabulation
# ---------------------------------------------------------------------------


class TestTables:
    def test_dataframe_has_one_row_per_result(self) -> None:
        results = [make_result(temperature=t) for t in (0.5, 1.0, 1.5)]
        frame = sweep_to_dataframe(results)
        assert len(frame) == 3
        for column in (
            "temperature",
            "top_p",
            "n_pv",
            "pv_rate",
            "sr_rate",
            "duplicate_rate",
            "mean_length",
            "tp_rate",
            "seconds",
        ):
            assert column in frame.columns
        assert list(frame["temperature"]) == [0.5, 1.0, 1.5]

    def test_markdown_has_a_header_naming_every_metric_and_one_row_per_result(self) -> None:
        results = [make_result(temperature=t, n_pv=n) for t, n in ((0.5, 90), (1.5, 10))]
        table = sweep_to_markdown(results)
        lines = [line for line in table.splitlines() if line.strip()]
        assert len(lines) == len(results) + 2  # header + separator + rows

        header = lines[0]
        for name in (
            "temperature",
            "top_p",
            "n_input",
            "n_sv",
            "n_tsd",
            "n_dd",
            "n_pv",
            "pv_rate",
            "sr_rate",
            "duplicate_rate",
            "mean_length",
            "tp_rate",
            "property_mean",
            "property_mae",
            "seconds",
        ):
            assert name in header
        assert set(lines[1]) <= set("|-: ")
        assert all(line.startswith("|") and line.endswith("|") for line in lines)

    def test_markdown_sorts_by_the_requested_metric(self) -> None:
        results = [make_result(temperature=1.5, n_pv=10), make_result(temperature=0.5, n_pv=90)]
        table = sweep_to_markdown(results, sort_by="pv_rate")
        rows = [line for line in table.splitlines() if line.strip()][2:]
        assert "0.5" in rows[0].split("|")[1]

    def test_markdown_keeps_input_order_when_sort_by_is_none(self) -> None:
        results = [make_result(temperature=1.5, n_pv=10), make_result(temperature=0.5, n_pv=90)]
        rows = [
            line for line in sweep_to_markdown(results, sort_by=None).splitlines() if line.strip()
        ][2:]
        assert "1.5" in rows[0].split("|")[1]

    def test_markdown_renders_missing_property_columns_without_fabricating(self) -> None:
        table = sweep_to_markdown([make_result(tp_rate=None)])
        assert "n/a" in table

    def test_markdown_of_an_empty_sweep_is_still_a_table(self) -> None:
        table = sweep_to_markdown([])
        lines = [line for line in table.splitlines() if line.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# incremental persistence
# ---------------------------------------------------------------------------


class TestIncrementalJsonl:
    def test_each_append_is_immediately_readable(self, tmp_path: Path) -> None:
        path = tmp_path / "sweep.jsonl"
        for temperature in (0.5, 1.0, 1.5):
            append_result_jsonl(path, make_result(temperature=temperature))
            rows = read_results_jsonl(path)
            assert all(isinstance(row, dict) for row in rows)
        assert [row["temperature"] for row in read_results_jsonl(path)] == [0.5, 1.0, 1.5]

    def test_an_interrupted_run_leaves_a_parseable_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sweep.jsonl"
        append_result_jsonl(path, make_result(temperature=0.5))
        append_result_jsonl(path, make_result(temperature=1.0))
        # Simulate a process killed halfway through writing the third row.
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"temperature": 1.5, "top_p": 0.9')

        rows = read_results_jsonl(path)
        assert len(rows) == 2
        assert [row["temperature"] for row in rows] == [0.5, 1.0]

    def test_rows_round_trip_back_into_results(self, tmp_path: Path) -> None:
        """sweep.jsonl is a checkpoint, not just a log: it must reload."""
        path = tmp_path / "sweep.jsonl"
        original = make_result(temperature=1.1, top_p=0.75, n_pv=42, tp_rate=0.33)
        append_result_jsonl(path, original)
        restored = SweepResult.from_dict(read_results_jsonl(path)[0])
        assert restored == original

    def test_from_dict_rejects_a_row_it_did_not_write(self) -> None:
        with pytest.raises(KeyError):
            SweepResult.from_dict({"temperature": 1.0, "top_p": 0.95})

    def test_missing_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_results_jsonl(tmp_path / "nope.jsonl") == []

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "sweep.jsonl"
        append_result_jsonl(path, make_result())
        assert path.is_file()
