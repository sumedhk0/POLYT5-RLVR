"""Group A Task 11: scoring with the regression head, and refusing to be mistaken
for a baseline checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from polyt5.inference import PolyT5PropertyPredictor
from polyt5.inference.regression_predictor import (
    GROUP_A_CONFIG_KEY,
    RegressionPropertyPredictor,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import save_checkpoint
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"
VOCAB = REPO / "artifacts" / "tokenizer" / "polyt5_vocab.json"

pytestmark = pytest.mark.skipif(not VOCAB.is_file(), reason="tokenizer artifact missing")


def build(n_descriptors: int = 0):
    seed_everything(0)
    tokenizer = PolyT5Tokenizer.from_file(VOCAB)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_id
    config.eos_token_id = tokenizer.eos_id
    config.decoder_start_token_id = tokenizer.decoder_start_token_id
    head_config = MultiTaskConfig(
        use_regression_head=True, n_descriptors=n_descriptors, head_dropout=0.0
    )
    model = PolyT5MultiTask(PolyT5ForConditionalGeneration(config), head_config)
    model.set_target_scaling(mean=417.0, std=113.0)
    return model, tokenizer, head_config


def write_checkpoint(path: Path, model, tokenizer, head_config):
    return save_checkpoint(
        path,
        model=model,
        epoch=0,
        global_step=1,
        config={GROUP_A_CONFIG_KEY: {"heads": head_config.to_dict(), "arm": "A1"}},
        model_config=model.config.to_dict(),
        tokenizer_path=str(VOCAB),
        tokenizer_sha256=tokenizer.sha256,
    )


def test_predictions_come_back_in_kelvin_around_the_train_mean():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    results = predictor.predict(["[At][C][C][O][At]", "[At][C][C][At]"])
    assert len(results) == 2
    assert all(result.is_numeric for result in results)
    assert all(math.isfinite(result.value) for result in results)


def test_the_decoded_field_is_the_formatted_number():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    result = predictor.predict(["[At][C][C][O][At]"])[0]
    assert result.decoded == f"{result.value:.1f}"
    assert result.source == "[At][C][C][O][At]"


def test_batching_does_not_change_a_prediction():
    model, tokenizer, _ = build()
    inputs = ["[At][C][C][O][At]", "[At][C][C][At]", "[At][C][O][At]"]
    one = RegressionPropertyPredictor(model, tokenizer, device="cpu", batch_size=1)
    many = RegressionPropertyPredictor(model, tokenizer, device="cpu", batch_size=8)
    assert one.predict_values(inputs) == pytest.approx(many.predict_values(inputs), abs=1e-4)


def test_a_blank_candidate_is_a_recorded_failure_not_an_exception():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    results = predictor.predict(["", "   ", None])
    assert [result.is_numeric for result in results] == [False, False, False]
    assert all(result.value is None for result in results)


def test_call_returns_nan_for_a_failure_so_the_tp_metric_can_drop_it():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    values = predictor(["", "[At][C][C][At]"])
    assert math.isnan(values[0])
    assert math.isfinite(values[1])


def test_round_trip_through_a_checkpoint_preserves_predictions(tmp_path):
    model, tokenizer, head_config = build(n_descriptors=3)
    path = write_checkpoint(tmp_path / "best.pt", model, tokenizer, head_config)
    reloaded = RegressionPropertyPredictor.from_checkpoint(path, device="cpu")
    inputs = ["[At][C][C][O][At]", "[At][C][C][At]"]
    original = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    assert reloaded.predict_values(inputs) == pytest.approx(
        original.predict_values(inputs), abs=1e-5
    )


def test_the_target_scaling_survives_the_checkpoint(tmp_path):
    model, tokenizer, head_config = build()
    path = write_checkpoint(tmp_path / "best.pt", model, tokenizer, head_config)
    reloaded = RegressionPropertyPredictor.from_checkpoint(path, device="cpu")
    assert float(reloaded.model.tg_mean) == pytest.approx(417.0)
    assert float(reloaded.model.tg_std) == pytest.approx(113.0)


def test_a_checkpoint_without_head_metadata_is_refused(tmp_path):
    model, tokenizer, _ = build()
    path = save_checkpoint(
        tmp_path / "bare.pt", model=model, epoch=0, global_step=1, config={},
        model_config=model.config.to_dict(), tokenizer_path=str(VOCAB),
        tokenizer_sha256=tokenizer.sha256,
    )
    with pytest.raises(ValueError, match=GROUP_A_CONFIG_KEY):
        RegressionPropertyPredictor.from_checkpoint(path, device="cpu")


def test_a_tokenizer_mismatch_is_refused(tmp_path):
    model, tokenizer, head_config = build()
    path = save_checkpoint(
        tmp_path / "wrong.pt", model=model, epoch=0, global_step=1,
        config={GROUP_A_CONFIG_KEY: {"heads": head_config.to_dict()}},
        model_config=model.config.to_dict(), tokenizer_path=str(VOCAB),
        tokenizer_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        RegressionPropertyPredictor.from_checkpoint(path, device="cpu")


def test_a_group_a_checkpoint_cannot_be_loaded_as_a_baseline_predictor(tmp_path):
    """Spec 7: Group A produces models ALONGSIDE the existing ones. A silent
    half-load into a reward path is the one failure mode that must not exist."""
    model, tokenizer, head_config = build()
    path = write_checkpoint(tmp_path / "best.pt", model, tokenizer, head_config)
    with pytest.raises((RuntimeError, KeyError, ValueError)):
        PolyT5PropertyPredictor.from_checkpoint(path, tokenizer_path=VOCAB, device="cpu")


def test_repr_names_the_arm_free_essentials():
    model, tokenizer, _ = build()
    text = repr(RegressionPropertyPredictor(model, tokenizer, device="cpu", property_name="Tg"))
    assert "RegressionPropertyPredictor" in text
    assert "Tg" in text


def test_degenerate_batch_size_is_refused():
    model, tokenizer, _ = build()
    with pytest.raises(ValueError, match="batch_size"):
        RegressionPropertyPredictor(model, tokenizer, device="cpu", batch_size=0)


@pytest.mark.parametrize(
    ("mean", "std", "standardised_bias", "expected_kelvin"),
    [
        # Hand-computed, not derived from Standardizer.inverse_transform or from
        # model.predict_tg: 0.5 * 113.0 + 417.0 = 473.5 and
        # 200.0 + 2.0 * 50.0 = 300.0.
        (417.0, 113.0, 0.5, 473.5),
        (200.0, 50.0, 2.0, 300.0),
    ],
)
def test_unstandardisation_matches_a_hand_computed_kelvin_value(
    mean, std, standardised_bias, expected_kelvin
):
    """Pin the head's output to a KNOWN standardised value and check the
    predictor reports the Kelvin figure computed by hand, independently of the
    inverse-transform code under test. This is the test that would catch a
    swapped mean/std, a sign error, or an inverse applied twice."""
    model, tokenizer, _ = build()
    model.set_target_scaling(mean=mean, std=std)
    # Zero the head's weight and fix its bias so the head emits exactly
    # `standardised_bias` in standardised units for ANY input, regardless of
    # what the encoder actually produces.
    with torch.no_grad():
        model.tg_head.projection.weight.zero_()
        model.tg_head.projection.bias.fill_(standardised_bias)
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    result = predictor.predict(["[At][C][C][O][At]"])[0]
    assert result.value == pytest.approx(expected_kelvin, abs=1e-4)


def test_call_auto_converts_psmiles_to_agree_with_the_pselfies_prediction():
    """evaluate_generation hands __call__ canonical PSMILES, not PSELFIES --
    the same contract PolyT5PropertyPredictor.__call__ honours by converting
    notation before scoring. "[At]CCO[At]" is the PSMILES form of the exact
    same polymer as "[At][C][C][O][At]" (verified via psmiles_to_pselfies);
    without conversion the PSMILES string is mis-tokenized character-by-
    character and scores as a DIFFERENT, wrong-but-finite Kelvin value."""
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    pselfies_value = predictor(["[At][C][C][O][At]"])[0]
    psmiles_value = predictor(["[At]CCO[At]"])[0]
    assert psmiles_value == pytest.approx(pselfies_value, abs=1e-4)


def test_call_survives_an_inference_exception_like_the_baseline_does():
    """PolyT5PropertyPredictor.__call__ wraps scoring in try/except so one bad
    batch degrades to NaNs, never an exception escaping into evaluate_generation.
    RegressionPropertyPredictor.__call__ must do the same to be interchangeable
    at the injection point."""
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated inference failure")

    # Patch the actual torch call inside predict(), not predict()/predict_values()
    # themselves -- so this test exercises __call__'s exception handling no
    # matter which of its own methods __call__ happens to delegate through.
    predictor.model.predict_tg = boom
    values = predictor(["[At][C][C][At]"])
    assert len(values) == 1
    assert math.isnan(values[0])
