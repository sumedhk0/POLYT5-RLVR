# tests/test_rl_rollout.py
from __future__ import annotations

import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.rl.reference_policy import ReferencePolicy
from polyt5.rl.rollout import RolloutBatch, sample_groups
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import save_checkpoint


def _tiny():
    tok = PolyT5Tokenizer.default()
    cfg = PolyT5Config(vocab_size=tok.vocab_size, d_model=64, d_kv=16, num_heads=4,
                       d_ff=128, num_layers=2, n_positions=64,
                       pad_token_id=tok.pad_id, eos_token_id=tok.eos_id,
                       decoder_start_token_id=tok.decoder_start_token_id)
    return PolyT5ForConditionalGeneration(cfg).eval(), tok


def test_sample_groups_shapes_and_grouping():
    model, tok = _tiny()
    batch = sample_groups(model, tok, targets=[300.0, 500.0], group_size=4,
                          max_length=32, temperature=1.0, top_p=0.95, seed=0,
                          device="cpu")
    assert isinstance(batch, RolloutBatch)
    assert batch.sequences.shape[0] == 8, "2 prompts x group of 4"
    assert batch.logprobs.shape == batch.mask.shape
    assert len(batch.texts) == 8
    # group members share their prompt's target, contiguously
    assert batch.targets[:4] == [300.0] * 4
    assert batch.targets[4:] == [500.0] * 4


def test_rollout_is_deterministic_under_a_seed():
    model, tok = _tiny()
    kw = dict(targets=[400.0], group_size=4, max_length=32, temperature=1.0,
              top_p=0.95, device="cpu")
    a = sample_groups(model, tok, seed=7, **kw)
    b = sample_groups(model, tok, seed=7, **kw)
    assert torch.equal(a.sequences, b.sequences)
    assert a.texts == b.texts


def test_mask_marks_padding_not_real_tokens():
    model, tok = _tiny()
    batch = sample_groups(model, tok, targets=[400.0], group_size=4, max_length=32,
                          temperature=1.0, top_p=0.95, seed=0, device="cpu")
    lengths = batch.mask.sum(dim=1)
    assert (lengths > 0).all()
    assert (lengths <= batch.sequences.shape[1]).all()


def test_logprobs_are_negative_and_finite():
    model, tok = _tiny()
    batch = sample_groups(model, tok, targets=[400.0], group_size=4, max_length=32,
                          temperature=1.0, top_p=0.95, seed=0, device="cpu")
    live = batch.logprobs[batch.mask.bool()]
    assert torch.isfinite(live).all()
    assert (live <= 0).all(), "log-probabilities cannot be positive"


# -- additional coverage: behavioural requirements the brief states in prose --
# rather than in the literal Step 1 test block (RolloutBatch.prompt_ids /
# prompt_mask, the 128-chunking rule, the unmodified-distribution guarantee,
# and ReferencePolicy's frozen contract). See task-6-report.md for rationale.


def test_prompt_ids_and_mask_are_exposed_and_grouped_contiguously():
    """RolloutBatch must carry the encoder side too -- ReferencePolicy.score
    needs it, and it must share sequences' contiguous group layout.
    """
    model, tok = _tiny()
    batch = sample_groups(model, tok, targets=[300.0, 500.0], group_size=3,
                          max_length=16, temperature=1.0, top_p=0.95, seed=0,
                          device="cpu")
    assert batch.prompt_ids.shape[0] == 6
    assert batch.prompt_mask.shape == batch.prompt_ids.shape
    decoded_prompts = tok.batch_decode(batch.prompt_ids.tolist())
    assert decoded_prompts[:3] == ["300.0"] * 3
    assert decoded_prompts[3:] == ["500.0"] * 3


def test_generation_is_chunked_at_128(monkeypatch):
    """Requirement 2: chunk at 128 regardless of how many candidates are asked
    for. Spies on the module-level `generate` call so this stays fast (no need
    to actually run 260 sequences through a model to prove the batching rule).
    """
    import polyt5.rl.rollout as rollout_mod

    model, tok = _tiny()
    seen_sizes: list[int] = []
    real_generate = rollout_mod.generate

    def spy(model_, input_ids, attention_mask=None, *, config, generator=None):
        seen_sizes.append(input_ids.shape[0])
        return real_generate(model_, input_ids, attention_mask, config=config, generator=generator)

    monkeypatch.setattr(rollout_mod, "generate", spy)

    targets = [float(i) for i in range(65)]  # 65 prompts x group_size 4 = 260
    batch = sample_groups(model, tok, targets=targets, group_size=4, max_length=4,
                          temperature=1.0, top_p=0.95, seed=0, device="cpu")

    assert batch.sequences.shape[0] == 260
    assert seen_sizes, "generate() was never called"
    assert all(size <= 128 for size in seen_sizes), seen_sizes
    assert sum(seen_sizes) == 260


def test_logprobs_are_from_unmodified_distribution_not_filtered():
    """Requirement 5, and mutant (c): token_logprobs must be the raw
    log_softmax of the model's logits, computed BEFORE temperature scaling and
    top-p/top-k filtering -- not the log-prob under the filtered proposal.

    Uses an aggressive temperature + top_k + top_p so the filtered and raw
    distributions diverge sharply, then independently recomputes the raw
    log-probs via a teacher-forced forward pass and requires an exact match.
    A `sample_groups` that instead logged probabilities from the filtered
    (processed) distribution would fail this to well beyond float tolerance.
    """
    model, tok = _tiny()
    batch = sample_groups(model, tok, targets=[400.0], group_size=4, max_length=8,
                          temperature=4.0, top_p=0.3, top_k=5, seed=0, device="cpu")

    start = torch.full((batch.sequences.shape[0], 1), tok.decoder_start_token_id,
                       dtype=torch.long)
    decoder_input_ids = torch.cat([start, batch.sequences[:, :-1]], dim=1)
    with torch.no_grad():
        out = model(batch.prompt_ids, attention_mask=batch.prompt_mask,
                    decoder_input_ids=decoder_input_ids)
    raw = torch.log_softmax(out.logits.float(), dim=-1)
    expected = raw.gather(2, batch.sequences.unsqueeze(-1)).squeeze(-1)
    expected = expected * batch.mask

    assert torch.allclose(batch.logprobs, expected, atol=1e-4)


def test_reference_policy_is_frozen():
    """Mutant (b): a reference left with requires_grad=True makes the KL
    anchor meaningless if it is ever swept into a backward pass.
    """
    model, _tok = _tiny()
    ref = ReferencePolicy(model)
    assert ref.model.training is False
    assert all(not p.requires_grad for p in ref.model.parameters())


def test_reference_policy_score_matches_manual_forward_and_has_no_grad():
    model, tok = _tiny()
    ref = ReferencePolicy(model)
    batch = sample_groups(model, tok, targets=[400.0], group_size=2, max_length=8,
                          temperature=1.0, top_p=0.95, seed=0, device="cpu")

    scores = ref.score(batch.sequences, batch.prompt_ids, batch.prompt_mask)
    assert scores.shape == batch.sequences.shape
    assert scores.requires_grad is False
    assert torch.isfinite(scores).all()

    start = torch.full((batch.sequences.shape[0], 1), tok.decoder_start_token_id,
                       dtype=torch.long)
    decoder_input_ids = torch.cat([start, batch.sequences[:, :-1]], dim=1)
    with torch.no_grad():
        out = model(batch.prompt_ids, attention_mask=batch.prompt_mask,
                    decoder_input_ids=decoder_input_ids)
    raw = torch.log_softmax(out.logits.float(), dim=-1)
    expected = raw.gather(2, batch.sequences.unsqueeze(-1)).squeeze(-1)

    # score() is deliberately unmasked (see its docstring); compare directly,
    # including whatever it reports at trailing-padding positions.
    assert torch.allclose(scores, expected, atol=1e-4)


def test_reference_policy_from_checkpoint_round_trips(tmp_path):
    model, tok = _tiny()
    ckpt_path = tmp_path / "reference.pt"
    save_checkpoint(ckpt_path, model=model, epoch=0, global_step=0, config={},
                    model_config=model.config.to_dict(), tokenizer_sha256=tok.sha256)

    ref = ReferencePolicy.from_checkpoint(ckpt_path)
    assert ref.model.training is False
    assert all(not p.requires_grad for p in ref.model.parameters())
    for original, loaded in zip(model.state_dict().values(), ref.model.state_dict().values(),
                                strict=True):
        assert torch.equal(original, loaded)
