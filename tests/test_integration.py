"""Cross-cutting integration tests for the supervised polyT5 pipeline.

Each track (chemistry, tokenizer, span corruption, model, training) is unit
tested in its own file. These tests exist to catch the failures those cannot
see: interface drift *between* layers. They walk the whole path the paper
describes, on real polymer structures::

    PSMILES -> [At] -> PSELFIES -> token ids -> span corruption -> padded batch
            -> polyT5 -> loss -> backward -> optimizer step -> checkpoint -> resume

They are deliberately CPU-only and small enough to run in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from polyt5.chemistry.conversion import psmiles_to_pselfies, star_to_at
from polyt5.chemistry.validity import validate_psmiles
from polyt5.data.collate import SpanCorruptionCollator
from polyt5.data.span_corruption import SpanCorruptionConfig, batch_span_corrupt
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import Trainer, TrainerConfig

#: Real polymer repeat units, star-terminated, spanning the functional-group
#: families the paper's Table S1 reports for its corpus.
REAL_PSMILES: tuple[str, ...] = (
    "*CC*",                                    # polyethylene
    "*CCO*",                                   # poly(ethylene oxide)
    "*CC(C)*",                                 # polypropylene
    "*CC(c1ccccc1)*",                          # polystyrene
    "*C(=O)c1ccc(cc1)C(=O)OCCO*",              # polyester
    "*C(=O)CCCCC(=O)NCCCCCCN*",                # polyamide
    "*CC(Cl)*",                                # poly(vinyl chloride)
    "*C(F)(F)C(F)(F)*",                        # PTFE
    "*[Si](C)(C)O*",                           # silicone
    "*Oc1ccc(cc1)C(C)(C)c1ccc(O*)cc1",         # bisphenol-A backbone
    "*CC(=O)OCC*",                             # vinyl ester
    "*NC(=O)c1ccc(cc1)C(=O)N*",                # aromatic polyamide
    "*OCCOC(=O)c1ccc(cc1)C(=O)*",              # PET-like
    "*CCCCCCN*",                               # aliphatic amine
)


@pytest.fixture(scope="module")
def tokenizer() -> PolyT5Tokenizer:
    return PolyT5Tokenizer.default()


@pytest.fixture(scope="module")
def pselfies() -> list[str]:
    """Real PSMILES carried through the chemistry layer to PSELFIES."""
    out = []
    for psmiles in REAL_PSMILES:
        capped = star_to_at(psmiles)
        verdict = validate_psmiles(capped, expected_termini=2)
        assert verdict.valid and verdict.correct_termini, f"{psmiles}: {verdict.reason}"
        encoded = psmiles_to_pselfies(capped)
        assert encoded is not None, f"selfies encoding failed for {psmiles}"
        out.append(encoded)
    return out


def _tiny_model(tokenizer: PolyT5Tokenizer) -> PolyT5ForConditionalGeneration:
    config = PolyT5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        d_kv=16,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        n_positions=64,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
    )
    return PolyT5ForConditionalGeneration(config)


# ---------------------------------------------------------------------------
# chemistry -> tokenizer
# ---------------------------------------------------------------------------


def test_real_polymers_survive_the_chemistry_layer(pselfies):
    assert len(pselfies) == len(REAL_PSMILES)
    assert all(s.count("[At]") == 2 for s in pselfies), "every PSELFIES keeps two termini"


def test_tokenizer_covers_real_polymer_tokens(tokenizer, pselfies):
    """No real polymer token may fall back to <unk>.

    This is the test that would catch a substitute base alphabet drifting away
    from the SELFIES symbols that real polymers actually produce.
    """
    unknown = 0
    for text in pselfies:
        ids = tokenizer.encode(text, add_eos=True)
        unknown += ids.count(tokenizer.unk_id)
    assert unknown == 0, f"{unknown} unknown tokens across {len(pselfies)} real polymers"


def test_tokenizer_round_trips_real_polymers(tokenizer, pselfies):
    for text in pselfies:
        ids = tokenizer.encode(text, add_eos=True)
        assert tokenizer.decode(ids, skip_special_tokens=True) == text


def test_tokenizer_artifact_covers_the_real_corpus():
    """Regression test for a bug that a handful of textbook polymers did not catch.

    A base alphabet derived from the ``selfies`` library instead of from the
    corpus passed every hand-written test above, then produced **2.1% unknown
    tokens across 22.4% of sequences** on the real PI1M corpus — the missing
    symbols being stereochemistry (``[/C]``, ``[\\C]``), bonded-terminus forms
    (``[=At]``, ``[/At]``), and explicit-hydrogen/charged species. Small curated
    examples cannot detect this; only real data can.

    Skips when the prepared corpus or the built artifact is absent, so a fresh
    checkout still runs green.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    corpus = repo / "data" / "processed" / "pi1m" / "train.txt"
    artifact = repo / "artifacts" / "tokenizer" / "polyt5_vocab.json"
    if not corpus.exists() or not artifact.exists():
        pytest.skip("prepared corpus or tokenizer artifact not present")

    tok = PolyT5Tokenizer.from_file(artifact)
    n_tokens = n_unknown = n_sequences = 0
    with corpus.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            ids = tok.encode(text, add_eos=True, max_length=200)
            n_tokens += len(ids)
            n_unknown += ids.count(tok.unk_id)
            n_sequences += 1
            if n_sequences >= 20_000:
                break

    rate = n_unknown / max(n_tokens, 1)
    # The training split defines the vocabulary, so its unknown rate must be
    # exactly zero. Any drift here means the artifact and the corpus disagree.
    assert rate == 0.0, (
        f"{n_unknown} unknown tokens in {n_tokens} tokens ({rate:.4%}) over "
        f"{n_sequences} training sequences — rebuild with "
        f"scripts/build_tokenizer.py --corpus {corpus}"
    )


# ---------------------------------------------------------------------------
# tokenizer -> span corruption
# ---------------------------------------------------------------------------


def test_span_corruption_on_real_polymers_obeys_the_paper(tokenizer, pselfies):
    """Every invariant the paper explicitly states, checked on real molecules."""
    config = SpanCorruptionConfig()
    ids = [tokenizer.encode(s, add_eos=True, max_length=200) for s in pselfies]
    results = batch_span_corrupt(
        ids,
        sentinel_ids=tokenizer.sentinel_ids,
        config=config,
        rng=np.random.default_rng(0),
        eos_id=tokenizer.eos_id,
    )

    corrupted_any = False
    for result in results:
        assert len(result.spans) <= config.max_spans
        assert all(length <= config.max_span_length for _, length in result.spans)
        assert result.corruption_fraction <= config.corruption_rate + 1e-9
        for (start_a, len_a), (start_b, _) in zip(result.spans, result.spans[1:], strict=False):
            assert start_b >= start_a + len_a + config.min_gap, "spans must not be adjacent"

        if result.spans:
            corrupted_any = True
            used = [i for i in result.input_ids if i in set(tokenizer.sentinel_ids)]
            assert used == tokenizer.sentinel_ids[: len(result.spans)], "sentinels in order"
            assert result.labels[0] == tokenizer.sentinel_ids[0]
    assert corrupted_any, "no real polymer was long enough to be corrupted — check the fixture"


# ---------------------------------------------------------------------------
# span corruption -> collator -> model
# ---------------------------------------------------------------------------


def test_collated_batch_feeds_the_model(tokenizer, pselfies):
    collator = SpanCorruptionCollator(
        sentinel_ids=tokenizer.sentinel_ids,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        config=SpanCorruptionConfig(),
        max_length=64,
        seed=0,
    )
    ids = [tokenizer.encode(s, add_eos=True, max_length=64) for s in pselfies]
    batch = collator(ids[:8])

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 8
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    # The paper: "padding tokens in the target sequences were replaced by -100".
    assert (batch["labels"] == -100).any()
    assert not (batch["labels"] == tokenizer.pad_id).all()

    model = _tiny_model(tokenizer)
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert output.logits.shape[0] == 8
    assert output.logits.shape[-1] == tokenizer.vocab_size
    assert torch.isfinite(output.loss)


# ---------------------------------------------------------------------------
# full pipeline through the Trainer
# ---------------------------------------------------------------------------


def _loader(tokenizer, pselfies, *, batch_size=4, n_batches=4, max_length=64):
    collator = SpanCorruptionCollator(
        sentinel_ids=tokenizer.sentinel_ids,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        config=SpanCorruptionConfig(),
        max_length=max_length,
        seed=0,
    )
    ids = [tokenizer.encode(s, add_eos=True, max_length=max_length) for s in pselfies]
    return [
        collator([ids[(b * batch_size + i) % len(ids)] for i in range(batch_size)])
        for b in range(n_batches)
    ]


def test_trainer_runs_the_real_pipeline(tmp_path, tokenizer, pselfies):
    from polyt5.utils import RunDirectory

    model = _tiny_model(tokenizer)
    batches = _loader(tokenizer, pselfies)
    config = TrainerConfig(
        max_epochs=1,
        physical_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        amp=False,
        scheduler="constant",
        device="cpu",
        seed=0,
    )
    run_dir = RunDirectory.create(tmp_path, "integration")
    trainer = Trainer(
        model,
        batches,
        config,
        val_loader=batches,
        run_dir=run_dir,
        tokenizer_sha256=tokenizer.sha256,
        run_config={"test": "integration"},
    )
    metrics = trainer.train()

    assert np.isfinite(metrics["train_loss"])
    assert run_dir.metrics_jsonl.exists()
    assert list(run_dir.checkpoints.glob("*.pt")), "an epoch checkpoint must be written"

    summary = trainer.summary()
    assert summary["effective_batch_size"] == 8


def test_effective_batch_size_must_match_the_declared_target():
    """The paper's batch size of 450 is asserted, never hard-coded."""
    TrainerConfig(
        max_epochs=1,
        physical_batch_size=18,
        gradient_accumulation_steps=25,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_effective_batch_size=450,
    )
    with pytest.raises(ValueError, match="450"):
        TrainerConfig(
            max_epochs=1,
            physical_batch_size=16,
            gradient_accumulation_steps=25,
            learning_rate=1e-3,
            weight_decay=0.0,
            target_effective_batch_size=450,
        )


def test_checkpoint_carries_tokenizer_identity(tmp_path, tokenizer, pselfies):
    """A checkpoint must be re-loadable *and* provably tied to its vocabulary."""
    from polyt5.training import load_checkpoint, save_checkpoint

    model = _tiny_model(tokenizer)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        epoch=0,
        global_step=1,
        config={"test": True},
        model_config=model.config.to_dict(),
        tokenizer_sha256=tokenizer.sha256,
    )
    state = load_checkpoint(path)
    assert state["tokenizer_sha256"] == tokenizer.sha256
    assert state["model_config"]["vocab_size"] == tokenizer.vocab_size

    reloaded = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(state["model_config"]))
    reloaded.load_state_dict(state["model_state"])

    ids = torch.tensor([tokenizer.encode(pselfies[0], add_eos=True)[:16]])
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        a = model(input_ids=ids, labels=ids).logits
        b = reloaded(input_ids=ids, labels=ids).logits
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# novelty index interchangeability
# ---------------------------------------------------------------------------


def test_scalable_novelty_index_is_a_drop_in_for_the_filter_cascade():
    """The two novelty implementations must agree exactly on the TSD stage.

    ``apply_filter_cascade`` tests ``canonical in training_index``. The set-based
    ``NoveltyIndex`` holds canonical PSMILES strings; ``ScalableNoveltyIndex``
    holds 64-bit hashes in a memory-mapped sorted array so it scales to 100M
    references. Evaluation must not care which one it was handed, or results
    become a function of which index the caller happened to build.

    Note the scalable index must be in ``canonical_psmiles`` space. A raw
    ``psmiles``-space index (which is what a corpus dedup sidecar produces)
    hashes the exact input text, so two different SMILES writings of one polymer
    would both count as novel — wrong for TSD specifically.
    """
    from polyt5.chemistry.conversion import pselfies_to_psmiles
    from polyt5.chemistry.novelty import NoveltyIndex
    from polyt5.chemistry.scalable_novelty import ScalableNoveltyIndex
    from polyt5.evaluation import apply_filter_cascade

    training = [
        "[At][C][C][At]",
        "[At][C][C][O][At]",
        "[At][C][C][Branch1][C][C][At]",
        "[At][C][C][Branch1][C][Cl][At]",
    ]
    candidates = [
        "[At][C][C][At]",                      # in training -> fails TSD
        "[At][C][C][O][At]",                   # in training -> fails TSD
        "[At][C][C][C][C][At]",                # novel
        "[At][C][C][C][C][At]",                # duplicate of the previous -> fails DD
        "[At][C][C][C][At]",                   # novel
        "[Zz][C][C][At]",                      # unparseable -> fails SV
    ]

    set_index = NoveltyIndex.from_pselfies(training)
    psmiles = [p for p in (pselfies_to_psmiles(t) for t in training) if p]
    scalable_index = ScalableNoveltyIndex.from_strings(
        psmiles, hash_space="canonical_psmiles"
    )
    assert len(set_index) == len(scalable_index)

    _, counts_set = apply_filter_cascade(candidates, training_index=set_index)
    _, counts_scalable = apply_filter_cascade(candidates, training_index=scalable_index)

    assert counts_set.to_dict() == counts_scalable.to_dict()
    # and the cascade actually exercised every stage, so the agreement is not vacuous
    assert counts_set.n_sv < counts_set.n_input
    assert counts_set.n_tsd < counts_set.n_sv
    assert counts_set.n_dd < counts_set.n_tsd
