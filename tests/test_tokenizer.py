"""Tests for the polyT5 tokenizer track.

These tests pin the *contract* described in the polyT5 Supplementary Information
("Tokenizer vocabulary") -- 199 base SELFIES tokens + 5 specials + 100 sentinels +
154 conditioning tokens = 458 -- plus the deterministic longest-match tokenization
behaviour this reproduction defines on top of it.

The tokenizer must remain torch-free: the future RLVR reward layer imports it
standalone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyt5.tokenization import (  # noqa: E402
    CONDITIONING_TOKENS,
    SENTINEL_COUNT,
    SPECIAL_TOKENS,
    PolyT5Tokenizer,
    TokenizerArtifact,
    build_default_vocab,
    build_vocab_from_corpus,
)
from polyt5.tokenization.vocab import (  # noqa: E402
    BASE_SELFIES_TARGET,
    CONDITIONING_TARGET,
    build_base_alphabet,
    default_selfies_alphabet,
)

# --- Prompt/target strings taken verbatim from the paper's SI examples. ---------------
PSELFIES_BARE = "[C][C][C][C][Branch1][C][At][C][At]"
PSELFIES_WITH_AT = "[C][C][Branch1][C][At][C][At]"
NUMBER_TARGET = "236.0"
DIELECTRIC_PROMPT = "property 4.1; polymer [C][C][Branch1][C][At][C][At]"
SOLUBILITY_PROMPT = (
    "polymer [C][C][Branch1][C][At][C][At]; solvent [C][C][C][O][C][Ring1][Branch1]"
)

ROUND_TRIP_CASES = [
    PSELFIES_BARE,
    PSELFIES_WITH_AT,
    NUMBER_TARGET,
    DIELECTRIC_PROMPT,
    SOLUBILITY_PROMPT,
    "soluble",
    "insoluble",
]


@pytest.fixture(scope="module")
def tok() -> PolyT5Tokenizer:
    return PolyT5Tokenizer.default()


# ---------------------------------------------------------------------------
# Vocabulary composition
# ---------------------------------------------------------------------------


def test_tokenizer_import_is_torch_free() -> None:
    """The tokenizer must stay torch-free so the RLVR reward layer can use it standalone.

    Checked in a subprocess: ``tests/test_model.py`` imports torch at collection time, so a
    plain ``sys.modules`` assertion in-process would only measure test ordering.
    """
    probe = (
        f"import sys; sys.path.insert(0, {str(_SRC)!r});"
        "import polyt5.tokenization as t;"
        "t.PolyT5Tokenizer.default().encode('[C][C]');"
        "banned = [m for m in ('torch', 'transformers', 'sentencepiece') if m in sys.modules];"
        "print(banned)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", f"tokenizer pulled in heavy deps: {result.stdout}"


def test_default_vocab_is_458_tokens(tok: PolyT5Tokenizer) -> None:
    assert tok.vocab_size == 458


def test_group_counts_match_paper(tok: PolyT5Tokenizer) -> None:
    assert len(SPECIAL_TOKENS) == 5
    assert SENTINEL_COUNT == 100
    assert BASE_SELFIES_TARGET == 199
    assert CONDITIONING_TARGET == 154
    assert len(CONDITIONING_TOKENS) == 154
    assert len(build_base_alphabet()) == 199
    assert 5 + 100 + 199 + 154 == 458


def test_no_duplicate_tokens(tok: PolyT5Tokenizer) -> None:
    tokens = list(tok.tokens)
    assert len(set(tokens)) == len(tokens)


def test_metadata_group_counts(tok: PolyT5Tokenizer) -> None:
    groups = tok.metadata["group_counts"]
    assert groups == {
        "special": 5,
        "sentinel": 100,
        "base_selfies": 199,
        "conditioning": 154,
    }


def test_metadata_documents_reproduction_gap(tok: PolyT5Tokenizer) -> None:
    note = tok.metadata["reproduction_note"]
    assert "not public" in note.lower() or "not publicly available" in note.lower()
    assert tok.metadata["selfies_version"]
    prov = tok.metadata["provenance"]
    assert prov["base_selfies"] == "substitute"
    assert prov["conditioning"] == "substitute"
    assert prov["special"] == "paper"
    assert prov["sentinel"] == "paper"


# ---------------------------------------------------------------------------
# ID-order contract
# ---------------------------------------------------------------------------


def test_id_order_contract(tok: PolyT5Tokenizer) -> None:
    assert tok.pad_id == 0
    assert tok.eos_id == 1
    assert tok.unk_id == 2
    assert tok.bos_id == 3
    assert tok.space_id == 4
    assert tok.id_to_token(0) == "<pad>"
    assert tok.id_to_token(1) == "</s>"
    assert tok.id_to_token(2) == "<unk>"
    assert tok.id_to_token(3) == "<s>"
    assert tok.id_to_token(4) == "▁"


def test_decoder_start_token_is_pad(tok: PolyT5Tokenizer) -> None:
    """T5 convention: decoder_start_token_id == pad_id, even though we carry a <s>."""
    assert tok.decoder_start_token_id == tok.pad_id == 0


def test_sentinel_ids_contiguous_and_ascending(tok: PolyT5Tokenizer) -> None:
    assert tok.sentinel_id(0) == 5
    assert tok.sentinel_id(99) == 104
    ids = tok.sentinel_ids
    assert ids == list(range(5, 105))
    assert ids == sorted(ids)
    assert tok.id_to_token(5) == "<extra_id_0>"
    assert tok.id_to_token(104) == "<extra_id_99>"


def test_sentinel_id_out_of_range_raises_index_error(tok: PolyT5Tokenizer) -> None:
    with pytest.raises(IndexError):
        tok.sentinel_id(100)
    with pytest.raises(IndexError):
        tok.sentinel_id(-1)


def test_base_alphabet_starts_at_105(tok: PolyT5Tokenizer) -> None:
    base = build_base_alphabet()
    assert tok.id_to_token(105) == base[0]
    assert tok.token_to_id(base[-1]) == 105 + 198


def test_conditioning_block_follows_base(tok: PolyT5Tokenizer) -> None:
    assert tok.id_to_token(304) == CONDITIONING_TOKENS[0]
    assert tok.id_to_token(457) == CONDITIONING_TOKENS[-1]


# ---------------------------------------------------------------------------
# Encode / decode round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_round_trip_is_exact(tok: PolyT5Tokenizer, text: str) -> None:
    ids = tok.encode(text)
    assert tok.unk_id not in ids, f"unexpected <unk> while encoding {text!r}"
    assert tok.decode(ids) == text


def test_at_terminus_token_is_in_vocab(tok: PolyT5Tokenizer) -> None:
    assert "[At]" in tok.tokens
    assert tok.encode("[At]", add_eos=False) == [tok.token_to_id("[At]")]


def test_add_eos_appends_exactly_once(tok: PolyT5Tokenizer) -> None:
    ids = tok.encode(PSELFIES_BARE, add_eos=True)
    assert ids[-1] == tok.eos_id
    assert ids.count(tok.eos_id) == 1
    assert tok.encode(PSELFIES_BARE, add_eos=False) == ids[:-1]


def test_decode_skips_eos_by_default(tok: PolyT5Tokenizer) -> None:
    ids = tok.encode(PSELFIES_BARE, add_eos=True)
    assert tok.decode(ids, skip_special_tokens=True) == PSELFIES_BARE
    assert tok.decode(ids, skip_special_tokens=False).endswith("</s>")


def test_truncation_yields_exactly_max_length(tok: PolyT5Tokenizer) -> None:
    long_text = PSELFIES_BARE * 20
    for n in (1, 2, 8, 17):
        ids = tok.encode(long_text, add_eos=True, max_length=n, truncation=True)
        assert len(ids) == n
        assert ids[-1] == tok.eos_id, "truncation must preserve a terminal </s>"
        ids_no_eos = tok.encode(long_text, add_eos=False, max_length=n, truncation=True)
        assert len(ids_no_eos) == n


def test_truncation_disabled_keeps_full_length(tok: PolyT5Tokenizer) -> None:
    long_text = PSELFIES_BARE * 20
    ids = tok.encode(long_text, add_eos=True, max_length=4, truncation=False)
    assert len(ids) > 4


def test_space_marker_only_between_non_bracket_tokens(tok: PolyT5Tokenizer) -> None:
    """Documented rule 4: whitespace becomes the U+2581 marker only between two
    non-bracket tokens; the space before a ``[`` group is implied by the bracket."""
    ids = tok.encode(DIELECTRIC_PROMPT, add_eos=False)
    # "property _ 4 . 1 ; _ polymer" -> two markers; the space before "[C]" is implied.
    assert ids.count(tok.space_id) == 2
    assert tok.encode("[C] [C]", add_eos=False).count(tok.space_id) == 0
    assert tok.decode(tok.encode("[C] [C]", add_eos=False)) == "[C][C]"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_unknown_bracket_token_maps_to_unk(tok: PolyT5Tokenizer) -> None:
    ids = tok.encode("[Zzz]", add_eos=False)
    assert ids == [tok.unk_id]


def test_unknown_run_collapses_to_single_unk(tok: PolyT5Tokenizer) -> None:
    ids = tok.encode("@@@@", add_eos=False)
    assert ids == [tok.unk_id]


def test_garbage_never_raises(tok: PolyT5Tokenizer) -> None:
    for junk in ["", "   ", "[[[[", "]]]]", "<<<>>>", "[Zzz][C]\x00中文", "[C" * 50]:
        ids = tok.encode(junk)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        tok.decode(ids)


def test_empty_string_rule(tok: PolyT5Tokenizer) -> None:
    assert tok.encode("", add_eos=False) == []
    assert tok.encode("", add_eos=True) == [tok.eos_id]


def test_decode_ignores_out_of_range_ids(tok: PolyT5Tokenizer) -> None:
    assert isinstance(tok.decode([10_000, -3]), str)


# ---------------------------------------------------------------------------
# Batch API (plain python containers only -- no torch, no numpy)
# ---------------------------------------------------------------------------


def test_batch_encode_pads_and_masks(tok: PolyT5Tokenizer) -> None:
    batch = tok.batch_encode([PSELFIES_BARE, NUMBER_TARGET, SOLUBILITY_PROMPT], padding=True)
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]
    assert isinstance(input_ids, list) and isinstance(mask, list)
    assert all(isinstance(row, list) for row in input_ids)
    assert all(isinstance(row, list) for row in mask)
    assert all(isinstance(i, int) for row in input_ids for i in row)
    widths = {len(row) for row in input_ids}
    assert len(widths) == 1, "padded rows must all be the same length"
    assert {len(row) for row in mask} == widths
    for row, m in zip(input_ids, mask, strict=True):
        for token_id, flag in zip(row, m, strict=True):
            assert flag == (0 if token_id == tok.pad_id else 1)
    assert sum(mask[1]) < sum(mask[2])


def test_batch_encode_without_padding(tok: PolyT5Tokenizer) -> None:
    batch = tok.batch_encode([PSELFIES_BARE, NUMBER_TARGET], padding=False)
    assert len({len(r) for r in batch["input_ids"]}) == 2


def test_batch_decode_round_trips(tok: PolyT5Tokenizer) -> None:
    texts = [PSELFIES_BARE, DIELECTRIC_PROMPT, NUMBER_TARGET]
    batch = tok.batch_encode(texts, padding=True)
    assert tok.batch_decode(batch["input_ids"]) == texts


def test_batch_encode_truncates(tok: PolyT5Tokenizer) -> None:
    batch = tok.batch_encode([PSELFIES_BARE * 20, "1"], max_length=6, truncation=True)
    assert all(len(r) == 6 for r in batch["input_ids"])


# ---------------------------------------------------------------------------
# Artifact persistence / determinism
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tok: PolyT5Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    tok.save(path)
    reloaded = PolyT5Tokenizer.from_file(path)
    assert reloaded.tokens == tok.tokens
    assert reloaded.sha256 == tok.sha256
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.decode(reloaded.encode(DIELECTRIC_PROMPT)) == DIELECTRIC_PROMPT
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sha256"] == tok.sha256
    assert set(payload) >= {"version", "sha256", "metadata", "tokens"}


def test_build_default_vocab_is_byte_identical_across_builds(tmp_path: Path) -> None:
    a = build_default_vocab()
    b = build_default_vocab()
    assert isinstance(a, TokenizerArtifact)
    assert a.tokens == b.tokens
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    PolyT5Tokenizer(a.tokens, a.metadata).save(pa)
    PolyT5Tokenizer(b.tokens, b.metadata).save(pb)
    assert pa.read_bytes() == pb.read_bytes()


def test_sha256_is_stable_and_content_sensitive(tok: PolyT5Tokenizer) -> None:
    assert tok.sha256 == PolyT5Tokenizer.default().sha256
    mutated = list(tok.tokens)
    mutated[-1] = "<cond_reserved_999>"
    assert PolyT5Tokenizer(mutated).sha256 != tok.sha256


# ---------------------------------------------------------------------------
# Corpus-derived variant
# ---------------------------------------------------------------------------


def test_build_vocab_from_corpus_keeps_target_size() -> None:
    corpus = [PSELFIES_BARE, PSELFIES_WITH_AT, "[C][C][O][C][C][O]"]
    art = build_vocab_from_corpus(corpus, target_base_size=199)
    assert len(art.tokens) == 458
    assert art.metadata["group_counts"]["base_selfies"] == 199
    assert art.metadata["base_alphabet"]["source"] == "corpus"
    assert art.metadata["base_alphabet"]["padded"] > 0
    tokenizer = PolyT5Tokenizer(art.tokens, art.metadata)
    assert tokenizer.unk_id not in tokenizer.encode(PSELFIES_BARE)


def test_build_base_alphabet_trims_when_corpus_is_too_large() -> None:
    corpus_tokens = [f"[X{i}]" for i in range(400)]
    alphabet = build_base_alphabet(corpus_tokens, target_size=199)
    assert len(alphabet) == 199
    assert len(set(alphabet)) == 199


def test_build_base_alphabet_respects_custom_target() -> None:
    assert len(build_base_alphabet(target_size=250)) == 250
    assert len(build_base_alphabet(target_size=30)) == 30


# ---------------------------------------------------------------------------
# Real chemistry integration -- the point of the whole alphabet exercise
# ---------------------------------------------------------------------------

# Real polymer PSMILES (PI1M / PoLyInfo style); the two [*] stars are the repeat-unit
# attachment points that polyT5 rewrites to the [At] terminus marker.
REAL_PSMILES = [
    "[*]CC[*]",                                          # polyethylene
    "[*]CC(c1ccccc1)[*]",                                # polystyrene
    "[*]Oc1ccc(cc1)C(C)(C)c1ccc(cc1)O[*]",               # bisphenol-A ether
    "[*]C(=O)c1ccc(cc1)C(=O)Nc1ccc(cc1)N[*]",            # aromatic polyamide
    "[*]CC(C)(C(=O)OC)[*]",                              # PMMA
    "[*]Nc1ccc(cc1)S(=O)(=O)c1ccc(cc1)N[*]",             # polysulfone-amine
    "[*]CC(Cl)[*]",                                      # PVC
    "[*]C(F)(F)C(F)(F)[*]",                              # PTFE
    "[*]CCOCCO[*]",                                      # PEG-like
    "[*]C[Si](C)(C)O[*]",                                # silicone
]


@pytest.mark.chem
def test_real_pselfies_tokens_are_all_in_vocab(tok: PolyT5Tokenizer) -> None:
    selfies = pytest.importorskip("selfies")
    seen: set[str] = set()
    for psmiles in REAL_PSMILES:
        pselfies = selfies.encoder(psmiles.replace("[*]", "[At]"))
        seen.update(selfies.split_selfies(pselfies))
        ids = tok.encode(pselfies, add_eos=False)
        assert tok.unk_id not in ids, f"<unk> produced for {psmiles!r} -> {pselfies!r}"
        assert tok.decode(ids) == pselfies
    missing = sorted(t for t in seen if t not in tok.tokens)
    assert not missing, f"real SELFIES tokens missing from the default alphabet: {missing}"


@pytest.mark.chem
def test_selfies_robust_alphabet_is_fully_covered(tok: PolyT5Tokenizer) -> None:
    """Anything selfies can legally emit under default constraints must be encodable."""
    selfies = pytest.importorskip("selfies")
    missing = sorted(t for t in selfies.get_semantic_robust_alphabet() if t not in tok.tokens)
    assert not missing, f"selfies robust alphabet not covered: {missing}"


@pytest.mark.chem
def test_default_alphabet_does_not_mutate_global_selfies_constraints() -> None:
    selfies = pytest.importorskip("selfies")
    before = selfies.get_semantic_constraints()
    default_selfies_alphabet()
    assert selfies.get_semantic_constraints() == before


# ---------------------------------------------------------------------------
# scripts/build_tokenizer.py
# ---------------------------------------------------------------------------


def _load_build_cli():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_tokenizer.py"
    spec = importlib.util.spec_from_file_location("build_tokenizer_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_artifact_matching_the_default_build(tmp_path: Path, capsys) -> None:
    cli = _load_build_cli()
    out = tmp_path / "vocab.json"
    assert cli.main(["--out", str(out), "--emit-sentencepiece-vocab"]) == 0

    tokenizer = PolyT5Tokenizer.from_file(out)
    assert tokenizer.tokens == PolyT5Tokenizer.default().tokens
    assert tokenizer.sha256 == PolyT5Tokenizer.default().sha256

    printed = capsys.readouterr().out
    assert tokenizer.sha256 in printed
    for group in ("special", "sentinel", "base_selfies", "conditioning"):
        assert group in printed
    assert "458" in printed
    assert "not publicly available" in printed

    sp_vocab = out.with_suffix(".vocab")
    lines = sp_vocab.read_text(encoding="utf-8").splitlines()
    assert len(lines) == tokenizer.vocab_size
    assert lines[0] == "<pad>\t0"


def test_cli_corpus_mode(tmp_path: Path) -> None:
    cli = _load_build_cli()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([PSELFIES_BARE, PSELFIES_WITH_AT, ""]), encoding="utf-8")
    out = tmp_path / "corpus_vocab.json"
    assert cli.main(["--out", str(out), "--corpus", str(corpus)]) == 0

    tokenizer = PolyT5Tokenizer.from_file(out)
    assert tokenizer.vocab_size == 458
    assert tokenizer.metadata["base_alphabet"]["source"] == "corpus"
    assert tokenizer.metadata["corpus_path"] == str(corpus)


def test_cli_missing_corpus_returns_error_code(tmp_path: Path) -> None:
    cli = _load_build_cli()
    args = ["--out", str(tmp_path / "x.json"), "--corpus", str(tmp_path / "nope.txt")]
    assert cli.main(args) == 2


def test_cli_custom_sizes(tmp_path: Path) -> None:
    cli = _load_build_cli()
    out = tmp_path / "small.json"
    args = ["--out", str(out), "--base-size", "50", "--sentinels", "8", "--conditioning-size", "20"]
    assert cli.main(args) == 0
    tokenizer = PolyT5Tokenizer.from_file(out)
    assert tokenizer.vocab_size == 5 + 8 + 50 + 20
    assert tokenizer.sentinel_id(7) == 12
    with pytest.raises(IndexError):
        tokenizer.sentinel_id(8)
