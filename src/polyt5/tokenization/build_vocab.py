"""Assemble the polyT5 vocabulary into a reproducible, hashable artifact.

The vocabulary is treated as a build artifact, not as code: it is constructed once here,
written to JSON with a SHA-256 over the token list, and reloaded identically by pre-training,
fine-tuning, evaluation, generation and the RLVR reward layer. Nothing downstream should ever
reconstruct a vocabulary by hand.

Two entry points are provided:

* :func:`build_default_vocab` -- the paper-shaped default (458 tokens), fully offline.
* :func:`build_vocab_from_corpus` -- the same layout, but with the base SELFIES block ranked
  by frequency over a supplied PSELFIES corpus, which is how the authors derived theirs.

See :mod:`polyt5.tokenization.vocab` for the id-order contract and for the paper quotes that
justify each group.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import selfies

from polyt5.tokenization.tokenizer import ARTIFACT_VERSION, TokenizerArtifact
from polyt5.tokenization.vocab import (
    BASE_SELFIES_TARGET,
    CONDITIONING_TARGET,
    SENTINEL_COUNT,
    SPECIAL_TOKENS,
    build_base_alphabet_detailed,
    build_conditioning_tokens,
    sentinel_tokens,
)

__all__ = [
    "REPRODUCTION_NOTE",
    "build_default_vocab",
    "build_vocab_from_corpus",
    "split_pselfies",
]

#: Recorded verbatim into every artifact so the substitution is impossible to lose track of.
REPRODUCTION_NOTE = (
    "polyT5 (Sahu et al., npj Artificial Intelligence 2026) reports a 458-token vocabulary: "
    "199 base SELFIES tokens + 5 special tokens + 100 sentinels + 154 conditioning tokens. "
    "The identities of the 199 base tokens and of the 154 conditioning tokens are NOT public: "
    "the base alphabet was derived from a 100M-polymer corpus the authors state is 'not "
    "publicly available due to IP protection', and the 154 conditioning tokens are never "
    "enumerated in the paper or its Supplementary Information. This artifact reproduces the "
    "paper's STRUCTURE and headline size exactly, but the base and conditioning token "
    "identities are a documented substitute chosen by this reproduction, not the authors' "
    "data. Only the special-token and sentinel groups are recoverable from the paper."
)


def split_pselfies(line: str) -> list[str]:
    """Split a PSELFIES string into bracketed tokens.

    Applies the paper's own rule -- "each substring enclosed within square brackets ... was
    treated as a distinct token" -- without depending on `selfies` grammar validation, so a
    corpus containing tokens outside the current `selfies` alphabet still contributes them.

    Args:
        line: One PSELFIES string, e.g. ``"[C][C][Branch1][C][At]"``.

    Returns:
        The bracketed tokens, in order. Text outside brackets is ignored.
    """
    tokens: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        start = line.find("[", i)
        if start == -1:
            break
        close = line.find("]", start)
        if close == -1:
            break
        tokens.append(line[start : close + 1])
        i = close + 1
    return tokens


def _assemble(
    base_alphabet: list[str],
    base_stats: dict[str, Any],
    *,
    sentinels: int,
    conditioning_size: int,
) -> TokenizerArtifact:
    """Concatenate the four groups in the documented id order.

    Args:
        base_alphabet: The base SELFIES block, already at its exact target size.
        base_stats: Adjustment statistics from :func:`build_base_alphabet_detailed`.
        sentinels: Number of sentinel tokens.
        conditioning_size: Exact size of the conditioning block.

    Returns:
        The assembled :class:`TokenizerArtifact`.

    Raises:
        ValueError: If the assembled vocabulary contains duplicate surface forms.
    """
    special = list(SPECIAL_TOKENS)
    sentinel = sentinel_tokens(sentinels)
    conditioning, cond_stats = build_conditioning_tokens(conditioning_size)

    tokens = special + sentinel + base_alphabet + conditioning
    if len(set(tokens)) != len(tokens):
        seen: dict[str, int] = {}
        for tok in tokens:
            seen[tok] = seen.get(tok, 0) + 1
        dupes = sorted(t for t, c in seen.items() if c > 1)
        raise ValueError(f"assembled vocabulary contains duplicates: {dupes[:10]}")

    group_counts = {
        "special": len(special),
        "sentinel": len(sentinel),
        "base_selfies": len(base_alphabet),
        "conditioning": len(conditioning),
    }
    offset = 0
    id_ranges: dict[str, list[int]] = {}
    for name, size in group_counts.items():
        id_ranges[name] = [offset, offset + size - 1] if size else [offset, offset - 1]
        offset += size

    metadata: dict[str, Any] = {
        "reproduction_note": REPRODUCTION_NOTE,
        "paper": (
            "Sahu et al., polyT5, npj Artificial Intelligence 2026 "
            "(Supplementary Information: 'Tokenizer vocabulary')"
        ),
        "selfies_version": selfies.__version__,
        "vocab_size": len(tokens),
        "group_counts": group_counts,
        "id_ranges": id_ranges,
        # Which groups are recoverable from the paper, and which we had to invent.
        "provenance": {
            "special": "paper",
            "sentinel": "paper",
            "base_selfies": "substitute",
            "conditioning": "substitute",
        },
        "provenance_detail": {
            "special": (
                "The paper names all five roles (start-of-sequence, end-of-sequence, unknown, "
                "padding, whitespace marker) but no surface forms. We use the T5/SentencePiece "
                "defaults <pad> </s> <unk> <s> and U+2581. [AMBIGUITY] stock T5 has no BOS and "
                "starts decoding from <pad>; decoder_start_token_id therefore stays pad_id."
            ),
            "sentinel": (
                "Count (100) and role (span masking during pre-training) are stated by the "
                "paper; the <extra_id_N> surface form is the standard T5 naming and is "
                "confirmed by paper Figure 2C."
            ),
            "base_selfies": (
                "[SUBSTITUTE] Size (199) is from the paper; identities are not. Derived here "
                "from selfies' robust alphabet under a widened polymer semantic-constraint "
                "table, plus the [At] terminus marker and the [#RingN] decode-only symbols."
            ),
            "conditioning": (
                "[SUBSTITUTE] Size (154) and the category list (property names, digits, "
                "decimal point, units, operators, booleans, polymer keywords) are from the "
                "paper; the individual tokens are never enumerated and are chosen here."
            ),
        },
        "base_alphabet": base_stats,
        "conditioning_block": cond_stats,
        "tokenization_rules": {
            "bracket_groups_atomic": True,
            "angle_special_forms_atomic": True,
            "longest_match_non_bracket": True,
            "space_marker_between_non_bracket_tokens_only": True,
            "unmatched_run_to_single_unk": True,
            "sentencepiece_model_trained": False,
        },
        "artifact_version": ARTIFACT_VERSION,
    }
    return TokenizerArtifact(ARTIFACT_VERSION, tokens, metadata)


def build_default_vocab(
    *,
    base_size: int = BASE_SELFIES_TARGET,
    sentinels: int = SENTINEL_COUNT,
    conditioning_size: int = CONDITIONING_TARGET,
) -> TokenizerArtifact:
    """Build the offline default vocabulary.

    With the default arguments this reproduces the paper's headline arithmetic exactly:
    ``5 + 100 + 199 + 154 = 458``.

    Args:
        base_size: Exact size of the base SELFIES block.
        sentinels: Number of sentinel tokens.
        conditioning_size: Exact size of the conditioning block.

    Returns:
        A deterministic :class:`TokenizerArtifact`; two calls produce identical output.
    """
    base_alphabet, base_stats = build_base_alphabet_detailed(None, base_size)
    return _assemble(
        base_alphabet, base_stats, sentinels=sentinels, conditioning_size=conditioning_size
    )


def build_vocab_from_corpus(
    pselfies_iter: Iterable[str],
    target_base_size: int = BASE_SELFIES_TARGET,
    *,
    sentinels: int = SENTINEL_COUNT,
    conditioning_size: int = CONDITIONING_TARGET,
    pre_split: bool = False,
) -> TokenizerArtifact:
    """Build a vocabulary whose base block is ranked by corpus frequency.

    This mirrors how the authors obtained their 199 tokens (most frequent bracketed symbols
    over a large polymer corpus). Ties are broken by ``token asc`` so the result is stable
    across runs, shuffles and platforms. Tokens beyond ``target_base_size`` are dropped and
    the drop count is recorded; a short corpus is padded with ``<unused_k>`` placeholders.

    Args:
        pselfies_iter: Iterable of PSELFIES strings (or of pre-split tokens if
            ``pre_split`` is set). Lines are consumed lazily.
        target_base_size: Exact size of the base SELFIES block.
        sentinels: Number of sentinel tokens.
        conditioning_size: Exact size of the conditioning block.
        pre_split: Treat each item as one already-split bracketed token.

    Returns:
        A deterministic :class:`TokenizerArtifact` with
        ``metadata["base_alphabet"]["source"] == "corpus"``.
    """
    if pre_split:
        tokens: Iterable[str] = pselfies_iter
    else:

        def _iter() -> Iterable[str]:
            for line in pselfies_iter:
                yield from split_pselfies(line)

        tokens = _iter()

    base_alphabet, base_stats = build_base_alphabet_detailed(tokens, target_base_size)
    return _assemble(
        base_alphabet, base_stats, sentinels=sentinels, conditioning_size=conditioning_size
    )
