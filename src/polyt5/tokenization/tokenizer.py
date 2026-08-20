"""Deterministic longest-match tokenizer over the predefined polyT5 vocabulary.

Why there is no SentencePiece model here
----------------------------------------
The polyT5 Supplementary Information ends its "Tokenizer vocabulary" section with:

    "To ensure compatibility with the SentencePiece tokenizer framework, all SELFIES tokens
    and additional custom tokens were included as **predefined tokens**."

Every one of the 458 units is predefined, so SentencePiece is used purely as a
serialization/interop wrapper -- there is no learned subword merge table to recover. Running
BPE/unigram training over an alphabet that is already fully enumerated is a no-op at best
and, if it ever split a bracketed SELFIES symbol, actively wrong: the paper's own rule is
that "each substring enclosed within square brackets ... was treated as a distinct token".

This module therefore implements the segmentation directly and imports ``sentencepiece``
nowhere, which also keeps the tokenizer dependency-light enough for the RLVR reward layer to
import standalone. ``scripts/build_tokenizer.py --emit-sentencepiece-vocab`` can still write
a plain ``.vocab`` side file for interop; nothing in this module reads it.

Segmentation algorithm
----------------------
Left-to-right, greedy, deterministic:

1. A ``[...]`` group is atomic -- the paper's explicit rule. The group runs to the first
   ``]``. If the exact group is in the vocabulary it becomes that token; otherwise it
   contributes to an unknown run.
2. A ``<...>`` group whose exact surface form is in the vocabulary (``<extra_id_7>``,
   ``</s>``, ``<pad>``, ``<unk>``, ``<s>``, reserved placeholders) is atomic. This is checked
   before step 3 so that ``<`` and ``<=`` -- which are ordinary conditioning tokens -- still
   work outside that shape.
3. Otherwise, longest match against the non-bracket token table, longest surface form first,
   so ``>=`` beats ``>`` and ``insoluble`` beats no shorter prefix. Matching is exact and
   case-sensitive.
4. Whitespace runs collapse to one separator, and a separator becomes the ``▁`` marker
   **only when it lies between two non-bracket tokens**. A space adjacent to a ``[`` group is
   dropped, because bracket boundaries are already self-delimiting; :meth:`decode`
   reinstates exactly one space at every non-bracket -> bracket boundary. Leading and
   trailing whitespace is dropped.
5. Any maximal run of characters that matches nothing becomes exactly one ``<unk>``.

Nothing in this module raises on adversarial input; unmatched text simply becomes ``<unk>``.

Round-trip fidelity
-------------------
``decode(encode(s)) == s`` holds exactly for every shape the paper's I/O formats use::

    [C][C][Branch1][C][At][C][At]
    236.0
    property 4.1; polymer [C][C][Branch1][C][At][C][At]
    polymer [C][C][At]; solvent [C][C][C][O][C][Ring1][Branch1]
    soluble / insoluble

It is lossy, by design, only for whitespace the rules above discard:

* whitespace between two bracket groups (``"[C] [C]"`` -> ``"[C][C]"``),
* whitespace at a bracket -> non-bracket boundary (``"[C] ;"`` -> ``"[C];"``),
* leading/trailing whitespace, and runs of >1 whitespace character (collapsed),
* unknown text, which is irrecoverably replaced by ``<unk>``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyt5.tokenization.vocab import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPACE_TOKEN,
    UNK_TOKEN,
)

__all__ = ["ARTIFACT_VERSION", "PolyT5Tokenizer", "TokenizerArtifact"]

#: Schema version of the on-disk JSON artifact. Bump on any breaking layout change.
ARTIFACT_VERSION = "1.0.0"

_SENTINEL_RE = re.compile(r"^<extra_id_(\d+)>$")
_RESERVED_RE = re.compile(r"^<(?:unused|cond_reserved)_\d+>$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TokenizerArtifact:
    """An immutable, hashable description of one built vocabulary.

    Attributes:
        version: Schema version of the artifact layout.
        tokens: The ordered token list; index == token id.
        metadata: Provenance -- which groups came from the paper, which are substitutes,
            padding/trim counts, the `selfies` version used, and the reproduction note.
    """

    version: str
    tokens: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class PolyT5Tokenizer:
    """Longest-match tokenizer over a fixed, predefined polyT5 vocabulary.

    The token list *is* the model: index equals token id, and the order is the contract
    documented in :mod:`polyt5.tokenization.vocab`.
    """

    def __init__(self, tokens: Sequence[str], metadata: Mapping[str, Any] | None = None) -> None:
        """Initialise from an explicit token list.

        Args:
            tokens: Ordered token surface forms; position ``i`` becomes id ``i``.
            metadata: Optional provenance mapping, copied verbatim onto the instance.

        Raises:
            ValueError: If ``tokens`` is empty, contains duplicates, or is missing one of
                the five required special tokens.
        """
        token_list = list(tokens)
        if not token_list:
            raise ValueError("tokens must not be empty")
        if len(set(token_list)) != len(token_list):
            counts = Counter(token_list)
            dupes = sorted(t for t, c in counts.items() if c > 1)
            raise ValueError(f"duplicate tokens in vocabulary: {dupes[:10]}")

        self._tokens: tuple[str, ...] = tuple(token_list)
        self._ids: dict[str, int] = {tok: i for i, tok in enumerate(token_list)}
        self._metadata: dict[str, Any] = dict(metadata or {})

        required = (PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, BOS_TOKEN, SPACE_TOKEN)
        missing = [t for t in required if t not in self._ids]
        if missing:
            raise ValueError(f"vocabulary is missing required special tokens: {missing}")

        self._pad_id = self._ids[PAD_TOKEN]
        self._eos_id = self._ids[EOS_TOKEN]
        self._unk_id = self._ids[UNK_TOKEN]
        self._bos_id = self._ids[BOS_TOKEN]
        self._space_id = self._ids[SPACE_TOKEN]

        # Sentinels, keyed by their extra_id index so the map is independent of layout.
        self._sentinel_by_index: dict[int, int] = {}
        reserved_ids: set[int] = set()
        plain: set[str] = set()
        for i, tok in enumerate(token_list):
            match = _SENTINEL_RE.match(tok)
            if match is not None:
                self._sentinel_by_index[int(match.group(1))] = i
                continue
            if _RESERVED_RE.match(tok) is not None:
                reserved_ids.add(i)
                continue
            if not _is_bracket(tok) and not _is_angle(tok):
                plain.add(tok)

        self._plain_tokens = plain
        self._max_plain_len = max((len(t) for t in plain), default=0)

        # decode(skip_special_tokens=True) drops these. SPACE is deliberately absent: it is
        # a content-bearing marker that decodes back to a literal space, so skipping it
        # would break round-trip fidelity on the paper's multi-field prompts.
        self._skip_ids: frozenset[int] = frozenset(
            {self._pad_id, self._eos_id, self._unk_id, self._bos_id}
            | set(self._sentinel_by_index.values())
            | reserved_ids
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def default(cls) -> PolyT5Tokenizer:
        """Build the paper-shaped 458-token vocabulary in memory.

        Returns:
            A tokenizer over the default vocabulary (5 specials + 100 sentinels +
            199 base SELFIES + 154 conditioning).
        """
        from polyt5.tokenization.build_vocab import build_default_vocab

        artifact = build_default_vocab()
        return cls(artifact.tokens, artifact.metadata)

    @classmethod
    def from_file(cls, path: str | Path) -> PolyT5Tokenizer:
        """Load a tokenizer from a JSON artifact written by :meth:`save`.

        Args:
            path: Path to the ``*.json`` artifact.

        Returns:
            The reconstructed tokenizer.

        Raises:
            ValueError: If the stored ``sha256`` does not match the stored token list.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokens = list(payload["tokens"])
        stored = payload.get("sha256")
        actual = _sha256_of(tokens)
        if stored is not None and stored != actual:
            raise ValueError(
                f"tokenizer artifact at {path} is corrupt: sha256 {stored} != {actual}"
            )
        return cls(tokens, payload.get("metadata", {}))

    @classmethod
    def from_artifact(cls, artifact: TokenizerArtifact) -> PolyT5Tokenizer:
        """Build a tokenizer from an in-memory :class:`TokenizerArtifact`.

        Args:
            artifact: The artifact to wrap.

        Returns:
            The tokenizer.
        """
        return cls(artifact.tokens, artifact.metadata)

    def to_artifact(self) -> TokenizerArtifact:
        """Return this tokenizer as a serialisable artifact.

        Returns:
            A :class:`TokenizerArtifact` holding the token list and metadata.
        """
        return TokenizerArtifact(ARTIFACT_VERSION, list(self._tokens), dict(self._metadata))

    def save(self, path: str | Path) -> Path:
        """Write the vocabulary to a deterministic JSON artifact.

        The payload is ``{version, sha256, metadata, tokens}`` serialised with sorted keys,
        two-space indent and ``ensure_ascii=False``. No timestamps or host details are
        recorded, so two builds of the same vocabulary produce byte-identical files.

        Args:
            path: Destination ``*.json`` path; parent directories are created.

        Returns:
            The path written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ARTIFACT_VERSION,
            "sha256": self.sha256,
            "metadata": self._metadata,
            "tokens": list(self._tokens),
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        target.write_text(text, encoding="utf-8")
        return target

    def save_sentencepiece_vocab(self, path: str | Path) -> Path:
        """Emit a plain SentencePiece-style ``token\\t0`` side file.

        Purely for interop with tooling that expects a ``.vocab``; nothing in this package
        reads it back, and the log-probability column is a constant ``0`` because every
        token is predefined rather than learned.

        Args:
            path: Destination ``*.vocab`` path; parent directories are created.

        Returns:
            The path written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"{tok}\t0\n" for tok in self._tokens), encoding="utf-8")
        return target

    # -- introspection -----------------------------------------------------

    @property
    def tokens(self) -> tuple[str, ...]:
        """The ordered token surface forms; index equals token id."""
        return self._tokens

    @property
    def metadata(self) -> dict[str, Any]:
        """Provenance metadata carried alongside the vocabulary."""
        return self._metadata

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._tokens)

    @property
    def sha256(self) -> str:
        """SHA-256 over the newline-joined UTF-8 token list."""
        return _sha256_of(self._tokens)

    @property
    def pad_id(self) -> int:
        """Id of ``<pad>``."""
        return self._pad_id

    @property
    def eos_id(self) -> int:
        """Id of ``</s>``."""
        return self._eos_id

    @property
    def unk_id(self) -> int:
        """Id of ``<unk>``."""
        return self._unk_id

    @property
    def bos_id(self) -> int:
        """Id of ``<s>``."""
        return self._bos_id

    @property
    def space_id(self) -> int:
        """Id of the ``▁`` whitespace marker."""
        return self._space_id

    @property
    def decoder_start_token_id(self) -> int:
        """Decoder start id.

        [AMBIGUITY] The paper claims a start-of-sequence marker and we carry ``<s>`` so the
        special-token count is the stated 5, but stock T5 starts decoding from ``<pad>`` and
        every public T5 checkpoint is trained that way. We follow the T5 convention here;
        override at the model layer if a checkpoint disagrees.
        """
        return self._pad_id

    @property
    def sentinel_ids(self) -> list[int]:
        """Sentinel ids in ascending ``extra_id`` order (``<extra_id_0>`` first)."""
        return [self._sentinel_by_index[i] for i in sorted(self._sentinel_by_index)]

    @property
    def all_special_ids(self) -> list[int]:
        """Ids that :meth:`decode` drops when ``skip_special_tokens=True``.

        Covers ``<pad>``, ``</s>``, ``<unk>``, ``<s>``, all sentinels and all reserved
        placeholders. The ``▁`` whitespace marker is intentionally excluded -- it decodes
        back to a literal space and dropping it would corrupt multi-field prompts.
        """
        return sorted(self._skip_ids)

    def sentinel_id(self, i: int) -> int:
        """Return the id of ``<extra_id_{i}>``.

        Args:
            i: Sentinel index, ``0 <= i < 100`` for the default vocabulary.

        Returns:
            The token id.

        Raises:
            IndexError: If no such sentinel exists in this vocabulary.
        """
        if i < 0 or i not in self._sentinel_by_index:
            raise IndexError(
                f"sentinel index {i} out of range; vocabulary has {len(self._sentinel_by_index)}"
            )
        return self._sentinel_by_index[i]

    def token_to_id(self, tok: str) -> int:
        """Map a surface form to its id, falling back to ``<unk>``.

        Args:
            tok: Exact token surface form.

        Returns:
            The token id, or :attr:`unk_id` if unknown. Never raises.
        """
        return self._ids.get(tok, self._unk_id)

    def id_to_token(self, i: int) -> str:
        """Map an id to its surface form.

        Args:
            i: Token id.

        Returns:
            The surface form.

        Raises:
            IndexError: If ``i`` is outside ``[0, vocab_size)``.
        """
        if not 0 <= i < len(self._tokens):
            raise IndexError(f"token id {i} out of range [0, {len(self._tokens)})")
        return self._tokens[i]

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self) -> str:
        return f"PolyT5Tokenizer(vocab_size={self.vocab_size}, sha256={self.sha256[:12]}...)"

    # -- encoding ----------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Segment ``text`` into vocabulary surface forms.

        Args:
            text: Raw input string.

        Returns:
            Token surface forms, with unmatched runs represented as ``<unk>``. Never raises.
        """
        return [self._tokens[i] for i in self._segment(text)]

    def encode(
        self,
        text: str,
        *,
        add_eos: bool = True,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> list[int]:
        """Encode a string to token ids.

        Args:
            text: Raw input string.
            add_eos: Append exactly one ``</s>``.
            max_length: Optional hard cap on the returned length.
            truncation: Whether to enforce ``max_length``. When truncation fires and
                ``add_eos`` is set, the final position is overwritten with ``</s>`` so the
                result is exactly ``max_length`` ids and still terminates properly.

        Returns:
            A plain list of ints. An empty string yields ``[]``, or ``[eos_id]`` when
            ``add_eos`` is set. Never raises on malformed input.
        """
        ids = self._segment(text)
        if add_eos:
            ids.append(self._eos_id)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos and ids:
                ids[-1] = self._eos_id
        return ids

    def batch_encode(
        self,
        texts: Iterable[str],
        *,
        add_eos: bool = True,
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, list[list[int]]]:
        """Encode many strings, optionally right-padding to the longest row.

        Args:
            texts: Iterable of raw input strings.
            add_eos: Append exactly one ``</s>`` per row.
            max_length: Optional hard cap on each row.
            padding: Right-pad every row with ``<pad>`` to the longest row in the batch.
            truncation: Whether to enforce ``max_length``.

        Returns:
            ``{"input_ids": [[int, ...], ...], "attention_mask": [[int, ...], ...]}`` --
            plain Python lists only, so the RLVR reward layer can consume them without
            torch or numpy.
        """
        rows = [
            self.encode(t, add_eos=add_eos, max_length=max_length, truncation=truncation)
            for t in texts
        ]
        masks = [[1] * len(row) for row in rows]
        if padding and rows:
            width = max(len(row) for row in rows)
            for row, mask in zip(rows, masks, strict=True):
                gap = width - len(row)
                if gap:
                    row.extend([self._pad_id] * gap)
                    mask.extend([0] * gap)
        return {"input_ids": rows, "attention_mask": masks}

    # -- decoding ----------------------------------------------------------

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to a string.

        Inverse of :meth:`encode` for well-formed input; see the module docstring for the
        exact set of lossy cases. Ids outside ``[0, vocab_size)`` are ignored rather than
        raising, so partially decoded model output can always be inspected.

        Args:
            ids: Token ids.
            skip_special_tokens: Drop :attr:`all_special_ids` (``<pad>``, ``</s>``,
                ``<unk>``, ``<s>``, sentinels, reserved placeholders). The ``▁`` marker is
                always rendered as a literal space.

        Returns:
            The reconstructed string.
        """
        pieces: list[str] = []
        previous = "start"
        for raw in ids:
            if not isinstance(raw, int) or not 0 <= raw < len(self._tokens):
                continue
            if skip_special_tokens and raw in self._skip_ids:
                continue
            if raw == self._space_id:
                pieces.append(" ")
                previous = "space"
                continue
            surface = self._tokens[raw]
            kind = "bracket" if _is_bracket(surface) else "plain"
            # Rule 4 inverse: reinstate the space that encoding dropped in front of a
            # bracket group. Bracket -> non-bracket boundaries stay tight, matching the
            # paper's "...[At]; solvent ..." prompt shape.
            if kind == "bracket" and previous == "plain":
                pieces.append(" ")
            pieces.append(surface)
            previous = kind
        return "".join(pieces)

    def batch_decode(
        self,
        sequences: Iterable[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of id sequences.

        Args:
            sequences: Iterable of id sequences.
            skip_special_tokens: Forwarded to :meth:`decode`.

        Returns:
            One decoded string per sequence.
        """
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    # -- internals ---------------------------------------------------------

    def _segment(self, text: str) -> list[int]:
        """Run the greedy scanner and resolve whitespace markers.

        Args:
            text: Raw input string.

        Returns:
            Token ids. Never raises.
        """
        if not text:
            return []
        pieces = self._scan(text)

        ids: list[int] = []
        previous_kind: str | None = None
        pending_space = False
        for token_id, kind in pieces:
            if kind == "ws":
                pending_space = True
                continue
            if pending_space:
                pending_space = False
                # Rule 4: a separator survives only between two non-bracket tokens.
                if previous_kind is not None and previous_kind != "bracket" and kind != "bracket":
                    ids.append(self._space_id)
            ids.append(token_id)
            previous_kind = kind
        return ids

    def _scan(self, text: str) -> list[tuple[int, str]]:
        """Greedy left-to-right scan into ``(token_id, kind)`` pairs.

        Args:
            text: Raw input string.

        Returns:
            Pairs whose ``kind`` is one of ``"bracket"``, ``"plain"``, ``"ws"``.
        """
        out: list[tuple[int, str]] = []
        n = len(text)
        i = 0
        unknown_start: int | None = None

        def flush_unknown() -> None:
            nonlocal unknown_start
            if unknown_start is not None:
                out.append((self._unk_id, "plain"))
                unknown_start = None

        while i < n:
            char = text[i]

            # 4. whitespace run
            if char.isspace():
                match = _WHITESPACE_RE.match(text, i)
                assert match is not None
                flush_unknown()
                out.append((-1, "ws"))
                i = match.end()
                continue

            # 1. bracketed SELFIES symbols are atomic (the paper's own rule)
            if char == "[":
                close = text.find("]", i)
                if close != -1:
                    group = text[i : close + 1]
                    known = self._ids.get(group)
                    if known is not None:
                        flush_unknown()
                        out.append((known, "bracket"))
                        i = close + 1
                        continue

            # 2. <...> special forms are atomic when the exact form is known
            if char == "<":
                close = text.find(">", i)
                if close != -1:
                    group = text[i : close + 1]
                    known = self._ids.get(group)
                    if known is not None:
                        flush_unknown()
                        out.append((known, "plain"))
                        i = close + 1
                        continue

            # 3. longest match over the non-bracket table (">=" beats ">")
            span = min(self._max_plain_len, n - i)
            matched = False
            for length in range(span, 0, -1):
                candidate = text[i : i + length]
                if candidate in self._plain_tokens:
                    flush_unknown()
                    out.append((self._ids[candidate], "plain"))
                    i += length
                    matched = True
                    break
            if matched:
                continue

            # 5. unmatched character -> extend the current unknown run
            if unknown_start is None:
                unknown_start = i
            i += 1

        flush_unknown()
        return out


def _is_bracket(token: str) -> bool:
    """Report whether ``token`` is a bracketed SELFIES symbol.

    Args:
        token: Surface form.

    Returns:
        ``True`` for ``[...]`` shapes.
    """
    return len(token) >= 2 and token.startswith("[") and token.endswith("]")


def _is_angle(token: str) -> bool:
    """Report whether ``token`` is an angle-bracketed special form.

    ``<`` and ``<=`` are ordinary conditioning tokens and deliberately do not qualify.

    Args:
        token: Surface form.

    Returns:
        ``True`` for ``<...>`` shapes of length >= 3.
    """
    return len(token) >= 3 and token.startswith("<") and token.endswith(">")


def _sha256_of(tokens: Sequence[str]) -> str:
    """Hash a token list.

    Args:
        tokens: Ordered surface forms.

    Returns:
        Hex SHA-256 digest of the newline-joined UTF-8 encoding.
    """
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
