"""T5 span-corruption objective as parameterized by polyT5 (Sahu et al., 2026).

Ground truth from the paper (verbatim):

    "The training objective follows the span corruption strategy introduced in
    the original T5 model. For each polymer sequence, up to 8 non-overlapping
    masked spans (each up to 3 tokens long) were randomly selected to mask up
    to 15% of the input tokens. These spans were replaced with sentinel tokens
    (<extra_id_n>) in the input sequence, and the target sequence was
    constructed by concatenating the masked spans, each prefixed with its
    corresponding sentinel token. The sentinel tokens were assigned in
    increasing numerical order of n and placed such that no two masked spans
    were adjacent, ensuring at least one unmasked token between them."

This parameterization DIFFERS from canonical T5 (corruption rate 15% with MEAN
span length 3 drawn from a distribution, span count derived from sequence
length): here the span count is HARD-CAPPED at 8 and the span length is
HARD-CAPPED at 3 tokens. With <= 8 spans x <= 3 tokens = <= 24 masked tokens,
the 15% cap binds only for sequences longer than 160 tokens; below that the
8 x 3 cap binds.

The paper leaves several details unspecified; every disambiguation we made is
marked ``[AMBIGUITY]`` at the point where it is implemented. Summary:

* order of the three "up to" limits: the 15% budget and the 8 x 3 cap are
  combined into a single token budget FIRST, then span lengths are drawn
  against that budget (final span truncated to fit).
* span length distribution: uniform on {1, ..., max_span_length}.
* the 15% is computed on the UNPADDED sequence (padding happens later, in the
  collator, and never participates in masking).
* sequences too short to host a span are returned unchanged with empty labels
  (n_masked == 0) -- never an exception.
* a trailing EOS token is never masked (``mask_eos=False`` by default).

This module is pure numpy/python: no torch import anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_CONFIG",
    "SpanCorruptionConfig",
    "SpanCorruptionResult",
    "batch_span_corrupt",
    "corruption_statistics",
    "span_corrupt",
]


@dataclass(frozen=True)
class SpanCorruptionConfig:
    """Hyperparameters of the polyT5 span-corruption objective.

    Attributes:
        max_spans: Hard cap on the number of masked spans per sequence
            (paper: 8).
        max_span_length: Hard cap on the length of a single span in tokens
            (paper: 3).
        corruption_rate: Maximum fraction of (maskable) tokens that may be
            masked (paper: 0.15, "up to 15%").
        min_gap: Minimum number of unmasked tokens between consecutive spans
            (paper: "at least one unmasked token between them"). ``0`` would
            permit adjacent spans and is allowed for ablations only.
        add_final_sentinel: Append ``sentinel_ids[n_spans]`` to the target as a
            terminator (standard T5 convention, visible in the paper's
            Figure 2C target).
        mask_eos: [AMBIGUITY] The paper does not say whether the EOS token may
            be masked. Our choice: never mask a trailing EOS (``False``) --
            masking the sequence terminator would teach the encoder that
            sequences can silently continue past their end.
    """

    max_spans: int = 8
    max_span_length: int = 3
    corruption_rate: float = 0.15
    min_gap: int = 1
    add_final_sentinel: bool = True
    mask_eos: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_spans, int) or self.max_spans < 1:
            raise ValueError(f"max_spans must be a positive int, got {self.max_spans!r}")
        if not isinstance(self.max_span_length, int) or self.max_span_length < 1:
            raise ValueError(
                f"max_span_length must be a positive int, got {self.max_span_length!r}"
            )
        if not 0.0 < self.corruption_rate <= 1.0:
            raise ValueError(
                f"corruption_rate must be in (0, 1], got {self.corruption_rate!r}"
            )
        if not isinstance(self.min_gap, int) or self.min_gap < 0:
            raise ValueError(f"min_gap must be a non-negative int, got {self.min_gap!r}")


DEFAULT_CONFIG = SpanCorruptionConfig()


@dataclass(frozen=True)
class SpanCorruptionResult:
    """One corrupted (encoder input, decoder target) pair.

    Attributes:
        input_ids: Corrupted encoder input: each masked span replaced by a
            single sentinel token, everything else unchanged.
        labels: Decoder target: for each span, its sentinel followed by the
            original span tokens; optionally terminated by a final sentinel
            and an EOS. EMPTY when the sequence was too short to corrupt
            (``n_masked == 0``) -- callers must detect and skip such rows.
        spans: ``(start, length)`` pairs in ORIGINAL token index space, sorted
            ascending by start.
        n_masked: Total number of original tokens covered by ``spans``.
        corruption_fraction: ``n_masked / len(original sequence)`` (0.0 for an
            empty sequence). Denominator is the full unpadded sequence length,
            including a protected trailing EOS, so the value is directly
            comparable to the paper's "up to 15% of the input tokens".
    """

    input_ids: list[int]
    labels: list[int]
    spans: tuple[tuple[int, int], ...]
    n_masked: int
    corruption_fraction: float


def _skip_result(tokens: list[int]) -> SpanCorruptionResult:
    """Result for a sequence left untouched (too short / zero budget)."""
    return SpanCorruptionResult(
        input_ids=tokens, labels=[], spans=(), n_masked=0, corruption_fraction=0.0
    )


def _sample_span_lengths(
    budget: int, max_spans: int, max_span_length: int, rng: np.random.Generator
) -> list[int]:
    """Draw span lengths uniformly from {1..max_span_length} against a budget.

    [AMBIGUITY] The paper gives no span-length distribution; canonical T5 uses
    a mean-3 random distribution, but here length is capped at 3, so we choose
    the maximum-entropy option: uniform on {1, ..., max_span_length}. Lengths
    are accumulated until either the token budget or ``max_spans`` is reached;
    the final span is TRUNCATED so the total never exceeds the budget (a
    trailing zero-length span can never appear because the loop stops as soon
    as the budget is consumed).
    """
    lengths: list[int] = []
    total = 0
    while total < budget and len(lengths) < max_spans:
        draw = int(rng.integers(1, max_span_length + 1))
        draw = min(draw, budget - total)
        lengths.append(draw)
        total += draw
    return lengths


def _place_spans(
    lengths: list[int], maskable_len: int, min_gap: int, rng: np.random.Generator
) -> tuple[tuple[int, int], ...]:
    """Place spans of the given lengths uniformly at random, non-adjacently.

    Uses the stars-and-bars "distribute the slack" construction: with ``n``
    spans occupying ``sum(lengths)`` tokens and ``(n - 1) * min_gap`` mandatory
    gap tokens, the remaining ``slack`` free positions are split uniformly at
    random into ``n + 1`` gaps (before, between and after the spans) by
    sampling ``n`` bar positions among ``slack + n`` slots without replacement.
    Every valid placement of the ordered lengths is equally likely -- no greedy
    left-to-right bias.

    If the lengths cannot fit, spans are dropped from the end one at a time
    (deterministic reduction, no resampling loop) until they fit or none
    remain. Never raises, never loops unboundedly.
    """
    while lengths:
        n = len(lengths)
        required = sum(lengths) + (n - 1) * min_gap
        if required <= maskable_len:
            break
        lengths = lengths[:-1]
    if not lengths:
        return ()

    n = len(lengths)
    slack = maskable_len - (sum(lengths) + (n - 1) * min_gap)
    if slack > 0:
        bars = np.sort(rng.choice(slack + n, size=n, replace=False))
        gaps = [int(bars[0])] + [int(bars[i] - bars[i - 1] - 1) for i in range(1, n)]
    else:
        gaps = [0] * n

    spans: list[tuple[int, int]] = []
    pos = 0
    for i, length in enumerate(lengths):
        pos += gaps[i] + (min_gap if i > 0 else 0)
        spans.append((pos, length))
        pos += length
    return tuple(spans)


def _validate_forced_spans(
    spans: Sequence[tuple[int, int]], maskable_len: int, min_gap: int
) -> tuple[tuple[int, int], ...]:
    """Validate the test-only ``_forced_spans`` injection (raises on misuse).

    The "never raise" guarantee applies to the sampling path; forcing invalid
    spans is a programming error in the test itself and fails loudly.
    """
    out = tuple((int(s), int(ln)) for s, ln in spans)
    prev_end: int | None = None
    for start, length in out:
        if length < 1 or start < 0 or start + length > maskable_len:
            raise ValueError(f"forced span {(start, length)} outside maskable region")
        if prev_end is not None and start < prev_end + min_gap:
            raise ValueError(f"forced spans violate min_gap={min_gap}: {out}")
        prev_end = start + length
    return out


def span_corrupt(
    token_ids: Sequence[int],
    *,
    sentinel_ids: Sequence[int],
    config: SpanCorruptionConfig = DEFAULT_CONFIG,
    rng: np.random.Generator,
    eos_id: int | None = None,
    _forced_spans: Sequence[tuple[int, int]] | None = None,
) -> SpanCorruptionResult:
    """Apply polyT5 span corruption to one token sequence.

    This implements OUR disambiguation of the paper's description; each choice
    the paper leaves open is marked ``[AMBIGUITY]`` here or in the helpers.

    Algorithm:
        1. Maskable region: all positions, minus a trailing ``eos_id`` when
           ``config.mask_eos`` is False.
        2. ``budget = min(max_spans * max_span_length,
           floor(corruption_rate * maskable_len))``.
           [AMBIGUITY] Order of the paper's three "up to" limits: we fold the
           15% cap and the 8 x 3 cap into one token budget up front, then draw
           spans against it -- so every limit is enforced simultaneously
           rather than sequentially. [AMBIGUITY] The 15% is computed on the
           UNPADDED sequence; padding is applied later by the collator and is
           never maskable.
        3. If the budget is empty or no span fits, the sequence is returned
           UNCHANGED with empty ``labels`` and ``n_masked=0``.
           [AMBIGUITY] The paper does not say what happens to short sequences;
           we never raise so that dataset iteration cannot crash, and callers
           can detect ``n_masked == 0`` and skip the row.
        4. Span lengths: uniform on {1..max_span_length}, accumulated until the
           budget or ``max_spans`` is hit; final span truncated to the budget.
        5. Placement: uniform among all valid non-adjacent placements via a
           random composition of the slack (see :func:`_place_spans`); spans
           are dropped from the end if they cannot fit. Sorted ascending.
        6. ``input_ids``: span ``i`` collapses to the single token
           ``sentinel_ids[i]``; all other tokens (including a protected EOS)
           pass through.
        7. ``labels``: ``sentinel_ids[i]`` + original span tokens for each
           span, then ``sentinel_ids[n_spans]`` if ``add_final_sentinel``.
           EOS rule: if ``eos_id`` is not None, it is appended as the very
           last label token (after the final sentinel), mirroring the HF T5
           convention where the tokenizer terminates targets with EOS.
        8. Sentinels are assigned in strictly increasing order left-to-right,
           exactly as the paper states.

    Args:
        token_ids: The original (unpadded) token id sequence.
        sentinel_ids: Sentinel token ids, ``sentinel_ids[n]`` == ``<extra_id_n>``.
            If fewer sentinels than spans are available, the span count is
            silently capped so a sentinel (plus final sentinel if configured)
            always exists.
        config: Objective hyperparameters.
        rng: Explicit numpy Generator; the only source of randomness.
        eos_id: Optional EOS token id -- protects a trailing EOS from masking
            (unless ``config.mask_eos``) and is appended to non-empty labels.
        _forced_spans: TEST-ONLY injection hook: bypasses steps 2-5 and uses
            exactly these ``(start, length)`` spans (validated, ascending).
            Used to reproduce the paper's Figure 2C example bit-for-bit.

    Returns:
        A :class:`SpanCorruptionResult`; never raises on degenerate sequences.
    """
    tokens = [int(t) for t in token_ids]
    seq_len = len(tokens)

    # Step 1: maskable region is the prefix that excludes a protected EOS.
    maskable_len = seq_len
    if (
        seq_len > 0
        and eos_id is not None
        and not config.mask_eos
        and tokens[-1] == eos_id
    ):
        maskable_len = seq_len - 1

    if _forced_spans is not None:
        spans = _validate_forced_spans(_forced_spans, maskable_len, config.min_gap)
    else:
        # Step 2: single combined token budget (+1e-9 guards against float
        # error making e.g. 0.15 * 160 floor to 23 instead of 24).
        budget = min(
            config.max_spans * config.max_span_length,
            int(config.corruption_rate * maskable_len + 1e-9),
        )
        # Cap span count by available sentinels (input sentinels + optional
        # final sentinel) so we can never index past ``sentinel_ids``.
        max_spans = min(
            config.max_spans,
            len(sentinel_ids) - (1 if config.add_final_sentinel else 0),
        )
        # Step 3: degenerate cases -> unchanged, empty labels, never raise.
        if budget < 1 or max_spans < 1 or maskable_len < 1:
            return _skip_result(tokens)

        # Steps 4-5: sample lengths, then place them uniformly.
        lengths = _sample_span_lengths(budget, max_spans, config.max_span_length, rng)
        spans = _place_spans(lengths, maskable_len, config.min_gap, rng)
        if not spans:
            return _skip_result(tokens)

    # Steps 6-8: build corrupted input and target with increasing sentinels.
    input_ids: list[int] = []
    labels: list[int] = []
    cursor = 0
    for i, (start, length) in enumerate(spans):
        input_ids.extend(tokens[cursor:start])
        input_ids.append(int(sentinel_ids[i]))
        labels.append(int(sentinel_ids[i]))
        labels.extend(tokens[start : start + length])
        cursor = start + length
    input_ids.extend(tokens[cursor:])
    if config.add_final_sentinel:
        labels.append(int(sentinel_ids[len(spans)]))
    if eos_id is not None:
        labels.append(int(eos_id))

    n_masked = sum(length for _, length in spans)
    return SpanCorruptionResult(
        input_ids=input_ids,
        labels=labels,
        spans=spans,
        n_masked=n_masked,
        corruption_fraction=n_masked / seq_len if seq_len else 0.0,
    )


def batch_span_corrupt(
    sequences: Sequence[Sequence[int]],
    *,
    sentinel_ids: Sequence[int],
    config: SpanCorruptionConfig = DEFAULT_CONFIG,
    rng: np.random.Generator,
    eos_id: int | None = None,
) -> list[SpanCorruptionResult]:
    """Corrupt a batch of sequences with one shared rng (order-dependent).

    Args:
        sequences: Iterable of unpadded token id sequences.
        sentinel_ids: See :func:`span_corrupt`.
        config: See :func:`span_corrupt`.
        rng: Shared generator; results are deterministic given the seed AND
            the order of ``sequences``.
        eos_id: See :func:`span_corrupt`.

    Returns:
        One :class:`SpanCorruptionResult` per input sequence, same order.
    """
    return [
        span_corrupt(seq, sentinel_ids=sentinel_ids, config=config, rng=rng, eos_id=eos_id)
        for seq in sequences
    ]


def corruption_statistics(results: Sequence[SpanCorruptionResult]) -> dict[str, float]:
    """Aggregate statistics for logging that the objective behaves as specified.

    A result with ``n_masked == 0`` counts as "skipped" (sequence was too
    short to host a span) and is excluded from the span-shape statistics but
    included in ``fraction_skipped``.

    Args:
        results: Corruption results, e.g. from :func:`batch_span_corrupt`.

    Returns:
        Dict with ``mean_corruption_fraction``, ``median_corruption_fraction``
        (both over non-skipped results), ``mean_span_count``,
        ``mean_span_length`` (masked tokens per span, over non-skipped
        results) and ``fraction_skipped``. All values are 0.0 when there is
        nothing to aggregate.
    """
    if not results:
        return {
            "mean_corruption_fraction": 0.0,
            "median_corruption_fraction": 0.0,
            "mean_span_count": 0.0,
            "mean_span_length": 0.0,
            "fraction_skipped": 0.0,
        }
    active = [r for r in results if r.n_masked > 0]
    fraction_skipped = 1.0 - len(active) / len(results)
    if not active:
        return {
            "mean_corruption_fraction": 0.0,
            "median_corruption_fraction": 0.0,
            "mean_span_count": 0.0,
            "mean_span_length": 0.0,
            "fraction_skipped": fraction_skipped,
        }
    fractions = np.array([r.corruption_fraction for r in active], dtype=np.float64)
    span_counts = np.array([len(r.spans) for r in active], dtype=np.float64)
    total_masked = sum(r.n_masked for r in active)
    total_spans = sum(len(r.spans) for r in active)
    return {
        "mean_corruption_fraction": float(fractions.mean()),
        "median_corruption_fraction": float(np.median(fractions)),
        "mean_span_count": float(span_counts.mean()),
        "mean_span_length": float(total_masked / total_spans),
        "fraction_skipped": fraction_skipped,
    }
