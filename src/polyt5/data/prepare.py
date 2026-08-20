"""PSMILES -> PSELFIES conversion, filtering and task-example construction.

This is the corpus preparation pipeline of Sahu et al. (npj AI 2026):

    raw PSMILES ("*CCO*" / "[*]CCO[*]")
        -> star_to_at                  -> "[At]CCO[At]"
        -> RDKit parse + terminus check (exactly two "[At]")
        -> selfies encoding            -> PSELFIES "[At][C][C][O][At]"
        -> length filter (<= 200 tokens)
        -> optional canonical deduplication

Every dropped row is counted in :class:`PreparationStats` -- corpus attrition
must never be invisible -- and the stats dict is written to the processed
dataset's ``stats.json`` sidecar by the preparation scripts.

Task I/O formats follow the paper's SI verbatim: property prediction maps the
bare PSELFIES (no task prefix) to a one-decimal numeric string ("236.0");
Tg-conditioned generation inverts that mapping.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from polyt5.chemistry import psmiles_to_pselfies, star_to_at, validate_psmiles
from polyt5.utils import get_logger

__all__ = [
    "PreparationStats",
    "build_generation_examples",
    "build_prediction_examples",
    "format_property_value",
    "parse_property_value",
    "prepare_labeled_corpus",
    "prepare_pselfies_corpus",
    "read_lamalab_tg",
    "read_pi1m",
]

_logger = get_logger("polyt5.data.prepare")

# One SELFIES token is one bracket group, e.g. "[C]", "[=Branch1]", "[At]".
_SELFIES_TOKEN_RE = re.compile(r"\[[^\]]+\]")

# LamaLab CSV column names (Zenodo record 15210035).
_LAMALAB_PSMILES_COLUMN = "PSMILES"
_LAMALAB_TG_COLUMN = "labels.Exp_Tg(K)"


class _Tokenizer(Protocol):
    """Duck type of the (concurrently developed) PolyT5Tokenizer."""

    def encode(self, text: str, **kwargs) -> list[int]:  # pragma: no cover - protocol
        ...


@dataclass
class PreparationStats:
    """Attrition accounting for one corpus-preparation run.

    The counters partition the input: ``n_input`` always equals
    ``n_parse_failed + n_wrong_termini + n_selfies_failed + n_too_long +
    n_duplicate + n_kept``. Each row is counted in exactly one bucket, checked
    in the order the pipeline applies its filters.

    Attributes:
        n_input: Rows read from the raw source.
        n_parse_failed: Rows RDKit could not parse (includes empty strings).
        n_wrong_termini: Parseable rows whose ``[At]`` count is not exactly 2.
        n_selfies_failed: Valid PSMILES the selfies encoder rejected.
        n_too_long: Rows whose PSELFIES exceeds the token budget.
        n_duplicate: Rows collapsing onto an earlier row's canonical PSMILES.
        n_kept: Rows that survived every filter.
    """

    n_input: int = 0
    n_parse_failed: int = 0
    n_wrong_termini: int = 0
    n_selfies_failed: int = 0
    n_too_long: int = 0
    n_duplicate: int = 0
    n_kept: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return the counters as a plain dict for JSON serialization."""
        return {
            "n_input": self.n_input,
            "n_parse_failed": self.n_parse_failed,
            "n_wrong_termini": self.n_wrong_termini,
            "n_selfies_failed": self.n_selfies_failed,
            "n_too_long": self.n_too_long,
            "n_duplicate": self.n_duplicate,
            "n_kept": self.n_kept,
        }


def _count_tokens(pselfies: str, tokenizer: _Tokenizer | None) -> int:
    """Count tokens in a PSELFIES string.

    With no tokenizer, a token is one SELFIES bracket group. With an injected
    tokenizer, the length of its untruncated encoding is used instead.

    [AMBIGUITY] The paper says "maximum sequence length of 200 tokens" without
    stating whether that is SELFIES symbols or tokenizer pieces, or whether it
    includes EOS. We count SELFIES bracket groups by default and let an
    injected tokenizer override; ``tokenizer.encode`` is called with its
    defaults (which append EOS), so the tokenizer-based count includes EOS.
    """
    if tokenizer is None:
        return len(_SELFIES_TOKEN_RE.findall(pselfies))
    return len(tokenizer.encode(pselfies))


@dataclass
class _RowOutcome:
    """Result of pushing one PSMILES row through the filter chain."""

    pselfies: str | None = None
    canonical: str | None = None
    bucket: str = "n_kept"


def _process_row(
    psmiles: str,
    *,
    max_tokens: int,
    tokenizer: _Tokenizer | None,
    seen: set[str] | None,
) -> _RowOutcome:
    """Classify one raw PSMILES row into exactly one attrition bucket.

    Args:
        psmiles: Raw PSMILES in either star or ``[At]`` notation.
        max_tokens: Token budget for the PSELFIES form.
        tokenizer: Optional injected tokenizer for length counting.
        seen: Set of canonical PSMILES already kept (None disables dedup).
            Mutated in place when the row is kept.

    Returns:
        A :class:`_RowOutcome` naming the stats bucket, with the PSELFIES
        string populated only for kept rows.
    """
    at_form = star_to_at(psmiles) if isinstance(psmiles, str) else ""
    validity = validate_psmiles(at_form)
    if not validity.valid:
        return _RowOutcome(bucket="n_parse_failed")
    if not validity.correct_termini:
        return _RowOutcome(bucket="n_wrong_termini")

    pselfies = psmiles_to_pselfies(at_form)
    if pselfies is None:
        return _RowOutcome(bucket="n_selfies_failed")

    if _count_tokens(pselfies, tokenizer) > max_tokens:
        return _RowOutcome(bucket="n_too_long")

    canonical = validity.canonical_psmiles or at_form
    if seen is not None:
        if canonical in seen:
            return _RowOutcome(bucket="n_duplicate")
        seen.add(canonical)
    return _RowOutcome(pselfies=pselfies, canonical=canonical, bucket="n_kept")


def prepare_pselfies_corpus(
    psmiles_iter: Iterable[str],
    *,
    max_tokens: int = 200,
    deduplicate: bool = True,
    tokenizer: _Tokenizer | None = None,
    progress: bool = False,
) -> tuple[list[str], PreparationStats]:
    """Convert raw PSMILES into a filtered, optionally deduplicated PSELFIES list.

    Args:
        psmiles_iter: Raw PSMILES strings (star or ``[At]`` notation).
        max_tokens: Maximum PSELFIES length; the paper uses 200.
        deduplicate: Collapse rows sharing a canonical PSMILES (first one wins).
        tokenizer: Optional duck-typed tokenizer; when given, its encode length
            replaces the SELFIES bracket-group count for the length filter.
        progress: Show a tqdm progress bar while iterating.

    Returns:
        ``(pselfies_list, stats)`` where ``stats`` partitions the input exactly.
    """
    iterator: Iterable[str] = psmiles_iter
    if progress:
        from tqdm import tqdm

        iterator = tqdm(psmiles_iter, desc="prepare_pselfies_corpus", unit="row")

    stats = PreparationStats()
    seen: set[str] | None = set() if deduplicate else None
    kept: list[str] = []
    for psmiles in iterator:
        stats.n_input += 1
        outcome = _process_row(psmiles, max_tokens=max_tokens, tokenizer=tokenizer, seen=seen)
        setattr(stats, outcome.bucket, getattr(stats, outcome.bucket) + 1)
        if outcome.pselfies is not None:
            kept.append(outcome.pselfies)
    return kept, stats


def prepare_labeled_corpus(
    pairs: Iterable[tuple[str, float]],
    *,
    max_tokens: int = 200,
    deduplicate: bool = True,
    tokenizer: _Tokenizer | None = None,
    progress: bool = False,
) -> tuple[list[tuple[str, float]], PreparationStats]:
    """Run the corpus pipeline on labeled ``(PSMILES, value)`` pairs.

    Identical filters to :func:`prepare_pselfies_corpus`, but the label rides
    along so supervised examples keep their pairing. On canonical duplicates
    the first occurrence (and its label) wins.

    [AMBIGUITY] The paper does not say how duplicate polymers with different
    labels were handled (its dataset is not public). We keep the first
    occurrence and count the rest as ``n_duplicate``.

    Args:
        pairs: ``(psmiles, value)`` tuples.
        max_tokens: Maximum PSELFIES length; the paper uses 200.
        deduplicate: Collapse rows sharing a canonical PSMILES.
        tokenizer: Optional duck-typed tokenizer for length counting.
        progress: Show a tqdm progress bar while iterating.

    Returns:
        ``([(pselfies, value), ...], stats)``.
    """
    iterator: Iterable[tuple[str, float]] = pairs
    if progress:
        from tqdm import tqdm

        iterator = tqdm(pairs, desc="prepare_labeled_corpus", unit="row")

    stats = PreparationStats()
    seen: set[str] | None = set() if deduplicate else None
    kept: list[tuple[str, float]] = []
    for psmiles, value in iterator:
        stats.n_input += 1
        outcome = _process_row(psmiles, max_tokens=max_tokens, tokenizer=tokenizer, seen=seen)
        setattr(stats, outcome.bucket, getattr(stats, outcome.bucket) + 1)
        if outcome.pselfies is not None:
            kept.append((outcome.pselfies, value))
    return kept, stats


def read_pi1m(path: str | Path, *, limit: int | None = None) -> Iterator[str]:
    """Yield PSMILES strings from the PI1M CSV (``SMILES,SA Score``).

    Args:
        path: Path to ``PI1M_v2.csv``.
        limit: Stop after this many rows (None reads everything).

    Yields:
        Star-terminated PSMILES strings from the ``SMILES`` column.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                return
            yield row["SMILES"]


def read_lamalab_tg(path: str | Path, *, limit: int | None = None) -> Iterator[tuple[str, float]]:
    """Yield ``(PSMILES, Tg_in_K)`` pairs from the LamaLab curated Tg CSV.

    Rows with a missing or non-finite ``labels.Exp_Tg(K)`` value are dropped
    and the drop count is logged when iteration finishes.

    Args:
        path: Path to ``LAMALAB_CURATED_Tg.csv``.
        limit: Stop after yielding this many valid pairs (None reads everything).

    Yields:
        ``(psmiles, tg_kelvin)`` tuples with a finite Tg.
    """
    path = Path(path)
    n_dropped = 0
    n_yielded = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if limit is not None and n_yielded >= limit:
                break
            psmiles = (row.get(_LAMALAB_PSMILES_COLUMN) or "").strip()
            raw_tg = (row.get(_LAMALAB_TG_COLUMN) or "").strip()
            try:
                tg = float(raw_tg)
            except ValueError:
                tg = math.nan
            if not psmiles or not math.isfinite(tg):
                n_dropped += 1
                continue
            n_yielded += 1
            yield psmiles, tg
    if n_dropped:
        _logger.info(
            "read_lamalab_tg: dropped %d rows with missing/non-finite Tg (%d yielded)",
            n_dropped,
            n_yielded,
        )


def format_property_value(value: float, decimals: int = 1) -> str:
    """Format a property value the way the paper's targets are written.

    The paper's SI shows one decimal place ("236.0" for Tg, "3.7" for the
    dielectric constant).

    Args:
        value: Numeric property value.
        decimals: Decimal places to keep.

    Returns:
        The fixed-point string, e.g. ``"236.0"``.
    """
    return f"{value:.{decimals}f}"


def parse_property_value(text: str) -> float | None:
    """Parse a model-emitted property string back into a float.

    This is the paper's "invalid or non-numeric outputs are filtered" step:
    anything that does not parse to a finite float returns None (so ``"nan"``
    and ``"inf"`` are rejected too).

    Args:
        text: Model output, e.g. ``"236.0"`` or ``"soluble"``.

    Returns:
        The finite float value, or None.
    """
    if not isinstance(text, str):
        return None
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def build_prediction_examples(
    pairs: Sequence[tuple[str, float]],
    *,
    property_name: str | None = None,
) -> list[tuple[str, str]]:
    """Build property-prediction examples: ``(PSELFIES, "236.0")``.

    The paper fine-tunes one model per thermal/electronic property with NO
    task prefix: the input is the bare PSELFIES and the output the formatted
    number. ``property_name`` is accepted for bookkeeping by callers (it names
    the per-property model/output directory) but deliberately never appears in
    the example text, matching the paper's SI format exactly.

    Args:
        pairs: ``(pselfies, value)`` tuples, already converted and filtered.
        property_name: Optional property label for caller-side bookkeeping;
            not embedded in the examples.

    Returns:
        ``[(source, target), ...]`` with targets formatted to one decimal.
    """
    del property_name  # documented no-op: the paper's format has no prefix
    return [(pselfies, format_property_value(value)) for pselfies, value in pairs]


def build_generation_examples(pairs: Sequence[tuple[str, float]]) -> list[tuple[str, str]]:
    """Build Tg-conditioned generation examples: ``("236.0", PSELFIES)``.

    The exact inverse of :func:`build_prediction_examples`: the input is the
    bare formatted number, the output the PSELFIES string.

    Args:
        pairs: ``(pselfies, value)`` tuples, already converted and filtered.

    Returns:
        ``[(source, target), ...]`` with sources formatted to one decimal.
    """
    return [(format_property_value(value), pselfies) for pselfies, value in pairs]
