# src/polyt5/data/tg_metadata.py
"""LamaLab Tg metadata: descriptors, measurement spread, reliability.

:mod:`polyt5.data.prepare` reads two columns of
``data/external/LAMALAB_CURATED_Tg.csv`` -- ``PSMILES`` and ``labels.Exp_Tg(K)``
-- and discards the other 110. This module reads the rest: the 100 precomputed
descriptor columns (30 ``backbonelevel``, 42 ``sidechainlevel``, 28
``fullpolymerlevel``) and the four measurement-provenance columns
``meta.num_of_points``, ``meta.std``, ``meta.reliability``,
``meta.polymer_class``.

**Index alignment is the contract.** Phase 4 reuses the frozen five splits,
whose indices point into the list :func:`polyt5.data.prepare.prepare_labeled_corpus`
produces. So :func:`prepare_labeled_rows` does not reimplement the filter chain
-- it *calls* ``prepare_labeled_corpus`` with each row's position riding along
in the value slot, which the corpus preparer carries through untouched. Same
filters by construction, and
``tests/test_group_a_metadata.py::test_row_order_matches_prepare_labeled_corpus``
pins it against the real corpus.

This module is torch-free: only ``csv``, ``math``, ``numpy``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polyt5.data.prepare import PreparationStats, prepare_labeled_corpus
from polyt5.utils import get_logger

__all__ = [
    "DESCRIPTOR_PREFIXES",
    "PSMILES_COLUMN",
    "TG_COLUMN",
    "TgExample",
    "TgRow",
    "descriptor_columns",
    "descriptor_group",
    "descriptor_matrix",
    "prepare_labeled_rows",
    "read_lamalab_rows",
]

_logger = get_logger("polyt5.data.tg_metadata")

#: Column-name prefix of each descriptor level, keyed by the name we report it under.
DESCRIPTOR_PREFIXES: dict[str, str] = {
    "backbone": "backbonelevel.features.",
    "sidechain": "sidechainlevel.features.",
    "fullpolymer": "fullpolymerlevel.features.",
}

PSMILES_COLUMN = "PSMILES"
TG_COLUMN = "labels.Exp_Tg(K)"


@dataclass(frozen=True)
class TgRow:
    """One CSV row's polymer, label, and everything the baseline discarded.

    Attributes:
        psmiles: Raw PSMILES from the ``PSMILES`` column.
        tg: Experimental Tg in Kelvin; always finite (rows without one are dropped).
        std: Experimental spread across repeated measurements; ``0.0`` for the
            7,088 single-measurement polymers.
        num_of_points: Independent measurements behind ``tg`` (1 to 10).
        reliability: Lower-cased curator flag: ``black``, ``gold``, ``yellow``, ``red``.
        polymer_class: ``meta.polymer_class``, carried for stratified reporting.
        descriptors: The 100 descriptor values, positionally aligned with the
            column list :func:`read_lamalab_rows` returns. A cell that does not
            parse is ``nan`` -- never ``0.0``, which would be an invented
            measurement.
    """

    psmiles: str
    tg: float
    std: float
    num_of_points: int
    reliability: str
    polymer_class: str
    descriptors: tuple[float, ...]


@dataclass(frozen=True)
class TgExample:
    """One prepared corpus entry: its PSELFIES and the row it came from."""

    pselfies: str
    row: TgRow


def descriptor_group(column: str) -> str | None:
    """Return ``"backbone"`` / ``"sidechain"`` / ``"fullpolymer"``, or ``None``.

    Args:
        column: A CSV column name.

    Returns:
        The descriptor level the column belongs to, or ``None`` when it is not
        a descriptor column at all.
    """
    for name, prefix in DESCRIPTOR_PREFIXES.items():
        if column.startswith(prefix):
            return name
    return None


def descriptor_columns(header: Sequence[str]) -> list[str]:
    """Descriptor column names, in header order.

    Args:
        header: The CSV's field names.

    Returns:
        Every column whose name starts with one of :data:`DESCRIPTOR_PREFIXES`.
    """
    return [column for column in header if descriptor_group(column) is not None]


def _to_float(value: object, default: float = math.nan) -> float:
    """Parse a CSV cell to a finite float, or return ``default``."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def read_lamalab_rows(
    path: str | Path, *, limit: int | None = None
) -> tuple[list[TgRow], list[str]]:
    """Read the curated Tg CSV into rows plus the descriptor column names.

    Rows with a missing or non-finite Tg are dropped and counted, matching
    :func:`polyt5.data.prepare.read_lamalab_tg` exactly, so the two loaders
    agree on which rows exist before any polymer filter runs.

    Args:
        path: Path to ``LAMALAB_CURATED_Tg.csv``.
        limit: Stop after this many kept rows (None reads everything).

    Returns:
        ``(rows, descriptor_column_names)``.
    """
    path = Path(path)
    rows: list[TgRow] = []
    n_dropped = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = descriptor_columns(reader.fieldnames or [])
        for record in reader:
            if limit is not None and len(rows) >= limit:
                break
            psmiles = (record.get(PSMILES_COLUMN) or "").strip()
            tg = _to_float(record.get(TG_COLUMN))
            if not psmiles or not math.isfinite(tg):
                n_dropped += 1
                continue
            rows.append(
                TgRow(
                    psmiles=psmiles,
                    tg=tg,
                    std=_to_float(record.get("meta.std"), default=0.0),
                    num_of_points=int(_to_float(record.get("meta.num_of_points"), default=1.0)),
                    reliability=(record.get("meta.reliability") or "").strip().lower(),
                    polymer_class=(record.get("meta.polymer_class") or "").strip(),
                    descriptors=tuple(_to_float(record.get(name)) for name in columns),
                )
            )
    if n_dropped:
        _logger.info(
            "read_lamalab_rows: dropped %d rows with missing/non-finite Tg (%d kept)",
            n_dropped, len(rows),
        )
    return rows, columns


def prepare_labeled_rows(
    rows: Sequence[TgRow],
    *,
    max_tokens: int = 200,
    deduplicate: bool = True,
    tokenizer: object | None = None,
) -> tuple[list[TgExample], PreparationStats]:
    """Run the corpus filter chain over ``rows``, keeping the metadata attached.

    The row's *position* rides in the value slot of
    :func:`polyt5.data.prepare.prepare_labeled_corpus`, which carries values
    through untouched. That makes the filtering literally the same code path as
    the baseline's, so the surviving order matches the frozen splits.

    Args:
        rows: Rows from :func:`read_lamalab_rows`.
        max_tokens: Maximum PSELFIES length. # [PAPER] 200.
        deduplicate: Collapse rows sharing a canonical PSMILES (first wins).
        tokenizer: Optional duck-typed tokenizer for the length filter.

    Returns:
        ``(examples, stats)`` with ``examples`` in surviving corpus order.
    """
    kept, stats = prepare_labeled_corpus(
        ((row.psmiles, float(index)) for index, row in enumerate(rows)),
        max_tokens=max_tokens,
        deduplicate=deduplicate,
        tokenizer=tokenizer,
    )
    return [
        TgExample(pselfies=pselfies, row=rows[int(index)]) for pselfies, index in kept
    ], stats


def descriptor_matrix(examples: Sequence[TgExample]) -> np.ndarray:
    """Stack the descriptor tuples into a ``(n_examples, n_columns)`` array.

    Args:
        examples: Prepared examples.

    Returns:
        A float64 array; missing cells are ``nan`` so a standardizer can drop
        the column rather than impute it.
    """
    if not examples:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray([example.row.descriptors for example in examples], dtype=np.float64)
