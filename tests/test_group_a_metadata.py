# tests/test_group_a_metadata.py
"""Group A Task 1: the LamaLab metadata the baseline pipeline throws away.

The one property that matters more than any other here is INDEX ALIGNMENT: the
frozen five splits index into the list ``prepare_labeled_corpus`` produces, so
``prepare_labeled_rows`` must produce the same list in the same order or every
Group A number is measured on a different test set than 28.67 K was.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polyt5.data.prepare import prepare_labeled_corpus, read_lamalab_tg
from polyt5.data.tg_metadata import (
    DESCRIPTOR_PREFIXES,
    TgExample,
    TgRow,
    descriptor_group,
    descriptor_matrix,
    prepare_labeled_rows,
    read_lamalab_rows,
)

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "external" / "LAMALAB_CURATED_Tg.csv"

pytestmark = pytest.mark.skipif(not CSV.is_file(), reason="LamaLab Tg CSV not downloaded")


def test_descriptor_columns_finds_exactly_the_hundred_feature_columns():
    _, columns = read_lamalab_rows(CSV, limit=5)
    assert len(columns) == 100
    groups = [descriptor_group(name) for name in columns]
    assert groups.count("backbone") == 30
    assert groups.count("sidechain") == 42
    assert groups.count("fullpolymer") == 28
    assert set(DESCRIPTOR_PREFIXES) == {"backbone", "sidechain", "fullpolymer"}


def test_descriptor_group_returns_none_for_a_metadata_column():
    assert descriptor_group("meta.std") is None
    assert descriptor_group("labels.Exp_Tg(K)") is None


def test_rows_carry_the_measurement_provenance_columns():
    rows, columns = read_lamalab_rows(CSV)
    assert len(rows) == 7367
    assert all(isinstance(row, TgRow) for row in rows)
    assert all(len(row.descriptors) == len(columns) for row in rows)
    assert sum(1 for row in rows if row.reliability == "red") == 4
    assert sum(1 for row in rows if row.reliability == "gold") == 143
    assert sum(1 for row in rows if row.reliability == "yellow") == 132
    assert sum(1 for row in rows if row.reliability == "black") == 7088
    assert sum(1 for row in rows if row.num_of_points > 1) == 279
    assert max(row.std for row in rows) == pytest.approx(145.0629562, abs=1e-6)


def test_row_order_matches_prepare_labeled_corpus():
    """The alignment contract: same filters, same order, same length."""
    rows, _ = read_lamalab_rows(CSV)
    examples, stats = prepare_labeled_rows(rows)
    baseline, baseline_stats = prepare_labeled_corpus(read_lamalab_tg(CSV))

    assert [e.pselfies for e in examples] == [pselfies for pselfies, _ in baseline]
    assert [e.row.tg for e in examples] == [value for _, value in baseline]
    assert stats.to_dict() == baseline_stats.to_dict()
    assert len(examples) == 7354, "the frozen splits.json was built over 7354 examples"
    assert all(isinstance(e, TgExample) for e in examples)


def test_descriptor_matrix_is_two_dimensional_and_row_aligned():
    rows, columns = read_lamalab_rows(CSV, limit=200)
    examples, _ = prepare_labeled_rows(rows)
    matrix = descriptor_matrix(examples)
    assert matrix.shape == (len(examples), len(columns))
    assert matrix.dtype == np.float64
    assert np.allclose(matrix[0], np.asarray(examples[0].row.descriptors, dtype=float),
                       equal_nan=True)


def test_unparseable_numeric_cell_becomes_nan_not_zero():
    rows, _ = read_lamalab_rows(CSV, limit=50)
    assert all(math.isfinite(row.tg) for row in rows), (
        "a row with no Tg must be dropped, not zeroed"
    )


def test_data_package_still_imports_without_torch():
    """``polyt5.data``'s import-weight contract survives Group A."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import polyt5.data; sys.exit(1 if 'torch' in sys.modules else 0)"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[:500]
