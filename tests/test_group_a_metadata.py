# tests/test_group_a_metadata.py
"""Group A Task 1: the LamaLab metadata the baseline pipeline throws away.

The one property that matters more than any other here is INDEX ALIGNMENT: the
frozen five splits index into the list ``prepare_labeled_corpus`` produces, so
``prepare_labeled_rows`` must produce the same list in the same order or every
Group A number is measured on a different test set than 28.67 K was.
"""

from __future__ import annotations

import csv
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

# Synthetic-CSV fixtures for tests that must not depend on what happens to be in
# the real CSV (round-1 review findings 1 and 3): a schema drift, or an
# unparseable descriptor cell, that the real file simply does not contain today.
_SYNTHETIC_BACKBONE_COLUMNS = [f"backbonelevel.features.f{i}" for i in range(30)]
_SYNTHETIC_SIDECHAIN_COLUMNS = [f"sidechainlevel.features.f{i}" for i in range(42)]
_SYNTHETIC_FULLPOLYMER_COLUMNS = [f"fullpolymerlevel.features.f{i}" for i in range(28)]
_SYNTHETIC_DESCRIPTOR_COLUMNS = (
    _SYNTHETIC_BACKBONE_COLUMNS + _SYNTHETIC_SIDECHAIN_COLUMNS + _SYNTHETIC_FULLPOLYMER_COLUMNS
)
_SYNTHETIC_REQUIRED_COLUMNS = [
    "PSMILES",
    "labels.Exp_Tg(K)",
    "meta.std",
    "meta.num_of_points",
    "meta.reliability",
    "meta.polymer_class",
]


def _synthetic_header(descriptor_columns=None):
    """A header with every required column plus the given descriptor columns."""
    if descriptor_columns is None:
        descriptor_columns = _SYNTHETIC_DESCRIPTOR_COLUMNS
    return _SYNTHETIC_REQUIRED_COLUMNS + list(descriptor_columns)


def _synthetic_record(header, overrides=None):
    """A CSV row (as a dict) covering exactly ``header``, valid by default."""
    defaults = {
        "PSMILES": "[At]CC[At]",
        "labels.Exp_Tg(K)": "300.0",
        "meta.std": "0.0",
        "meta.num_of_points": "1",
        "meta.reliability": "gold",
        "meta.polymer_class": "synthetic",
    }
    record = {name: defaults.get(name, "1.0") for name in header}
    if overrides:
        record.update(overrides)
    return record


def _write_csv(path, header, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


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


def test_descriptor_matrix_values_match_independently_authored_descriptors():
    """Non-tautological value check (round-1 review finding 4).

    ``expected`` below is typed out by hand; it is not read back from the
    ``TgRow.descriptors`` the matrix is built from, so this can actually fail
    if :func:`descriptor_matrix` mis-stacks or mis-orders rows or columns.
    """
    row_a = TgRow(
        psmiles="[At]CC[At]",
        tg=300.0,
        std=0.0,
        num_of_points=1,
        reliability="gold",
        polymer_class="vinyl",
        descriptors=(1.0, 2.5, float("nan")),
    )
    row_b = TgRow(
        psmiles="[At]CCO[At]",
        tg=250.0,
        std=1.5,
        num_of_points=2,
        reliability="black",
        polymer_class="ether",
        descriptors=(4.0, -5.5, 6.25),
    )
    examples = [
        TgExample(pselfies="[At][C][C][At]", row=row_a),
        TgExample(pselfies="[At][C][C][O][At]", row=row_b),
    ]

    matrix = descriptor_matrix(examples)

    expected = np.array([[1.0, 2.5, float("nan")], [4.0, -5.5, 6.25]], dtype=np.float64)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, expected, equal_nan=True)


def test_unparseable_tg_cell_is_dropped_not_zeroed():
    rows, _ = read_lamalab_rows(CSV, limit=50)
    assert all(math.isfinite(row.tg) for row in rows), (
        "a row with no Tg must be dropped, not zeroed"
    )


def test_unparseable_descriptor_cell_becomes_nan_not_zero(tmp_path):
    """Round-1 review finding 3: the real CSV has zero unparseable descriptor
    cells, so this path needs a synthetic fixture to be exercised at all.
    """
    header = _synthetic_header()
    bad_row = _synthetic_record(header, overrides={_SYNTHETIC_BACKBONE_COLUMNS[0]: "not-a-number"})
    good_row = _synthetic_record(header, overrides={_SYNTHETIC_BACKBONE_COLUMNS[0]: "3.5"})
    csv_path = _write_csv(tmp_path / "unparseable_descriptor.csv", header, [bad_row, good_row])

    rows, columns = read_lamalab_rows(csv_path)

    assert len(rows) == 2
    index = columns.index(_SYNTHETIC_BACKBONE_COLUMNS[0])
    assert math.isnan(rows[0].descriptors[index]), (
        "an unparseable descriptor cell must become nan, not 0.0"
    )
    assert rows[1].descriptors[index] == pytest.approx(3.5)


def test_read_lamalab_rows_raises_on_missing_required_column(tmp_path):
    """Round-1 review finding 1: a renamed/missing required column must raise,
    not silently drop every row into the 'no Tg' bucket.
    """
    header = [name for name in _synthetic_header() if name != "meta.std"]
    record = _synthetic_record(header)
    csv_path = _write_csv(tmp_path / "missing_required_column.csv", header, [record])

    with pytest.raises(ValueError, match="meta.std"):
        read_lamalab_rows(csv_path)


def test_read_lamalab_rows_raises_on_wrong_descriptor_composition(tmp_path):
    """Round-1 review finding 1: a dropped descriptor column must raise, not
    silently yield a shorter (but still internally consistent) feature matrix.
    """
    truncated_backbone = _SYNTHETIC_BACKBONE_COLUMNS[:-1]  # 29 instead of 30
    header = _synthetic_header(
        truncated_backbone + _SYNTHETIC_SIDECHAIN_COLUMNS + _SYNTHETIC_FULLPOLYMER_COLUMNS
    )
    record = _synthetic_record(header)
    csv_path = _write_csv(tmp_path / "wrong_descriptor_composition.csv", header, [record])

    with pytest.raises(ValueError, match="backbone"):
        read_lamalab_rows(csv_path)


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
