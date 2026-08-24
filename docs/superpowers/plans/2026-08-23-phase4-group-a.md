# Phase 4 Group A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build five independently-switchable fine-tuning changes (regression head, descriptor auxiliaries, invariance augmentation, reliability weighting, shared-encoder multi-task) plus a seven-configuration ablation runner, so the Tg fine-tuning stage can be measured against the frozen 28.67 ± 0.76 K baseline on the same five splits.

**Architecture:** One shared pretrained encoder wrapped by `PolyT5MultiTask`, which owns the untouched `PolyT5ForConditionalGeneration` backbone plus two new heads on masked-mean-pooled encoder states — a scalar Tg regression head and a 100-column descriptor head. Every switch is a boolean on a `GroupAConfig`, the seven arms are a code table (not YAML prose) so A6 is provably the union of A1–A5, and all data-side work (metadata, standardisation, weighting, augmentation) is torch-free and fitted on train indices only.

**Tech Stack:** PyTorch 2.9, numpy 2.5, RDKit 2026.03.5, selfies 2.1.1 (no `transformers`, no RL library). Existing project modules: `polyt5.model`, `polyt5.data`, `polyt5.chemistry`, `polyt5.training`, `polyt5.inference`, `polyt5.evaluation`, `polyt5.tokenization`, `polyt5.utils`.

**Spec:** `docs/superpowers/specs/2026-08-23-phase4-group-a-design.md`

## Global Constraints

- Run tests with: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`. Never `pip install`.
- Baseline suite: **1069 passed, 1 skipped, 1 xfailed**, and `ruff check .` clean. Nothing moves but what the task adds — after each task the count is 1069 + (that task's new tests), still 1 skipped and 1 xfailed.
- `ruff check .` must pass. Line-length 100. `from __future__ import annotations` at the top of every new module. Google-style docstrings.
- **No `transformers` dependency.** Anywhere, ever.
- `polyt5/rewards/` and `polyt5/chemistry/` stay torch-free. `polyt5.chemistry.enumeration` (Task 4) uses only rdkit.
- `import polyt5.data` and `import polyt5.evaluation` must still not pull in torch. Torch-importing new modules (`polyt5/data/multitask.py`) are **not** re-exported from `polyt5/data/__init__.py`, exactly as `polyt5/data/datasets.py` already is not.
- **Nothing outside `rl/` imports `rl/`.** No Group A module may import `polyt5.rl` or `polyt5.rewards`.
- **Do not add a new package under `src/polyt5/`.** `tests/test_dependency_direction.py::test_supervised_packages_covers_every_package_on_disk` enumerates every directory with an `__init__.py` under `src/polyt5/` and fails on any it has not classified. Every Group A module lands in an *existing* package (`data`, `model`, `training`, `inference`, `evaluation`, `chemistry`).
- **Success is pre-registered at 27.91 K** — the frozen baseline's 28.6733 K minus one standard deviation (0.7591 K). A configuration whose five-split mean MAE lands inside `28.6733 ± 0.7591` is **no effect**, not a small win. Configurations that **hurt** are reported with equal prominence.
- Never retrain, overwrite, or edit `artifacts/baseline/frozen_baseline.json`, `results/tg_prediction_5splits_medium92m/`, or any existing baseline artifact. Group A writes only under `results/group_a/`.
- **A GRPO run may be in flight writing to `results/grpo_control/`.** No task in this plan starts training. Tasks 1–14 are code + tests only; the runner in Task 13 is *written and unit-tested*, not executed.
- **Documentation discipline.** Three claims never merge: *"the paper reports"* (5,130 labels, RMSE 40.82), *"our reproduction obtains"* (28.6733 ± 0.7591 K on 7,367 LamaLab labels), *"our extension obtains"* (Group A). **No Group A configuration has been trained**, so nothing this plan produces may be described as a result. Every emitted matrix row carries an explicit claim category.
- Windows shell: never combine `cd` with a write in one Bash call (`cd X && cat > f`); use absolute paths, or the Write/Edit tools. Never use process substitution `<(...)` or `>(...)`.

## Spec ambiguities resolved here

Three points the spec does not settle. Each is decided once, in code, with the reason attached — an implementer must not re-decide them per task.

1. **Which splits does the `red` drop apply to?** §4.4 says "drop `reliability == red`" without naming a split. Dropping red rows from **test** would change the evaluation set and make MAE incomparable to the frozen 28.6733 K, which is the one number every arm is measured against. **Decision: drop from train and val only; test is never filtered**, and `SplitTensors.n_red_in_test` reports the residual so it is visible rather than hidden. (Task 7.)
2. **Is B0 rerun, or is 28.67 K carried over?** §5's table annotates B0 "already measured: 28.67 K", but §5 also says all seven run on the same five splits. Carrying the number over would leave any harness difference — this plan's `assemble_split`, its example-weighted loss reporting — unattributed. **Decision: B0 is rerun in-harness**, and its result is the in-harness reference; the frozen 28.6733 ± 0.7591 K remains the verdict threshold's source. (Tasks 8, 12, 13.)
3. **7,367 or 7,354?** §1 and §4.4 quote the CSV's 7,367 rows. The frozen splits index into the **7,354** examples that survive corpus preparation (13 lost to parse/terminus/length/dedup filters). **Decision: the corpus size that governs split reuse is 7,354**, asserted by `load_frozen_splits`; 7,367 remains correct as the label count. (Tasks 1, 13.)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/polyt5/data/tg_metadata.py` | Read the 100 descriptor columns + `meta.std` / `meta.num_of_points` / `meta.reliability` / `meta.polymer_class`; keep row order aligned with the frozen corpus |
| `src/polyt5/data/standardize.py` | `Standardizer`: fit on train rows only, drop constant/non-finite columns **and log the drop**, invert at inference |
| `src/polyt5/data/weighting.py` | Drop `reliability == red`; `1/max(std, floor)` weights with a floor that forbids infinite weight |
| `src/polyt5/chemistry/enumeration.py` | Deterministic enumeration of alternative PSELFIES writings of one polymer (rdkit only, torch-free) |
| `src/polyt5/data/augment.py` | Split-safe expansion: every writing carries its `source_index`, so a writing can never cross a split boundary |
| `src/polyt5/model/heads.py` | `masked_mean_pool`, `RegressionHead`, `weighted_huber_loss`, `weighted_lm_loss` |
| `src/polyt5/model/multitask.py` | `PolyT5MultiTask` — backbone + heads, three forward paths, target scaling as buffers |
| `src/polyt5/data/multitask.py` | `TaskItem`, `TaskDataset`, `TaskCollator`, `assemble_split` (torch; **not** re-exported) |
| `src/polyt5/training/group_a.py` | `GroupAConfig` and the seven-arm table `arm_config` |
| `src/polyt5/training/multitask_trainer.py` | `InterleavedLoader`, `GroupATrainer` |
| `src/polyt5/training/cycle.py` | Cycle consistency behind a flag, default OFF |
| `src/polyt5/inference/regression_predictor.py` | `RegressionPropertyPredictor` — inverts standardisation, no beam search |
| `src/polyt5/evaluation/ablation.py` | `ArmResult`, `classify_effect`, `build_ablation_matrix`, `format_ablation_matrix` |
| `src/polyt5/evaluation/generation_regression.py` | Arm-B generation regression check |
| `src/polyt5/training/trainer.py` | *Modified*: extract `_batch_weight` as an override point (behaviour identical) |
| `configs/finetune/group_a.yaml` | Shared hyperparameters; the arm switches come from code, not YAML |
| `scripts/run_group_a.py` | The seven-configuration ablation runner |
| `scripts/check_group_a_generation.py` | Generation regression check CLI |

Tests, one file per task so a reviewer can reject one task without unpicking its neighbour:
`tests/test_group_a_metadata.py`, `test_group_a_standardize.py`, `test_group_a_weighting.py`, `test_group_a_augment.py`, `test_group_a_heads.py`, `test_group_a_multitask_model.py`, `test_group_a_batches.py`, `test_group_a_arms.py`, `test_group_a_trainer.py`, `test_group_a_cycle.py`, `test_group_a_predictor.py`, `test_group_a_ablation.py`, `test_run_group_a.py`, `test_group_a_generation_check.py`.

---

### Task 1: Tg metadata loader with frozen-corpus index alignment

The ablation must reuse `results/tg_prediction_5splits_medium92m/splits.json` verbatim. Those split indices point into the list `prepare_labeled_corpus(read_lamalab_tg(csv))` produces (`n = 7354` from 7,367 CSV rows). So the metadata loader must not reimplement the filter chain — it must produce rows in exactly that order, with exactly that length.

**Files:**
- Create: `src/polyt5/data/tg_metadata.py`
- Modify: `src/polyt5/data/__init__.py` (re-export; the module is torch-free)
- Test: `tests/test_group_a_metadata.py`

**Interfaces:**
- Consumes: `polyt5.data.prepare.prepare_labeled_corpus(pairs: Iterable[tuple[str, float]], *, max_tokens: int, deduplicate: bool, tokenizer) -> tuple[list[tuple[str, float]], PreparationStats]`, `polyt5.data.prepare.read_lamalab_tg`, `polyt5.data.prepare.PreparationStats`, `polyt5.utils.get_logger`
- Produces:
  - `DESCRIPTOR_PREFIXES: dict[str, str]` — `{"backbone": "backbonelevel.features.", "sidechain": "sidechainlevel.features.", "fullpolymer": "fullpolymerlevel.features."}`
  - `TG_COLUMN: str = "labels.Exp_Tg(K)"`, `PSMILES_COLUMN: str = "PSMILES"`
  - `descriptor_columns(header: Sequence[str]) -> list[str]`
  - `descriptor_group(column: str) -> str | None`
  - `TgRow` frozen dataclass: `psmiles: str`, `tg: float`, `std: float`, `num_of_points: int`, `reliability: str`, `polymer_class: str`, `descriptors: tuple[float, ...]`
  - `TgExample` frozen dataclass: `pselfies: str`, `row: TgRow`
  - `read_lamalab_rows(path: str | Path, *, limit: int | None = None) -> tuple[list[TgRow], list[str]]`
  - `prepare_labeled_rows(rows: Sequence[TgRow], *, max_tokens: int = 200, deduplicate: bool = True, tokenizer: object | None = None) -> tuple[list[TgExample], PreparationStats]`
  - `descriptor_matrix(examples: Sequence[TgExample]) -> np.ndarray` — shape `(len(examples), n_descriptor_columns)`, dtype float64

- [ ] **Step 1: Write the failing test**

```python
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
    descriptor_columns,
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
    assert all(math.isfinite(row.tg) for row in rows), "a row with no Tg must be dropped, not zeroed"


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_metadata.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.data.tg_metadata'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

Then add to `src/polyt5/data/__init__.py`, in the existing import block (alphabetical among the `polyt5.data.*` imports) and in `__all__`:

```python
from polyt5.data.tg_metadata import (
    DESCRIPTOR_PREFIXES,
    TgExample,
    TgRow,
    descriptor_columns,
    descriptor_group,
    descriptor_matrix,
    prepare_labeled_rows,
    read_lamalab_rows,
)
```

and the `__all__` entries `"DESCRIPTOR_PREFIXES"`, `"TgExample"`, `"TgRow"`, `"descriptor_columns"`, `"descriptor_group"`, `"descriptor_matrix"`, `"prepare_labeled_rows"`, `"read_lamalab_rows"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_metadata.py -o addopts="" -q`
Expected: PASS — 7 passed

- [ ] **Step 5: Run the full suite and ruff**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1076 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/polyt5/data/tg_metadata.py src/polyt5/data/__init__.py tests/test_group_a_metadata.py
git commit -m "feat(group-a): read LamaLab descriptors and measurement provenance, index-aligned"
```

---

### Task 2: Standardizer — train-split statistics only, logged drops, never imputed

Spec §4.1: "standardise using train-split statistics only, invert at inference". Spec §4.2: "Descriptors are standardised per column on train statistics. Columns constant or absent across the train split are dropped and the drop is logged, never silently imputed." One class serves both: the Tg target is a one-column matrix.

**Files:**
- Create: `src/polyt5/data/standardize.py`
- Modify: `src/polyt5/data/__init__.py`
- Test: `tests/test_group_a_standardize.py`

**Interfaces:**
- Consumes: `numpy`, `polyt5.utils.get_logger`
- Produces:
  - `Standardizer` frozen dataclass with fields `columns: tuple[str, ...]`, `keep_index: tuple[int, ...]`, `mean: tuple[float, ...]`, `std: tuple[float, ...]`, `dropped: tuple[str, ...]`
  - `Standardizer.fit(values: np.ndarray, columns: Sequence[str], *, min_std: float = 1e-8) -> Standardizer` (classmethod)
  - `Standardizer.transform(self, values: np.ndarray) -> np.ndarray` — `(n, n_input_columns) -> (n, n_features)`
  - `Standardizer.inverse_transform(self, values: np.ndarray) -> np.ndarray` — `(n, n_features) -> (n, n_features)`
  - `Standardizer.n_features: int` (property)
  - `Standardizer.to_dict(self) -> dict[str, Any]` / `Standardizer.from_dict(payload: dict[str, Any]) -> Standardizer` (classmethod)
  - `fit_target_standardizer(values: Sequence[float], *, name: str = "Tg") -> Standardizer`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_standardize.py
"""Group A Task 2: train-only standardisation with logged, never-imputed drops."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from polyt5.data.standardize import Standardizer, fit_target_standardizer

TRAIN = np.array(
    [
        [1.0, 5.0, 7.0, np.nan],
        [2.0, 5.0, 9.0, 1.0],
        [3.0, 5.0, 11.0, 2.0],
        [4.0, 5.0, 13.0, 3.0],
    ]
)
COLUMNS = ["varying", "constant", "spread", "has_nan"]


def test_fit_keeps_only_the_informative_columns():
    std = Standardizer.fit(TRAIN, COLUMNS)
    assert std.columns == ("varying", "spread")
    assert std.dropped == ("constant", "has_nan")
    assert std.keep_index == (0, 2)
    assert std.n_features == 2


def test_dropped_columns_are_logged_by_name(caplog):
    """'the drop is logged, never silently imputed' -- the names must appear."""
    with caplog.at_level(logging.INFO, logger="polyt5.data.standardize"):
        Standardizer.fit(TRAIN, COLUMNS)
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "constant" in message
    assert "has_nan" in message


def test_transform_uses_train_statistics_on_unseen_rows():
    std = Standardizer.fit(TRAIN, COLUMNS)
    test = np.array([[2.5, 5.0, 10.0, 99.0]])
    out = std.transform(test)
    assert out.shape == (1, 2)
    # train mean of "varying" is 2.5, so it standardises to exactly 0.
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(0.0)


def test_inverse_transform_round_trips():
    std = Standardizer.fit(TRAIN, COLUMNS)
    forward = std.transform(TRAIN)
    back = std.inverse_transform(forward)
    assert np.allclose(back, TRAIN[:, [0, 2]])


def test_a_nan_column_is_dropped_not_imputed():
    std = Standardizer.fit(TRAIN, COLUMNS)
    assert "has_nan" not in std.columns
    assert not np.isnan(std.transform(TRAIN)).any()


def test_fit_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="columns"):
        Standardizer.fit(TRAIN, ["a", "b"])
    with pytest.raises(ValueError, match="2-D"):
        Standardizer.fit(np.array([1.0, 2.0]), ["a"])


def test_round_trips_through_dict():
    std = Standardizer.fit(TRAIN, COLUMNS)
    assert Standardizer.from_dict(std.to_dict()) == std


def test_target_standardizer_scales_a_scalar_property():
    values = [200.0, 300.0, 400.0, 500.0]
    std = fit_target_standardizer(values)
    assert std.columns == ("Tg",)
    scaled = std.transform(np.asarray(values, dtype=float)[:, None])
    assert scaled.mean() == pytest.approx(0.0)
    assert scaled.std() == pytest.approx(1.0)
    kelvin = std.inverse_transform(scaled)
    assert np.allclose(kelvin.ravel(), values)
    assert std.mean[0] == pytest.approx(350.0)


def test_target_standardizer_refuses_a_constant_property():
    """A zero-variance target would divide by zero at inference."""
    with pytest.raises(ValueError, match="no usable"):
        fit_target_standardizer([310.0, 310.0, 310.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_standardize.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.data.standardize'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/data/standardize.py
"""Per-column standardisation fitted on the train split only.

Raw Kelvin (mu ~ 417, sigma ~ 113) makes the regression loss landscape awkward,
and the 100 LamaLab descriptors span wildly different scales. Both are handled
by the same object: the Tg target is a one-column matrix.

Two rules are load-bearing:

* **Train statistics only.** ``fit`` is called on the train rows of one split;
  ``transform`` is then applied to val and test with those same numbers.
  Fitting on all rows would leak the test set's location and scale.
* **Drop, log, never impute.** A column that is constant or carries any
  non-finite value on the fitting rows is removed and named in the log and in
  :attr:`Standardizer.dropped`. Filling it with a mean would invent
  measurements the CSV does not contain.

Torch-free: numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from polyt5.utils import get_logger

__all__ = ["Standardizer", "fit_target_standardizer"]

_logger = get_logger("polyt5.data.standardize")


@dataclass(frozen=True)
class Standardizer:
    """Column means and standard deviations, plus the columns that were dropped.

    Attributes:
        columns: Names of the kept columns, in output order.
        keep_index: Positions of those columns in the ORIGINAL input matrix, so
            :meth:`transform` can select them from a full-width array.
        mean: Per-kept-column mean from the fitting rows.
        std: Per-kept-column standard deviation from the fitting rows; every
            entry is ``> min_std`` by construction, so division is safe.
        dropped: Names of columns removed as constant or non-finite.
    """

    columns: tuple[str, ...]
    keep_index: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    dropped: tuple[str, ...]

    @property
    def n_features(self) -> int:
        """How many columns survive standardisation."""
        return len(self.columns)

    @classmethod
    def fit(
        cls, values: np.ndarray, columns: Sequence[str], *, min_std: float = 1e-8
    ) -> Standardizer:
        """Fit on ``values``, dropping columns that carry no usable signal.

        Args:
            values: ``(n_rows, n_columns)`` array of the FITTING rows only.
            columns: One name per column of ``values``.
            min_std: Standard deviations at or below this count as constant.

        Returns:
            A fitted :class:`Standardizer`.

        Raises:
            ValueError: If ``values`` is not 2-D, or its width does not match
                ``columns``.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"values must be 2-D (n_rows, n_columns), got shape {array.shape}")
        if array.shape[1] != len(columns):
            raise ValueError(
                f"values has {array.shape[1]} columns but {len(columns)} column names were given"
            )

        finite = np.isfinite(array).all(axis=0)
        # Compute the std only where every value is finite: a column with a NaN
        # would otherwise produce a NaN std and a RuntimeWarning on comparison.
        std = np.zeros(array.shape[1], dtype=np.float64)
        if finite.any():
            std[finite] = array[:, finite].std(axis=0, ddof=0)
        keep = finite & (std > min_std)

        dropped = tuple(
            name for name, kept in zip(columns, keep, strict=True) if not kept
        )
        if dropped:
            _logger.info(
                "Standardizer.fit: dropped %d of %d columns as constant or non-finite on the "
                "fitting rows (never imputed): %s",
                len(dropped), len(columns), ", ".join(dropped),
            )
        keep_index = tuple(int(i) for i in np.flatnonzero(keep))
        return cls(
            columns=tuple(name for name, kept in zip(columns, keep, strict=True) if kept),
            keep_index=keep_index,
            mean=tuple(float(v) for v in array[:, keep].mean(axis=0)),
            std=tuple(float(v) for v in std[keep]),
            dropped=dropped,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Select the kept columns of ``values`` and standardise them.

        Args:
            values: ``(n_rows, n_original_columns)`` array.

        Returns:
            ``(n_rows, n_features)`` standardised array.
        """
        array = np.asarray(values, dtype=np.float64)
        selected = array[:, list(self.keep_index)]
        return (selected - np.asarray(self.mean)) / np.asarray(self.std)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map standardised values back to their original units.

        Args:
            values: ``(n_rows, n_features)`` standardised array.

        Returns:
            ``(n_rows, n_features)`` array in original units.
        """
        array = np.asarray(values, dtype=np.float64)
        return array * np.asarray(self.std) + np.asarray(self.mean)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "columns": list(self.columns),
            "keep_index": list(self.keep_index),
            "mean": list(self.mean),
            "std": list(self.std),
            "dropped": list(self.dropped),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Standardizer:
        """Rebuild a standardizer from :meth:`to_dict` output."""
        return cls(
            columns=tuple(payload["columns"]),
            keep_index=tuple(int(i) for i in payload["keep_index"]),
            mean=tuple(float(v) for v in payload["mean"]),
            std=tuple(float(v) for v in payload["std"]),
            dropped=tuple(payload.get("dropped", ())),
        )


def fit_target_standardizer(values: Sequence[float], *, name: str = "Tg") -> Standardizer:
    """Fit the scalar property standardizer on one split's train targets.

    Args:
        values: Train-split property values in their natural units (Kelvin).
        name: Column name recorded in the standardizer.

    Returns:
        A one-column :class:`Standardizer`.

    Raises:
        ValueError: If the targets have no variance, which would make the
            inverse transform meaningless and the forward one a division by
            zero.
    """
    standardizer = Standardizer.fit(np.asarray(values, dtype=np.float64)[:, None], [name])
    if standardizer.n_features == 0:
        raise ValueError(
            f"no usable variance in {len(values)} target values: a constant target cannot be "
            "standardised, and a model trained on it would predict one number forever"
        )
    return standardizer
```

Then add `from polyt5.data.standardize import Standardizer, fit_target_standardizer` to `src/polyt5/data/__init__.py` and `"Standardizer"`, `"fit_target_standardizer"` to its `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_standardize.py -o addopts="" -q`
Expected: PASS — 9 passed

- [ ] **Step 5: Run the full suite and ruff**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1085 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/polyt5/data/standardize.py src/polyt5/data/__init__.py tests/test_group_a_standardize.py
git commit -m "feat(group-a): train-only standardizer that drops and logs unusable columns"
```

---

### Task 3: Reliability weighting — drop red, floor the weight

Spec §4.4: weight by `1 / max(std, floor)` and drop `reliability == red` (4 rows). The floor exists so a single-measurement polymer (`std == 0`, 7,088 of them) cannot acquire infinite weight. Weights are renormalised to mean 1.0 so switching A4 on does not silently change the effective learning rate — otherwise A4 would be confounded with an LR change and the ablation could not attribute its result.

**Files:**
- Create: `src/polyt5/data/weighting.py`
- Modify: `src/polyt5/data/__init__.py`
- Test: `tests/test_group_a_weighting.py`

**Interfaces:**
- Consumes: `polyt5.data.tg_metadata.TgExample`, `polyt5.utils.get_logger`
- Produces:
  - `RED_RELIABILITY: str = "red"`
  - `DEFAULT_STD_FLOOR: float = 5.6`
  - `drop_red_reliability(examples: Sequence[TgExample]) -> tuple[list[TgExample], list[TgExample]]` — returns `(kept, dropped)`
  - `reliability_weights(examples: Sequence[TgExample], *, floor: float = DEFAULT_STD_FLOOR, normalize: bool = True) -> list[float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_weighting.py
"""Group A Task 3: reliability weighting, and the floor that forbids infinity."""

from __future__ import annotations

import pytest

from polyt5.data.tg_metadata import TgExample, TgRow
from polyt5.data.weighting import (
    DEFAULT_STD_FLOOR,
    RED_RELIABILITY,
    drop_red_reliability,
    reliability_weights,
)


def make(std: float, reliability: str = "black") -> TgExample:
    return TgExample(
        pselfies="[At][C][C][At]",
        row=TgRow(
            psmiles="[At]CC[At]",
            tg=300.0,
            std=std,
            num_of_points=1 if std == 0.0 else 3,
            reliability=reliability,
            polymer_class="Polyolefins",
            descriptors=(1.0, 2.0),
        ),
    )


def test_red_rows_are_dropped_and_returned_for_the_log():
    examples = [make(0.0), make(4.0, RED_RELIABILITY), make(9.0)]
    kept, dropped = drop_red_reliability(examples)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0].row.reliability == RED_RELIABILITY
    assert all(e.row.reliability != RED_RELIABILITY for e in kept)


def test_gold_and_yellow_are_kept():
    kept, dropped = drop_red_reliability([make(1.0, "gold"), make(1.0, "yellow")])
    assert len(kept) == 2
    assert dropped == []


def test_a_single_measurement_polymer_does_not_get_infinite_weight():
    """std == 0 for 7088 of 7367 rows; without the floor every one is infinite."""
    weights = reliability_weights([make(0.0)], normalize=False)
    assert weights == [pytest.approx(1.0 / DEFAULT_STD_FLOOR)]


def test_a_noisy_label_is_downweighted_relative_to_a_precise_one():
    weights = reliability_weights([make(0.0), make(145.0629562)], normalize=False)
    assert weights[0] > weights[1]
    assert weights[1] == pytest.approx(1.0 / 145.0629562)


def test_std_below_the_floor_is_clamped_to_the_floor():
    weights = reliability_weights([make(1.0), make(0.0)], normalize=False)
    assert weights[0] == pytest.approx(weights[1])


def test_normalisation_keeps_the_mean_weight_at_one():
    """A4 must not smuggle in an effective learning-rate change."""
    weights = reliability_weights([make(0.0), make(20.0), make(145.0)])
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert all(w > 0.0 for w in weights)


def test_a_nonpositive_floor_is_refused():
    with pytest.raises(ValueError, match="floor"):
        reliability_weights([make(0.0)], floor=0.0)
    with pytest.raises(ValueError, match="floor"):
        reliability_weights([make(0.0)], floor=-1.0)


def test_empty_input_is_empty_output_not_a_division_by_zero():
    assert reliability_weights([]) == []
    assert drop_red_reliability([]) == ([], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_weighting.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.data.weighting'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/data/weighting.py
"""Per-example training weights from the curated measurement provenance.

Every LamaLab row carries ``meta.num_of_points``, ``meta.std`` and
``meta.reliability``, all of which the baseline fine-tune ignores. 279 polymers
have repeated measurements; their spread has median 5.6 K and reaches 145 K.
A 145 K label is close to noise, and training on it at full weight lets it
compete with labels that are 25x tighter.

Two guards, both deliberate:

* **The floor.** 7,088 of 7,367 rows are single measurements with ``std == 0``.
  Unfloored, ``1/std`` is infinite for every one of them, which is not
  "confident" -- it is "unmeasured". The floor defaults to the observed median
  spread, 5.6 K, so a single measurement is treated as typical rather than
  perfect.
* **Renormalisation.** Weights are rescaled to mean 1.0, so switching this on
  does not also change the effective learning rate. Without it, arm A4 would
  confound label weighting with an LR change and its result would be
  unattributable.

``reliability == red`` rows (4 of them) are dropped, and the dropped rows are
returned rather than discarded so the caller can log what left.

Torch-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from polyt5.data.tg_metadata import TgExample
from polyt5.utils import get_logger

__all__ = [
    "DEFAULT_STD_FLOOR",
    "RED_RELIABILITY",
    "drop_red_reliability",
    "reliability_weights",
]

_logger = get_logger("polyt5.data.weighting")

#: Curator flag marking a measurement the LamaLab curation distrusts.
RED_RELIABILITY = "red"

#: Median experimental spread over the 279 repeatedly measured polymers (K).
#: Used as the weight floor: a single measurement is typical, not perfect.
DEFAULT_STD_FLOOR = 5.6


def drop_red_reliability(
    examples: Sequence[TgExample],
) -> tuple[list[TgExample], list[TgExample]]:
    """Split examples into those to train on and those flagged ``red``.

    Args:
        examples: Prepared examples.

    Returns:
        ``(kept, dropped)``, both in input order. ``dropped`` is returned
        rather than thrown away so the caller can record what left and why.
    """
    kept: list[TgExample] = []
    dropped: list[TgExample] = []
    for example in examples:
        (dropped if example.row.reliability == RED_RELIABILITY else kept).append(example)
    if dropped:
        _logger.info(
            "drop_red_reliability: dropped %d of %d rows flagged %r",
            len(dropped), len(examples), RED_RELIABILITY,
        )
    return kept, dropped


def reliability_weights(
    examples: Sequence[TgExample],
    *,
    floor: float = DEFAULT_STD_FLOOR,
    normalize: bool = True,
) -> list[float]:
    """Weight each example by ``1 / max(std, floor)``.

    Args:
        examples: Prepared examples, already filtered by
            :func:`drop_red_reliability`.
        floor: Lower bound on the standard deviation. Must be positive.
        normalize: Rescale so the weights average exactly 1.0, keeping the
            effective learning rate unchanged when weighting is switched on.

    Returns:
        One weight per example, in input order. Empty input gives empty output.

    Raises:
        ValueError: If ``floor`` is not strictly positive -- an unfloored
            ``std == 0`` row would acquire infinite weight.
    """
    if floor <= 0.0:
        raise ValueError(
            f"floor must be > 0, got {floor}: without a positive floor a single-measurement "
            "polymer (std = 0) acquires infinite weight and dominates every gradient"
        )
    raw = [1.0 / max(float(example.row.std), floor) for example in examples]
    if not raw or not normalize:
        return raw
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]
```

Then add `from polyt5.data.weighting import DEFAULT_STD_FLOOR, RED_RELIABILITY, drop_red_reliability, reliability_weights` to `src/polyt5/data/__init__.py` and the four names to its `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_weighting.py -o addopts="" -q`
Expected: PASS — 8 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1093 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/data/weighting.py src/polyt5/data/__init__.py tests/test_group_a_weighting.py
git commit -m "feat(group-a): reliability weighting with a floor and mean-1 renormalisation"
```

---

### Task 4: Invariance augmentation that cannot cross a split boundary

Spec §4.3: N valid PSELFIES writings per polymer, same target. **"Augmentation must respect split boundaries — every writing of a polymer belongs to the same split as the original. A writing of a train polymer appearing in test would be leakage indistinguishable from memorisation."**

The design makes that structural rather than hoped-for: `augment_indices` takes *one split's index list* and expands only those positions, and every produced item carries `source_index` back to the polymer it came from. The tests include a **negative control** — an augment-then-split ordering — that must be caught, so the guard is proven to have teeth rather than merely passing on correct input.

RDKit gotcha, verified on this machine: `Chem.MolToRandomSmilesVect(mol, n, randomSeed=0)` is **not** deterministic — 0 means "pick a random seed". `randomSeed=7` is. The implementation therefore passes `randomSeed=seed + 1` and a test pins reproducibility at `seed=0`.

**Files:**
- Create: `src/polyt5/chemistry/enumeration.py`, `src/polyt5/data/augment.py`
- Modify: `src/polyt5/chemistry/__init__.py`, `src/polyt5/data/__init__.py`
- Test: `tests/test_group_a_augment.py`

**Interfaces:**
- Consumes: `polyt5.chemistry.canonicalization.canonical_psmiles`, `polyt5.chemistry.conversion._mol_from_psmiles`, `polyt5.chemistry.conversion.psmiles_to_pselfies`, `polyt5.chemistry.conversion.pselfies_to_psmiles`, `polyt5.data.prepare._count_tokens`, `polyt5.data.tg_metadata.TgExample`
- Produces:
  - `enumerate_pselfies_writings(psmiles: str, n_writings: int, *, seed: int = 0) -> list[str]`
  - `AugmentedExample` frozen dataclass: `pselfies: str`, `source_index: int`, `is_original: bool`
  - `augment_indices(examples: Sequence[TgExample], indices: Iterable[int], *, n_writings: int, seed: int = 0, max_tokens: int = 200, tokenizer: object | None = None) -> list[AugmentedExample]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_augment.py
"""Group A Task 4: invariance augmentation, and the split boundary it must not cross."""

from __future__ import annotations

import pytest

from polyt5.chemistry import canonical_psmiles, pselfies_to_psmiles
from polyt5.chemistry.enumeration import enumerate_pselfies_writings
from polyt5.data.augment import AugmentedExample, augment_indices
from polyt5.data.tg_metadata import TgExample, TgRow

PSMILES = [
    "[*]CC(C)(C(=O)OC)[*]",
    "[*]CCO[*]",
    "[*]CC(c1ccccc1)[*]",
    "[*]CC(Cl)[*]",
    "[*]CC(C#N)[*]",
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]",
]


def build_examples() -> list[TgExample]:
    from polyt5.chemistry import psmiles_to_pselfies, star_to_at

    out: list[TgExample] = []
    for index, psmiles in enumerate(PSMILES):
        pselfies = psmiles_to_pselfies(star_to_at(psmiles))
        assert pselfies is not None, psmiles
        out.append(
            TgExample(
                pselfies=pselfies,
                row=TgRow(
                    psmiles=psmiles, tg=300.0 + index, std=0.0, num_of_points=1,
                    reliability="black", polymer_class="test", descriptors=(float(index),),
                ),
            )
        )
    return out


def canonical_of(pselfies: str) -> str | None:
    psmiles = pselfies_to_psmiles(pselfies)
    return None if psmiles is None else canonical_psmiles(psmiles)


# ------------------------------------------------------------------- enumeration
def test_enumeration_produces_several_distinct_writings_of_one_polymer():
    writings = enumerate_pselfies_writings("[*]CC(C)(C(=O)OC)[*]", 5, seed=3)
    assert len(writings) >= 2
    assert len(set(writings)) == len(writings)


def test_every_writing_canonicalises_back_to_the_same_polymer():
    reference = canonical_psmiles("[*]CC(C)(C(=O)OC)[*]")
    for writing in enumerate_pselfies_writings("[*]CC(C)(C(=O)OC)[*]", 6, seed=3):
        assert canonical_of(writing) == reference


def test_enumeration_is_deterministic_including_at_seed_zero():
    """RDKit's randomSeed=0 means 'pick a random seed'; we must not pass it through."""
    first = enumerate_pselfies_writings("[*]CC(c1ccccc1)[*]", 6, seed=0)
    second = enumerate_pselfies_writings("[*]CC(c1ccccc1)[*]", 6, seed=0)
    assert first == second
    assert first != enumerate_pselfies_writings("[*]CC(c1ccccc1)[*]", 6, seed=1) or len(first) <= 1


def test_enumeration_returns_empty_on_junk_instead_of_raising():
    assert enumerate_pselfies_writings("not a molecule at all", 4) == []
    assert enumerate_pselfies_writings("", 4) == []


def test_enumeration_refuses_a_nonpositive_count():
    with pytest.raises(ValueError, match="n_writings"):
        enumerate_pselfies_writings("[*]CCO[*]", 0)


# ------------------------------------------------------------------ augmentation
def test_n_writings_one_reproduces_the_unaugmented_corpus_exactly():
    examples = build_examples()
    out = augment_indices(examples, [0, 2, 4], n_writings=1)
    assert [a.pselfies for a in out] == [examples[i].pselfies for i in (0, 2, 4)]
    assert all(a.is_original for a in out)
    assert [a.source_index for a in out] == [0, 2, 4]


def test_augmentation_multiplies_the_train_set():
    examples = build_examples()
    out = augment_indices(examples, [0, 1, 2], n_writings=4, seed=5)
    assert len(out) > 3
    assert all(isinstance(a, AugmentedExample) for a in out)
    assert sum(1 for a in out if a.is_original) == 3


def test_every_writing_points_back_at_a_requested_index():
    examples = build_examples()
    train = [0, 1, 2, 3]
    out = augment_indices(examples, train, n_writings=4, seed=5)
    assert {a.source_index for a in out} <= set(train)


def test_no_augmented_train_writing_lands_in_the_test_split():
    """The leakage check. The corpus is deduplicated on canonical PSMILES, so a
    train writing sharing a test polymer's canonical form can only mean the
    augmentation crossed the boundary."""
    examples = build_examples()
    train, test = [0, 1, 2], [3, 4, 5]
    train_canonical = {canonical_of(a.pselfies)
                       for a in augment_indices(examples, train, n_writings=6, seed=5)}
    test_canonical = {canonical_of(examples[i].pselfies) for i in test}
    assert train_canonical.isdisjoint(test_canonical)


def test_the_leakage_check_catches_augment_before_split():
    """Negative control: a guard that never fires proves nothing.

    Here augmentation is (wrongly) run over the WHOLE corpus and the result
    sliced afterwards -- the exact ordering mistake spec 4.3 warns about. The
    same assertion as the test above must now fail.
    """
    examples = build_examples()
    everything = augment_indices(examples, range(len(examples)), n_writings=6, seed=5)
    wrong_train = [a for a in everything if a.source_index in {0, 1, 2, 3, 4, 5}][:12]
    test_canonical = {canonical_of(examples[i].pselfies) for i in (3, 4, 5)}
    wrong_canonical = {canonical_of(a.pselfies) for a in wrong_train}
    assert not wrong_canonical.isdisjoint(test_canonical), (
        "the leakage assertion must FAIL on an augment-then-split ordering, or it is "
        "not testing anything"
    )


def test_a_writing_longer_than_the_token_budget_is_dropped():
    examples = build_examples()
    out = augment_indices(examples, [0], n_writings=6, seed=5, max_tokens=4)
    assert all(a.is_original for a in out), "only the original survives a 4-token budget"


def test_out_of_range_index_is_refused():
    examples = build_examples()
    with pytest.raises(IndexError):
        augment_indices(examples, [99], n_writings=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_augment.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.chemistry.enumeration'`

- [ ] **Step 3: Write the enumeration module**

```python
# src/polyt5/chemistry/enumeration.py
"""Deterministic enumeration of alternative writings of one polymer.

One polymer has many valid SMILES writings, and therefore many valid PSELFIES
writings. Register entry A-05 established that canonicalisation collapses them
back to exactly one form, deterministically across processes -- which is what
makes them safe to train on: the model learns Tg is a property of the molecule,
not of the string.

RDKit note, verified on this machine (rdkit 2026.03.5):
``Chem.MolToRandomSmilesVect(mol, n, randomSeed=0)`` is NOT deterministic --
``0`` is RDKit's "choose a seed for me" sentinel. Every call here passes
``randomSeed=seed + 1`` so a caller's ``seed=0`` still reproduces.

Torch-free: rdkit only, like the rest of :mod:`polyt5.chemistry`. Nothing here
raises on adversarial input; an unparseable string yields an empty list.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

from .canonicalization import canonical_psmiles
from .conversion import _mol_from_psmiles, psmiles_to_pselfies

RDLogger.DisableLog("rdApp.*")

__all__ = ["enumerate_pselfies_writings"]

#: How many random SMILES to request per writing wanted. RDKit returns
#: duplicates freely, and some writings fail the SELFIES round trip, so the
#: request is oversubscribed rather than looped.
_OVERSAMPLE = 6


def enumerate_pselfies_writings(psmiles: str, n_writings: int, *, seed: int = 0) -> list[str]:
    """Return up to ``n_writings`` distinct PSELFIES writings of one polymer.

    Args:
        psmiles: A PSMILES string in either terminus notation.
        n_writings: Maximum number of writings to return (>= 1).
        seed: Reproducibility seed. The same ``(psmiles, n_writings, seed)``
            always yields the same list, in the same order.

    Returns:
        Distinct PSELFIES strings, the canonical writing first, each of which
        canonicalises back to the same polymer as ``psmiles``. An empty list
        when ``psmiles`` cannot be parsed. Never raises on bad input.

    Raises:
        ValueError: If ``n_writings`` is below 1. That is a caller bug, not
            adversarial data.
    """
    if n_writings < 1:
        raise ValueError(f"n_writings must be >= 1, got {n_writings}")

    mol = _mol_from_psmiles(psmiles)
    if mol is None:
        return []
    reference = canonical_psmiles(psmiles)
    if reference is None:
        return []

    candidates: list[str] = [reference]
    try:
        candidates.extend(
            Chem.MolToRandomSmilesVect(mol, n_writings * _OVERSAMPLE, randomSeed=seed + 1)
        )
    except Exception:  # RDKit failures are data problems, not caller bugs
        pass

    writings: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if canonical_psmiles(candidate) != reference:
            continue
        pselfies = psmiles_to_pselfies(candidate)
        if pselfies is None or pselfies in seen:
            continue
        seen.add(pselfies)
        writings.append(pselfies)
        if len(writings) >= n_writings:
            break
    return writings
```

Add to `src/polyt5/chemistry/__init__.py`: `from .enumeration import enumerate_pselfies_writings` and `"enumerate_pselfies_writings"` in `__all__`.

- [ ] **Step 4: Write the split-safe augmentation module**

```python
# src/polyt5/data/augment.py
"""Split-safe invariance augmentation.

Training on N writings of one polymer with one target teaches the model that Tg
is a property of the molecule, not of the string, and multiplies the effective
training set at no labelling cost.

**The split boundary is the whole safety story.** :func:`augment_indices` takes
ONE split's index list and expands only those positions; every produced item
carries ``source_index`` back to the polymer it came from. A writing of a train
polymer appearing in test would be leakage indistinguishable from memorisation,
so ``tests/test_group_a_augment.py`` asserts the disjointness AND includes a
negative control that an augment-then-split ordering trips it.

The length filter is the corpus filter -- ``polyt5.data.prepare._count_tokens``
is imported rather than re-implemented, so an augmented writing can never be
admitted under a budget the original corpus would have rejected.

Torch-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from polyt5.chemistry.enumeration import enumerate_pselfies_writings
from polyt5.data.prepare import _count_tokens
from polyt5.data.tg_metadata import TgExample

__all__ = ["AugmentedExample", "augment_indices"]


@dataclass(frozen=True)
class AugmentedExample:
    """One writing of one polymer, with the corpus position it belongs to.

    Attributes:
        pselfies: The PSELFIES writing to train on.
        source_index: Position in the un-augmented example list. Every
            downstream target, descriptor vector and weight is looked up
            through this, so a writing can never acquire another polymer's
            label -- and a writing can never appear in a split its source does
            not belong to.
        is_original: Whether this is the corpus's own writing rather than an
            enumerated alternative.
    """

    pselfies: str
    source_index: int
    is_original: bool


def augment_indices(
    examples: Sequence[TgExample],
    indices: Iterable[int],
    *,
    n_writings: int,
    seed: int = 0,
    max_tokens: int = 200,
    tokenizer: object | None = None,
) -> list[AugmentedExample]:
    """Expand ONE split's indices into up to ``n_writings`` writings each.

    Args:
        examples: The full, un-augmented example list.
        indices: Positions belonging to the split being expanded -- typically
            one split's ``train`` list. Nothing outside this iterable is ever
            read, which is what keeps writings inside their own split.
        n_writings: Maximum writings per polymer, counting the original.
            ``1`` reproduces the un-augmented list exactly.
        seed: Reproducibility seed; each polymer uses ``seed + its index``.
        max_tokens: Token budget an alternative writing must also satisfy.
        tokenizer: Optional duck-typed tokenizer for the length count.

    Returns:
        The expanded list, grouped by source polymer in the order ``indices``
        gives, original writing first within each group.

    Raises:
        ValueError: If ``n_writings`` is below 1.
        IndexError: If an index is out of range for ``examples``. Silently
            skipping it would quietly shrink a split.
    """
    if n_writings < 1:
        raise ValueError(f"n_writings must be >= 1, got {n_writings}")

    out: list[AugmentedExample] = []
    for position in indices:
        example = examples[position]  # IndexError on a bad index, deliberately
        out.append(
            AugmentedExample(pselfies=example.pselfies, source_index=position, is_original=True)
        )
        if n_writings == 1:
            continue

        seen = {example.pselfies}
        for pselfies in enumerate_pselfies_writings(
            example.row.psmiles, n_writings * 2, seed=seed + position
        ):
            if pselfies in seen or _count_tokens(pselfies, tokenizer) > max_tokens:
                continue
            seen.add(pselfies)
            out.append(
                AugmentedExample(
                    pselfies=pselfies, source_index=position, is_original=False
                )
            )
            if len(seen) >= n_writings:
                break
    return out
```

Add to `src/polyt5/data/__init__.py`: `from polyt5.data.augment import AugmentedExample, augment_indices` plus the two `__all__` entries.

- [ ] **Step 5: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_augment.py -o addopts="" -q`
Expected: PASS — 12 passed

- [ ] **Step 6: Confirm the chemistry package is still torch-free**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -c "import sys, polyt5.chemistry; print('torch' in sys.modules)"`
Expected: `False`

- [ ] **Step 7: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1105 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/chemistry/enumeration.py src/polyt5/chemistry/__init__.py src/polyt5/data/augment.py src/polyt5/data/__init__.py tests/test_group_a_augment.py
git commit -m "feat(group-a): split-safe invariance augmentation with a leakage negative control"
```

---

### Task 5: Pooling and the two loss functions

Spec §4.1 gives the pooling and loss verbatim: masked mean over non-pad tokens (T5 has no `[CLS]` and never trained one), Huber rather than MSE because some labels carry up to 145 K of experimental spread and MSE would let those dominate the gradient.

`weighted_lm_loss` is here too, because arm A4 needs a *weighted* text cross-entropy. Its unweighted form is deliberately NOT used for arm B0 — B0 must keep the backbone's own token-mean cross-entropy so it reproduces the baseline exactly; Task 6 pins that.

**Files:**
- Create: `src/polyt5/model/heads.py`
- Modify: `src/polyt5/model/__init__.py`
- Test: `tests/test_group_a_heads.py`

**Interfaces:**
- Consumes: `torch`, `torch.nn`
- Produces:
  - `masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor` — `(B, L, D), (B, L) -> (B, D)`
  - `RegressionHead(nn.Module)` — `__init__(self, d_model: int, n_outputs: int = 1, *, dropout: float = 0.1)`, attribute `n_outputs: int`, `forward(self, pooled: Tensor) -> Tensor` `(B, D) -> (B, n_outputs)`
  - `weighted_huber_loss(predictions: Tensor, targets: Tensor, *, delta: float = 1.0, weights: Tensor | None = None) -> Tensor` — scalar
  - `weighted_lm_loss(logits: Tensor, labels: Tensor, *, weights: Tensor | None = None) -> Tensor` — scalar

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_heads.py
"""Group A Task 5: masked mean pooling, the regression head, and the two losses."""

from __future__ import annotations

import pytest
import torch

from polyt5.model.heads import (
    RegressionHead,
    masked_mean_pool,
    weighted_huber_loss,
    weighted_lm_loss,
)


def test_pooling_ignores_padded_positions():
    """Padding must not drag the pooled vector toward zero."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_pooling_matches_a_plain_mean_when_nothing_is_padded():
    hidden = torch.randn(3, 7, 5)
    mask = torch.ones(3, 7, dtype=torch.long)
    assert torch.allclose(masked_mean_pool(hidden, mask), hidden.mean(dim=1), atol=1e-6)


def test_pooling_of_an_all_pad_row_does_not_divide_by_zero():
    hidden = torch.randn(1, 4, 3)
    pooled = masked_mean_pool(hidden, torch.zeros(1, 4, dtype=torch.long))
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled, torch.zeros(1, 3))


def test_pooling_is_invariant_to_extra_padding():
    hidden = torch.randn(1, 4, 6)
    short = masked_mean_pool(hidden[:, :4], torch.tensor([[1, 1, 1, 1]]))
    padded_hidden = torch.cat([hidden, torch.randn(1, 3, 6)], dim=1)
    long = masked_mean_pool(padded_hidden, torch.tensor([[1, 1, 1, 1, 0, 0, 0]]))
    assert torch.allclose(short, long, atol=1e-6)


def test_regression_head_shapes():
    head = RegressionHead(16, 1, dropout=0.0)
    assert head.n_outputs == 1
    assert head(torch.randn(5, 16)).shape == (5, 1)
    wide = RegressionHead(16, 100, dropout=0.0)
    assert wide(torch.randn(5, 16)).shape == (5, 100)


def test_regression_head_rejects_degenerate_sizes():
    with pytest.raises(ValueError, match="d_model"):
        RegressionHead(0, 1)
    with pytest.raises(ValueError, match="n_outputs"):
        RegressionHead(16, 0)


def test_huber_is_quadratic_near_zero_and_linear_far_out():
    small = weighted_huber_loss(torch.tensor([0.1]), torch.tensor([0.0]), delta=1.0)
    assert small == pytest.approx(0.5 * 0.1**2)
    big = weighted_huber_loss(torch.tensor([100.0]), torch.tensor([0.0]), delta=1.0)
    assert big == pytest.approx(1.0 * (100.0 - 0.5))
    # the whole point of Huber over MSE: a 145 K outlier is not squared
    assert big < weighted_huber_loss(torch.tensor([200.0]), torch.tensor([0.0]), delta=1.0)


def test_uniform_weights_equal_no_weights():
    pred, target = torch.randn(8), torch.randn(8)
    assert weighted_huber_loss(pred, target) == pytest.approx(
        float(weighted_huber_loss(pred, target, weights=torch.ones(8))), abs=1e-6
    )


def test_a_zero_weight_example_does_not_contribute():
    pred = torch.tensor([0.0, 1000.0])
    target = torch.tensor([0.0, 0.0])
    loss = weighted_huber_loss(pred, target, weights=torch.tensor([1.0, 0.0]))
    assert loss == pytest.approx(0.0)


def test_huber_reduces_a_multi_output_target_per_example_first():
    pred = torch.zeros(2, 4)
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    loss = weighted_huber_loss(pred, target, weights=torch.tensor([1.0, 0.0]))
    assert loss == pytest.approx(0.0), "per-example weights must apply after the column mean"


def test_weighted_lm_loss_matches_token_cross_entropy_on_equal_length_rows():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    reference = torch.nn.functional.cross_entropy(
        logits.view(-1, 7), labels.view(-1), ignore_index=-100
    )
    assert weighted_lm_loss(logits, labels) == pytest.approx(float(reference), abs=1e-6)


def test_weighted_lm_loss_ignores_padded_label_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 7)
    padded = torch.tensor([[1, 2, -100, -100]])
    trimmed = torch.tensor([[1, 2]])
    assert weighted_lm_loss(logits, padded) == pytest.approx(
        float(weighted_lm_loss(logits[:, :2], trimmed)), abs=1e-6
    )


def test_weighted_lm_loss_honours_a_zero_weight_row():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    only_first = weighted_lm_loss(logits, labels, weights=torch.tensor([1.0, 0.0]))
    assert only_first == pytest.approx(
        float(weighted_lm_loss(logits[:1], labels[:1])), abs=1e-6
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_heads.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.model.heads'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/model/heads.py
"""Encoder pooling, the regression head, and the two weighted losses.

Tg is currently emitted as TEXT, one character at a time under beam search: the
model must learn digit place-value, can emit non-numeric output, and token
cross-entropy gives it no sense that 236 and 237 are adjacent while 236 and 936
are not. A scalar head on the pooled encoder state removes all three problems.

Design points, each from the Group A spec:

* **Masked mean pooling, not first-token.** T5 has no ``[CLS]`` and never
  trained one, so position 0 carries no summary. The mean is taken over
  non-pad positions only, and an all-pad row pools to zero rather than dividing
  by zero.
* **Huber, not MSE.** Some Tg labels carry up to 145 K of experimental spread.
  Squaring that lets a handful of near-noise labels dominate the gradient.
* **Per-example weights apply after the per-example reduction.** For the
  100-column descriptor head that means the column mean first, then the weight
  -- otherwise a reliability weight would be applied 100 times.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "RegressionHead",
    "masked_mean_pool",
    "weighted_huber_loss",
    "weighted_lm_loss",
]


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool encoder states over non-pad positions.

    Args:
        hidden_states: ``(batch, seq, d_model)`` encoder output.
        attention_mask: ``(batch, seq)`` 1/0 mask; 1 marks a real token.

    Returns:
        ``(batch, d_model)``. A row with no unmasked position pools to zeros
        rather than NaN -- a degenerate input is not a crash.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class RegressionHead(nn.Module):
    """Dropout + linear projection from the pooled encoder state.

    Used twice: width 1 for Tg, width ``n_kept_descriptors`` for the auxiliary
    descriptor targets.
    """

    def __init__(self, d_model: int, n_outputs: int = 1, *, dropout: float = 0.1) -> None:
        """Build the head.

        Args:
            d_model: Encoder hidden size.
            n_outputs: Number of scalar outputs.
            dropout: Dropout applied to the pooled vector.

        Raises:
            ValueError: If ``d_model`` or ``n_outputs`` is below 1.
        """
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        self.n_outputs = int(n_outputs)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, self.n_outputs)
        # Match T5's scheme for a projection out of the residual stream, and
        # start from a zero bias so an untrained head predicts the standardised
        # mean (0.0), i.e. the train mean in Kelvin.
        nn.init.normal_(self.projection.weight, mean=0.0, std=d_model**-0.5)
        nn.init.zeros_(self.projection.bias)

    def forward(self, pooled: Tensor) -> Tensor:
        """Project the pooled state.

        Args:
            pooled: ``(batch, d_model)`` output of :func:`masked_mean_pool`.

        Returns:
            ``(batch, n_outputs)``.
        """
        return self.projection(self.dropout(pooled))


def _apply_example_weights(per_example: Tensor, weights: Tensor | None) -> Tensor:
    """Reduce per-example losses to a scalar, optionally weighted."""
    if weights is None:
        return per_example.mean()
    weight = weights.to(per_example.dtype)
    denominator = weight.sum().clamp(min=torch.finfo(per_example.dtype).eps)
    return (per_example * weight).sum() / denominator


def weighted_huber_loss(
    predictions: Tensor,
    targets: Tensor,
    *,
    delta: float = 1.0,
    weights: Tensor | None = None,
) -> Tensor:
    """Huber loss, meaned over any trailing dimensions, then weighted per example.

    Args:
        predictions: ``(batch,)`` or ``(batch, n_outputs)``.
        targets: Same shape as ``predictions``.
        delta: Huber transition point, in standardised units.
        weights: Optional ``(batch,)`` per-example weights.

    Returns:
        A scalar tensor.
    """
    per_element = nn.functional.huber_loss(
        predictions, targets, reduction="none", delta=delta
    )
    per_example = (
        per_element
        if per_element.dim() <= 1
        else per_element.mean(dim=tuple(range(1, per_element.dim())))
    )
    return _apply_example_weights(per_example, weights)


def weighted_lm_loss(
    logits: Tensor, labels: Tensor, *, weights: Tensor | None = None
) -> Tensor:
    """Token cross-entropy, meaned within each example, then weighted per example.

    With ``weights=None`` and equal token counts per row this equals the
    backbone's own token-mean cross-entropy; with unequal counts it is the
    example-mean instead. Callers that must reproduce the baseline exactly use
    the backbone's loss, not this function -- see
    :meth:`polyt5.model.multitask.PolyT5MultiTask.forward_text`.

    Args:
        logits: ``(batch, tgt_len, vocab)``.
        labels: ``(batch, tgt_len)`` with ``-100`` at ignored positions.
        weights: Optional ``(batch,)`` per-example weights.

    Returns:
        A scalar tensor.
    """
    per_token = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)
    valid = (labels != -100).to(per_token.dtype)
    per_example = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    return _apply_example_weights(per_example, weights)
```

Add to `src/polyt5/model/__init__.py`: `from polyt5.model.heads import RegressionHead, masked_mean_pool, weighted_huber_loss, weighted_lm_loss` plus the four `__all__` entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_heads.py -o addopts="" -q`
Expected: PASS — 13 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1118 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/model/heads.py src/polyt5/model/__init__.py tests/test_group_a_heads.py
git commit -m "feat(group-a): masked mean pooling, regression head, weighted Huber and LM losses"
```

---

### Task 6: `PolyT5MultiTask` — one encoder, three forward paths

Spec §3: one shared encoder receiving gradients from both tasks; a regression head and descriptor heads on the prediction side, the decoder for generation. Spec §4.1: "Generation is untouched" — the generation path must be bit-identical to the bare backbone's, and a test proves it.

Three forward paths, because the seven arms need all three:

| path | used by | objective |
|---|---|---|
| `forward_regression` | A1, A6 | Huber on the pooled scalar head (+ λ·descriptors) |
| `forward_text` | B0, A2, A3, A4, A5 | the backbone's seq2seq CE (+ λ·descriptors, + weights) |
| `forward_generation` | A5, A6 | the backbone's seq2seq CE, untouched |

`forward_text` returns the backbone's own loss verbatim when neither descriptors nor weights are active — that is how B0 reproduces the baseline rather than approximating it.

**Files:**
- Create: `src/polyt5/model/multitask.py`
- Modify: `src/polyt5/model/__init__.py`
- Test: `tests/test_group_a_multitask_model.py`

**Interfaces:**
- Consumes: `polyt5.model.transformer.PolyT5ForConditionalGeneration`, `polyt5.model.transformer.Seq2SeqLMOutput`, `polyt5.model.config.PolyT5Config`, `polyt5.model.heads.{RegressionHead, masked_mean_pool, weighted_huber_loss, weighted_lm_loss}`
- Produces:
  - `MultiTaskConfig` frozen dataclass: `use_regression_head: bool = False`, `n_descriptors: int = 0`, `descriptor_lambda: float = 0.1`, `huber_delta: float = 1.0`, `head_dropout: float = 0.1`; methods `to_dict(self) -> dict[str, Any]`, `from_dict(payload: dict[str, Any]) -> MultiTaskConfig` (classmethod)
  - `HeadOutput` dataclass: `loss: Tensor | None`, `tg_standardised: Tensor | None`, `tg_kelvin: Tensor | None`, `descriptors: Tensor | None`, `tg_loss: Tensor | None`, `descriptor_loss: Tensor | None`
  - `PolyT5MultiTask(nn.Module)`:
    - `__init__(self, backbone: PolyT5ForConditionalGeneration, config: MultiTaskConfig)`
    - attributes `backbone`, `head_config: MultiTaskConfig`, `tg_head: RegressionHead | None`, `descriptor_head: RegressionHead | None`; buffers `tg_mean`, `tg_std` (0-dim float tensors)
    - `config` property `-> PolyT5Config` (delegates to the backbone, so `Trainer.save` writes a loadable `model_config`)
    - `set_target_scaling(self, mean: float, std: float) -> None`
    - `forward_regression(self, input_ids: Tensor, attention_mask: Tensor, *, tg_targets: Tensor | None = None, descriptor_targets: Tensor | None = None, weights: Tensor | None = None) -> HeadOutput`
    - `forward_text(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor, *, descriptor_targets: Tensor | None = None, weights: Tensor | None = None) -> HeadOutput`
    - `forward_generation(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Seq2SeqLMOutput`
    - `predict_tg(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor` — Kelvin, no grad
    - `num_parameters(self, trainable_only: bool = True) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_multitask_model.py
"""Group A Task 6: the shared-encoder wrapper and its three forward paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import HeadOutput, MultiTaskConfig, PolyT5MultiTask
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"


def tiny_backbone(seed: int = 0) -> PolyT5ForConditionalGeneration:
    seed_everything(seed)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    return PolyT5ForConditionalGeneration(config)


def wrap(**kwargs) -> PolyT5MultiTask:
    seed_everything(0)
    return PolyT5MultiTask(
        tiny_backbone(), MultiTaskConfig(head_dropout=0.0, **kwargs)
    )


def batch(n: int = 3, length: int = 6, vocab: int = 458):
    generator = torch.Generator().manual_seed(11)
    return (
        torch.randint(2, vocab, (n, length), generator=generator),
        torch.ones(n, length, dtype=torch.long),
    )


def test_wrapper_exposes_the_backbone_config_so_checkpoints_stay_loadable():
    model = wrap(use_regression_head=True)
    assert isinstance(model.config, PolyT5Config)
    assert model.config is model.backbone.config
    assert PolyT5Config.from_dict(model.config.to_dict()) == model.config


def test_state_dict_is_namespaced_under_backbone():
    model = wrap(use_regression_head=True, n_descriptors=4)
    keys = list(model.state_dict())
    assert any(key.startswith("backbone.") for key in keys)
    assert any(key.startswith("tg_head.") for key in keys)
    assert any(key.startswith("descriptor_head.") for key in keys)
    assert "tg_mean" in keys and "tg_std" in keys


def test_generation_is_untouched():
    """Spec 4.1: 'Generation is untouched.' Bit-identical, not merely similar."""
    backbone = tiny_backbone()
    model = PolyT5MultiTask(backbone, MultiTaskConfig(use_regression_head=True,
                                                      head_dropout=0.0))
    model.eval()
    backbone.eval()
    input_ids, mask = batch()
    labels = torch.randint(2, 458, (3, 5), generator=torch.Generator().manual_seed(12))
    wrapped = model.forward_generation(input_ids, mask, labels)
    bare = backbone(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert torch.equal(wrapped.logits, bare.logits)
    assert wrapped.loss == pytest.approx(float(bare.loss))


def test_forward_text_without_extras_returns_the_backbone_loss_verbatim():
    """B0 must REPRODUCE the baseline, not approximate it."""
    backbone = tiny_backbone()
    model = PolyT5MultiTask(backbone, MultiTaskConfig(head_dropout=0.0))
    model.eval()
    backbone.eval()
    input_ids, mask = batch()
    labels = torch.tensor([[5, 6, -100], [7, 8, 9], [10, -100, -100]])
    out = model.forward_text(input_ids, mask, labels)
    bare = backbone(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert isinstance(out, HeadOutput)
    assert out.loss == pytest.approx(float(bare.loss), abs=1e-7)
    assert out.descriptor_loss is None


def test_regression_forward_produces_one_scalar_per_example():
    model = wrap(use_regression_head=True)
    model.eval()
    input_ids, mask = batch()
    out = model.forward_regression(input_ids, mask)
    assert out.tg_standardised.shape == (3,)
    assert out.tg_kelvin.shape == (3,)
    assert out.loss is None


def test_target_scaling_inverts_at_inference():
    model = wrap(use_regression_head=True)
    model.set_target_scaling(mean=417.0, std=113.0)
    model.eval()
    input_ids, mask = batch()
    out = model.forward_regression(input_ids, mask)
    assert torch.allclose(out.tg_kelvin, out.tg_standardised * 113.0 + 417.0, atol=1e-5)
    assert torch.allclose(model.predict_tg(input_ids, mask), out.tg_kelvin, atol=1e-5)


def test_set_target_scaling_refuses_a_degenerate_std():
    model = wrap(use_regression_head=True)
    with pytest.raises(ValueError, match="std"):
        model.set_target_scaling(mean=417.0, std=0.0)


def test_descriptor_loss_enters_with_lambda():
    model = wrap(use_regression_head=True, n_descriptors=4, descriptor_lambda=0.25)
    model.eval()
    input_ids, mask = batch()
    targets = torch.zeros(3)
    descriptors = torch.zeros(3, 4)
    out = model.forward_regression(
        input_ids, mask, tg_targets=targets, descriptor_targets=descriptors
    )
    assert out.tg_loss is not None and out.descriptor_loss is not None
    assert float(out.loss) == pytest.approx(
        float(out.tg_loss) + 0.25 * float(out.descriptor_loss), abs=1e-6
    )


def test_descriptor_auxiliaries_ride_on_the_text_path_too():
    """Arm A2 is descriptors WITHOUT a regression head; it must still work."""
    model = wrap(n_descriptors=4, descriptor_lambda=0.5)
    model.eval()
    input_ids, mask = batch()
    labels = torch.randint(2, 458, (3, 4), generator=torch.Generator().manual_seed(13))
    out = model.forward_text(input_ids, mask, labels, descriptor_targets=torch.zeros(3, 4))
    assert out.descriptor_loss is not None
    assert out.tg_loss is not None
    assert float(out.loss) == pytest.approx(
        float(out.tg_loss) + 0.5 * float(out.descriptor_loss), abs=1e-6
    )


def test_regression_forward_without_a_head_is_a_loud_error():
    model = wrap()
    input_ids, mask = batch()
    with pytest.raises(RuntimeError, match="use_regression_head"):
        model.forward_regression(input_ids, mask)


def test_gradients_reach_the_shared_encoder_from_the_regression_head():
    model = wrap(use_regression_head=True)
    input_ids, mask = batch()
    out = model.forward_regression(input_ids, mask, tg_targets=torch.zeros(3))
    out.loss.backward()
    encoder_grads = [
        p.grad for p in model.backbone.encoder.parameters() if p.grad is not None
    ]
    assert encoder_grads, "the point of the shared encoder is that it receives this gradient"
    assert any(float(g.abs().sum()) > 0.0 for g in encoder_grads)


def test_multitask_config_round_trips_through_dict():
    config = MultiTaskConfig(use_regression_head=True, n_descriptors=97,
                             descriptor_lambda=0.3, huber_delta=2.0, head_dropout=0.05)
    assert MultiTaskConfig.from_dict(config.to_dict()) == config
    assert MultiTaskConfig.from_dict({"n_descriptors": 5}).n_descriptors == 5


def test_num_parameters_counts_the_heads():
    plain = wrap()
    with_heads = wrap(use_regression_head=True, n_descriptors=8)
    assert with_heads.num_parameters() > plain.num_parameters()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_multitask_model.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.model.multitask'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/model/multitask.py
"""One pretrained encoder, a regression head, descriptor heads, and the decoder.

Today prediction and generation are separate fine-tunes from the same
pretrained checkpoint -- siblings that share nothing after pretraining, so
anything learned about what makes a polymer high-Tg is invisible to the
generator. That separation follows the paper training them as distinct tasks,
not any property of T5, which is a multi-task architecture by design::

    pretrained polyT5 encoder      (shared; gradients from both tasks)
    |-- regression head   -> Tg as a scalar           (prediction)
    |-- descriptor heads  -> the 100 LamaLab features (auxiliary)
    `-- decoder           -> PSELFIES                 (generation)

The wrapper OWNS the backbone rather than subclassing it, so:

* ``state_dict`` keys are namespaced under ``backbone.``. A Group A checkpoint
  therefore fails loudly in :class:`polyt5.inference.PolyT5PropertyPredictor`
  instead of half-loading -- Group A produces new models ALONGSIDE the existing
  ones and must never be mistaken for them.
* :attr:`config` still returns the backbone's :class:`PolyT5Config`, so
  ``Trainer.save`` records a ``model_config`` that rebuilds the backbone.
* :meth:`forward_generation` delegates unchanged. Generation is untouched.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from polyt5.model.config import PolyT5Config
from polyt5.model.heads import (
    RegressionHead,
    masked_mean_pool,
    weighted_huber_loss,
    weighted_lm_loss,
)
from polyt5.model.transformer import PolyT5ForConditionalGeneration, Seq2SeqLMOutput

__all__ = ["HeadOutput", "MultiTaskConfig", "PolyT5MultiTask"]


@dataclass(frozen=True)
class MultiTaskConfig:
    """Which heads exist and how their losses combine.

    Attributes:
        use_regression_head: Attach the scalar Tg head. When ``False`` the Tg
            objective is the backbone's text decode, as in the baseline.
        n_descriptors: Width of the auxiliary descriptor head; ``0`` disables it.
            This is the number of columns the train-split standardizer KEPT,
            not the raw 100.
        descriptor_lambda: Weight of the descriptor term in
            ``L = L_Tg + lambda * L_descriptors``. 100 auxiliary targets against
            one Tg target can swamp the objective we care about, so this is
            configurable and its sensitivity is reported.
        huber_delta: Huber transition point, in standardised units.
        head_dropout: Dropout on the pooled vector before each head.
    """

    use_regression_head: bool = False
    n_descriptors: int = 0
    descriptor_lambda: float = 0.1
    huber_delta: float = 1.0
    head_dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, stored in the run config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiTaskConfig:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass
class HeadOutput:
    """What one prediction-side forward pass produced.

    Attributes:
        loss: The combined objective, or ``None`` when no targets were given.
        tg_standardised: ``(batch,)`` head output in standardised units, or
            ``None`` on the text path (where Tg is decoded, not regressed).
        tg_kelvin: ``tg_standardised`` mapped back to Kelvin, or ``None``.
        descriptors: ``(batch, n_descriptors)`` head output, or ``None``.
        tg_loss: The Tg term alone.
        descriptor_loss: The descriptor term alone, BEFORE ``descriptor_lambda``.
    """

    loss: Tensor | None
    tg_standardised: Tensor | None
    tg_kelvin: Tensor | None
    descriptors: Tensor | None
    tg_loss: Tensor | None
    descriptor_loss: Tensor | None


class PolyT5MultiTask(nn.Module):
    """A polyT5 backbone plus the Group A prediction heads."""

    def __init__(
        self, backbone: PolyT5ForConditionalGeneration, config: MultiTaskConfig
    ) -> None:
        """Wrap ``backbone``.

        Args:
            backbone: A pretrained or freshly built conditional-generation model.
            config: Which heads to attach and how to combine their losses.
        """
        super().__init__()
        self.backbone = backbone
        self.head_config = config
        d_model = backbone.config.d_model
        self.tg_head = (
            RegressionHead(d_model, 1, dropout=config.head_dropout)
            if config.use_regression_head
            else None
        )
        self.descriptor_head = (
            RegressionHead(d_model, config.n_descriptors, dropout=config.head_dropout)
            if config.n_descriptors > 0
            else None
        )
        # Buffers, so the target scaling travels inside state_dict and an
        # inference-time inverse transform can never silently use the wrong
        # numbers.
        self.register_buffer("tg_mean", torch.zeros(()))
        self.register_buffer("tg_std", torch.ones(()))

    @property
    def config(self) -> PolyT5Config:
        """The BACKBONE's config, so a saved checkpoint can rebuild the model."""
        return self.backbone.config

    def set_target_scaling(self, mean: float, std: float) -> None:
        """Record the train-split target statistics used to invert predictions.

        Args:
            mean: Train-split mean Tg in Kelvin.
            std: Train-split standard deviation in Kelvin.

        Raises:
            ValueError: If ``std`` is not finite and positive.
        """
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"target std must be finite and > 0, got {std}")
        with torch.no_grad():
            self.tg_mean.fill_(float(mean))
            self.tg_std.fill_(float(std))

    def _pooled(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return masked_mean_pool(
            self.backbone.encode(input_ids, attention_mask=attention_mask), attention_mask
        )

    def _descriptor_term(
        self,
        pooled: Tensor,
        descriptor_targets: Tensor | None,
        weights: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return ``(head_output, unweighted_loss)`` for the descriptor head."""
        if self.descriptor_head is None:
            return None, None
        predictions = self.descriptor_head(pooled)
        if descriptor_targets is None:
            return predictions, None
        loss = weighted_huber_loss(
            predictions,
            descriptor_targets.to(predictions.dtype),
            delta=self.head_config.huber_delta,
            weights=weights,
        )
        return predictions, loss

    def forward_regression(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        tg_targets: Tensor | None = None,
        descriptor_targets: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> HeadOutput:
        """Predict Tg with the scalar head.

        Args:
            input_ids: ``(batch, src_len)`` PSELFIES token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.
            tg_targets: ``(batch,)`` STANDARDISED targets; enables the loss.
            descriptor_targets: ``(batch, n_descriptors)`` standardised targets.
            weights: ``(batch,)`` per-example weights.

        Returns:
            A :class:`HeadOutput`.

        Raises:
            RuntimeError: If no regression head was built.
        """
        if self.tg_head is None:
            raise RuntimeError(
                "forward_regression needs MultiTaskConfig(use_regression_head=True); this "
                "model has no scalar head, so its Tg objective is the text decode"
            )
        pooled = self._pooled(input_ids, attention_mask)
        standardised = self.tg_head(pooled).squeeze(-1)
        descriptors, descriptor_loss = self._descriptor_term(
            pooled, descriptor_targets, weights
        )

        tg_loss: Tensor | None = None
        total: Tensor | None = None
        if tg_targets is not None:
            tg_loss = weighted_huber_loss(
                standardised,
                tg_targets.to(standardised.dtype),
                delta=self.head_config.huber_delta,
                weights=weights,
            )
            total = tg_loss
        if descriptor_loss is not None:
            scaled = self.head_config.descriptor_lambda * descriptor_loss
            total = scaled if total is None else total + scaled

        return HeadOutput(
            loss=total,
            tg_standardised=standardised,
            tg_kelvin=standardised * self.tg_std + self.tg_mean,
            descriptors=descriptors,
            tg_loss=tg_loss,
            descriptor_loss=descriptor_loss,
        )

    def forward_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        *,
        descriptor_targets: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> HeadOutput:
        """Predict Tg as TEXT, the baseline objective, plus optional extras.

        With neither ``descriptor_targets`` nor ``weights``, the returned loss
        is the backbone's own token-mean cross-entropy, unchanged -- that is how
        arm B0 reproduces the frozen baseline rather than approximating it.

        Args:
            input_ids: ``(batch, src_len)`` PSELFIES token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.
            labels: ``(batch, tgt_len)`` numeric-string target ids, ``-100`` padded.
            descriptor_targets: ``(batch, n_descriptors)`` standardised targets.
            weights: ``(batch,)`` per-example weights.

        Returns:
            A :class:`HeadOutput` whose ``tg_standardised`` and ``tg_kelvin``
            are ``None`` -- on this path Tg is decoded, not regressed.
        """
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        tg_loss = (
            output.loss
            if weights is None
            else weighted_lm_loss(output.logits, labels, weights=weights)
        )

        descriptors: Tensor | None = None
        descriptor_loss: Tensor | None = None
        if self.descriptor_head is not None:
            pooled = masked_mean_pool(output.encoder_last_hidden_state, attention_mask)
            descriptors, descriptor_loss = self._descriptor_term(
                pooled, descriptor_targets, weights
            )

        total = tg_loss
        if descriptor_loss is not None:
            total = total + self.head_config.descriptor_lambda * descriptor_loss

        return HeadOutput(
            loss=total,
            tg_standardised=None,
            tg_kelvin=None,
            descriptors=descriptors,
            tg_loss=tg_loss,
            descriptor_loss=descriptor_loss,
        )

    def forward_generation(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> Seq2SeqLMOutput:
        """Tg-conditioned generation, delegated to the backbone untouched.

        Args:
            input_ids: ``(batch, src_len)`` conditioning-number token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.
            labels: ``(batch, tgt_len)`` PSELFIES target ids, ``-100`` padded.

        Returns:
            The backbone's :class:`~polyt5.model.transformer.Seq2SeqLMOutput`.
        """
        return self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    @torch.no_grad()
    def predict_tg(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Predict Tg in KELVIN, with the train-split scaling inverted.

        Args:
            input_ids: ``(batch, src_len)`` PSELFIES token ids.
            attention_mask: ``(batch, src_len)`` 1/0 padding mask.

        Returns:
            ``(batch,)`` predictions in Kelvin.
        """
        was_training = self.training
        self.eval()
        try:
            output = self.forward_regression(input_ids, attention_mask)
        finally:
            self.train(was_training)
        assert output.tg_kelvin is not None  # forward_regression always sets it
        return output.tg_kelvin

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, counting tied weights once.

        Args:
            trainable_only: Count only parameters with ``requires_grad``.

        Returns:
            Total parameter count across backbone and heads.
        """
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        unique = {p.data_ptr(): p for p in params}
        return sum(p.numel() for p in unique.values())
```

Add to `src/polyt5/model/__init__.py`: `from polyt5.model.multitask import HeadOutput, MultiTaskConfig, PolyT5MultiTask` plus the three `__all__` entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_multitask_model.py -o addopts="" -q`
Expected: PASS — 13 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1131 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/model/multitask.py src/polyt5/model/__init__.py tests/test_group_a_multitask_model.py
git commit -m "feat(group-a): shared-encoder multi-task wrapper with three forward paths"
```

---

### Task 7: Batch assembly — one `TaskItem` shape for all seven arms

Every arm produces batches with the same keys, so one collator and one trainer serve all seven and the switches stay independent. `assemble_split` is where the ordering rules live, and they are the correctness story of the whole plan:

1. **Drop `red` from train and val only.** The test split is NEVER filtered. Removing rows from test would change the evaluation set and make MAE incomparable to the frozen 28.6733 K, which is the one thing every arm is measured against. `n_red_in_test` is reported so the residual is visible rather than hidden.
2. **Fit the standardizers on train rows only**, after the red drop, **before** augmentation — augmenting first would let a polymer with many writings drag the mean.
3. **Compute reliability weights on train rows**, before augmentation; each writing inherits its source polymer's weight.
4. **Augment train only.** Val is never augmented (it selects checkpoints) and test never is.
5. **Generation items are built from train polymers only**, and only when multi-task is on.

**Files:**
- Create: `src/polyt5/data/multitask.py` (imports torch; deliberately **not** re-exported from `polyt5/data/__init__.py`)
- Test: `tests/test_group_a_batches.py`

**Interfaces:**
- Consumes: `polyt5.data.tg_metadata.{TgExample, descriptor_matrix}`, `polyt5.data.standardize.{Standardizer, fit_target_standardizer}`, `polyt5.data.weighting.{drop_red_reliability, reliability_weights}`, `polyt5.data.augment.augment_indices`, `polyt5.data.collate.{LABEL_IGNORE_ID, pad_sequences}`, `polyt5.data.prepare.format_property_value`
- Produces:
  - `PREDICTION_TASK: int = 0`, `GENERATION_TASK: int = 1`
  - `TaskItem` frozen dataclass: `input_ids: tuple[int, ...]`, `label_ids: tuple[int, ...]`, `tg_standardised: float`, `descriptors: tuple[float, ...]`, `weight: float`, `task_id: int`
  - `TaskDataset(torch.utils.data.Dataset)`: `__init__(self, items: Sequence[TaskItem])`, `__len__`, `__getitem__(self, index: int) -> TaskItem`, `stats` property `-> dict[str, Any]`
  - `TaskCollator`: `__init__(self, pad_id: int, *, max_source_length: int = 200, max_target_length: int = 200)`, `__call__(self, batch: Sequence[TaskItem]) -> dict[str, torch.Tensor]` with keys `input_ids`, `attention_mask`, `labels`, `tg_targets`, `descriptor_targets`, `weights`, `task_id`
  - `SplitTensors` frozen dataclass: `train: list[TaskItem]`, `train_generation: list[TaskItem]`, `val: list[TaskItem]`, `test_pselfies: list[str]`, `test_tg: list[float]`, `target_standardizer: Standardizer`, `descriptor_standardizer: Standardizer | None`, `dropped_descriptor_columns: tuple[str, ...]`, `n_train_polymers: int`, `n_train_writings: int`, `n_dropped_red: int`, `n_red_in_test: int`; method `to_manifest(self) -> dict[str, Any]`
  - `assemble_split(examples: Sequence[TgExample], descriptor_names: Sequence[str], *, train_indices: Sequence[int], val_indices: Sequence[int], test_indices: Sequence[int], tokenizer, use_regression_head: bool, use_descriptors: bool, n_writings: int, use_reliability_weighting: bool, std_floor: float, build_generation: bool, seed: int, max_source_length: int = 200, max_target_length: int = 200) -> SplitTensors`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_batches.py
"""Group A Task 7: one batch shape for seven arms, and the ordering rules."""

from __future__ import annotations

import pytest
import torch

from polyt5.data.multitask import (
    GENERATION_TASK,
    PREDICTION_TASK,
    SplitTensors,
    TaskCollator,
    TaskDataset,
    TaskItem,
    assemble_split,
)
from polyt5.data.tg_metadata import TgExample, TgRow


class FakeTokenizer:
    """Character-level stand-in; ``polyt5.data`` never constructs a tokenizer."""

    pad_id = 0
    eos_id = 1

    def encode(self, text, *, add_eos=True, max_length=None, truncation=True):
        ids = [2 + (ord(ch) % 60) for ch in str(text)]
        if add_eos:
            ids.append(self.eos_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


def make(index: int, *, std: float = 0.0, reliability: str = "black") -> TgExample:
    return TgExample(
        pselfies=f"[At][C]{'[C]' * index}[At]",
        row=TgRow(
            psmiles="[*]CCO[*]" if index % 2 else "[*]CC(C)[*]",
            tg=300.0 + 10.0 * index,
            std=std,
            num_of_points=1,
            reliability=reliability,
            polymer_class="test",
            descriptors=(float(index), 5.0, float(index) * 2.0),
        ),
    )


NAMES = ["varies", "constant", "doubles"]


def build(**kwargs) -> SplitTensors:
    examples = [make(i) for i in range(10)]
    defaults = dict(
        train_indices=[0, 1, 2, 3, 4, 5],
        val_indices=[6, 7],
        test_indices=[8, 9],
        tokenizer=FakeTokenizer(),
        use_regression_head=True,
        use_descriptors=False,
        n_writings=1,
        use_reliability_weighting=False,
        std_floor=5.6,
        build_generation=False,
        seed=0,
    )
    defaults.update(kwargs)
    return assemble_split(examples, NAMES, **defaults)


def test_test_split_is_never_augmented_weighted_or_filtered():
    out = build(n_writings=4, use_reliability_weighting=True)
    assert out.test_pselfies == [make(8).pselfies, make(9).pselfies]
    assert out.test_tg == [380.0, 390.0]


def test_red_rows_leave_train_but_the_test_split_is_untouched():
    """Filtering test would break comparability with the frozen 28.67 K."""
    examples = [make(i) for i in range(10)]
    examples[2] = make(2, reliability="red")
    examples[9] = make(9, reliability="red")
    out = assemble_split(
        examples, NAMES, train_indices=[0, 1, 2, 3], val_indices=[4],
        test_indices=[8, 9], tokenizer=FakeTokenizer(), use_regression_head=True,
        use_descriptors=False, n_writings=1, use_reliability_weighting=False,
        std_floor=5.6, build_generation=False, seed=0,
    )
    assert out.n_dropped_red == 1
    assert out.n_train_polymers == 3
    assert len(out.test_pselfies) == 2, "test keeps its red row"
    assert out.n_red_in_test == 1


def test_target_standardizer_is_fitted_on_train_only():
    out = build()
    train_tg = [300.0 + 10.0 * i for i in range(6)]
    assert out.target_standardizer.mean[0] == pytest.approx(
        sum(train_tg) / len(train_tg)
    )


def test_standardised_targets_have_zero_mean_on_train():
    out = build()
    values = [item.tg_standardised for item in out.train]
    assert sum(values) / len(values) == pytest.approx(0.0, abs=1e-9)


def test_descriptor_columns_that_are_constant_on_train_are_dropped_and_named():
    out = build(use_descriptors=True)
    assert out.descriptor_standardizer is not None
    assert out.dropped_descriptor_columns == ("constant",)
    assert out.descriptor_standardizer.columns == ("varies", "doubles")
    assert all(len(item.descriptors) == 2 for item in out.train)


def test_descriptors_are_empty_when_the_switch_is_off():
    out = build(use_descriptors=False)
    assert out.descriptor_standardizer is None
    assert out.dropped_descriptor_columns == ()
    assert all(item.descriptors == () for item in out.train)


def test_weights_are_all_one_when_weighting_is_off():
    out = build(use_reliability_weighting=False)
    assert {item.weight for item in out.train} == {1.0}


def test_weights_are_inherited_by_every_writing_of_a_polymer():
    examples = [make(i, std=float(i) * 20.0) for i in range(6)]
    out = assemble_split(
        examples, NAMES, train_indices=[0, 1, 2, 3], val_indices=[4], test_indices=[5],
        tokenizer=FakeTokenizer(), use_regression_head=True, use_descriptors=False,
        n_writings=3, use_reliability_weighting=True, std_floor=5.6,
        build_generation=False, seed=0,
    )
    by_target = {}
    for item in out.train:
        by_target.setdefault(item.tg_standardised, set()).add(item.weight)
    assert all(len(weights) == 1 for weights in by_target.values())


def test_augmentation_grows_train_but_not_val():
    plain = build(n_writings=1)
    grown = build(n_writings=4)
    assert grown.n_train_writings >= plain.n_train_writings
    assert grown.n_train_polymers == plain.n_train_polymers == 6
    assert len(grown.val) == len(plain.val) == 2


def test_regression_arms_carry_no_text_labels_and_text_arms_carry_no_scalar_target():
    regression = build(use_regression_head=True)
    assert all(item.label_ids == () for item in regression.train)
    text = build(use_regression_head=False)
    assert all(item.label_ids for item in text.train)


def test_generation_items_are_built_only_when_asked_and_only_from_train():
    off = build(build_generation=False)
    assert off.train_generation == []
    on = build(build_generation=True)
    assert len(on.train_generation) == 6
    assert {item.task_id for item in on.train_generation} == {GENERATION_TASK}
    assert {item.task_id for item in on.train} == {PREDICTION_TASK}


def test_collator_emits_only_tensors_with_the_expected_keys():
    out = build(use_descriptors=True, n_writings=2, use_reliability_weighting=True)
    collator = TaskCollator(pad_id=0, max_source_length=200, max_target_length=200)
    batch = collator(out.train[:4])
    assert set(batch) == {
        "input_ids", "attention_mask", "labels", "tg_targets",
        "descriptor_targets", "weights", "task_id",
    }
    assert all(isinstance(value, torch.Tensor) for value in batch.values())
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["tg_targets"].dtype == torch.float32
    assert batch["descriptor_targets"].shape == (4, 2)
    assert batch["weights"].shape == (4,)
    assert batch["task_id"].tolist() == [PREDICTION_TASK] * 4


def test_collator_pads_labels_with_the_ignore_id():
    from polyt5.data.collate import LABEL_IGNORE_ID

    items = [
        TaskItem((5, 6, 1), (7, 1), 0.0, (), 1.0, PREDICTION_TASK),
        TaskItem((5, 1), (7, 8, 9, 1), 0.0, (), 1.0, PREDICTION_TASK),
    ]
    batch = TaskCollator(pad_id=0)(items)
    assert batch["labels"][0].tolist() == [7, 1, LABEL_IGNORE_ID, LABEL_IGNORE_ID]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]


def test_collator_handles_items_with_no_text_labels():
    items = [TaskItem((5, 6, 1), (), 0.5, (), 1.0, PREDICTION_TASK)]
    batch = TaskCollator(pad_id=0)(items)
    assert batch["labels"].shape == (1, 0)


def test_dataset_is_picklable_for_dataloader_workers():
    import pickle

    dataset = TaskDataset(build().train)
    clone = pickle.loads(pickle.dumps(dataset))
    assert len(clone) == len(dataset)
    assert clone[0] == dataset[0]
    assert dataset.stats["n_items"] == len(dataset)


def test_manifest_records_what_was_dropped():
    out = build(use_descriptors=True)
    manifest = out.to_manifest()
    assert manifest["dropped_descriptor_columns"] == ["constant"]
    assert manifest["n_train_polymers"] == 6
    assert "n_red_in_test" in manifest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_batches.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.data.multitask'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/data/multitask.py
"""Group A batch construction: one item shape for all seven ablation arms.

Every arm -- text head or regression head, augmented or not, weighted or not,
single-task or multi-task -- produces batches with the same keys, so one
collator and one trainer serve all seven and the five switches stay genuinely
independent. Unused slots are empty tensors rather than absent keys, because
``Trainer._to_device`` moves every value and a missing key would branch the
training loop.

:func:`assemble_split` owns the ordering rules, and they are load-bearing:

1. ``reliability == red`` rows leave TRAIN and VAL. **Test is never filtered** --
   changing the evaluation set would make MAE incomparable to the frozen
   28.6733 K, which is the only number every arm is measured against.
   ``n_red_in_test`` reports the residual instead of hiding it.
2. Standardizers are fitted on TRAIN rows only, after the red drop and
   **before** augmentation -- augmenting first lets a polymer with many
   writings drag the mean.
3. Reliability weights are computed on train rows before augmentation; each
   writing inherits its source polymer's weight.
4. Only TRAIN is augmented. Val selects checkpoints; test is the measurement.
5. Generation items come from train polymers only, and only when asked.

This module imports torch and is therefore NOT re-exported from
``polyt5.data.__init__`` -- same rule as ``polyt5.data.datasets``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.utils.data

from polyt5.data.augment import augment_indices
from polyt5.data.collate import LABEL_IGNORE_ID, pad_sequences
from polyt5.data.prepare import format_property_value
from polyt5.data.standardize import Standardizer, fit_target_standardizer
from polyt5.data.tg_metadata import TgExample, descriptor_matrix
from polyt5.data.weighting import drop_red_reliability, reliability_weights

__all__ = [
    "GENERATION_TASK",
    "PREDICTION_TASK",
    "SplitTensors",
    "TaskCollator",
    "TaskDataset",
    "TaskItem",
    "assemble_split",
]

#: Task markers carried in every batch as a ``(batch,)`` long tensor.
PREDICTION_TASK = 0
GENERATION_TASK = 1


@dataclass(frozen=True)
class TaskItem:
    """One training example, whichever arm produced it.

    Attributes:
        input_ids: Encoder input ids. PSELFIES for prediction, the formatted
            conditioning number for generation.
        label_ids: Decoder target ids, or ``()`` when the regression head owns
            the Tg objective and no text is decoded.
        tg_standardised: The Tg target in standardised units; ``0.0`` and
            unused on the text path.
        descriptors: Standardised descriptor targets, or ``()`` when the
            descriptor switch is off.
        weight: Per-example loss weight; ``1.0`` when weighting is off.
        task_id: :data:`PREDICTION_TASK` or :data:`GENERATION_TASK`.
    """

    input_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    tg_standardised: float
    descriptors: tuple[float, ...]
    weight: float
    task_id: int


class TaskDataset(torch.utils.data.Dataset):
    """A plain, picklable list of :class:`TaskItem` for ``DataLoader``."""

    def __init__(self, items: Sequence[TaskItem]) -> None:
        """Wrap already-tokenized items.

        Args:
            items: Output of :func:`assemble_split`.
        """
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> TaskItem:
        """Return one item, already tokenized."""
        return self._items[index]

    @property
    def stats(self) -> dict[str, Any]:
        """Cheap descriptive statistics for run manifests."""
        return {
            "n_items": len(self._items),
            "n_prediction": sum(1 for i in self._items if i.task_id == PREDICTION_TASK),
            "n_generation": sum(1 for i in self._items if i.task_id == GENERATION_TASK),
        }


class TaskCollator:
    """Pad and tensorize a batch of :class:`TaskItem`.

    Every batch is single-task by construction (the loaders never mix), so
    ``task_id`` is uniform and the trainer can read element 0.
    """

    def __init__(
        self, pad_id: int, *, max_source_length: int = 200, max_target_length: int = 200
    ) -> None:
        """Initialize the collator.

        Args:
            pad_id: Encoder padding token id; labels always pad with ``-100``.
            max_source_length: Source truncation length.
            max_target_length: Target truncation length.
        """
        self.pad_id = int(pad_id)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)

    def __call__(self, batch: Sequence[TaskItem]) -> dict[str, torch.Tensor]:
        """Tensorize ``batch``.

        Returns:
            ``{"input_ids", "attention_mask", "labels", "tg_targets",
            "descriptor_targets", "weights", "task_id"}``. ``labels`` has width
            0 when no item decodes text; ``descriptor_targets`` has width 0
            when the descriptor switch is off.
        """
        inputs, mask = pad_sequences(
            [list(item.input_ids) for item in batch],
            pad_id=self.pad_id,
            max_length=self.max_source_length,
        )
        labels, _ = pad_sequences(
            [list(item.label_ids) for item in batch],
            pad_id=LABEL_IGNORE_ID,
            max_length=self.max_target_length,
        )
        n_descriptors = len(batch[0].descriptors) if batch else 0
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long).reshape(len(batch), -1),
            "tg_targets": torch.tensor(
                [item.tg_standardised for item in batch], dtype=torch.float32
            ),
            "descriptor_targets": torch.tensor(
                [list(item.descriptors) for item in batch], dtype=torch.float32
            ).reshape(len(batch), n_descriptors),
            "weights": torch.tensor([item.weight for item in batch], dtype=torch.float32),
            "task_id": torch.tensor([item.task_id for item in batch], dtype=torch.long),
        }


@dataclass(frozen=True)
class SplitTensors:
    """Everything one split needs, plus what was dropped getting there."""

    train: list[TaskItem]
    train_generation: list[TaskItem]
    val: list[TaskItem]
    test_pselfies: list[str]
    test_tg: list[float]
    target_standardizer: Standardizer
    descriptor_standardizer: Standardizer | None
    dropped_descriptor_columns: tuple[str, ...]
    n_train_polymers: int
    n_train_writings: int
    n_dropped_red: int
    n_red_in_test: int

    def to_manifest(self) -> dict[str, Any]:
        """Return the attrition and scaling record for the run manifest."""
        return {
            "n_train_polymers": self.n_train_polymers,
            "n_train_writings": self.n_train_writings,
            "n_train_generation": len(self.train_generation),
            "n_val": len(self.val),
            "n_test": len(self.test_pselfies),
            "n_dropped_red": self.n_dropped_red,
            "n_red_in_test": self.n_red_in_test,
            "dropped_descriptor_columns": list(self.dropped_descriptor_columns),
            "n_descriptors_kept": (
                0 if self.descriptor_standardizer is None
                else self.descriptor_standardizer.n_features
            ),
            "target_mean": self.target_standardizer.mean[0],
            "target_std": self.target_standardizer.std[0],
        }


def _encode(tokenizer, text: str, max_length: int) -> tuple[int, ...]:
    return tuple(
        tokenizer.encode(text, add_eos=True, max_length=max_length, truncation=True)
    )


def assemble_split(
    examples: Sequence[TgExample],
    descriptor_names: Sequence[str],
    *,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    tokenizer,
    use_regression_head: bool,
    use_descriptors: bool,
    n_writings: int,
    use_reliability_weighting: bool,
    std_floor: float,
    build_generation: bool,
    seed: int,
    max_source_length: int = 200,
    max_target_length: int = 200,
) -> SplitTensors:
    """Turn one split's indices into train/val items and a raw test set.

    Args:
        examples: The full prepared corpus, in frozen-splits order.
        descriptor_names: The 100 descriptor column names, positionally aligned
            with ``TgRow.descriptors``.
        train_indices: This split's train positions.
        val_indices: This split's validation positions (may be empty).
        test_indices: This split's held-out positions.
        tokenizer: Duck-typed tokenizer with ``encode``.
        use_regression_head: When True the Tg objective is the scalar head and
            items carry no text labels; when False they carry the numeric string.
        use_descriptors: Attach standardised descriptor targets.
        n_writings: Writings per train polymer (1 disables augmentation).
        use_reliability_weighting: Weight train examples by measurement spread.
        std_floor: Floor passed to :func:`reliability_weights`.
        build_generation: Also build Tg-conditioned generation items from train.
        seed: Augmentation seed.
        max_source_length: Encoder truncation length.
        max_target_length: Decoder truncation length.

    Returns:
        A :class:`SplitTensors`.
    """
    train_pool = [examples[i] for i in train_indices]
    val_pool = [examples[i] for i in val_indices]

    kept_train, dropped_train = drop_red_reliability(train_pool)
    kept_val, dropped_val = drop_red_reliability(val_pool)
    n_dropped_red = len(dropped_train) + len(dropped_val)
    n_red_in_test = sum(1 for i in test_indices if examples[i].row.reliability == "red")

    target_standardizer = fit_target_standardizer([e.row.tg for e in kept_train])

    descriptor_standardizer: Standardizer | None = None
    train_descriptors: np.ndarray | None = None
    if use_descriptors:
        matrix = descriptor_matrix(kept_train)
        descriptor_standardizer = Standardizer.fit(matrix, descriptor_names)
        train_descriptors = descriptor_standardizer.transform(matrix)

    weights = (
        reliability_weights(kept_train, floor=std_floor)
        if use_reliability_weighting
        else [1.0] * len(kept_train)
    )

    def scalar(example: TgExample) -> float:
        return float(
            target_standardizer.transform(np.asarray([[example.row.tg]], dtype=float))[0, 0]
        )

    def build_item(
        pselfies: str, example: TgExample, descriptors: tuple[float, ...], weight: float
    ) -> TaskItem:
        return TaskItem(
            input_ids=_encode(tokenizer, pselfies, max_source_length),
            label_ids=(
                ()
                if use_regression_head
                else _encode(
                    tokenizer, format_property_value(example.row.tg), max_target_length
                )
            ),
            tg_standardised=scalar(example),
            descriptors=descriptors,
            weight=weight,
            task_id=PREDICTION_TASK,
        )

    writings = augment_indices(
        kept_train,
        range(len(kept_train)),
        n_writings=n_writings,
        seed=seed,
        max_tokens=max_source_length,
        tokenizer=tokenizer,
    )
    train_items = [
        build_item(
            writing.pselfies,
            kept_train[writing.source_index],
            (
                tuple(float(v) for v in train_descriptors[writing.source_index])
                if train_descriptors is not None
                else ()
            ),
            weights[writing.source_index],
        )
        for writing in writings
    ]

    val_descriptors = (
        descriptor_standardizer.transform(descriptor_matrix(kept_val))
        if descriptor_standardizer is not None and kept_val
        else None
    )
    val_items = [
        build_item(
            example.pselfies,
            example,
            (
                tuple(float(v) for v in val_descriptors[position])
                if val_descriptors is not None
                else ()
            ),
            1.0,
        )
        for position, example in enumerate(kept_val)
    ]

    generation_items: list[TaskItem] = []
    if build_generation:
        generation_items = [
            TaskItem(
                input_ids=_encode(
                    tokenizer, format_property_value(example.row.tg), max_source_length
                ),
                label_ids=_encode(tokenizer, example.pselfies, max_target_length),
                tg_standardised=scalar(example),
                descriptors=(),
                weight=1.0,
                task_id=GENERATION_TASK,
            )
            for example in kept_train
        ]

    return SplitTensors(
        train=train_items,
        train_generation=generation_items,
        val=val_items,
        test_pselfies=[examples[i].pselfies for i in test_indices],
        test_tg=[examples[i].row.tg for i in test_indices],
        target_standardizer=target_standardizer,
        descriptor_standardizer=descriptor_standardizer,
        dropped_descriptor_columns=(
            () if descriptor_standardizer is None else descriptor_standardizer.dropped
        ),
        n_train_polymers=len(kept_train),
        n_train_writings=len(train_items),
        n_dropped_red=n_dropped_red,
        n_red_in_test=n_red_in_test,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_batches.py -o addopts="" -q`
Expected: PASS — 16 passed

- [ ] **Step 5: Confirm `polyt5.data` is still torch-free**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -c "import sys, polyt5.data; print('torch' in sys.modules)"`
Expected: `False` — `polyt5.data.multitask` must NOT be re-exported.

- [ ] **Step 6: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1147 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/data/multitask.py tests/test_group_a_batches.py
git commit -m "feat(group-a): split assembly with train-only fitting, weighting and augmentation"
```

---

### Task 8: The seven arms as a code table

Spec §5's table, as code rather than YAML prose, so "A6 = all five combined" is a property a test can assert instead of a convention someone maintains by hand. λ, N and the std floor stay configurable (spec §8 requires their sensitivity to be reportable), but the *switches* are fixed.

| id | regression_head | descriptors | augment | reliability_weighting | multitask |
|---|---|---|---|---|---|
| B0 | – | – | – | – | – |
| A1 | yes | – | – | – | – |
| A2 | – | yes | – | – | – |
| A3 | – | – | yes | – | – |
| A4 | – | – | – | yes | – |
| A5 | – | – | – | – | yes |
| A6 | yes | yes | yes | yes | yes |

Cycle consistency is **not** on any arm. Spec §4.5: it ships behind a flag, default OFF, because a model can satisfy it by being consistently wrong. A test pins that no arm turns it on.

**Files:**
- Create: `src/polyt5/training/group_a.py`
- Modify: `src/polyt5/training/__init__.py`
- Test: `tests/test_group_a_arms.py`

**Interfaces:**
- Consumes: nothing outside the standard library
- Produces:
  - `ARM_IDS: tuple[str, ...] = ("B0", "A1", "A2", "A3", "A4", "A5", "A6")`
  - `SWITCH_NAMES: tuple[str, ...] = ("regression_head", "descriptors", "augment", "reliability_weighting", "multitask")`
  - `GroupAConfig` frozen dataclass: `arm: str`, `regression_head: bool = False`, `descriptors: bool = False`, `augment: bool = False`, `reliability_weighting: bool = False`, `multitask: bool = False`, `cycle_consistency: bool = False`, `descriptor_lambda: float = 0.1`, `n_writings: int = 4`, `std_floor: float = 5.6`, `huber_delta: float = 1.0`, `cycle_weight: float = 0.1`
    - `switches(self) -> dict[str, bool]`
    - `effective_n_writings(self) -> int` — `n_writings` when `augment` else `1`
    - `to_dict(self) -> dict[str, Any]`
    - `__post_init__` validation
  - `arm_config(arm: str, **overrides: Any) -> GroupAConfig`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_arms.py
"""Group A Task 8: the seven-arm table, and the invariants it must satisfy."""

from __future__ import annotations

import pytest

from polyt5.training.group_a import ARM_IDS, SWITCH_NAMES, GroupAConfig, arm_config


def test_there_are_exactly_seven_arms_in_spec_order():
    assert ARM_IDS == ("B0", "A1", "A2", "A3", "A4", "A5", "A6")


def test_b0_turns_nothing_on():
    """B0 is the baseline: current text head, single task."""
    assert arm_config("B0").switches() == dict.fromkeys(SWITCH_NAMES, False)


def test_each_single_change_arm_turns_on_exactly_one_switch():
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        on = [name for name, value in arm_config(arm).switches().items() if value]
        assert len(on) == 1, f"{arm} flips {on}; individual ablations must be individual"


def test_the_five_single_change_arms_cover_all_five_switches():
    covered = {
        name
        for arm in ("A1", "A2", "A3", "A4", "A5")
        for name, value in arm_config(arm).switches().items()
        if value
    }
    assert covered == set(SWITCH_NAMES)


def test_a6_is_exactly_the_union_of_a1_through_a5():
    """'all five combined' is a property, not a convention someone maintains."""
    union = {name: False for name in SWITCH_NAMES}
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        for name, value in arm_config(arm).switches().items():
            union[name] = union[name] or value
    assert arm_config("A6").switches() == union
    assert all(union.values())


def test_no_arm_enables_cycle_consistency():
    """Spec 4.5: behind a flag, default OFF, never a primary objective."""
    assert all(not arm_config(arm).cycle_consistency for arm in ARM_IDS)


def test_overrides_reach_the_hyperparameters_but_not_the_switches():
    config = arm_config("A6", descriptor_lambda=0.5, n_writings=8, std_floor=10.0)
    assert config.descriptor_lambda == 0.5
    assert config.n_writings == 8
    assert config.std_floor == 10.0
    assert config.switches() == arm_config("A6").switches()


def test_overriding_a_switch_is_refused():
    with pytest.raises(ValueError, match="switch"):
        arm_config("A1", regression_head=False)


def test_unknown_arm_is_refused_with_the_valid_list():
    with pytest.raises(ValueError, match="B0"):
        arm_config("A9")


def test_effective_n_writings_is_one_unless_augmentation_is_on():
    assert arm_config("A1", n_writings=6).effective_n_writings() == 1
    assert arm_config("A3", n_writings=6).effective_n_writings() == 6


def test_degenerate_hyperparameters_are_refused():
    with pytest.raises(ValueError, match="descriptor_lambda"):
        arm_config("A2", descriptor_lambda=-1.0)
    with pytest.raises(ValueError, match="n_writings"):
        arm_config("A3", n_writings=0)
    with pytest.raises(ValueError, match="std_floor"):
        arm_config("A4", std_floor=0.0)


def test_cycle_consistency_requires_a_regression_head_to_score_with():
    with pytest.raises(ValueError, match="regression_head"):
        GroupAConfig(arm="X", cycle_consistency=True, regression_head=False)


def test_config_round_trips_to_a_dict_for_the_run_manifest():
    payload = arm_config("A6").to_dict()
    assert payload["arm"] == "A6"
    assert payload["regression_head"] is True
    assert payload["cycle_consistency"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_arms.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.training.group_a'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/training/group_a.py
"""The seven Group A ablation configurations.

Spec section 5 runs seven configurations over the SAME five splits the frozen
baseline used, so every number is directly comparable to 28.6733 +/- 0.7591 K::

    B0  baseline -- current text head, single task
    A1  + regression head
    A2  + descriptor auxiliaries
    A3  + invariance augmentation
    A4  + label weighting
    A5  + multi-task shared encoder
    A6  all five combined

The table lives here rather than in YAML because "A6 is all five combined" then
becomes a property a test asserts. Individual ablations run AS WELL AS the
combination: a combined gain with no per-change attribution cannot tell you
which idea to keep.

Cycle consistency (spec 4.5) is a field here but is OFF on every arm. A model
can satisfy it by being consistently wrong -- generate something odd,
confidently mispredict it, incur zero loss -- so it is a regulariser anchored by
real labels, never a primary objective, and it ships behind a flag that no
ablation arm sets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ARM_IDS", "SWITCH_NAMES", "GroupAConfig", "arm_config"]

#: The seven configurations, in spec section 5's order.
ARM_IDS: tuple[str, ...] = ("B0", "A1", "A2", "A3", "A4", "A5", "A6")

#: The five independently switchable changes.
SWITCH_NAMES: tuple[str, ...] = (
    "regression_head",
    "descriptors",
    "augment",
    "reliability_weighting",
    "multitask",
)

_ARM_SWITCHES: dict[str, tuple[str, ...]] = {
    "B0": (),
    "A1": ("regression_head",),
    "A2": ("descriptors",),
    "A3": ("augment",),
    "A4": ("reliability_weighting",),
    "A5": ("multitask",),
    "A6": SWITCH_NAMES,
}


@dataclass(frozen=True)
class GroupAConfig:
    """One ablation configuration: five switches plus their hyperparameters.

    Attributes:
        arm: The configuration id, e.g. ``"A3"``.
        regression_head: Predict Tg with the pooled scalar head instead of
            decoding it as text.
        descriptors: Add the auxiliary descriptor heads,
            ``L = L_Tg + descriptor_lambda * L_descriptors``.
        augment: Train on several PSELFIES writings per polymer.
        reliability_weighting: Weight examples by ``1 / max(std, std_floor)``
            and drop ``reliability == red``.
        multitask: Train prediction and generation together on the shared
            encoder, alternating batches.
        cycle_consistency: OFF on every arm; see the module docstring.
        descriptor_lambda: Weight of the descriptor term. Configurable because
            100 auxiliary targets against one Tg target risk swamping the
            objective we care about (spec section 8).
        n_writings: Writings per polymer when ``augment`` is on. Configurable
            because a large N means fewer distinct chemistries per epoch.
        std_floor: Weight floor in Kelvin; must be positive.
        huber_delta: Huber transition point, in standardised units.
        cycle_weight: Weight of the cycle term when it is enabled at all.
    """

    arm: str
    regression_head: bool = False
    descriptors: bool = False
    augment: bool = False
    reliability_weighting: bool = False
    multitask: bool = False
    cycle_consistency: bool = False
    descriptor_lambda: float = 0.1
    n_writings: int = 4
    std_floor: float = 5.6
    huber_delta: float = 1.0
    cycle_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.descriptor_lambda < 0.0:
            raise ValueError(
                f"descriptor_lambda must be >= 0, got {self.descriptor_lambda}"
            )
        if self.n_writings < 1:
            raise ValueError(f"n_writings must be >= 1, got {self.n_writings}")
        if self.std_floor <= 0.0:
            raise ValueError(f"std_floor must be > 0, got {self.std_floor}")
        if self.huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")
        if self.cycle_consistency and not self.regression_head:
            raise ValueError(
                "cycle_consistency needs regression_head=True: the cycle scores its own "
                "generations with the regression head, and without one there is nothing "
                "to close the loop with"
            )
        if self.cycle_consistency and self.cycle_weight <= 0.0:
            raise ValueError(
                f"cycle_weight must be > 0 when cycle_consistency is on, got "
                f"{self.cycle_weight}"
            )

    def switches(self) -> dict[str, bool]:
        """The five switches as a plain dict, in :data:`SWITCH_NAMES` order."""
        return {name: bool(getattr(self, name)) for name in SWITCH_NAMES}

    def effective_n_writings(self) -> int:
        """Writings per polymer actually used: 1 unless ``augment`` is on."""
        return self.n_writings if self.augment else 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the run manifest."""
        return asdict(self)


def arm_config(arm: str, **overrides: Any) -> GroupAConfig:
    """Build the configuration for one of the seven arms.

    Args:
        arm: One of :data:`ARM_IDS`.
        **overrides: Hyperparameter overrides (``descriptor_lambda``,
            ``n_writings``, ``std_floor``, ``huber_delta``, ``cycle_weight``,
            ``cycle_consistency``). The five switches cannot be overridden --
            they define the arm.

    Returns:
        The configuration.

    Raises:
        ValueError: If ``arm`` is unknown, or an override names a switch.
    """
    if arm not in _ARM_SWITCHES:
        raise ValueError(f"unknown arm {arm!r}; valid arms are {list(ARM_IDS)}")
    clashing = sorted(set(overrides) & set(SWITCH_NAMES))
    if clashing:
        raise ValueError(
            f"cannot override the switch(es) {clashing} on arm {arm!r}: the switches define "
            "the arm, and changing one would make the ablation row unattributable"
        )
    enabled = dict.fromkeys(_ARM_SWITCHES[arm], True)
    return GroupAConfig(arm=arm, **enabled, **overrides)
```

Add to `src/polyt5/training/__init__.py`: `from polyt5.training.group_a import ARM_IDS, SWITCH_NAMES, GroupAConfig, arm_config` plus the four `__all__` entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_arms.py -o addopts="" -q`
Expected: PASS — 13 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1160 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/training/group_a.py src/polyt5/training/__init__.py tests/test_group_a_arms.py
git commit -m "feat(group-a): the seven-arm ablation table with A6 proved to be A1..A5"
```

---

### Task 9: `GroupATrainer` and the interleaved loader

Spec §4.5: "Train prediction and generation together on the shared encoder, **alternating batches**." Rather than rewrite the training loop, alternation is a *loader*: `InterleavedLoader` yields one prediction batch, then one generation batch, and the existing `Trainer.train_epoch` runs unchanged. `GroupATrainer` overrides only three small hooks.

The base `Trainer` needs one override point it does not have: `train_epoch` and `evaluate` both inline `int((batch["labels"] != -100).sum().item())`, which is meaningless for a regression batch with a width-0 `labels` tensor. This task extracts that expression into `Trainer._batch_weight` — a pure refactor, identical behaviour, and the existing `tests/test_training.py` proves it.

**Files:**
- Create: `src/polyt5/training/multitask_trainer.py`
- Modify: `src/polyt5/training/trainer.py` (extract `_batch_weight`), `src/polyt5/training/__init__.py`
- Test: `tests/test_group_a_trainer.py`

**Interfaces:**
- Consumes: `polyt5.training.trainer.{Trainer, TrainerConfig, Batch}`, `polyt5.training.group_a.GroupAConfig`, `polyt5.model.multitask.PolyT5MultiTask`, `polyt5.data.multitask.{PREDICTION_TASK, GENERATION_TASK}`
- Produces:
  - `Trainer._batch_weight(self, batch: Batch) -> int` (new protected method on the existing class)
  - `InterleavedLoader`: `__init__(self, prediction: Iterable[Batch], generation: Iterable[Batch] | None = None)`, `__iter__(self) -> Iterator[Batch]`, `__len__(self) -> int`
  - `GroupATrainer(Trainer)`: `__init__(self, model: PolyT5MultiTask, train_loader: Iterable[Batch], config: TrainerConfig, *, group_a: GroupAConfig, cycle_loss: Callable[[Tensor], Tensor] | None = None, **kwargs: Any)`; overrides `_to_device`, `_forward_loss`, `_batch_weight`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_trainer.py
"""Group A Task 9: alternating batches, and one loss router for seven arms."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polyt5.data.multitask import GENERATION_TASK, PREDICTION_TASK
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.training import Trainer, TrainerConfig
from polyt5.training.group_a import arm_config
from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"


def build_model(**kwargs) -> PolyT5MultiTask:
    seed_everything(0)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    return PolyT5MultiTask(
        PolyT5ForConditionalGeneration(config), MultiTaskConfig(head_dropout=0.0, **kwargs)
    )


def make_batch(task_id: int, *, n: int = 2, n_descriptors: int = 0, text: bool = True):
    generator = torch.Generator().manual_seed(17 + task_id)
    return {
        "input_ids": torch.randint(2, 458, (n, 5), generator=generator),
        "attention_mask": torch.ones(n, 5, dtype=torch.long),
        "labels": (
            torch.randint(2, 458, (n, 3), generator=generator)
            if text
            else torch.zeros(n, 0, dtype=torch.long)
        ),
        "tg_targets": torch.zeros(n, dtype=torch.float32),
        "descriptor_targets": torch.zeros(n, n_descriptors, dtype=torch.float32),
        "weights": torch.ones(n, dtype=torch.float32),
        "task_id": torch.full((n,), task_id, dtype=torch.long),
    }


def trainer_config(**kwargs) -> TrainerConfig:
    defaults = dict(
        max_epochs=1, physical_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=3e-4, weight_decay=0.01, scheduler="constant", amp=False,
        device="cpu", log_every=1000,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


# ------------------------------------------------------- the base-class refactor
def test_base_trainer_batch_weight_still_counts_label_tokens():
    """The extraction must not change what the base trainer measures."""
    model = PolyT5ForConditionalGeneration(PolyT5Config.from_yaml(TINY_YAML))
    trainer = Trainer(model, [], trainer_config())
    batch = {"labels": torch.tensor([[1, 2, -100], [3, -100, -100]])}
    assert trainer._batch_weight(batch) == 3


# --------------------------------------------------------------- interleaving
def test_interleaved_loader_alternates_prediction_and_generation():
    prediction = [make_batch(PREDICTION_TASK) for _ in range(3)]
    generation = [make_batch(GENERATION_TASK) for _ in range(3)]
    tasks = [
        int(batch["task_id"][0]) for batch in InterleavedLoader(prediction, generation)
    ]
    assert tasks == [PREDICTION_TASK, GENERATION_TASK] * 3
    assert len(InterleavedLoader(prediction, generation)) == 6


def test_interleaved_loader_without_generation_is_a_passthrough():
    prediction = [make_batch(PREDICTION_TASK) for _ in range(4)]
    loader = InterleavedLoader(prediction, None)
    assert len(loader) == 4
    assert all(int(b["task_id"][0]) == PREDICTION_TASK for b in loader)


def test_interleaved_loader_cycles_the_shorter_side():
    prediction = [make_batch(PREDICTION_TASK) for _ in range(4)]
    generation = [make_batch(GENERATION_TASK)]
    assert len(list(InterleavedLoader(prediction, generation))) == 8


def test_interleaved_loader_is_reiterable():
    loader = InterleavedLoader([make_batch(PREDICTION_TASK)], [make_batch(GENERATION_TASK)])
    assert [int(b["task_id"][0]) for b in loader] == [int(b["task_id"][0]) for b in loader]


# ------------------------------------------------------------- the loss router
def test_regression_arm_uses_the_scalar_head():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    loss = trainer._forward_loss(make_batch(PREDICTION_TASK, text=False))
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_text_arm_matches_the_backbone_loss_exactly():
    """B0 reproduces the baseline objective, it does not approximate it."""
    model = build_model()
    model.eval()
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("B0"))
    batch = make_batch(PREDICTION_TASK)
    reference = model.backbone(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss
    assert float(trainer._forward_loss(batch)) == pytest.approx(float(reference), abs=1e-6)


def test_generation_batches_route_to_the_decoder():
    model = build_model(use_regression_head=True)
    model.eval()
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A6"))
    batch = make_batch(GENERATION_TASK)
    reference = model.backbone(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss
    assert float(trainer._forward_loss(batch)) == pytest.approx(float(reference), abs=1e-6)


def test_descriptor_arm_adds_a_term_to_the_text_loss():
    plain = build_model()
    plain.eval()
    with_descriptors = build_model(n_descriptors=3, descriptor_lambda=1.0)
    with_descriptors.eval()
    batch = make_batch(PREDICTION_TASK, n_descriptors=3)
    base = GroupATrainer(plain, [], trainer_config(), group_a=arm_config("B0"))
    aux = GroupATrainer(
        with_descriptors, [], trainer_config(), group_a=arm_config("A2")
    )
    assert float(aux._forward_loss(batch)) != pytest.approx(float(base._forward_loss(batch)))


def test_batch_weight_counts_examples_not_tokens():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    assert trainer._batch_weight(make_batch(PREDICTION_TASK, n=5, text=False)) == 5


def test_to_device_tolerates_a_non_tensor_value():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A1"))
    moved = trainer._to_device({**make_batch(PREDICTION_TASK, text=False), "arm": "A1"})
    assert moved["arm"] == "A1"
    assert moved["input_ids"].device.type == "cpu"


def test_a_mixed_task_batch_is_refused():
    model = build_model(use_regression_head=True)
    trainer = GroupATrainer(model, [], trainer_config(), group_a=arm_config("A6"))
    batch = make_batch(PREDICTION_TASK, n=2)
    batch["task_id"] = torch.tensor([PREDICTION_TASK, GENERATION_TASK])
    with pytest.raises(ValueError, match="single-task"):
        trainer._forward_loss(batch)


def test_one_epoch_runs_end_to_end_and_moves_the_weights():
    model = build_model(use_regression_head=True, n_descriptors=3)
    before = model.tg_head.projection.weight.detach().clone()
    loader = InterleavedLoader(
        [make_batch(PREDICTION_TASK, n_descriptors=3, text=False) for _ in range(4)],
        [make_batch(GENERATION_TASK) for _ in range(4)],
    )
    trainer = GroupATrainer(model, loader, trainer_config(), group_a=arm_config("A6"))
    metrics = trainer.train()
    assert metrics["global_step"] == 8
    assert not torch.allclose(before, model.tg_head.projection.weight)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_trainer.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.training.multitask_trainer'`

- [ ] **Step 3: Extract `_batch_weight` in the base trainer**

In `src/polyt5/training/trainer.py`, insert this method immediately after `_forward_loss` (before `_optimizer_step`):

```python
    def _batch_weight(self, batch: Batch) -> int:
        """Weight of this batch in the epoch's mean loss.

        The base trainer reports a TOKEN-level mean, so the weight is the
        number of label positions that are not ``-100``. Subclasses whose
        batches are not token-scored (e.g. a regression head) override this;
        the epoch mean is then over whatever unit they return, and their
        docstring says which.
        """
        return int((batch["labels"] != -100).sum().item())
```

Then replace both occurrences of the inlined expression (one in `train_epoch`, one in `evaluate`) — they are byte-identical, so one `replace_all` edit does both:

```python
# before (2 occurrences)
            num_tokens = int((batch["labels"] != -100).sum().item())
# after
            num_tokens = self._batch_weight(batch)
```

- [ ] **Step 4: Run the existing training tests to prove the refactor changed nothing**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_training.py tests/test_integration.py -o addopts="" -q`
Expected: PASS, same counts as before the edit.

- [ ] **Step 5: Write the interleaved loader and the Group A trainer**

```python
# src/polyt5/training/multitask_trainer.py
"""Alternating-batch training for the Group A arms.

Spec section 4.5 asks for prediction and generation trained together on the
shared encoder, "alternating batches". That is expressed here as a LOADER --
:class:`InterleavedLoader` yields one prediction batch, then one generation
batch -- so :meth:`polyt5.training.Trainer.train_epoch` runs completely
unchanged and every gradient-accumulation, AMP, clipping and checkpoint
behaviour is shared with the baseline trainer rather than re-implemented.

:class:`GroupATrainer` overrides exactly three hooks:

* ``_to_device`` -- tolerates non-tensor values in a batch dict.
* ``_forward_loss`` -- routes a batch to one of the model's three forward paths
  by its ``task_id``.
* ``_batch_weight`` -- weights the epoch mean by EXAMPLES, not label tokens,
  because a regression batch has no label tokens at all. The reported
  ``train_loss`` for a Group A run is therefore an example-weighted mean and is
  not comparable to the baseline trainer's token-weighted one. MAE on the held
  out split is the comparable number, and that is what the ablation reports.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import cycle
from typing import Any

import torch
from torch import Tensor

from polyt5.data.multitask import GENERATION_TASK, PREDICTION_TASK
from polyt5.model.multitask import PolyT5MultiTask
from polyt5.training.group_a import GroupAConfig
from polyt5.training.trainer import Batch, Trainer, TrainerConfig

__all__ = ["GroupATrainer", "InterleavedLoader"]


class InterleavedLoader:
    """Alternate batches from two loaders, one from each, until the longer ends.

    The shorter side is cycled, so a small generation set is reused rather than
    silently truncating the prediction set. With no generation loader this is a
    transparent pass-through, which is what the six single-task arms use.
    """

    def __init__(
        self, prediction: Iterable[Batch], generation: Iterable[Batch] | None = None
    ) -> None:
        """Initialize the loader.

        Args:
            prediction: Loader of prediction batches.
            generation: Optional loader of generation batches.
        """
        self.prediction = prediction
        self.generation = generation

    def _lengths(self) -> tuple[int, int]:
        n_prediction = len(self.prediction)  # type: ignore[arg-type]
        n_generation = (
            0 if self.generation is None else len(self.generation)  # type: ignore[arg-type]
        )
        return n_prediction, n_generation

    def __len__(self) -> int:
        """Total batches yielded per epoch."""
        n_prediction, n_generation = self._lengths()
        if n_generation == 0:
            return n_prediction
        return 2 * max(n_prediction, n_generation)

    def __iter__(self) -> Iterator[Batch]:
        """Yield prediction, generation, prediction, generation, ..."""
        if self.generation is None:
            yield from self.prediction
            return
        n_prediction, n_generation = self._lengths()
        rounds = max(n_prediction, n_generation)
        predictions = cycle(self.prediction) if n_prediction < rounds else iter(self.prediction)
        generations = cycle(self.generation) if n_generation < rounds else iter(self.generation)
        for _ in range(rounds):
            yield next(predictions)
            yield next(generations)


class GroupATrainer(Trainer):
    """The baseline trainer, with a task-routing loss and example weighting."""

    def __init__(
        self,
        model: PolyT5MultiTask,
        train_loader: Iterable[Batch],
        config: TrainerConfig,
        *,
        group_a: GroupAConfig,
        cycle_loss: Callable[[Tensor], Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the trainer.

        Args:
            model: The wrapped multi-task model.
            train_loader: Usually an :class:`InterleavedLoader`.
            config: Standard trainer configuration.
            group_a: Which of the five switches this arm turns on.
            cycle_loss: Optional callable taking the batch's standardised Tg
                targets and returning a scalar cycle-consistency loss. ``None``
                on every ablation arm; see :mod:`polyt5.training.cycle`.
            **kwargs: Forwarded to :class:`~polyt5.training.Trainer`
                (``val_loader``, ``run_dir``, ``tokenizer_path``,
                ``tokenizer_sha256``, ``run_config``, ``logger``).

        Raises:
            ValueError: If the arm enables cycle consistency but no
                ``cycle_loss`` was supplied, or supplies one for an arm that
                does not enable it.
        """
        if group_a.cycle_consistency and cycle_loss is None:
            raise ValueError(
                f"arm {group_a.arm!r} enables cycle_consistency but no cycle_loss was given"
            )
        if cycle_loss is not None and not group_a.cycle_consistency:
            raise ValueError(
                f"a cycle_loss was given for arm {group_a.arm!r}, which has "
                "cycle_consistency=False; an objective that is off must not be trained"
            )
        super().__init__(model, train_loader, config, **kwargs)
        self.group_a = group_a
        self.cycle_loss = cycle_loss

    def _to_device(self, batch: Batch) -> Batch:
        """Move tensor values to the device, passing anything else through."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _batch_weight(self, batch: Batch) -> int:
        """Weight the epoch mean by EXAMPLES; a regression batch has no tokens."""
        return int(batch["input_ids"].shape[0])

    def _task_of(self, batch: Batch) -> int:
        task_ids = batch["task_id"]
        unique = torch.unique(task_ids)
        if unique.numel() != 1:
            raise ValueError(
                f"every batch must be single-task, got task ids {unique.tolist()}; the "
                "loaders are built per task so a mixed batch is a plumbing bug"
            )
        return int(unique.item())

    def _forward_loss(self, batch: Batch) -> Tensor:
        """Route the batch to one of the model's three forward paths.

        Args:
            batch: A batch from :class:`polyt5.data.multitask.TaskCollator`.

        Returns:
            The scalar loss for this batch.
        """
        task = self._task_of(batch)
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        weights = batch["weights"] if self.group_a.reliability_weighting else None
        descriptors = batch["descriptor_targets"] if self.group_a.descriptors else None

        with torch.amp.autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            if task == GENERATION_TASK:
                return self.model.forward_generation(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                ).loss
            if task != PREDICTION_TASK:
                raise ValueError(f"unknown task id {task}")

            if self.group_a.regression_head:
                output = self.model.forward_regression(
                    batch["input_ids"],
                    batch["attention_mask"],
                    tg_targets=batch["tg_targets"],
                    descriptor_targets=descriptors,
                    weights=weights,
                )
            else:
                output = self.model.forward_text(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                    descriptor_targets=descriptors,
                    weights=weights,
                )
            assert output.loss is not None  # targets were supplied
            loss = output.loss
            if self.cycle_loss is not None:
                loss = loss + self.group_a.cycle_weight * self.cycle_loss(batch["tg_targets"])
            return loss
```

Add to `src/polyt5/training/__init__.py`: `from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader` plus the two `__all__` entries.

- [ ] **Step 6: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_trainer.py -o addopts="" -q`
Expected: PASS — 14 passed

- [ ] **Step 7: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1174 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/training/multitask_trainer.py src/polyt5/training/trainer.py src/polyt5/training/__init__.py tests/test_group_a_trainer.py
git commit -m "feat(group-a): alternating-batch trainer routing three forward paths"
```

---

### Task 10: Cycle consistency behind a flag, default OFF

Spec §4.5, in full: "Generating a polymer for a target Tg and then scoring it with the model's own regression head is a signal the model can satisfy by being *consistently wrong* — generate something odd, confidently mispredict it as 500 K, incur zero loss. Anchored by 7,367 real labels it is a legitimate semi-supervised regulariser... As a primary objective it would be circular. It ships behind a flag and is measured as an ablation, never assumed."

So the module enforces three things in code rather than in prose: `enabled` defaults to `False`; `build_cycle_loss` refuses a model with no regression head (nothing to close the loop with); and Task 8 already pins that no ablation arm turns it on. Task 9's `GroupATrainer` already refuses a `cycle_loss` for an arm that has the flag off.

**Files:**
- Create: `src/polyt5/training/cycle.py`
- Modify: `src/polyt5/training/__init__.py`
- Test: `tests/test_group_a_cycle.py`

**Interfaces:**
- Consumes: `polyt5.generation.{GenerationConfig, generate}`, `polyt5.model.multitask.PolyT5MultiTask`, `polyt5.model.heads.weighted_huber_loss`, `polyt5.data.prepare.format_property_value`
- Produces:
  - `CycleConfig` frozen dataclass: `enabled: bool = False`, `max_length: int = 200`, `temperature: float = 1.0`, `top_p: float = 0.95`, `seed: int = 0`, `huber_delta: float = 1.0`
  - `build_cycle_loss(model: PolyT5MultiTask, tokenizer, *, config: CycleConfig, device: str) -> Callable[[Tensor], Tensor] | None` — returns `None` when disabled

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_cycle.py
"""Group A Task 10: cycle consistency is a flag, and the flag is off."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training.cycle import CycleConfig, build_cycle_loss
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"
VOCAB = REPO / "artifacts" / "tokenizer" / "polyt5_vocab.json"

pytestmark = pytest.mark.skipif(not VOCAB.is_file(), reason="tokenizer artifact missing")


def build_model(*, regression_head: bool = True) -> tuple[PolyT5MultiTask, PolyT5Tokenizer]:
    seed_everything(0)
    tokenizer = PolyT5Tokenizer.from_file(VOCAB)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_id
    config.eos_token_id = tokenizer.eos_id
    config.decoder_start_token_id = tokenizer.decoder_start_token_id
    model = PolyT5MultiTask(
        PolyT5ForConditionalGeneration(config),
        MultiTaskConfig(use_regression_head=regression_head, head_dropout=0.0),
    )
    model.set_target_scaling(mean=417.0, std=113.0)
    return model, tokenizer


def test_the_default_is_off():
    assert CycleConfig().enabled is False


def test_disabled_config_builds_no_loss_at_all():
    model, tokenizer = build_model()
    assert build_cycle_loss(model, tokenizer, config=CycleConfig(), device="cpu") is None


def test_enabled_config_builds_a_callable_returning_a_scalar():
    model, tokenizer = build_model()
    loss_fn = build_cycle_loss(
        model, tokenizer, config=CycleConfig(enabled=True, max_length=16), device="cpu"
    )
    assert loss_fn is not None
    loss = loss_fn(torch.zeros(2))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_the_cycle_loss_carries_a_gradient_into_the_encoder():
    model, tokenizer = build_model()
    loss_fn = build_cycle_loss(
        model, tokenizer, config=CycleConfig(enabled=True, max_length=16), device="cpu"
    )
    loss_fn(torch.zeros(2)).backward()
    grads = [p.grad for p in model.backbone.encoder.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0.0 for g in grads)


def test_a_model_without_a_regression_head_cannot_close_the_loop():
    model, tokenizer = build_model(regression_head=False)
    with pytest.raises(ValueError, match="regression head"):
        build_cycle_loss(model, tokenizer, config=CycleConfig(enabled=True), device="cpu")


def test_the_same_seed_reproduces_the_same_cycle_loss():
    model, tokenizer = build_model()
    config = CycleConfig(enabled=True, max_length=16, seed=5)
    first = build_cycle_loss(model, tokenizer, config=config, device="cpu")(torch.zeros(2))
    second = build_cycle_loss(model, tokenizer, config=config, device="cpu")(torch.zeros(2))
    assert float(first) == pytest.approx(float(second))


def test_an_empty_target_batch_returns_zero_rather_than_nan():
    model, tokenizer = build_model()
    loss_fn = build_cycle_loss(
        model, tokenizer, config=CycleConfig(enabled=True, max_length=16), device="cpu"
    )
    loss = loss_fn(torch.zeros(0))
    assert float(loss) == pytest.approx(0.0)


def test_degenerate_sampling_settings_are_refused():
    model, tokenizer = build_model()
    with pytest.raises(ValueError, match="temperature"):
        build_cycle_loss(
            model, tokenizer, config=CycleConfig(enabled=True, temperature=0.0), device="cpu"
        )
    with pytest.raises(ValueError, match="top_p"):
        build_cycle_loss(
            model, tokenizer, config=CycleConfig(enabled=True, top_p=0.0), device="cpu"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_cycle.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.training.cycle'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/training/cycle.py
"""Cycle consistency: generate for a target Tg, then score what you generated.

**This ships behind a flag and the flag defaults to OFF.**

The signal is circular on its own. A model can satisfy it by being
*consistently wrong*: generate something odd, confidently mispredict it as
500 K, incur zero loss. Anchored by 7,367 real labels it is a legitimate
semi-supervised regulariser -- back-translation is anchored by real parallel
text in exactly the same way -- but as a primary objective it measures nothing.
So:

* :attr:`CycleConfig.enabled` is ``False`` by default, and
  :func:`build_cycle_loss` returns ``None`` when it is.
* No arm in :mod:`polyt5.training.group_a` turns it on, and a test pins that.
* :class:`polyt5.training.multitask_trainer.GroupATrainer` refuses a cycle loss
  for an arm whose flag is off, and refuses an arm whose flag is on without one.

Sampling happens under ``no_grad`` -- the gradient flows through the RESCORING
pass, not through the sampling, which is not differentiable anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from polyt5.data.prepare import format_property_value
from polyt5.generation import GenerationConfig, generate
from polyt5.model.heads import weighted_huber_loss
from polyt5.model.multitask import PolyT5MultiTask

__all__ = ["CycleConfig", "build_cycle_loss"]


@dataclass(frozen=True)
class CycleConfig:
    """Sampling and weighting knobs for the cycle term.

    Attributes:
        enabled: Off by default. See the module docstring for why.
        max_length: Decode cap for the generated PSELFIES.
        temperature: Sampling temperature; must be positive.
        top_p: Nucleus cutoff; must be in ``(0, 1]``.
        seed: Sampling seed, so a cycle loss is reproducible.
        huber_delta: Huber transition point, in standardised units.
    """

    enabled: bool = False
    max_length: int = 200
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0
    huber_delta: float = 1.0


def build_cycle_loss(
    model: PolyT5MultiTask,
    tokenizer,
    *,
    config: CycleConfig,
    device: str,
) -> Callable[[Tensor], Tensor] | None:
    """Build the cycle-consistency loss callable, or ``None`` when disabled.

    Args:
        model: The multi-task model; needs a regression head to score with.
        tokenizer: The tokenizer the model was trained with.
        config: Sampling and weighting knobs.
        device: Torch device string.

    Returns:
        A callable taking ``(batch,)`` STANDARDISED Tg targets and returning a
        scalar loss, or ``None`` when ``config.enabled`` is ``False``.

    Raises:
        ValueError: If enabled on a model with no regression head, or with a
            non-positive temperature or an out-of-range ``top_p``.
    """
    if not config.enabled:
        return None
    if model.tg_head is None:
        raise ValueError(
            "cycle consistency needs a regression head to score its own generations with; "
            "this model has none, so there is nothing to close the loop"
        )
    if config.temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {config.temperature}")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {config.top_p}")

    def cycle_loss(standardised_targets: Tensor) -> Tensor:
        if standardised_targets.numel() == 0:
            return torch.zeros((), device=device)

        kelvin = standardised_targets.detach() * model.tg_std + model.tg_mean
        prompts = [format_property_value(float(value)) for value in kelvin]
        encoded = tokenizer.batch_encode(
            prompts, add_eos=True, max_length=config.max_length,
            padding=True, truncation=True,
        )
        prompt_ids = torch.tensor(encoded["input_ids"], device=device)
        prompt_mask = torch.tensor(encoded["attention_mask"], device=device)

        with torch.no_grad():
            sampled = generate(
                model.backbone, prompt_ids, prompt_mask,
                config=GenerationConfig(
                    max_length=config.max_length,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                    decoder_start_token_id=tokenizer.decoder_start_token_id,
                    seed=config.seed,
                ),
            )
        texts = tokenizer.batch_decode(sampled.sequences.tolist(), skip_special_tokens=True)
        rescored = tokenizer.batch_encode(
            texts, add_eos=True, max_length=config.max_length,
            padding=True, truncation=True,
        )
        # The RESCORING pass carries the gradient; sampling above did not.
        output = model.forward_regression(
            torch.tensor(rescored["input_ids"], device=device),
            torch.tensor(rescored["attention_mask"], device=device),
        )
        assert output.tg_standardised is not None
        return weighted_huber_loss(
            output.tg_standardised,
            standardised_targets.to(output.tg_standardised.dtype),
            delta=config.huber_delta,
        )

    return cycle_loss
```

Add to `src/polyt5/training/__init__.py`: `from polyt5.training.cycle import CycleConfig, build_cycle_loss` plus the two `__all__` entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_cycle.py -o addopts="" -q`
Expected: PASS — 8 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1182 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/training/cycle.py src/polyt5/training/__init__.py tests/test_group_a_cycle.py
git commit -m "feat(group-a): cycle consistency behind an off-by-default flag"
```

---

### Task 11: `RegressionPropertyPredictor`

The regression head replaces text-decoded numbers for prediction, so the ablation needs a predictor that runs the head and inverts the train-split standardisation. `non_numeric_rate` is structurally 0.0 for this predictor and is still reported — spec §6 asks for it "where applicable", and reporting an honest zero is better than omitting the column and inviting the reader to assume the old one.

A Group A checkpoint must **fail loudly** in the existing `PolyT5PropertyPredictor`: its `state_dict` keys are namespaced under `backbone.`, and spec §7 says Group A "produces new fine-tuned models alongside the existing ones". Silently half-loading one into a reward path would be the worst possible outcome, so a test pins the failure.

**Files:**
- Create: `src/polyt5/inference/regression_predictor.py`
- Modify: `src/polyt5/inference/__init__.py`
- Test: `tests/test_group_a_predictor.py`

**Interfaces:**
- Consumes: `polyt5.inference.predictor.{PredictionResult, NON_NUMERIC_VALUE, DEFAULT_MAX_SOURCE_LENGTH}`, `polyt5.model.{PolyT5Config, PolyT5ForConditionalGeneration}`, `polyt5.model.multitask.{MultiTaskConfig, PolyT5MultiTask}`, `polyt5.tokenization.PolyT5Tokenizer`, `polyt5.training.load_checkpoint`, `polyt5.data.prepare.format_property_value`, `polyt5.utils.select_device`
- Produces:
  - `GROUP_A_CONFIG_KEY: str = "group_a"` — the run-config key under which head metadata is stored
  - `RegressionPropertyPredictor`:
    - `__init__(self, model: PolyT5MultiTask, tokenizer: PolyT5Tokenizer, *, device: str = "auto", batch_size: int = 64, max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH, property_name: str | None = None)`
    - `from_checkpoint(cls, checkpoint_path: str | Path, tokenizer_path: str | Path | None = None, *, device: str = "auto", allow_unverified_tokenizer: bool = False, **kwargs: Any) -> RegressionPropertyPredictor` (classmethod)
    - `predict(self, pselfies: Sequence[str]) -> list[PredictionResult]`
    - `predict_values(self, pselfies: Sequence[str]) -> list[float | None]`
    - `__call__(self, candidates: Sequence[str]) -> list[float]`
    - `__repr__(self) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_predictor.py
"""Group A Task 11: scoring with the regression head, and refusing to be mistaken
for a baseline checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from polyt5.inference import PolyT5PropertyPredictor
from polyt5.inference.regression_predictor import (
    GROUP_A_CONFIG_KEY,
    RegressionPropertyPredictor,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import save_checkpoint
from polyt5.utils import seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"
VOCAB = REPO / "artifacts" / "tokenizer" / "polyt5_vocab.json"

pytestmark = pytest.mark.skipif(not VOCAB.is_file(), reason="tokenizer artifact missing")


def build(n_descriptors: int = 0):
    seed_everything(0)
    tokenizer = PolyT5Tokenizer.from_file(VOCAB)
    config = PolyT5Config.from_yaml(TINY_YAML)
    config.dropout_rate = 0.0
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_id
    config.eos_token_id = tokenizer.eos_id
    config.decoder_start_token_id = tokenizer.decoder_start_token_id
    head_config = MultiTaskConfig(
        use_regression_head=True, n_descriptors=n_descriptors, head_dropout=0.0
    )
    model = PolyT5MultiTask(PolyT5ForConditionalGeneration(config), head_config)
    model.set_target_scaling(mean=417.0, std=113.0)
    return model, tokenizer, head_config


def write_checkpoint(path: Path, model, tokenizer, head_config):
    return save_checkpoint(
        path,
        model=model,
        epoch=0,
        global_step=1,
        config={GROUP_A_CONFIG_KEY: {"heads": head_config.to_dict(), "arm": "A1"}},
        model_config=model.config.to_dict(),
        tokenizer_path=str(VOCAB),
        tokenizer_sha256=tokenizer.sha256,
    )


def test_predictions_come_back_in_kelvin_around_the_train_mean():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    results = predictor.predict(["[At][C][C][O][At]", "[At][C][C][At]"])
    assert len(results) == 2
    assert all(result.is_numeric for result in results)
    assert all(math.isfinite(result.value) for result in results)


def test_the_decoded_field_is_the_formatted_number():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    result = predictor.predict(["[At][C][C][O][At]"])[0]
    assert result.decoded == f"{result.value:.1f}"
    assert result.source == "[At][C][C][O][At]"


def test_batching_does_not_change_a_prediction():
    model, tokenizer, _ = build()
    inputs = ["[At][C][C][O][At]", "[At][C][C][At]", "[At][C][O][At]"]
    one = RegressionPropertyPredictor(model, tokenizer, device="cpu", batch_size=1)
    many = RegressionPropertyPredictor(model, tokenizer, device="cpu", batch_size=8)
    assert one.predict_values(inputs) == pytest.approx(many.predict_values(inputs), abs=1e-4)


def test_a_blank_candidate_is_a_recorded_failure_not_an_exception():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    results = predictor.predict(["", "   ", None])
    assert [result.is_numeric for result in results] == [False, False, False]
    assert all(result.value is None for result in results)


def test_call_returns_nan_for_a_failure_so_the_tp_metric_can_drop_it():
    model, tokenizer, _ = build()
    predictor = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    values = predictor(["", "[At][C][C][At]"])
    assert math.isnan(values[0])
    assert math.isfinite(values[1])


def test_round_trip_through_a_checkpoint_preserves_predictions(tmp_path):
    model, tokenizer, head_config = build(n_descriptors=3)
    path = write_checkpoint(tmp_path / "best.pt", model, tokenizer, head_config)
    reloaded = RegressionPropertyPredictor.from_checkpoint(path, device="cpu")
    inputs = ["[At][C][C][O][At]", "[At][C][C][At]"]
    original = RegressionPropertyPredictor(model, tokenizer, device="cpu")
    assert reloaded.predict_values(inputs) == pytest.approx(
        original.predict_values(inputs), abs=1e-5
    )


def test_the_target_scaling_survives_the_checkpoint(tmp_path):
    model, tokenizer, head_config = build()
    path = write_checkpoint(tmp_path / "best.pt", model, tokenizer, head_config)
    reloaded = RegressionPropertyPredictor.from_checkpoint(path, device="cpu")
    assert float(reloaded.model.tg_mean) == pytest.approx(417.0)
    assert float(reloaded.model.tg_std) == pytest.approx(113.0)


def test_a_checkpoint_without_head_metadata_is_refused(tmp_path):
    model, tokenizer, _ = build()
    path = save_checkpoint(
        tmp_path / "bare.pt", model=model, epoch=0, global_step=1, config={},
        model_config=model.config.to_dict(), tokenizer_path=str(VOCAB),
        tokenizer_sha256=tokenizer.sha256,
    )
    with pytest.raises(ValueError, match=GROUP_A_CONFIG_KEY):
        RegressionPropertyPredictor.from_checkpoint(path, device="cpu")


def test_a_tokenizer_mismatch_is_refused(tmp_path):
    model, tokenizer, head_config = build()
    path = save_checkpoint(
        tmp_path / "wrong.pt", model=model, epoch=0, global_step=1,
        config={GROUP_A_CONFIG_KEY: {"heads": head_config.to_dict()}},
        model_config=model.config.to_dict(), tokenizer_path=str(VOCAB),
        tokenizer_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        RegressionPropertyPredictor.from_checkpoint(path, device="cpu")


def test_a_group_a_checkpoint_cannot_be_loaded_as_a_baseline_predictor(tmp_path):
    """Spec 7: Group A produces models ALONGSIDE the existing ones. A silent
    half-load into a reward path is the one failure mode that must not exist."""
    model, tokenizer, head_config = build()
    path = write_checkpoint(tmp_path / "best.pt", model, tokenizer, head_config)
    with pytest.raises((RuntimeError, KeyError, ValueError)):
        PolyT5PropertyPredictor.from_checkpoint(path, tokenizer_path=VOCAB, device="cpu")


def test_repr_names_the_arm_free_essentials():
    model, tokenizer, _ = build()
    text = repr(RegressionPropertyPredictor(model, tokenizer, device="cpu", property_name="Tg"))
    assert "RegressionPropertyPredictor" in text
    assert "Tg" in text


def test_degenerate_batch_size_is_refused():
    model, tokenizer, _ = build()
    with pytest.raises(ValueError, match="batch_size"):
        RegressionPropertyPredictor(model, tokenizer, device="cpu", batch_size=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_predictor.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.inference.regression_predictor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/inference/regression_predictor.py
"""Score PSELFIES with the Group A regression head instead of decoding text.

The baseline predictor generates the number one character at a time under beam
search, so a decode can fail to be a number at all. The regression head cannot:
it emits one float per input. ``non_numeric_rate`` is therefore structurally
0.0 for this predictor, and it is still reported -- an honest zero is better
than an omitted column that invites the reader to carry the old number over.

A Group A checkpoint deliberately does NOT load in
:class:`polyt5.inference.PolyT5PropertyPredictor`: its ``state_dict`` keys are
namespaced under ``backbone.`` and it carries head metadata the baseline loader
knows nothing about. Group A produces new models ALONGSIDE the existing five,
never replacements, and a silent half-load into a reward path is the one
failure this design refuses to allow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from polyt5.data.prepare import format_property_value
from polyt5.inference.predictor import (
    DEFAULT_MAX_SOURCE_LENGTH,
    NON_NUMERIC_VALUE,
    PredictionResult,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import load_checkpoint
from polyt5.utils import select_device

__all__ = ["GROUP_A_CONFIG_KEY", "RegressionPropertyPredictor"]

#: Run-config key under which a Group A checkpoint stores its head metadata.
GROUP_A_CONFIG_KEY = "group_a"


class RegressionPropertyPredictor:
    """Load a Group A checkpoint and score PSELFIES with its regression head."""

    def __init__(
        self,
        model: PolyT5MultiTask,
        tokenizer: PolyT5Tokenizer,
        *,
        device: str = "auto",
        batch_size: int = 64,
        max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
        property_name: str | None = None,
    ) -> None:
        """Wrap a loaded model for inference.

        Args:
            model: A :class:`~polyt5.model.multitask.PolyT5MultiTask` whose
                target scaling has been set.
            tokenizer: The tokenizer the model was trained with.
            device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit device.
            batch_size: Candidates per forward pass. A throughput knob only:
                the head is independent per row, so batching must not change
                any prediction.
            max_source_length: Input truncation length. # [PAPER] 200. Clamped
                to the model's ``n_positions`` when that is smaller.
            property_name: Optional label ("Tg") carried for bookkeeping.

        Raises:
            ValueError: If ``batch_size`` is below 1, or the model has no
                regression head.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if model.tg_head is None:
            raise ValueError(
                "this model has no regression head; use PolyT5PropertyPredictor for a "
                "text-decoding checkpoint"
            )
        self.device = select_device("auto") if device == "auto" else str(device)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.property_name = property_name

        n_positions = getattr(model.config, "n_positions", None)
        self.max_source_length = (
            min(int(max_source_length), int(n_positions))
            if isinstance(n_positions, int) and n_positions > 0
            else int(max_source_length)
        )
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        *,
        device: str = "auto",
        allow_unverified_tokenizer: bool = False,
        **kwargs: Any,
    ) -> RegressionPropertyPredictor:
        """Rebuild a predictor from a Group A checkpoint.

        Args:
            checkpoint_path: A ``.pt`` file written during a Group A run.
            tokenizer_path: Tokenizer artifact; defaults to the path recorded
                inside the checkpoint.
            device: Passed to :meth:`__init__`.
            allow_unverified_tokenizer: Permit a checkpoint with no recorded
                ``tokenizer_sha256``. Off by default: a wrong vocabulary yields
                plausible wrong NUMBERS rather than a crash.
            **kwargs: Forwarded to :meth:`__init__`.

        Returns:
            A ready-to-use predictor in ``eval`` mode.

        Raises:
            ValueError: If no tokenizer can be located or verified, or the
                checkpoint carries no ``group_a`` head metadata.
        """
        checkpoint_path = Path(checkpoint_path)
        payload = load_checkpoint(checkpoint_path, map_location="cpu")

        resolved = tokenizer_path or payload.get("tokenizer_path")
        if resolved is None:
            raise ValueError(
                f"{checkpoint_path} records no tokenizer_path and none was supplied"
            )
        tokenizer = PolyT5Tokenizer.from_file(Path(resolved))
        recorded_sha = payload.get("tokenizer_sha256")
        if recorded_sha is None:
            if not allow_unverified_tokenizer:
                raise ValueError(
                    f"{checkpoint_path} recorded no tokenizer_sha256, so the vocabulary "
                    "cannot be verified; pass allow_unverified_tokenizer=True to accept it"
                )
        elif recorded_sha != tokenizer.sha256:
            raise ValueError(
                "tokenizer mismatch: the checkpoint was trained with vocabulary "
                f"{recorded_sha[:16]} but the supplied tokenizer is "
                f"{tokenizer.sha256[:16]}"
            )

        group_a = (payload.get("config") or {}).get(GROUP_A_CONFIG_KEY)
        if not isinstance(group_a, dict) or "heads" not in group_a:
            raise ValueError(
                f"{checkpoint_path} carries no {GROUP_A_CONFIG_KEY!r}.heads metadata, so its "
                "head widths are unknown; it is not a Group A checkpoint"
            )

        backbone = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(payload["model_config"]))
        model = PolyT5MultiTask(backbone, MultiTaskConfig.from_dict(group_a["heads"]))
        model.load_state_dict(payload["model_state"])
        return cls(model, tokenizer, device=device, **kwargs)

    @torch.no_grad()
    def predict(self, pselfies: Sequence[str]) -> list[PredictionResult]:
        """Score PSELFIES strings with the regression head.

        Args:
            pselfies: Candidate polymers as PSELFIES. Entries that are not
                non-empty strings are reported as non-numeric without reaching
                the model.

        Returns:
            One :class:`~polyt5.inference.PredictionResult` per input, in input
            order. Never raises on model output.
        """
        results: list[PredictionResult | None] = [None] * len(pselfies)
        live: list[tuple[int, str]] = []
        for position, entry in enumerate(pselfies):
            if isinstance(entry, str) and entry.strip():
                live.append((position, entry))
            else:
                results[position] = PredictionResult(
                    source=entry if isinstance(entry, str) else str(entry),
                    decoded="", value=None, is_numeric=False,
                )

        for start in range(0, len(live), self.batch_size):
            chunk = live[start : start + self.batch_size]
            encoded = self.tokenizer.batch_encode(
                [text for _, text in chunk], add_eos=True,
                max_length=self.max_source_length, padding=True, truncation=True,
            )
            values = self.model.predict_tg(
                torch.tensor(encoded["input_ids"], device=self.device),
                torch.tensor(encoded["attention_mask"], device=self.device),
            )
            for (position, text), value in zip(chunk, values.tolist(), strict=True):
                finite = bool(value == value and abs(value) != float("inf"))
                results[position] = PredictionResult(
                    source=text,
                    decoded=format_property_value(value) if finite else "",
                    value=float(value) if finite else None,
                    is_numeric=finite,
                )
        return [result for result in results if result is not None]

    def predict_values(self, pselfies: Sequence[str]) -> list[float | None]:
        """Score PSELFIES and return values only, ``None`` for failures."""
        return [result.value for result in self.predict(pselfies)]

    def __call__(self, candidates: Sequence[str]) -> list[float]:
        """Injection point for :func:`polyt5.evaluation.evaluate_generation`.

        Args:
            candidates: Candidate polymers as PSELFIES.

        Returns:
            One float per candidate; a failure is
            :data:`~polyt5.inference.NON_NUMERIC_VALUE` (NaN), which the TP
            metric drops from both numerator and denominator.
        """
        return [
            NON_NUMERIC_VALUE if result.value is None else result.value
            for result in self.predict(candidates)
        ]

    def __repr__(self) -> str:
        return (
            f"RegressionPropertyPredictor(property_name={self.property_name!r}, "
            f"device={self.device!r}, batch_size={self.batch_size})"
        )
```

Add to `src/polyt5/inference/__init__.py`: `from polyt5.inference.regression_predictor import GROUP_A_CONFIG_KEY, RegressionPropertyPredictor` plus the two `__all__` entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_predictor.py -o addopts="" -q`
Expected: PASS — 12 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1194 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/inference/regression_predictor.py src/polyt5/inference/__init__.py tests/test_group_a_predictor.py
git commit -m "feat(group-a): regression-head predictor that refuses to pass as a baseline"
```

---

### Task 12: The ablation matrix and the pre-registered verdict

Spec §6, fixed before any run: a change **helps** if its five-split mean MAE is below **27.9142 K** (28.6733 − 0.7591). A change landing inside the baseline's own spread is recorded as **no effect**, not a small win. A configuration that **hurts** is reported with the same prominence as one that helps.

Spec §9: three claims never merge. This module stamps every matrix with an explicit `claim_category` and refuses to render a row for an unrun configuration as anything but `not run` with `None` metrics, so an empty cell cannot be misread as a zero.

**Files:**
- Create: `src/polyt5/evaluation/ablation.py`
- Modify: `src/polyt5/evaluation/__init__.py`
- Test: `tests/test_group_a_ablation.py`

**Interfaces:**
- Consumes: `polyt5.evaluation.regression_metrics.{RegressionReport, aggregate_over_splits}` (numpy only — no torch)
- Produces:
  - `BASELINE_ARM: str = "B0"`
  - `EFFECT_HELPS: str = "helps"`, `EFFECT_NO_EFFECT: str = "no effect"`, `EFFECT_HURTS: str = "hurts"`, `EFFECT_NOT_RUN: str = "not run"`
  - `CLAIM_CATEGORY: str = "our extension obtains"`
  - `success_threshold(baseline_mean: float, baseline_std: float) -> float`
  - `classify_effect(mae_mean: float | None, *, baseline_mean: float, baseline_std: float) -> str`
  - `ArmResult` frozen dataclass: `arm: str`, `switches: dict[str, bool]`, `n_splits: int`, `mae_mean: float | None`, `mae_std: float | None`, `rmse_mean: float | None`, `rmse_std: float | None`, `r2_mean: float | None`, `r2_std: float | None`, `non_numeric_rate_mean: float | None`; classmethod `from_reports(arm: str, switches: dict[str, bool], reports: Sequence[RegressionReport]) -> ArmResult`; method `to_dict(self) -> dict[str, Any]`
  - `build_ablation_matrix(results: Sequence[ArmResult], *, baseline_mean: float, baseline_std: float) -> dict[str, Any]`
  - `format_ablation_matrix(matrix: dict[str, Any]) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_ablation.py
"""Group A Task 12: the pre-registered verdict and the matrix that reports it."""

from __future__ import annotations

import pytest

from polyt5.evaluation.ablation import (
    BASELINE_ARM,
    CLAIM_CATEGORY,
    EFFECT_HELPS,
    EFFECT_HURTS,
    EFFECT_NO_EFFECT,
    EFFECT_NOT_RUN,
    ArmResult,
    build_ablation_matrix,
    classify_effect,
    format_ablation_matrix,
    success_threshold,
)
from polyt5.evaluation.regression_metrics import RegressionReport

FROZEN_MEAN = 28.6733
FROZEN_STD = 0.7591


def report(mae: float, rmse: float = 44.0, r2: float = 0.84) -> RegressionReport:
    return RegressionReport(
        n_total=1471, n_valid_numeric=1471, n_non_numeric=0, non_numeric_rate=0.0,
        mae=mae, rmse=rmse, r2=r2, pearson_r=0.92,
    )


def arm(name: str, maes, switches=None) -> ArmResult:
    return ArmResult.from_reports(
        name, switches or {"regression_head": name == "A1"}, [report(m) for m in maes]
    )


def test_the_threshold_is_the_baseline_minus_one_standard_deviation():
    assert success_threshold(FROZEN_MEAN, FROZEN_STD) == pytest.approx(27.9142, abs=1e-4)


def test_a_gain_inside_the_baseline_spread_is_no_effect_not_a_small_win():
    assert classify_effect(28.2, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NO_EFFECT
    )
    assert classify_effect(28.6733, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NO_EFFECT
    )


def test_a_mae_below_the_threshold_helps():
    assert classify_effect(27.0, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_HELPS
    )


def test_a_mae_above_the_upper_spread_hurts():
    assert classify_effect(31.0, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_HURTS
    )


def test_an_unrun_configuration_is_not_run_not_zero():
    assert classify_effect(None, baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD) == (
        EFFECT_NOT_RUN
    )


def test_arm_result_aggregates_the_five_splits():
    result = arm("A1", [27.0, 27.4, 26.8, 27.2, 27.1])
    assert result.n_splits == 5
    assert result.mae_mean == pytest.approx(27.1, abs=1e-6)
    assert result.mae_std is not None
    assert result.non_numeric_rate_mean == pytest.approx(0.0)


def test_arm_result_with_no_reports_carries_none_metrics():
    result = ArmResult.from_reports("A5", {"multitask": True}, [])
    assert result.n_splits == 0
    assert result.mae_mean is None
    assert result.rmse_mean is None


def test_the_matrix_reports_every_arm_including_the_unrun_ones():
    matrix = build_ablation_matrix(
        [arm("B0", [28.7]), arm("A1", [27.0]), ArmResult.from_reports("A2", {}, [])],
        baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD,
    )
    assert [row["arm"] for row in matrix["rows"]] == ["B0", "A1", "A2"]
    assert matrix["rows"][2]["effect"] == EFFECT_NOT_RUN
    assert matrix["rows"][2]["mae_mean"] is None


def test_the_matrix_stamps_the_claim_category_and_the_comparison_point():
    matrix = build_ablation_matrix([arm("A1", [27.0])], baseline_mean=FROZEN_MEAN,
                                   baseline_std=FROZEN_STD)
    assert matrix["claim_category"] == CLAIM_CATEGORY
    assert matrix["baseline_arm"] == BASELINE_ARM
    assert matrix["baseline_mae_mean"] == pytest.approx(FROZEN_MEAN)
    assert matrix["success_threshold"] == pytest.approx(27.9142, abs=1e-4)
    assert "paper" in matrix["claim_note"].lower()


def test_the_matrix_refuses_a_duplicate_arm():
    with pytest.raises(ValueError, match="duplicate"):
        build_ablation_matrix([arm("A1", [27.0]), arm("A1", [26.0])],
                              baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD)


def test_the_matrix_refuses_a_nonpositive_baseline_spread():
    with pytest.raises(ValueError, match="baseline_std"):
        build_ablation_matrix([arm("A1", [27.0])], baseline_mean=FROZEN_MEAN, baseline_std=0.0)


def test_a_hurting_arm_is_rendered_as_prominently_as_a_helping_one():
    """Spec 6: 'reported with the same prominence'. Row order is arm order, not
    a leaderboard, and no arm is elided."""
    matrix = build_ablation_matrix(
        [arm("A1", [27.0]), arm("A2", [33.0]), arm("A3", [28.4])],
        baseline_mean=FROZEN_MEAN, baseline_std=FROZEN_STD,
    )
    rendered = format_ablation_matrix(matrix)
    assert [row["arm"] for row in matrix["rows"]] == ["A1", "A2", "A3"]
    for arm_id, verdict in (("A1", EFFECT_HELPS), ("A2", EFFECT_HURTS),
                            ("A3", EFFECT_NO_EFFECT)):
        line = next(line for line in rendered.splitlines() if line.strip().startswith(arm_id))
        assert verdict in line


def test_the_rendered_table_names_the_threshold_and_the_claim_category():
    matrix = build_ablation_matrix([arm("A1", [27.0])], baseline_mean=FROZEN_MEAN,
                                   baseline_std=FROZEN_STD)
    rendered = format_ablation_matrix(matrix)
    assert "27.91" in rendered
    assert CLAIM_CATEGORY in rendered
    assert "non_numeric" in rendered


def test_evaluation_package_still_imports_without_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import polyt5.evaluation; sys.exit(1 if 'torch' in sys.modules else 0)"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[:500]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_ablation.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.evaluation.ablation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/polyt5/evaluation/ablation.py
"""The Group A ablation matrix and its pre-registered verdict.

The criterion was fixed before any run (spec section 6). A configuration
**helps** only if its five-split mean MAE is below the frozen baseline's mean
minus one standard deviation -- 28.6733 - 0.7591 = 27.9142 K. A configuration
landing inside that spread is **no effect**, not a small win, and one above it
**hurts**.

Reporting discipline (spec section 9). Three claims never merge:

* *the paper reports* -- Tg prediction on 5,130 withheld labels;
* *our reproduction obtains* -- 28.6733 +/- 0.7591 K on 7,367 LamaLab labels;
* *our extension obtains* -- Group A.

Group A results are the third category and are stamped as such. An unrun
configuration renders as ``not run`` with ``None`` metrics: an empty cell must
never be mistakable for a zero. Rows are emitted in the order they are given
(arm order), never sorted into a leaderboard, so a configuration that hurts is
as prominent as one that helps.

Torch-free, like the rest of :mod:`polyt5.evaluation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from polyt5.evaluation.regression_metrics import RegressionReport, aggregate_over_splits

__all__ = [
    "BASELINE_ARM",
    "CLAIM_CATEGORY",
    "CLAIM_NOTE",
    "EFFECT_HELPS",
    "EFFECT_HURTS",
    "EFFECT_NOT_RUN",
    "EFFECT_NO_EFFECT",
    "ArmResult",
    "build_ablation_matrix",
    "classify_effect",
    "success_threshold",
]

#: The configuration every other one is compared against.
BASELINE_ARM = "B0"

EFFECT_HELPS = "helps"
EFFECT_NO_EFFECT = "no effect"
EFFECT_HURTS = "hurts"
EFFECT_NOT_RUN = "not run"

#: Which of the three never-merged claims a Group A number is.
CLAIM_CATEGORY = "our extension obtains"

CLAIM_NOTE = (
    "Group A numbers are our extension's improvements to OUR OWN reproduction. They are "
    "not comparable to the paper's reported Tg figures, which are a different quantity "
    "(5,130 withheld labels) measured on a different dataset, and must never be presented "
    "as closing a gap to the paper."
)


def success_threshold(baseline_mean: float, baseline_std: float) -> float:
    """The pre-registered MAE a configuration must beat to count as helping.

    Args:
        baseline_mean: Frozen baseline five-split mean MAE (K).
        baseline_std: Its standard deviation over those splits (K).

    Returns:
        ``baseline_mean - baseline_std``.
    """
    return baseline_mean - baseline_std


def classify_effect(
    mae_mean: float | None, *, baseline_mean: float, baseline_std: float
) -> str:
    """Apply the pre-registered verdict to one configuration's mean MAE.

    Args:
        mae_mean: The configuration's five-split mean MAE, or ``None`` when it
            has not been run.
        baseline_mean: Frozen baseline mean MAE.
        baseline_std: Frozen baseline standard deviation.

    Returns:
        One of :data:`EFFECT_HELPS`, :data:`EFFECT_NO_EFFECT`,
        :data:`EFFECT_HURTS`, :data:`EFFECT_NOT_RUN`.
    """
    if mae_mean is None:
        return EFFECT_NOT_RUN
    if mae_mean < success_threshold(baseline_mean, baseline_std):
        return EFFECT_HELPS
    if mae_mean > baseline_mean + baseline_std:
        return EFFECT_HURTS
    return EFFECT_NO_EFFECT


@dataclass(frozen=True)
class ArmResult:
    """One configuration's aggregated five-split metrics."""

    arm: str
    switches: dict[str, bool]
    n_splits: int
    mae_mean: float | None
    mae_std: float | None
    rmse_mean: float | None
    rmse_std: float | None
    r2_mean: float | None
    r2_std: float | None
    non_numeric_rate_mean: float | None

    @classmethod
    def from_reports(
        cls, arm: str, switches: dict[str, bool], reports: Sequence[RegressionReport]
    ) -> ArmResult:
        """Aggregate one configuration's per-split reports.

        Args:
            arm: Configuration id.
            switches: Which of the five changes this configuration turns on.
            reports: One report per completed split; an empty sequence yields
                an all-``None`` result rather than zeros.

        Returns:
            An :class:`ArmResult`.
        """
        aggregate = aggregate_over_splits(reports)

        def value(metric: str, field: str) -> float | None:
            entry = aggregate.get(metric) or {}
            return entry.get(field)

        return cls(
            arm=arm,
            switches=dict(switches),
            n_splits=len(reports),
            mae_mean=value("mae", "mean"),
            mae_std=value("mae", "std"),
            rmse_mean=value("rmse", "mean"),
            rmse_std=value("rmse", "std"),
            r2_mean=value("r2", "mean"),
            r2_std=value("r2", "std"),
            non_numeric_rate_mean=value("non_numeric_rate", "mean"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "arm": self.arm,
            "switches": dict(self.switches),
            "n_splits": self.n_splits,
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "rmse_mean": self.rmse_mean,
            "rmse_std": self.rmse_std,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
            "non_numeric_rate_mean": self.non_numeric_rate_mean,
        }


def build_ablation_matrix(
    results: Sequence[ArmResult], *, baseline_mean: float, baseline_std: float
) -> dict[str, Any]:
    """Assemble the arm x metric matrix with its verdicts and provenance.

    Args:
        results: One :class:`ArmResult` per configuration, in the order they
            should be reported.
        baseline_mean: Frozen baseline five-split mean MAE.
        baseline_std: Its standard deviation.

    Returns:
        A JSON-serialisable matrix.

    Raises:
        ValueError: If an arm appears twice, or ``baseline_std`` is not
            positive (which would collapse the verdict to a point comparison).
    """
    if baseline_std <= 0.0:
        raise ValueError(
            f"baseline_std must be > 0, got {baseline_std}: without a spread there is no "
            "'no effect' band and every difference would read as a result"
        )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.arm in seen:
            raise ValueError(f"duplicate arm {result.arm!r} in the ablation matrix")
        seen.add(result.arm)
        rows.append(
            {
                **result.to_dict(),
                "effect": classify_effect(
                    result.mae_mean, baseline_mean=baseline_mean, baseline_std=baseline_std
                ),
                "mae_delta_vs_baseline": (
                    None if result.mae_mean is None else result.mae_mean - baseline_mean
                ),
            }
        )
    return {
        "claim_category": CLAIM_CATEGORY,
        "claim_note": CLAIM_NOTE,
        "baseline_arm": BASELINE_ARM,
        "baseline_mae_mean": baseline_mean,
        "baseline_mae_std": baseline_std,
        "success_threshold": success_threshold(baseline_mean, baseline_std),
        "rows": rows,
    }


def _cell(row: dict[str, Any], key: str, width: int, digits: int) -> str:
    """Render one numeric cell, or right-aligned ``n/a`` when it is ``None``."""
    value = row.get(key)
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def format_ablation_matrix(matrix: dict[str, Any]) -> str:
    """Render the matrix as a fixed-width table.

    Args:
        matrix: Output of :func:`build_ablation_matrix`.

    Returns:
        The table as a string. The caller prints it; this module never does.
    """
    header = (
        f"{'arm':<5}{'MAE':>10}{'+/-':>8}{'RMSE':>10}{'R2':>8}"
        f"{'non_numeric':>13}{'splits':>8}  verdict"
    )
    lines = [
        f"Group A ablation -- {matrix['claim_category']}",
        (
            f"baseline {matrix['baseline_arm']} = {matrix['baseline_mae_mean']:.4f} "
            f"+/- {matrix['baseline_mae_std']:.4f} K; "
            f"helps below {matrix['success_threshold']:.4f} K"
        ),
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for row in matrix["rows"]:
        lines.append(
            f"{row['arm']:<5}{_cell(row, 'mae_mean', 10, 4)}{_cell(row, 'mae_std', 8, 4)}"
            f"{_cell(row, 'rmse_mean', 10, 4)}{_cell(row, 'r2_mean', 8, 4)}"
            f"{_cell(row, 'non_numeric_rate_mean', 13, 4)}{row['n_splits']:>8}  "
            f"{row['effect']}"
        )
    lines.extend(["-" * len(header), matrix["claim_note"]])
    return "\n".join(lines)
```

Add to `src/polyt5/evaluation/__init__.py`: `from .ablation import BASELINE_ARM, CLAIM_CATEGORY, CLAIM_NOTE, EFFECT_HELPS, EFFECT_HURTS, EFFECT_NOT_RUN, EFFECT_NO_EFFECT, ArmResult, build_ablation_matrix, classify_effect, success_threshold` plus the eleven `__all__` entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_ablation.py -o addopts="" -q`
Expected: PASS — 14 passed

- [ ] **Step 5: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1208 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/evaluation/ablation.py src/polyt5/evaluation/__init__.py tests/test_group_a_ablation.py
git commit -m "feat(group-a): pre-registered ablation verdict at 27.91 K and its matrix"
```

---

### Task 13: The ablation runner

Seven configurations over the **same five splits** the frozen baseline used. The splits are not rebuilt — they are **loaded from `results/tg_prediction_5splits_medium92m/splits.json` and validated**, because a silently rebuilt split would produce numbers that look comparable to 28.6733 K and are not.

**This task writes and unit-tests the runner. It does not run it.** A GRPO run may be in flight writing to `results/grpo_control/`; the runner's default output root is `results/group_a/`, and no step here executes training.

**Files:**
- Create: `configs/finetune/group_a.yaml`, `scripts/run_group_a.py`
- Test: `tests/test_run_group_a.py`

**Interfaces:**
- Consumes: `polyt5.data.tg_metadata.{prepare_labeled_rows, read_lamalab_rows}`, `polyt5.data.multitask.{TaskCollator, TaskDataset, assemble_split}`, `polyt5.data.splits.load_splits`, `polyt5.training.group_a.{ARM_IDS, GroupAConfig, arm_config}`, `polyt5.training.multitask_trainer.{GroupATrainer, InterleavedLoader}`, `polyt5.training.{TrainerConfig, load_checkpoint}`, `polyt5.model.{PolyT5Config, PolyT5ForConditionalGeneration}`, `polyt5.model.multitask.{MultiTaskConfig, PolyT5MultiTask}`, `polyt5.inference.regression_predictor.{GROUP_A_CONFIG_KEY, RegressionPropertyPredictor}`, `polyt5.inference.PolyT5PropertyPredictor`, `polyt5.evaluation.{ArmResult, build_ablation_matrix, format_ablation_matrix, regression_report}`, `polyt5.utils.{RunDirectory, get_logger, load_config, parse_dotted_overrides, require, save_config, seed_everything, select_device}`
- Produces (in `scripts/run_group_a.py`):
  - `FrozenSplit` frozen dataclass: `index: int`, `train: list[int]`, `val: list[int]`, `test: list[int]`
  - `load_frozen_splits(path: str | Path, *, n_examples: int) -> list[FrozenSplit]`
  - `resolve_arms(requested: Sequence[str] | None, **overrides: Any) -> list[GroupAConfig]`
  - `load_baseline_reference(path: str | Path) -> tuple[float, float]` — `(mae_mean, mae_std)` from `frozen_baseline.json` → `tg_prediction_5split.mae`
  - `build_arm_model(group_a: GroupAConfig, n_descriptors: int, *, model_config, init_checkpoint: Path | None, tokenizer, logger, device: str) -> tuple[PolyT5MultiTask, MultiTaskConfig, bool]`
  - `parse_args(argv: list[str] | None = None) -> argparse.Namespace`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_group_a.py
"""Group A Task 13: the runner's guards. Nothing here trains anything."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from polyt5.training.group_a import ARM_IDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_group_a  # noqa: E402


def write_splits(path: Path, *, n: int = 10, n_splits: int = 2) -> Path:
    splits = []
    for index in range(n_splits):
        order = [(i + index) % n for i in range(n)]
        splits.append(
            {"index": index, "train": order[:6], "val": order[6:8], "test": order[8:]}
        )
    path.write_text(
        json.dumps({"task": "tg_prediction", "n": n, "base_seed": 0,
                    "n_splits": n_splits, "splits": splits}),
        encoding="utf-8",
    )
    return path


def test_frozen_splits_load_and_validate(tmp_path):
    path = write_splits(tmp_path / "splits.json")
    splits = run_group_a.load_frozen_splits(path, n_examples=10)
    assert [s.index for s in splits] == [0, 1]
    assert len(splits[0].train) == 6
    assert len(splits[0].test) == 2


def test_a_corpus_of_the_wrong_size_is_refused(tmp_path):
    """The one guard that keeps every Group A number comparable to 28.67 K."""
    path = write_splits(tmp_path / "splits.json", n=10)
    with pytest.raises(ValueError, match="7354|9|indices"):
        run_group_a.load_frozen_splits(path, n_examples=9)


def test_overlapping_train_and_test_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"n": 4, "n_splits": 1,
                    "splits": [{"index": 0, "train": [0, 1, 2], "val": [], "test": [2, 3]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_a_split_that_does_not_cover_every_index_is_refused(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(
        json.dumps({"n": 4, "n_splits": 1,
                    "splits": [{"index": 0, "train": [0, 1], "val": [], "test": [2]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cover"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_a_file_with_no_splits_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"n": 4, "splits": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no splits"):
        run_group_a.load_frozen_splits(path, n_examples=4)


def test_resolve_arms_defaults_to_all_seven_in_order():
    assert [c.arm for c in run_group_a.resolve_arms(None)] == list(ARM_IDS)


def test_resolve_arms_honours_a_subset_and_deduplicates():
    assert [c.arm for c in run_group_a.resolve_arms(["A3", "B0", "A3"])] == ["B0", "A3"]


def test_resolve_arms_passes_hyperparameter_overrides_through():
    configs = run_group_a.resolve_arms(["A2", "A3"], descriptor_lambda=0.4, n_writings=6)
    assert configs[0].descriptor_lambda == 0.4
    assert configs[1].n_writings == 6


def test_resolve_arms_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="A9"):
        run_group_a.resolve_arms(["A9"])


def test_baseline_reference_comes_from_the_frozen_artifact(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps({"tg_prediction_5split": {"mae": {"mean": 28.6733, "std": 0.7591}}}),
        encoding="utf-8",
    )
    assert run_group_a.load_baseline_reference(path) == (28.6733, 0.7591)


def test_baseline_reference_refuses_an_artifact_without_the_key(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"tg_prediction_5split": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mae"):
        run_group_a.load_baseline_reference(path)


def test_the_real_frozen_artifact_still_says_what_the_plan_assumes():
    repo = Path(__file__).resolve().parents[1]
    artifact = repo / "artifacts" / "baseline" / "frozen_baseline.json"
    if not artifact.is_file():
        pytest.skip("frozen baseline artifact missing")
    mean, std = run_group_a.load_baseline_reference(artifact)
    assert mean == pytest.approx(28.6733, abs=1e-4)
    assert std == pytest.approx(0.7591, abs=1e-4)


def test_default_output_root_is_not_a_live_run_directory():
    args = run_group_a.parse_args([])
    assert Path(args.out).as_posix().endswith("results/group_a")
    assert "grpo" not in Path(args.out).as_posix()


def test_arms_can_be_selected_on_the_command_line():
    args = run_group_a.parse_args(["--arm", "A1", "--arm", "A4"])
    assert args.arm == ["A1", "A4"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_run_group_a.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_group_a'`

- [ ] **Step 3: Write the config**

```yaml
# configs/finetune/group_a.yaml
# Phase 4 Group A ablation (docs/superpowers/specs/2026-08-23-phase4-group-a-design.md).
#
# NOT part of the published polyT5 method. The five switches per arm come from
# polyt5.training.group_a.arm_config, NOT from this file: "A6 is all five
# combined" must be a property a test asserts, not a convention maintained here.
# This file carries only what is shared across arms.
#
# PAPER values (inherited from tg_prediction.yaml): 30 epochs, batch 16, AdamW,
#   lr 3e-4, weight decay 0.01, max length 200, beam width 4, five splits.
# OURS: the head hyperparameters, and reusing the frozen splits verbatim.

task: tg_group_a
seed: 0
data_mode: proxy

data:
  source: lamalab_tg
  csv_path: data/external/LAMALAB_CURATED_Tg.csv
  property_name: Tg
  decimals: 1               # PAPER: targets like "236.0"
  max_length: 200           # PAPER
  deduplicate: true

splits:
  # OURS: reuse the frozen splits verbatim. Rebuilding them would produce
  # numbers that LOOK comparable to 28.6733 K and are not.
  frozen_file: results/tg_prediction_5splits_medium92m/splits.json

baseline:
  frozen_file: artifacts/baseline/frozen_baseline.json

model:
  config: configs/model/polyt5_medium.yaml

group_a:
  descriptor_lambda: 0.1    # OURS  [AMBIGUITY] sensitivity is reported, not assumed
  n_writings: 4             # OURS  [AMBIGUITY] N is measured by A3, not assumed
  std_floor: 5.6            # OURS: median spread over the 279 repeated measurements
  huber_delta: 1.0          # OURS: in standardised units
  head_dropout: 0.1         # OURS: matches the backbone's dropout_rate

training:
  epochs: 30                # PAPER
  batch_size: 16            # PAPER
  optimizer: adamw          # PAPER
  learning_rate: 3.0e-4     # PAPER
  weight_decay: 0.01        # PAPER
  max_source_length: 200    # PAPER
  max_target_length: 200    # PAPER
  scheduler: constant
  amp: true
  amp_dtype: bf16
  num_workers: 0

evaluation:
  batch_size: 32
  beam_width: 4             # PAPER, for the text-head arms only
  max_target_length: 32
  length_penalty: 1.0
```

- [ ] **Step 4: Write the runner**

```python
# scripts/run_group_a.py
"""Run the seven Group A configurations over the FROZEN five splits.

Spec section 5: seven configurations, each on the same five splits the frozen
baseline used, so every number is directly comparable to 28.6733 +/- 0.7591 K.
Individual ablations run AS WELL AS the combination -- a combined gain with no
per-change attribution cannot tell you which idea to keep.

The splits are LOADED and VALIDATED, never rebuilt: a silently rebuilt split
would produce numbers that look comparable and are not.
:func:`load_frozen_splits` refuses a corpus whose size does not match the
frozen file, refuses overlapping train/val/test, and refuses a split that does
not cover every index.

Nothing here touches the frozen artifacts or any existing results directory.
Output goes under ``--out`` (default ``results/group_a``).

Usage:
    python scripts/run_group_a.py --init-checkpoint <pretrained.pt>
    python scripts/run_group_a.py --arm A3 --set group_a.n_writings=8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.multitask import TaskCollator, TaskDataset, assemble_split  # noqa: E402
from polyt5.data.splits import load_splits  # noqa: E402
from polyt5.data.tg_metadata import prepare_labeled_rows, read_lamalab_rows  # noqa: E402
from polyt5.evaluation import (  # noqa: E402
    ArmResult,
    RegressionReport,
    build_ablation_matrix,
    format_ablation_matrix,
    regression_report,
)
from polyt5.inference.regression_predictor import (  # noqa: E402
    GROUP_A_CONFIG_KEY,
    RegressionPropertyPredictor,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import TrainerConfig, load_checkpoint  # noqa: E402
from polyt5.training.group_a import ARM_IDS, GroupAConfig, arm_config  # noqa: E402
from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader  # noqa: E402
from polyt5.utils import (  # noqa: E402
    RunDirectory,
    describe_device,
    get_logger,
    load_config,
    parse_dotted_overrides,
    require,
    save_config,
    seed_everything,
    select_device,
)

RESULTS_FILENAME = "results.json"
MATRIX_FILENAME = "ablation_matrix.json"


@dataclass(frozen=True)
class FrozenSplit:
    """One split loaded verbatim from the frozen splits file."""

    index: int
    train: list[int]
    val: list[int]
    test: list[int]


def _resolve(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_frozen_splits(path: str | Path, *, n_examples: int) -> list[FrozenSplit]:
    """Load the frozen splits and refuse anything that would break comparability.

    Args:
        path: Path to the frozen ``splits.json``.
        n_examples: Size of the corpus this run prepared.

    Returns:
        One :class:`FrozenSplit` per split, in file order.

    Raises:
        ValueError: If the file records no splits, if its corpus size does not
            match ``n_examples``, if a split's parts overlap, or if they do not
            cover every index.
    """
    payload = load_splits(path)
    splits = payload.get("splits") or []
    if not splits:
        raise ValueError(f"{path} contains no splits")

    recorded = payload.get("n")
    if recorded is not None and int(recorded) != n_examples:
        raise ValueError(
            f"{path} was built over {recorded} examples but this run prepared {n_examples}. "
            "The split indices would point at different polymers, so every Group A number "
            "would be measured on a different test set than the 28.6733 K baseline."
        )

    out: list[FrozenSplit] = []
    for entry in splits:
        train = [int(i) for i in entry["train"]]
        val = [int(i) for i in entry.get("val", [])]
        test = [int(i) for i in entry["test"]]
        parts = {"train": set(train), "val": set(val), "test": set(test)}
        for left in ("train", "val"):
            for right in ("val", "test"):
                if left != right and parts[left] & parts[right]:
                    raise ValueError(
                        f"split {entry['index']}: {left} and {right} must be disjoint, "
                        f"{len(parts[left] & parts[right])} indices appear in both"
                    )
        if parts["train"] | parts["val"] | parts["test"] != set(range(n_examples)):
            raise ValueError(
                f"split {entry['index']} does not cover every index in range({n_examples})"
            )
        out.append(
            FrozenSplit(index=int(entry["index"]), train=train, val=val, test=test)
        )
    return out


def resolve_arms(requested: Sequence[str] | None, **overrides: Any) -> list[GroupAConfig]:
    """Build the configurations to run, in :data:`ARM_IDS` order.

    Args:
        requested: Arm ids to run; ``None`` or empty means all seven.
        **overrides: Hyperparameter overrides applied to every arm.

    Returns:
        One :class:`GroupAConfig` per arm, deduplicated, in spec order.

    Raises:
        ValueError: If an id is not one of :data:`ARM_IDS`.
    """
    wanted = set(requested) if requested else set(ARM_IDS)
    unknown = sorted(wanted - set(ARM_IDS))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; valid arms are {list(ARM_IDS)}")
    return [arm_config(arm, **overrides) for arm in ARM_IDS if arm in wanted]


def load_baseline_reference(path: str | Path) -> tuple[float, float]:
    """Read the frozen five-split Tg MAE mean and standard deviation.

    Args:
        path: Path to ``artifacts/baseline/frozen_baseline.json``.

    Returns:
        ``(mae_mean, mae_std)`` in Kelvin.

    Raises:
        ValueError: If the artifact does not carry ``tg_prediction_5split.mae``
            with both a mean and a std. Hard-coding the numbers instead would
            let the plan drift from the artifact it claims to compare against.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mae = ((payload.get("tg_prediction_5split") or {}).get("mae")) or {}
    mean, std = mae.get("mean"), mae.get("std")
    if mean is None or std is None:
        raise ValueError(
            f"{path} carries no tg_prediction_5split.mae mean/std; the comparison point "
            "must come from the frozen artifact, never from a constant in the runner"
        )
    return float(mean), float(std)


def build_arm_model(
    group_a: GroupAConfig,
    n_descriptors: int,
    *,
    model_config_path: Path,
    init_checkpoint: Path | None,
    tokenizer: PolyT5Tokenizer,
    logger: Any,
    device: str,
) -> tuple[PolyT5MultiTask, MultiTaskConfig, bool]:
    """Build one arm's model, warm-started from the pretrained checkpoint.

    Args:
        group_a: The arm's switches.
        n_descriptors: Width of the descriptor head (kept columns), 0 when off.
        model_config_path: YAML describing the backbone when not warm-starting.
        init_checkpoint: Pretrained polyT5 checkpoint, or ``None``.
        tokenizer: The tokenizer every arm shares.
        logger: Progress logger.
        device: Torch device string.

    Returns:
        ``(model_on_device, head_config, was_pretrained)``.

    Raises:
        ValueError: If the checkpoint was trained on another vocabulary.
    """
    if init_checkpoint is not None:
        state = load_checkpoint(_resolve(init_checkpoint), map_location="cpu")
        recorded = state.get("tokenizer_sha256")
        if recorded and recorded != tokenizer.sha256:
            raise ValueError(
                "tokenizer mismatch: the checkpoint was trained with vocabulary "
                f"{recorded[:16]} but the configured tokenizer is {tokenizer.sha256[:16]}"
            )
        backbone_config = PolyT5Config.from_dict(state["model_config"])
        backbone = PolyT5ForConditionalGeneration(backbone_config)
        backbone.load_state_dict(state["model_state"])
        pretrained = True
        logger.info("arm %s: warm-started from %s", group_a.arm, init_checkpoint)
    else:
        backbone_config = PolyT5Config.from_yaml(model_config_path)
        backbone_config.vocab_size = tokenizer.vocab_size
        backbone_config.pad_token_id = tokenizer.pad_id
        backbone_config.eos_token_id = tokenizer.eos_id
        backbone_config.decoder_start_token_id = tokenizer.decoder_start_token_id
        backbone = PolyT5ForConditionalGeneration(backbone_config)
        pretrained = False
        logger.info("arm %s: RANDOM initialisation (no pretrained checkpoint)", group_a.arm)

    head_config = MultiTaskConfig(
        use_regression_head=group_a.regression_head,
        n_descriptors=n_descriptors if group_a.descriptors else 0,
        descriptor_lambda=group_a.descriptor_lambda,
        huber_delta=group_a.huber_delta,
        head_dropout=backbone_config.dropout_rate,
    )
    return PolyT5MultiTask(backbone, head_config).to(device), head_config, pretrained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "finetune" / "group_a.yaml")
    parser.add_argument("--arm", action="append", default=None, choices=list(ARM_IDS),
                        help="Run only this arm; repeatable. Default: all seven.")
    parser.add_argument("--splits-file", type=Path, default=None,
                        help="Override splits.frozen_file from the config.")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/group_a"))
    parser.add_argument("--only-split", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-run arm/split pairs whose results already exist.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap optimizer steps per split (debug).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of corpus rows read (debug).")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _score_split(
    model: PolyT5MultiTask,
    tokenizer: PolyT5Tokenizer,
    group_a: GroupAConfig,
    tensors: Any,
    *,
    cfg: dict[str, Any],
    device: str,
    logger: Any,
) -> tuple[RegressionReport, list[str]]:
    """Decode or regress the held-out split and score it."""
    eval_cfg = cfg.get("evaluation", {})
    if group_a.regression_head:
        predictor = RegressionPropertyPredictor(
            model, tokenizer, device=device,
            batch_size=int(eval_cfg.get("batch_size", 32)), property_name="Tg",
        )
        predictions = [result.decoded for result in predictor.predict(tensors.test_pselfies)]
    else:
        from polyt5.generation import BeamSearchConfig, beam_search

        model.eval()
        predictions = []
        batch_size = int(eval_cfg.get("batch_size", 32))
        for start in range(0, len(tensors.test_pselfies), batch_size):
            chunk = tensors.test_pselfies[start : start + batch_size]
            encoded = tokenizer.batch_encode(
                chunk, add_eos=True, max_length=int(cfg["data"]["max_length"]),
                padding=True, truncation=True,
            )
            with torch.no_grad():
                output = beam_search(
                    model.backbone,
                    torch.tensor(encoded["input_ids"], device=device),
                    torch.tensor(encoded["attention_mask"], device=device),
                    config=BeamSearchConfig(
                        num_beams=int(eval_cfg.get("beam_width", 4)),  # [PAPER] 4
                        max_length=int(eval_cfg.get("max_target_length", 32)),
                        length_penalty=float(eval_cfg.get("length_penalty", 1.0)),
                        eos_token_id=tokenizer.eos_id,
                        pad_token_id=tokenizer.pad_id,
                        decoder_start_token_id=tokenizer.decoder_start_token_id,
                    ),
                )
            predictions.extend(
                tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
            )
            logger.info("  decoded %d/%d", len(predictions), len(tensors.test_pselfies))
    return regression_report(tensors.test_tg, predictions), predictions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=parse_dotted_overrides(args.set))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    out_root = _resolve(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("polyt5.run_group_a", log_file=out_root / "run_group_a.log")
    device = select_device(cfg.get("train", {}).get("device", "auto"))
    logger.info("device=%s", describe_device(device).to_dict())

    tokenizer_path = _resolve(
        cfg.get("tokenizer", {}).get("path", "artifacts/tokenizer/polyt5_vocab.json")
    )
    tokenizer = PolyT5Tokenizer.from_file(tokenizer_path)

    csv_path = _resolve(require(cfg, "data.csv_path"))
    rows, descriptor_names = read_lamalab_rows(csv_path, limit=args.limit)
    examples, stats = prepare_labeled_rows(
        rows,
        max_tokens=int(cfg["data"]["max_length"]),
        deduplicate=bool(cfg["data"].get("deduplicate", True)),
        tokenizer=tokenizer,
    )
    logger.info("corpus: %d usable rows (attrition %s)", len(examples),
                json.dumps(stats.to_dict()))

    splits_file = _resolve(args.splits_file or require(cfg, "splits.frozen_file"))
    splits = load_frozen_splits(splits_file, n_examples=len(examples))
    logger.info("reusing %d frozen splits from %s", len(splits), splits_file)

    baseline_mean, baseline_std = load_baseline_reference(
        _resolve(require(cfg, "baseline.frozen_file"))
    )
    group_cfg = cfg.get("group_a", {})
    arms = resolve_arms(
        args.arm,
        descriptor_lambda=float(group_cfg.get("descriptor_lambda", 0.1)),
        n_writings=int(group_cfg.get("n_writings", 4)),
        std_floor=float(group_cfg.get("std_floor", 5.6)),
        huber_delta=float(group_cfg.get("huber_delta", 1.0)),
    )
    train_cfg = {**cfg.get("training", {}), **cfg.get("train", {})}
    max_source = int(train_cfg.get("max_source_length", 200))
    max_target = int(train_cfg.get("max_target_length", 200))
    collator = TaskCollator(pad_id=tokenizer.pad_id, max_source_length=max_source,
                            max_target_length=max_target)

    arm_results: list[ArmResult] = []
    for group_a in arms:
        reports: list[RegressionReport] = []
        for split in splits:
            if args.only_split is not None and split.index != args.only_split:
                continue
            run_dir = RunDirectory.create(out_root, f"{group_a.arm}/split_{split.index}")
            results_path = run_dir.root / RESULTS_FILENAME
            if results_path.is_file() and not args.force:
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                reports.append(RegressionReport(**payload["evaluation"]))
                logger.info("%s split %d: already done — skipping", group_a.arm, split.index)
                continue

            seed_everything(seed + split.index)
            tensors = assemble_split(
                examples, descriptor_names,
                train_indices=split.train, val_indices=split.val, test_indices=split.test,
                tokenizer=tokenizer,
                use_regression_head=group_a.regression_head,
                use_descriptors=group_a.descriptors,
                n_writings=group_a.effective_n_writings(),
                use_reliability_weighting=group_a.reliability_weighting,
                std_floor=group_a.std_floor,
                build_generation=group_a.multitask,
                seed=seed + split.index,
                max_source_length=max_source,
                max_target_length=max_target,
            )
            logger.info("%s split %d: %s", group_a.arm, split.index,
                        json.dumps(tensors.to_manifest()))

            n_descriptors = (
                0 if tensors.descriptor_standardizer is None
                else tensors.descriptor_standardizer.n_features
            )
            model, head_config, pretrained = build_arm_model(
                group_a, n_descriptors,
                model_config_path=_resolve(require(cfg, "model.config")),
                init_checkpoint=args.init_checkpoint,
                tokenizer=tokenizer, logger=logger, device=device,
            )
            model.set_target_scaling(
                mean=tensors.target_standardizer.mean[0],
                std=tensors.target_standardizer.std[0],
            )

            batch_size = int(train_cfg.get("batch_size", 16))  # [PAPER] 16
            trainer_config = TrainerConfig(
                max_epochs=int(train_cfg.get("epochs", 30)),   # [PAPER] 30
                physical_batch_size=batch_size,
                gradient_accumulation_steps=int(train_cfg.get(
                    "gradient_accumulation_steps", 1)),
                learning_rate=float(train_cfg.get("learning_rate", 3e-4)),  # [PAPER]
                weight_decay=float(train_cfg.get("weight_decay", 0.01)),    # [PAPER]
                scheduler=str(train_cfg.get("scheduler", "constant")),
                amp=bool(train_cfg.get("amp", True)),
                amp_dtype=str(train_cfg.get("amp_dtype", "bf16")),
                max_steps=args.max_steps,
                seed=seed + split.index,
                device=device,
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
            prediction_loader = DataLoader(
                TaskDataset(tensors.train), batch_size=batch_size, shuffle=True,
                collate_fn=collator, num_workers=trainer_config.num_workers,
            )
            generation_loader = (
                DataLoader(TaskDataset(tensors.train_generation), batch_size=batch_size,
                           shuffle=True, collate_fn=collator,
                           num_workers=trainer_config.num_workers)
                if tensors.train_generation else None
            )
            val_loader = (
                DataLoader(TaskDataset(tensors.val), batch_size=batch_size, shuffle=False,
                           collate_fn=collator, num_workers=trainer_config.num_workers)
                if tensors.val else None
            )

            run_config = {
                **cfg,
                GROUP_A_CONFIG_KEY: {
                    "arm": group_a.arm,
                    "switches": group_a.switches(),
                    "config": group_a.to_dict(),
                    "heads": head_config.to_dict(),
                    "split_index": split.index,
                    "splits_file": str(splits_file),
                    "standardizers": {
                        "target": tensors.target_standardizer.to_dict(),
                        "descriptors": (
                            None if tensors.descriptor_standardizer is None
                            else tensors.descriptor_standardizer.to_dict()
                        ),
                    },
                    "attrition": tensors.to_manifest(),
                },
            }
            save_config(run_config, run_dir.config_path)
            run_dir.write_manifest({
                "stage": "group_a_ablation",
                "arm": group_a.arm,
                "split_index": split.index,
                "pretrained": pretrained,
                "tokenizer_sha256": tokenizer.sha256,
                "tokenizer_path": str(tokenizer_path),
                "model_parameters": model.num_parameters(),
                **tensors.to_manifest(),
            })

            started = time.time()
            trainer = GroupATrainer(
                model, InterleavedLoader(prediction_loader, generation_loader),
                trainer_config, group_a=group_a, val_loader=val_loader, run_dir=run_dir,
                tokenizer_path=tokenizer_path, tokenizer_sha256=tokenizer.sha256,
                run_config=run_config, logger=logger,
            )
            train_metrics = trainer.train()
            train_seconds = time.time() - started

            checkpoint_path = run_dir.checkpoints / "best.pt"
            if not checkpoint_path.exists():
                trainer.save(path=checkpoint_path, train_metrics=train_metrics)

            report, predictions = _score_split(
                model, tokenizer, group_a, tensors, cfg=cfg, device=device, logger=logger
            )
            run_dir.append_jsonl("predictions.jsonl", [
                {"source": source, "target": target, "prediction": prediction}
                for source, target, prediction in zip(
                    tensors.test_pselfies, tensors.test_tg, predictions, strict=True
                )
            ])
            run_dir.write_json(RESULTS_FILENAME, {
                "arm": group_a.arm,
                "split_index": split.index,
                "pretrained": pretrained,
                "train_seconds": train_seconds,
                "training": train_metrics,
                "evaluation": report.to_dict(),
                "checkpoint": str(checkpoint_path),
                "attrition": tensors.to_manifest(),
            })
            logger.info("%s split %d: MAE=%s RMSE=%s R2=%s (non-numeric %.4f)",
                        group_a.arm, split.index, report.mae, report.rmse, report.r2,
                        report.non_numeric_rate)
            reports.append(report)

        arm_results.append(
            ArmResult.from_reports(group_a.arm, group_a.switches(), reports)
        )

    matrix = build_ablation_matrix(
        arm_results, baseline_mean=baseline_mean, baseline_std=baseline_std
    )
    (out_root / MATRIX_FILENAME).write_text(
        json.dumps(matrix, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("\n%s", format_ablation_matrix(matrix))
    logger.info("wrote %s", out_root / MATRIX_FILENAME)
    return 0


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_run_group_a.py -o addopts="" -q`
Expected: PASS — 13 passed

- [ ] **Step 6: Confirm the runner is wired without running it**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" scripts/run_group_a.py --help`
Expected: the usage text listing `--arm {B0,A1,A2,A3,A4,A5,A6}`, `--splits-file`, `--out`.

**Do not run the ablation.** A GRPO run may be writing to `results/grpo_control/`; training here would contend for the same GPU and is out of scope for this plan.

- [ ] **Step 7: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1221 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add scripts/run_group_a.py configs/finetune/group_a.yaml tests/test_run_group_a.py
git commit -m "feat(group-a): seven-arm ablation runner over the frozen five splits"
```

---

### Task 14: The generation regression check against Arm B

Spec §6: "Generation is evaluated separately and must not regress. Any configuration touching the shared encoder is checked against the frozen generation baseline (Arm B: PV 55.8%, TP 58.8%). A prediction gain bought with a generation loss is a trade to surface, not a success to report."

The reference rates are **read from `artifacts/baseline/frozen_baseline.json` → `arm_b_tuned_sampling`**, never hard-coded, so the check cannot drift from the artifact it claims to compare against. Only the arms that touch the shared encoder with a second objective need it — `A5` and `A6` — and `requires_generation_check` says which those are.

**Files:**
- Create: `src/polyt5/evaluation/generation_regression.py`, `scripts/check_group_a_generation.py`
- Modify: `src/polyt5/evaluation/__init__.py`
- Test: `tests/test_group_a_generation_check.py`

**Interfaces:**
- Consumes: `polyt5.evaluation.generation_metrics.GenerationReport`, `polyt5.training.group_a.GroupAConfig` is **not** imported (evaluation must not depend on training) — `requires_generation_check` takes a plain `switches: dict[str, bool]`
- Produces:
  - `DEFAULT_REGRESSION_TOLERANCE: float = 0.02`
  - `GenerationRegressionCheck` frozen dataclass: `arm: str`, `pv_rate: float`, `tp_rate: float | None`, `baseline_pv_rate: float`, `baseline_tp_rate: float`, `pv_delta: float`, `tp_delta: float | None`, `tolerance: float`, `regressed: bool`, `verdict: str`; method `to_dict(self) -> dict[str, Any]`
  - `load_arm_b_generation_baseline(path: str | Path) -> tuple[float, float]`
  - `requires_generation_check(switches: dict[str, bool]) -> bool`
  - `check_generation_regression(report: GenerationReport, *, arm: str, baseline_pv_rate: float, baseline_tp_rate: float, tolerance: float = DEFAULT_REGRESSION_TOLERANCE) -> GenerationRegressionCheck`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group_a_generation_check.py
"""Group A Task 14: a prediction gain bought with a generation loss is a trade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polyt5.evaluation.filters import FilterCounts
from polyt5.evaluation.generation_metrics import GenerationReport
from polyt5.evaluation.generation_regression import (
    DEFAULT_REGRESSION_TOLERANCE,
    check_generation_regression,
    load_arm_b_generation_baseline,
    requires_generation_check,
)

BASELINE_PV = 0.558
BASELINE_TP = 0.5878


def make_report(pv_rate: float, tp_rate: float | None) -> GenerationReport:
    n_input = 1000
    return GenerationReport(
        counts=FilterCounts(n_input=n_input, n_sv=n_input, n_tsd=n_input, n_dd=n_input,
                            n_pv=round(pv_rate * n_input)),
        sr_rate=1.0, n_unique=n_input, n_novel=n_input, duplicate_rate=0.0,
        sa_available=True, n_sa_scored=n_input, sa_mean=3.0, sa_median=3.0,
        sa_fraction_above_6=0.0,
        property_target=500.0, property_tolerance=50.0, n_property_values=n_input,
        property_mean=500.0, property_median=500.0, property_std=10.0,
        target_property_rate=tp_rate,
        fingerprints_available=True, diversity={},
    )


def test_only_the_arms_that_share_the_encoder_with_generation_need_the_check():
    assert requires_generation_check({"multitask": True}) is True
    assert requires_generation_check({"multitask": False, "regression_head": True}) is False
    assert requires_generation_check({}) is False


def test_matching_the_baseline_is_not_a_regression():
    check = check_generation_regression(
        make_report(BASELINE_PV, BASELINE_TP), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False
    assert check.pv_delta == pytest.approx(0.0, abs=1e-6)
    assert check.verdict == "no regression"


def test_a_drop_beyond_the_tolerance_is_a_regression():
    check = check_generation_regression(
        make_report(0.40, BASELINE_TP), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is True
    assert check.pv_delta < 0
    assert "regress" in check.verdict


def test_a_drop_inside_the_tolerance_is_not_a_regression():
    check = check_generation_regression(
        make_report(BASELINE_PV - 0.01, BASELINE_TP), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False
    assert check.tolerance == DEFAULT_REGRESSION_TOLERANCE


def test_a_tp_drop_alone_is_enough_to_fire():
    check = check_generation_regression(
        make_report(BASELINE_PV, 0.40), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is True


def test_an_improvement_is_reported_as_such_not_hidden():
    check = check_generation_regression(
        make_report(0.70, 0.70), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.regressed is False
    assert check.pv_delta > 0
    assert check.verdict == "improved"


def test_a_missing_tp_rate_is_none_and_does_not_read_as_zero():
    check = check_generation_regression(
        make_report(BASELINE_PV, None), arm="A5",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    )
    assert check.tp_rate is None
    assert check.tp_delta is None
    assert check.regressed is False, "an unmeasured TP is not a failed TP"


def test_a_nonpositive_tolerance_is_refused():
    with pytest.raises(ValueError, match="tolerance"):
        check_generation_regression(
            make_report(BASELINE_PV, BASELINE_TP), arm="A5",
            baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP, tolerance=0.0,
        )


def test_the_baseline_rates_come_from_the_frozen_artifact(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps({"arm_b_tuned_sampling": {"pv_rate": 0.558, "tp_rate": 0.5878}}),
        encoding="utf-8",
    )
    assert load_arm_b_generation_baseline(path) == (0.558, 0.5878)


def test_an_artifact_without_arm_b_is_refused(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"arm_a_default_sampling": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="arm_b_tuned_sampling"):
        load_arm_b_generation_baseline(path)


def test_the_real_frozen_artifact_still_says_pv_558_and_tp_588():
    repo = Path(__file__).resolve().parents[1]
    artifact = repo / "artifacts" / "baseline" / "frozen_baseline.json"
    if not artifact.is_file():
        pytest.skip("frozen baseline artifact missing")
    pv_rate, tp_rate = load_arm_b_generation_baseline(artifact)
    assert pv_rate == pytest.approx(0.558, abs=1e-4)
    assert tp_rate == pytest.approx(0.5878, abs=1e-4)


def test_the_check_serialises_for_the_run_directory():
    payload = check_generation_regression(
        make_report(0.50, 0.50), arm="A6",
        baseline_pv_rate=BASELINE_PV, baseline_tp_rate=BASELINE_TP,
    ).to_dict()
    assert payload["arm"] == "A6"
    assert payload["regressed"] is True
    assert payload["baseline_pv_rate"] == pytest.approx(BASELINE_PV)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_generation_check.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'polyt5.evaluation.generation_regression'`

- [ ] **Step 3: Write the check module**

```python
# src/polyt5/evaluation/generation_regression.py
"""Did a shared-encoder arm cost us generation quality?

The multi-task arms give one encoder two objectives, and a gain on prediction
may be paid for out of generation. Spec section 6 requires the trade to be
SURFACED, not absorbed: "a prediction gain bought with a generation loss is a
trade to surface, not a success to report".

The reference rates are read from ``artifacts/baseline/frozen_baseline.json``
under ``arm_b_tuned_sampling`` -- never hard-coded here -- so this check cannot
drift away from the artifact it claims to compare against. Arm B is the tuned
sampling point (temperature 0.7, top_p 0.95); comparing against a different
sampling point would make the row incomparable on every column.

An unmeasured TP rate stays ``None``. It is not a failed TP, and it must not
be averaged as a zero.

Torch-free, like the rest of :mod:`polyt5.evaluation`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyt5.evaluation.generation_metrics import GenerationReport

__all__ = [
    "DEFAULT_REGRESSION_TOLERANCE",
    "GenerationRegressionCheck",
    "check_generation_regression",
    "load_arm_b_generation_baseline",
    "requires_generation_check",
]

#: How far below the frozen rate an arm may land before it counts as a
#: regression. Two percentage points; the frozen rates are quoted to one
#: decimal place (55.8%, 58.8%), so a tighter band would be reading noise.
DEFAULT_REGRESSION_TOLERANCE = 0.02

_ARM_B_KEY = "arm_b_tuned_sampling"


@dataclass(frozen=True)
class GenerationRegressionCheck:
    """One arm's generation rates against the frozen Arm B rates."""

    arm: str
    pv_rate: float
    tp_rate: float | None
    baseline_pv_rate: float
    baseline_tp_rate: float
    pv_delta: float
    tp_delta: float | None
    tolerance: float
    regressed: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "arm": self.arm,
            "pv_rate": self.pv_rate,
            "tp_rate": self.tp_rate,
            "baseline_pv_rate": self.baseline_pv_rate,
            "baseline_tp_rate": self.baseline_tp_rate,
            "pv_delta": self.pv_delta,
            "tp_delta": self.tp_delta,
            "tolerance": self.tolerance,
            "regressed": self.regressed,
            "verdict": self.verdict,
        }


def load_arm_b_generation_baseline(path: str | Path) -> tuple[float, float]:
    """Read Arm B's frozen PV and TP rates.

    Args:
        path: Path to ``artifacts/baseline/frozen_baseline.json``.

    Returns:
        ``(pv_rate, tp_rate)``.

    Raises:
        ValueError: If the artifact carries no ``arm_b_tuned_sampling`` with
            both rates.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arm_b = payload.get(_ARM_B_KEY) or {}
    pv_rate, tp_rate = arm_b.get("pv_rate"), arm_b.get("tp_rate")
    if pv_rate is None or tp_rate is None:
        raise ValueError(
            f"{path} carries no {_ARM_B_KEY} pv_rate/tp_rate; the generation comparison "
            "point must come from the frozen artifact, not from a constant"
        )
    return float(pv_rate), float(tp_rate)


def requires_generation_check(switches: dict[str, bool]) -> bool:
    """Whether a configuration needs the generation regression check.

    Only a configuration that trains a second objective through the shared
    encoder can trade generation away, so this is the ``multitask`` switch.

    Args:
        switches: The configuration's five switches.

    Returns:
        ``True`` when generation must be re-measured for this configuration.
    """
    return bool(switches.get("multitask", False))


def check_generation_regression(
    report: GenerationReport,
    *,
    arm: str,
    baseline_pv_rate: float,
    baseline_tp_rate: float,
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> GenerationRegressionCheck:
    """Compare one arm's generation rates against Arm B's frozen rates.

    Args:
        report: The arm's generation evaluation, sampled at Arm B's
            temperature and top_p.
        arm: The configuration id, for the record.
        baseline_pv_rate: Arm B's PV rate from the frozen artifact.
        baseline_tp_rate: Arm B's TP rate from the frozen artifact.
        tolerance: How far below either rate is still acceptable.

    Returns:
        A :class:`GenerationRegressionCheck`.

    Raises:
        ValueError: If ``tolerance`` is not positive -- a zero tolerance would
            make sampling noise read as a regression.
    """
    if tolerance <= 0.0:
        raise ValueError(
            f"tolerance must be > 0, got {tolerance}: with no band, sampling noise reads "
            "as a regression and the check becomes uninformative"
        )
    pv_rate = report.counts.pv_rate
    tp_rate = report.target_property_rate
    pv_delta = pv_rate - baseline_pv_rate
    tp_delta = None if tp_rate is None else tp_rate - baseline_tp_rate

    regressed = pv_delta < -tolerance or (tp_delta is not None and tp_delta < -tolerance)
    if regressed:
        verdict = "regressed — a prediction gain bought here is a trade, not a success"
    elif pv_delta > tolerance and (tp_delta is None or tp_delta > -tolerance):
        verdict = "improved"
    else:
        verdict = "no regression"

    return GenerationRegressionCheck(
        arm=arm,
        pv_rate=pv_rate,
        tp_rate=tp_rate,
        baseline_pv_rate=baseline_pv_rate,
        baseline_tp_rate=baseline_tp_rate,
        pv_delta=pv_delta,
        tp_delta=tp_delta,
        tolerance=tolerance,
        regressed=regressed,
        verdict=verdict,
    )
```

Add to `src/polyt5/evaluation/__init__.py`: `from .generation_regression import DEFAULT_REGRESSION_TOLERANCE, GenerationRegressionCheck, check_generation_regression, load_arm_b_generation_baseline, requires_generation_check` plus the five `__all__` entries.

- [ ] **Step 4: Write the CLI**

```python
# scripts/check_group_a_generation.py
"""Check one Group A arm's generation quality against the frozen Arm B rates.

Spec section 6: generation is evaluated separately and must not regress. Any
configuration touching the shared encoder is sampled at ARM B's sampling point
(temperature 0.7, top_p 0.95 -- a different point makes the comparison
meaningless) and scored with the same cascade the frozen numbers came from.

Usage:
    python scripts/check_group_a_generation.py \
        --checkpoint results/group_a/A5/split_0/checkpoints/best.pt \
        --arm A5 --n-samples 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.prepare import format_property_value  # noqa: E402
from polyt5.evaluation import evaluate_generation  # noqa: E402
from polyt5.evaluation.generation_regression import (  # noqa: E402
    check_generation_regression,
    load_arm_b_generation_baseline,
)
from polyt5.generation import GenerationConfig, generate  # noqa: E402
from polyt5.inference.regression_predictor import RegressionPropertyPredictor  # noqa: E402
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import get_logger, seed_everything, select_device  # noqa: E402

#: Arm B's tuned sampling point. A different point makes the row incomparable.
ARM_B_TEMPERATURE = 0.7
ARM_B_TOP_P = 0.95


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tokenizer", type=Path,
                        default=REPO_ROOT / "artifacts" / "tokenizer" / "polyt5_vocab.json")
    parser.add_argument("--frozen-baseline", type=Path,
                        default=REPO_ROOT / "artifacts" / "baseline" / "frozen_baseline.json")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--target-property", type=float, default=500.0)
    parser.add_argument("--tolerance", type=float, default=50.0,
                        help="TP acceptance half-window in Kelvin. [PAPER] 50.")
    parser.add_argument("--regression-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the check JSON (default: beside the checkpoint).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code; 2 marks a regression."""
    args = parse_args(argv)
    logger = get_logger("polyt5.check_group_a_generation")
    seed_everything(args.seed)
    device = select_device("auto")

    tokenizer = PolyT5Tokenizer.from_file(args.tokenizer)
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    heads = ((payload.get("config") or {}).get("group_a") or {}).get("heads") or {}
    backbone = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(payload["model_config"]))
    model = PolyT5MultiTask(backbone, MultiTaskConfig.from_dict(heads))
    model.load_state_dict(payload["model_state"])
    model = model.to(device)
    model.eval()

    prompt = format_property_value(args.target_property)
    generated: list[str] = []
    for start in range(0, args.n_samples, args.batch_size):
        size = min(args.batch_size, args.n_samples - start)
        encoded = tokenizer.batch_encode(
            [prompt] * size, add_eos=True, max_length=200, padding=True, truncation=True
        )
        with torch.no_grad():
            output = generate(
                model.backbone,
                torch.tensor(encoded["input_ids"], device=device),
                torch.tensor(encoded["attention_mask"], device=device),
                config=GenerationConfig(
                    max_length=200,
                    do_sample=True,
                    temperature=ARM_B_TEMPERATURE,
                    top_p=ARM_B_TOP_P,
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                    decoder_start_token_id=tokenizer.decoder_start_token_id,
                    seed=args.seed + start,
                ),
            )
        generated.extend(
            tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
        )
        logger.info("sampled %d/%d", len(generated), args.n_samples)

    predictor = (
        RegressionPropertyPredictor(model, tokenizer, device=device, property_name="Tg")
        if model.tg_head is not None
        else None
    )
    report = evaluate_generation(
        generated,
        target_property=args.target_property,
        tolerance=args.tolerance,
        property_predictor=predictor,
    )
    baseline_pv_rate, baseline_tp_rate = load_arm_b_generation_baseline(args.frozen_baseline)
    check = check_generation_regression(
        report, arm=args.arm, baseline_pv_rate=baseline_pv_rate,
        baseline_tp_rate=baseline_tp_rate, tolerance=args.regression_tolerance,
    )

    destination = args.out or args.checkpoint.parent.parent / "generation_check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"check": check.to_dict(), "generation": report.to_dict()},
                   indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("%s: PV %.4f (baseline %.4f), TP %s (baseline %.4f) -> %s",
                args.arm, check.pv_rate, check.baseline_pv_rate,
                "n/a" if check.tp_rate is None else f"{check.tp_rate:.4f}",
                check.baseline_tp_rate, check.verdict)
    logger.info("wrote %s", destination)
    return 2 if check.regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest tests/test_group_a_generation_check.py -o addopts="" -q`
Expected: PASS — 12 passed

- [ ] **Step 6: Confirm the CLI is wired without running it**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" scripts/check_group_a_generation.py --help`
Expected: the usage text listing `--checkpoint`, `--arm`, `--regression-tolerance`.

- [ ] **Step 7: Run the full suite and ruff, then commit**

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m pytest -o addopts="" -q`
Expected: `1233 passed, 1 skipped, 1 xfailed`

Run: `"C:/Users/sumedh/.venvs/polyt5-rlvr/Scripts/python.exe" -m ruff check .`
Expected: `All checks passed!`

```bash
git add src/polyt5/evaluation/generation_regression.py src/polyt5/evaluation/__init__.py scripts/check_group_a_generation.py tests/test_group_a_generation_check.py
git commit -m "feat(group-a): generation regression check against the frozen Arm B rates"
```

---

## Running the ablation (NOT part of this plan)

Every task above is code plus tests. Executing the ablation is a separate, deliberate act, and these are its preconditions:

1. **No GRPO run is in flight.** `results/grpo_control/` must be finished. Group A and GRPO contend for the same GPU, and spec §7 is explicit that no Group A configuration enters a reward path until a full round is rerun on it deliberately.
2. The pretrained checkpoint path is passed with `--init-checkpoint`; `results/tg_prediction_5splits_medium92m/splits.json` is present and unmodified.
3. Budget: six new configurations × five splits × ~6.7 min ≈ 3.5 hours (spec §5), plus the generation check for A5 and A6.

```bash
python scripts/run_group_a.py --init-checkpoint <pretrained.pt> --out results/group_a
python scripts/check_group_a_generation.py --checkpoint results/group_a/A5/split_0/checkpoints/best.pt --arm A5
python scripts/check_group_a_generation.py --checkpoint results/group_a/A6/split_0/checkpoints/best.pt --arm A6
```

A5 and A6 are the arms that need the generation check, and that list is not hand-maintained:
`polyt5.evaluation.requires_generation_check(config.switches())` (Task 14) returns `True` for
exactly the configurations whose `multitask` switch is on, i.e. the ones giving the shared
encoder a second objective.

### Deliberately not scheduled here

Spec §8 asks that **λ's sensitivity be reported** and that N be measured rather than assumed.
This plan makes both configurable (`--set group_a.descriptor_lambda=…`,
`--set group_a.n_writings=…`) and reports the value used in every run manifest, but it does
**not** schedule a sweep: spec §5's budget is seven configurations × five splits ≈ 3.5 hours,
and a λ or N sweep is additional runs on top of that. A2's and A3's results tell you whether
the mechanism helps at the default setting at all; a sweep is only worth paying for if it does.
That follow-up needs its own budget decision and is out of this plan's scope.

When results exist, they are written up as a **fourth** claim category — our extension's improvements to our own reproduction — and never as closing a gap to the paper. Until then, **no Group A configuration has been trained, and nothing here is a result.**
