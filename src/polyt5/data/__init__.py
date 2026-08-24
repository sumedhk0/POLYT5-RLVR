"""Data pipeline: corpus acquisition, PSELFIES preparation, splits, the polyT5
span-corruption objective, and batch collators.

Import-weight contract for this package
---------------------------------------
``import polyt5.data`` must NOT pull in torch. ``span_corruption``, ``sources``,
``prepare`` and ``splits`` are pure numpy/python, and ``collate`` imports torch
lazily inside its call paths. This matters because the future RLVR reward
workers (see ``docs/rlvr_plan.md``) run without a model in the process.

``polyt5.data.datasets`` is the one exception: it subclasses
``torch.utils.data.Dataset`` and therefore imports torch at module level. It is
deliberately **not** re-exported here — import it explicitly::

    from polyt5.data.datasets import PSelfiesCorpus, Seq2SeqDataset
"""

from __future__ import annotations

from polyt5.data.collate import (
    LABEL_IGNORE_ID,
    Seq2SeqCollator,
    SpanCorruptionCollator,
    pad_sequences,
)
from polyt5.data.prepare import (
    PreparationStats,
    build_generation_examples,
    build_prediction_examples,
    format_property_value,
    parse_property_value,
    prepare_labeled_corpus,
    prepare_pselfies_corpus,
    read_lamalab_tg,
    read_pi1m,
)
from polyt5.data.sources import (
    SOURCES,
    ConfirmationRequiredError,
    DataSource,
    DownloadRecord,
    download,
)
from polyt5.data.span_corruption import (
    DEFAULT_CONFIG,
    SpanCorruptionConfig,
    SpanCorruptionResult,
    batch_span_corrupt,
    corruption_statistics,
    span_corrupt,
)
from polyt5.data.splits import (
    load_splits,
    make_kfold_random_splits,
    make_pretraining_splits,
    random_split,
    save_splits,
)
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

__all__ = [
    "DEFAULT_CONFIG",
    "LABEL_IGNORE_ID",
    "SOURCES",
    "ConfirmationRequiredError",
    "DataSource",
    "DownloadRecord",
    "DESCRIPTOR_PREFIXES",
    "PreparationStats",
    "Seq2SeqCollator",
    "SpanCorruptionCollator",
    "SpanCorruptionConfig",
    "SpanCorruptionResult",
    "TgExample",
    "TgRow",
    "batch_span_corrupt",
    "build_generation_examples",
    "build_prediction_examples",
    "corruption_statistics",
    "descriptor_columns",
    "descriptor_group",
    "descriptor_matrix",
    "download",
    "format_property_value",
    "load_splits",
    "make_kfold_random_splits",
    "make_pretraining_splits",
    "pad_sequences",
    "parse_property_value",
    "prepare_labeled_corpus",
    "prepare_labeled_rows",
    "prepare_pselfies_corpus",
    "random_split",
    "read_lamalab_rows",
    "read_lamalab_tg",
    "read_pi1m",
    "save_splits",
    "span_corrupt",
]
