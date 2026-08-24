"""Inference layer: the torch-side predictor that closes the evaluation loop.

:mod:`polyt5.evaluation` deliberately takes its property predictor as an
injected ``Callable[[Sequence[str]], Sequence[float]]`` so that it -- and the
future RL reward worker -- never imports torch. This package supplies that
callable, and the dependency arrow runs one way only::

    polyt5.inference  ---->  polyt5.evaluation
    polyt5.evaluation  -/->  polyt5.inference

============================  ==============================================
Object                        Responsibility
============================  ==============================================
:class:`PolyT5PropertyPredictor`  Load a fine-tuned property checkpoint and
                              score PSELFIES with the paper's beam-search
                              width-4 decode. Its ``__call__`` is the
                              injection point.
:class:`EnsemblePropertyPredictor`  Average several members and report their
                              disagreement, the RL phase's uncertainty signal.
:func:`conditioning_report`   Score each candidate against ITS OWN
                              conditioning target -- the metric the RLVR
                              reward optimises, and the one a single fixed
                              window cannot express.
:class:`CachedPredictor`      Memoise any predictor, since the RL loop
                              re-scores the same candidates repeatedly.
:class:`PredictionResult`     One prediction, with the decode that produced it.
============================  ==============================================

A decode that is not a number is a recorded failure, never an exception and
never a zero: :attr:`PredictionResult.value` is ``None`` and the ``float``-typed
``__call__`` returns :data:`NON_NUMERIC_VALUE` (NaN), which
:func:`polyt5.evaluation.target_property_rate` drops from both the numerator and
the denominator of the TP metric.

Example:
    >>> from polyt5.inference import PolyT5PropertyPredictor   # doctest: +SKIP
    >>> predictor = PolyT5PropertyPredictor.from_checkpoint(   # doctest: +SKIP
    ...     "results/finetune_tg_prediction/checkpoints/best.pt",
    ...     tokenizer_path="artifacts/tokenizer/polyt5_vocab.json",
    ... )
    >>> report = evaluate_generation(                          # doctest: +SKIP
    ...     candidates, target_property=500.0, tolerance=50.0,
    ...     property_predictor=predictor,
    ... )
"""

from __future__ import annotations

from polyt5.inference.predictor import (
    DEFAULT_MAX_SOURCE_LENGTH,
    DEFAULT_TOLERANCES,
    NON_NUMERIC_VALUE,
    CachedPredictor,
    CacheInfo,
    ConditioningReport,
    EnsemblePropertyPredictor,
    PolyT5PropertyPredictor,
    PredictionResult,
    conditioning_report,
    looks_like_pselfies,
)
from polyt5.inference.regression_predictor import GROUP_A_CONFIG_KEY, RegressionPropertyPredictor

__all__ = [
    "DEFAULT_MAX_SOURCE_LENGTH",
    "DEFAULT_TOLERANCES",
    "GROUP_A_CONFIG_KEY",
    "NON_NUMERIC_VALUE",
    "CacheInfo",
    "CachedPredictor",
    "ConditioningReport",
    "EnsemblePropertyPredictor",
    "PolyT5PropertyPredictor",
    "PredictionResult",
    "RegressionPropertyPredictor",
    "conditioning_report",
    "looks_like_pselfies",
]
