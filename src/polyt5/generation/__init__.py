"""Decoding for polyT5 -- sampling (conditional generation) and beam search
(property prediction).

The paper (Sahu et al., npj Artificial Intelligence 2026) uses two DIFFERENT
decoding regimes and conflating them is a reproduction error:

* Property prediction -> :func:`beam_search`, beam width 4.
* Conditional generation -> :func:`generate` with ``do_sample=True``,
  explicitly "instead of beam search": ``top_p`` in {0.75, 0.95}, temperature
  0.1 -> 2.0 in steps of 0.1, max output length 200 tokens.

Everything operates on integer token ids and tensors; nothing here imports the
tokenizer or the data pipeline.
"""

from polyt5.generation.beam_search import BeamSearchConfig, beam_search, length_penalized_score
from polyt5.generation.generate import (
    GenerationConfig,
    GenerationOutput,
    batch_generate,
    generate,
)
from polyt5.generation.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "BeamSearchConfig",
    "GenerationConfig",
    "GenerationOutput",
    "apply_repetition_penalty",
    "apply_temperature",
    "batch_generate",
    "beam_search",
    "generate",
    "length_penalized_score",
    "top_k_filter",
    "top_p_filter",
]
