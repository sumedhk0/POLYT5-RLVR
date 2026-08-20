"""polyT5 tokenization: a predefined 458-token vocabulary and a longest-match tokenizer.

Deliberately torch-free and sentencepiece-free so the RLVR reward layer can import it
standalone. See :mod:`polyt5.tokenization.vocab` for the paper grounding and the id-order
contract, and :mod:`polyt5.tokenization.tokenizer` for the segmentation rules.
"""

from polyt5.tokenization.build_vocab import (
    REPRODUCTION_NOTE,
    build_default_vocab,
    build_vocab_from_corpus,
    split_pselfies,
)
from polyt5.tokenization.tokenizer import (
    ARTIFACT_VERSION,
    PolyT5Tokenizer,
    TokenizerArtifact,
)
from polyt5.tokenization.vocab import (
    BASE_SELFIES_TARGET,
    CONDITIONING_TARGET,
    CONDITIONING_TOKENS,
    SENTINEL_COUNT,
    SPECIAL_TOKENS,
    build_base_alphabet,
    build_conditioning_tokens,
    conditioning_groups,
    default_selfies_alphabet,
    sentinel_tokens,
)

__all__ = [
    "ARTIFACT_VERSION",
    "BASE_SELFIES_TARGET",
    "CONDITIONING_TARGET",
    "CONDITIONING_TOKENS",
    "REPRODUCTION_NOTE",
    "SENTINEL_COUNT",
    "SPECIAL_TOKENS",
    "PolyT5Tokenizer",
    "TokenizerArtifact",
    "build_base_alphabet",
    "build_conditioning_tokens",
    "build_default_vocab",
    "build_vocab_from_corpus",
    "conditioning_groups",
    "default_selfies_alphabet",
    "sentinel_tokens",
    "split_pselfies",
]
