"""Declarative definition of the polyT5 vocabulary.

Paper grounding
---------------
polyT5 (Sahu et al., *npj Artificial Intelligence* 2026), Supplementary Information,
section "Tokenizer vocabulary", states verbatim:

    "For tokenizing the PSELFIES strings, each substring enclosed within square brackets
    (e.g., [C], [O]) was treated as a distinct token, resulting in a base vocabulary of
    199 unique tokens. Several special tokens were also introduced, including start- and
    end-of-sequence markers, unknown and padding tokens, a whitespace marker, and 100
    sentinel tokens for masking during pre-training. To further expand the vocabulary and
    enable property-conditional generation and prediction, an additional 154 tokens were
    incorporated. These included property names, numerical digits (0-9), decimal point (.),
    units, arithmetic and relational operators (+, -, >, <, =, etc.), boolean values, and a
    set of common polymer-related keywords. This resulted in a final vocabulary size of 458
    tokens. To ensure compatibility with the SentencePiece tokenizer framework, all SELFIES
    tokens and additional custom tokens were included as predefined tokens."

The arithmetic closes exactly::

    199 base SELFIES + 5 specials + 100 sentinels + 154 conditioning = 458

What is reproducible and what is not
------------------------------------
* Reproducible: the *structure* (four groups), the group sizes, the sentinel surface form
  ``<extra_id_{i}>`` (standard T5 naming, confirmed by paper Figure 2C), and the bracket
  tokenization rule.
* **Not reproducible**: the identity of the 199 base tokens and of the 154 conditioning
  tokens. The 199 were derived from a 100M-polymer corpus the authors state is "not
  publicly available due to IP protection", and the 154 are never enumerated anywhere in
  the paper or its SI. Everything below marked ``[SUBSTITUTE]`` is this reproduction's
  documented stand-in, not the authors' data.

ID order contract (stable across rebuilds -- downstream code depends on it)
--------------------------------------------------------------------------
=========  ===========================================================
ids        contents
=========  ===========================================================
0..4       specials, in order ``PAD, EOS, UNK, BOS, SPACE``
5..104     the 100 sentinels, ``<extra_id_0>`` first (``sentinel_id(i) == 5 + i``)
105..303   the base SELFIES alphabet, deterministically sorted
304..457   the conditioning tokens, in declaration order
=========  ===========================================================

Sentinels are laid out in *ascending* ``extra_id`` order because the span-corruption
pre-training objective consumes ``sentinel_id(0)`` first within a sequence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

import selfies

__all__ = [
    "BASE_SELFIES_TARGET",
    "CONDITIONING_TARGET",
    "CONDITIONING_TOKENS",
    "POLYMER_SEMANTIC_CONSTRAINTS",
    "SENTINEL_COUNT",
    "SPECIAL_TOKENS",
    "build_base_alphabet",
    "build_base_alphabet_detailed",
    "build_conditioning_tokens",
    "conditioning_groups",
    "default_selfies_alphabet",
    "sentinel_tokens",
]

# ---------------------------------------------------------------------------
# Group 1 -- special tokens (5, named by the paper)
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
SPACE_TOKEN = "▁"  # U+2581 LOWER ONE EIGHTH BLOCK, the SentencePiece whitespace marker

#: The five specials the paper names, in the fixed id order 0..4.
#:
#: Surface-form choices:
#:
#: * ``<pad>``, ``</s>``, ``<unk>`` -- the T5/SentencePiece defaults, so checkpoints and
#:   generation utilities written against T5 conventions behave unchanged.
#: * ``<s>`` -- [AMBIGUITY] the paper explicitly claims a "start-of-sequence marker", but
#:   stock T5 has **no** BOS token and instead uses ``<pad>`` as ``decoder_start_token_id``.
#:   We include ``<s>`` so the group count is 5 and the 458 arithmetic closes, while
#:   :attr:`PolyT5Tokenizer.decoder_start_token_id` still defaults to the PAD id (T5
#:   convention). The paper never states which id its start marker occupies.
#: * ``▁`` -- [AMBIGUITY] the paper says "a whitespace marker" without giving a surface
#:   form; U+2581 is the SentencePiece convention and the paper explicitly targets
#:   "compatibility with the SentencePiece tokenizer framework".
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, BOS_TOKEN, SPACE_TOKEN)

# ---------------------------------------------------------------------------
# Group 2 -- sentinels (100, named and counted by the paper)
# ---------------------------------------------------------------------------

SENTINEL_COUNT = 100


def sentinel_tokens(count: int = SENTINEL_COUNT) -> list[str]:
    """Return the sentinel surface forms in ascending ``extra_id`` order.

    Args:
        count: Number of sentinels to emit.

    Returns:
        ``["<extra_id_0>", ..., "<extra_id_{count-1}>"]``.
    """
    return [f"<extra_id_{i}>" for i in range(count)]


# ---------------------------------------------------------------------------
# Group 3 -- base SELFIES alphabet (199, size given by the paper, identities not)
# ---------------------------------------------------------------------------

BASE_SELFIES_TARGET = 199

#: The polymer terminus marker. polyT5 rewrites each ``[*]`` attachment point of a PSMILES
#: repeat unit to astatine before SELFIES encoding, so ``[At]`` must always be present.
TERMINUS_TOKEN = "[At]"

# [SUBSTITUTE] The authors' 199 base tokens came from a private 100M-polymer corpus. We
# instead derive a deterministic alphabet from `selfies` itself, widening its default
# semantic-constraint table to the heteroatoms and metals that actually occur in polymer
# repeat units (silicones, polyphosphazenes, selenopolymers, organotin/organoboron
# chains, metallopolymers and coordination polymers). `selfies.get_semantic_robust_alphabet`
# then enumerates, for every entry, the bond-order variants its valence permits -- which is
# exactly the closure a corpus-derived alphabet would have converged to.
#
# Charge states are limited to {-1, 0, +1} for main-group elements because SELFIES emits one
# token per (bond order, element, charge) triple and higher charges do not appear in neutral
# organic polymer repeat units.
POLYMER_SEMANTIC_CONSTRAINTS: Mapping[str, int] = {
    # hydrogen, halogens, and the [At] polymer terminus marker
    "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1, "At": 1,
    # group 13 -- polyborazines, organoboron and organoaluminium chains
    "B": 3, "B+1": 2, "B-1": 4,
    "Al": 3, "Al+1": 2, "Al-1": 4,
    "Ga": 3, "In": 3,
    # group 14 -- carbon backbones, polysilanes/siloxanes, organogermanium/tin
    "C": 4, "C+1": 3, "C-1": 3,
    "Si": 4, "Si+1": 3, "Si-1": 3,
    "Ge": 4, "Ge+1": 3, "Ge-1": 3,
    "Sn": 4, "Sn+1": 3, "Sn-1": 3,
    "Pb": 4,
    # group 15 -- polyamides/imides/ureas, polyphosphazenes, phosphate esters
    "N": 3, "N+1": 4, "N-1": 2,
    "P": 5, "P+1": 4, "P-1": 6,
    "As": 3, "As+1": 4, "As-1": 2,
    "Sb": 3, "Sb+1": 4, "Sb-1": 2,
    "Bi": 3,
    # group 16 -- polyethers/esters, polysulfones/sulfides, seleno- and telluropolymers
    "O": 2, "O+1": 3, "O-1": 1,
    "S": 6, "S+1": 5, "S-1": 5,
    "Se": 6, "Se+1": 5, "Se-1": 1,
    "Te": 6, "Te+1": 5, "Te-1": 1,
    # counter-ions and metal centres in ionomers, metallopolymers, coordination polymers
    "Li": 1, "Na": 1, "K": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Cu": 2,
    "Fe": 6, "Co": 6, "Ni": 4, "Mn": 6, "Cr": 6,
    "Ti": 4, "Zr": 4, "Pd": 4, "Pt": 4, "Ru": 6, "Rh": 6, "Ag": 1, "Au": 3,
    # selfies' wildcard valence for anything outside the table above
    "?": 8,
}  # fmt: skip

# [SUBSTITUTE] `selfies.get_semantic_robust_alphabet()` omits `[#Ring{n}]` because its
# encoder never emits a triple ring-closure bond, but its *decoder* accepts them. Including
# them keeps the tokenizer total on any grammar-valid SELFIES string, which matters for
# RLVR rollouts that score model-generated (not encoder-generated) text.
EXTRA_GRAMMAR_TOKENS: tuple[str, ...] = ("[#Ring1]", "[#Ring2]", "[#Ring3]")


def default_selfies_alphabet() -> list[str]:
    """Build the deterministic default base SELFIES alphabet.

    Widens the global ``selfies`` semantic-constraint table to
    :data:`POLYMER_SEMANTIC_CONSTRAINTS`, snapshots the robust alphabet it induces, then
    restores the previous global constraints so that callers of :func:`selfies.encoder`
    elsewhere in the process are unaffected.

    The result already contains every branch/ring control symbol selfies v2 emits
    (``[Branch1..3]``, ``[=Branch1..3]``, ``[#Branch1..3]``, ``[Ring1..3]``,
    ``[=Ring1..3]``) and every index symbol used as a branch/ring *length* encoding
    (``selfies.grammar_rules.INDEX_ALPHABET`` -- ``[C] [Ring1] [Ring2] [Branch1] [=Branch1]
    [#Branch1] [Branch2] [=Branch2] [#Branch2] [O] [N] [=N] [=C] [#C] [S] [P]``), because
    every one of those is a member of the robust alphabet. Both facts are asserted below
    rather than hard-coded, so a `selfies` upgrade that changes them fails loudly.

    Returns:
        Sorted, de-duplicated list of bracketed SELFIES tokens.
    """
    previous = selfies.get_semantic_constraints()
    try:
        selfies.set_semantic_constraints(dict(POLYMER_SEMANTIC_CONSTRAINTS))
        alphabet = set(selfies.get_semantic_robust_alphabet())
    finally:
        selfies.set_semantic_constraints(previous)

    alphabet.update(EXTRA_GRAMMAR_TOKENS)
    alphabet.add(TERMINUS_TOKEN)

    # Guard rails: the branch/ring control symbols and the index alphabet must survive.
    from selfies.grammar_rules import INDEX_ALPHABET

    required = set(INDEX_ALPHABET) | {
        f"{bond}{kind}{n}]"
        for bond in ("[", "[=", "[#")
        for kind, ns in (("Branch", (1, 2, 3)), ("Ring", (1, 2, 3)))
        for n in ns
    }
    missing = sorted(required - alphabet)
    if missing:  # pragma: no cover - only reachable on a breaking selfies upgrade
        raise RuntimeError(f"selfies control symbols missing from default alphabet: {missing}")
    if TERMINUS_TOKEN not in alphabet:  # pragma: no cover - defensive
        raise RuntimeError("the [At] polymer terminus marker must be in the base alphabet")

    return sorted(alphabet)


def build_base_alphabet_detailed(
    corpus_tokens: Iterable[str] | None = None,
    target_size: int = BASE_SELFIES_TARGET,
) -> tuple[list[str], dict[str, object]]:
    """Build the base alphabet at an exact size and report how it was adjusted.

    Selection rule:

    * With a corpus, tokens are ranked by ``(frequency desc, token asc)``; the second key
      makes ties deterministic across runs and platforms.
    * Without a corpus, :func:`default_selfies_alphabet` is used (already sorted).

    The size is then forced to ``target_size``. It is **never** silently changed:

    * short -> pad with reserved placeholders ``<unused_0>``, ``<unused_1>``, ...
    * long  -> keep the first ``target_size`` by the ranking above and record the drop count

    Args:
        corpus_tokens: Optional iterable of already-split SELFIES tokens.
        target_size: Exact number of base tokens to return.

    Returns:
        ``(alphabet, stats)`` where ``stats`` records source, pre-adjustment size,
        ``padded`` and ``dropped`` counts, and the dropped tokens (capped for readability).

    Raises:
        ValueError: If ``target_size`` is negative.
    """
    if target_size < 0:
        raise ValueError(f"target_size must be non-negative, got {target_size}")

    if corpus_tokens is None:
        ranked = default_selfies_alphabet()
        source = "selfies_robust_alphabet"
    else:
        counts = Counter(corpus_tokens)
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        source = "corpus"

    natural_size = len(ranked)
    kept = ranked[:target_size]
    dropped = ranked[target_size:]
    padded = [f"<unused_{k}>" for k in range(target_size - len(kept))]

    stats: dict[str, object] = {
        "source": source,
        "natural_size": natural_size,
        "target_size": target_size,
        "padded": len(padded),
        "dropped": len(dropped),
        "dropped_tokens": dropped[:50],
        "padding_prefix": "<unused_",
    }
    return kept + padded, stats


def build_base_alphabet(
    corpus_tokens: Iterable[str] | None = None,
    target_size: int = BASE_SELFIES_TARGET,
) -> list[str]:
    """Build the base SELFIES alphabet at exactly ``target_size`` tokens.

    Thin wrapper over :func:`build_base_alphabet_detailed` for callers that do not need
    the adjustment statistics.

    Args:
        corpus_tokens: Optional iterable of already-split SELFIES tokens.
        target_size: Exact number of base tokens to return.

    Returns:
        List of exactly ``target_size`` tokens.
    """
    return build_base_alphabet_detailed(corpus_tokens, target_size)[0]


# ---------------------------------------------------------------------------
# Group 4 -- conditioning tokens (154, size given by the paper, identities not)
# ---------------------------------------------------------------------------

CONDITIONING_TARGET = 154

# [SUBSTITUTE] The paper never enumerates its 154 conditioning tokens; it only lists the
# *categories* ("property names, numerical digits (0-9), decimal point (.), units,
# arithmetic and relational operators (+, -, >, <, =, etc.), boolean values, and a set of
# common polymer-related keywords"). The groups below follow those categories in that
# order, seeded from the field labels and class words that actually appear in the SI's I/O
# examples and from the six properties the paper models.

#: Field labels and property names. ``property``/``polymer``/``solvent`` are verbatim from
#: the SI prompt formats; the rest name the six modelled properties and their aliases.
_FIELD_LABELS: tuple[str, ...] = (
    "property", "polymer", "solvent",
    "Tg", "Tm", "Td", "Eg", "dielectric_constant", "solubility",
    "bandgap", "glass_transition", "melting", "decomposition",
    "frequency", "log_frequency", "target",
)  # fmt: skip

#: Digits and the decimal point -- explicitly named by the paper.
_NUMERIC: tuple[str, ...] = tuple("0123456789") + (".",)

# [AMBIGUITY] The paper says "units" but names none. These cover the units the six modelled
# properties are reported in (K for Tg/Tm/Td, eV for Eg, Hz/log10(Hz) for dielectric
# frequency) plus common polymer-property units.
_UNITS: tuple[str, ...] = (
    "K", "eV", "Hz", "log10(Hz)", "C", "g/mol", "GPa", "MPa",
    "%", "cm3", "kJ/mol", "S/cm", "unitless",
)  # fmt: skip

#: Arithmetic and relational operators. The paper names ``+ - > < =`` and abbreviates the
#: rest as "etc."; the multi-character forms are listed so longest-match prefers ``>=``
#: over ``>``.
_OPERATORS: tuple[str, ...] = (
    "+", "-", "*", "/", ">", "<", ">=", "<=", "=", "==", "!=", "~",
)  # fmt: skip

#: Boolean values (named by the paper) plus the classification labels the SI's solubility
#: task emits verbatim (``soluble`` / ``insoluble``).
_BOOLEANS: tuple[str, ...] = (
    "true", "false", "yes", "no", "soluble", "insoluble", "valid", "invalid",
)  # fmt: skip

# [AMBIGUITY] Separators are not called out as a category by the paper, but ``;`` appears in
# every multi-field SI prompt, so the set must contain at least that.
_PUNCTUATION: tuple[str, ...] = (";", ":", ",", "|", "(", ")")

# [SUBSTITUTE] "common polymer-related keywords" -- chemistry families, reaction/click
# handles, topology terms, functional groups, heteroatom names, and morphology/processing
# descriptors drawn from the paper's own tables. ``bandgap_kw`` and ``soluble_kw`` carry a
# suffix purely to stay distinct from the identically spelled property name and class label
# above; the vocabulary must not contain duplicates.
_POLYMER_KEYWORDS: tuple[str, ...] = (
    # chemistry families
    "polyamide", "polyimide", "polyester", "polyether", "polyurea",
    "polyurethane", "polycarbonate", "polyolefin",
    # polymerisation / click chemistry handles
    "ROMP", "CuAAC", "SPAAC", "thiol-ene", "thiol-yne", "Diels-Alder",
    "SuFEx", "oxime", "click",
    # topology
    "homopolymer", "copolymer", "monomer", "repeat_unit", "backbone", "sidechain",
    "aromatic", "aliphatic", "ring", "branch", "crosslink",
    # functional groups
    "ether", "amide", "ester", "amine", "imide", "urea", "urethane", "thioether",
    "sulfoxide", "hydroxyl", "nitrile", "acetal", "allyl", "alkyne", "acrylate",
    "methacrylate", "epoxide", "anhydride", "phosphate", "hydrazide",
    "carboxylic_acid", "amidine",
    # heteroatom vocabulary
    "halogen", "sulfur", "nitrogen", "oxygen", "silicon", "phosphorus", "unsaturation",
    # morphology and processing descriptors
    "crystalline", "amorphous", "semicrystalline", "thermoplastic", "thermoset",
    "elastomer", "dielectric", "bandgap_kw", "soluble_kw", "processability", "stability",
)  # fmt: skip


def conditioning_groups() -> dict[str, tuple[str, ...]]:
    """Return the conditioning token groups, in the order they are concatenated.

    Returns:
        Mapping from group name to its ordered token tuple. Excludes the reserved
        placeholder padding, which is appended by :func:`build_conditioning_tokens`.
    """
    return {
        "field_labels": _FIELD_LABELS,
        "numeric": _NUMERIC,
        "units": _UNITS,
        "operators": _OPERATORS,
        "booleans": _BOOLEANS,
        "punctuation": _PUNCTUATION,
        "polymer_keywords": _POLYMER_KEYWORDS,
    }


def build_conditioning_tokens(
    target_size: int = CONDITIONING_TARGET,
) -> tuple[list[str], dict[str, object]]:
    """Assemble the conditioning block at exactly ``target_size`` tokens.

    Groups are concatenated in :func:`conditioning_groups` order, then padded with
    ``<cond_reserved_k>`` placeholders (or trimmed from the tail) to land on exactly
    ``target_size``. The arithmetic is done in code and asserted -- never hand-counted.

    Args:
        target_size: Exact number of conditioning tokens to return.

    Returns:
        ``(tokens, stats)`` where ``stats`` records the per-group counts and the
        padding/trim counts.

    Raises:
        ValueError: If ``target_size`` is negative or a duplicate token is declared.
    """
    if target_size < 0:
        raise ValueError(f"target_size must be non-negative, got {target_size}")

    groups = conditioning_groups()
    ordered: list[str] = []
    for tokens in groups.values():
        ordered.extend(tokens)

    duplicates = sorted({t for t in ordered if ordered.count(t) > 1})
    if duplicates:
        raise ValueError(f"duplicate conditioning tokens declared: {duplicates}")

    natural_size = len(ordered)
    kept = ordered[:target_size]
    trimmed = ordered[target_size:]
    padded = [f"<cond_reserved_{k}>" for k in range(target_size - len(kept))]
    result = kept + padded

    assert len(result) == target_size, "conditioning block must land on the exact target"

    stats: dict[str, object] = {
        "group_sizes": {name: len(tokens) for name, tokens in groups.items()},
        "natural_size": natural_size,
        "target_size": target_size,
        "padded": len(padded),
        "trimmed": len(trimmed),
        "trimmed_tokens": trimmed[:50],
        "padding_prefix": "<cond_reserved_",
    }
    return result, stats


#: The 154 conditioning tokens, materialised at import time so the id layout is a constant.
CONDITIONING_TOKENS: tuple[str, ...] = tuple(build_conditioning_tokens()[0])
