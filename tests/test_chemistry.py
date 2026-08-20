"""Tests for the polyT5 chemistry layer.

The chemistry layer is deliberately free of any torch/model dependency: it is
reused by the RLVR reward system, which must be able to score adversarial model
output without ever raising.

Terminology used throughout (kept strictly distinct):
    PSMILES  -- SMILES for a polymer repeat unit, with exactly two chain-end
                termini written as ``[At]`` (or ``[*]``/``*`` in some sources).
    PSELFIES -- ``selfies.encoder()`` applied to an ``[At]``-capped PSMILES.
    SMILES   -- an ordinary small-molecule SMILES, no terminus semantics.
    SELFIES  -- ``selfies.encoder()`` applied to an ordinary SMILES.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The project is not pip-installed in this environment, so make the src/ layout
# importable when pytest is run straight from the repo root. This is a no-op
# once `pip install -e .` has been run.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyt5.chemistry import (  # noqa: E402
    STAR_TOKENS,
    TERMINUS,
    GenerationChemMetrics,
    NoveltyIndex,
    ValidityResult,
    at_to_star,
    canonical_pselfies,
    canonical_psmiles,
    cleave_and_cap,
    count_termini,
    cyclize_psmiles,
    evaluate_generations,
    is_same_polymer,
    novelty_rate,
    pselfies_to_psmiles,
    psmiles_to_pselfies,
    psmiles_to_pselfies_loop_break,
    selfies_reproducible,
    selfies_to_smiles,
    smiles_to_selfies,
    star_to_at,
    synthetic_accessibility,
    validate_pselfies,
    validate_psmiles,
)
from polyt5.chemistry.metrics import SA_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.chem


# Five real polymer repeat units, written as [At]-capped PSMILES.
POLYMERS = {
    "polyethylene": "[At]CC[At]",
    "poly(ethylene oxide)": "[At]CCO[At]",
    "PET-like polyester": "[At]OCCOC(=O)c1ccc(cc1)C(=O)[At]",
    "nylon-6-like polyamide": "[At]NCCCCCC(=O)[At]",
    "bisphenol-A-like aromatic": "[At]c1ccc(cc1)C(C)(C)c1ccc([At])cc1",
}

GARBAGE = [
    "",
    "   ",
    "not_a_molecule",
    "C(C",            # unbalanced parenthesis
    "[At]CCO[At",     # unbalanced bracket
    "(((",
    "[Qq][Zz]",       # bogus bracket tokens
]


# --------------------------------------------------------------------------
# import hygiene
# --------------------------------------------------------------------------


def test_chemistry_layer_does_not_import_torch():
    """The chemistry layer must be importable with zero torch dependency.

    Checked in a SUBPROCESS on purpose: pytest imports every test module at
    collection time, and other test modules import torch at module level, so an
    in-process ``sys.modules`` assertion would be testing the test runner rather
    than the chemistry package. A future RL reward worker must be able to import
    this layer without torch present at all.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import polyt5.chemistry; "
            "sys.exit(1 if 'torch' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"polyt5.chemistry pulled in torch: rc={result.returncode} stderr={result.stderr[:500]}"
    )


def test_module_constants():
    assert TERMINUS == "[At]"
    assert STAR_TOKENS == ("[*]", "*")


# --------------------------------------------------------------------------
# conversion: star <-> At
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[*]CCO[*]", "[At]CCO[At]"),
        ("*CCO*", "[At]CCO[At]"),
        ("[*]CC[*]", "[At]CC[At]"),
        ("*CC*", "[At]CC[At]"),
        ("[*]C(C)(C)C[*]", "[At]C(C)(C)C[At]"),
        ("*C(*)CO*", "[At]C([At])CO[At]"),
        ("[At]CCO[At]", "[At]CCO[At]"),  # already [At]-style: idempotent
        ("CCO", "CCO"),                  # no termini at all: unchanged
    ],
)
def test_star_to_at(raw, expected):
    assert star_to_at(raw) == expected


@pytest.mark.parametrize(
    ("at_form", "expected"),
    [
        ("[At]CCO[At]", "[*]CCO[*]"),
        ("[At]CC[At]", "[*]CC[*]"),
        ("[At]C([At])CO[At]", "[*]C([*])CO[*]"),
        ("CCO", "CCO"),
    ],
)
def test_at_to_star(at_form, expected):
    assert at_to_star(at_form) == expected


@pytest.mark.parametrize("psmiles", list(POLYMERS.values()))
def test_star_at_round_trip(psmiles):
    """[At] -> [*] -> [At] must be the identity."""
    assert star_to_at(at_to_star(psmiles)) == psmiles


def test_star_conversions_never_raise_on_non_string():
    assert star_to_at(None) == ""  # type: ignore[arg-type]
    assert at_to_star(12345) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# conversion: PSMILES <-> PSELFIES
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "psmiles"), sorted(POLYMERS.items()))
def test_psmiles_pselfies_round_trip(name, psmiles):
    """Encode then decode must recover the same polymer (canonically)."""
    pselfies = psmiles_to_pselfies(psmiles)
    assert pselfies is not None, name
    assert pselfies.startswith("[")

    back = pselfies_to_psmiles(pselfies)
    assert back is not None, name
    assert count_termini(back) == 2, name
    assert is_same_polymer(psmiles, back), name


def test_psmiles_to_pselfies_known_value():
    """Anchor one exact string so encoder behaviour changes are visible."""
    assert psmiles_to_pselfies("[At]CCO[At]") == "[At][C][C][O][At]"
    assert pselfies_to_psmiles("[At][C][C][O][At]") == "[At]CCO[At]"


def test_psmiles_to_pselfies_accepts_star_notation():
    """`*` is not encodable by selfies, so star input must be normalised first."""
    assert psmiles_to_pselfies("[*]CCO[*]") == "[At][C][C][O][At]"
    assert psmiles_to_pselfies("*CCO*") == "[At][C][C][O][At]"


@pytest.mark.parametrize("bad", GARBAGE)
def test_psmiles_to_pselfies_returns_none_on_garbage(bad):
    assert psmiles_to_pselfies(bad) is None


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "garbage", "[Bogus]", "[C][Nope]", "[C][C][NotAToken][C]", "((("],
)
def test_pselfies_to_psmiles_returns_none_on_garbage(bad):
    assert pselfies_to_psmiles(bad) is None


# --------------------------------------------------------------------------
# conversion: ordinary molecules (no terminus logic)
# --------------------------------------------------------------------------


def test_smiles_selfies_round_trip():
    selfies_str = smiles_to_selfies("CCO")
    assert selfies_str == "[C][C][O]"
    assert selfies_to_smiles(selfies_str) == "CCO"


def test_smiles_to_selfies_does_not_touch_termini():
    """Ordinary-molecule helpers must not silently rewrite `*` into [At]."""
    assert smiles_to_selfies("*CCO*") is None


@pytest.mark.parametrize("bad", GARBAGE)
def test_ordinary_conversions_return_none_on_garbage(bad):
    assert smiles_to_selfies(bad) is None
    assert selfies_to_smiles(bad) is None


# --------------------------------------------------------------------------
# conversion: count_termini
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("psmiles", "n"),
    [
        ("CCO", 0),
        ("c1ccccc1", 0),
        ("[At]CCO", 1),
        ("[At]CCO[At]", 2),
        ("[At]CC[At]", 2),
        ("[At]C([At])CO[At]", 3),
        ("[*]CCO[*]", 2),   # star notation is normalised before counting
        ("*CCO*", 2),
        ("garbage", 0),     # unparseable -> 0
        ("", 0),
        ("C(C", 0),
        # Unparseable strings that *contain* the literal "[At]": counting must
        # go through RDKit, so these are 0 rather than a naive substring count.
        ("[At]CCO[At", 0),
        ("[At]C(C", 0),
        ("[At]CCO[At]]", 0),
    ],
)
def test_count_termini(psmiles, n):
    assert count_termini(psmiles) == n


# --------------------------------------------------------------------------
# canonicalization and dedup
# --------------------------------------------------------------------------


def test_canonical_psmiles_preserves_termini():
    canon = canonical_psmiles("[At]CCO[At]")
    assert canon is not None
    assert canon.count("[At]") == 2
    assert count_termini(canon) == 2


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("[At]CCO[At]", "[At]OCC[At]"),                     # written from either end
        ("[At]c1ccccc1[At]", "[At]C1=CC=CC=C1[At]"),        # aromatic vs kekulized
        ("[At]CCO[At]", "[*]OCC[*]"),                       # star vs At notation
        ("[At]C(C)(C)C[At]", "[At]CC(C)(C)[At]"),
    ],
)
def test_canonical_dedup_collapses_equivalent_writings(a, b):
    ca, cb = canonical_psmiles(a), canonical_psmiles(b)
    assert ca is not None and cb is not None
    assert ca == cb
    assert is_same_polymer(a, b)
    assert len({ca, cb}) == 1


def test_canonical_dedup_keeps_different_polymers_apart():
    assert canonical_psmiles("[At]CC[At]") != canonical_psmiles("[At]CCO[At]")
    assert not is_same_polymer("[At]CC[At]", "[At]CCO[At]")


def test_canonical_pselfies_collapses_equivalent_writings():
    a = psmiles_to_pselfies("[At]CCO[At]")
    b = psmiles_to_pselfies("[At]OCC[At]")
    assert a != b  # the raw encodings genuinely differ
    ca, cb = canonical_pselfies(a), canonical_pselfies(b)
    assert ca is not None and ca == cb
    assert is_same_polymer(a, b, kind="pselfies")


@pytest.mark.parametrize("bad", GARBAGE)
def test_canonicalization_returns_none_on_garbage(bad):
    assert canonical_psmiles(bad) is None
    assert canonical_pselfies(bad) is None


def test_is_same_polymer_false_for_garbage():
    assert not is_same_polymer("garbage", "garbage")
    assert not is_same_polymer("[At]CC[At]", "garbage")
    assert not is_same_polymer("", "")


def test_is_same_polymer_rejects_unknown_kind():
    with pytest.raises(ValueError):
        is_same_polymer("[At]CC[At]", "[At]CC[At]", kind="inchi")


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------


def test_validate_pselfies_happy_path():
    res = validate_pselfies("[At][C][C][O][At]")
    assert isinstance(res, ValidityResult)
    assert res.valid
    assert res.parseable
    assert res.n_termini == 2
    assert res.correct_termini
    assert res.canonical_psmiles == canonical_psmiles("[At]CCO[At]")
    assert res.reason is None


def test_validity_result_is_frozen():
    res = validate_pselfies("[At][C][C][O][At]")
    # frozen dataclasses raise FrozenInstanceError, which subclasses AttributeError
    with pytest.raises(AttributeError):
        res.valid = False  # type: ignore[misc]


def test_validate_pselfies_decode_failure():
    res = validate_pselfies("[Bogus][Token]")
    assert not res.parseable
    assert not res.valid
    assert res.n_termini == 0
    assert not res.correct_termini
    assert res.canonical_psmiles is None
    assert res.reason == "decode_failed"


def test_validate_pselfies_wrong_termini_count():
    """A chemically fine molecule that is not a polymer repeat unit."""
    res = validate_pselfies("[C][C][O]")
    assert res.parseable
    assert res.valid          # RDKit is happy: ethanol is a real molecule
    assert res.n_termini == 0
    assert not res.correct_termini
    assert res.reason == "wrong_termini_count"


@pytest.mark.parametrize(
    ("psmiles", "n_termini"),
    [("CCO", 0), ("[At]CCO", 1), ("[At]C([At])CO[At]", 3)],
)
def test_validate_psmiles_wrong_termini(psmiles, n_termini):
    res = validate_psmiles(psmiles)
    assert res.valid
    assert res.n_termini == n_termini
    assert not res.correct_termini
    assert res.reason == "wrong_termini_count"


def test_validate_psmiles_expected_termini_is_configurable():
    res = validate_psmiles("[At]CCO", expected_termini=1)
    assert res.correct_termini
    assert res.reason is None


def test_validate_psmiles_rdkit_failure():
    res = validate_psmiles("C(C")
    assert res.parseable      # non-empty string, but RDKit rejects it
    assert not res.valid
    assert res.canonical_psmiles is None
    assert res.reason == "rdkit_parse_failed"


def test_validate_psmiles_empty_input():
    res = validate_psmiles("")
    assert not res.parseable
    assert not res.valid
    assert res.reason == "empty_input"


@pytest.mark.parametrize("bad", GARBAGE)
def test_validation_never_raises(bad):
    for res in (validate_pselfies(bad), validate_psmiles(bad)):
        assert isinstance(res, ValidityResult)
        assert not res.valid
        assert res.reason is not None


# --------------------------------------------------------------------------
# SELFIES reproducibility (SR)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("psmiles", list(POLYMERS.values()))
def test_selfies_reproducible_true_for_encoder_output(psmiles):
    """Anything the encoder itself emits is by construction reproducible."""
    pselfies = psmiles_to_pselfies(psmiles)
    assert pselfies is not None
    assert selfies_reproducible(pselfies)


def test_selfies_reproducible_false_case():
    """A valid-but-non-canonical PSELFIES fails the strict (paper) SR test.

    ``[At][C][Branch1][Ring2][C][C][At]`` decodes to ``[At]CCC[At]`` because the
    branch-length token overruns the remaining symbols, so the branch collapses
    into the main chain. Re-encoding that PSMILES yields
    ``[At][C][C][C][At]``, which differs from the input string -- exactly the
    failure mode the paper's SR metric measures.
    """
    non_canonical = "[At][C][Branch1][Ring2][C][C][At]"
    assert pselfies_to_psmiles(non_canonical) == "[At]CCC[At]"
    assert psmiles_to_pselfies("[At]CCC[At]") == "[At][C][C][C][At]"

    # Strict (default, paper definition): fails, the string is not reproduced.
    assert not selfies_reproducible(non_canonical)

    # Lenient (semantic round trip): passes, the molecule is preserved.
    assert selfies_reproducible(non_canonical, strict=False)


@pytest.mark.parametrize("bad", GARBAGE)
def test_selfies_reproducible_false_and_silent_on_garbage(bad):
    assert selfies_reproducible(bad) is False
    assert selfies_reproducible(bad, strict=False) is False


# --------------------------------------------------------------------------
# novelty
# --------------------------------------------------------------------------


def test_novelty_index_membership():
    index = NoveltyIndex(["[At]CC[At]", "[At]CCO[At]"])
    assert len(index) == 2

    assert "[At]CC[At]" in index
    assert "[At]OCC[At]" in index          # equivalent writing of PEO
    assert "[*]CC[*]" in index             # star notation
    assert "[At]CCC[At]" not in index

    assert not index.is_novel("[At]CC[At]")
    assert index.is_novel("[At]CCC[At]")


def test_novelty_index_dedups_and_skips_garbage():
    index = NoveltyIndex(["[At]CC[At]", "[At]CC[At]", "garbage", "", "C(C"])
    assert len(index) == 1
    assert "garbage" not in index


def test_novelty_index_from_pselfies():
    index = NoveltyIndex.from_pselfies(["[At][C][C][At]", "[At][C][C][O][At]", "[Bogus]"])
    assert len(index) == 2
    assert "[At]CC[At]" in index
    assert "[At]CCO[At]" in index


def test_novelty_rate_arithmetic():
    index = NoveltyIndex(["[At]CC[At]", "[At]CCO[At]"])
    candidates = [
        "[At]CC[At]",     # known
        "[At]OCC[At]",    # known (equivalent writing)
        "[At]CCC[At]",    # novel
        "[At]CCCC[At]",   # novel
    ]
    assert novelty_rate(candidates, index) == pytest.approx(0.5)
    assert novelty_rate(["[At]CC[At]"], index) == pytest.approx(0.0)
    assert novelty_rate(["[At]CCCC[At]"], index) == pytest.approx(1.0)


def test_novelty_rate_empty_candidates_is_zero():
    assert novelty_rate([], NoveltyIndex(["[At]CC[At]"])) == 0.0


def test_novelty_rate_against_empty_index():
    assert novelty_rate(["[At]CC[At]"], NoveltyIndex([])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# aggregate metrics
# --------------------------------------------------------------------------


# Hand-built generation batch with known counts. Annotations:
#   1. PE               -> valid, 2 termini, SR ok
#   2. PEO              -> valid, 2 termini, SR ok
#   3. PEO, other end   -> valid, 2 termini, SR ok, duplicate of #2 after canon
#   4. PE again         -> exact duplicate of #1
#   5. ethanol          -> valid molecule but 0 termini (not a polymer)
#   6. bogus token      -> not parseable
#   7. empty string     -> not parseable
#   8. non-canonical PP -> valid, 2 termini, SR FAILS, unique
BATCH = [
    "[At][C][C][At]",
    "[At][C][C][O][At]",
    "[At][O][C][C][At]",
    "[At][C][C][At]",
    "[C][C][O]",
    "[Bogus]",
    "",
    "[At][C][Branch1][Ring2][C][C][At]",
]


@pytest.fixture()
def batch_metrics() -> GenerationChemMetrics:
    index = NoveltyIndex(["[At]CC[At]"])  # polyethylene is "known"
    return evaluate_generations(BATCH, novelty_index=index)


def test_evaluate_generations_counts(batch_metrics):
    m = batch_metrics
    assert m.n_generated == 8
    assert m.n_parseable == 6        # bogus token and empty string fail to decode
    assert m.n_valid == 6            # every decoded string is RDKit-valid
    assert m.n_correct_termini == 5  # ethanol has no termini
    assert m.n_reproducible == 5     # only the non-canonical branch string fails SR
    assert m.n_unique == 3           # {PE, PEO, PP} among the 5 polymer-valid entries
    assert m.n_novel == 2            # PE is in the reference index


def test_evaluate_generations_rates_exactly(batch_metrics):
    m = batch_metrics
    assert m.validity_rate == pytest.approx(6 / 8)
    assert m.uniqueness_rate == pytest.approx(3 / 5)
    assert m.duplicate_rate == pytest.approx(2 / 5)
    assert m.novelty_rate == pytest.approx(2 / 3)
    assert m.reproducibility_rate == pytest.approx(5 / 8)
    assert m.uniqueness_rate + m.duplicate_rate == pytest.approx(1.0)


def test_evaluate_generations_without_novelty_index():
    m = evaluate_generations(BATCH)
    assert m.n_novel == 0
    assert m.novelty_rate == 0.0
    # every other statistic is unaffected by the missing index
    assert m.n_unique == 3
    assert m.validity_rate == pytest.approx(6 / 8)


def test_evaluate_generations_empty_batch():
    m = evaluate_generations([])
    assert m.n_generated == 0
    for rate in (
        m.validity_rate,
        m.uniqueness_rate,
        m.novelty_rate,
        m.reproducibility_rate,
        m.duplicate_rate,
    ):
        assert rate == 0.0


def test_evaluate_generations_all_garbage_never_raises():
    m = evaluate_generations(GARBAGE)
    assert m.n_generated == len(GARBAGE)
    assert m.n_valid == 0
    assert m.n_unique == 0
    assert m.validity_rate == 0.0


def test_evaluate_generations_respects_expected_termini():
    m = evaluate_generations(["[At][C][C][O]"], expected_termini=1)
    assert m.n_correct_termini == 1
    assert m.n_unique == 1


def test_metrics_to_dict_is_json_serializable(batch_metrics):
    d = batch_metrics.to_dict()
    assert isinstance(d, dict)
    assert d["n_generated"] == 8
    assert d["validity_rate"] == pytest.approx(6 / 8)
    assert set(d) == {
        "n_generated",
        "n_parseable",
        "n_valid",
        "n_correct_termini",
        "n_reproducible",
        "n_unique",
        "n_novel",
        "validity_rate",
        "uniqueness_rate",
        "novelty_rate",
        "reproducibility_rate",
        "duplicate_rate",
    }
    json.dumps(d)  # must not raise


# --------------------------------------------------------------------------
# synthetic accessibility
# --------------------------------------------------------------------------


def test_sa_available_is_a_bool():
    assert isinstance(SA_AVAILABLE, bool)


@pytest.mark.skipif(not SA_AVAILABLE, reason="RDKit Contrib SA_Score is not importable")
def test_synthetic_accessibility_scores():
    """SA scores run 1 (easy) to 10 (hard); known polymers sit low (paper: 2-3)."""
    peo = synthetic_accessibility("[At]CCO[At]")
    assert peo is not None
    assert 1.0 <= peo <= 10.0

    # [At] capping must not change the score relative to the H-capped monomer.
    assert peo == pytest.approx(synthetic_accessibility("CCO"))
    assert peo == pytest.approx(synthetic_accessibility("[*]CCO[*]"))


@pytest.mark.skipif(not SA_AVAILABLE, reason="RDKit Contrib SA_Score is not importable")
@pytest.mark.parametrize(("name", "psmiles"), sorted(POLYMERS.items()))
def test_synthetic_accessibility_known_polymers_are_easy(name, psmiles):
    """The paper reports nearly all known polymers below 6, most in the 2-3 band.

    Anything above 6 is what the paper calls structurally complex, so the whole
    reference set landing under that threshold is the real behavioural check.
    """
    score = synthetic_accessibility(psmiles)
    assert score is not None, name
    assert 1.0 <= score <= 10.0, name
    assert score < 6.0, f"{name} scored {score}, above the paper's complexity threshold"


@pytest.mark.skipif(not SA_AVAILABLE, reason="RDKit Contrib SA_Score is not importable")
def test_synthetic_accessibility_ranks_complexity():
    """A gnarly polycyclic natural product must score harder than a simple chain."""
    simple = synthetic_accessibility("[At]CCO[At]")
    complex_ = synthetic_accessibility("C1CC2CCC3C(CCC4C3CCC3(C)C(C(=O)O)CCC43)C2(C)CC1")
    assert simple is not None and complex_ is not None
    assert complex_ > simple
    assert complex_ > 3.0


@pytest.mark.skipif(not SA_AVAILABLE, reason="RDKit Contrib SA_Score is not importable")
def test_synthetic_accessibility_is_deterministic():
    scores = {synthetic_accessibility("[At]CCO[At]") for _ in range(5)}
    assert len(scores) == 1


@pytest.mark.skipif(SA_AVAILABLE, reason="SA_Score is importable in this environment")
def test_synthetic_accessibility_returns_none_when_unavailable():
    assert synthetic_accessibility("[At]CCO[At]") is None


@pytest.mark.parametrize("bad", GARBAGE)
def test_synthetic_accessibility_returns_none_on_garbage(bad):
    assert synthetic_accessibility(bad) is None


# --------------------------------------------------------------------------
# loop-canonicalize-break conversion (the paper's actual PSMILES -> PSELFIES)
# --------------------------------------------------------------------------
#
# Paper, verbatim: "the two polymer ends were first joined by removing the [*]
# tokens to form a cyclic structure, followed by canonicalization to mitigate
# any initial sequence bias. Subsequently, a single bond within the backbone was
# strategically cleaved, and Astatine (At) atoms were attached at the cleavage
# points."


# Equivalent writings of one polymer. Each group must collapse to a single
# loop-break PSELFIES -- that is the entire point of canonicalizing the ring
# before choosing where to cleave it.
EQUIVALENT_WRITINGS = {
    "PEO": [
        "[At]CCO[At]",
        "[At]OCC[At]",
        "C([At])CO[At]",
        "[*]CCO[*]",
    ],
    "PET-like": [
        "[At]OCCOC(=O)c1ccc(cc1)C(=O)[At]",
        "[At]C(=O)c1ccc(cc1)C(=O)OCCO[At]",
        "O(CCO[At])C(=O)c1ccc(cc1)C(=O)[At]",
        "[At]OCCOC(=O)C1=CC=C(C=C1)C(=O)[At]",  # kekulized
    ],
    "nylon-6-like": [
        "[At]NCCCCCC(=O)[At]",
        "[At]C(=O)CCCCCN[At]",
        "C(CCCC(=O)[At])CN[At]",
    ],
    "bisphenol-A-like": [
        "[At]c1ccc(cc1)C(C)(C)c1ccc([At])cc1",
        "[At]c1ccc(C(C)(C)c2ccc([At])cc2)cc1",
        "CC(C)(c1ccc([At])cc1)c1ccc([At])cc1",
    ],
}

# Repeat units that cannot be cyclized: joining the ends would need a one- or
# two-membered ring, which no molecular graph can express. Vinyl polymers all
# land here, so this is a common case rather than an exotic one.
DEGENERATE_PSMILES = [
    "[At]CC[At]",        # polyethylene: the two neighbours are already bonded
    "[At]CC(C)[At]",     # polypropylene
    "[At]CC(Cl)[At]",    # PVC
    "[At]c1ccccc1[At]",  # ortho-phenylene: neighbours adjacent in the ring
    "[At]C([At])C",      # both termini on the same atom
    "[At]C[At]",         # polymethylene, likewise
]


def test_cyclize_psmiles_forms_a_ring():
    """[At]CCO[At] becomes oxirane: a real ring, with both termini consumed."""
    mol = cyclize_psmiles("[At]CCO[At]")
    assert mol is not None

    ring_info = mol.GetRingInfo()
    assert ring_info.NumRings() == 1
    assert mol.GetNumAtoms() == 3  # two carbons and one oxygen, no astatine
    assert all(atom.GetSymbol() != "At" for atom in mol.GetAtoms())
    assert all(atom.IsInRing() for atom in mol.GetAtoms())


@pytest.mark.parametrize(
    ("psmiles", "n_ring_atoms"),
    [
        ("[At]CCO[At]", 3),
        ("[At]NCCCCCC(=O)[At]", 7),
        ("[*]CCO[*]", 3),  # star notation is accepted too
    ],
)
def test_cyclize_psmiles_ring_size(psmiles, n_ring_atoms):
    mol = cyclize_psmiles(psmiles)
    assert mol is not None
    assert mol.GetRingInfo().NumRings() >= 1
    assert sum(1 for atom in mol.GetAtoms() if atom.IsInRing()) >= n_ring_atoms
    assert all(atom.GetSymbol() != "At" for atom in mol.GetAtoms())


@pytest.mark.parametrize("psmiles", DEGENERATE_PSMILES)
def test_cyclize_psmiles_degenerate_cases_return_none(psmiles):
    """Same-atom and already-bonded termini cannot be cyclized; never raises."""
    assert cyclize_psmiles(psmiles) is None


@pytest.mark.parametrize(
    "psmiles",
    ["CCO", "[At]CCO", "[At]C([At])CO[At]", "c1ccccc1"],
)
def test_cyclize_psmiles_requires_exactly_two_termini(psmiles):
    assert cyclize_psmiles(psmiles) is None


@pytest.mark.parametrize("bad", GARBAGE)
def test_cyclize_psmiles_returns_none_on_garbage(bad):
    assert cyclize_psmiles(bad) is None


def test_cleave_and_cap_restores_two_termini():
    mol = cyclize_psmiles("[At]OCCOC(=O)c1ccc(cc1)C(=O)[At]")
    assert mol is not None
    result = cleave_and_cap(mol)
    assert result is not None
    assert count_termini(result) == 2
    assert validate_psmiles(result).valid


def test_cleave_and_cap_rejects_unknown_strategy():
    mol = cyclize_psmiles("[At]CCO[At]")
    assert mol is not None
    with pytest.raises(ValueError):
        cleave_and_cap(mol, strategy="whatever")


def test_cleave_and_cap_returns_none_for_unmarked_mol_under_original():
    """The "original" strategy needs the join bond recorded by cyclize_psmiles."""
    from rdkit import Chem

    plain_ring = Chem.MolFromSmiles("C1CCCCC1")
    assert cleave_and_cap(plain_ring, strategy="original") is None
    # ...but the canonical-rank rule works on any ring-bearing molecule.
    assert cleave_and_cap(plain_ring, strategy="canonical_rank") is not None


def test_cleave_and_cap_none_when_no_cleavable_bond():
    """Benzene has only aromatic ring bonds, so there is nothing to cleave."""
    from rdkit import Chem

    assert cleave_and_cap(Chem.MolFromSmiles("c1ccccc1")) is None
    assert cleave_and_cap(Chem.MolFromSmiles("CCO")) is None  # acyclic
    assert cleave_and_cap(None) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "psmiles"), sorted(POLYMERS.items()))
def test_loop_break_round_trips_to_a_valid_polymer(name, psmiles):
    """The full pipeline must yield a decodable, 2-terminus, valid polymer."""
    pselfies = psmiles_to_pselfies_loop_break(psmiles)
    assert pselfies is not None, name

    decoded = pselfies_to_psmiles(pselfies)
    assert decoded is not None, name

    result = validate_psmiles(decoded)
    assert result.valid, name
    assert result.n_termini == 2, name
    assert result.correct_termini, name


@pytest.mark.parametrize("strategy", ["canonical_rank", "original"])
@pytest.mark.parametrize(("name", "psmiles"), sorted(POLYMERS.items()))
def test_loop_break_is_deterministic(name, psmiles, strategy):
    """Same input, same strategy -> byte-identical output, every time."""
    outputs = {
        psmiles_to_pselfies_loop_break(psmiles, strategy=strategy) for _ in range(10)
    }
    assert len(outputs) == 1, f"{name}/{strategy} was not deterministic: {outputs}"


@pytest.mark.parametrize(("name", "writings"), sorted(EQUIVALENT_WRITINGS.items()))
def test_loop_break_is_canonically_invariant(name, writings):
    """Equivalent writings of one polymer must give ONE loop-break PSELFIES.

    This is the empirical claim the paper's "canonicalization to mitigate any
    initial sequence bias" step rests on. It is asserted at full strength: if
    RDKit's canonical ranking were not order-invariant here, this would fail
    rather than be weakened.
    """
    results = {psmiles_to_pselfies_loop_break(w) for w in writings}
    assert None not in results, f"{name}: some writing failed to convert"
    assert len(results) == 1, f"{name} produced {len(results)} distinct PSELFIES: {results}"


def test_loop_break_canonical_invariance_holds_across_all_groups():
    """Cross-check: every group collapses, and different polymers stay distinct."""
    per_group = {
        name: {psmiles_to_pselfies_loop_break(w) for w in writings}
        for name, writings in EQUIVALENT_WRITINGS.items()
    }
    assert all(len(results) == 1 for results in per_group.values())

    collapsed = [results.pop() for results in per_group.values()]
    assert len(set(collapsed)) == len(collapsed), "distinct polymers collided"


@pytest.mark.parametrize(
    ("name", "psmiles"),
    [
        ("bisphenol-A-like", "[At]c1ccc(cc1)C(C)(C)c1ccc([At])cc1"),
        ("para-phenylene", "[At]c1ccc([At])cc1"),
    ],
)
def test_original_strategy_reproduces_direct_substitution(name, psmiles):
    """Where the input's own terminus bond IS the canonical-rank pick, the
    loop-break pipeline agrees with the direct substitution.

    Both polymers here are symmetric enough that cleaving the canonical-rank
    bond lands back on the original terminus positions. For asymmetric backbones
    such as PEO the two disagree by construction -- see
    ``test_loop_break_differs_from_direct_substitution`` -- so this equality is
    asserted only for cases where it is genuinely expected, never in general.
    """
    direct = psmiles_to_pselfies(psmiles)
    original = psmiles_to_pselfies_loop_break(psmiles, strategy="original")
    canonical = psmiles_to_pselfies_loop_break(psmiles, strategy="canonical_rank")

    assert direct is not None and original is not None and canonical is not None
    # "original" reproduces the input's own terminus positions, up to the
    # canonicalization that both paths apply to the final PSMILES.
    assert is_same_polymer(original, direct, kind="pselfies")
    assert original == canonical


def test_original_strategy_recovers_input_terminus_positions():
    """The "original" strategy round-trips cleavage back to the input termini."""
    for psmiles in ("[At]CCO[At]", "[At]NCCCCCC(=O)[At]", "[At]OCCOC(=O)c1ccc(cc1)C(=O)[At]"):
        mol = cyclize_psmiles(psmiles)
        assert mol is not None
        recovered = cleave_and_cap(mol, strategy="original")
        assert recovered == canonical_psmiles(psmiles), psmiles


def test_loop_break_differs_from_direct_substitution():
    """The whole reason this pipeline matters: it can pick a different bond.

    PEO cyclizes to oxirane; the canonical-rank rule cleaves the C-C bond rather
    than the C-O bond the input's termini sat on. The result is a different
    repeat-unit *phase* of the same infinite chain -- ...CH2-CH2-O... written as
    [At]COC[At] instead of [At]CCO[At] -- so the PSELFIES string differs while
    the polymer does not.
    """
    direct = psmiles_to_pselfies("[At]CCO[At]")
    loop = psmiles_to_pselfies_loop_break("[At]CCO[At]")
    assert direct == "[At][C][C][O][At]"
    assert loop is not None
    assert loop != direct
    assert pselfies_to_psmiles(loop) == "[At]COC[At]"


@pytest.mark.parametrize("psmiles", DEGENERATE_PSMILES)
def test_loop_break_falls_back_to_direct_for_degenerate_cases(psmiles):
    """Un-cyclizable repeat units fall back to the direct substitution."""
    result = psmiles_to_pselfies_loop_break(psmiles)
    assert result == psmiles_to_pselfies(psmiles)


@pytest.mark.parametrize("psmiles", DEGENERATE_PSMILES)
def test_loop_break_can_refuse_to_fall_back(psmiles):
    """With the fallback disabled the degenerate cases report failure honestly."""
    assert psmiles_to_pselfies_loop_break(psmiles, fallback_to_direct=False) is None


def test_loop_break_fully_cyclic_degenerate_case():
    """A repeat unit that is already a ring is handled without raising."""
    # Para-phenylene cyclizes into a bridged bicycle; the bridge is the only
    # non-aromatic single ring bond, so cleaving it returns the input.
    assert psmiles_to_pselfies_loop_break("[At]c1ccc([At])cc1") is not None

    # A ring bearing no termini at all cannot enter the pipeline.
    assert psmiles_to_pselfies_loop_break("c1ccccc1", fallback_to_direct=False) is None


@pytest.mark.parametrize("bad", GARBAGE)
def test_loop_break_returns_none_on_garbage(bad):
    assert psmiles_to_pselfies_loop_break(bad) is None
    assert psmiles_to_pselfies_loop_break(bad, strategy="original") is None
    assert psmiles_to_pselfies_loop_break(bad, fallback_to_direct=False) is None


def test_loop_break_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        psmiles_to_pselfies_loop_break("[At]CCO[At]", strategy="nonsense")


def test_loop_break_output_is_selfies_reproducible():
    """Loop-break output is encoder output, so it must survive the SR test."""
    for psmiles in POLYMERS.values():
        pselfies = psmiles_to_pselfies_loop_break(psmiles)
        assert pselfies is not None
        assert selfies_reproducible(pselfies), psmiles
