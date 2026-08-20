"""Tests for the evaluation layer: the paper's filter cascade, metrics, and reporting.

Every expected number in this file is either hand-computable or recomputed
independently with numpy inside the test, so a regression in the library cannot
be papered over by copying the library's own output back into the assertion.

The chemistry fixtures below were chosen against the real ``selfies``/``rdkit``
installed in this environment; the decoded PSMILES for each PSELFIES string is
recorded in the fixture table so the intent stays readable.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from rdkit import Chem, RDLogger

from polyt5.chemistry import NoveltyIndex
from polyt5.evaluation import (
    CandidateRecord,
    FilterCounts,
    GenerationReport,
    RegressionReport,
    aggregate_over_splits,
    apply_filter_cascade,
    build_reference_fingerprints,
    diversity_metrics,
    ecfp6,
    evaluate_generation,
    format_console_summary,
    has_valid_termini,
    loop_closed_mol,
    max_similarity_to_reference,
    parse_numeric_predictions,
    regression_report,
    tanimoto,
    target_property_rate,
    write_generation_report,
)
from polyt5.evaluation import filters as filters_mod
from polyt5.evaluation import generation_metrics as gen_mod
from polyt5.evaluation.similarity import FINGERPRINTS_AVAILABLE

RDLogger.DisableLog("rdApp.*")

requires_fingerprints = pytest.mark.skipif(
    not FINGERPRINTS_AVAILABLE,
    reason="RDKit Morgan fingerprints unavailable in this environment",
)


# ===========================================================================
# has_valid_termini -- the paper's PV rule (exactly two [At], each valency 1)
# ===========================================================================


def test_has_valid_termini_accepts_two_monovalent_astatines():
    assert has_valid_termini("[At]CCO[At]") is True
    assert has_valid_termini("[At]CC[At]") is True
    assert has_valid_termini("[At]COC[At]") is True


def test_has_valid_termini_rejects_wrong_terminus_count():
    assert has_valid_termini("[At]CCO") is False  # one terminus
    assert has_valid_termini("CCO") is False  # none
    assert has_valid_termini("[At]CC([At])C[At]") is False  # three


def test_has_valid_termini_rejects_divalent_astatine_from_smiles():
    """``[At]=C[At]`` parses in RDKit but the first astatine has valency 2.

    This is exactly the gap the paper's PV rule closes and that a bare terminus
    *count* misses: ``count_termini`` reports 2 here.
    """
    from polyt5.chemistry import count_termini

    psmiles = "[At]=C[At]"
    assert count_termini(psmiles) == 2, "fixture assumption: the count check passes"
    assert has_valid_termini(psmiles) is False


def test_has_valid_termini_rejects_divalent_astatine_built_with_rwmol():
    """Same rule, on a molecule assembled atom-by-atom rather than parsed."""
    rw = Chem.RWMol()
    atoms = []
    for atomic_number in (85, 6, 85):
        atom = Chem.Atom(atomic_number)
        if atomic_number == 85:
            # Pin the hydrogen count so the divalent astatine stays divalent
            # instead of being promoted to the next allowed halogen valence.
            atom.SetNoImplicit(True)
        atoms.append(rw.AddAtom(atom))
    at_a, carbon, at_b = atoms
    rw.AddBond(at_a, carbon, Chem.BondType.DOUBLE)
    rw.AddBond(carbon, at_b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    valences = sorted(a.GetTotalValence() for a in mol.GetAtoms() if a.GetSymbol() == "At")
    assert valences == [1, 2], "fixture assumption: one astatine is divalent"
    assert has_valid_termini(mol) is False


def test_has_valid_termini_rejects_charged_or_disconnected_terminus():
    assert has_valid_termini("[At+]CCO[At]") is False
    assert has_valid_termini("[At].[At]CCO") is False  # a free astatine, degree 0


def test_has_valid_termini_is_total_on_garbage():
    for junk in ["", "   ", "not a molecule", "[[[", None, 42]:
        assert has_valid_termini(junk) is False


def test_has_valid_termini_honours_expected():
    assert has_valid_termini("[At]CC([At])C[At]", expected=3) is True
    assert has_valid_termini("[At]CCO[At]", expected=3) is False


# ===========================================================================
# apply_filter_cascade -- SV -> TSD -> DD -> PV, sequential and nested
# ===========================================================================

# index -> (pselfies, decoded psmiles, expected failure stage)
CASCADE_CASES: list[tuple[str, str, str | None]] = [
    ("garbage", "<undecodable>", "SV"),
    ("[Xx][Yy]", "<undecodable>", "SV"),
    ("[At][C][C][O][At]", "[At]CCO[At]", "TSD"),  # in the training set
    ("[C][C]", "CC", "TSD"),  # in training AND terminus-invalid -> TSD wins
    ("[At][C][C][At]", "[At]CC[At]", None),  # the only full pass
    ("[C][Branch1][C][At][C][At]", "C([At])C[At]", "DD"),  # same polymer as above
    ("[At][C][Ring1][C][At]", "[At]=C[At]", "PV"),  # divalent astatine
    ("[At][C][C][Branch3][C][C][At]", "[At]CC", "PV"),  # one terminus
]
CASCADE_INPUT = [case[0] for case in CASCADE_CASES]
CASCADE_STAGES = [case[2] for case in CASCADE_CASES]


@pytest.fixture()
def training_index() -> NoveltyIndex:
    """Training set holding ``[At]CCO[At]`` and the terminus-free ``CC``."""
    index = NoveltyIndex(["[At]CCO[At]", "CC"])
    assert len(index) == 2
    return index


def test_cascade_counts_are_exact(training_index):
    records, counts = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)

    assert len(records) == 8
    assert counts.n_input == 8
    assert counts.n_sv == 6
    assert counts.n_tsd == 4
    assert counts.n_dd == 3
    assert counts.n_pv == 1

    assert counts.sv_rate == pytest.approx(6 / 8)
    assert counts.tsd_rate == pytest.approx(4 / 8)
    assert counts.dd_rate == pytest.approx(3 / 8)
    assert counts.pv_rate == pytest.approx(1 / 8)


def test_cascade_nesting_holds(training_index):
    _, counts = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    assert counts.n_input >= counts.n_sv >= counts.n_tsd >= counts.n_dd >= counts.n_pv


def test_cascade_records_the_failure_stage_of_every_candidate(training_index):
    records, _ = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    assert [r.failure_stage for r in records] == CASCADE_STAGES


def test_cascade_is_sequential_training_duplicate_beats_terminus_failure(training_index):
    """``[C][C]`` decodes to ``CC``: in the training set *and* zero termini.

    The paper's filters are nested, so it must be charged to TSD -- the first
    stage it fails -- and must never reach the PV stage.
    """
    records, _ = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    record = records[3]

    assert record.canonical_psmiles == "CC"
    assert record.failure_stage == "TSD"
    assert record.passed_sv is True
    assert record.passed_tsd is False
    assert record.passed_dd is False  # not evaluated -> not passed
    assert record.passed_pv is False
    assert has_valid_termini(record.canonical_psmiles) is False, "would also fail PV"


def test_cascade_flags_are_monotone_per_record(training_index):
    records, _ = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    for record in records:
        if record.passed_pv:
            assert record.passed_dd and record.passed_tsd and record.passed_sv
        if record.passed_dd:
            assert record.passed_tsd and record.passed_sv
        if record.passed_tsd:
            assert record.passed_sv


def test_cascade_dedup_keeps_the_first_writing_of_a_polymer(training_index):
    records, _ = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    first, duplicate = records[4], records[5]

    assert first.canonical_psmiles == duplicate.canonical_psmiles == "[At]CC[At]"
    assert first.failure_stage is None
    assert duplicate.failure_stage == "DD"


def test_cascade_populates_string_fields(training_index):
    records, _ = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)

    assert records[0].raw_pselfies == "garbage"
    assert records[0].psmiles is None
    assert records[0].canonical_psmiles is None

    assert records[4].raw_pselfies == "[At][C][C][At]"
    assert records[4].psmiles == "[At]CC[At]"
    assert records[4].canonical_psmiles == "[At]CC[At]"


def test_cascade_records_selfies_reproducibility(training_index):
    records, _ = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    assert [r.reproducible for r in records] == [
        False,  # garbage
        False,  # [Xx][Yy]
        True,  # [At][C][C][O][At]
        True,  # [C][C]
        True,  # [At][C][C][At]
        True,  # [C][Branch1][C][At][C][At]
        False,  # [At][C][Ring1][C][At]
        False,  # [At][C][C][Branch3][C][C][At]
    ]


def test_cascade_without_training_index_skips_tsd():
    records, counts = apply_filter_cascade(CASCADE_INPUT, training_index=None)
    assert counts.n_sv == 6
    assert counts.n_tsd == 6, "no reference set -> nothing can be a training duplicate"
    assert counts.n_dd == 5, "[C][C] is now a unique survivor"
    assert counts.n_pv == 2, "[At]CCO[At] is no longer screened out as a training duplicate"
    assert records[2].failure_stage is None, "was TSD, now a full pass"
    assert records[3].failure_stage == "PV", "now reaches the terminus check"


def test_cascade_is_total_on_adversarial_input():
    junk = ["", "   ", "\n", "[At]" * 400, "((((", "🙂", "[C][C][C]" * 200]
    records, counts = apply_filter_cascade(junk, training_index=None)
    assert counts.n_input == len(junk) == len(records)
    assert counts.n_pv <= counts.n_dd <= counts.n_tsd <= counts.n_sv <= counts.n_input


def test_cascade_on_empty_input():
    records, counts = apply_filter_cascade([], training_index=None)
    assert records == []
    assert counts.n_input == 0
    assert counts.sv_rate == 0.0
    assert counts.pv_rate == 0.0


def test_filter_counts_to_dict_is_json_serialisable(training_index):
    _, counts = apply_filter_cascade(CASCADE_INPUT, training_index=training_index)
    payload = counts.to_dict()
    assert payload["n_input"] == 8
    assert payload["n_pv"] == 1
    json.dumps(payload)


def test_cascade_computes_sa_only_for_full_passes(training_index):
    records, _ = apply_filter_cascade(
        CASCADE_INPUT, training_index=training_index, compute_sa=True
    )
    assert records[4].sa_score is not None
    assert records[4].sa_score == pytest.approx(2.7476, abs=1e-3)
    for i, record in enumerate(records):
        if i != 4:
            assert record.sa_score is None


def test_cascade_skips_sa_when_not_requested(training_index):
    records, _ = apply_filter_cascade(
        CASCADE_INPUT, training_index=training_index, compute_sa=False
    )
    assert all(r.sa_score is None for r in records)


def test_cascade_expected_termini_is_configurable():
    records, counts = apply_filter_cascade(
        ["[At][C][C][Branch1][C][At][C][At]"], training_index=None, expected_termini=3
    )
    assert records[0].psmiles == "[At]CC([At])C[At]"
    assert counts.n_pv == 1


# ===========================================================================
# SELFIES reproducibility (SR)
# ===========================================================================


def test_sr_rate_on_known_reproducible_and_non_reproducible_strings():
    report = evaluate_generation(
        ["[At][C][C][At]", "[At][C][Ring1][C][At]"], compute_sa=False
    )
    assert report.sr_rate == pytest.approx(0.5)


def test_sr_rate_over_the_cascade_fixture(training_index):
    report = evaluate_generation(
        CASCADE_INPUT, training_index=training_index, compute_sa=False
    )
    assert report.sr_rate == pytest.approx(4 / 8)


def test_sr_rate_is_zero_for_empty_input():
    assert evaluate_generation([], compute_sa=False).sr_rate == 0.0


# ===========================================================================
# target_property_rate (the paper's TP metric)
# ===========================================================================


def test_target_property_rate_hand_values():
    values = [450.0, 500.0, 550.0, 551.0, 449.0, 500.0]
    assert target_property_rate(values, 500.0, 50.0) == pytest.approx(4 / 6)


def test_target_property_rate_boundaries_are_inclusive():
    assert target_property_rate([450.0, 550.0], 500.0, 50.0) == pytest.approx(1.0)
    assert target_property_rate([449.9, 550.1], 500.0, 50.0) == pytest.approx(0.0)


def test_target_property_rate_ignores_non_finite_values():
    values = [500.0, float("nan"), None, float("inf"), 1000.0]
    assert target_property_rate(values, 500.0, 50.0) == pytest.approx(0.5)


def test_target_property_rate_on_empty_input():
    assert target_property_rate([], 500.0, 50.0) == 0.0
    assert target_property_rate([float("nan")], 500.0, 50.0) == 0.0


# ===========================================================================
# regression_metrics
# ===========================================================================


def test_parse_numeric_predictions_counts_non_numeric():
    texts = ["soluble", "", "abc", "1.2.3", "412.5", " 3.0 ", "-1e-3", "nan"]
    values, n_non_numeric = parse_numeric_predictions(texts)

    assert values == [None, None, None, None, 412.5, 3.0, -1e-3, None]
    assert n_non_numeric == 5


def test_parse_numeric_predictions_accepts_plain_numbers():
    values, n_non_numeric = parse_numeric_predictions([1, 2.5, "3"])
    assert values == [1.0, 2.5, 3.0]
    assert n_non_numeric == 0


def test_parse_numeric_predictions_is_total_on_junk():
    values, n_non_numeric = parse_numeric_predictions([None, object(), "∞", "1,234.5"])
    assert values == [None, None, None, None]
    assert n_non_numeric == 4


def test_regression_report_hand_computed():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = ["1.5", "2.5", "soluble", "3.5"]

    report = regression_report(y_true, y_pred)

    assert report.n_total == 4
    assert report.n_valid_numeric == 3
    assert report.n_non_numeric == 1
    assert report.non_numeric_rate == pytest.approx(0.25)

    # Errors on the surviving pairs are +0.5, +0.5, -0.5.
    assert report.mae == pytest.approx(0.5)
    assert report.rmse == pytest.approx(0.5)

    kept_true = np.array([1.0, 2.0, 4.0])
    kept_pred = np.array([1.5, 2.5, 3.5])
    ss_res = float(((kept_true - kept_pred) ** 2).sum())
    ss_tot = float(((kept_true - kept_true.mean()) ** 2).sum())
    assert report.r2 == pytest.approx(1.0 - ss_res / ss_tot)
    assert report.pearson_r == pytest.approx(float(np.corrcoef(kept_true, kept_pred)[0, 1]))


def test_regression_report_perfect_prediction():
    report = regression_report([1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"])
    assert report.mae == pytest.approx(0.0)
    assert report.rmse == pytest.approx(0.0)
    assert report.r2 == pytest.approx(1.0)
    assert report.pearson_r == pytest.approx(1.0)
    assert report.n_non_numeric == 0


def test_regression_report_all_non_numeric_returns_none_not_nan():
    report = regression_report([1.0, 2.0], ["soluble", "polymer"])

    assert report.n_total == 2
    assert report.n_valid_numeric == 0
    assert report.n_non_numeric == 2
    assert report.non_numeric_rate == pytest.approx(1.0)
    for metric in (report.mae, report.rmse, report.r2, report.pearson_r):
        assert metric is None


def test_regression_report_constant_y_true_gives_none_r2():
    report = regression_report([5.0, 5.0, 5.0], ["5.0", "6.0", "4.0"])

    assert report.r2 is None, "R^2 is undefined when the target has zero variance"
    assert report.pearson_r is None
    assert report.mae == pytest.approx(2 / 3)
    assert report.rmse == pytest.approx(math.sqrt(2 / 3))


def test_regression_report_single_pair_has_no_correlation_metrics():
    report = regression_report([1.0], ["2.0"])
    assert report.mae == pytest.approx(1.0)
    assert report.rmse == pytest.approx(1.0)
    assert report.r2 is None
    assert report.pearson_r is None


def test_regression_report_refuses_to_drop_when_asked_not_to():
    report = regression_report([1.0, 2.0], ["1.0", "soluble"], drop_non_numeric=False)
    assert report.n_non_numeric == 1
    for metric in (report.mae, report.rmse, report.r2, report.pearson_r):
        assert metric is None


def test_regression_report_length_mismatch_is_a_caller_error():
    with pytest.raises(ValueError):
        regression_report([1.0, 2.0], ["1.0"])


def test_regression_report_empty():
    report = regression_report([], [])
    assert report.n_total == 0
    assert report.non_numeric_rate == 0.0
    assert report.mae is None


def test_regression_report_to_dict_is_json_serialisable():
    report = regression_report([1.0, 2.0, 3.0], ["1.1", "2.1", "abc"])
    payload = report.to_dict()
    assert payload["n_total"] == 3
    assert payload["n_non_numeric"] == 1
    json.dumps(payload)


def test_aggregate_over_splits_mean_and_std():
    reports = [
        RegressionReport(
            n_total=10,
            n_valid_numeric=10,
            n_non_numeric=0,
            non_numeric_rate=0.0,
            mae=mae,
            rmse=2 * mae,
            r2=0.5,
            pearson_r=None,
        )
        for mae in (1.0, 2.0, 3.0)
    ]

    agg = aggregate_over_splits(reports)

    assert agg["n_splits"] == 3
    assert agg["mae"]["mean"] == pytest.approx(2.0)
    assert agg["mae"]["std"] == pytest.approx(math.sqrt(2 / 3))
    assert agg["mae"]["n"] == 3
    assert agg["rmse"]["mean"] == pytest.approx(4.0)
    assert agg["rmse"]["std"] == pytest.approx(2 * math.sqrt(2 / 3))
    assert agg["r2"]["mean"] == pytest.approx(0.5)
    assert agg["r2"]["std"] == pytest.approx(0.0)
    assert agg["pearson_r"]["mean"] is None
    assert agg["pearson_r"]["n"] == 0


def test_aggregate_over_splits_sample_std_option():
    reports = [
        RegressionReport(10, 10, 0, 0.0, mae, None, None, None) for mae in (1.0, 2.0, 3.0)
    ]
    agg = aggregate_over_splits(reports, ddof=1)
    assert agg["mae"]["std"] == pytest.approx(1.0)


def test_aggregate_over_splits_empty():
    agg = aggregate_over_splits([])
    assert agg["n_splits"] == 0
    assert agg["mae"]["mean"] is None


# ===========================================================================
# similarity: loop closure, ECFP6, Tanimoto
# ===========================================================================


def test_loop_closed_mol_forms_a_ring_and_removes_astatine():
    mol = loop_closed_mol("[At]CCO[At]")

    assert mol is not None
    assert mol.GetRingInfo().NumRings() == 1
    assert all(atom.GetSymbol() != "At" for atom in mol.GetAtoms())
    assert Chem.MolToSmiles(mol) == "C1CO1"


def test_loop_closed_mol_on_a_longer_repeat_unit():
    mol = loop_closed_mol("[At]CCCCO[At]")
    assert mol is not None
    assert mol.GetRingInfo().NumRings() == 1
    assert Chem.MolToSmiles(mol) == "C1CCOC1"


def test_loop_closed_mol_accepts_star_notation():
    from_star = loop_closed_mol("[*]CCO[*]")
    assert from_star is not None
    assert Chem.MolToSmiles(from_star) == "C1CO1"


def test_loop_closed_mol_returns_none_on_bad_input():
    assert loop_closed_mol("not a molecule") is None
    assert loop_closed_mol("") is None
    assert loop_closed_mol("[At]CCO") is None, "needs exactly two termini"
    assert loop_closed_mol("CCO") is None


def test_loop_closed_mol_returns_none_when_the_ends_are_already_bonded():
    """``[At]CC[At]`` would need a two-membered ring; RDKit cannot express one."""
    assert loop_closed_mol("[At]CC[At]") is None


@requires_fingerprints
def test_ecfp6_identical_polymers_have_tanimoto_one():
    a = ecfp6("[At]CCO[At]")
    b = ecfp6("[At]CCO[At]")
    assert a is not None and b is not None
    assert a.GetNumBits() == 2048
    assert tanimoto(a, b) == pytest.approx(1.0)


@requires_fingerprints
def test_ecfp6_is_invariant_to_the_written_form():
    a = ecfp6("[At]CCO[At]")
    b = ecfp6("[At]OCC[At]")
    assert tanimoto(a, b) == pytest.approx(1.0)


@requires_fingerprints
def test_ecfp6_different_polymers_are_less_than_one():
    a = ecfp6("[At]CCO[At]")
    b = ecfp6("[At]c1ccc(cc1)C(=O)OCC[At]")
    assert a is not None and b is not None
    similarity = tanimoto(a, b)
    assert 0.0 <= similarity < 1.0


@requires_fingerprints
def test_ecfp6_respects_n_bits_and_loop_closed_flag():
    assert ecfp6("[At]CCO[At]", n_bits=1024).GetNumBits() == 1024
    assert ecfp6("[At]CC[At]", loop_closed=True) is None, "loop closure fails here"
    assert ecfp6("[At]CC[At]", loop_closed=False) is not None


@requires_fingerprints
def test_ecfp6_returns_none_on_garbage():
    assert ecfp6("not a molecule") is None
    assert ecfp6("") is None


def test_tanimoto_handles_none():
    assert tanimoto(None, None) == 0.0
    assert tanimoto(ecfp6("[At]CCO[At]"), None) == 0.0
    assert tanimoto(None, ecfp6("[At]CCO[At]")) == 0.0


@requires_fingerprints
def test_max_similarity_to_reference():
    reference = build_reference_fingerprints(["[At]CCO[At]", "[At]CCCCO[At]"])
    assert len(reference) == 2

    assert max_similarity_to_reference("[At]CCO[At]", reference) == pytest.approx(1.0)

    partial = max_similarity_to_reference("[At]CCCCO[At]", reference)
    assert partial == pytest.approx(1.0)

    unrelated = max_similarity_to_reference("[At]c1ccc(cc1)C(=O)OCC[At]", reference)
    assert unrelated is not None and unrelated < 1.0


def test_max_similarity_to_reference_returns_none_when_undefined():
    assert max_similarity_to_reference("[At]CCO[At]", []) is None
    reference = build_reference_fingerprints(["[At]CCO[At]"])
    assert max_similarity_to_reference("garbage", reference) is None


def test_build_reference_fingerprints_skips_unusable_entries():
    reference = build_reference_fingerprints(["[At]CCO[At]", "garbage", "", "[At]CC[At]"])
    # "garbage"/"" cannot be parsed and "[At]CC[At]" cannot be loop-closed.
    assert len(reference) == 1


# ===========================================================================
# diversity_metrics
# ===========================================================================


def test_diversity_metrics_counts():
    result = diversity_metrics(["[At]CCO[At]", "[At]CCO[At]", "[At]CC[At]", "[At]COC[At]"])
    assert result["n_total"] == 4
    assert result["n_unique"] == 3
    assert result["unique_fraction"] == pytest.approx(0.75)
    assert result["most_common_count"] == 2
    assert result["most_common_fraction"] == pytest.approx(0.5)


def test_diversity_metrics_empty():
    result = diversity_metrics([])
    assert result["n_total"] == 0
    assert result["n_unique"] == 0
    assert result["unique_fraction"] == 0.0
    assert result["mean_pairwise_tanimoto"] is None


@requires_fingerprints
def test_diversity_metrics_pairwise_tanimoto():
    result = diversity_metrics(["[At]CCO[At]", "[At]CCCCO[At]", "[At]c1ccccc1CCO[At]"])
    assert result["mean_pairwise_tanimoto"] is not None
    assert 0.0 <= result["mean_pairwise_tanimoto"] <= 1.0
    assert result["n_pairwise_sampled"] == 3


@requires_fingerprints
def test_diversity_metrics_pairwise_sample_is_capped():
    polymers = [f"[At]{'C' * n}O[At]" for n in range(2, 30)]
    result = diversity_metrics(polymers, max_pairwise=5)
    assert result["pairwise_sample_cap"] == 5
    assert result["n_pairwise_sampled"] == 5


def test_diversity_metrics_is_deterministic_under_a_seed():
    polymers = [f"[At]{'C' * n}O[At]" for n in range(2, 30)]
    first = diversity_metrics(polymers, max_pairwise=5, seed=7)
    second = diversity_metrics(polymers, max_pairwise=5, seed=7)
    assert first == second


# ===========================================================================
# evaluate_generation end-to-end
# ===========================================================================

CLEAN_BATCH = [
    "[At][C][C][At]",
    "[At][C][C][O][At]",
    "[At][C][O][C][At]",
    "[At][C][C][C][At]",
]


def test_evaluate_generation_end_to_end(training_index):
    report = evaluate_generation(CASCADE_INPUT, training_index=training_index, compute_sa=True)

    assert isinstance(report, GenerationReport)
    assert report.counts.n_input == 8
    assert report.counts.n_pv == 1
    assert report.sr_rate == pytest.approx(4 / 8)
    # SV survivors canonicalise to [At]CCO[At], CC, [At]CC[At] (twice),
    # [At]C=[At] and CC[At]: five distinct polymers, two of which are the
    # training entries [At]CCO[At] and CC.
    assert report.n_unique == 5, "distinct canonical polymers among the SV survivors"
    assert report.n_novel == 3, "of those, three are absent from the training set"
    assert report.duplicate_rate == pytest.approx(1 - 5 / 6)
    json.dumps(report.to_dict())


def test_evaluate_generation_without_predictor_reports_no_property_stats():
    report = evaluate_generation(CLEAN_BATCH, compute_sa=False)

    assert report.n_property_values is None
    assert report.property_mean is None
    assert report.property_median is None
    assert report.property_std is None
    assert report.target_property_rate is None
    assert report.property_target is None
    assert report.property_tolerance is None


def test_evaluate_generation_with_injected_predictor():
    seen: list[list[str]] = []

    def fake_predictor(psmiles_list):
        seen.append(list(psmiles_list))
        return [480.0, 500.0, 560.0, 500.0]

    report = evaluate_generation(
        CLEAN_BATCH,
        target_property=500.0,
        tolerance=50.0,
        property_predictor=fake_predictor,
        compute_sa=False,
    )

    assert report.counts.n_pv == 4, "fixture assumption: all four survive the cascade"
    assert len(seen) == 1 and len(seen[0]) == 4
    assert report.n_property_values == 4
    assert report.target_property_rate == pytest.approx(0.75)
    assert report.property_mean == pytest.approx((480 + 500 + 560 + 500) / 4)
    assert report.property_median == pytest.approx(500.0)
    assert report.property_target == pytest.approx(500.0)
    assert report.property_tolerance == pytest.approx(50.0)


def test_evaluate_generation_survives_a_raising_predictor():
    def broken(_psmiles_list):
        raise RuntimeError("the model fell over")

    report = evaluate_generation(
        CLEAN_BATCH, target_property=500.0, property_predictor=broken, compute_sa=False
    )
    assert report.n_property_values is None
    assert report.target_property_rate is None


def test_evaluate_generation_sa_statistics_when_available():
    report = evaluate_generation(CLEAN_BATCH, compute_sa=True)
    if not report.sa_available:
        pytest.skip("SA scoring unavailable in this environment")

    assert report.n_sa_scored == 4
    assert report.sa_mean is not None
    assert report.sa_median is not None
    assert report.sa_fraction_above_6 == pytest.approx(0.0)
    assert 1.0 <= report.sa_mean <= 10.0


def test_evaluate_generation_sa_statistics_are_none_when_unavailable(monkeypatch):
    """No SA scorer must mean ``None``, never ``0.0`` and never an estimate."""
    monkeypatch.setattr(filters_mod, "SA_AVAILABLE", False)
    monkeypatch.setattr(gen_mod, "SA_AVAILABLE", False)
    monkeypatch.setattr(filters_mod, "synthetic_accessibility", lambda _s: None)

    report = evaluate_generation(CLEAN_BATCH, compute_sa=True)

    assert report.sa_available is False
    assert report.n_sa_scored == 0
    assert report.sa_mean is None
    assert report.sa_median is None
    assert report.sa_fraction_above_6 is None


def test_evaluate_generation_on_empty_input():
    report = evaluate_generation([], compute_sa=True)
    assert report.counts.n_input == 0
    assert report.sr_rate == 0.0
    assert report.n_unique == 0
    assert report.n_novel == 0
    assert report.duplicate_rate == 0.0
    json.dumps(report.to_dict())


def test_evaluate_generation_exposes_the_records(training_index):
    report = evaluate_generation(CASCADE_INPUT, training_index=training_index, compute_sa=False)
    assert len(report.records) == 8
    assert all(isinstance(record, CandidateRecord) for record in report.records)
    assert report.screened_psmiles == ["[At]CC[At]"]


# ===========================================================================
# report.py
# ===========================================================================


def test_write_generation_report(tmp_path):
    from polyt5.utils import RunDirectory

    run_dir = RunDirectory.create(tmp_path, "eval_test")
    report = evaluate_generation(CLEAN_BATCH, compute_sa=False)

    write_generation_report(run_dir, report, generations=report.records)

    metrics_path = run_dir.root / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["counts"]["n_input"] == 4

    generations_path = run_dir.root / "generations.jsonl"
    assert generations_path.exists()
    rows = [
        json.loads(line)
        for line in generations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    assert rows[0]["raw_pselfies"] == "[At][C][C][At]"


def test_write_generation_report_without_generations(tmp_path):
    from polyt5.utils import RunDirectory

    run_dir = RunDirectory.create(tmp_path, "eval_test_2")
    write_generation_report(run_dir, evaluate_generation(CLEAN_BATCH, compute_sa=False))

    assert (run_dir.root / "metrics.json").exists()
    assert not (run_dir.root / "generations.jsonl").exists()


def test_format_console_summary_returns_a_string(training_index):
    report = evaluate_generation(CASCADE_INPUT, training_index=training_index, compute_sa=False)
    summary = format_console_summary(report)

    assert isinstance(summary, str)
    for label in ("SV", "TSD", "DD", "PV", "SR"):
        assert label in summary
    assert "\n" in summary


def test_format_console_summary_handles_missing_capabilities():
    report = evaluate_generation([], compute_sa=False)
    summary = format_console_summary(report)
    assert isinstance(summary, str) and summary


# ===========================================================================
# structural constraints
# ===========================================================================


def test_evaluation_package_does_not_import_torch():
    """The RL reward worker must be able to import this layer without torch."""
    import subprocess
    import sys

    code = (
        "import sys; import polyt5.evaluation; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dataclasses_are_frozen():
    _, counts = apply_filter_cascade(["[At][C][C][At]"], training_index=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        counts.n_input = 99  # type: ignore[misc]
    assert isinstance(counts, FilterCounts)
