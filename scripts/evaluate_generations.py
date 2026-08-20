"""Score a generations file with the paper's full screen, TP metric included.

This closes the loop the supervised baseline was missing.
:func:`polyt5.evaluation.evaluate_generation` takes its property predictor as an
*injected* callable so the evaluation layer stays free of torch; nothing supplied
one, so ``target_property_rate`` -- the paper's **TP** metric, "the proportion of
candidates with predicted Tg within 500 +- 50 K" -- always came back ``None``.
Here the predictor is built from a fine-tuned Tg-prediction checkpoint and
injected, so TP is actually computed.

TWO MODES, and which one applies is never left implicit:

``per-row`` (default when the rows carry ``target_property``)
    Each candidate is scored against ITS OWN conditioning target. This is the
    metric that matters when a generations file was produced by conditioning on
    a validation set's own Tg values, so the targets span a whole range: a
    single fixed window would score every candidate on a request nobody made of
    it. It is also what the RLVR reward will optimise.

``fixed`` (forced by passing ``--target``)
    Every candidate is scored against one window, which is what the paper does
    with its 500 +- 50 K screen. Correct when the whole batch really was
    conditioned on the same value.

Two flags decide exactly what the conditioning report measures, and both are
recorded in ``evaluation.json`` because both move the numbers:
``--score-stage`` picks the cascade survivors that enter it (PV by default, the
fully screened set), and ``--score-input`` picks the string the predictor is fed
(``raw``, what the model actually emitted, by default). Scoring the canonical
PSMILES instead -- which is what :func:`evaluate_generation` does internally --
rewrote 661 of 665 SELFIES strings on the real Tg run and moved MAE by ~8 K.

Usage:
    # per-row: the honest default for a range-conditioned file
    python scripts/evaluate_generations.py \
        --generations results/finetune_tg_generation/generations.jsonl \
        --predictor-checkpoint results/finetune_tg_prediction/checkpoints/best.pt \
        --tokenizer artifacts/tokenizer/polyt5_vocab.json \
        --training-corpus data/processed/tg/generation/train.jsonl \
        --out results/finetune_tg_generation/

    # fixed: the paper's 500 K screen
    python scripts/evaluate_generations.py ... --target 500 --tolerance 50

Several ``--predictor-checkpoint`` flags build an ENSEMBLE (one member per
split, e.g. from ``scripts/run_splits.py``); the reported property values are
then the ensemble mean and the report also carries the mean member disagreement,
which is the RL phase's cheapest uncertainty signal.

Without ``--predictor-checkpoint`` the run is still valid, but every property
statistic is ``None`` and the script says so loudly. It never substitutes an
estimate for a measurement it could not make.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.chemistry import NoveltyIndex  # noqa: E402
from polyt5.data.prepare import parse_property_value  # noqa: E402
from polyt5.evaluation import (  # noqa: E402
    GenerationReport,
    evaluate_generation,
    format_console_summary,
)
from polyt5.inference import (  # noqa: E402
    DEFAULT_TOLERANCES,
    CachedPredictor,
    ConditioningReport,
    EnsemblePropertyPredictor,
    PolyT5PropertyPredictor,
    conditioning_report,
    looks_like_pselfies,
)
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import load_config  # noqa: E402

EVALUATION_FILENAME = "evaluation.json"

MODE_PER_ROW = "per_row"
MODE_FIXED = "fixed"

_NO_PREDICTOR_BANNER = """
!!  NO PROPERTY PREDICTOR SUPPLIED  !!
    --predictor-checkpoint was omitted, so NO property statistics were computed.
    property_mean / property_std / target_property_rate (the paper's TP metric)
    are null in evaluation.json because they were not measured -- not because
    they are zero. Re-run with --predictor-checkpoint to obtain TP.
""".strip()

_CIRCULARITY_TEXT = {
    "shared_dataset": (
        "CIRCULARITY: the property predictor and the generator were fine-tuned on the SAME "
        "underlying dataset, so the predictor scoring the generator is partly "
        "self-referential. Both models learned the same corpus's quirks, and a candidate "
        "that exploits a blind spot shared by both is scored as a success. Treat these "
        "property numbers as an in-domain sanity check, NOT as independent validation. "
        "This is precisely the reward-hacking risk an RLVR loop would amplify: reserve a "
        "held-out ensemble member as an auditor before optimising against this signal."
    ),
    "unknown_provenance": (
        "CIRCULARITY (unverified): the dataset provenance of the predictor and/or the "
        "generator could not be read, so shared training data cannot be ruled out. A "
        "predictor scoring a generator trained on the same corpus is partly "
        "self-referential; treat the property numbers as an in-domain sanity check until "
        "the provenance is confirmed."
    ),
    "distinct_datasets": (
        "The predictor and the generator record DIFFERENT dataset provenance, so the "
        "usual same-corpus circularity does not apply. A learned predictor is still a "
        "proxy and remains gameable under RL optimisation; an independent auditor is "
        "still worthwhile."
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generations", type=Path, required=True,
                        help='JSONL of {"target_property": "500.0", "generated": "<PSELFIES>"}')
    parser.add_argument("--predictor-checkpoint", type=Path, action="append", default=None,
                        metavar="PATH",
                        help="Fine-tuned Tg-prediction checkpoint. Repeat for an ensemble.")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Tokenizer artifact the predictor checkpoint was trained with. "
                             "Defaults to the path recorded inside the checkpoint; required "
                             "when the checkpoint records none. Its sha256 is verified "
                             "against the checkpoint either way.")
    parser.add_argument("--training-corpus", type=Path, default=None,
                        help="Training JSONL used to build the TSD novelty index.")
    parser.add_argument("--target", type=float, default=None,
                        help="Force FIXED-target mode at this value in K. [PAPER] 500. "
                             "Omit to score each candidate against its own row target.")
    parser.add_argument("--tolerance", type=float, default=50.0,
                        help="Half-width of the fixed TP window in K. [PAPER] 50.")
    parser.add_argument("--tolerances", nargs="+", type=float, default=list(DEFAULT_TOLERANCES),
                        metavar="K",
                        help="Acceptance half-widths for the per-row conditioning report.")
    parser.add_argument("--score-stage", choices=sorted(STAGE_FLAGS), default="pv",
                        help="Which cascade survivors enter the conditioning report. "
                             "'pv' (default) is the fully screened set the paper's TP is "
                             "defined over; 'dd' keeps candidates that are unique and novel "
                             "but have malformed termini, which is a larger and easier "
                             "population. This choice moves every rate, so it is recorded.")
    parser.add_argument("--score-input", choices=["raw", "canonical"], default="raw",
                        help="What the predictor is fed for the conditioning report. 'raw' "
                             "(default) is the PSELFIES the model actually emitted, which is "
                             "what an RLVR reward must score. 'canonical' reproduces the "
                             "canonical-PSMILES round trip that evaluate_generation performs "
                             "internally; on the real Tg run that rewrote 661 of 665 strings "
                             "and cost ~8 K of MAE.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Directory for evaluation.json.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4,
                        help="Predictor beam width. [PAPER] 4.")
    parser.add_argument("--no-sa", action="store_true",
                        help="Skip synthetic-accessibility scoring (the slowest filter step).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N generations (debug).")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    """Resolve a path relative to the repo root unless it is absolute."""
    return path if path.is_absolute() else REPO_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank and unparseable lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_generations(
    path: Path, *, limit: int | None = None
) -> tuple[list[str], list[float | None]]:
    """Read the generated candidates and their conditioning targets.

    Args:
        path: JSONL written by ``scripts/finetune.py --task generation``.
        limit: Keep only the first N rows.

    Returns:
        ``(generated, targets)``, positionally aligned. A target is ``None``
        when the row carried no ``target_property`` or it did not parse as a
        finite number -- recorded as missing, never defaulted to ``0.0``, which
        would be scored as a genuine request for 0 K.
    """
    rows = _read_jsonl(path)
    if limit:
        rows = rows[:limit]

    generated: list[str] = []
    targets: list[float | None] = []
    for row in rows:
        generated.append(str(row.get("generated", "")))
        raw = row.get("target_property")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            targets.append(float(raw))
        else:
            targets.append(parse_property_value(raw) if isinstance(raw, str) else None)
    return generated, targets


def read_training_pselfies(path: Path) -> list[str]:
    """Extract the training-set PSELFIES for the novelty (TSD) index.

    The prediction and generation corpora put the polymer on opposite sides of
    the pair (``source`` for prediction, ``target`` for generation), so both are
    inspected and whichever field is written in SELFIES bracket-token notation
    is taken.

    Args:
        path: A prepared corpus JSONL.

    Returns:
        The training polymers as PSELFIES strings.
    """
    polymers: list[str] = []
    for row in _read_jsonl(path):
        for key in ("pselfies", "target", "source"):
            value = row.get(key)
            if isinstance(value, str) and looks_like_pselfies(value):
                polymers.append(value)
                break
    return polymers


#: Which cascade stage a candidate must reach to enter the conditioning report.
STAGE_FLAGS: dict[str, str] = {
    "sv": "passed_sv",
    "tsd": "passed_tsd",
    "dd": "passed_dd",
    "pv": "passed_pv",
}


def scored_candidates(
    report: GenerationReport, stage: str = "pv", source: str = "raw"
) -> tuple[list[int], list[str]]:
    """Select the candidates to score, and remember which row each came from.

    :attr:`GenerationReport.screened_psmiles` holds the PV survivors in input
    order but drops the association with their rows, which is exactly what a
    per-row target needs. The per-candidate ``records`` preserve it, and they
    also allow the population and the predictor's input form to be chosen
    deliberately instead of inherited by accident.

    **Stage** moves the denominator and therefore every rate in the conditioning
    report: PV additionally requires exactly two ``[At]`` termini of valency 1,
    which a large fraction of candidates fails, and the survivors are not a
    random sample of the rest.

    **Source** decides what string the predictor actually sees, and it is not a
    cosmetic choice. ``evaluate_generation`` hands its injected predictor the
    *canonical PSMILES*, which this package re-encodes to PSELFIES; that round
    trip rewrites the SELFIES phase for almost every candidate (measured on the
    real Tg run: 661 of 665 strings changed) and shifts the measured conditioning
    MAE by roughly 8 K. ``"raw"`` scores what the model actually emitted, which
    is what an RLVR reward must do -- the policy is credited for the string it
    produced, not for a normalised rewrite of it.
    # [AMBIGUITY] The paper never states which form its property predictor was
    # fed when scoring generated candidates.

    Args:
        report: A generation evaluation.
        stage: One of ``"sv"``, ``"tsd"``, ``"dd"``, ``"pv"``. ``"pv"`` (the
            full screen) is the population the paper's TP metric is defined over.
        source: ``"raw"`` for the generated PSELFIES exactly as emitted, or
            ``"canonical"`` for the canonical PSMILES, which reproduces what
            ``evaluate_generation`` feeds its predictor internally.

    Returns:
        ``(row_indices, strings)``, positionally aligned.

    Raises:
        ValueError: If ``stage`` or ``source`` is not recognised.
    """
    flag = STAGE_FLAGS.get(stage)
    if flag is None:
        raise ValueError(f"stage must be one of {sorted(STAGE_FLAGS)}, got {stage!r}")
    if source not in ("raw", "canonical"):
        raise ValueError(f"source must be 'raw' or 'canonical', got {source!r}")

    rows: list[int] = []
    strings: list[str] = []
    for index, record in enumerate(report.records):
        if not getattr(record, flag) or record.canonical_psmiles is None:
            continue
        rows.append(index)
        strings.append(
            record.raw_pselfies if source == "raw" else record.canonical_psmiles
        )
    return rows, strings


def _config_dataset_fingerprint(config: Any) -> str | None:
    """Extract a comparable dataset identity from a resolved run config."""
    if not isinstance(config, dict):
        return None
    data = config.get("data")
    if not isinstance(data, dict):
        return None
    csv_path = data.get("csv_path")
    if not isinstance(csv_path, str) or not csv_path.strip():
        return None
    return str(_resolve(Path(csv_path))).replace("\\", "/").lower()


def _generator_fingerprint(generations_path: Path) -> str | None:
    """Read the generating run's dataset identity from its run directory."""
    run_dir = generations_path.parent
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        try:
            return _config_dataset_fingerprint(load_config(config_path))
        except Exception:
            return None
    return None


def _predictor_fingerprint(checkpoint_path: Path) -> str | None:
    """Read a predictor checkpoint's dataset identity from its stored config."""
    try:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception:
        return None
    return _config_dataset_fingerprint(payload.get("config"))


def assess_circularity(
    generations_path: Path, checkpoints: list[Path]
) -> dict[str, Any]:
    """Decide whether the predictor and the generator share training data.

    A learned predictor scoring a generator that was fine-tuned on the same
    corpus is partly self-referential: both absorbed the same quirks, so a
    candidate exploiting a shared blind spot is scored as a success. The warning
    is emitted unconditionally when provenance cannot be established, because
    silence would read as "checked, and fine".

    Args:
        generations_path: The generations file; its run directory carries the
            generator's config.
        checkpoints: Predictor checkpoints, whose stored configs carry theirs.

    Returns:
        A JSON-serialisable dict with ``status``, ``warning``, and the
        fingerprints compared. ``status`` is one of ``"shared_dataset"``,
        ``"distinct_datasets"``, ``"unknown_provenance"``, or ``"not_applicable"``
        when no predictor was used at all.
    """
    if not checkpoints:
        return {"status": "not_applicable", "warning": None,
                "generator_dataset": None, "predictor_datasets": []}

    generator = _generator_fingerprint(generations_path)
    predictors = [_predictor_fingerprint(path) for path in checkpoints]
    known = [entry for entry in predictors if entry is not None]

    if generator is None or not known:
        status = "unknown_provenance"
    elif any(entry == generator for entry in known):
        status = "shared_dataset"
    else:
        status = "distinct_datasets"

    return {
        "status": status,
        "warning": _CIRCULARITY_TEXT[status],
        "generator_dataset": generator,
        "predictor_datasets": predictors,
    }


def build_predictor(
    checkpoints: list[Path],
    tokenizer_path: Path | None,
    *,
    device: str,
    batch_size: int,
    num_beams: int,
) -> tuple[Any, EnsemblePropertyPredictor | None]:
    """Build the injected predictor from one or more checkpoints.

    Args:
        checkpoints: One checkpoint for a single predictor, several for an
            ensemble.
        tokenizer_path: Tokenizer artifact, or ``None`` to use the path each
            checkpoint recorded. Every checkpoint's recorded hash is verified
            against whichever tokenizer is loaded.
        device: Torch device string.
        batch_size: Decode batch size.
        num_beams: Beam width. # [PAPER] 4.

    Returns:
        ``(injectable, ensemble_or_None)``. The injectable is wrapped in a
        :class:`polyt5.inference.CachedPredictor` so that re-scoring the
        screened set for the conditioning report costs nothing.
    """
    kwargs = {"batch_size": batch_size, "num_beams": num_beams, "property_name": "Tg"}
    if len(checkpoints) == 1:
        predictor = PolyT5PropertyPredictor.from_checkpoint(
            checkpoints[0], tokenizer_path, device=device, **kwargs
        )
        return CachedPredictor(predictor), None
    ensemble = EnsemblePropertyPredictor.from_checkpoints(
        checkpoints, tokenizer_path, device=device, **kwargs
    )
    return CachedPredictor(ensemble), ensemble


def choose_mode(targets: list[float | None], explicit_target: float | None) -> str:
    """Decide between per-row and fixed-target scoring.

    Args:
        targets: Per-row conditioning targets, ``None`` where absent.
        explicit_target: The value passed to ``--target``, or ``None``.

    Returns:
        :data:`MODE_FIXED` when ``--target`` was given or no row carries a
        target; :data:`MODE_PER_ROW` otherwise.
    """
    if explicit_target is not None:
        return MODE_FIXED
    return MODE_PER_ROW if any(target is not None for target in targets) else MODE_FIXED


def format_conditioning_summary(
    report: ConditioningReport, *, stage: str = "pv", source: str = "raw", unit: str = "K"
) -> str:
    """Render a conditioning report as a readable block.

    Args:
        report: The report to render.
        stage: The cascade stage the scored population was drawn from, named in
            the header so the denominator is never implicit.
        source: Which string form the predictor was fed, named for the same
            reason -- it moves the numbers.
        unit: Unit label for the property.

    Returns:
        A multi-line string. The caller prints it; this function does not.
    """
    input_label = {
        "raw": "raw generated PSELFIES",
        "canonical": "canonical PSMILES (re-encoded)",
    }.get(source, source)
    lines = [
        "Conditioning accuracy (each candidate vs ITS OWN target)",
        f"population: {stage.upper()} survivors   predictor input: {input_label}",
        "-" * 58,
        f"{'scored candidates':<34}{report.n_scored:>10}",
        f"{'missing target (skipped)':<34}{report.n_missing_target:>10}",
        f"{'no prediction (skipped)':<34}{report.n_non_numeric_prediction:>10}",
    ]
    if report.n_scored == 0:
        lines.append("nothing could be scored — every statistic below is n/a")
        return "\n".join(lines)

    lines += [
        f"{'MAE vs own target':<34}{report.mae:>10.1f} {unit}",
        f"{'median absolute error':<34}{report.median_abs_error:>10.1f} {unit}",
        f"{'RMSE':<34}{report.rmse:>10.1f} {unit}",
        f"{'mean signed bias':<34}{report.mean_signed_bias:>+10.1f} {unit}",
        "",
    ]
    for tolerance in report.tolerances:
        rate = report.within_tolerance[f"{tolerance:g}"]
        lines.append(f"{'within +/- ' + f'{tolerance:g} ' + unit:<34}{rate * 100:>9.1f}%")
    lines += [
        "",
        f"{'target range in file':<34}"
        f"{report.target_min:>10.1f} .. {report.target_max:.1f} {unit}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)

    generations_path = _resolve(args.generations)
    if not generations_path.is_file():
        print(f"ERROR: generations file not found: {generations_path}", file=sys.stderr)
        return 1
    candidates, row_targets = read_generations(generations_path, limit=args.limit)
    if not candidates:
        print(f"ERROR: no generations in {generations_path}", file=sys.stderr)
        return 1

    mode = choose_mode(row_targets, args.target)
    n_with_target = sum(1 for target in row_targets if target is not None)
    present = [target for target in row_targets if target is not None]
    print(f"generations:  {generations_path}  ({len(candidates)} candidates)")
    if present:
        print(f"row targets:  {n_with_target}/{len(candidates)} rows carry target_property "
              f"({min(present):.1f} .. {max(present):.1f} K)")
    if mode == MODE_PER_ROW:
        print("MODE:         PER-ROW — each candidate is scored against its own "
              "conditioning target.")
        print("              The paper's fixed-window TP is NOT reported: this file spans a "
              "range of\n              targets, so one window would score candidates on a "
              "request nobody made.\n              Pass --target to force the fixed-window "
              "measurement anyway.")
    else:
        fixed_target = args.target if args.target is not None else 500.0
        print(f"MODE:         FIXED — every candidate scored against "
              f"{fixed_target:g} +/- {args.tolerance:g} K (the paper's screen).")
        if args.target is None:
            print("              No row carried target_property, so the paper's default "
                  "500 K window is used.")

    training_index = None
    n_reference = 0
    if args.training_corpus is not None:
        corpus_path = _resolve(args.training_corpus)
        if not corpus_path.is_file():
            print(f"ERROR: training corpus not found: {corpus_path}", file=sys.stderr)
            return 1
        reference = read_training_pselfies(corpus_path)
        training_index = NoveltyIndex.from_pselfies(reference)
        n_reference = len(training_index)
        print(f"novelty index: {corpus_path}  ({n_reference} distinct polymers)")
    else:
        print("WARNING: no --training-corpus; TSD and novelty are measured against an "
              "EMPTY reference set, so every candidate counts as novel.", file=sys.stderr)

    predictor = None
    ensemble = None
    checkpoints = [_resolve(path) for path in (args.predictor_checkpoint or [])]
    tokenizer_path = _resolve(args.tokenizer) if args.tokenizer is not None else None
    if checkpoints:
        try:
            predictor, ensemble = build_predictor(
                checkpoints,
                tokenizer_path,
                device=args.device,
                batch_size=args.batch_size,
                num_beams=args.num_beams,
            )
        except (ValueError, FileNotFoundError) as error:
            # A tokenizer mismatch or a checkpoint with no recorded vocabulary.
            # Reported as a message, not a traceback: it is a usage problem.
            print(f"ERROR: could not build the property predictor.\n{error}", file=sys.stderr)
            return 1
        for path in checkpoints:
            print(f"predictor:    {path}")
    else:
        print(_NO_PREDICTOR_BANNER, file=sys.stderr)

    circularity = assess_circularity(generations_path, checkpoints)
    if circularity["warning"] is not None:
        print()
        print(f"WARNING: {circularity['warning']}", file=sys.stderr)

    # In per-row mode no fixed window is passed, so target_property_rate comes
    # back None -- honestly "not measured" rather than an answer to the wrong
    # question. The conditioning report below is the measurement that applies.
    fixed_target = args.target if mode == MODE_FIXED else None
    if mode == MODE_FIXED and fixed_target is None:
        fixed_target = 500.0  # [PAPER] the screen's centre
    report = evaluate_generation(
        candidates,
        training_index=training_index,
        target_property=fixed_target,
        tolerance=args.tolerance,
        property_predictor=predictor,
        compute_sa=not args.no_sa,
    )

    print()
    print(format_console_summary(report))
    print()

    window = f"TP (within {fixed_target:g} +/- {args.tolerance:g} K)" if fixed_target \
        else "TP (fixed window)"
    tp_status: str | None = None
    if mode == MODE_PER_ROW:
        tp_status = (
            "per-row mode — a single fixed window does not describe a file whose rows "
            "were conditioned on different targets (see the conditioning accuracy below)"
        )
    elif not checkpoints:
        tp_status = "no property predictor was supplied"
    elif report.target_property_rate is None:
        # The predictor ran but nothing it emitted parsed as a number. That is a
        # very different failure from "no predictor", and must not be reported
        # as the same thing.
        tp_status = (
            f"the predictor ran but returned no finite value for any of the "
            f"{len(report.screened_psmiles)} screened candidates — its decodes were "
            "non-numeric (an undertrained or mismatched predictor checkpoint)"
        )
    if tp_status is not None:
        print(f"{window}: NOT MEASURED — {tp_status}.")
    else:
        print(f"{window}: {report.target_property_rate * 100:.1f}%  "
              f"({report.n_property_values} of {len(report.screened_psmiles)} screened "
              "candidates received a prediction)")

    # ------------------------------------------------- conditioning accuracy
    conditioning: ConditioningReport | None = None
    scored_rows, scored_strings = scored_candidates(
        report, args.score_stage, args.score_input
    )
    if predictor is not None and scored_rows and n_with_target:
        # With --score-input canonical these are the exact strings
        # evaluate_generation already scored, so the cache serves them for free.
        predicted = list(predictor(scored_strings))
        conditioning = conditioning_report(
            predicted,
            [row_targets[index] for index in scored_rows],
            tolerances=args.tolerances,
        )
        print()
        print(format_conditioning_summary(
            conditioning, stage=args.score_stage, source=args.score_input
        ))

    payload: dict[str, Any] = {
        "generations": str(generations_path),
        "n_generations": len(candidates),
        "mode": mode,
        "n_rows_with_target": n_with_target,
        "row_target_min": min(present) if present else None,
        "row_target_max": max(present) if present else None,
        "training_corpus": str(_resolve(args.training_corpus))
        if args.training_corpus else None,
        "n_reference_polymers": n_reference,
        "predictor_checkpoints": [str(path) for path in checkpoints],
        "predictor_available": bool(checkpoints),
        "tokenizer": str(tokenizer_path) if tokenizer_path is not None else None,
        "target_property": fixed_target,
        "tolerance": args.tolerance,
        "tolerances": list(args.tolerances),
        "score_stage": args.score_stage,
        "score_input": args.score_input,
        "n_score_stage_candidates": len(scored_rows),
        "num_beams": args.num_beams,
        "circularity": circularity,
        "report": report.to_dict(),
        "note_property_stats_input": (
            "report.property_* and report.target_property_rate are computed inside "
            "polyt5.evaluation, which always feeds its predictor the CANONICAL PSMILES of "
            "the PV survivors. The conditioning block uses score_stage/score_input instead, "
            "so the two populations and string forms can differ by design."
        ) if predictor is not None else None,
        "conditioning": conditioning.to_dict() if conditioning is not None else None,
    }
    if tp_status is not None:
        payload["tp_not_measured_because"] = tp_status
    if conditioning is None and predictor is not None and n_with_target:
        payload["conditioning_not_measured_because"] = (
            f"no candidate reached the {args.score_stage.upper()} stage, so there was "
            "nothing to score"
        )

    if ensemble is not None and report.screened_psmiles:
        spreads = [
            std for _, std, n in ensemble.predict_with_uncertainty(report.screened_psmiles)
            if n > 1
        ]
        payload["ensemble"] = {
            "n_members": len(ensemble),
            "n_with_disagreement": len(spreads),
            "mean_member_std": (sum(spreads) / len(spreads)) if spreads else None,
            "max_member_std": max(spreads) if spreads else None,
        }
        print(f"\nensemble disagreement (mean std over {len(spreads)} candidates): "
              f"{payload['ensemble']['mean_member_std']}")

    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / EVALUATION_FILENAME
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
