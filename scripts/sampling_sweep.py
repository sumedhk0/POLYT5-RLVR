"""Run the sampling-hyperparameter sweep -- "Arm B" of the RLVR comparison.

Arm A is supervised polyT5 with default sampling, Arm B is supervised polyT5
with sampling tuned on validation (this script), Arm C is supervised polyT5
plus GRPO/RLVR. Arm B exists because much of what RL appears to buy can often
be had by turning the temperature down, so the eventual comparison is only
honest against a properly tuned baseline.

It is simultaneously a reproduction of the paper's own sweep (Sahu et al., npj
Artificial Intelligence 2026): ``top_p`` in {0.75, 0.95}, temperature
0.1 -> 2.0 in steps of 0.1, epochs 1 -> 15, 10,000 polymers per configuration,
1,800 configurations in total. The paper publishes those per-configuration
counts only as unlabeled heatmaps; this script writes the numeric table.

Cost discipline: the full paper grid is not runnable on one laptop GPU, so the
script prints the planned configuration count and a measured runtime estimate
BEFORE committing to the run, refuses any grid larger than ``--max-configs``
without ``--force``, and appends every finished configuration to ``sweep.jsonl``
so an interrupted run keeps its data.

Example:
    python scripts/sampling_sweep.py --grid default --targets 500 --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from polyt5.chemistry import NoveltyIndex, psmiles_to_pselfies  # noqa: E402
from polyt5.data.prepare import parse_property_value  # noqa: E402
from polyt5.evaluation.sweep import (  # noqa: E402
    DEFAULT_GRID,
    OBJECTIVES,
    PAPER_CONFIGURATIONS,
    PAPER_GRID,
    PAPER_SAMPLES_PER_CONFIG,
    SweepPoint,
    SweepResult,
    append_result_jsonl,
    read_results_jsonl,
    run_sweep_point,
    select_best,
    sweep_grid,
    sweep_to_dataframe,
    sweep_to_markdown,
)
from polyt5.generation import BeamSearchConfig, beam_search  # noqa: E402
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import (  # noqa: E402
    # noqa: E402,
    RunDirectory,
    describe_device,
    get_logger,
    resolve_under,
    seed_everything,
    select_device,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--generation-checkpoint",
        default="results/finetune_tg_generation/checkpoints/best.pt",
        help="Tg -> PSELFIES checkpoint to sweep.",
    )
    parser.add_argument(
        "--predictor-checkpoint",
        default=None,
        help=(
            "Optional PSELFIES -> Tg checkpoint. Without it the TP and property columns "
            "stay empty; they are never fabricated."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default="artifacts/tokenizer/polyt5_vocab.json",
        help="Tokenizer artifact (falls back to the built-in vocabulary).",
    )
    parser.add_argument(
        "--training-corpus",
        default="data/processed/tg/generation/train.jsonl",
        help="Training polymers for the TSD novelty index (.jsonl or .txt of PSELFIES).",
    )
    parser.add_argument(
        "--max-corpus-rows",
        type=int,
        default=0,
        help="Cap on training rows indexed (0 = all). Indexing is RDKit-bound.",
    )
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[300.0, 400.0, 500.0],
        help="Property values to condition on; prompts cycle through them round-robin.",
    )
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Candidates per configuration (the paper uses 10000).")
    parser.add_argument("--grid", choices=["paper", "default", "custom"], default="default",
                        help="paper = the full 40-point temperature x top_p plane.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None,
                        help="Temperature axis for --grid custom.")
    parser.add_argument("--top-ps", type=float, nargs="+", default=None,
                        help="top_p axis for --grid custom.")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="pv_rate",
                        help="Objective that picks the winning configuration.")
    parser.add_argument("--target-property", type=float, default=None,
                        help="Fixed TP window centre; default centres it on each prompt target.")
    parser.add_argument("--tolerance", type=float, default=50.0,
                        help="TP window half-width (the paper uses 50 K).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/sampling_sweep", help="Run directory.")
    parser.add_argument("--max-configs", type=int, default=64,
                        help="Hard safety cap on grid size; exceeding it requires --force.")
    parser.add_argument("--force", action="store_true",
                        help="Run a grid larger than --max-configs anyway.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configurations already present in sweep.jsonl.")
    parser.add_argument("--batch-size", type=int, default=64, help="Prompts per decoding batch.")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Maximum generated tokens (the paper uses 200).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repository root when it is not absolute."""
    return resolve_under(REPO_ROOT, path)


def load_generation_model(
    checkpoint: Path, tokenizer: PolyT5Tokenizer, device: str, logger
) -> PolyT5ForConditionalGeneration:
    """Rebuild the generation model from a checkpoint.

    Args:
        checkpoint: Path to a checkpoint written by ``polyt5.training``.
        tokenizer: The tokenizer, used only to warn about artifact mismatches.
        device: Torch device string.
        logger: Where to report provenance.

    Returns:
        The model in eval mode on ``device``.
    """
    payload = load_checkpoint(checkpoint, map_location="cpu")
    config = PolyT5Config.from_dict(payload["model_config"])
    model = PolyT5ForConditionalGeneration(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()

    stored_sha = payload.get("tokenizer_sha256")
    if stored_sha and stored_sha != tokenizer.sha256:
        logger.warning(
            "tokenizer sha mismatch: checkpoint=%s tokenizer=%s -- decoded strings may be garbage",
            str(stored_sha)[:16], tokenizer.sha256[:16],
        )
    logger.info(
        "generation model: %s (epoch=%s step=%s params=%.1fM)",
        checkpoint, payload.get("epoch"), payload.get("global_step"),
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model


class CheckpointPropertyPredictor:
    """A property predictor loaded straight from a checkpoint.

    This deliberately does NOT use ``polyt5.inference``: that module is owned by
    another track and is being written concurrently. The sweep only ever needs
    the injected-callable contract
    ``Callable[[Sequence[str]], Sequence[float]]``, so a local loader keeps this
    script independent -- and keeps ``polyt5.evaluation.sweep`` free of any
    model import at all.

    Decoding follows the paper's PROPERTY-PREDICTION regime: beam search with
    width 4, not sampling.
    """

    def __init__(
        self,
        checkpoint: Path,
        tokenizer: PolyT5Tokenizer,
        device: str,
        *,
        batch_size: int = 64,
        max_target_length: int = 32,
        num_beams: int = 4,
    ) -> None:
        """Load the regressor.

        Args:
            checkpoint: PSELFIES -> property checkpoint.
            tokenizer: The polyT5 tokenizer.
            device: Torch device string.
            batch_size: Rows per beam-search batch.
            max_target_length: Maximum decoded characters-as-tokens for a number.
            num_beams: Beam width (the paper uses 4).
        """
        payload = load_checkpoint(checkpoint, map_location="cpu")
        config = PolyT5Config.from_dict(payload["model_config"])
        self._model = PolyT5ForConditionalGeneration(config)
        self._model.load_state_dict(payload["model_state"])
        self._model.to(device).eval()
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._config = BeamSearchConfig(
            num_beams=num_beams,
            max_length=max_target_length,
            length_penalty=1.0,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            decoder_start_token_id=tokenizer.decoder_start_token_id,
        )

    @torch.no_grad()
    def __call__(self, psmiles: Sequence[str]) -> list[float]:
        """Predict the property for each PSMILES.

        Args:
            psmiles: Canonical PSMILES of screened candidates.

        Returns:
            One float per input, ALIGNED with the input: a candidate that
            cannot be re-encoded to PSELFIES, or whose decoded string is not a
            number, becomes ``nan`` rather than being dropped, so the caller's
            per-candidate bookkeeping stays valid.
        """
        values = [math.nan] * len(psmiles)
        encodable: list[tuple[int, str]] = []
        for index, entry in enumerate(psmiles):
            try:
                pselfies = psmiles_to_pselfies(entry)
            except Exception:
                pselfies = None
            if pselfies:
                encodable.append((index, pselfies))

        for start in range(0, len(encodable), self._batch_size):
            chunk = encodable[start : start + self._batch_size]
            encoded = self._tokenizer.batch_encode(
                [text for _, text in chunk], add_eos=True, max_length=200,
                padding=True, truncation=True,
            )
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=self._device)
            mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=self._device)
            output = beam_search(self._model, input_ids, mask, config=self._config)
            decoded = self._tokenizer.batch_decode(
                output.sequences.tolist(), skip_special_tokens=True
            )
            for (index, _), text in zip(chunk, decoded, strict=False):
                parsed = parse_property_value(text)
                if parsed is not None:
                    values[index] = parsed
        return values


def load_training_index(path: Path, limit: int, logger) -> NoveltyIndex | None:
    """Build the TSD reference index from a training corpus.

    Args:
        path: ``.jsonl`` with a ``target`` (or ``source``) PSELFIES field, or a
            ``.txt`` with one PSELFIES per line.
        limit: Maximum rows to index (``0`` = all).
        logger: Where to report what was indexed.

    Returns:
        A :class:`polyt5.chemistry.NoveltyIndex`, or ``None`` when the corpus is
        missing -- in which case TSD is a no-op and the run says so.

    Note:
        # [AMBIGUITY] The paper says TSD removes candidates "already in the
        # training set" without saying whether that is the pre-training corpus
        # (PI1M, ~890k polymers) or the fine-tuning set. The fine-tuning set is
        # used here: it is the data that taught the model this conditional task,
        # and indexing PI1M costs minutes of RDKit time per run.
    """
    if not path.is_file():
        logger.warning("training corpus %s not found -- TSD will be a no-op", path)
        return None

    entries: list[str] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("target") or row.get("source") or ""
                if value.startswith("["):
                    entries.append(value)
                if limit and len(entries) >= limit:
                    break
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    entries.append(stripped)
                if limit and len(entries) >= limit:
                    break

    index = NoveltyIndex.from_pselfies(entries)
    logger.info("TSD index: %d distinct polymers from %d rows of %s", len(index),
                len(entries), path)
    return index


def build_grid(args: argparse.Namespace) -> list[SweepPoint]:
    """Build the requested grid.

    Args:
        args: Parsed command line.

    Returns:
        The sweep points to run.

    Raises:
        ValueError: If ``--grid custom`` is chosen without both axes.
    """
    if args.grid == "paper":
        return list(PAPER_GRID)
    if args.grid == "default":
        return list(DEFAULT_GRID)
    if not args.temperatures or not args.top_ps:
        raise ValueError("--grid custom requires both --temperatures and --top-ps.")
    return sweep_grid(temperatures=args.temperatures, top_ps=args.top_ps)


def format_duration(seconds: float) -> str:
    """Render a duration as a compact human string."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def report_paper_cost(seconds_per_sample: float, logger) -> dict[str, float]:
    """Extrapolate the cost of the paper's full sweep from a measured rate.

    Args:
        seconds_per_sample: Measured wall-clock seconds per generated candidate.
        logger: Where to report the projection.

    Returns:
        A dict with the projected totals, for the run manifest.
    """
    full_samples = PAPER_CONFIGURATIONS * PAPER_SAMPLES_PER_CONFIG
    plane_samples = len(PAPER_GRID) * PAPER_SAMPLES_PER_CONFIG
    projection = {
        "seconds_per_sample": seconds_per_sample,
        "paper_configurations": PAPER_CONFIGURATIONS,
        "paper_samples_total": full_samples,
        "paper_full_sweep_seconds": full_samples * seconds_per_sample,
        "paper_one_checkpoint_plane_seconds": plane_samples * seconds_per_sample,
    }
    logger.info(
        "PAPER COST: full sweep = %d configs x %d samples = %d polymers ~ %s on this hardware "
        "(NOT run); one checkpoint's 40-point plane at 10k samples ~ %s",
        PAPER_CONFIGURATIONS, PAPER_SAMPLES_PER_CONFIG, full_samples,
        format_duration(projection["paper_full_sweep_seconds"]),
        format_duration(projection["paper_one_checkpoint_plane_seconds"]),
    )
    return projection


def main(argv: list[str] | None = None) -> int:
    """Run the sweep."""
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.out)
    run_dir = RunDirectory.create(_resolve(out.parent), out.name)
    logger = get_logger("polyt5.sampling_sweep", log_file=run_dir.logs / "sampling_sweep.log")
    device = select_device(args.device)
    logger.info("device: %s", json.dumps(describe_device(device).to_dict()))

    # ------------------------------------------------------------------ grid
    try:
        grid = build_grid(args)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    jsonl_path = run_dir.root / "sweep.jsonl"
    done: set[tuple[float, float, int | None]] = set()
    if args.resume:
        done = {
            (row["temperature"], row["top_p"], row.get("epoch"))
            for row in read_results_jsonl(jsonl_path)
        }
        grid = [p for p in grid if (p.temperature, p.top_p, p.epoch) not in done]
        logger.info("resuming: %d configurations already done, %d remain", len(done), len(grid))
    elif jsonl_path.exists():
        jsonl_path.unlink()

    total_samples = len(grid) * args.n_samples
    logger.info(
        "PLANNED: %d configurations x %d samples = %d polymers (targets=%s, grid=%s)",
        len(grid), args.n_samples, total_samples,
        ", ".join(f"{t:g}" for t in args.targets), args.grid,
    )
    if not grid and not (args.resume and done):
        logger.error("nothing to run")
        return 1
    if len(grid) > args.max_configs and not args.force:
        logger.error(
            "REFUSING: %d configurations exceeds --max-configs %d. Re-run with --force if you "
            "really mean it, or shrink the grid. For scale: the paper's full sweep is %d "
            "configurations x %d samples.",
            len(grid), args.max_configs, PAPER_CONFIGURATIONS, PAPER_SAMPLES_PER_CONFIG,
        )
        return 3

    results: list[SweepResult] = []
    projection: dict[str, float] = {}

    if grid:
        # -------------------------------------------------------- components
        tokenizer_path = _resolve(args.tokenizer)
        tokenizer = (
            PolyT5Tokenizer.from_file(tokenizer_path)
            if tokenizer_path.is_file()
            else PolyT5Tokenizer.default()
        )
        logger.info("tokenizer: %d tokens sha=%s", tokenizer.vocab_size, tokenizer.sha256[:16])

        model = load_generation_model(
            _resolve(args.generation_checkpoint), tokenizer, device, logger
        )
        training_index = load_training_index(
            _resolve(args.training_corpus), args.max_corpus_rows, logger
        )

        predictor = None
        if args.predictor_checkpoint:
            predictor = CheckpointPropertyPredictor(
                _resolve(args.predictor_checkpoint), tokenizer, device, batch_size=args.batch_size
            )
            logger.info("property predictor: %s (beam search, width 4)", args.predictor_checkpoint)
        else:
            logger.warning(
                "NO PREDICTOR CHECKPOINT: the tp_rate, property_mean and property_mae columns "
                "will be empty. They are left empty on purpose -- pass --predictor-checkpoint "
                "results/finetune_tg_prediction/checkpoints/best.pt to fill them."
            )

        def run(point: SweepPoint, seed_offset: int) -> SweepResult:
            return run_sweep_point(
                model,
                tokenizer,
                point=point,
                targets=args.targets,
                n_samples=args.n_samples,
                training_index=training_index,
                property_predictor=predictor,
                target_property=args.target_property,
                tolerance=args.tolerance,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed + seed_offset,
                max_length=args.max_length,
            )

        # ----------------------------------------------------------- warm-up
        logger.info("warm-up: timing configuration 1/%d before committing to the rest", len(grid))
        first = run(grid[0], 0)
        append_result_jsonl(jsonl_path, first)
        results.append(first)

        per_point = first.seconds
        logger.info(
            "WARM-UP: %.1fs for %d samples (%.3f s/polymer). ESTIMATED TOTAL for %d "
            "configurations: %s. Continuing with the remaining %d.",
            per_point, args.n_samples, per_point / max(args.n_samples, 1), len(grid),
            format_duration(per_point * len(grid)), len(grid) - 1,
        )
        projection = report_paper_cost(per_point / max(args.n_samples, 1), logger)

        # ------------------------------------------------------------- sweep
        for index, point in enumerate(grid[1:], start=1):
            result = run(point, index)
            append_result_jsonl(jsonl_path, result)
            results.append(result)
            elapsed = sum(r.seconds for r in results)
            remaining = (len(grid) - len(results)) * (elapsed / len(results))
            logger.info(
                "[%2d/%2d] T=%.2f top_p=%.2f -> pv=%.4f (n_pv=%d) dup=%.4f sr=%.4f %s eta=%s",
                len(results), len(grid), point.temperature, point.top_p,
                result.counts["pv_rate"], result.counts["n_pv"], result.duplicate_rate,
                result.sr_rate,
                f"tp={result.tp_rate:.4f}" if result.tp_rate is not None else "tp=n/a",
                format_duration(remaining),
            )
    else:
        logger.info("every requested configuration is already in %s; re-tabulating only",
                    jsonl_path)

    if args.resume:
        # sweep.jsonl is the checkpoint, not just a log: reload everything,
        # including configurations this invocation skipped, so the tables and
        # the winner are computed over the whole sweep.
        results = [SweepResult.from_dict(row) for row in read_results_jsonl(jsonl_path)]

    # --------------------------------------------------------------- outputs
    table = sweep_to_markdown(results, sort_by="pv_rate")
    (run_dir.root / "sweep.md").write_text(table + "\n", encoding="utf-8")
    frame = sweep_to_dataframe(results)
    frame.to_csv(run_dir.root / "sweep.csv", index=False)

    best = select_best(results, objective=args.objective)
    best_payload = {
        "objective": args.objective,
        "objective_value": OBJECTIVES[args.objective](best),
        "best": best.to_dict(),
        "n_configurations": len(results),
        "n_samples_per_configuration": args.n_samples,
        "targets": list(args.targets),
        "seed": args.seed,
        "generation_checkpoint": str(args.generation_checkpoint),
        "predictor_checkpoint": args.predictor_checkpoint,
        "paper_cost_projection": projection,
        "paper_reported_optima": {
            "small": {"epochs": 14, "temperature": 0.9, "top_p": 0.75},
            "medium": {"epochs": 6, "temperature": 1.1, "top_p": 0.75},
            "large": {"epochs": 8, "temperature": 1.1, "top_p": 0.95},
        },
    }
    (run_dir.root / "best.json").write_text(
        json.dumps(best_payload, indent=2, default=str), encoding="utf-8"
    )
    run_dir.write_manifest({"script": "sampling_sweep", "args": vars(args)})

    print(table)
    print()
    print(f"objective={args.objective} selected T={best.point.temperature:g} "
          f"top_p={best.point.top_p:g} "
          f"(value={best_payload['objective_value']:.4f}) "
          f"[paper's polyT5-small optimum: T=0.9, top_p=0.75]")
    print("top 5 by the selected objective:")
    ranked = sorted(
        results,
        key=lambda r: (OBJECTIVES[args.objective](r) is None,
                       -(OBJECTIVES[args.objective](r) or 0.0)),
    )
    for rank, result in enumerate(ranked[:5], start=1):
        value = OBJECTIVES[args.objective](result)
        rendered = "n/a" if value is None else f"{value:.4f}"
        # pv_rate is only worth repeating when it is not the objective itself.
        context = "" if args.objective == "pv_rate" else f"pv_rate={result.counts['pv_rate']:.4f} "
        print(
            f"  {rank}. T={result.point.temperature:<5g} top_p={result.point.top_p:<5g} "
            f"{args.objective}={rendered} {context}"
            f"n_pv={result.counts['n_pv']} dup={result.duplicate_rate:.4f}"
        )
    print(f"\nwrote {run_dir.root}/sweep.jsonl, sweep.csv, sweep.md, best.json")
    logger.info("objective %r selected %s", args.objective, best.point)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
