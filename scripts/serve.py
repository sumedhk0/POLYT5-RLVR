#!/usr/bin/env python
"""Serve the local polyT5 demo app.

Our counterpart to Figure S13 of Sahu et al.: a single-page console over the two
fine-tuned checkpoints in this repository. It binds to 127.0.0.1 by default,
makes no outside network calls, and keeps no state between requests.

Defaults point at the checkpoints this repository trains. A default that is not
on disk does **not** stop the server: the corresponding feature is disabled and
``/api/health`` says so, which is more useful than a crash at 3 a.m. two hours
into a training run.

Examples:
    python scripts/serve.py
    python scripts/serve.py --port 8123 --device cpu
    python scripts/serve.py --generation-checkpoint results/other/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.app.server import create_app  # noqa: E402
from polyt5.utils import get_logger, resolve_under  # noqa: E402

#: Defaults, relative to the repository root.
DEFAULT_GENERATION = "results/finetune_tg_generation/checkpoints/best.pt"
DEFAULT_PREDICTION = "results/finetune_tg_prediction/checkpoints/best.pt"
DEFAULT_TOKENIZER = "artifacts/tokenizer/polyt5_vocab.json"
# The TSD filter asks "is this candidate already in the generation model's
# training set?", so the generation split is the right reference -- not PI1M,
# which the model only ever saw through span corruption.
DEFAULT_CORPUS = "data/processed/tg/generation/train.jsonl"


def _resolve(path: str | None) -> Path | None:
    """Resolve a possibly-relative path against the repository root.

    Args:
        path: A path string, or ``None``.

    Returns:
        The absolute path, or ``None`` when ``path`` is ``None``/empty.
    """
    if not path:
        return None
    return resolve_under(REPO_ROOT, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Serve the local polyT5 demo app (offline, 127.0.0.1 by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generation-checkpoint", default=DEFAULT_GENERATION)
    parser.add_argument("--prediction-checkpoint", default=DEFAULT_PREDICTION)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--training-corpus",
        default=DEFAULT_CORPUS,
        help="Corpus for the TSD filter. Omit with --no-training-corpus.",
    )
    parser.add_argument(
        "--no-training-corpus",
        action="store_true",
        help="Skip the training index; the TSD stage then becomes a no-op.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep it local.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Load the checkpoints at startup instead of on the first request.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes (development only; re-imports the app).",
    )
    parser.add_argument(
        "--log-level", default="info", choices=("critical", "error", "warning", "info", "debug")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Start the server.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        A process exit code (0 on a clean shutdown).
    """
    args = parse_args(argv)
    logger = get_logger("polyt5.serve")

    generation = _resolve(args.generation_checkpoint)
    prediction = _resolve(args.prediction_checkpoint)
    tokenizer = _resolve(args.tokenizer)
    corpus = None if args.no_training_corpus else _resolve(args.training_corpus)

    disabled: list[str] = []
    for label, path in (
        ("generation", generation),
        ("Tg prediction", prediction),
        ("tokenizer", tokenizer),
        ("training-set deduplication", corpus),
    ):
        if path is not None and not path.is_file():
            disabled.append(f"{label} ({path})")
    for note in disabled:
        logger.warning("not found, feature disabled: %s", note)

    if tokenizer is None or not tokenizer.is_file():
        logger.warning(
            "no tokenizer vocabulary: every model endpoint will refuse until one is "
            "supplied with --tokenizer"
        )

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn is a stated dependency here
        logger.error("uvicorn is not installed; cannot serve. Install it or use create_app().")
        return 1

    url = f"http://{args.host}:{args.port}/"
    logger.info("polyT5 demo app starting")
    logger.info("  generation checkpoint : %s", generation)
    logger.info("  prediction checkpoint : %s", prediction)
    logger.info("  tokenizer             : %s", tokenizer)
    logger.info("  training corpus (TSD) : %s", corpus)
    logger.info("  device                : %s", args.device)
    print(f"\nOpen {url} in a browser. Press Ctrl+C to stop.\n", flush=True)

    if args.reload:
        # uvicorn's reloader re-imports the app in a child process, so the
        # configuration has to travel through the environment rather than a
        # closure over an already-built object.
        os.environ["POLYT5_APP_GENERATION_CHECKPOINT"] = str(generation or "")
        os.environ["POLYT5_APP_PREDICTION_CHECKPOINT"] = str(prediction or "")
        os.environ["POLYT5_APP_TOKENIZER"] = str(tokenizer or "")
        os.environ["POLYT5_APP_TRAINING_CORPUS"] = str(corpus or "")
        os.environ["POLYT5_APP_DEVICE"] = args.device
        uvicorn.run(
            "polyt5.app.server:create_app_from_env",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
        return 0

    app = create_app(
        generation_checkpoint=generation,
        prediction_checkpoint=prediction,
        tokenizer_path=tokenizer,
        training_corpus=corpus,
        device=args.device,
        lazy=not args.eager,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
