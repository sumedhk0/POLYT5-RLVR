"""Check one Group A arm's generation quality against the frozen Arm B rates.

Spec section 6: generation is evaluated separately and must not regress. Any
configuration touching the shared encoder is sampled at ARM B's sampling point
(temperature 0.7, top_p 0.95 -- a different point makes the comparison
meaningless) and scored with the same cascade the frozen numbers came from.

TP (target-property rate) needs a property predictor, and whole-branch review
finding 3 is about WHICH one. The frozen Arm B TP of 58.78% was produced by
``scripts/evaluate_generations.py`` with EXTERNAL predictor checkpoints
(``--predictor-checkpoint``); using an arm's OWN regression head (when it has
one, e.g. A6) to score its OWN generations is the generator grading its own
output with the very component under test, a different instrument from the
one that produced 58.78% -- not a measurement of generation quality. So TP is
NEVER scored by the arm's own head here: pass ``--predictor-checkpoint`` (the
same external checkpoints used for the frozen number, repeatable for an
ensemble) to get a TP verdict at all; without it, TP is ``None`` for every
arm -- A5 and A6 alike -- and the script says so loudly rather than silently
falling back to a circular instrument.

Usage:
    python scripts/check_group_a_generation.py \
        --checkpoint results/group_a/A5/split_0_<fingerprint>/checkpoints/best.pt \
        --arm A5 --n-samples 1000 \
        --predictor-checkpoint results/finetune_tg_prediction/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.prepare import format_property_value  # noqa: E402
from polyt5.evaluation import evaluate_generation  # noqa: E402
from polyt5.evaluation.generation_regression import (  # noqa: E402
    check_generation_regression,
    load_arm_b_generation_baseline,
)
from polyt5.generation import GenerationConfig, generate  # noqa: E402
from polyt5.inference import EnsemblePropertyPredictor  # noqa: E402
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import get_logger, seed_everything, select_device  # noqa: E402

#: Arm B's tuned sampling point. A different point makes the row incomparable.
ARM_B_TEMPERATURE = 0.7
ARM_B_TOP_P = 0.95


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, action="append", default=None,
                        metavar="PATH",
                        help="External Tg-prediction checkpoint -- the SAME instrument "
                             "that produced the frozen Arm B TP of 58.78%%. Repeat for an "
                             "ensemble. Without this, TP is never computed: an arm's OWN "
                             "regression head (e.g. A6) is refused as a TP instrument, "
                             "since a checkpoint grading its own generations is circular.")
    parser.add_argument("--tokenizer", type=Path,
                        default=REPO_ROOT / "artifacts" / "tokenizer" / "polyt5_vocab.json")
    parser.add_argument("--frozen-baseline", type=Path,
                        default=REPO_ROOT / "artifacts" / "baseline" / "frozen_baseline.json")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--target-property", type=float, default=500.0)
    parser.add_argument("--tolerance", type=float, default=50.0,
                        help="TP acceptance half-window in Kelvin. [PAPER] 50.")
    parser.add_argument("--regression-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the check JSON (default: beside the checkpoint).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code; 2 marks a regression."""
    args = parse_args(argv)
    logger = get_logger("polyt5.check_group_a_generation")
    seed_everything(args.seed)
    device = select_device("auto")

    tokenizer = PolyT5Tokenizer.from_file(args.tokenizer)
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    heads = ((payload.get("config") or {}).get("group_a") or {}).get("heads") or {}
    backbone = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(payload["model_config"]))
    model = PolyT5MultiTask(backbone, MultiTaskConfig.from_dict(heads))
    model.load_state_dict(payload["model_state"])
    model = model.to(device)
    model.eval()

    prompt = format_property_value(args.target_property)
    generated: list[str] = []
    for start in range(0, args.n_samples, args.batch_size):
        size = min(args.batch_size, args.n_samples - start)
        encoded = tokenizer.batch_encode(
            [prompt] * size, add_eos=True, max_length=200, padding=True, truncation=True
        )
        with torch.no_grad():
            output = generate(
                model.backbone,
                torch.tensor(encoded["input_ids"], device=device),
                torch.tensor(encoded["attention_mask"], device=device),
                config=GenerationConfig(
                    max_length=200,
                    do_sample=True,
                    temperature=ARM_B_TEMPERATURE,
                    top_p=ARM_B_TOP_P,
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                    decoder_start_token_id=tokenizer.decoder_start_token_id,
                    seed=args.seed + start,
                ),
            )
        generated.extend(
            tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
        )
        logger.info("sampled %d/%d", len(generated), args.n_samples)

    predictor = None
    if args.predictor_checkpoint:
        predictor = EnsemblePropertyPredictor.from_checkpoints(
            args.predictor_checkpoint, device=device,
        )
    elif model.tg_head is not None:
        logger.warning(
            "arm %s has its own regression head, but no --predictor-checkpoint was given: "
            "TP will NOT be scored by the arm's own head -- that would be the generator "
            "grading its own output, a different instrument from the external predictor "
            "that produced the frozen Arm B TP of 58.78%%. Re-run with "
            "--predictor-checkpoint for a TP verdict.",
            args.arm,
        )
    report = evaluate_generation(
        generated,
        target_property=args.target_property,
        tolerance=args.tolerance,
        property_predictor=predictor,
    )
    baseline_pv_rate, baseline_tp_rate = load_arm_b_generation_baseline(args.frozen_baseline)
    check = check_generation_regression(
        report, arm=args.arm, baseline_pv_rate=baseline_pv_rate,
        baseline_tp_rate=baseline_tp_rate, tolerance=args.regression_tolerance,
    )

    destination = args.out or args.checkpoint.parent.parent / "generation_check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({
            "check": check.to_dict(),
            "generation": report.to_dict(),
            # Provenance: which instrument (if any) scored TP, so a reader
            # never has to guess whether this row is comparable to the
            # frozen 58.78% or was left as None on purpose.
            "predictor_checkpoints": [str(path) for path in (args.predictor_checkpoint or [])],
        }, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("%s: PV %.4f (baseline %.4f), TP %s (baseline %.4f) -> %s",
                args.arm, check.pv_rate, check.baseline_pv_rate,
                "n/a" if check.tp_rate is None else f"{check.tp_rate:.4f}",
                check.baseline_tp_rate, check.verdict)
    logger.info("wrote %s", destination)
    return 2 if check.regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
