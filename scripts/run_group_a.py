# scripts/run_group_a.py
"""Run the seven Group A configurations over the FROZEN five splits.

Spec section 5: seven configurations, each on the same five splits the frozen
baseline used, so every number is directly comparable to 28.6733 +/- 0.7591 K.
Individual ablations run AS WELL AS the combination -- a combined gain with no
per-change attribution cannot tell you which idea to keep.

The splits are LOADED and VALIDATED, never rebuilt: a silently rebuilt split
would produce numbers that look comparable and are not.
:func:`load_frozen_splits` refuses a corpus whose size does not match the
frozen file, refuses overlapping train/val/test, and refuses a split that does
not cover every index.

Nothing here touches the frozen artifacts or any existing results directory.
Output goes under ``--out`` (default ``results/group_a``).

Usage:
    python scripts/run_group_a.py --init-checkpoint <pretrained.pt>
    python scripts/run_group_a.py --arm A3 --set group_a.n_writings=8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.data.multitask import TaskCollator, TaskDataset, assemble_split  # noqa: E402
from polyt5.data.splits import load_splits  # noqa: E402
from polyt5.data.tg_metadata import prepare_labeled_rows, read_lamalab_rows  # noqa: E402
from polyt5.evaluation import (  # noqa: E402
    ArmResult,
    RegressionReport,
    build_ablation_matrix,
    format_ablation_matrix,
    regression_report,
)
from polyt5.inference.regression_predictor import (  # noqa: E402
    GROUP_A_CONFIG_KEY,
    RegressionPropertyPredictor,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.model.multitask import MultiTaskConfig, PolyT5MultiTask  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import TrainerConfig, load_checkpoint  # noqa: E402
from polyt5.training.group_a import ARM_IDS, GroupAConfig, arm_config  # noqa: E402
from polyt5.training.multitask_trainer import GroupATrainer, InterleavedLoader  # noqa: E402
from polyt5.utils import (  # noqa: E402
    RunDirectory,
    describe_device,
    get_logger,
    load_config,
    parse_dotted_overrides,
    require,
    save_config,
    seed_everything,
    select_device,
)

RESULTS_FILENAME = "results.json"
MATRIX_FILENAME = "ablation_matrix.json"


@dataclass(frozen=True)
class FrozenSplit:
    """One split loaded verbatim from the frozen splits file."""

    index: int
    train: list[int]
    val: list[int]
    test: list[int]


def _resolve(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _config_fingerprint(group_a: GroupAConfig) -> str:
    """Short, deterministic fingerprint of an arm's full configuration.

    Folded into the run-directory name so a hyperparameter override (e.g. the
    brief's own ``--arm A3 --set group_a.n_writings=8`` example) cannot
    collide with -- and silently reuse or overwrite -- a cached
    ``results.json`` written under a different configuration of the SAME arm.
    Two :class:`GroupAConfig` values that compare equal field-for-field
    always produce the same fingerprint (so a resumed run still finds its own
    cache); any field that differs changes it.

    Args:
        group_a: The arm's fully-resolved switches and hyperparameters.

    Returns:
        A 12-hex-character digest of ``group_a.to_dict()``.
    """
    payload = json.dumps(group_a.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_frozen_splits(path: str | Path, *, n_examples: int) -> list[FrozenSplit]:
    """Load the frozen splits and refuse anything that would break comparability.

    Args:
        path: Path to the frozen ``splits.json``.
        n_examples: Size of the corpus this run prepared.

    Returns:
        One :class:`FrozenSplit` per split, in file order.

    Raises:
        ValueError: If the file records no splits, if its corpus size does not
            match ``n_examples``, if a split's parts overlap, or if they do not
            cover every index.
    """
    payload = load_splits(path)
    splits = payload.get("splits") or []
    if not splits:
        raise ValueError(f"{path} contains no splits")

    recorded = payload.get("n")
    if recorded is not None and int(recorded) != n_examples:
        raise ValueError(
            f"{path} was built over {recorded} examples but this run prepared {n_examples}. "
            "The split indices would point at different polymers, so every Group A number "
            "would be measured on a different test set than the 28.6733 K baseline."
        )

    out: list[FrozenSplit] = []
    for entry in splits:
        train = [int(i) for i in entry["train"]]
        val = [int(i) for i in entry.get("val", [])]
        test = [int(i) for i in entry["test"]]
        parts = {"train": set(train), "val": set(val), "test": set(test)}
        for left in ("train", "val"):
            for right in ("val", "test"):
                if left != right and parts[left] & parts[right]:
                    raise ValueError(
                        f"split {entry['index']}: {left} and {right} must be disjoint, "
                        f"{len(parts[left] & parts[right])} indices appear in both"
                    )
        if parts["train"] | parts["val"] | parts["test"] != set(range(n_examples)):
            raise ValueError(
                f"split {entry['index']} does not cover every index in range({n_examples})"
            )
        out.append(
            FrozenSplit(index=int(entry["index"]), train=train, val=val, test=test)
        )
    return out


def resolve_arms(requested: Sequence[str] | None, **overrides: Any) -> list[GroupAConfig]:
    """Build the configurations to run, in :data:`ARM_IDS` order.

    Args:
        requested: Arm ids to run; ``None`` or empty means all seven.
        **overrides: Hyperparameter overrides applied to every arm.

    Returns:
        One :class:`GroupAConfig` per arm, deduplicated, in spec order.

    Raises:
        ValueError: If an id is not one of :data:`ARM_IDS`.
    """
    wanted = set(requested) if requested else set(ARM_IDS)
    unknown = sorted(wanted - set(ARM_IDS))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; valid arms are {list(ARM_IDS)}")
    return [arm_config(arm, **overrides) for arm in ARM_IDS if arm in wanted]


def load_baseline_reference(path: str | Path) -> tuple[float, float]:
    """Read the frozen five-split Tg MAE mean and standard deviation.

    Args:
        path: Path to ``artifacts/baseline/frozen_baseline.json``.

    Returns:
        ``(mae_mean, mae_std)`` in Kelvin.

    Raises:
        ValueError: If the artifact does not carry ``tg_prediction_5split.mae``
            with both a mean and a std. Hard-coding the numbers instead would
            let the plan drift from the artifact it claims to compare against.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mae = ((payload.get("tg_prediction_5split") or {}).get("mae")) or {}
    mean, std = mae.get("mean"), mae.get("std")
    if mean is None or std is None:
        raise ValueError(
            f"{path} carries no tg_prediction_5split.mae mean/std; the comparison point "
            "must come from the frozen artifact, never from a constant in the runner"
        )
    return float(mean), float(std)


def build_arm_model(
    group_a: GroupAConfig,
    n_descriptors: int,
    *,
    model_config_path: Path,
    init_checkpoint: Path | None,
    tokenizer: PolyT5Tokenizer,
    logger: Any,
    device: str,
) -> tuple[PolyT5MultiTask, MultiTaskConfig, bool]:
    """Build one arm's model, warm-started from the pretrained checkpoint.

    Args:
        group_a: The arm's switches.
        n_descriptors: Width of the descriptor head (kept columns), 0 when off.
        model_config_path: YAML describing the backbone when not warm-starting.
        init_checkpoint: Pretrained polyT5 checkpoint, or ``None``.
        tokenizer: The tokenizer every arm shares.
        logger: Progress logger.
        device: Torch device string.

    Returns:
        ``(model_on_device, head_config, was_pretrained)``.

    Raises:
        ValueError: If the checkpoint was trained on another vocabulary.
    """
    if init_checkpoint is not None:
        state = load_checkpoint(_resolve(init_checkpoint), map_location="cpu")
        recorded = state.get("tokenizer_sha256")
        if recorded and recorded != tokenizer.sha256:
            raise ValueError(
                "tokenizer mismatch: the checkpoint was trained with vocabulary "
                f"{recorded[:16]} but the configured tokenizer is {tokenizer.sha256[:16]}"
            )
        backbone_config = PolyT5Config.from_dict(state["model_config"])
        backbone = PolyT5ForConditionalGeneration(backbone_config)
        backbone.load_state_dict(state["model_state"])
        pretrained = True
        logger.info("arm %s: warm-started from %s", group_a.arm, init_checkpoint)
    else:
        backbone_config = PolyT5Config.from_yaml(model_config_path)
        backbone_config.vocab_size = tokenizer.vocab_size
        backbone_config.pad_token_id = tokenizer.pad_id
        backbone_config.eos_token_id = tokenizer.eos_id
        backbone_config.decoder_start_token_id = tokenizer.decoder_start_token_id
        backbone = PolyT5ForConditionalGeneration(backbone_config)
        pretrained = False
        logger.info("arm %s: RANDOM initialisation (no pretrained checkpoint)", group_a.arm)

    head_config = MultiTaskConfig(
        use_regression_head=group_a.regression_head,
        n_descriptors=n_descriptors if group_a.descriptors else 0,
        descriptor_lambda=group_a.descriptor_lambda,
        huber_delta=group_a.huber_delta,
        head_dropout=backbone_config.dropout_rate,
    )
    return PolyT5MultiTask(backbone, head_config).to(device), head_config, pretrained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "finetune" / "group_a.yaml")
    parser.add_argument("--arm", action="append", default=None, choices=list(ARM_IDS),
                        help="Run only this arm; repeatable. Default: all seven.")
    parser.add_argument("--splits-file", type=Path, default=None,
                        help="Override splits.frozen_file from the config.")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/group_a"))
    parser.add_argument("--only-split", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-run arm/split pairs whose results already exist.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap optimizer steps per split (debug).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of corpus rows read (debug).")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _score_split(
    model: PolyT5MultiTask,
    tokenizer: PolyT5Tokenizer,
    group_a: GroupAConfig,
    tensors: Any,
    *,
    cfg: dict[str, Any],
    device: str,
    logger: Any,
) -> tuple[RegressionReport, list[str]]:
    """Decode or regress the held-out split and score it."""
    eval_cfg = cfg.get("evaluation", {})
    if group_a.regression_head:
        predictor = RegressionPropertyPredictor(
            model, tokenizer, device=device,
            batch_size=int(eval_cfg.get("batch_size", 32)), property_name="Tg",
        )
        predictions = [result.decoded for result in predictor.predict(tensors.test_pselfies)]
    else:
        from polyt5.generation import BeamSearchConfig, beam_search

        model.eval()
        predictions = []
        batch_size = int(eval_cfg.get("batch_size", 32))
        for start in range(0, len(tensors.test_pselfies), batch_size):
            chunk = tensors.test_pselfies[start : start + batch_size]
            encoded = tokenizer.batch_encode(
                chunk, add_eos=True, max_length=int(cfg["data"]["max_length"]),
                padding=True, truncation=True,
            )
            with torch.no_grad():
                output = beam_search(
                    model.backbone,
                    torch.tensor(encoded["input_ids"], device=device),
                    torch.tensor(encoded["attention_mask"], device=device),
                    config=BeamSearchConfig(
                        num_beams=int(eval_cfg.get("beam_width", 4)),  # [PAPER] 4
                        max_length=int(eval_cfg.get("max_target_length", 32)),
                        length_penalty=float(eval_cfg.get("length_penalty", 1.0)),
                        eos_token_id=tokenizer.eos_id,
                        pad_token_id=tokenizer.pad_id,
                        decoder_start_token_id=tokenizer.decoder_start_token_id,
                    ),
                )
            predictions.extend(
                tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)
            )
            logger.info("  decoded %d/%d", len(predictions), len(tensors.test_pselfies))
    return regression_report(tensors.test_tg, predictions), predictions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=parse_dotted_overrides(args.set))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    # "train" is read second so a CLI/--set override under train.* still wins,
    # but every documented knob (including "training.device") lives under
    # "training" -- computed once, up front, so device selection and the
    # per-split trainer config below read the exact same merged mapping.
    train_cfg = {**cfg.get("training", {}), **cfg.get("train", {})}

    out_root = _resolve(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("polyt5.run_group_a", log_file=out_root / "run_group_a.log")
    device = select_device(train_cfg.get("device", "auto"))
    logger.info("device=%s", describe_device(device).to_dict())

    tokenizer_path = _resolve(
        cfg.get("tokenizer", {}).get("path", "artifacts/tokenizer/polyt5_vocab.json")
    )
    tokenizer = PolyT5Tokenizer.from_file(tokenizer_path)

    csv_path = _resolve(require(cfg, "data.csv_path"))
    rows, descriptor_names = read_lamalab_rows(csv_path, limit=args.limit)
    examples, stats = prepare_labeled_rows(
        rows,
        max_tokens=int(cfg["data"]["max_length"]),
        deduplicate=bool(cfg["data"].get("deduplicate", True)),
        tokenizer=tokenizer,
    )
    logger.info("corpus: %d usable rows (attrition %s)", len(examples),
                json.dumps(stats.to_dict()))

    splits_file = _resolve(args.splits_file or require(cfg, "splits.frozen_file"))
    splits = load_frozen_splits(splits_file, n_examples=len(examples))
    logger.info("reusing %d frozen splits from %s", len(splits), splits_file)

    baseline_mean, baseline_std = load_baseline_reference(
        _resolve(require(cfg, "baseline.frozen_file"))
    )
    group_cfg = cfg.get("group_a", {})
    arms = resolve_arms(
        args.arm,
        descriptor_lambda=float(group_cfg.get("descriptor_lambda", 0.1)),
        n_writings=int(group_cfg.get("n_writings", 4)),
        std_floor=float(group_cfg.get("std_floor", 5.6)),
        huber_delta=float(group_cfg.get("huber_delta", 1.0)),
    )
    max_source = int(train_cfg.get("max_source_length", 200))
    max_target = int(train_cfg.get("max_target_length", 200))
    collator = TaskCollator(pad_id=tokenizer.pad_id, max_source_length=max_source,
                            max_target_length=max_target)

    arm_results: list[ArmResult] = []
    for group_a in arms:
        reports: list[RegressionReport] = []
        for split in splits:
            if args.only_split is not None and split.index != args.only_split:
                continue
            fingerprint = _config_fingerprint(group_a)
            run_dir = RunDirectory.create(
                out_root, f"{group_a.arm}/split_{split.index}_{fingerprint}"
            )
            results_path = run_dir.root / RESULTS_FILENAME
            if results_path.is_file() and not args.force:
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                reports.append(RegressionReport(**payload["evaluation"]))
                logger.info("%s split %d: already done — skipping", group_a.arm, split.index)
                continue

            seed_everything(seed + split.index)
            tensors = assemble_split(
                examples, descriptor_names,
                train_indices=split.train, val_indices=split.val, test_indices=split.test,
                tokenizer=tokenizer,
                use_regression_head=group_a.regression_head,
                use_descriptors=group_a.descriptors,
                n_writings=group_a.effective_n_writings(),
                use_reliability_weighting=group_a.reliability_weighting,
                std_floor=group_a.std_floor,
                build_generation=group_a.multitask,
                seed=seed + split.index,
                max_source_length=max_source,
                max_target_length=max_target,
            )
            logger.info("%s split %d: %s", group_a.arm, split.index,
                        json.dumps(tensors.to_manifest()))

            n_descriptors = (
                0 if tensors.descriptor_standardizer is None
                else tensors.descriptor_standardizer.n_features
            )
            model, head_config, pretrained = build_arm_model(
                group_a, n_descriptors,
                model_config_path=_resolve(require(cfg, "model.config")),
                init_checkpoint=args.init_checkpoint,
                tokenizer=tokenizer, logger=logger, device=device,
            )
            model.set_target_scaling(
                mean=tensors.target_standardizer.mean[0],
                std=tensors.target_standardizer.std[0],
            )

            batch_size = int(train_cfg.get("batch_size", 16))  # [PAPER] 16
            trainer_config = TrainerConfig(
                max_epochs=int(train_cfg.get("epochs", 30)),   # [PAPER] 30
                physical_batch_size=batch_size,
                gradient_accumulation_steps=int(train_cfg.get(
                    "gradient_accumulation_steps", 1)),
                learning_rate=float(train_cfg.get("learning_rate", 3e-4)),  # [PAPER]
                weight_decay=float(train_cfg.get("weight_decay", 0.01)),    # [PAPER]
                scheduler=str(train_cfg.get("scheduler", "constant")),
                amp=bool(train_cfg.get("amp", True)),
                amp_dtype=str(train_cfg.get("amp_dtype", "bf16")),
                # Matches the frozen baseline's retention (scripts/run_splits.py):
                # the TrainerConfig default of 3 would keep ~12.6 GB of
                # checkpoints across the 35 arm/split runs instead of ~6.3 GB.
                keep_last_checkpoints=int(train_cfg.get("keep_last_checkpoints", 1)),
                max_steps=args.max_steps,
                seed=seed + split.index,
                device=device,
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
            prediction_loader = DataLoader(
                TaskDataset(tensors.train), batch_size=batch_size, shuffle=True,
                collate_fn=collator, num_workers=trainer_config.num_workers,
            )
            generation_loader = (
                DataLoader(TaskDataset(tensors.train_generation), batch_size=batch_size,
                           shuffle=True, collate_fn=collator,
                           num_workers=trainer_config.num_workers)
                if tensors.train_generation else None
            )
            val_loader = (
                DataLoader(TaskDataset(tensors.val), batch_size=batch_size, shuffle=False,
                           collate_fn=collator, num_workers=trainer_config.num_workers)
                if tensors.val else None
            )

            run_config = {
                **cfg,
                GROUP_A_CONFIG_KEY: {
                    "arm": group_a.arm,
                    "switches": group_a.switches(),
                    "config": group_a.to_dict(),
                    "heads": head_config.to_dict(),
                    "split_index": split.index,
                    "splits_file": str(splits_file),
                    "standardizers": {
                        "target": tensors.target_standardizer.to_dict(),
                        "descriptors": (
                            None if tensors.descriptor_standardizer is None
                            else tensors.descriptor_standardizer.to_dict()
                        ),
                    },
                    "attrition": tensors.to_manifest(),
                },
            }
            save_config(run_config, run_dir.config_path)
            run_dir.write_manifest({
                "stage": "group_a_ablation",
                "arm": group_a.arm,
                "split_index": split.index,
                "pretrained": pretrained,
                "tokenizer_sha256": tokenizer.sha256,
                "tokenizer_path": str(tokenizer_path),
                "model_parameters": model.num_parameters(),
                **tensors.to_manifest(),
            })

            started = time.time()
            trainer = GroupATrainer(
                model, InterleavedLoader(prediction_loader, generation_loader),
                trainer_config, group_a=group_a, val_loader=val_loader, run_dir=run_dir,
                tokenizer_path=tokenizer_path, tokenizer_sha256=tokenizer.sha256,
                run_config=run_config, logger=logger,
            )
            train_metrics = trainer.train()
            train_seconds = time.time() - started

            checkpoint_path = run_dir.checkpoints / "best.pt"
            if not checkpoint_path.exists():
                trainer.save(path=checkpoint_path, train_metrics=train_metrics)

            report, predictions = _score_split(
                model, tokenizer, group_a, tensors, cfg=cfg, device=device, logger=logger
            )
            run_dir.append_jsonl("predictions.jsonl", [
                {"source": source, "target": target, "prediction": prediction}
                for source, target, prediction in zip(
                    tensors.test_pselfies, tensors.test_tg, predictions, strict=True
                )
            ])
            run_dir.write_json(RESULTS_FILENAME, {
                "arm": group_a.arm,
                "split_index": split.index,
                "config_fingerprint": fingerprint,
                "pretrained": pretrained,
                "train_seconds": train_seconds,
                "training": train_metrics,
                "evaluation": report.to_dict(),
                "checkpoint": str(checkpoint_path),
                # "evaluation" is scored from the model's FINAL-EPOCH in-memory
                # weights (Trainer.train() never reloads a checkpoint before
                # returning), not necessarily from the state saved at
                # "checkpoint" -- which may hold an earlier, better-val-loss
                # epoch if GroupATrainer already wrote it mid-training. This
                # mirrors scripts/run_splits.py's baseline protocol exactly, so
                # comparability with the frozen 28.6733 K holds; only the label
                # was ambiguous.
                "scored_weights": "final_epoch_in_memory",
                "attrition": tensors.to_manifest(),
            })
            logger.info("%s split %d: MAE=%s RMSE=%s R2=%s (non-numeric %.4f)",
                        group_a.arm, split.index, report.mae, report.rmse, report.r2,
                        report.non_numeric_rate)
            reports.append(report)

        arm_results.append(
            ArmResult.from_reports(group_a.arm, group_a.switches(), reports)
        )

    matrix = build_ablation_matrix(
        arm_results, baseline_mean=baseline_mean, baseline_std=baseline_std
    )
    (out_root / MATRIX_FILENAME).write_text(
        json.dumps(matrix, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("\n%s", format_ablation_matrix(matrix))
    logger.info("wrote %s", out_root / MATRIX_FILENAME)
    return 0


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    raise SystemExit(main())
