"""GRPO/RLVR training CLI for the four polyT5 arms.

Phase 3. NOT part of the published polyT5 method - see ``docs/rlvr_plan.md``.

Every run is anchored to ``artifacts/baseline/frozen_baseline.json`` (see
``configs/rl/*.yaml``): every checkpoint this script opens has its SHA-256
verified against that record before it is loaded, and the property-prediction
ensemble that backs the reward is built EXCLUSIVELY from the four checkpoints
listed under ``reward_ensemble``. The fifth split, ``auditor``, is held out of
every REWARD path. ``scripts/compare_arms.py`` uses it for confirmation scoring
after training.

The auditor and the training process
------------------------------------
Spec section 4.4 promises an ``auditor gap`` logged during training, so this
script does load the auditor -- but only into
:class:`~polyt5.rl.drift.DriftMonitor`, which is handed to
:class:`~polyt5.rl.trainer.GRPOTrainer` in an attribute of its own and is never
passed to the reward arm, never consulted while a reward is computed, and can
only ever write into the step stats. :func:`build_drift_monitor` additionally
re-checks that the auditor key is absent from ``reward_ensemble`` before
constructing anything, on top of the two guards
:func:`load_frozen_baseline` and :func:`build_reward_ensemble` already apply.
``--no-drift-monitor`` skips loading it at all.

Usage:
    python scripts/train_grpo.py --arm accuracy
    python scripts/train_grpo.py --arm composite --set train.max_steps=50
    python scripts/train_grpo.py --arm constraint --max-steps 10 --device cpu
    python scripts/train_grpo.py --arm accuracy --resume results/grpo_accuracy
    python scripts/train_grpo.py --arm accuracy --no-drift-monitor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from polyt5.chemistry import ScalableNoveltyIndex, pselfies_to_psmiles  # noqa: E402
from polyt5.inference import EnsemblePropertyPredictor, PolyT5PropertyPredictor  # noqa: E402
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.rewards import (  # noqa: E402
    DEFAULT_MIN_COVERAGE,
    DEFAULT_SIGMA0,
    DEFAULT_SIGMA_UNKNOWN,
    TgRewardConfig,
    build_arm,
)
from polyt5.rl import DriftMonitor, GRPOTrainer, GRPOTrainerConfig  # noqa: E402
from polyt5.rl.drift import DEFAULT_MAX_REFERENCE  # noqa: E402
from polyt5.rl.rollout import ROLLOUT_CHUNK_SIZE  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402
from polyt5.training import load_checkpoint  # noqa: E402
from polyt5.utils import (  # noqa: E402
    RunDirectory,
    get_logger,
    load_config,
    parse_dotted_overrides,
    save_config,
    seed_everything,
    select_device,
)

#: The four arms Task 6 implements. Order matches the paper's C1-C4 numbering.
ARMS: tuple[str, ...] = ("accuracy", "validity", "composite", "constraint")

#: Arms whose reward needs a TSD / novelty reference set (see
#: ``polyt5.rewards.composite`` -- ``ValidityArm`` requires one by default,
#: ``CompositeArm`` and ``ConstraintArm`` score a ``novelty`` term with one).
ARMS_NEEDING_NOVELTY_INDEX: frozenset[str] = frozenset({"validity", "composite", "constraint"})

#: Checkpoint used to initialise BOTH the policy and the reference: the
#: paper's fine-tuned Tg-conditioned generation model. GRPO refines it further;
#: it is not trained from the raw pretrained checkpoint.
POLICY_ARTIFACT_KEY = "generation"

#: Default novelty index, built by ``scripts/build_novelty_index.py`` from the
#: same generation training split the frozen baseline's models were trained on.
DEFAULT_NOVELTY_INDEX = REPO_ROOT / "artifacts" / "novelty" / "tg_generation_train"

#: Labelled Tg polymers the drift monitor measures max-Tanimoto against -- the
#: same generation training split the novelty index is built from, so "novel"
#: (exact canonical absence) and "similar" (ECFP6) are talking about the same
#: reference set.
DEFAULT_DRIFT_REFERENCE = REPO_ROOT / "data" / "processed" / "tg" / "generation" / "train.jsonl"


def _resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repo root unless it is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file's bytes, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, expected_sha256: str, *, label: str) -> None:
    """Raise unless ``path`` exists and its SHA-256 matches the frozen record.

    Args:
        path: File to verify.
        expected_sha256: The hash recorded in ``frozen_baseline.json``.
        label: Human-readable name for error messages (e.g. ``"generation"``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file's hash does not match ``expected_sha256``. RL
            may never run against an unrecorded or modified baseline artifact.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {path}")
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} artifact at {path} failed SHA-256 verification: "
            f"got {actual}, expected {expected_sha256} (from frozen_baseline.json). "
            "Refusing to train against an unverified baseline artifact."
        )


def load_frozen_baseline(path: Path) -> dict[str, Any]:
    """Load and sanity-check the immutable supervised-baseline record.

    Args:
        path: Path to ``frozen_baseline.json``.

    Returns:
        The parsed record.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If a required top-level key is missing.
        ValueError: If the auditor split also appears in the reward ensemble --
            that would mean the held-out confirmation split leaked into the
            reward path.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen baseline not found: {path}. RL may not run against an "
            "unrecorded baseline; run the Phase 1/2 pipeline and freeze it first."
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    for key in ("artifacts", "reward_ensemble", "auditor", "evaluation_protocol",
                "success_criterion"):
        if key not in frozen:
            raise KeyError(f"frozen baseline at {path} is missing required key {key!r}")
    if frozen["auditor"] in frozen["reward_ensemble"]:
        raise ValueError(
            f"{path}: auditor split {frozen['auditor']!r} also appears in reward_ensemble "
            f"{frozen['reward_ensemble']!r}. The auditor must never enter a reward path."
        )
    return frozen


def build_verified_tokenizer(frozen: dict[str, Any]) -> PolyT5Tokenizer:
    """Load the tokenizer artifact, verifying its SHA-256 against the baseline."""
    meta = frozen["artifacts"]["tokenizer"]
    path = _resolve(meta["path"])
    verify_artifact(path, meta["sha256"], label="tokenizer")
    return PolyT5Tokenizer.from_file(path)


def load_verified_model(
    frozen: dict[str, Any], artifact_key: str
) -> tuple[PolyT5ForConditionalGeneration, dict[str, Any]]:
    """Load ONE fresh model instance from a frozen-baseline checkpoint.

    Every call constructs a brand-new :class:`PolyT5ForConditionalGeneration`
    and loads the state dict into it, so calling this twice for the same
    ``artifact_key`` yields two SEPARATE model objects with identical weights
    -- exactly what :class:`~polyt5.rl.trainer.GRPOTrainer` requires for
    ``policy`` and ``reference`` (it raises if they are the same object).

    Args:
        frozen: The parsed ``frozen_baseline.json``.
        artifact_key: Key into ``frozen["artifacts"]``, e.g. ``"generation"``.

    Returns:
        ``(model, checkpoint_payload)``.
    """
    meta = frozen["artifacts"][artifact_key]
    path = _resolve(meta["path"])
    verify_artifact(path, meta["sha256"], label=artifact_key)
    payload = load_checkpoint(path, map_location="cpu")
    model_config = PolyT5Config.from_dict(payload["model_config"])
    model = PolyT5ForConditionalGeneration(model_config)
    model.load_state_dict(payload["model_state"])
    return model, payload


def build_reward_ensemble(
    frozen: dict[str, Any],
    tokenizer_path: Path,
    *,
    device: str,
    batch_size: int,
    num_beams: int,
) -> EnsemblePropertyPredictor:
    """Build the Tg-prediction ensemble from ``reward_ensemble`` splits only.

    Args:
        frozen: The parsed ``frozen_baseline.json``.
        tokenizer_path: Tokenizer artifact shared by every ensemble member.
        device: Torch device string.
        batch_size: Decode batch size per member.
        num_beams: Beam width. [PAPER] 4.

    Returns:
        The assembled ensemble.

    Raises:
        ValueError: If the auditor split is (still) present in
            ``reward_ensemble`` -- a second guard alongside
            :func:`load_frozen_baseline`'s, kept here because this is the
            function that actually decides which checkpoints become reward
            models.
    """
    ensemble_keys = frozen["reward_ensemble"]
    auditor_key = frozen["auditor"]
    if auditor_key in ensemble_keys:
        raise ValueError(
            f"auditor split {auditor_key!r} must never be used to build the reward ensemble"
        )
    paths: list[Path] = []
    for key in ensemble_keys:
        meta = frozen["artifacts"][key]
        path = _resolve(meta["path"])
        verify_artifact(path, meta["sha256"], label=key)
        paths.append(path)
    return EnsemblePropertyPredictor.from_checkpoints(
        paths,
        tokenizer_path,
        device=device,
        batch_size=batch_size,
        num_beams=num_beams,
        property_name="Tg",
    )


def load_drift_reference(path: Path, *, limit: int, field: str = "target") -> list[str]:
    """Read the labelled Tg polymers the drift monitor fingerprints against.

    Args:
        path: A JSONL file whose ``field`` holds a PSELFIES string -- the same
            ``data/processed/tg/generation/train.jsonl`` the novelty index is
            built from.
        limit: Stop after this many usable rows.
        field: Which key holds the polymer.

    Returns:
        Canonical-ish PSMILES strings (whatever
        :func:`~polyt5.chemistry.pselfies_to_psmiles` returns), with
        unconvertible rows dropped. An empty list -- a missing or unreadable
        file included -- disables the max-Tanimoto half rather than raising:
        this is a diagnostic, and a run must not die because a monitor could
        not be built.
    """
    if not path.is_file():
        return []
    out: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line).get(field)
                psmiles = pselfies_to_psmiles(value) if value else None
            except Exception:
                continue
            if psmiles:
                out.append(psmiles)
            if len(out) >= limit:
                break
    return out


def build_drift_monitor(
    frozen: dict[str, Any],
    cfg: dict[str, Any],
    tokenizer_path: Path,
    *,
    device: str,
    batch_size: int,
    num_beams: int,
    seed: int,
) -> DriftMonitor:
    """Build spec section 4.4's drift monitor, auditor included.

    Args:
        frozen: The parsed ``frozen_baseline.json``.
        cfg: The resolved run config; reads the optional ``drift`` block
            (``reference``, ``max_reference``).
        tokenizer_path: Tokenizer artifact for the auditor predictor.
        device: Torch device string.
        batch_size: Decode batch size for the auditor.
        num_beams: Beam width for the auditor. [PAPER] 4.
        seed: Seed for the reference subsample.

    Returns:
        A :class:`~polyt5.rl.drift.DriftMonitor`. The auditor goes in as a
        plain callable and the monitor exposes no way to get it back out; see
        that class's module docstring for why that matters and
        :class:`~polyt5.rl.trainer.GRPOTrainer` for the second half of the
        containment.

    Raises:
        ValueError: If the auditor split appears in ``reward_ensemble``. This
            is the third independent check of the same invariant, placed here
            because this is the only function in the training path that opens
            the auditor's checkpoint at all.
    """
    auditor_key = frozen["auditor"]
    if auditor_key in frozen["reward_ensemble"]:
        raise ValueError(
            f"auditor split {auditor_key!r} appears in reward_ensemble "
            f"{frozen['reward_ensemble']!r}; refusing to load it anywhere in the training "
            "process."
        )
    meta = frozen["artifacts"][auditor_key]
    auditor_path = _resolve(meta["path"])
    verify_artifact(auditor_path, meta["sha256"], label=auditor_key)
    auditor = PolyT5PropertyPredictor.from_checkpoint(
        auditor_path, tokenizer_path, device=device, batch_size=batch_size,
        num_beams=num_beams, property_name="Tg",
    )

    drift_cfg = cfg.get("drift", {})
    reference_path = _resolve(drift_cfg.get("reference", DEFAULT_DRIFT_REFERENCE))
    max_reference = int(drift_cfg.get("max_reference", DEFAULT_MAX_REFERENCE))
    reference = load_drift_reference(reference_path, limit=max_reference)
    return DriftMonitor(
        auditor=auditor, reference_psmiles=reference,
        max_reference=max_reference, seed=seed,
    )


#: Sentinel distinguishing "no override was given" (resolve the index from
#: ``cfg`` as before) from "the override IS None" (use no index at all, e.g.
#: ``scripts/compare_arms.py --allow-missing-novelty-index``). Plain ``None``
#: cannot serve as that sentinel because it is also the deliberate override
#: value in the second case -- using it for both would make the second case
#: fall through to re-opening ``cfg``'s path, which is exactly the file the
#: caller already established is missing.
_NO_NOVELTY_INDEX_OVERRIDE = object()


def build_reward_arm(
    arm_name: str, cfg: dict[str, Any], *, novelty_index: Any = _NO_NOVELTY_INDEX_OVERRIDE,
    ensemble_size: int = 1,
):
    """Construct the reward arm from a resolved config's ``reward`` block.

    Args:
        arm_name: One of :data:`ARMS`.
        cfg: The fully-resolved run config.
        ensemble_size: How many members the property predictor whose
            ``(mean, std, n_contributing_members)`` triples this arm will
            receive actually has -- ``len(predictor)`` for the reward
            ensemble, ``1`` for a single-model predictor such as the auditor.
            It is the denominator of the coverage term in the confidence
            weight (see :mod:`polyt5.rewards.tg`); getting it wrong raises at
            the first prediction rather than silently understating the
            penalty. It is NOT read from ``cfg``: the ensemble is defined by
            ``frozen_baseline.json``'s ``reward_ensemble`` list, not by an
            arm's own config, and a ``--set`` override must not be able to
            change it.
        novelty_index: Pre-opened novelty index to use instead of opening one
            from ``cfg["reward"]["novelty_index"]``. Passed by
            ``scripts/compare_arms.py`` so that every arm it scores shares
            the exact same (already SHA-256-recorded) index object rather
            than each opening its own file handle on the same path. Pass
            ``None`` explicitly (not the default) to force "no index" without
            attempting to open ``cfg``'s path at all -- see
            :data:`_NO_NOVELTY_INDEX_OVERRIDE`.

    Returns:
        An :class:`~polyt5.rewards.ArmReward`.

    Note:
        ``reward.tolerance`` and ``reward.window_tolerance`` are deliberately
        SEPARATE keys, not one key doing double duty: ``tolerance`` feeds
        :class:`~polyt5.rewards.TgRewardConfig` (the Kelvin at which the tg
        term's continuous closeness reaches zero, spec Sec 4.2, default
        100.0), while ``window_tolerance`` feeds ``_BaseArm.tolerance`` --
        concretely, only :class:`~polyt5.rewards.composite.ConstraintArm`
        reads it, as C4's in-window half-width (spec Sec 4.3, default 50.0).
        The two were previously the same key with two different defaults,
        which could not be set independently and read as a mistake even
        though no arm actually consumed both meanings at once.
    """
    reward_cfg = cfg.get("reward", {})
    tg_config = TgRewardConfig(
        tolerance=float(reward_cfg.get("tolerance", 100.0)),
        sigma0=float(reward_cfg.get("sigma0", DEFAULT_SIGMA0)),
        sigma_unknown=float(reward_cfg.get("sigma_unknown", DEFAULT_SIGMA_UNKNOWN)),
        min_coverage=float(reward_cfg.get("min_coverage", DEFAULT_MIN_COVERAGE)),
    )
    kwargs: dict[str, Any] = {
        "tolerance": float(reward_cfg.get("window_tolerance", 50.0)),
        "sa_max": float(reward_cfg.get("sa_max", 6.0)),
        "tg_config": tg_config,
        "ensemble_size": int(ensemble_size),
    }
    if arm_name in ARMS_NEEDING_NOVELTY_INDEX:
        if novelty_index is not _NO_NOVELTY_INDEX_OVERRIDE:
            # Explicit override, including None -- never re-derive from cfg.
            kwargs["novelty_index"] = novelty_index
        else:
            index_path = _resolve(reward_cfg.get("novelty_index", DEFAULT_NOVELTY_INDEX))
            kwargs["novelty_index"] = ScalableNoveltyIndex.open(index_path)
    if arm_name == "composite":
        weights = reward_cfg.get("weights")
        if weights is not None:
            kwargs["weights"] = dict(weights)
    return build_arm(arm_name, **kwargs)


#: ``reward.*`` keys Ruling F fixed to one correct, safety-critical value (see
#: ``polyt5.rewards.tg``'s module docstring): together they are the
#: substitution and coverage-fraction machinery that stops candidates the
#: reward ensemble cannot parse from out-scoring candidates it agrees on
#: (measured: a 1-of-4 candidate scored 1.0000 before the fix, 0.0683 after,
#: at equal closeness). A ``--set reward.sigma_unknown=0.0`` silently reverts
#: that fix -- nothing in a resolved ``config.yaml`` flags a value as
#: "the one Ruling F picked" versus "an ordinary tuning knob". These three are
#: refused outright rather than merely logged: unlike ``tolerance``/``sa_max``/
#: ``window_tolerance``/``weights``, which are design choices a legitimate
#: ablation might sweep, these three have one correct value, matched by every
#: shipped ``configs/rl/*.yaml`` to ``polyt5.rewards.tg.DEFAULT_SIGMA0`` /
#: ``DEFAULT_SIGMA_UNKNOWN`` / ``DEFAULT_MIN_COVERAGE``, and the entire point
#: of fixing them was that they cannot be quietly changed back per-run.
PINNED_REWARD_PARAMS: frozenset[str] = frozenset({"sigma0", "sigma_unknown", "min_coverage"})


def check_reward_overrides(overrides: dict[str, Any]) -> list[str]:
    """Refuse a ``--set`` that would touch one of Ruling F's pinned reward constants.

    Args:
        overrides: The parsed ``--set`` / ``--max-steps`` overrides dict,
            BEFORE merging into the resolved config -- this must run before
            ``load_config`` so a refusal happens before anything is loaded.

    Returns:
        Human-readable refusal reasons, one per pinned parameter touched.
        Empty means none of :data:`PINNED_REWARD_PARAMS` were touched.
    """
    reward_overrides = overrides.get("reward")
    if not isinstance(reward_overrides, dict):
        return []
    problems: list[str] = []
    for key in sorted(PINNED_REWARD_PARAMS & set(reward_overrides)):
        problems.append(
            f"reward.{key}={reward_overrides[key]!r} is refused -- {key} is one of Ruling F's "
            "pinned reward constants and overriding it would silently change what the reward "
            "ensemble optimizes (see PINNED_REWARD_PARAMS)."
        )
    return problems


def summarize_reward_overrides(
    overrides: dict[str, Any], canonical_cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build an explicit before/after record of any ``--set reward.*`` override actually applied.

    Not refused (see :func:`check_reward_overrides` for the three that are),
    but a pre-registered reward's whole value is that a reader cannot miss
    when one of its knobs was moved for this particular run -- so every
    ``reward.*`` override this run applies is recorded here, not only the
    pinned ones, and the caller writes the result into both the run manifest
    and a log line.

    Args:
        overrides: The parsed ``--set`` overrides dict.
        canonical_cfg: The SAME config file loaded with no overrides, i.e.
            what this run would have used had ``--set`` said nothing about
            ``reward``. The comparison baseline, not a hard-coded default --
            so this also catches overriding a key that already differs from
            ``polyt5.rewards.tg``'s own defaults in a custom ``--config``.

    Returns:
        ``{key: {"canonical": ..., "overridden_to": ...}}`` for every
        ``reward.*`` key this run's overrides touch. Empty when none do.
    """
    reward_overrides = overrides.get("reward")
    if not isinstance(reward_overrides, dict):
        return {}
    canonical_reward = canonical_cfg.get("reward", {})
    return {
        key: {"canonical": canonical_reward.get(key), "overridden_to": value}
        for key, value in reward_overrides.items()
    }


def build_trainer_config(cfg: dict[str, Any], *, seed: int,
                          device_override: str | None) -> GRPOTrainerConfig:
    """Build a :class:`GRPOTrainerConfig` from a resolved config's ``train`` block.

    ``--max-steps`` has no separate parameter here: it is folded into
    ``cfg["train"]["max_steps"]`` by :func:`main` before this is called (via
    the same ``overrides`` dict ``--set`` uses), so there is exactly one
    mechanism for overriding it, not two.
    """
    train_cfg = cfg.get("train", {})
    return GRPOTrainerConfig(
        group_size=int(train_cfg.get("group_size", 16)),
        prompts_per_step=int(train_cfg.get("prompts_per_step", 32)),
        max_steps=int(train_cfg.get("max_steps", 2000)),
        max_length=int(train_cfg.get("max_length", 200)),
        target_min=float(train_cfg.get("target_min", 250.0)),
        target_max=float(train_cfg.get("target_max", 600.0)),
        temperature=float(train_cfg.get("temperature", 0.7)),
        top_p=float(train_cfg.get("top_p", 0.95)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        clip_eps=float(train_cfg.get("clip_eps", 0.2)),
        kl_coef=float(train_cfg.get("kl_coef", 0.02)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        device=str(device_override or train_cfg.get("device", "auto")),
        seed=seed,
        log_every=int(train_cfg.get("log_every", 10)),
        save_every=int(train_cfg.get("save_every", 250)),
        rollout_batch_size=int(train_cfg.get("rollout_batch_size", ROLLOUT_CHUNK_SIZE)),
        drift_every=int(cfg.get("drift", {}).get("every", 50)),
    )


def _latest_checkpoint(path: Path) -> Path:
    """Resolve ``--resume`` to a concrete checkpoint file.

    Args:
        path: Either a ``.pt`` file, or a run directory / ``checkpoints``
            directory containing ``step_NNNNNN.pt`` files.

    Returns:
        The checkpoint to resume from -- ``path`` itself if it is a file, or
        the lexicographically-last (i.e. highest-step) ``*.pt`` inside it.

    Raises:
        FileNotFoundError: If ``path`` is a directory with no checkpoints.
    """
    if path.is_file():
        return path
    checkpoints_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"--resume {path}: no checkpoints found under {checkpoints_dir}")
    return candidates[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ARMS, required=True,
                        help="Which reward arm to train (C1-C4).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Defaults to configs/rl/<arm>.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dotted-path config overrides, e.g. train.max_steps=50")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Shorthand for --set train.max_steps=N; wins if both are given.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint file, or run/checkpoints directory, to resume "
                             "policy + optimizer + step count from.")
    parser.add_argument("--device", default=None, help="Overrides train.device.")
    parser.add_argument("--novelty-index", type=Path, default=None,
                        help="Overrides reward.novelty_index for arms that need one.")
    parser.add_argument("--predictor-batch-size", type=int, default=64,
                        help="Decode batch size for the reward ensemble.")
    parser.add_argument("--predictor-num-beams", type=int, default=4,
                        help="Beam width for the reward ensemble. [PAPER] 4.")
    parser.add_argument("--no-drift-monitor", action="store_true",
                        help="Skip spec 4.4 drift monitoring. The monitor loads the held-out "
                             "auditor into the training process (for logging only -- it is "
                             "never passed to the reward arm); pass this to keep the auditor "
                             "checkpoint out of the process entirely.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)

    config_path = args.config or (REPO_ROOT / "configs" / "rl" / f"{args.arm}.yaml")
    overrides = parse_dotted_overrides(args.set)
    if args.max_steps is not None:
        train_overrides = dict(overrides.get("train", {}))
        train_overrides["max_steps"] = args.max_steps
        overrides = {**overrides, "train": train_overrides}

    # Refused BEFORE anything is loaded, not merely logged after the fact --
    # see PINNED_REWARD_PARAMS. A `--set reward.sigma_unknown=0.0` must not
    # be able to silently revert Ruling F.
    reward_override_problems = check_reward_overrides(overrides)
    if reward_override_problems:
        print(
            "ERROR: refusing to start -- " + "; ".join(reward_override_problems) + "\n"
            "These are safety-critical constants pinned by Ruling F, not tunable knobs; their "
            "whole value is that they cannot be changed after the fact by a per-run flag. If "
            "the constant itself must legitimately change, edit polyt5.rewards.tg."
            "DEFAULT_SIGMA0 / DEFAULT_SIGMA_UNKNOWN / DEFAULT_MIN_COVERAGE directly -- that "
            "changes every arm's default in one visible diff instead of one silent override.",
            file=sys.stderr,
        )
        return 1

    try:
        cfg = load_config(config_path, overrides=overrides)
    except FileNotFoundError as error:
        print(f"ERROR: config not found: {error}", file=sys.stderr)
        return 1

    if cfg.get("arm") != args.arm:
        print(f"ERROR: {config_path} names arm {cfg.get('arm')!r}, but --arm {args.arm!r} "
              "was requested.", file=sys.stderr)
        return 1
    if args.novelty_index is not None:
        cfg.setdefault("reward", {})["novelty_index"] = str(args.novelty_index)

    # Not refused (see check_reward_overrides for what is), but a reader must
    # not be able to miss when a reward.* knob was moved for this particular
    # run -- compared against the SAME file with no overrides applied.
    canonical_cfg = load_config(config_path)
    reward_override_record = summarize_reward_overrides(overrides, canonical_cfg)

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    run_dir = RunDirectory.create(_resolve(cfg.get("output", {}).get("results_root", "results")),
                                  cfg.get("experiment_name") or f"grpo_{args.arm}")
    logger = get_logger("polyt5.train_grpo", log_file=run_dir.logs / "train_grpo.log")
    logger.info("arm=%s config=%s run_dir=%s", args.arm, config_path, run_dir.root)
    if reward_override_record:
        logger.warning(
            "reward parameter override(s) active for this run -- differs from the canonical "
            "%s: %s", config_path, reward_override_record)

    try:
        frozen = load_frozen_baseline(_resolve(cfg["baseline"]))
        tokenizer = build_verified_tokenizer(frozen)
        device = select_device(args.device or cfg.get("train", {}).get("device", "auto"))

        policy, _ = load_verified_model(frozen, POLICY_ARTIFACT_KEY)
        reference, _ = load_verified_model(frozen, POLICY_ARTIFACT_KEY)

        tokenizer_path = _resolve(frozen["artifacts"]["tokenizer"]["path"])
        predictor = build_reward_ensemble(
            frozen, tokenizer_path,
            device=str(device), batch_size=args.predictor_batch_size,
            num_beams=args.predictor_num_beams,
        )
        # The arm MUST know how many members produced the triples it will be
        # fed: n_contributing == 1 means "full coverage" for a single-model
        # predictor and "three of four members could not read this candidate"
        # for the reward ensemble, and only this denominator separates them.
        arm = build_reward_arm(args.arm, cfg, ensemble_size=len(predictor))

        drift_monitor = None
        drift_cfg = cfg.get("drift", {})
        drift_enabled = bool(drift_cfg.get("enabled", True)) and not args.no_drift_monitor
        if drift_enabled:
            drift_monitor = build_drift_monitor(
                frozen, cfg, tokenizer_path, device=str(device),
                batch_size=args.predictor_batch_size, num_beams=args.predictor_num_beams,
                seed=seed,
            )
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("could not build the run: %s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    trainer_config = build_trainer_config(cfg, seed=seed, device_override=str(device))

    save_config(cfg, run_dir.config_path)
    run_dir.write_manifest({
        "stage": "grpo_train",
        "arm": args.arm,
        "tokenizer_sha256": tokenizer.sha256,
        "baseline": str(_resolve(cfg["baseline"])),
        "reward_ensemble": list(frozen["reward_ensemble"]),
        "auditor_excluded_from_reward": frozen["auditor"],
        "auditor_loaded_for_drift_monitoring": drift_monitor is not None,
        "reward_ensemble_size": len(predictor),
        "device": str(device),
        # Loud, unmistakable record of any --set reward.* override this run
        # actually applied (see summarize_reward_overrides) -- empty when
        # none did. The three parameters Ruling F pins are refused outright,
        # earlier in main(), and never reach here.
        "reward_parameter_overrides": reward_override_record,
    })
    logger.info("reward ensemble: %s (%d members; auditor %r excluded from every reward path)",
                frozen["reward_ensemble"], len(predictor), frozen["auditor"])
    if drift_monitor is not None:
        logger.info(
            "drift monitor: every %d steps; auditor=%s (LOGGING ONLY -- never passed to the "
            "reward arm) reference_fingerprints=%d",
            trainer_config.drift_every, drift_monitor.has_auditor, drift_monitor.n_reference,
        )
    else:
        logger.info("drift monitor: disabled (spec 4.4 auditor gap and max-Tanimoto not logged)")
    logger.info("trainer config: %s", asdict(trainer_config))

    trainer = GRPOTrainer(
        policy=policy, reference=reference, tokenizer=tokenizer, arm=arm,
        predictor=predictor, config=trainer_config, run_dir=run_dir,
        drift_monitor=drift_monitor,
    )

    start_step = 0
    if args.resume is not None:
        try:
            resume_path = _latest_checkpoint(_resolve(args.resume))
            payload = load_checkpoint(resume_path)
        except FileNotFoundError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        trainer.policy.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state") is not None:
            trainer.optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload.get("global_step", 0))
        logger.info("resumed from %s at step %d", resume_path, start_step)

    if start_step >= trainer_config.max_steps:
        logger.info("nothing to do: already at step %d >= max_steps %d",
                    start_step, trainer_config.max_steps)
        result = {"num_steps": 0, "history": []}
    else:
        # start_step=0 (the common, non-resumed case) behaves exactly as
        # before this parameter existed -- see GRPOTrainer.train's docstring.
        result = trainer.train(start_step=start_step)

    logger.info("done: %d steps run, results in %s", result["num_steps"], run_dir.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
