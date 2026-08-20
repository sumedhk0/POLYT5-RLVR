"""End-to-end vertical slice of the supervised polyT5 pipeline.

Runs every layer of the reproduction on real polymer structures, in the order
the project brief specifies::

    raw PSMILES
        -> chemistry: [*] -> [At], validate, encode to PSELFIES
        -> tokenizer: the deterministic 458-token artifact
        -> span corruption: the paper's <=8 spans / <=3 tokens / <=15% objective
        -> collate into a padded batch with -100 label padding
        -> polyT5 (small by default)
        -> forward -> loss -> backward -> optimizer step
        -> checkpoint -> reload -> assert identical logits

This is a correctness and wiring check, not a training run. It exists so that
every stage can be exercised on one machine in seconds before any large-scale
training is attempted.

Example:
    python scripts/run_vertical_slice.py --model-config configs/model/polyt5_small.yaml --steps 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from polyt5.chemistry.conversion import psmiles_to_pselfies, star_to_at
from polyt5.chemistry.validity import validate_psmiles
from polyt5.data.collate import SpanCorruptionCollator
from polyt5.data.span_corruption import (
    SpanCorruptionConfig,
    batch_span_corrupt,
    corruption_statistics,
)
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.utils import (
    RunDirectory,
    describe_device,
    get_logger,
    memory_stats,
    reset_memory_stats,
    seed_everything,
    select_device,
)

#: Real polymer repeat units in star-terminated PSMILES form. Small and diverse
#: on purpose: aliphatic, ether, ester, aromatic, amide, halogenated, silicone.
DEMO_PSMILES: tuple[str, ...] = (
    "*CC*",                                          # polyethylene
    "*CCO*",                                         # poly(ethylene oxide)
    "*CC(C)*",                                       # polypropylene
    "*CC(c1ccccc1)*",                                # polystyrene
    "*C(=O)c1ccc(cc1)C(=O)OCCO*",                    # PET-like polyester
    "*C(=O)CCCCC(=O)NCCCCCCN*",                      # nylon-6,6-like polyamide
    "*CC(Cl)*",                                      # poly(vinyl chloride)
    "*C(F)(F)C(F)(F)*",                              # PTFE
    "*[Si](C)(C)O*",                                 # poly(dimethylsiloxane)
    "*Oc1ccc(cc1)C(C)(C)c1ccc(O*)cc1",               # bisphenol-A backbone
    "*CC(=O)OCC*",                                   # poly(vinyl acetate)-like
    "*NC(=O)c1ccc(cc1)C(=O)N*",                      # aromatic polyamide
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model/polyt5_small.yaml",
                        help="Model config YAML (Table S2 sizes live in configs/model/).")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/polyt5_vocab.json",
                        help="Tokenizer artifact; falls back to the built-in default vocabulary.")
    parser.add_argument("--steps", type=int, default=5, help="Optimizer steps to run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Physical batch size.")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Maximum sequence length (the paper uses 200).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the smoke run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", default="results/vertical_slice",
                        help="Run directory for artifacts and metrics.")
    return parser.parse_args(argv)


def stage_chemistry(logger) -> list[str]:
    """PSMILES -> [At]-capped PSMILES -> PSELFIES, dropping anything invalid."""
    pselfies: list[str] = []
    dropped: list[tuple[str, str]] = []
    for psmiles in DEMO_PSMILES:
        capped = star_to_at(psmiles)
        verdict = validate_psmiles(capped, expected_termini=2)
        if not (verdict.valid and verdict.correct_termini):
            dropped.append((psmiles, verdict.reason or "unknown"))
            continue
        encoded = psmiles_to_pselfies(capped)
        if encoded is None:
            dropped.append((psmiles, "selfies_encode_failed"))
            continue
        pselfies.append(encoded)

    logger.info("chemistry  : %d/%d polymers converted to PSELFIES", len(pselfies),
                len(DEMO_PSMILES))
    for psmiles, reason in dropped:
        logger.warning("chemistry  : dropped %-30s reason=%s", psmiles, reason)
    if not pselfies:
        raise RuntimeError("no polymers survived conversion; the chemistry layer is broken")
    logger.info("chemistry  : example %s -> %s", DEMO_PSMILES[0], pselfies[0])
    return pselfies


def stage_tokenizer(path: str, logger) -> PolyT5Tokenizer:
    artifact = Path(path)
    if artifact.exists():
        tokenizer = PolyT5Tokenizer.from_file(artifact)
        source = str(artifact)
    else:
        tokenizer = PolyT5Tokenizer.default()
        source = "built-in default (artifact not found; run scripts/build_tokenizer.py)"
    logger.info("tokenizer  : vocab_size=%d pad=%d eos=%d sentinels=%d..%d sha256=%s",
                tokenizer.vocab_size, tokenizer.pad_id, tokenizer.eos_id,
                tokenizer.sentinel_ids[0], tokenizer.sentinel_ids[-1], tokenizer.sha256[:16])
    logger.info("tokenizer  : source=%s", source)
    return tokenizer


def stage_span_corruption(token_id_lists, tokenizer, max_length: int, seed: int, logger):
    """Apply the paper's span-corruption objective and report its statistics."""
    config = SpanCorruptionConfig()
    rng = np.random.default_rng(seed)
    results = batch_span_corrupt(token_id_lists, sentinel_ids=tokenizer.sentinel_ids,
                                 config=config, rng=rng, eos_id=tokenizer.eos_id)
    stats = corruption_statistics(results)
    logger.info("span corr. : %s", json.dumps({k: round(float(v), 4) for k, v in stats.items()}))

    # Show a sequence that actually hosted spans; short polymers are skipped by
    # design when the 15% budget cannot fit even one span.
    example = max(results, key=lambda r: len(r.spans))
    logger.info("span corr. : example spans  %s", list(example.spans))
    logger.info("span corr. : example input  %s",
                tokenizer.decode(example.input_ids, skip_special_tokens=False)[:110])
    logger.info("span corr. : example labels %s",
                tokenizer.decode(example.labels, skip_special_tokens=False)[:110])

    # The paper's stated invariants, asserted rather than assumed.
    for result in results:
        assert len(result.spans) <= config.max_spans, "more than 8 spans"
        assert all(length <= config.max_span_length for _, length in result.spans), "span > 3"
        assert result.corruption_fraction <= config.corruption_rate + 1e-9, "corruption > 15%"
        for (start_a, len_a), (start_b, _) in zip(result.spans, result.spans[1:], strict=False):
            assert start_b >= start_a + len_a + config.min_gap, "adjacent spans"
    logger.info("span corr. : paper invariants hold (<=8 spans, <=3 tokens, <=15% of tokens, "
                "non-adjacent, sentinels in increasing order)")
    return config, stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed, deterministic=True)

    run_dir = RunDirectory.create(Path(args.out).parent, Path(args.out).name)
    logger = get_logger("vertical_slice", log_file=run_dir.logs / "vertical_slice.log")
    device = select_device(args.device)
    device_info = describe_device(device)
    logger.info("device     : %s", json.dumps(device_info.to_dict()))

    # ---------------------------------------------------------------- stages
    pselfies = stage_chemistry(logger)
    tokenizer = stage_tokenizer(args.tokenizer, logger)

    token_id_lists = [tokenizer.encode(s, add_eos=True, max_length=args.max_length)
                      for s in pselfies]
    lengths = [len(ids) for ids in token_id_lists]
    unk_count = sum(ids.count(tokenizer.unk_id) for ids in token_id_lists)
    logger.info("tokenizer  : token lengths min=%d max=%d mean=%.1f unknown_tokens=%d",
                min(lengths), max(lengths), sum(lengths) / len(lengths), unk_count)
    if unk_count:
        logger.warning("tokenizer  : %d unknown tokens — the base alphabet is missing symbols "
                       "that real polymers use", unk_count)

    span_config, span_stats = stage_span_corruption(token_id_lists, tokenizer, args.max_length,
                                                    args.seed, logger)

    collator = SpanCorruptionCollator(
        sentinel_ids=tokenizer.sentinel_ids,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        config=span_config,
        max_length=args.max_length,
        seed=args.seed,
    )

    # ---------------------------------------------------------------- model
    model_config = PolyT5Config.from_yaml(args.model_config)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_id
    model_config.eos_token_id = tokenizer.eos_id
    model_config.decoder_start_token_id = tokenizer.decoder_start_token_id
    model = PolyT5ForConditionalGeneration(model_config).to(device)
    n_params = model.num_parameters()
    logger.info("model      : %s d_model=%d layers=%d/%d heads=%d params=%d",
                Path(args.model_config).stem, model_config.d_model, model_config.num_layers,
                model_config.num_decoder_layers_resolved, model_config.num_heads, n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # ---------------------------------------------------------------- train
    reset_memory_stats(device)
    model.train()
    losses: list[float] = []
    total_target_tokens = 0
    start = time.perf_counter()
    for step in range(args.steps):
        batch_ids = [token_id_lists[i % len(token_id_lists)]
                     for i in range(step * args.batch_size, (step + 1) * args.batch_size)]
        batch = {k: v.to(device) for k, v in collator(batch_ids).items()}

        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"])
        loss = output.loss
        if not torch.isfinite(loss):
            logger.error("loss is not finite at step %d", step)
            return 1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        total_target_tokens += int((batch["labels"] != -100).sum().item())
        logger.info("step %d/%d : loss=%.4f grad_norm=%.3f input=%s labels=%s",
                    step + 1, args.steps, losses[-1], float(grad_norm),
                    tuple(batch["input_ids"].shape), tuple(batch["labels"].shape))
        run_dir.log_metrics({"step": step + 1, "loss": losses[-1],
                             "grad_norm": float(grad_norm)})
    elapsed = time.perf_counter() - start

    if losses[-1] >= losses[0]:
        logger.warning("loss did not decrease over %d steps (%.4f -> %.4f); with so few steps "
                       "this is not necessarily a bug", args.steps, losses[0], losses[-1])

    # ---------------------------------------------------------------- checkpoint
    from polyt5.training.checkpoint import load_checkpoint, save_checkpoint

    ckpt_path = run_dir.checkpoints / "vertical_slice.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=args.steps,
        config={"script": "run_vertical_slice", "args": vars(args)},
        model_config=model_config.to_dict(),
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=tokenizer.sha256,
        train_metrics={"final_loss": losses[-1]},
    )
    logger.info("checkpoint : saved %s (%.1f MB)", ckpt_path,
                ckpt_path.stat().st_size / 1024**2)

    model.eval()
    eval_batch = {k: v.to(device) for k, v in collator(token_id_lists[: args.batch_size]).items()}
    with torch.no_grad():
        before = model(input_ids=eval_batch["input_ids"],
                       attention_mask=eval_batch["attention_mask"],
                       labels=eval_batch["labels"]).logits

    reloaded = PolyT5ForConditionalGeneration(model_config).to(device)
    state = load_checkpoint(ckpt_path, map_location=device)
    reloaded.load_state_dict(state["model_state"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(input_ids=eval_batch["input_ids"],
                         attention_mask=eval_batch["attention_mask"],
                         labels=eval_batch["labels"]).logits

    if not torch.allclose(before, after, atol=1e-6):
        logger.error("checkpoint : reloaded model does not reproduce the original logits "
                     "(max abs diff %.3e)", float((before - after).abs().max()))
        return 1
    logger.info("checkpoint : reload reproduces identical logits")

    # ---------------------------------------------------------------- report
    memory = memory_stats(device)
    summary = {
        "model_config": Path(args.model_config).stem,
        "parameters": n_params,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.sha256,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "max_sequence_length": args.max_length,
        "steps": args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "target_tokens": total_target_tokens,
        "tokens_per_second": round(total_target_tokens / elapsed, 1),
        "seconds": round(elapsed, 2),
        "device": device_info.to_dict(),
        "vram": memory,
        "span_corruption": {k: float(v) for k, v in span_stats.items()},
    }
    run_dir.write_json("slice_summary.json", summary)
    run_dir.write_manifest({"stage": "vertical_slice", "tokenizer_sha256": tokenizer.sha256})

    logger.info("=" * 78)
    for key, value in summary.items():
        logger.info("%-24s %s", key, value)
    logger.info("=" * 78)
    logger.info("VERTICAL SLICE OK — chemistry -> tokenizer -> span corruption -> batch -> "
                "polyT5 -> loss -> backward -> step -> checkpoint -> reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
