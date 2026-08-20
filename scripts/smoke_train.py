"""Vertical-slice smoke test: tiny model -> N optimizer steps -> checkpoint ->
reload -> identical logits -> summary report. Depends only on the model and
training tracks (synthetic random token batches; no tokenizer, no data files).

Usage:
    python scripts/smoke_train.py --model-config configs/model/polyt5_tiny.yaml \
        --steps 3 --device auto --out results/smoke

Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration  # noqa: E402
from polyt5.training import Trainer, TrainerConfig, resume  # noqa: E402
from polyt5.utils import RunDirectory, seed_everything  # noqa: E402

ACCUM = 2  # micro-batches per optimizer step in this smoke run


def synthetic_batches(cfg: PolyT5Config, n: int, batch_size: int, seq_len: int) -> list[dict]:
    """Random token-id batches shaped like the real collator output."""
    g = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(n):
        batches.append(
            {
                "input_ids": torch.randint(2, cfg.vocab_size, (batch_size, seq_len), generator=g),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "labels": torch.randint(2, cfg.vocab_size, (batch_size, seq_len), generator=g),
            }
        )
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model/polyt5_tiny.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results/smoke")
    args = parser.parse_args()

    seed_everything(0)
    model_cfg = PolyT5Config.from_yaml(args.model_config)
    model = PolyT5ForConditionalGeneration(model_cfg)

    seq_len = min(32, model_cfg.n_positions)
    batches = synthetic_batches(model_cfg, n=args.steps * ACCUM, batch_size=4, seq_len=seq_len)

    out = Path(args.out)
    run = RunDirectory.create(out.parent, out.name)
    trainer_cfg = TrainerConfig(
        max_epochs=1,
        physical_batch_size=4,
        gradient_accumulation_steps=ACCUM,
        learning_rate=3e-4,
        weight_decay=0.01,
        scheduler="inverse_sqrt",
        warmup_steps=1,
        max_steps=args.steps,
        log_every=1,
        seed=0,
        device=args.device,
    )
    trainer = Trainer(model, batches, trainer_cfg, run_dir=run, run_config=vars(args))
    metrics = trainer.train()
    assert torch.isfinite(torch.tensor(metrics["train_loss"])), "non-finite training loss"
    assert trainer.global_step == args.steps, (trainer.global_step, args.steps)

    # Checkpoint -> reload into a FRESH model -> logits must match bitwise.
    ckpt_path = trainer.save(run.checkpoints / "smoke.pt", train_metrics=metrics)
    reloaded = PolyT5ForConditionalGeneration(PolyT5Config.from_yaml(args.model_config))
    resume(ckpt_path, model=reloaded, restore_rng=False)
    reloaded.to(trainer.device)

    probe = {k: v.to(trainer.device) for k, v in batches[0].items()}
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        logits_a = model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"],
                         labels=probe["labels"]).logits
        logits_b = reloaded(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"],
                            labels=probe["labels"]).logits
    assert torch.equal(logits_a, logits_b), "reloaded model produced different logits"

    summary = trainer.summary()
    print("\n=== smoke_train summary ===")
    print(f"  model config            : {args.model_config}")
    print(f"  parameters              : {summary['num_parameters']:,}")
    print(f"  physical batch size     : {summary['physical_batch_size']}")
    print(f"  grad accumulation steps : {summary['gradient_accumulation_steps']}")
    print(f"  effective batch size    : {summary['effective_batch_size']}")
    print(f"  sequence length         : {summary['sequence_length']}")
    print(f"  optimizer steps         : {summary['global_step']}")
    print(f"  device / amp            : {summary['device']} / {summary['amp_dtype']}")
    print(f"  peak VRAM               : {summary['peak_vram'] or 'n/a (cpu)'}")
    print(f"  tokens/second           : {summary['tokens_per_second']:,}")
    print(f"  final train loss        : {metrics['train_loss']:.4f}")
    print(f"  checkpoint round-trip   : OK ({ckpt_path})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
