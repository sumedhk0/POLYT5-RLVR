"""Tests for the training system: optimizer/scheduler, checkpointing, Trainer.

CPU-only and fast: everything uses the deliberately tiny debug config
(configs/model/polyt5_tiny.yaml). One CUDA smoke test is skipped when no GPU
is present. Data loaders are duck-typed plain lists of dict batches — the
trainer must accept any iterable of {"input_ids", "attention_mask", "labels"}
dicts, so the real collators (owned by another track) are never imported.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import pytest
import torch

from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.training import (
    Trainer,
    TrainerConfig,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    resume,
    rotate_checkpoints,
    save_checkpoint,
)
from polyt5.utils import RunDirectory, seed_everything

REPO = Path(__file__).resolve().parents[1]
TINY_YAML = REPO / "configs" / "model" / "polyt5_tiny.yaml"


# --------------------------------------------------------------------- helpers
def tiny_config(dropout: float = 0.0) -> PolyT5Config:
    cfg = PolyT5Config.from_yaml(TINY_YAML)
    cfg.dropout_rate = dropout
    return cfg


def build_model(seed: int = 0, dropout: float = 0.0) -> PolyT5ForConditionalGeneration:
    seed_everything(seed)
    return PolyT5ForConditionalGeneration(tiny_config(dropout=dropout))


def make_batch(
    batch_size: int = 2,
    src_len: int = 12,
    tgt_len: int = 12,
    vocab: int = 458,
    seed: int = 1234,
) -> dict[str, torch.Tensor]:
    """A synthetic batch with NO ignored label positions (equal token counts)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(2, vocab, (batch_size, src_len), generator=g),
        "attention_mask": torch.ones(batch_size, src_len, dtype=torch.long),
        "labels": torch.randint(2, vocab, (batch_size, tgt_len), generator=g),
    }


def base_trainer_config(**overrides) -> TrainerConfig:
    kwargs = dict(
        max_epochs=1,
        physical_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_grad_norm=None,
        amp=False,
        scheduler="constant",
        warmup_steps=0,
        log_every=1000,
        seed=0,
        device="cpu",
    )
    kwargs.update(overrides)
    return TrainerConfig(**kwargs)


def capture_grads_on_step(trainer: Trainer, model: torch.nn.Module) -> dict:
    """Patch the optimizer so the first .step() snapshots post-clip gradients."""
    captured: dict[str, torch.Tensor] = {}
    original = trainer.optimizer.step

    def spy(*args, **kwargs):
        if not captured:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    captured[name] = param.grad.detach().clone()
        return original(*args, **kwargs)

    trainer.optimizer.step = spy
    return captured


# ---------------------------------------------------------------------- optim
def test_one_batch_training_step():
    model = build_model(seed=0)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    batch = make_batch()

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert out.loss is not None and torch.isfinite(out.loss)

    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed, "optimizer.step() changed no parameters"


def test_weight_decay_param_groups():
    model = build_model(seed=0)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert len(optimizer.param_groups) == 2
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # Every trainable parameter appears exactly once across the two groups.
    total = sum(len(g["params"]) for g in optimizer.param_groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)
    # LayerNorm weights (1-D) and the shared embedding must be in no-decay.
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert id(model.shared.weight) in no_decay_ids
    for p in decay["params"]:
        assert p.ndim >= 2, "1-D tensors (biases/norms) must not be weight-decayed"


def test_loss_decreases_when_memorizing_one_batch():
    model = build_model(seed=0, dropout=0.0)
    batch = make_batch(batch_size=4, seed=7)
    loader = [batch] * 30

    def eval_loss() -> float:
        model.eval()
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        return out.loss.item()

    initial = eval_loss()
    trainer = Trainer(model, loader, base_trainer_config(learning_rate=2e-3, physical_batch_size=4))
    trainer.train()
    final = eval_loss()
    assert final < 0.5 * initial, f"loss did not halve: {initial:.3f} -> {final:.3f}"


@pytest.mark.filterwarnings("ignore:Seems like `optimizer.step\\(\\)`:UserWarning")
def test_gradient_accumulation_equivalence():
    """K micro-batches of size B must produce ~the same gradients as one K*B batch."""
    model_a = build_model(seed=0, dropout=0.0)
    model_b = copy.deepcopy(model_a)

    b1 = make_batch(batch_size=2, seed=11)
    b2 = make_batch(batch_size=2, seed=22)
    big = {k: torch.cat([b1[k], b2[k]], dim=0) for k in b1}

    trainer_a = Trainer(
        model_a,
        [b1, b2],
        base_trainer_config(gradient_accumulation_steps=2, max_steps=1),
    )
    grads_a = capture_grads_on_step(trainer_a, model_a)
    trainer_a.train()

    trainer_b = Trainer(
        model_b,
        [big],
        base_trainer_config(physical_batch_size=4, max_steps=1),
    )
    grads_b = capture_grads_on_step(trainer_b, model_b)
    trainer_b.train()

    assert grads_a and grads_b and grads_a.keys() == grads_b.keys()
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=1e-4, atol=1e-6)


def test_effective_batch_size_target_mismatch_raises():
    with pytest.raises(ValueError, match=r"450") as exc:
        TrainerConfig(
            max_epochs=1,
            physical_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=1e-3,
            weight_decay=0.01,
            target_effective_batch_size=450,
        )
    assert "32" in str(exc.value)  # names the actual effective batch size too


def test_effective_batch_size_computed_and_target_match():
    cfg = TrainerConfig(
        max_epochs=1,
        physical_batch_size=8,
        gradient_accumulation_steps=56,
        learning_rate=1e-3,
        weight_decay=0.01,
        target_effective_batch_size=448,
    )
    assert cfg.effective_batch_size == 448


@pytest.mark.filterwarnings("ignore:Seems like `optimizer.step\\(\\)`:UserWarning")
def test_gradient_clipping_bounds_grad_norm():
    max_norm = 0.01
    model = build_model(seed=0, dropout=0.0)
    batch = make_batch(batch_size=2, seed=3)

    # Sanity: the unclipped gradient norm genuinely exceeds the bound.
    probe = copy.deepcopy(model)
    out = probe(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"])
    out.loss.backward()
    raw_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in probe.parameters()])
    )
    assert raw_norm > max_norm

    trainer = Trainer(model, [batch], base_trainer_config(max_grad_norm=max_norm, max_steps=1))
    grads = capture_grads_on_step(trainer, model)
    trainer.train()
    clipped_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in grads.values()])
    )
    assert clipped_norm <= max_norm * (1 + 1e-4)


# ------------------------------------------------------------------ scheduler
def test_scheduler_warmup_inverse_sqrt_shape():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    sched = build_scheduler(opt, name="inverse_sqrt", num_training_steps=100, num_warmup_steps=4)
    lrs = []
    for _ in range(16):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Warmup: linear ramp to base lr at step 4.
    assert lrs[0] == pytest.approx(0.25)
    assert lrs[3] == pytest.approx(1.0)
    # Decay: lr(step) = sqrt(warmup / step) for step > warmup.
    assert lrs[8] == pytest.approx(math.sqrt(4 / 9))
    assert lrs[15] == pytest.approx(math.sqrt(4 / 16))
    assert all(
        a >= b for a, b in zip(lrs[3:-1], lrs[4:], strict=True)
    ), "must decay monotonically after warmup"


def test_scheduler_linear_constant_cosine_none():
    def lrs_for(name, steps=10, **kw):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = build_scheduler(opt, name=name, num_training_steps=10, **kw)
        out = []
        for _ in range(steps):
            opt.step()
            sched.step()
            out.append(opt.param_groups[0]["lr"])
        return out

    linear = lrs_for("linear", num_warmup_steps=2)
    assert linear[1] == pytest.approx(1.0)
    assert linear[5] == pytest.approx((10 - 6) / (10 - 2))
    assert linear[9] == pytest.approx(0.0)

    constant = lrs_for("constant", num_warmup_steps=2)
    assert constant[1:] == pytest.approx([1.0] * 9)

    cosine = lrs_for("cosine", num_warmup_steps=2)
    assert cosine[1] == pytest.approx(1.0)
    assert cosine[9] == pytest.approx(0.0, abs=1e-6)
    mid = cosine[5]
    assert 0.0 < mid < 1.0

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    assert build_scheduler(opt, name="none", num_training_steps=10) is None


def test_scheduler_stepped_per_optimizer_step_not_micro_batch():
    model = build_model(seed=0, dropout=0.0)
    batches = [make_batch(seed=i) for i in range(8)]
    trainer = Trainer(
        model,
        batches,
        base_trainer_config(
            gradient_accumulation_steps=2, scheduler="inverse_sqrt", warmup_steps=2
        ),
    )
    trainer.train()
    assert trainer.global_step == 4  # 8 micro-batches / accumulation 2
    assert trainer.scheduler is not None
    assert trainer.scheduler.last_epoch == trainer.global_step  # NOT 8


def test_trailing_partial_accumulation_window_still_steps():
    model = build_model(seed=0, dropout=0.0)
    batches = [make_batch(seed=i) for i in range(5)]  # 5 micro-batches, accum 2
    trainer = Trainer(model, batches, base_trainer_config(gradient_accumulation_steps=2))
    trainer.train()
    assert trainer.global_step == 3  # 2 full windows + 1 trailing partial window
    if trainer.scheduler is not None:
        assert trainer.scheduler.last_epoch == 3


# ----------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip(tmp_path):
    model = build_model(seed=0, dropout=0.0)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = build_scheduler(
        optimizer, name="inverse_sqrt", num_training_steps=100, num_warmup_steps=2
    )
    batch = make_batch(seed=5)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        out.loss.backward()
        optimizer.step()
        scheduler.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=3,
        config={"experiment": "roundtrip"},
        model_config=model.config.to_dict(),
        train_metrics={"train_loss": out.loss.item()},
    )
    payload = load_checkpoint(path)
    for key in ("model_state", "optimizer_state", "scheduler_state", "model_config",
                "config", "torch_version", "created_utc", "rng_state"):
        assert key in payload, f"missing checkpoint key: {key}"
    assert payload["epoch"] == 0 and payload["global_step"] == 3

    model2 = build_model(seed=99, dropout=0.0)  # different init on purpose
    optimizer2 = build_optimizer(model2, lr=1e-3, weight_decay=0.01)
    scheduler2 = build_scheduler(
        optimizer2, name="inverse_sqrt", num_training_steps=100, num_warmup_steps=2
    )
    epoch, global_step = resume(
        path, model=model2, optimizer=optimizer2, scheduler=scheduler2
    )
    assert (epoch, global_step) == (0, 3)
    assert scheduler2.last_epoch == scheduler.last_epoch
    assert optimizer2.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

    model.eval()
    model2.eval()
    with torch.no_grad():
        logits1 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["labels"]).logits
        logits2 = model2(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         labels=batch["labels"]).logits
    assert torch.equal(logits1, logits2)

    # Optimizer moments restored.
    state1 = optimizer.state_dict()["state"]
    state2 = optimizer2.state_dict()["state"]
    assert state1.keys() == state2.keys()
    some_key = next(iter(state1))
    torch.testing.assert_close(state1[some_key]["exp_avg"], state2[some_key]["exp_avg"])


def test_save_checkpoint_without_model_config_raises(tmp_path):
    model = build_model(seed=0)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    with pytest.raises(ValueError, match="model_config"):
        save_checkpoint(
            tmp_path / "bad.pt",
            model=model,
            optimizer=optimizer,
            epoch=0,
            global_step=0,
            config={},
            model_config=None,
        )
    assert not (tmp_path / "bad.pt").exists()


def test_resume_matches_uninterrupted_training(tmp_path):
    """Strongest resume test: 4 steps + checkpoint + 4 steps == 8 straight steps."""
    batches = [make_batch(seed=i) for i in range(4)]

    def cfg(max_epochs: int) -> TrainerConfig:
        return base_trainer_config(
            max_epochs=max_epochs,
            max_steps=8,
            scheduler="linear",
            warmup_steps=2,
            weight_decay=0.01,
        )

    # Run A: 2 epochs (8 optimizer steps) uninterrupted. Dropout ON so that RNG
    # restoration is actually exercised.
    model_a = build_model(seed=0, dropout=0.1)
    Trainer(model_a, batches, cfg(max_epochs=2)).train()

    # Run B: 1 epoch, checkpoint, fresh trainer, resume, 1 more epoch.
    model_b = build_model(seed=0, dropout=0.1)
    run_dir = RunDirectory.create(tmp_path, "resume_test")
    Trainer(model_b, batches, cfg(max_epochs=1), run_dir=run_dir).train()
    ckpts = sorted(run_dir.checkpoints.glob("epoch_*.pt"))
    assert ckpts, "trainer did not save an epoch checkpoint"

    model_b2 = build_model(seed=123, dropout=0.1)  # init irrelevant, overwritten
    trainer_b2 = Trainer(model_b2, batches, cfg(max_epochs=2))
    trainer_b2.resume_from(ckpts[-1])
    assert trainer_b2.global_step == 4
    trainer_b2.train()
    assert trainer_b2.global_step == 8

    params_a = dict(model_a.named_parameters())
    for name, param in model_b2.named_parameters():
        torch.testing.assert_close(param, params_a[name], rtol=0, atol=0)


def test_rotate_checkpoints_keeps_last_and_best(tmp_path):
    directory = tmp_path / "ckpts"
    directory.mkdir()
    base = 1_700_000_000
    for i in range(5):
        p = directory / f"epoch_{i:04d}.pt"
        torch.save({"val_metrics": {"val_loss": float(i)}}, p)  # epoch_0000 is best
        os.utime(p, (base + i, base + i))
    best = directory / "best.pt"
    torch.save({"val_metrics": {"val_loss": 0.0}}, best)
    os.utime(best, (base - 100, base - 100))  # oldest file of all

    kept = rotate_checkpoints(directory, keep_last=2)
    remaining = {p.name for p in directory.glob("*.pt")}
    assert remaining == {"epoch_0003.pt", "epoch_0004.pt", "best.pt"}
    assert {p.name for p in kept} == remaining

    # keep_best retains the best-by-metric checkpoint as well.
    for i in range(5):
        p = directory / f"epoch_{i:04d}.pt"
        torch.save({"val_metrics": {"val_loss": float(i)}}, p)
        os.utime(p, (base + i, base + i))
    rotate_checkpoints(directory, keep_last=2, keep_best="val_loss")
    remaining = {p.name for p in directory.glob("*.pt")}
    assert remaining == {"epoch_0000.pt", "epoch_0003.pt", "epoch_0004.pt", "best.pt"}


# -------------------------------------------------------------------- trainer
def test_run_directory_metrics_rows(tmp_path):
    model = build_model(seed=0, dropout=0.0)
    train_batches = [make_batch(seed=i) for i in range(4)]
    val_batches = [make_batch(seed=100)]
    run_dir = RunDirectory.create(tmp_path, "metrics_test")
    trainer = Trainer(
        model, train_batches, base_trainer_config(log_every=2), val_loader=val_batches,
        run_dir=run_dir,
    )
    trainer.train()

    assert run_dir.metrics_jsonl.exists() and run_dir.metrics_csv.exists()
    import csv as csv_mod
    import json

    rows = [json.loads(line) for line in run_dir.metrics_jsonl.read_text().splitlines()]
    expected = {"epoch", "global_step", "train_loss", "val_loss", "lr",
                "tokens_per_second", "epoch_seconds"}
    epoch_rows = [r for r in rows if expected <= set(r)]
    assert epoch_rows, f"no epoch row with keys {expected}; rows={rows}"
    assert epoch_rows[-1]["val_loss"] is not None

    with run_dir.metrics_csv.open(newline="") as fh:
        header = set(next(csv_mod.reader(fh)))
    assert expected <= header


def test_determinism_same_seed_same_loss():
    def run() -> tuple[float, dict[str, torch.Tensor]]:
        model = build_model(seed=0, dropout=0.1)  # dropout ON: seeding must tame it
        batches = [make_batch(seed=i) for i in range(4)]
        trainer = Trainer(model, batches, base_trainer_config(max_epochs=2, seed=0))
        metrics = trainer.train()
        return metrics["train_loss"], {n: p.detach().clone()
                                       for n, p in model.named_parameters()}

    loss1, params1 = run()
    loss2, params2 = run()
    assert loss1 == loss2
    for name in params1:
        assert torch.equal(params1[name], params2[name])


def test_trainer_summary_keys():
    model = build_model(seed=0, dropout=0.0)
    trainer = Trainer(model, [make_batch()], base_trainer_config())
    trainer.train()
    summary = trainer.summary()
    for key in ("num_parameters", "physical_batch_size", "gradient_accumulation_steps",
                "effective_batch_size", "sequence_length", "peak_vram", "tokens_per_second"):
        assert key in summary, f"summary missing {key}"
    assert summary["num_parameters"] == model.num_parameters()
    assert summary["effective_batch_size"] == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_bf16_amp_two_steps():
    from polyt5.utils import memory_stats

    model = build_model(seed=0, dropout=0.0)
    batches = [make_batch(seed=i) for i in range(2)]
    trainer = Trainer(
        model,
        batches,
        base_trainer_config(amp=True, device="cuda", max_steps=2, max_grad_norm=1.0),
    )
    metrics = trainer.train()
    assert math.isfinite(metrics["train_loss"])
    stats = memory_stats("cuda")
    assert stats, "memory_stats returned empty on CUDA"
    assert stats["max_allocated_gb"] > 0
