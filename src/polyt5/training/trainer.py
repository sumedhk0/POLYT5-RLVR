"""A plain-PyTorch trainer for polyT5 pretraining and fine-tuning.

Ground truth from Sahu et al. (npj AI 2026):
    * Pretraining: batch size 450, AdamW, token-level cross-entropy,
      a checkpoint after every epoch, a single NVIDIA L40S.
    * Fine-tuning: AdamW, LR 3e-4, weight decay 0.01, batch size 16,
      cross-entropy with padded labels replaced by -100, evaluation at the
      end of every epoch.

Everything the paper does NOT state (LR schedule, warmup, mixed precision,
gradient accumulation, clipping, seeds) is a ``[AMBIGUITY]`` — flagged on the
corresponding :class:`TrainerConfig` default below.

Our hardware is a 12 GB RTX 4080 Laptop GPU, so the paper's batch size of 450
is reproduced as ``physical_batch_size x gradient_accumulation_steps`` (e.g.
8 x 56 = 448) and is never hard-coded: configs set the two factors, and
``target_effective_batch_size`` asserts the product.

The loop is deliberately readable — it is a research artifact meant to be
inspected and modified, not a framework.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from polyt5.training.checkpoint import resume as _resume
from polyt5.training.checkpoint import rotate_checkpoints, save_checkpoint
from polyt5.training.optim import build_optimizer, build_scheduler
from polyt5.utils import get_logger, memory_stats, seed_everything, select_device

Batch = dict[str, torch.Tensor]


@dataclass
class TrainerConfig:
    """All experimental knobs. Nothing here is hard-coded in the loop."""

    max_epochs: int
    physical_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    # [AMBIGUITY] Clipping is not mentioned in the paper; 1.0 is the common
    # transformer default and protects the fp16/bf16 paths.
    max_grad_norm: float | None = 1.0
    # [AMBIGUITY] The paper never mentions mixed precision. We default to bf16
    # AMP (no loss-scaling pathologies, supported on our Ada GPU) to fit
    # training in 12 GB; AMP is only active on CUDA.
    amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" | "fp16" | "off"
    # [AMBIGUITY] The paper states no LR schedule at all. Default is
    # "inverse_sqrt" — the original T5 pretraining schedule — since polyT5 is
    # a T5. Fine-tuning configs should override to "constant".
    scheduler: str = "inverse_sqrt"
    # [AMBIGUITY] Warmup is not mentioned in the paper.
    warmup_steps: int = 0
    log_every: int = 50
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    keep_last_checkpoints: int = 3
    max_steps: int | None = None
    # [AMBIGUITY] The paper reports no seeds.
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    # Set this to the paper's batch size (450) in configs: it asserts that
    # physical_batch_size * gradient_accumulation_steps reproduces it.
    target_effective_batch_size: int | None = None
    effective_batch_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.effective_batch_size = self.physical_batch_size * self.gradient_accumulation_steps
        if (
            self.target_effective_batch_size is not None
            and self.effective_batch_size != self.target_effective_batch_size
        ):
            raise ValueError(
                "Effective batch size mismatch: physical_batch_size "
                f"{self.physical_batch_size} x gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps} = {self.effective_batch_size}, "
                f"but target_effective_batch_size is {self.target_effective_batch_size}."
            )
        if self.amp_dtype not in ("bf16", "fp16", "off"):
            raise ValueError(f"amp_dtype must be 'bf16', 'fp16' or 'off', got {self.amp_dtype!r}")


class Trainer:
    """Seq2seq trainer over any iterable of dict batches.

    Batches must be dicts with ``input_ids``, ``attention_mask`` and
    ``labels`` (labels padded with -100). Any ``torch.utils.data.DataLoader``
    producing those works; so does a plain list of dicts (used in tests).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[Batch],
        config: TrainerConfig,
        *,
        val_loader: Iterable[Batch] | None = None,
        run_dir: Any = None,
        tokenizer_path: str | Path | None = None,
        tokenizer_sha256: str | None = None,
        run_config: dict[str, Any] | None = None,
        logger: Any = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.tokenizer_path = tokenizer_path
        self.tokenizer_sha256 = tokenizer_sha256
        self.run_config = run_config or {}
        self.logger = logger or get_logger("polyt5.trainer")

        # Seed BEFORE any RNG-consuming training op (dropout). resume_from()
        # afterwards overrides this with the checkpoint's RNG snapshot.
        seed_everything(config.seed)

        self.device = select_device(config.device)
        self.model = model.to(self.device)

        self.optimizer = build_optimizer(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            name=config.scheduler,
            num_training_steps=self._estimate_total_steps(),
            num_warmup_steps=config.warmup_steps,
        )

        # AMP: autocast on CUDA only; GradScaler ONLY for fp16 (bf16 has the
        # fp32 exponent range and needs no loss scaling).
        self.amp_enabled = (
            config.amp and config.amp_dtype != "off" and self.device.startswith("cuda")
        )
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and config.amp_dtype == "fp16"
        )

        self.epoch = -1  # last completed epoch
        self.global_step = 0  # optimizer steps taken
        self._start_epoch = 0
        self._best_val_loss = float("inf")
        self._last_tokens_per_second = 0.0
        self._last_seq_len: int | None = None

    # ------------------------------------------------------------- internals
    def _estimate_total_steps(self) -> int:
        """Optimizer-step horizon for schedulers that need one."""
        if self.config.max_steps is not None:
            return self.config.max_steps
        try:
            batches_per_epoch = len(self.train_loader)  # type: ignore[arg-type]
        except TypeError:
            return 10_000  # sized loaders unavailable; harmless for constant/inverse_sqrt
        steps_per_epoch = math.ceil(
            batches_per_epoch / self.config.gradient_accumulation_steps
        )
        return max(1, steps_per_epoch * self.config.max_epochs)

    def _to_device(self, batch: Batch) -> Batch:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forward_loss(self, batch: Batch) -> torch.Tensor:
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
        return out.loss

    def _batch_weight(self, batch: Batch) -> int:
        """Weight of this batch in the epoch's mean loss.

        The base trainer reports a TOKEN-level mean, so the weight is the
        number of label positions that are not ``-100``. Subclasses whose
        batches are not token-scored (e.g. a regression head) override this;
        the epoch mean is then over whatever unit they return, and their
        docstring says which.
        """
        return int((batch["labels"] != -100).sum().item())

    def _optimizer_step(self) -> None:
        """Unscale -> clip -> step -> scaler update -> zero -> scheduler."""
        if self.config.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)  # no-op unless fp16 scaling is on
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()  # per OPTIMIZER step, never per micro-batch
        self.global_step += 1

    def _lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _log(self, record: dict[str, Any]) -> None:
        if self.run_dir is not None:
            self.run_dir.log_metrics(record)

    # ------------------------------------------------------------ public API
    def train_epoch(self, epoch: int) -> dict[str, Any]:
        """One pass over ``train_loader`` with gradient accumulation."""
        cfg = self.config
        self.model.train()
        start = time.perf_counter()
        loss_sum = 0.0  # sum of (batch token-mean loss * batch token count)
        token_sum = 0
        micro_idx = 0

        iterator = iter(self.train_loader)
        batch = next(iterator, None)
        while batch is not None:
            if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                break
            next_batch = next(iterator, None)
            is_last = next_batch is None

            batch = self._to_device(batch)
            loss = self._forward_loss(batch)
            # Divide by the accumulation window so the accumulated gradient
            # equals the gradient of the mean over the whole effective batch.
            scaled = loss / cfg.gradient_accumulation_steps
            self.scaler.scale(scaled).backward()

            num_tokens = self._batch_weight(batch)
            token_sum += num_tokens
            loss_sum += loss.item() * num_tokens
            self._last_seq_len = int(batch["input_ids"].shape[1])
            micro_idx += 1

            # A trailing partial window at the end of the epoch still steps.
            if micro_idx % cfg.gradient_accumulation_steps == 0 or is_last:
                self._optimizer_step()
                if self.global_step % cfg.log_every == 0:
                    elapsed = time.perf_counter() - start
                    self._log(
                        {
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "train_loss": loss.item(),
                            "lr": self._lr(),
                            "tokens_per_second": round(token_sum / max(elapsed, 1e-9), 1),
                        }
                    )
            batch = next_batch

        epoch_seconds = time.perf_counter() - start
        self._last_tokens_per_second = token_sum / max(epoch_seconds, 1e-9)
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "train_loss": loss_sum / max(token_sum, 1),  # token-level mean
            "tokens_per_second": round(self._last_tokens_per_second, 1),
            "epoch_seconds": round(epoch_seconds, 3),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Token-weighted mean cross-entropy over ``val_loader``."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        loss_sum = 0.0
        token_sum = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss = self._forward_loss(batch)
            num_tokens = self._batch_weight(batch)
            loss_sum += loss.item() * num_tokens
            token_sum += num_tokens
        return {"val_loss": loss_sum / max(token_sum, 1)}

    def train(self) -> dict[str, Any]:
        """Full training run. Returns the final epoch's metrics."""
        cfg = self.config
        final: dict[str, Any] = {}
        for epoch in range(self._start_epoch, cfg.max_epochs):
            train_metrics = self.train_epoch(epoch)
            self.epoch = epoch

            val_metrics: dict[str, Any] = {}
            if self.val_loader is not None and (epoch + 1) % cfg.eval_every_epochs == 0:
                val_metrics = self.evaluate()

            record = {
                **train_metrics,
                "val_loss": val_metrics.get("val_loss"),
                "lr": self._lr(),
            }
            if self.device.startswith("cuda"):
                record.update(memory_stats(self.device))
            self._log(record)
            self.logger.info(
                "epoch %d | step %d | train_loss %.4f | val_loss %s | lr %.3g",
                epoch,
                self.global_step,
                record["train_loss"],
                f"{record['val_loss']:.4f}" if record["val_loss"] is not None else "-",
                record["lr"],
            )

            if self.run_dir is not None and (epoch + 1) % cfg.save_every_epochs == 0:
                self.save(train_metrics=train_metrics, val_metrics=val_metrics or None)
                val_loss = val_metrics.get("val_loss")
                if val_loss is not None and val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self.save(
                        path=Path(self.run_dir.checkpoints) / "best.pt",
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                rotate_checkpoints(self.run_dir.checkpoints, cfg.keep_last_checkpoints)

            final = record
            if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                break
        return final

    def save(
        self,
        path: str | Path | None = None,
        *,
        train_metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Write a full resume checkpoint (defaults to epoch_XXXX.pt in run_dir)."""
        if path is None:
            if self.run_dir is None:
                raise ValueError("No path given and no run_dir to save into")
            path = Path(self.run_dir.checkpoints) / f"epoch_{max(self.epoch, 0):04d}.pt"
        model_config = getattr(self.model, "config", None)
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.run_config,
            model_config=model_config.to_dict() if model_config is not None else None,
            tokenizer_path=self.tokenizer_path,
            tokenizer_sha256=self.tokenizer_sha256,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

    def resume_from(self, path: str | Path) -> None:
        """Restore model/optimizer/scheduler/RNG and continue at the right step."""
        epoch, global_step = _resume(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=True,
        )
        self.epoch = epoch
        self._start_epoch = epoch + 1
        self.global_step = global_step
        self.logger.info("resumed from %s at epoch %d, step %d", path, epoch, global_step)

    def summary(self) -> dict[str, Any]:
        """The run report: parameters, batch shape, VRAM, throughput."""
        model_config = getattr(self.model, "config", None)
        if hasattr(self.model, "num_parameters"):
            num_params = self.model.num_parameters()
        else:
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "num_parameters": num_params,
            "physical_batch_size": self.config.physical_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": self.config.effective_batch_size,
            "sequence_length": self._last_seq_len
            or (getattr(model_config, "n_positions", None) if model_config else None),
            "peak_vram": memory_stats(self.device),
            "tokens_per_second": round(self._last_tokens_per_second, 1),
            "device": self.device,
            "global_step": self.global_step,
            "amp": self.amp_enabled,
            "amp_dtype": self.config.amp_dtype if self.amp_enabled else "off",
        }
