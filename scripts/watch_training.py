"""Live training monitor: progress bar, loss trend, throughput, ETA, GPU.

Reads a run's ``metrics.jsonl`` as the trainer appends to it, so it attaches to
an already-running job and costs the training process nothing. Nothing is
imported from torch and no GPU context is created — this is a reader.

Usage:
    python scripts/watch_training.py                      # newest run under results/
    python scripts/watch_training.py --run results/pretrain_medium_polyone_92m
    python scripts/watch_training.py --once               # one snapshot, no loop
    python scripts/watch_training.py --interval 30        # refresh every 30 s

Press Ctrl-C to detach; the training job is unaffected.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Eighth-blocks for the inline loss sparkline.
SPARK = "▁▂▃▄▅▆▇█"
CLEAR = "\033[2J\033[H"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _newest_run(results_root: Path) -> Path | None:
    """Return the run directory whose metrics file was written most recently."""
    candidates = [p.parent for p in results_root.glob("*/metrics.jsonl")]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "metrics.jsonl").stat().st_mtime)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read metrics.jsonl, tolerating a partially-written trailing line."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # The trainer is mid-write; that row arrives next refresh.
                continue
    return rows


def _load_plan(run: Path) -> dict[str, Any]:
    """Recover total-step count from the run's manifest and config.

    Returns an empty dict when the numbers are not recoverable; the display then
    degrades to "no total known" rather than inventing a denominator.
    """
    plan: dict[str, Any] = {}
    manifest = run / "manifest.json"
    config = run / "config.yaml"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            plan["n_train"] = data.get("n_train_sequences") or data.get("n_corpus_sequences")
            plan["params"] = data.get("model_parameters")
            plan["effective_batch"] = data.get("effective_batch_size")
            plan["device"] = (data.get("device") or {}).get("name")
        except Exception:
            pass
    if config.exists():
        try:
            import yaml

            cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            train = {**cfg.get("training", {}), **cfg.get("train", {})}
            plan.setdefault("effective_batch", train.get("target_effective_batch_size"))
            if not plan.get("effective_batch"):
                pb = train.get("physical_batch_size")
                ga = train.get("gradient_accumulation_steps", 1)
                plan["effective_batch"] = (pb or 0) * (ga or 1) or None
            plan["max_epochs"] = train.get("max_epochs")
        except Exception:
            pass
    n_train = plan.get("n_train")
    batch = plan.get("effective_batch")
    epochs = plan.get("max_epochs")
    if n_train and batch and epochs:
        plan["steps_per_epoch"] = n_train // batch
        plan["total_steps"] = plan["steps_per_epoch"] * epochs
    return plan


def _gpu() -> str:
    """One-line GPU summary, or '' when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return ""
        util, used, total = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
        return f"{util}% util · {int(used) / 1024:.1f}/{int(total) / 1024:.1f} GB"
    except Exception:
        return ""


def _sparkline(values: list[float], width: int = 48) -> str:
    """Render a loss trend as unicode eighth-blocks."""
    if len(values) < 2:
        return ""
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]
    lo, hi = min(sampled), max(sampled)
    if hi - lo < 1e-12:
        return SPARK[0] * len(sampled)
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / (hi - lo) * (len(SPARK) - 1)))]
                   for v in sampled)


def _bar(fraction: float, width: int = 42) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def render(run: Path) -> str:
    """Build the full monitor frame for one run."""
    rows = _read_rows(run / "metrics.jsonl")
    plan = _load_plan(run)
    lines: list[str] = []
    width = min(shutil.get_terminal_size((100, 30)).columns, 100)

    lines.append(f"{BOLD}{run.name}{RESET}")
    meta = []
    if plan.get("params"):
        meta.append(f"{plan['params']:,} params")
    if plan.get("effective_batch"):
        meta.append(f"effective batch {plan['effective_batch']}")
    if plan.get("device"):
        meta.append(str(plan["device"]))
    if meta:
        lines.append(f"{DIM}{' · '.join(meta)}{RESET}")
    lines.append("─" * width)

    if not rows:
        lines.append("waiting for the first metrics row…")
        gpu = _gpu()
        if gpu:
            lines.append(f"GPU        {gpu}")
        return "\n".join(lines)

    last = rows[-1]
    step = last.get("global_step", 0)
    total = plan.get("total_steps")

    if total:
        frac = step / total
        lines.append(f"progress   {_bar(frac)} {frac:6.1%}   step {step:,}/{total:,}")
    else:
        lines.append(f"progress   step {step:,} (total unknown)")

    epoch = last.get("epoch")
    if epoch is not None and plan.get("max_epochs"):
        lines.append(f"epoch      {epoch + 1} of {plan['max_epochs']}")

    losses = [r["train_loss"] for r in rows if isinstance(r.get("train_loss"), (int, float))]
    if losses:
        first, now = losses[0], losses[-1]
        trend = _sparkline(losses)
        lines.append(f"train loss {now:.4f}   {DIM}(from {first:.4f}){RESET}")
        if trend:
            lines.append(f"           {trend}")

    val_rows = [r for r in rows if r.get("val_loss") is not None]
    if val_rows:
        vals = ", ".join(f"e{r['epoch'] + 1}:{r['val_loss']:.4f}" for r in val_rows[-4:])
        lines.append(f"val loss   {val_rows[-1]['val_loss']:.4f}   {DIM}({vals}){RESET}")

    # Rate from the last two rows, which reflects current conditions rather
    # than an average dragged down by startup.
    if len(rows) >= 2:
        a, b = rows[-2], rows[-1]
        try:
            dt = (datetime.fromisoformat(b["wall_utc"])
                  - datetime.fromisoformat(a["wall_utc"])).total_seconds()
            dstep = b["global_step"] - a["global_step"]
            if dt > 0 and dstep > 0:
                rate = dstep / dt
                extra = f"   {last['tokens_per_second']:,.0f} target tok/s" \
                    if last.get("tokens_per_second") else ""
                lines.append(f"rate       {rate:.2f} steps/s{extra}")
                if total and step < total:
                    eta = (total - step) / rate
                    done = datetime.now().timestamp() + eta
                    lines.append(f"eta        {_duration(eta)} remaining   "
                                 f"{DIM}(~{datetime.fromtimestamp(done):%a %H:%M}){RESET}")
        except Exception:
            pass

    if last.get("lr") is not None:
        lines.append(f"lr         {last['lr']:.3e}")

    mem = {k: last[k] for k in ("max_allocated_gb", "max_reserved_gb") if k in last}
    if mem:
        lines.append("peak vram  " + "  ".join(f"{k.replace('_gb', '')} {v:.2f} GB"
                                               for k, v in mem.items()))
    gpu = _gpu()
    if gpu:
        lines.append(f"gpu        {gpu}")

    age = time.time() - (run / "metrics.jsonl").stat().st_mtime
    stale = f"   {DIM}(last row {_duration(age)} ago){RESET}" if age > 120 else ""
    lines.append(f"{DIM}rows {len(rows)}{RESET}{stale}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=None,
                        help="Run directory. Defaults to the most recently updated one.")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--interval", type=float, default=15.0, help="Refresh seconds.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    args = parser.parse_args(argv)

    run = args.run
    if run is None:
        run = _newest_run(args.results_root)
        if run is None:
            print(f"no runs with metrics.jsonl under {args.results_root}")
            return 1
    if not run.exists():
        print(f"no such run directory: {run}")
        return 1

    if args.once:
        print(render(run))
        return 0

    try:
        while True:
            frame = render(run)
            sys.stdout.write(CLEAR + frame + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\ndetached (training is unaffected)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
