#!/usr/bin/env python3
"""Live dashboard for the round-1 GRPO arm chain.

Reads each arm's ``metrics.csv`` straight off disk, so it never touches the
trainer and can be started, stopped and restarted at will.

    python scripts/watch_rl.py              # refresh every 10s
    python scripts/watch_rl.py --once       # one frame, then exit
    python scripts/watch_rl.py --interval 30

``unique_fraction`` is coloured because it is the metric four arms have already
destroyed while their own reward climbed: it ended at 0.535 (``accuracy``),
0.516 (``novelty``) and 0.221 (``synthesisability``). Red there is not
decoration, it is this study's central failure mode showing up live.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Round-1 arms, in the order they were run. Anything else found on disk (the round-2
#: kl_coef sweep, and whatever comes after) is discovered rather than listed here --
#: a hardcoded list silently omits new runs, which is how the sweep's first launch
#: went unmonitored.
ROUND_1 = [
    "validity",
    "control",
    "accuracy",
    "novelty",
    "synthesisability",
    "composite",
    "constraint",
]
MAX_STEPS = 2000
SPARK = "▁▂▃▄▅▆▇█"

C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C = dict.fromkeys(C, "")


def discover() -> list[str]:
    """Every run on disk: round-1 arms in order, then anything else alphabetically.

    Discovered rather than listed so a new experiment appears without editing this
    file. The round-2 kl_coef sweep writes to ``results/grpo_composite_kl05`` and
    friends, which a hardcoded list would silently omit.
    """
    found = {
        p.name[len("grpo_"):]
        for p in (REPO / "results").glob("grpo_*")
        if p.is_dir() and (p / "metrics.csv").is_file()
    }
    ordered = [a for a in ROUND_1 if a in found]
    return ordered + sorted(found - set(ordered))


def kl_coef(arm: str) -> float | None:
    """The run's kl_coef, shown when it differs from round 1's 0.02.

    It is the swept variable in round 2, so a progress line without it cannot be
    told apart from its neighbours.
    """
    path = REPO / "results" / f"grpo_{arm}" / "config.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = (loaded.get("train") or {}).get("kl_coef")
        return float(value) if value is not None else None
    except Exception:
        return None


def read_rows(arm: str) -> list[dict]:
    """Metric rows for one arm, or empty if it has not started."""
    path = REPO / "results" / f"grpo_{arm}" / "metrics.csv"
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return [r for r in csv.DictReader(handle) if r.get("step")]
    except (OSError, csv.Error):
        return []  # mid-write is normal; the next refresh will catch up


def num(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def spark(values: list, width: int = 26) -> str:
    """A sparkline over the last ``width`` non-null values."""
    vals = [v for v in values if v is not None][-width:]
    if len(vals) < 2:
        return C["grey"] + "·" * 8 + C["reset"]
    low, high = min(vals), max(vals)
    if high - low < 1e-12:
        return C["grey"] + SPARK[0] * len(vals) + C["reset"]
    out = []
    for value in vals:
        idx = int((value - low) / (high - low) * (len(SPARK) - 1))
        out.append(SPARK[min(idx, len(SPARK) - 1)])
    return "".join(out)


def bar(fraction: float, width: int) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    remainder = int((fraction * width - filled) * 8)
    body = "█" * filled
    if remainder and filled < width:
        body += "▏▎▍▌▋▊▉█"[remainder - 1]
    return body.ljust(width, "·")


def human(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def recent_rate(rows: list[dict], window: int = 6) -> float | None:
    """Seconds per step over the last few rows.

    Deliberately not the lifetime average: ``composite`` has run anywhere between
    24 and 98 s/step inside one session, so a lifetime figure produces a
    confidently wrong ETA.
    """
    points = []
    for row in rows[-(window + 1) :]:
        try:
            points.append((int(row["step"]), dt.datetime.fromisoformat(row["wall_utc"])))
        except (KeyError, ValueError):
            continue
    if len(points) < 2:
        return None
    (first_step, first_time), (last_step, last_time) = points[0], points[-1]
    if last_step <= first_step:
        return None
    return (last_time - first_time).total_seconds() / (last_step - first_step)


def gpu_line() -> str:
    query = (
        "utilization.gpu,memory.used,memory.total,clocks.sm,temperature.gpu"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        util, used, total, clock, temp = (
            x.strip() for x in result.stdout.strip().splitlines()[0].split(",")
        )
        colour = C["green"] if int(util) > 50 else C["yellow"] if int(util) > 10 else C["red"]
        return (
            f"{colour}{util:>3}%{C['reset']} util   "
            f"{int(used) / 1024:.1f}/{int(total) / 1024:.0f} GB   {clock} MHz   {temp}°C"
        )
    except Exception:
        return C["grey"] + "nvidia-smi unavailable" + C["reset"]


def live_arm() -> str | None:
    """Which arm the trainer is on, read from the process table."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "train_grpo.py"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if "--arm" in parts:
                return parts[parts.index("--arm") + 1]
    except Exception:
        pass
    return None


def metric_row(label, value, fmt, series, *, warn=None, good=None) -> str:
    if value is None:
        return f"  {C['grey']}{label:<21}       —{C['reset']}"
    colour = ""
    if warn is not None and value <= warn:
        colour = C["red"]
    elif good is not None and value >= good:
        colour = C["green"]
    return (
        f"  {label:<21}{colour}{format(value, fmt):>8}{C['reset']}  "
        f"{C['cyan']}{spark(series)}{C['reset']}"
    )


def frame() -> str:
    width = min(shutil.get_terminal_size((92, 40)).columns, 98)
    live = live_arm()
    lines = [f"{C['bold']}{C['blue']}┌{' polyT5 · GRPO round 1 ':─^{width - 2}}┐{C['reset']}"]

    arms = discover()
    width_name = max((len(a) for a in arms), default=16) + 1
    seen_extra = False
    for arm in arms:
        rows = read_rows(arm)
        if not rows:
            continue
        if arm not in ROUND_1 and not seen_extra:
            seen_extra = True
            lines.append(f"{C['grey']}   round 2 · kl_coef sweep{C['reset']}")
        step = max(int(r["step"]) for r in rows)
        is_live = arm == live
        complete = step >= MAX_STEPS
        mark = (
            f"{C['green']}●{C['reset']}"
            if is_live
            else f"{C['blue']}✓{C['reset']}"
            if complete
            else f"{C['grey']}○{C['reset']}"
        )
        name = f"{C['bold']}{arm}{C['reset']}" if is_live else arm
        pad = " " * (width_name - len(arm))
        coef = kl_coef(arm)
        tag = f"{C['grey']}kl{coef:g}{C['reset']} " if coef is not None and coef != 0.02 else ""
        rate = recent_rate(rows)
        eta = "done" if complete else human((MAX_STEPS - step) * rate) if rate else "?"
        rate_text = f"{rate:.0f}s/step" if rate and not complete else "—"
        colour = C["green"] if is_live else C["blue"] if complete else C["grey"]
        lines.append(
            f" {mark} {name}{pad}{tag}{colour}{bar(step / MAX_STEPS, 18)}{C['reset']} "
            f"{step:>4}/{MAX_STEPS} {step / MAX_STEPS:>4.0%} {rate_text:>9} {eta:>6}"
        )

    if not arms:
        lines.append(f"  {C['grey']}no run has written metrics yet{C['reset']}")

    rows = read_rows(live) if live else []
    if rows:
        last = rows[-1]
        lines.append(f"{C['blue']}├{'─' * (width - 2)}┤{C['reset']}")
        lines.append(f"  {C['bold']}{live}{C['reset']} {C['grey']}· live telemetry{C['reset']}")
        for label, key, fmt, warn, good in [
            ("reward_mean", "reward_mean", ".4f", None, 0.9),
            ("unique_fraction", "unique_fraction", ".4f", 0.6, None),
            ("zero_variance_groups", "zero_variance_group_fraction", ".4f", None, None),
            ("clip_fraction", "clip_fraction", ".4f", None, None),
            ("kl", "kl", ".5f", None, None),
            ("mean_length", "mean_length", ".1f", None, None),
        ]:
            lines.append(
                metric_row(
                    label,
                    num(last, key),
                    fmt,
                    [num(r, key) for r in rows],
                    warn=warn,
                    good=good,
                )
            )
        unique = num(last, "unique_fraction")
        flat = num(last, "zero_variance_group_fraction")
        if unique is not None and unique < 0.6:
            lines.append(
                f"  {C['grey']}!{C['reset']} {C['red']}diversity collapsing{C['reset']} "
                f"{C['grey']}(accuracy ended 0.535, novelty 0.516, "
                f"synthesisability 0.221){C['reset']}"
            )
        if flat is not None and flat > 0.5:
            lines.append(
                f"  {C['grey']}!{C['reset']} {C['yellow']}{flat:.0%} of groups flat"
                f"{C['reset']} {C['grey']}— those steps yield no gradient{C['reset']}"
            )

    lines.append(f"{C['blue']}├{'─' * (width - 2)}┤{C['reset']}")
    lines.append(f"  GPU  {gpu_line()}")
    state = (
        f"{C['green']}training {live}{C['reset']}"
        if live
        else f"{C['red']}no trainer running{C['reset']}"
    )
    lines.append(f"  {state}   {C['grey']}{dt.datetime.now():%H:%M:%S}{C['reset']}")
    lines.append(f"{C['blue']}└{'─' * (width - 2)}┘{C['reset']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10.0, help="refresh seconds")
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    args = parser.parse_args(argv)

    if args.once:
        print(frame())
        return 0
    try:
        sys.stdout.write("\033[?25l")
        while True:
            sys.stdout.write("\033[H\033[J" + frame() + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
