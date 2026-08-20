"""YAML experiment configuration.

Rule for this repository (see ``docs/reproduction.md``): *no experimental
setting lives inside a Python script*. Every run is defined by a YAML file plus
a seed, and the fully-resolved config is written next to the results so that a
run can be replayed exactly.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_INCLUDE_KEY = "_include"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; ``override`` wins."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving ``_include`` and applying overrides.

    ``_include`` may be a single path or a list of paths, resolved relative to
    the including file. Later includes win over earlier ones, and the including
    file wins over all of its includes.

    Args:
        path: Path to the YAML file.
        overrides: Nested dict merged last (e.g. from CLI ``--set`` flags).

    Returns:
        The resolved configuration dictionary.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    includes = raw.pop(_INCLUDE_KEY, [])
    if isinstance(includes, str):
        includes = [includes]

    merged: dict[str, Any] = {}
    for inc in includes:
        merged = _deep_merge(merged, load_config(path.parent / inc))
    merged = _deep_merge(merged, raw)

    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def parse_dotted_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["train.batch_size=8", "seed=1"]`` into a nested dict.

    Values are parsed as YAML scalars, so ``8`` becomes an int and ``true`` a
    bool. This backs the CLI ``--set`` flag of every script in ``scripts/``.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got: {pair!r}")
        key, _, value = pair.partition("=")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    """Write the fully-resolved config beside the results of a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    return path


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Fetch a nested key, raising a clear error if it is missing."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"missing required config key: {dotted_key!r}")
        node = node[part]
    return node
