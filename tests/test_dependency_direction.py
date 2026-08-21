"""Automated guard for the one-way dependency rule between the Phase-3
RL/rewards extension and the supervised codebase.

Spec Section 6: "Dependency direction is one-way: ``rl/`` -> {model,
tokenization, chemistry, generation, evaluation, inference}. Nothing in the
supervised codebase imports ``rl/``." Plan Global Constraints: "Nothing in
those packages may import ``rl/`` or ``rewards/``." This is Task 9's
automated version of the plan's Step 3, which asks a human to run:

    grep -rn "polyt5.rl\\|polyt5.rewards" src/polyt5/model src/polyt5/data \\
        src/polyt5/chemistry src/polyt5/generation src/polyt5/evaluation \\
        src/polyt5/training src/polyt5/tokenization

**That literal command does not actually report "no matches" on this repo**,
which is the defect that motivated writing a real test instead of trusting
the manual step: an unescaped ``.`` in a basic regex is "any character", so
the substring pattern ``polyt5.rl`` also matches the project's own PyPI-style
distribution name ``polyt5-rlvr`` wherever it appears in a string literal
(``src/polyt5/data/sources.py``'s HTTP User-Agent, ``src/polyt5/data/
tokenized_corpus.py``'s ``importlib.metadata.version("polyt5-rlvr")``,
``src/polyt5/chemistry/_sa_compat.py``'s docstring), and ``polyt5.rewards``
matches a legitimate cross-package Sphinx docstring reference
(``:class:`polyt5.rewards.ArmReward``` in ``src/polyt5/evaluation/
sweep.py``). None of those is an import -- the rule this test actually
enforces is about real ``import``/``from ... import`` statements, so this
parses each file's AST instead of grepping raw text.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The supervised packages that must never import ``polyt5.rl`` or
#: ``polyt5.rewards`` -- the exact directory list from the plan's Step 3 grep
#: command (``.superpowers/sdd/2026-08-20-grpo-rlvr/task-9-brief.md``).
SUPERVISED_PACKAGES: tuple[str, ...] = (
    "model", "data", "chemistry", "generation", "evaluation", "training", "tokenization",
)

#: Module roots the supervised codebase may not import, or import from.
FORBIDDEN_ROOTS: tuple[str, ...] = ("polyt5.rl", "polyt5.rewards")


def _imported_module_names(path: Path) -> list[str]:
    """Every dotted module a file imports, from real ``import`` statements only.

    Relative ``from . import x`` / ``from .. import y`` inside ``rl/`` or
    ``rewards/`` themselves never resolve to a supervised package, so they
    are irrelevant here; ``node.module`` is ``None`` for a bare
    ``from . import x`` and is skipped rather than crashing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.append(node.module)
    return names


def _violations_in_package(package: str) -> list[str]:
    package_dir = REPO_ROOT / "src" / "polyt5" / package
    found: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        for name in _imported_module_names(path):
            if any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS):
                found.append(f"{path.relative_to(REPO_ROOT)}: import {name}")
    return found


def test_supervised_package_dirs_exist():
    """A typo in ``SUPERVISED_PACKAGES`` would silently check nothing."""
    for package in SUPERVISED_PACKAGES:
        assert (REPO_ROOT / "src" / "polyt5" / package).is_dir(), package


def test_no_supervised_package_imports_rl_or_rewards():
    """The supervised codebase must never import ``polyt5.rl``/``polyt5.rewards``.

    This is the real, automated version of the one-way dependency rule --
    "the supervised codebase must not know RL exists" -- checked by parsing
    every ``.py`` file's AST in each supervised package, not by a raw-text
    grep (see the module docstring for why that would report false
    positives on this repo).
    """
    violations: list[str] = []
    for package in SUPERVISED_PACKAGES:
        violations.extend(_violations_in_package(package))
    assert not violations, (
        "supervised code must never import polyt5.rl / polyt5.rewards:\n" + "\n".join(violations)
    )


def test_rl_package_itself_is_exempt_from_its_own_check():
    """Sanity check on the checker: ``rl/`` legitimately imports ``rewards``-
    adjacent and supervised code, so it is deliberately NOT one of
    :data:`SUPERVISED_PACKAGES` -- this pins that omission is intentional,
    not an oversight, so a future edit cannot "fix" it by adding ``rl`` back.
    """
    assert "rl" not in SUPERVISED_PACKAGES
    assert "rewards" not in SUPERVISED_PACKAGES
