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
#: ``polyt5.rewards``. This used to be the plan's Step 3 grep list, which was
#: written from a partial survey and covered only seven of the ten supervised
#: packages -- ``app``, ``inference`` and ``utils`` were unguarded, and
#: ``inference`` is precisely the package ``rl/`` depends on for the predictor
#: and therefore the most plausible future back-edge. The README's and spec
#: section 6's contract is unqualified ("nothing outside ``rl/`` imports
#: ``rl/``"), so the list is now everything under ``src/polyt5/`` except the
#: two exempt packages, and
#: :func:`test_supervised_packages_covers_every_package_on_disk` fails if a new
#: package appears and is not classified either way.
SUPERVISED_PACKAGES: tuple[str, ...] = (
    "app", "chemistry", "data", "evaluation", "generation", "inference", "model",
    "tokenization", "training", "utils",
)

#: The Phase-3 packages the rule is ABOUT; they are deliberately not checked
#: against themselves.
EXEMPT_PACKAGES: tuple[str, ...] = ("rl", "rewards")

#: Module roots the supervised codebase may not import, or import from.
FORBIDDEN_ROOTS: tuple[str, ...] = ("polyt5.rl", "polyt5.rewards")


def _packages_on_disk() -> list[str]:
    """Every importable package directory directly under ``src/polyt5``."""
    root = REPO_ROOT / "src" / "polyt5"
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def _imported_module_names(path: Path) -> list[str]:
    """Every dotted module a file imports, from real ``import`` statements only.

    Relative ``from . import x`` / ``from .. import y`` inside ``rl/`` or
    ``rewards/`` themselves never resolve to a supervised package, so they
    are irrelevant here; ``node.module`` is ``None`` for a bare
    ``from . import x`` and is skipped rather than crashing.

    For ``from a.b import c`` BOTH ``a.b`` and ``a.b.c`` are recorded. Only
    recording ``node.module`` would let ``from polyt5 import rl`` through the
    guard -- it names the forbidden package in the alias, not the module --
    which is exactly the evasion the review of Task 9 found.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
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


def test_supervised_packages_covers_every_package_on_disk():
    """Finding 10: the guard must not silently cover only part of the codebase.

    The original list was the plan's grep list -- 7 of the 10 supervised
    packages -- so ``app``, ``inference`` and ``utils`` were unguarded while
    the stated contract was unqualified. Enumerating the filesystem means a
    NEW package is a test failure until someone classifies it, rather than an
    unguarded hole nobody notices.
    """
    on_disk = set(_packages_on_disk())
    classified = set(SUPERVISED_PACKAGES) | set(EXEMPT_PACKAGES)
    assert on_disk == classified, (
        "every package under src/polyt5 must be either guarded (SUPERVISED_PACKAGES) or "
        f"explicitly exempt (EXEMPT_PACKAGES). Unclassified: {sorted(on_disk - classified)}; "
        f"listed but absent: {sorted(classified - on_disk)}"
    )


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
    assert set(EXEMPT_PACKAGES) == {"rl", "rewards"}


def test_checker_detects_the_from_package_import_module_form(tmp_path):
    """``from polyt5 import rl`` must be caught, not just ``import polyt5.rl``.

    The Task 9 review found this evasion: the forbidden package is named in
    the ImportFrom's ALIAS, while ``node.module`` is only the harmless
    ``polyt5``. A checker that reads ``node.module`` alone reports clean on a
    real violation, which is the one failure mode a guard must not have.
    """
    source = tmp_path / "sneaky.py"
    source.write_text("from polyt5 import rl\n", encoding="utf-8")
    names = _imported_module_names(source)
    assert "polyt5.rl" in names, names

    source.write_text("from polyt5 import rewards as r\n", encoding="utf-8")
    assert "polyt5.rewards" in _imported_module_names(source)


def test_checker_does_not_flag_a_legitimate_sibling_name(tmp_path):
    """Guard against over-correction: ``polyt5.rlvr_utils`` is not ``polyt5.rl``.

    The prefix match must be on a dotted boundary, or closing the evasion
    above would start failing on unrelated modules whose names merely begin
    with the forbidden string -- the same unescaped-substring mistake the
    plan's original grep made.
    """
    source = tmp_path / "innocent.py"
    source.write_text("from polyt5 import rlvr_utils\n", encoding="utf-8")
    names = _imported_module_names(source)
    assert not any(
        name == root or name.startswith(root + ".") for name in names for root in FORBIDDEN_ROOTS
    ), names
