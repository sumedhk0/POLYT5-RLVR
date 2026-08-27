r"""Repo-relative path resolution that survives crossing operating systems.

Fourteen call sites had their own copy of this two-line function, and every copy
had the same defect: none normalised backslashes. ``frozen_baseline.json`` records
its artifact paths as ``artifacts\tokenizer\polyt5_vocab.json`` because it was
written on Windows, where ``str(Path)`` uses backslash separators. POSIX treats a
backslash as an ordinary filename character, so each of those paths resolved to one
nonexistent file and every artifact lookup failed the moment the study moved to
WSL2.

The function lives here so the next script gets it for free rather than
reintroducing the bug.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_under"]


def resolve_under(root: Path, path: str | Path) -> Path:
    """Resolve ``path`` under ``root`` unless it is already absolute.

    Backslashes become forward slashes first, so a path recorded on Windows
    resolves correctly on POSIX and vice versa.

    Args:
        root: Directory that relative paths are taken against, normally the repo root.
        path: Absolute path, or one relative to ``root``, in either separator style.

    Returns:
        ``path`` itself when absolute, otherwise ``root / path``.

    Note:
        A genuine POSIX filename containing a backslash is mangled by this. That is
        a deliberate trade for a repository that records checkpoint and config paths
        and never adversarial filenames -- the alternative is being unable to read
        artifact records written on the other OS.
    """
    normalised = Path(str(path).replace("\\", "/"))
    return normalised if normalised.is_absolute() else root / normalised
