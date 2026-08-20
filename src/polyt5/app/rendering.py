"""Structure drawing and table flattening for the polyT5 demo app.

A chemist reads a polymer repeat unit as a picture, not as a token string, so
the app draws every candidate. Two details matter here:

* **Notation.** polyT5 writes chain ends as ``[At]`` because the ``selfies``
  package cannot encode ``*``. Astatine is a representation device, never real
  chemistry, so the drawing converts back to the ``[*]`` form first -- a
  chemist reads ``*`` as "the chain continues here", and a drawing captioned
  with astatine atoms would be actively misleading.
* **Availability.** ``rdkit.Chem.Draw`` loads extra native modules, and on a
  Windows machine with Application Control enabled one of them can be blocked
  at import time. That is a *capability* question, not an error condition, so
  the import is guarded and :data:`RENDERING_AVAILABLE` reports the answer.
  With drawing unavailable every function here returns ``None`` and the API
  returns ``svg: null`` -- the app still works, it just shows text.

Everything in this module is total: adversarial or unparseable input returns
``None`` and nothing ever raises or prints.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from polyt5.chemistry import at_to_star

__all__ = [
    "RENDERING_AVAILABLE",
    "RENDERING_UNAVAILABLE_REASON",
    "psmiles_to_svg",
    "summary_table",
]

try:  # pragma: no cover - the failure branch depends on the host machine
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    #: Whether structure drawing is usable in this process.
    RENDERING_AVAILABLE = True
    #: Why drawing is unavailable, or ``None`` when it is available.
    RENDERING_UNAVAILABLE_REASON: str | None = None
except Exception as exc:  # pragma: no cover - see the module docstring
    Chem = None  # type: ignore[assignment]
    rdMolDraw2D = None  # type: ignore[assignment]
    RENDERING_AVAILABLE = False
    RENDERING_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"

# MolDraw2DSVG emits an XML prolog, which is illegal inside an HTML document.
_XML_DECLARATION_RE = re.compile(r"^\s*<\?xml[^>]*\?>\s*", re.IGNORECASE)

# Fields copied verbatim from a CandidateRecord into a UI row.
_RECORD_FIELDS = (
    "raw_pselfies",
    "psmiles",
    "canonical_psmiles",
    "passed_sv",
    "passed_tsd",
    "passed_dd",
    "passed_pv",
    "reproducible",
    "failure_stage",
    "sa_score",
)


def psmiles_to_svg(
    psmiles: str,
    *,
    width: int = 320,
    height: int = 240,
    highlight_termini: bool = True,
) -> str | None:
    """Draw a polymer repeat unit as an inlineable SVG string.

    Args:
        psmiles: A PSMILES string in either terminus notation. ``[At]`` is
            rewritten to ``[*]`` before drawing.
        width: Canvas width in pixels. The SVG keeps a matching ``viewBox``,
            so CSS can scale it freely.
        height: Canvas height in pixels.
        highlight_termini: Highlight the chain-end atoms so the two positions
            where the repeat unit joins its neighbours are obvious.

    Returns:
        The SVG markup with its XML declaration stripped, or ``None`` when
        drawing is unavailable, the input is not a string, or RDKit cannot
        parse it. Never raises.

    Note:
        The canvas background is left transparent so the page's own card
        colour shows through in both light and dark themes; the caller is
        responsible for putting the drawing on a light surface, because RDKit
        draws bonds in near-black.
    """
    if not RENDERING_AVAILABLE:
        return None
    if not isinstance(psmiles, str) or not psmiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(at_to_star(psmiles))
        if mol is None:
            return None
        highlight: list[int] = []
        if highlight_termini:
            # A "[*]" terminus parses as a wildcard atom, atomic number 0.
            highlight = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
        options = drawer.drawOptions()
        options.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception:
        return None

    if not isinstance(svg, str):
        return None
    cleaned = _XML_DECLARATION_RE.sub("", svg).strip()
    return cleaned or None


def summary_table(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten screening records into plain dicts for the UI.

    Args:
        records: An iterable of :class:`polyt5.evaluation.CandidateRecord`
            objects, or of mappings with the same field names.

    Returns:
        One dict per record, in input order, with an added ``index`` key and
        ``pselfies`` aliased from ``raw_pselfies`` (the UI's vocabulary). Any
        entry that is neither a record nor a mapping is skipped rather than
        raising.
    """
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if isinstance(record, dict):
            source: dict[str, Any] = record
        elif hasattr(record, "to_dict"):
            try:
                source = record.to_dict()
            except Exception:
                continue
        else:
            continue
        row: dict[str, Any] = {"index": index}
        for name in _RECORD_FIELDS:
            row[name] = source.get(name)
        row["pselfies"] = source.get("raw_pselfies", source.get("pselfies"))
        rows.append(row)
    return rows
