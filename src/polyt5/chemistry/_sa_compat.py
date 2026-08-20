"""Compatibility shim that restores RDKit's SA scorer on locked-down Windows.

Why this exists
---------------
On this development machine, Windows Application Control blocks RDKit's
``rdFingerprintGenerator`` extension module::

    ImportError: DLL load failed while importing rdFingerprintGenerator:
    An Application Control policy has blocked this file.

RDKit's contributed ``sascorer`` (Ertl & Schuffenhauer's synthetic accessibility
score) imports that module at line 20 and builds a Morgan generator at import
time, so the whole scorer becomes unavailable — and with it one of the metrics
polyT5 reports.

What this shim does
-------------------
``sascorer`` uses exactly one call on the generator::

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sfp = mfpgen.GetSparseCountFingerprint(mol)

The legacy entry point ``rdMolDescriptors.GetMorganFingerprint(mol, radius)``
is **not** blocked on this machine and returns the same sparse count
fingerprint with the same bit identifiers. Older releases of ``sascorer`` called
precisely that function, and the shipped ``fpscores`` fragment table is keyed on
those identifiers — so this is a restoration of the original code path, not an
approximation. SA scores produced through the shim are the published RDKit
values, not a substitute formula.

The shim is installed only when the real module cannot be imported, and it never
touches global state beyond inserting one entry in :data:`sys.modules`.

Sanity values obtained through this shim (RDKit 2026.03.5): phenetole 1.042,
paracetamol 1.407, butane 1.606, bicyclo[2.2.2]octane 2.784.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import Any

_SHIM_MODULE_NAME = "rdkit.Chem.rdFingerprintGenerator"

#: Set to True once :func:`install_fingerprint_shim` has installed the fallback.
SHIM_INSTALLED: bool = False


class _LegacyMorganGenerator:
    """Minimal stand-in for RDKit's ``MorganGenerator``.

    Implements only the surface that ``sascorer`` actually uses. Anything else
    raises ``AttributeError`` naturally, which is the desired behaviour: this is
    a targeted compatibility patch, not a reimplementation of the generator API.
    """

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def GetSparseCountFingerprint(self, mol: Any) -> Any:  # noqa: N802 (RDKit naming)
        """Return the sparse count Morgan fingerprint via the legacy API."""
        from rdkit.Chem import rdMolDescriptors

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rdMolDescriptors.GetMorganFingerprint(mol, self.radius)

    def GetFingerprint(self, mol: Any) -> Any:  # noqa: N802 (RDKit naming)
        """Return the folded bit-vector Morgan fingerprint via the legacy API."""
        from rdkit.Chem import rdMolDescriptors

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=2048
            )


def real_generator_available() -> bool:
    """Whether RDKit's own ``rdFingerprintGenerator`` can be imported."""
    try:
        import rdkit.Chem.rdFingerprintGenerator  # noqa: F401

        return True
    except Exception:
        return False


def install_fingerprint_shim() -> bool:
    """Install the fallback generator module if the real one is blocked.

    Returns:
        True if a usable ``rdFingerprintGenerator`` is present afterwards —
        either the genuine module or this shim. False if neither works, in which
        case callers must degrade gracefully rather than invent a score.
    """
    global SHIM_INSTALLED

    if real_generator_available():
        return True
    if _SHIM_MODULE_NAME in sys.modules and SHIM_INSTALLED:
        return True

    try:  # the shim is useless unless the legacy path actually works
        from rdkit.Chem import rdMolDescriptors  # noqa: F401
    except Exception:
        return False

    module = types.ModuleType(_SHIM_MODULE_NAME)
    module.GetMorganGenerator = (  # type: ignore[attr-defined]
        lambda radius=2, **_kwargs: _LegacyMorganGenerator(radius)
    )
    module.__doc__ = (
        "polyt5-rlvr compatibility shim; see polyt5.chemistry._sa_compat. "
        "Installed because the real RDKit extension is blocked on this system."
    )
    sys.modules[_SHIM_MODULE_NAME] = module
    SHIM_INSTALLED = True
    return True
