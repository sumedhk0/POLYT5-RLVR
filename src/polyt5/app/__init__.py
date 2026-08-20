"""Local web application for interacting with the trained polyT5 models.

Our counterpart to Figure S13 of Sahu et al., which shows polyT5 wrapped in "an
AI framework for natural language interaction". This package provides the same
affordance for the reproduction: a single-page app that loads our checkpoints,
drives both fine-tuned tasks conversationally, draws the generated polymers, and
applies the paper's SV -> TSD -> DD -> PV evaluation cascade live.

============  =========================================================
Module        Responsibility
============  =========================================================
``server``    The FastAPI application factory, endpoints, and lazy
              checkpoint loading with tokenizer-provenance checks.
``intents``   Deterministic, dependency-free natural-language parsing.
              Regex and keyword rules only -- no LLM, no network.
``rendering`` RDKit structure drawing (guarded: a machine without a
              usable ``rdkit.Chem.Draw`` degrades to ``svg: null``).
============  =========================================================

The app is a research demo: no accounts, no database, no external calls, no
telemetry. It binds to 127.0.0.1 and runs fully offline.

Example:
    >>> from polyt5.app import create_app
    >>> app = create_app(tokenizer_path="artifacts/tokenizer/polyt5_vocab.json")
    >>> app.title
    'polyT5 local demo'
"""

from __future__ import annotations

from polyt5.app.intents import Intent, format_reply, parse_intent
from polyt5.app.rendering import RENDERING_AVAILABLE, psmiles_to_svg, summary_table
from polyt5.app.server import MAX_CANDIDATES, create_app

__all__ = [
    "MAX_CANDIDATES",
    "RENDERING_AVAILABLE",
    "Intent",
    "create_app",
    "format_reply",
    "parse_intent",
    "psmiles_to_svg",
    "summary_table",
]
