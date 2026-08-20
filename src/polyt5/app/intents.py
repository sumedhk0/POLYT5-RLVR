"""Deterministic natural-language intent parsing for the polyT5 demo app.

The paper wraps polyT5 in "an AI framework for natural language interaction"
(Figure S13). This module is the equivalent front door for *our* reproduction,
with one deliberate difference: **there is no language model here.** Parsing is
regex and keyword rules only -- no LLM, no network, no learned component -- so
every routing decision is inspectable, reproducible, and unit-testable, and the
whole app runs offline on a laptop with the GPU idle.

The grammar is small on purpose. Five intents:

===========  ==============================================================
``generate`` Sample polymers conditioned on a target Tg.
``predict``  Predict Tg for a structure the user supplied.
``params``   Adjust decoding knobs; applies to the previous intent (see
             :func:`merge_with_history`).
``help``     Explain what the app understands.
``unknown``  Nothing matched -- answered with an explanation of what *would*
             have matched, never with silence.
===========  ==============================================================

Units
-----
The paper reports Tg in Kelvin and the fine-tuning corpus is in Kelvin, so a
**bare number is read as Kelvin**. A Celsius reading is honoured only when the
user writes the unit explicitly (``C``, ``°C``, ``celsius``, ``centigrade``),
in which case it is converted to Kelvin and the conversion is stated in
:attr:`Intent.explanation` -- which the app surfaces in its reply, so the user
is never left guessing which unit was assumed.

# [AMBIGUITY] The paper's Figure S13 chatbot is a screenshot, not a
# specification: it never states how free-text numbers are interpreted, nor
# whether its assistant asks or assumes. Assuming Kelvin matches the training
# targets ("236.0" is Kelvin throughout the corpus); stating the assumption in
# every reply is our addition, because a silent 273 K error is the single most
# damaging thing this interface could do.

Structure recognition
---------------------
A whitespace-delimited token is treated as a chemical structure when it either

* consists entirely of ``[...]`` groups and has at least two of them
  (PSELFIES, e.g. ``[At][C][C][O][At]``), or
* contains a terminus marker -- ``*``, ``[*]`` or ``[At]`` -- anywhere
  (PSMILES, e.g. ``[*]CCO[*]`` or ``[At]CCO[At]``).

# [AMBIGUITY] A terminus-free PSMILES such as ``CCO`` is deliberately NOT
# auto-detected in free text: it is indistinguishable from an English word, and
# a false positive would silently answer the wrong question. Such structures
# are still accepted by ``POST /api/predict``, where the field is unambiguous.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CELSIUS_OFFSET",
    "HELP_TEXT",
    "INTENT_NAMES",
    "Intent",
    "detect_structure_kind",
    "extract_structures",
    "format_reply",
    "merge_with_history",
    "parse_intent",
]

#: 0 degrees Celsius in Kelvin.
CELSIUS_OFFSET = 273.15

#: Every intent name :func:`parse_intent` can return.
INTENT_NAMES: tuple[str, ...] = ("generate", "predict", "params", "help", "unknown")

HELP_TEXT = (
    "I drive the two polyT5 models this repository trained, and I read plain "
    "English.\n"
    "\n"
    "Generation (target Tg -> polymer):\n"
    "  generate 20 polymers with Tg near 450 K\n"
    "  design a polymer with a glass transition of 500\n"
    "  make me 5 candidates around 380 kelvin\n"
    "\n"
    "Prediction (polymer -> Tg):\n"
    "  predict Tg for [*]CCO[*]\n"
    "  what is the glass transition of [At][C][C][O][At]\n"
    "\n"
    "Decoding knobs (applied to the previous request):\n"
    "  use temperature 0.9 and top_p 0.95\n"
    "\n"
    "Numbers without a unit are Kelvin, which is the unit the paper reports. "
    "Write '100 C' or '100 celsius' for Celsius and I will convert it.\n"
    "Structures are recognised as PSELFIES ([At][C][C][O][At]) or as PSMILES "
    "with star/At termini ([*]CCO[*], [At]CCO[At])."
)

_UNKNOWN_EXPLANATION = (
    "I could not map that to one of my two tasks. I understand requests to "
    "generate polymers for a target glass transition temperature (for example "
    "'generate 20 polymers with Tg near 450 K'), requests to predict Tg for a "
    "structure you supply as PSMILES or PSELFIES (for example 'predict Tg for "
    "[*]CCO[*]'), decoding-parameter changes ('use temperature 0.9 and top_p "
    "0.95'), and 'help'. Bare numbers are read as Kelvin."
)

# --------------------------------------------------------------------------
# regexes
# --------------------------------------------------------------------------

_NUMBER = r"-?\d+(?:\.\d+)?"
_CONNECTOR = r"(?:of|is|at|to|=|:|~|near|around|about|approximately|close\s+to)"
_UNIT = r"(?:°\s*)?(k|kelvin|c|celsius|centigrade)\b"

# "Tg near 450 K", "glass transition of 500", "glass transition temperature = 420"
_TG_NAMED_RE = re.compile(
    rf"(?:\btg\b|glass[\s\-]transition(?:\s+temperature)?)\s*"
    rf"(?:{_CONNECTOR}\s*)*({_NUMBER})\s*(?:{_UNIT})?",
    re.IGNORECASE,
)

# "around 380 kelvin", "at 100 C" -- a unit is REQUIRED when Tg is not named.
_TG_UNIT_RE = re.compile(rf"({_NUMBER})\s*{_UNIT}", re.IGNORECASE)

_COUNT_NOUN_RE = re.compile(
    r"\b(\d+)\s+(?:polymers?|candidates?|structures?|molecules?|samples?|monomers?"
    r"|repeat\s+units?|designs?|options?|suggestions?)\b",
    re.IGNORECASE,
)
_COUNT_ASSIGN_RE = re.compile(r"\bn\s*(?:=|:)\s*(\d+)\b", re.IGNORECASE)

_TEMPERATURE_RE = re.compile(
    rf"\b(?:temperature|temp)\b\s*(?:{_CONNECTOR}\s*)*({_NUMBER})", re.IGNORECASE
)
_TOP_P_RE = re.compile(rf"\btop[\s_\-]?p\b\s*(?:{_CONNECTOR}\s*)*({_NUMBER})", re.IGNORECASE)
_TOP_K_RE = re.compile(rf"\btop[\s_\-]?k\b\s*(?:{_CONNECTOR}\s*)*(\d+)", re.IGNORECASE)
_SEED_RE = re.compile(rf"\bseed\b\s*(?:{_CONNECTOR}\s*)*(\d+)", re.IGNORECASE)

_HELP_EXACT_RE = re.compile(
    r"^\s*(?:help|\?+|usage|commands?|options?)\s*[!.?]*\s*$", re.IGNORECASE
)
_HELP_PHRASES = (
    "what can you do",
    "what can i ask",
    "what do you do",
    "how do i use",
    "how does this work",
    "who are you",
    "what are you",
    "show me the commands",
)

_GENERATE_VERBS = (
    "generate",
    "design",
    "make",
    "create",
    "produce",
    "propose",
    "suggest",
    "invent",
    "sample",
    "give me",
    "come up with",
)
_PREDICT_VERBS = ("predict", "estimate", "what is", "what's", "whats", "compute", "calculate")

_BRACKET_GROUP_RE = re.compile(r"\[[^\[\]\s]*\]")
_PURE_BRACKET_RUN_RE = re.compile(r"^(?:\[[^\[\]\s]*\])+$")
_TRAILING_PUNCTUATION = "?!,;:."


@dataclass
class Intent:
    """One parsed user request.

    Attributes:
        name: One of :data:`INTENT_NAMES`.
        params: Extracted arguments. Keys are only present when the user
            actually supplied them, so the server's own defaults stay in
            charge of everything else. Possible keys: ``target_tg`` (Kelvin),
            ``n``, ``structure``, ``kind``, ``temperature``, ``top_p``,
            ``top_k``, ``seed``.
        confidence: Rough 0-1 confidence in the routing decision. It is a
            rule-count heuristic, not a calibrated probability, and exists so
            the UI can flag a shaky parse rather than to gate anything.
        explanation: One or more sentences saying how the message was read,
            including any unit assumption. Never empty.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the intent."""
        return {
            "name": self.name,
            "params": dict(self.params),
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


# --------------------------------------------------------------------------
# structure extraction
# --------------------------------------------------------------------------


def detect_structure_kind(text: str) -> str | None:
    """Classify a single token as ``"pselfies"``, ``"psmiles"``, or ``None``.

    Args:
        text: A candidate structure string, already stripped of surrounding
            prose punctuation.

    Returns:
        ``"pselfies"`` when the token is nothing but two or more ``[...]``
        groups, ``"psmiles"`` when it carries a terminus marker but is not a
        pure bracket run, and ``None`` when it looks like neither.
    """
    if not isinstance(text, str):
        return None
    token = text.strip()
    if not token:
        return None
    if _PURE_BRACKET_RUN_RE.match(token) and len(_BRACKET_GROUP_RE.findall(token)) >= 2:
        return "pselfies"
    if "*" in token or "[At]" in token:
        return "psmiles"
    return None


def extract_structures(message: str) -> list[tuple[str, str]]:
    """Find every chemical structure in a free-text message.

    Args:
        message: The raw user message.

    Returns:
        ``[(structure, kind), ...]`` in the order they appear. Empty when the
        message contains no recognisable structure.
    """
    if not isinstance(message, str):
        return []
    found: list[tuple[str, str]] = []
    for token in message.split():
        stripped = token.strip(_TRAILING_PUNCTUATION)
        kind = detect_structure_kind(stripped)
        if kind is not None:
            found.append((stripped, kind))
    return found


# --------------------------------------------------------------------------
# scalar extraction
# --------------------------------------------------------------------------


def _blank(text: str, span: tuple[int, int]) -> str:
    """Replace ``text[span]`` with spaces so later regexes cannot re-match it."""
    start, end = span
    return text[:start] + " " * (end - start) + text[end:]


def _to_kelvin(value: float, unit: str | None) -> tuple[float, str]:
    """Convert a parsed number to Kelvin and describe the assumption made.

    Args:
        value: The number the user wrote.
        unit: The unit token they wrote, or ``None`` for a bare number.

    Returns:
        ``(kelvin, note)`` where ``note`` is a human sentence about the unit.
    """
    normalized = (unit or "").strip().lower()
    if normalized in {"c", "celsius", "centigrade"}:
        return value + CELSIUS_OFFSET, (
            f"Read {value:g} as degrees Celsius and converted it to "
            f"{value + CELSIUS_OFFSET:.2f} K."
        )
    if normalized in {"k", "kelvin"}:
        return value, f"Read {value:g} K."
    return value, f"Read {value:g} as Kelvin, the unit the paper reports Tg in."


def _extract_target_tg(text: str) -> tuple[float | None, str, str | None]:
    """Pull a target Tg out of the message.

    Args:
        text: Working text (structures already blanked out).

    Returns:
        ``(kelvin, remaining_text, unit_note)``. ``kelvin`` is ``None`` when no
        target was written, in which case ``unit_note`` is ``None`` too.
    """
    for pattern in (_TG_NAMED_RE, _TG_UNIT_RE):
        match = pattern.search(text)
        if match is None:
            continue
        raw, unit = match.group(1), match.group(2)
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - the regex only matches numbers
            continue
        kelvin, note = _to_kelvin(value, unit)
        return kelvin, _blank(text, match.span()), note
    return None, text, None


def _extract_count(text: str) -> tuple[int | None, str]:
    """Pull an explicit candidate count out of the message.

    Args:
        text: Working text (target Tg already blanked out).

    Returns:
        ``(n, remaining_text)``; ``n`` is ``None`` when no count was written.
    """
    for pattern in (_COUNT_NOUN_RE, _COUNT_ASSIGN_RE):
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1)), _blank(text, match.span())
    return None, text


def _extract_overrides(text: str) -> dict[str, Any]:
    """Pull decoding knobs out of the message.

    Args:
        text: Working text (structures, target and count already blanked out,
            so ``glass transition temperature of 500`` can never be misread as
            a sampling temperature).

    Returns:
        A dict with any of ``temperature``, ``top_p``, ``top_k``, ``seed``.
    """
    overrides: dict[str, Any] = {}
    for key, pattern, caster in (
        ("temperature", _TEMPERATURE_RE, float),
        ("top_p", _TOP_P_RE, float),
        ("top_k", _TOP_K_RE, int),
        ("seed", _SEED_RE, int),
    ):
        match = pattern.search(text)
        if match is not None:
            try:
                overrides[key] = caster(match.group(1))
            except ValueError:  # pragma: no cover - regex guarantees numerics
                continue
    return overrides


def _is_help(message: str) -> bool:
    """Return whether the message is asking what the app can do."""
    lowered = message.strip().lower()
    if _HELP_EXACT_RE.match(lowered):
        return True
    return any(phrase in lowered for phrase in _HELP_PHRASES)


def _mentions(message: str, needles: Sequence[str]) -> bool:
    """Return whether any needle occurs in the lower-cased message."""
    lowered = message.lower()
    return any(needle in lowered for needle in needles)


# --------------------------------------------------------------------------
# the parser
# --------------------------------------------------------------------------


def parse_intent(message: str) -> Intent:
    """Parse a user message into a routable :class:`Intent`.

    The rules are applied in a fixed order so the result is a pure function of
    the string: help first, then structures (which force ``predict``), then a
    target Tg (which forces ``generate``), then bare decoding knobs (which
    become ``params``), and finally ``unknown``.

    Args:
        message: Raw user text. ``None``, non-strings and blanks are accepted
            and answered with the ``unknown`` intent.

    Returns:
        An :class:`Intent`. This function never raises and never returns an
        empty :attr:`Intent.explanation`.
    """
    if not isinstance(message, str) or not message.strip():
        return Intent("unknown", {}, 0.0, _UNKNOWN_EXPLANATION)

    if _is_help(message):
        return Intent("help", {}, 1.0, "Interpreted as a request for usage help.")

    working = message
    structures = extract_structures(message)
    for structure, _kind in structures:
        index = working.find(structure)
        if index >= 0:
            working = _blank(working, (index, index + len(structure)))

    target_tg, working, unit_note = _extract_target_tg(working)
    count, working = _extract_count(working)
    overrides = _extract_overrides(working)

    # ---------------------------------------------------------------- predict
    if structures:
        structure, kind = structures[0]
        params: dict[str, Any] = {"structure": structure, "kind": kind}
        params.update(overrides)
        notes = [
            f"Read {structure!r} as {kind.upper()} and routed it to Tg prediction "
            f"(beam search, width 4)."
        ]
        if len(structures) > 1:
            notes.append(
                f"{len(structures) - 1} further structure(s) in the message were ignored; "
                "ask about one at a time."
            )
        confidence = 0.95 if _mentions(message, _PREDICT_VERBS) else 0.8
        return Intent("predict", params, confidence, " ".join(notes))

    # --------------------------------------------------------------- generate
    if target_tg is not None:
        params = {"target_tg": target_tg}
        if count is not None:
            params["n"] = count
        params.update(overrides)
        notes = [f"Routed to conditional generation at Tg = {target_tg:g} K."]
        if unit_note:
            notes.append(unit_note)
        if count is None:
            notes.append("No candidate count given, so the server default is used.")
        wants_generation = _mentions(message, _GENERATE_VERBS)
        confidence = 0.95 if wants_generation else 0.7
        return Intent("generate", params, confidence, " ".join(notes))

    # ----------------------------------------------------------------- params
    if overrides:
        readable = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        return Intent(
            "params",
            dict(overrides),
            0.9,
            f"Read a decoding-parameter change ({readable}). It applies to the "
            "previous request; send it together with a target Tg to start a new one.",
        )

    # ---------------------------------------------------------------- unknown
    hint = _UNKNOWN_EXPLANATION
    if _mentions(message, _GENERATE_VERBS):
        hint = (
            "That looks like a generation request, but I could not find a target "
            "glass transition temperature in it. Try 'generate 20 polymers with "
            "Tg near 450 K'. " + _UNKNOWN_EXPLANATION
        )
    elif _mentions(message, _PREDICT_VERBS):
        hint = (
            "That looks like a prediction request, but I could not find a "
            "structure in it. Give me PSELFIES ([At][C][C][O][At]) or PSMILES "
            "with star/At termini ([*]CCO[*]). " + _UNKNOWN_EXPLANATION
        )
    return Intent("unknown", {}, 0.0, hint)


def merge_with_history(
    intent: Intent, history: Sequence[Mapping[str, Any]] | None
) -> Intent:
    """Resolve a bare parameter change against the most recent real request.

    "use temperature 0.9 and top_p 0.95" on its own is not actionable. This
    walks the conversation backwards, re-parses each user turn, and re-issues
    the newest ``generate``/``predict`` intent with the new knobs merged in --
    which is what a chemist means by "now try that again, but hotter".

    Args:
        intent: The intent parsed from the current message.
        history: Prior turns as ``{"role": ..., "content": ...}`` mappings.
            Non-mappings and non-user roles are skipped. ``None`` is allowed.

    Returns:
        The merged intent, or ``intent`` unchanged when it is not a ``params``
        intent or when no prior actionable request exists.

    Note:
        Only the *message text* is replayed; no server state is kept between
        requests, so the app stays stateless and two browser tabs cannot
        interfere with one another.
        # [AMBIGUITY] The paper shows a chatbot transcript but says nothing
        # about how (or whether) it carries context between turns.
    """
    if intent.name != "params" or not history:
        return intent

    for turn in reversed(list(history)):
        if not isinstance(turn, Mapping):
            continue
        if str(turn.get("role", "user")).lower() != "user":
            continue
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        previous = parse_intent(content)
        if previous.name not in {"generate", "predict"}:
            continue
        params = dict(previous.params)
        params.update(intent.params)
        return Intent(
            previous.name,
            params,
            min(previous.confidence, intent.confidence),
            f"Applied {intent.explanation.split('.')[0].lower()} to the previous "
            f"{previous.name} request. {previous.explanation}",
        )
    return intent


# --------------------------------------------------------------------------
# reply formatting
# --------------------------------------------------------------------------


def _format_number(value: Any, digits: int = 1) -> str | None:
    """Format a number for prose, or return ``None`` when it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return None


def _generate_reply(intent: Intent, data: Mapping[str, Any]) -> str:
    """Compose the prose summary of a generation run."""
    aggregate = data.get("aggregate") or {}
    counts = aggregate.get("counts") or {}
    target = data.get("target_tg", intent.params.get("target_tg"))
    n_input = counts.get("n_input", data.get("n_requested", 0))

    target_text = _format_number(target, 0)
    parts = [
        f"Generated {n_input} candidates for Tg = {target_text} K."
        if target_text is not None
        else f"Generated {n_input} candidates."
    ]
    if counts:
        parts.append(
            f"{counts.get('n_sv', 0)} passed SMILES validity, "
            f"{counts.get('n_tsd', 0)} were not in the training set, "
            f"{counts.get('n_dd', 0)} were distinct within the batch, and "
            f"{counts.get('n_pv', 0)} passed the PSMILES-validity filter "
            "(exactly two [At] termini, each valency 1)."
        )
    sa_mean = _format_number(aggregate.get("sa_mean"), 2)
    if sa_mean is not None:
        parts.append(f"Mean synthetic accessibility {sa_mean}.")
    mean_tg = _format_number(aggregate.get("mean_predicted_tg"), 0)
    if mean_tg is not None:
        parts.append(
            f"Mean predicted Tg {mean_tg} K over "
            f"{aggregate.get('n_predicted', 0)} of them."
        )
        tp_rate = aggregate.get("tp_rate")
        tolerance = _format_number(aggregate.get("tp_tolerance"), 0)
        if tp_rate is not None and tolerance is not None:
            parts.append(
                f"{float(tp_rate) * 100:.0f}% land within +/-{tolerance} K of the target."
            )
    else:
        parts.append("No property model is configured, so no Tg was predicted back.")
    return " ".join(parts)


def _predict_reply(data: Mapping[str, Any]) -> str:
    """Compose the prose summary of a property prediction."""
    canonical = data.get("canonical_psmiles") or data.get("psmiles") or data.get("structure")
    predicted = _format_number(data.get("predicted_tg"), 1)
    parts: list[str] = []
    if predicted is not None:
        parts.append(f"Predicted Tg {predicted} K for {canonical} (beam search, width 4).")
    else:
        raw = data.get("raw_output")
        parts.append(
            f"The property model did not emit a number for {canonical}; it decoded "
            f"{raw!r}, which the paper's pipeline filters out as non-numeric."
        )
    parts.append(
        f"{data.get('n_termini', 0)} terminus atom(s) found; "
        + ("passes" if data.get("passes_pv") else "does not pass")
        + " the PSMILES-validity rule."
    )
    sa_score = _format_number(data.get("sa_score"), 2)
    if sa_score is not None:
        parts.append(f"Synthetic accessibility {sa_score}.")
    return " ".join(parts)


def format_reply(intent: Intent, data: Mapping[str, Any] | None = None) -> str:
    """Turn an intent plus its result payload into one short factual reply.

    Args:
        intent: The routed intent.
        data: The endpoint payload the intent produced, if any.

    Returns:
        A plain-prose sentence or two. Purely factual: every number in the
        string comes from ``data``, nothing is embellished, and no claim is
        made that the payload does not support.
    """
    payload: Mapping[str, Any] = data or {}
    if intent.name == "generate":
        return _generate_reply(intent, payload)
    if intent.name == "predict":
        return _predict_reply(payload)
    if intent.name == "help":
        return HELP_TEXT
    if intent.name == "params":
        return intent.explanation
    return intent.explanation or _UNKNOWN_EXPLANATION
