"""FastAPI application for driving the trained polyT5 models locally.

This is our counterpart to Figure S13 of Sahu et al., where polyT5 is wrapped in
"an AI framework for natural language interaction". It is a **research demo**,
not a product: no accounts, no database, no external API calls, no telemetry, no
persisted state. It binds to 127.0.0.1 and runs fully offline.

What it exposes
---------------
=========================  =================================================
``GET  /``                 The single-page UI.
``GET  /api/health``       Device, feature availability, provenance.
``POST /api/generate``     Target Tg -> polymers, screened through the
                           paper's nested SV -> TSD -> DD -> PV cascade,
                           scored for synthetic accessibility, and (when a
                           property model is configured) predicted back.
``POST /api/predict``      Structure -> Tg, by beam search with width 4.
``POST /api/chat``         Natural-language front door over the two above.
=========================  =================================================

Design rules this module holds to
---------------------------------
* **Lazy.** Checkpoints load on first use, never at import or at ``--help``
  time, and are then cached on ``app.state``. ``/api/health`` answers with
  every model still unloaded, so a laptop can start the app with the GPU idle.
* **Provenance-checked.** Every checkpoint records the sha256 of the
  vocabulary it was trained with. A mismatch against the loaded tokenizer is
  refused loudly, because token ids from a different vocabulary decode into
  plausible-looking nonsense rather than failing.
* **Degrading, not crashing.** A missing checkpoint disables one feature and
  says so through ``/api/health``; it never prevents the app from starting.
* **No tracebacks over the wire.** Every failure becomes JSON with an
  explanatory message and a 4xx/5xx status.
* **Two decoding regimes, never conflated.** Generation samples
  (``do_sample=True``, the paper's "instead of beam search"); prediction uses
  beam search with width 4.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from polyt5.app import rendering
from polyt5.app.intents import HELP_TEXT, Intent, format_reply, merge_with_history, parse_intent
from polyt5.chemistry import (
    SA_AVAILABLE,
    NoveltyIndex,
    canonical_psmiles,
    count_termini,
    pselfies_to_psmiles,
    psmiles_to_pselfies,
    synthetic_accessibility,
    validate_psmiles,
)
from polyt5.data.prepare import format_property_value, parse_property_value
from polyt5.evaluation import apply_filter_cascade, has_valid_termini, target_property_rate
from polyt5.generation import BeamSearchConfig, GenerationConfig, beam_search, generate
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import load_checkpoint
from polyt5.utils import get_logger, select_device

__all__ = [
    "DEFAULT_TOP_P",
    "DEFAULT_TEMPERATURE",
    "MAX_CANDIDATES",
    "AppError",
    "create_app",
    "create_app_from_env",
]

logger = get_logger("polyt5.app")

#: Hard cap on candidates per ``/api/generate`` call.
#:
#: This is a demo, not a sweep runner. 200 samples of up to 200 tokens is about
#: a second on a GPU and a few seconds on CPU, which keeps the page responsive
#: and keeps one browser tab from monopolising the device. The paper's own
#: figures use 10,000 candidates per configuration -- that is what
#: ``scripts/`` and ``polyt5.evaluation.sweep`` are for, not this endpoint.
MAX_CANDIDATES = 200

#: The paper's best "medium" conditional-generation setting (temperature 1.1,
#: top_p 0.75); used as the default so the demo shows the reproduction's
#: headline configuration rather than an arbitrary one.
DEFAULT_TEMPERATURE = 1.1
DEFAULT_TOP_P = 0.75

#: Maximum tokens decoded for a property prediction. Targets are short numeric
#: strings such as ``"236.0"``; 32 leaves generous room without wasting beams.
PREDICTION_MAX_TOKENS = 32

#: Beam width for property prediction -- the paper's value.
PREDICTION_NUM_BEAMS = 4

#: Rows per forward pass when predicting Tg back for a batch of candidates.
PREDICTION_BATCH_SIZE = 32

#: Half-width of the TP acceptance window, in Kelvin. Figure S10 reports the
#: "proportion of candidates with predicted Tg within 500 +- 50 K".
#:
#: # [AMBIGUITY] The paper states that window only for its 500 K case study and
#: # never says whether the tolerance is absolute or relative to the target. We
#: # keep it absolute at 50 K for every target and report the value alongside
#: # the rate, so a reader can always see which window produced the number.
TP_TOLERANCE_K = 50.0

#: A linear repeat unit has exactly two chain ends.
EXPECTED_TERMINI = 2

#: Cap on training-corpus entries indexed for the TSD filter.
#:
#: # [AMBIGUITY] The paper deduplicates against its whole training set. Loading
#: # PI1M's ~890k rows would add tens of seconds to the first request of a demo
#: # session, so the index is capped and ``/api/health`` reports how many
#: # entries were actually indexed -- the number is visible, never silently
#: # assumed to be "all of them".
DEFAULT_CORPUS_LIMIT = 50_000

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class AppError(Exception):
    """An error the app can explain to the user.

    Attributes:
        message: A complete, self-contained explanation. It is sent verbatim
            to the client, so it must never contain a traceback or a path the
            user cannot act on.
        status_code: The HTTP status to answer with.
    """

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        """Initialise the error.

        Args:
            message: The user-facing explanation.
            status_code: HTTP status code, 4xx or 5xx.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class FeatureUnavailable(AppError):
    """A requested feature has no checkpoint configured, or it failed to load."""

    def __init__(self, message: str) -> None:
        """Initialise with a 503 status."""
        super().__init__(message, status_code=503)


# --------------------------------------------------------------------------
# request / response models
# --------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Body of ``POST /api/generate``."""

    target_tg: float = Field(
        ...,
        description="Target glass transition temperature in Kelvin.",
        gt=0.0,
        le=2000.0,
    )
    n: int = Field(20, ge=1, le=MAX_CANDIDATES, description="Number of candidates to sample.")
    temperature: float = Field(DEFAULT_TEMPERATURE, gt=0.0, le=5.0)
    top_p: float = Field(DEFAULT_TOP_P, gt=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1)
    seed: int | None = Field(None, ge=0, description="Seed for reproducible sampling.")
    max_length: int = Field(200, ge=1, le=512, description="Maximum tokens per candidate.")


class Candidate(BaseModel):
    """One generated polymer and everything the cascade learned about it."""

    index: int
    pselfies: str
    psmiles: str | None
    canonical_psmiles: str | None
    passed_sv: bool
    passed_tsd: bool
    passed_dd: bool
    passed_pv: bool
    reproducible: bool
    failure_stage: str | None
    sa_score: float | None
    predicted_tg: float | None
    tg_error: float | None
    svg: str | None


class GenerateAggregate(BaseModel):
    """Batch-level statistics over one generation call."""

    counts: dict[str, int]
    rates: dict[str, float]
    sr_rate: float
    sa_mean: float | None
    sa_scored: int
    tp_rate: float | None
    tp_tolerance: float
    mean_predicted_tg: float | None
    mean_abs_tg_error: float | None
    n_predicted: int


class GenerateResponse(BaseModel):
    """Body of a successful ``POST /api/generate``."""

    target_tg: float
    n_requested: int
    candidates: list[Candidate]
    aggregate: GenerateAggregate
    settings: dict[str, Any]
    elapsed_s: float


class PredictRequest(BaseModel):
    """Body of ``POST /api/predict``."""

    structure: str = Field(..., min_length=1, max_length=4000)
    kind: Literal["psmiles", "pselfies", "auto"] = "auto"


class PredictResponse(BaseModel):
    """Body of a successful ``POST /api/predict``."""

    structure: str
    kind: str
    pselfies: str | None
    psmiles: str | None
    canonical_psmiles: str | None
    valid: bool
    n_termini: int
    passes_pv: bool
    predicted_tg: float | None
    raw_output: str
    sa_score: float | None
    svg: str | None
    elapsed_s: float


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``."""

    message: str = Field(..., min_length=1, max_length=4000)
    history: list[dict[str, Any]] | None = None


class ChatResponse(BaseModel):
    """Body of a successful ``POST /api/chat``."""

    reply: str
    intent: str
    params: dict[str, Any]
    confidence: float
    data: dict[str, Any] | None


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------


class _ModelSlot:
    """One lazily loaded checkpoint (the generation or the prediction model)."""

    def __init__(self, name: str, path: str | Path | None) -> None:
        """Initialise the slot.

        Args:
            name: Human name used in error messages, e.g. ``"generation"``.
            path: Checkpoint path, or ``None`` when the feature is disabled.
        """
        self.name = name
        self.path: Path | None = Path(path) if path is not None else None
        self.model: PolyT5ForConditionalGeneration | None = None
        self.config: PolyT5Config | None = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None

    @property
    def configured(self) -> bool:
        """Whether a checkpoint path was supplied at all."""
        return self.path is not None

    @property
    def present(self) -> bool:
        """Whether the configured checkpoint actually exists on disk."""
        return self.path is not None and self.path.is_file()

    @property
    def loaded(self) -> bool:
        """Whether the model is in memory."""
        return self.model is not None

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable provenance summary of this slot."""
        return {
            "configured": self.configured,
            "path": str(self.path) if self.path is not None else None,
            "exists": self.present,
            "loaded": self.loaded,
            "error": self.error,
            **{k: v for k, v in self.metadata.items() if k != "model_config"},
        }


class AppResources:
    """Lazily loaded models, tokenizer, and training index shared by requests.

    Nothing here is touched until an endpoint needs it, and everything is
    cached afterwards. A single lock serialises loading so two concurrent
    requests cannot both pay for (or race on) the same checkpoint.
    """

    def __init__(
        self,
        *,
        generation_checkpoint: str | Path | None,
        prediction_checkpoint: str | Path | None,
        tokenizer_path: str | Path | None,
        training_corpus: str | Path | None,
        device: str = "auto",
        corpus_limit: int = DEFAULT_CORPUS_LIMIT,
    ) -> None:
        """Initialise the resource holder without loading anything.

        Args:
            generation_checkpoint: Path to the Tg-conditioned generation model.
            prediction_checkpoint: Path to the Tg property-prediction model.
            tokenizer_path: Path to the vocabulary JSON artifact.
            training_corpus: Path to the corpus used for the TSD filter.
            device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
            corpus_limit: Maximum training entries to index.
        """
        self.generation = _ModelSlot("generation", generation_checkpoint)
        self.prediction = _ModelSlot("prediction", prediction_checkpoint)
        self.tokenizer_path: Path | None = (
            Path(tokenizer_path) if tokenizer_path is not None else None
        )
        self.corpus_path: Path | None = (
            Path(training_corpus) if training_corpus is not None else None
        )
        self.corpus_limit = corpus_limit
        self._device_preference = device
        self._device: str | None = None
        self._tokenizer: PolyT5Tokenizer | None = None
        self._tokenizer_error: str | None = None
        self._training_index: NoveltyIndex | None = None
        self._training_index_size: int | None = None
        self._corpus_error: str | None = None
        self._lock = threading.Lock()

    # -- device ------------------------------------------------------------

    @property
    def device(self) -> str:
        """Resolve (once) and return the torch device string.

        Note:
            Resolution calls ``torch.cuda.is_available()``, which queries the
            driver but allocates nothing, so ``/api/health`` can report the
            device without loading a model onto it.
        """
        if self._device is None:
            try:
                self._device = select_device(self._device_preference)
            except Exception as exc:
                logger.warning("device selection failed (%s); falling back to cpu", exc)
                self._device = "cpu"
        return self._device

    # -- tokenizer ---------------------------------------------------------

    @property
    def tokenizer_sha256(self) -> str | None:
        """The sha256 of the loaded vocabulary, or ``None`` if unavailable."""
        try:
            return self.tokenizer().sha256
        except AppError:
            return None

    def tokenizer(self) -> PolyT5Tokenizer:
        """Return the tokenizer, loading it on first use.

        Returns:
            The cached :class:`~polyt5.tokenization.PolyT5Tokenizer`.

        Raises:
            FeatureUnavailable: If no vocabulary is configured, or the file is
                missing or corrupt.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is not None:  # pragma: no cover - race guard
                return self._tokenizer
            if self.tokenizer_path is None:
                self._tokenizer_error = "no tokenizer configured"
                raise FeatureUnavailable(
                    "No tokenizer vocabulary is configured. Start the server with "
                    "--tokenizer pointing at artifacts/tokenizer/polyt5_vocab.json."
                )
            if not self.tokenizer_path.is_file():
                self._tokenizer_error = f"missing file: {self.tokenizer_path}"
                raise FeatureUnavailable(
                    f"The tokenizer vocabulary {self.tokenizer_path} does not exist. "
                    "Run scripts/build_tokenizer.py or pass --tokenizer."
                )
            try:
                self._tokenizer = PolyT5Tokenizer.from_file(self.tokenizer_path)
            except Exception as exc:
                self._tokenizer_error = str(exc)
                raise FeatureUnavailable(
                    f"The tokenizer vocabulary {self.tokenizer_path} could not be "
                    f"loaded: {exc}"
                ) from exc
            self._tokenizer_error = None
            logger.info(
                "tokenizer loaded: %s (%d tokens, sha256 %s)",
                self.tokenizer_path,
                self._tokenizer.vocab_size,
                self._tokenizer.sha256[:16],
            )
            return self._tokenizer

    # -- models ------------------------------------------------------------

    def model(self, slot: _ModelSlot) -> tuple[PolyT5ForConditionalGeneration, PolyT5Config]:
        """Return a slot's model, loading and verifying it on first use.

        Args:
            slot: The generation or prediction slot.

        Returns:
            ``(model, model_config)``, with the model in ``eval`` mode on the
            resolved device.

        Raises:
            FeatureUnavailable: If the slot is unconfigured, the file is
                missing, the tokenizer hash does not match, or loading fails.
        """
        if slot.model is not None and slot.config is not None:
            return slot.model, slot.config

        if not slot.configured:
            raise FeatureUnavailable(
                f"No {slot.name} checkpoint is configured, so that feature is "
                f"disabled. Restart with --{slot.name}-checkpoint to enable it."
            )
        if not slot.present:
            slot.error = "checkpoint file not found"
            raise FeatureUnavailable(
                f"The {slot.name} checkpoint {slot.path} does not exist, so that "
                "feature is disabled. Train it first, or point the server at a "
                "different checkpoint."
            )

        tokenizer = self.tokenizer()
        with self._lock:
            if slot.model is not None and slot.config is not None:  # pragma: no cover
                return slot.model, slot.config
            try:
                state = load_checkpoint(slot.path, map_location="cpu")
            except Exception as exc:
                slot.error = str(exc)
                raise FeatureUnavailable(
                    f"The {slot.name} checkpoint {slot.path} could not be read: {exc}"
                ) from exc

            checkpoint_sha = state.get("tokenizer_sha256")
            if checkpoint_sha and checkpoint_sha != tokenizer.sha256:
                slot.error = "tokenizer sha256 mismatch"
                raise FeatureUnavailable(
                    f"Refusing to serve the {slot.name} model: tokenizer mismatch. "
                    f"The checkpoint {slot.path} was trained with vocabulary "
                    f"{checkpoint_sha[:16]} but the loaded tokenizer is "
                    f"{tokenizer.sha256[:16]}. Token ids from a different "
                    "vocabulary decode into plausible-looking nonsense instead of "
                    "failing, so this is refused rather than warned about."
                )

            try:
                config = PolyT5Config.from_dict(state["model_config"])
                model = PolyT5ForConditionalGeneration(config)
                model.load_state_dict(state["model_state"])
                model.to(self.device)
                model.eval()
            except Exception as exc:
                slot.error = str(exc)
                raise FeatureUnavailable(
                    f"The {slot.name} model could not be instantiated from "
                    f"{slot.path}: {exc}"
                ) from exc

            slot.model = model
            slot.config = config
            slot.error = None
            slot.metadata = {
                "epoch": state.get("epoch"),
                "global_step": state.get("global_step"),
                "tokenizer_sha256": checkpoint_sha,
                "d_model": config.d_model,
                "num_layers": config.num_layers,
                "n_positions": config.n_positions,
                "created_utc": state.get("created_utc"),
            }
            logger.info(
                "%s model loaded from %s onto %s (step %s)",
                slot.name,
                slot.path,
                self.device,
                state.get("global_step"),
            )
            return model, config

    # -- training index ----------------------------------------------------

    def training_index(self) -> NoveltyIndex | None:
        """Return the TSD reference index, building it on first use.

        Returns:
            A :class:`~polyt5.chemistry.NoveltyIndex`, or ``None`` when no
            corpus is configured or it could not be read. ``None`` makes the
            TSD stage a documented no-op rather than a silent failure -- see
            :func:`polyt5.evaluation.apply_filter_cascade`.
        """
        if self._training_index is not None or self.corpus_path is None:
            return self._training_index
        with self._lock:
            if self._training_index is not None:  # pragma: no cover - race guard
                return self._training_index
            if not self.corpus_path.is_file():
                self._corpus_error = f"missing file: {self.corpus_path}"
                logger.warning(
                    "training corpus %s not found; the TSD filter is a no-op",
                    self.corpus_path,
                )
                return None
            try:
                entries = list(_read_corpus_psmiles(self.corpus_path, limit=self.corpus_limit))
                self._training_index = NoveltyIndex(entries)
                self._training_index_size = len(self._training_index)
                self._corpus_error = None
                logger.info(
                    "training index built from %s (%d distinct polymers)",
                    self.corpus_path,
                    self._training_index_size,
                )
            except Exception as exc:
                self._corpus_error = str(exc)
                logger.warning("training corpus %s could not be read: %s", self.corpus_path, exc)
                return None
            return self._training_index

    # -- reporting ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return the ``/api/health`` payload without loading any model."""
        tokenizer_sha = self.tokenizer_sha256
        return {
            "status": "ok",
            "device": self.device,
            "models_loaded": {
                "tokenizer": self._tokenizer is not None,
                "generation": self.generation.loaded,
                "prediction": self.prediction.loaded,
            },
            "tokenizer_sha256": tokenizer_sha,
            "checkpoints": {
                "generation": self.generation.describe(),
                "prediction": self.prediction.describe(),
                "tokenizer": {
                    "configured": self.tokenizer_path is not None,
                    "path": str(self.tokenizer_path) if self.tokenizer_path else None,
                    "exists": bool(self.tokenizer_path and self.tokenizer_path.is_file()),
                    "sha256": tokenizer_sha,
                    "error": self._tokenizer_error,
                },
                "training_corpus": {
                    "configured": self.corpus_path is not None,
                    "path": str(self.corpus_path) if self.corpus_path else None,
                    "exists": bool(self.corpus_path and self.corpus_path.is_file()),
                    "indexed": self._training_index_size,
                    "limit": self.corpus_limit,
                    "error": self._corpus_error,
                },
            },
            "features": {
                "generation": self.generation.present and self.tokenizer_path is not None,
                "prediction": self.prediction.present and self.tokenizer_path is not None,
                "rendering": rendering.RENDERING_AVAILABLE,
                "synthetic_accessibility": SA_AVAILABLE,
                "training_deduplication": self.corpus_path is not None
                and self.corpus_path.is_file(),
            },
            "rendering_available": rendering.RENDERING_AVAILABLE,
            "rendering_unavailable_reason": rendering.RENDERING_UNAVAILABLE_REASON,
            "max_candidates": MAX_CANDIDATES,
            "defaults": {
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "n": 20,
                "tp_tolerance": TP_TOLERANCE_K,
            },
        }


def _read_corpus_psmiles(path: Path, *, limit: int) -> Iterable[str]:
    """Yield PSMILES strings from a training corpus file.

    Two layouts are understood, both produced by this repository:

    * ``.jsonl`` with ``{"source": ..., "target": ...}`` rows -- whichever of
      the two fields looks like a structure is taken, so the prediction
      corpus (``PSELFIES -> "236.0"``) and the generation corpus
      (``"236.0" -> PSELFIES``) both work.
    * plain text, one PSELFIES or PSMILES per line.

    Args:
        path: Corpus file.
        limit: Maximum number of entries to yield.

    Yields:
        PSMILES strings, decoded from PSELFIES where necessary.
    """
    is_jsonl = path.suffix.lower() == ".jsonl"
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if yielded >= limit:
                return
            text = line.strip()
            if not text:
                continue
            entry: str | None = None
            if is_jsonl:
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                for key in ("target", "source", "pselfies", "psmiles"):
                    value = row.get(key) if isinstance(row, dict) else None
                    if isinstance(value, str) and "[" in value:
                        entry = value
                        break
            else:
                entry = text
            if not entry:
                continue
            psmiles = pselfies_to_psmiles(entry) if _looks_like_pselfies(entry) else entry
            if psmiles:
                yielded += 1
                yield psmiles


def _looks_like_pselfies(text: str) -> bool:
    """Return whether a string is a pure run of bracket tokens (i.e. PSELFIES).

    Args:
        text: A candidate structure string.

    Returns:
        ``True`` when the string is nothing but two or more ``[...]`` groups
        and carries no ``*`` terminus, which is exactly the PSELFIES shape.
        ``[At]CCO[At]`` and ``[*]CCO[*]`` are PSMILES and return ``False``.
    """
    stripped = text.strip()
    if "*" in stripped or " " in stripped or stripped.count("[") < 2:
        return False
    return _bracket_only(stripped)


def _bracket_only(text: str) -> bool:
    """Return whether ``text`` contains nothing outside ``[...]`` groups."""
    depth = 0
    for char in text:
        if char == "[":
            if depth:
                return False
            depth = 1
        elif char == "]":
            if not depth:
                return False
            depth = 0
        elif depth == 0:
            return False
    return depth == 0


# --------------------------------------------------------------------------
# inference helpers
# --------------------------------------------------------------------------


def _encode(tokenizer: PolyT5Tokenizer, texts: Sequence[str], *, max_length: int, device: str):
    """Encode a batch of source strings into padded tensors on ``device``.

    Args:
        tokenizer: The loaded tokenizer.
        texts: Source strings.
        max_length: Hard cap on the source length.
        device: Torch device string.

    Returns:
        ``(input_ids, attention_mask)`` tensors.
    """
    encoded = tokenizer.batch_encode(
        list(texts), add_eos=True, max_length=max_length, padding=True, truncation=True
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device)
    return input_ids, attention_mask


@torch.no_grad()
def _sample_candidates(
    resources: AppResources, request: GenerateRequest
) -> list[str]:
    """Sample ``n`` PSELFIES candidates for a target Tg.

    Args:
        resources: The app's resource holder.
        request: The validated generation request.

    Returns:
        The decoded model outputs, one per requested candidate.
    """
    model, config = resources.model(resources.generation)
    tokenizer = resources.tokenizer()
    source = format_property_value(request.target_tg)
    input_ids, attention_mask = _encode(
        tokenizer, [source], max_length=config.n_positions, device=resources.device
    )
    output = generate(
        model,
        input_ids,
        attention_mask,
        config=GenerationConfig(
            max_length=request.max_length,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.n,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            decoder_start_token_id=tokenizer.decoder_start_token_id,
            seed=request.seed,
        ),
    )
    return tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True)


@torch.no_grad()
def _predict_values(
    resources: AppResources, sources: Sequence[str]
) -> list[tuple[str, float | None]]:
    """Predict Tg for a batch of PSELFIES strings using beam search, width 4.

    Args:
        resources: The app's resource holder.
        sources: PSELFIES strings, one per structure.

    Returns:
        ``[(raw_decoder_output, value_or_None), ...]`` in input order. A
        non-numeric decode yields ``None``, which is the paper's "invalid or
        non-numeric outputs are filtered" step, not an error.
    """
    if not sources:
        return []
    model, config = resources.model(resources.prediction)
    tokenizer = resources.tokenizer()
    results: list[tuple[str, float | None]] = []
    for start in range(0, len(sources), PREDICTION_BATCH_SIZE):
        chunk = list(sources[start : start + PREDICTION_BATCH_SIZE])
        input_ids, attention_mask = _encode(
            tokenizer, chunk, max_length=config.n_positions, device=resources.device
        )
        output = beam_search(
            model,
            input_ids,
            attention_mask,
            config=BeamSearchConfig(
                num_beams=PREDICTION_NUM_BEAMS,
                max_length=PREDICTION_MAX_TOKENS,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
                decoder_start_token_id=tokenizer.decoder_start_token_id,
            ),
        )
        for text in tokenizer.batch_decode(output.sequences.tolist(), skip_special_tokens=True):
            results.append((text, parse_property_value(text)))
    return results


def _safe_sa(psmiles: str | None) -> float | None:
    """Score synthetic accessibility, returning ``None`` on any failure."""
    if not psmiles or not SA_AVAILABLE:
        return None
    try:
        return synthetic_accessibility(psmiles)
    except Exception:
        return None


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` for an empty sequence."""
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------
# endpoint implementations
# --------------------------------------------------------------------------


def _run_generate(resources: AppResources, request: GenerateRequest) -> dict[str, Any]:
    """Sample, screen, score and back-predict one batch of candidates.

    Args:
        resources: The app's resource holder.
        request: The validated request.

    Returns:
        A dict matching :class:`GenerateResponse`.
    """
    started = time.perf_counter()
    raw = _sample_candidates(resources, request)

    records, counts = apply_filter_cascade(
        raw,
        training_index=resources.training_index(),
        expected_termini=EXPECTED_TERMINI,
        compute_sa=SA_AVAILABLE,
    )

    survivors = [index for index, record in enumerate(records) if record.passed_pv]
    predictions: dict[int, tuple[str, float | None]] = {}
    if survivors and resources.prediction.present:
        try:
            values = _predict_values(resources, [records[i].raw_pselfies for i in survivors])
            predictions = dict(zip(survivors, values, strict=True))
        except FeatureUnavailable as exc:
            # A missing property model is a documented degradation, not a
            # failure of the generation request the user actually made.
            logger.warning("back-prediction skipped: %s", exc.message)

    candidates: list[dict[str, Any]] = []
    predicted_values: list[float] = []
    absolute_errors: list[float] = []
    for index, record in enumerate(records):
        raw_text, value = predictions.get(index, ("", None))
        error = None if value is None else value - request.target_tg
        if value is not None:
            predicted_values.append(value)
            absolute_errors.append(abs(value - request.target_tg))
        drawable = record.canonical_psmiles or record.psmiles
        candidates.append(
            {
                "index": index,
                "pselfies": record.raw_pselfies,
                "psmiles": record.psmiles,
                "canonical_psmiles": record.canonical_psmiles,
                "passed_sv": record.passed_sv,
                "passed_tsd": record.passed_tsd,
                "passed_dd": record.passed_dd,
                "passed_pv": record.passed_pv,
                "reproducible": record.reproducible,
                "failure_stage": record.failure_stage,
                "sa_score": record.sa_score,
                "predicted_tg": value,
                "tg_error": error,
                "svg": rendering.psmiles_to_svg(drawable) if record.passed_sv else None,
                "raw_prediction": raw_text or None,
            }
        )

    sa_scores = [r.sa_score for r in records if r.sa_score is not None]
    counts_dict = counts.to_dict()
    aggregate = {
        "counts": {k: v for k, v in counts_dict.items() if k.startswith("n_")},
        "rates": {k: v for k, v in counts_dict.items() if k.endswith("_rate")},
        "sr_rate": (
            sum(1 for r in records if r.reproducible) / len(records) if records else 0.0
        ),
        "sa_mean": _mean(sa_scores),
        "sa_scored": len(sa_scores),
        "tp_rate": (
            target_property_rate(predicted_values, request.target_tg, TP_TOLERANCE_K)
            if predicted_values
            else None
        ),
        "tp_tolerance": TP_TOLERANCE_K,
        "mean_predicted_tg": _mean(predicted_values),
        "mean_abs_tg_error": _mean(absolute_errors),
        "n_predicted": len(predicted_values),
    }

    return {
        "target_tg": request.target_tg,
        "n_requested": request.n,
        "candidates": candidates,
        "aggregate": aggregate,
        "settings": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "seed": request.seed,
            "max_length": request.max_length,
            "expected_termini": EXPECTED_TERMINI,
            "prompt": format_property_value(request.target_tg),
            "decoding": "sampling (the paper uses sampling, not beam search, for generation)",
        },
        "elapsed_s": round(time.perf_counter() - started, 4),
    }


def _resolve_structure(structure: str, kind: str) -> tuple[str, str]:
    """Normalise a user-supplied structure into ``(pselfies, psmiles)``.

    Args:
        structure: The raw structure string.
        kind: ``"psmiles"``, ``"pselfies"`` or ``"auto"``.

    Returns:
        ``(pselfies, psmiles)`` -- both non-empty.

    Raises:
        AppError: If the structure cannot be interpreted as either notation.
    """
    text = structure.strip()
    if not text:
        raise AppError("The structure is empty. Send a PSMILES or PSELFIES string.")

    resolved = kind
    if kind == "auto":
        resolved = "pselfies" if _bracket_only(text) and text.count("[") >= 2 else "psmiles"

    if resolved == "pselfies":
        psmiles = pselfies_to_psmiles(text)
        if psmiles:
            return text, psmiles
        # Fall through: an [At]-style PSMILES also starts with a bracket.
        resolved = "psmiles"

    pselfies = psmiles_to_pselfies(text)
    if pselfies:
        decoded = pselfies_to_psmiles(pselfies) or text
        return pselfies, decoded

    raise AppError(
        f"Could not read {text!r} as a polymer. Send PSELFIES such as "
        "'[At][C][C][O][At]', or PSMILES with star or At termini such as "
        "'[*]CCO[*]'. Nothing was changed and no model was run."
    )


def _run_predict(resources: AppResources, request: PredictRequest) -> dict[str, Any]:
    """Predict Tg for one user-supplied structure.

    Args:
        resources: The app's resource holder.
        request: The validated request.

    Returns:
        A dict matching :class:`PredictResponse`.
    """
    started = time.perf_counter()
    pselfies, psmiles = _resolve_structure(request.structure, request.kind)

    result = validate_psmiles(psmiles, expected_termini=EXPECTED_TERMINI)
    canonical = result.canonical_psmiles or canonical_psmiles(psmiles)
    termini = count_termini(psmiles)
    passes_pv = has_valid_termini(canonical or psmiles, expected=EXPECTED_TERMINI)

    raw_text, value = "", None
    if resources.prediction.present:
        predicted = _predict_values(resources, [pselfies])
        if predicted:
            raw_text, value = predicted[0]

    return {
        "structure": request.structure,
        "kind": request.kind,
        "pselfies": pselfies,
        "psmiles": psmiles,
        "canonical_psmiles": canonical,
        "valid": bool(result.valid),
        "n_termini": termini,
        "passes_pv": passes_pv,
        "predicted_tg": value,
        "raw_output": raw_text,
        "sa_score": _safe_sa(canonical or psmiles),
        "svg": rendering.psmiles_to_svg(canonical or psmiles),
        "elapsed_s": round(time.perf_counter() - started, 4),
    }


def _generate_request_from_intent(intent: Intent) -> GenerateRequest:
    """Build a validated :class:`GenerateRequest` from a parsed intent.

    Args:
        intent: A ``generate`` intent.

    Returns:
        The request, with every unspecified field left at its default.

    Raises:
        AppError: If the extracted parameters are out of range.
    """
    payload: dict[str, Any] = {"target_tg": intent.params.get("target_tg")}
    for key in ("n", "temperature", "top_p", "top_k", "seed"):
        if key in intent.params:
            payload[key] = intent.params[key]
    try:
        return GenerateRequest(**payload)
    except Exception as exc:
        raise AppError(
            f"I understood a generation request but the numbers do not work: {exc}. "
            f"Candidate counts must be between 1 and {MAX_CANDIDATES}."
        ) from exc


def _run_chat(resources: AppResources, request: ChatRequest) -> dict[str, Any]:
    """Parse a message, dispatch it, and compose a prose reply.

    Args:
        resources: The app's resource holder.
        request: The validated chat request.

    Returns:
        A dict matching :class:`ChatResponse`.
    """
    intent = merge_with_history(parse_intent(request.message), request.history)

    data: dict[str, Any] | None = None
    if intent.name == "generate":
        data = _run_generate(resources, _generate_request_from_intent(intent))
    elif intent.name == "predict":
        data = _run_predict(
            resources,
            PredictRequest(
                structure=str(intent.params.get("structure", "")),
                kind=str(intent.params.get("kind", "auto")),  # type: ignore[arg-type]
            ),
        )

    reply = format_reply(intent, data)
    if intent.name in {"generate", "predict"} and intent.explanation:
        reply = f"{reply} ({intent.explanation})"

    return {
        "reply": reply,
        "intent": intent.name,
        "params": dict(intent.params),
        "confidence": intent.confidence,
        "data": data,
    }


# --------------------------------------------------------------------------
# application factory
# --------------------------------------------------------------------------


def create_app(
    *,
    generation_checkpoint: str | Path | None = None,
    prediction_checkpoint: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    training_corpus: str | Path | None = None,
    device: str = "auto",
    lazy: bool = True,
    corpus_limit: int = DEFAULT_CORPUS_LIMIT,
) -> FastAPI:
    """Build the polyT5 demo application.

    Args:
        generation_checkpoint: Checkpoint for the Tg-conditioned generation
            model. ``None`` disables ``/api/generate``.
        prediction_checkpoint: Checkpoint for the Tg property-prediction
            model. ``None`` disables ``/api/predict`` and the back-prediction
            step of ``/api/generate``; everything else still works.
        tokenizer_path: The vocabulary JSON artifact. Required for any model.
        training_corpus: Corpus for the TSD (training-set deduplication)
            filter. ``None`` makes TSD a documented no-op.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        lazy: Load models on first use (the default). ``False`` warms them at
            construction time, logging -- not raising -- on failure, so a
            broken checkpoint still yields a serving app that reports the
            problem through ``/api/health``.
        corpus_limit: Maximum training entries indexed for TSD.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    resources = AppResources(
        generation_checkpoint=generation_checkpoint,
        prediction_checkpoint=prediction_checkpoint,
        tokenizer_path=tokenizer_path,
        training_corpus=training_corpus,
        device=device,
        corpus_limit=corpus_limit,
    )

    app = FastAPI(
        title="polyT5 local demo",
        version="0.1.0",
        description=(
            "Local, offline interface to the polyT5 reproduction: Tg-conditioned "
            "polymer generation with the paper's SV/TSD/DD/PV screen, and Tg "
            "prediction by beam search."
        ),
    )
    app.state.resources = resources

    # -- error handling ----------------------------------------------------

    @app.exception_handler(AppError)
    async def _app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        """Render an :class:`AppError` as explanatory JSON."""
        logger.info("request refused (%d): %s", exc.status_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "status_code": exc.status_code},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Render an HTTP error as explanatory JSON rather than HTML."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail), "status_code": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Summarise a body-validation failure in one readable sentence."""
        problems = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            problems.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
        message = "; ".join(problems) or "the request body is invalid"
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    f"Invalid request -- {message}. Candidate counts must be between "
                    f"1 and {MAX_CANDIDATES}, top_p in (0, 1], temperature > 0."
                ),
                "status_code": 422,
            },
        )

    # -- routes ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        """Serve the single-page UI.

        Returns:
            The page as ``text/html``.

        Raises:
            AppError: If the page file is missing from the installed package.
        """
        if not _INDEX_HTML.is_file():
            raise AppError(
                f"The UI page {_INDEX_HTML} is missing from the installation. "
                "The JSON API under /api still works.",
                status_code=500,
            )
        return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Report device, feature availability and provenance.

        Returns:
            The health payload. No model is loaded to answer this.
        """
        return resources.health()

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate_endpoint(request: GenerateRequest) -> dict[str, Any]:
        """Sample polymers for a target Tg and screen them like the paper does.

        Args:
            request: The generation request.

        Returns:
            Per-candidate rows plus the aggregate cascade statistics.

        Raises:
            AppError: If the generation model is unavailable or decoding fails.
        """
        try:
            return _run_generate(resources, request)
        except AppError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("generation failed")
            raise AppError(
                f"Generation failed: {type(exc).__name__}: {exc}", status_code=500
            ) from exc

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict_endpoint(request: PredictRequest) -> dict[str, Any]:
        """Predict Tg for one structure.

        Args:
            request: The prediction request.

        Returns:
            The prediction plus the structure's validity verdict and drawing.

        Raises:
            AppError: If the structure is unreadable or the model is missing.
        """
        try:
            return _run_predict(resources, request)
        except AppError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("prediction failed")
            raise AppError(
                f"Prediction failed: {type(exc).__name__}: {exc}", status_code=500
            ) from exc

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
        """Route a natural-language message to generation, prediction or help.

        Args:
            request: The chat request.

        Returns:
            The prose reply, the routed intent, its parameters, and the raw
            endpoint payload so the UI can draw structures too.

        Raises:
            AppError: If the routed feature is unavailable.
        """
        try:
            return _run_chat(resources, request)
        except AppError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("chat failed")
            raise AppError(f"Chat failed: {type(exc).__name__}: {exc}", status_code=500) from exc

    @app.get("/api/help")
    async def help_endpoint() -> dict[str, str]:
        """Return the same usage text the chat ``help`` intent produces."""
        return {"help": HELP_TEXT}

    if not lazy:
        for slot in (resources.generation, resources.prediction):
            if slot.present:
                try:
                    resources.model(slot)
                except AppError as exc:
                    logger.warning("eager load of %s failed: %s", slot.name, exc.message)
        resources.training_index()

    logger.info(
        "polyT5 app ready (generation=%s, prediction=%s, rendering=%s)",
        resources.generation.present,
        resources.prediction.present,
        rendering.RENDERING_AVAILABLE,
    )
    return app


def create_app_from_env() -> FastAPI:
    """Build the app from ``POLYT5_APP_*`` environment variables.

    This exists so ``scripts/serve.py --reload`` can hand uvicorn an import
    string (uvicorn's reloader re-imports the app in a child process and so
    cannot be given an already-built object).

    Returns:
        The configured application.
    """
    import os

    def _path(name: str) -> str | None:
        value = os.environ.get(name)
        return value or None

    return create_app(
        generation_checkpoint=_path("POLYT5_APP_GENERATION_CHECKPOINT"),
        prediction_checkpoint=_path("POLYT5_APP_PREDICTION_CHECKPOINT"),
        tokenizer_path=_path("POLYT5_APP_TOKENIZER"),
        training_corpus=_path("POLYT5_APP_TRAINING_CORPUS"),
        device=os.environ.get("POLYT5_APP_DEVICE", "auto"),
    )
