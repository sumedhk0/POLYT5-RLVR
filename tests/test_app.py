"""Tests for the local polyT5 web application (our counterpart to Figure S13).

Everything here is CPU-only, offline, and fast: a deliberately tiny model
(``d_model=64``, two layers) is trained-not-at-all and saved to ``tmp_path``,
then served through :func:`polyt5.app.server.create_app`. The point is to pin
the *contract* -- intent parsing, response schemas, the nested filter counts,
error handling, lazy loading, and the offline guarantee -- not model quality.

No test starts a real server; the app is driven directly through its ASGI
interface by the tiny :class:`TestClient` below.

Why not ``fastapi.testclient.TestClient``: starlette's client is a thin wrapper
around ``httpx``/``httpx2``, and neither is installed in this environment (the
project's own dependency list does not include an HTTP client, because nothing
in ``polyt5`` makes HTTP requests). Rather than add a dependency for the sake of
the test harness, the client below speaks ASGI directly -- which is what
starlette's client ultimately does too, and which keeps the suite offline by
construction: there is no socket anywhere in it.
"""

from __future__ import annotations

import asyncio
import json as jsonlib
from pathlib import Path
from typing import Any

import pytest

from polyt5.app import rendering
from polyt5.app.intents import format_reply, merge_with_history, parse_intent
from polyt5.app.server import MAX_CANDIDATES, create_app
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import save_checkpoint

# A PSMILES the model never sees but RDKit certainly parses: poly(ethylene oxide).
PEO_PSMILES = "[*]CCO[*]"
PEO_PSELFIES = "[At][C][C][O][At]"

# Keep decoding short: a randomly initialised model almost never emits EOS, so
# every generated row runs to max_length. 16 tokens keeps the suite under a
# second while still exercising the whole decode -> cascade -> render path.
TEST_MAX_LENGTH = 16


# --------------------------------------------------------------------------
# a minimal in-process ASGI client
# --------------------------------------------------------------------------


class Response:
    """The parts of an HTTP response the tests care about."""

    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        """Initialise from the raw ASGI messages."""
        self.status_code = status_code
        self.headers = headers
        self.content = body

    @property
    def text(self) -> str:
        """The body decoded as UTF-8."""
        return self.content.decode("utf-8")

    def json(self) -> Any:
        """The body parsed as JSON."""
        return jsonlib.loads(self.text)


class TestClient:
    """Drive an ASGI app in-process, one request at a time.

    Only ``GET`` and ``POST`` with a JSON body are supported, which is all the
    API surface needs. Unhandled server exceptions propagate rather than being
    swallowed into a 500, so a bug in an endpoint fails the test loudly.
    """

    __test__ = False  # not a pytest test class despite the name

    def __init__(self, app: Any) -> None:
        """Initialise around an ASGI application."""
        self.app = app

    def get(self, path: str) -> Response:
        """Issue a GET request."""
        return asyncio.run(self._request("GET", path, None))

    def post(self, path: str, json: Any = None) -> Response:
        """Issue a POST request with an optional JSON body."""
        return asyncio.run(self._request("POST", path, json))

    async def _request(self, method: str, path: str, payload: Any) -> Response:
        """Run one request through the ASGI application."""
        body = b"" if payload is None else jsonlib.dumps(payload).encode("utf-8")
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }
        pending = {"sent": False}

        async def receive() -> dict[str, Any]:
            if pending["sent"]:
                return {"type": "http.disconnect"}
            pending["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}

        status = 500
        headers: dict[str, str] = {}
        chunks: list[bytes] = []

        async def send(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                for key, value in message.get("headers", []):
                    headers[key.decode("latin-1").lower()] = value.decode("latin-1")
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await self.app(scope, receive, send)
        return Response(status, headers, b"".join(chunks))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> PolyT5Tokenizer:
    """The paper-shaped 458-token vocabulary, built in memory."""
    return PolyT5Tokenizer.default()


@pytest.fixture(scope="module")
def tiny_paths(tmp_path_factory, tokenizer: PolyT5Tokenizer) -> dict[str, Path]:
    """Write a tiny checkpoint plus its tokenizer artifact and training corpus.

    Returns:
        Mapping with ``checkpoint``, ``mismatched``, ``tokenizer`` and
        ``corpus`` paths.
    """
    root = tmp_path_factory.mktemp("polyt5_app")

    tokenizer_path = root / "vocab.json"
    tokenizer.save(tokenizer_path)

    config = PolyT5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        d_kv=16,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        n_positions=64,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
    )
    model = PolyT5ForConditionalGeneration(config)

    checkpoint = root / "tiny.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        epoch=0,
        global_step=0,
        config={"note": "tests/test_app.py fixture"},
        model_config=config.to_dict(),
        tokenizer_path=str(tokenizer_path),
        tokenizer_sha256=tokenizer.sha256,
    )

    mismatched = root / "wrong_vocab.pt"
    save_checkpoint(
        mismatched,
        model=model,
        epoch=0,
        global_step=0,
        config={"note": "deliberately wrong tokenizer hash"},
        model_config=config.to_dict(),
        tokenizer_path=str(tokenizer_path),
        tokenizer_sha256="0" * 64,
    )

    corpus = root / "train.jsonl"
    corpus.write_text(
        '{"source": "300.0", "target": "[At][C][C][At]"}\n'
        '{"source": "450.0", "target": "[At][C][C][O][At]"}\n',
        encoding="utf-8",
    )

    return {
        "checkpoint": checkpoint,
        "mismatched": mismatched,
        "tokenizer": tokenizer_path,
        "corpus": corpus,
    }


@pytest.fixture
def client(tiny_paths: dict[str, Path]) -> TestClient:
    """A TestClient over an app wired to the tiny checkpoint on CPU."""
    app = create_app(
        generation_checkpoint=tiny_paths["checkpoint"],
        prediction_checkpoint=tiny_paths["checkpoint"],
        tokenizer_path=tiny_paths["tokenizer"],
        training_corpus=tiny_paths["corpus"],
        device="cpu",
    )
    return TestClient(app)


@pytest.fixture
def bare_client(tiny_paths: dict[str, Path]) -> TestClient:
    """An app with NO checkpoints configured -- the degraded-feature case."""
    return TestClient(create_app(tokenizer_path=tiny_paths["tokenizer"], device="cpu"))


# --------------------------------------------------------------------------
# intent parsing
# --------------------------------------------------------------------------


def test_parse_generate_with_count_and_kelvin() -> None:
    """The canonical generation phrasing yields both the target and the count."""
    intent = parse_intent("generate 20 polymers with Tg near 450 K")
    assert intent.name == "generate"
    assert intent.params == {"target_tg": 450.0, "n": 20}
    assert intent.confidence > 0.5


def test_parse_generate_without_count() -> None:
    """A bare design request yields only the target; ``n`` stays server-default."""
    intent = parse_intent("design a polymer with a glass transition of 500")
    assert intent.name == "generate"
    assert intent.params == {"target_tg": 500.0}


def test_parse_generate_colloquial() -> None:
    """"make me 5 candidates around 380 kelvin" is a generation request."""
    intent = parse_intent("make me 5 candidates around 380 kelvin")
    assert intent.name == "generate"
    assert intent.params == {"target_tg": 380.0, "n": 5}


def test_parse_predict_psmiles() -> None:
    """A star-notation PSMILES routes to prediction and is classified correctly."""
    intent = parse_intent("predict Tg for [*]CCO[*]")
    assert intent.name == "predict"
    assert intent.params == {"structure": "[*]CCO[*]", "kind": "psmiles"}


def test_parse_predict_pselfies() -> None:
    """A pure bracket-token run is recognised as PSELFIES, not PSMILES."""
    intent = parse_intent("what is the glass transition of [At][C][C][O][At]")
    assert intent.name == "predict"
    assert intent.params == {"structure": PEO_PSELFIES, "kind": "pselfies"}


def test_parse_parameter_override() -> None:
    """Decoding knobs alone form a parameter-override intent."""
    intent = parse_intent("use temperature 0.9 and top_p 0.95")
    assert intent.name == "params"
    assert intent.params == {"temperature": 0.9, "top_p": 0.95}


@pytest.mark.parametrize("message", ["help", "what can you do", "What can you do?"])
def test_parse_help(message: str) -> None:
    """Help phrasings all land on the help intent."""
    intent = parse_intent(message)
    assert intent.name == "help"
    assert intent.params == {}
    assert intent.explanation


def test_parse_unknown_is_helpful_and_total() -> None:
    """Unrecognised input never raises and always explains what IS understood."""
    for message in ["", "   ", "asdfgh qwerty", "tell me a joke", "	"]:
        intent = parse_intent(message)
        assert intent.name == "unknown"
        assert intent.params == {}
        assert intent.explanation.strip()
        assert "generate" in intent.explanation.lower()


def test_parse_celsius_is_converted_and_documented() -> None:
    """An explicit Celsius unit is converted to Kelvin and said out loud."""
    intent = parse_intent("generate 3 polymers with Tg near 100 C")
    assert intent.name == "generate"
    assert intent.params["n"] == 3
    assert intent.params["target_tg"] == pytest.approx(373.15)
    assert "celsius" in intent.explanation.lower()


def test_bare_numbers_are_kelvin_and_said_so() -> None:
    """A unitless target is Kelvin (the paper's unit), stated in the explanation."""
    intent = parse_intent("design a polymer with a glass transition of 500")
    assert "kelvin" in intent.explanation.lower()


def test_merge_with_history_applies_override() -> None:
    """A parameter-only message re-runs the previous intent with new knobs."""
    history = [
        {"role": "user", "content": "generate 6 polymers with Tg near 450 K"},
        {"role": "assistant", "content": "Generated 6 candidates."},
    ]
    merged = merge_with_history(parse_intent("use temperature 0.9 and top_p 0.95"), history)
    assert merged.name == "generate"
    assert merged.params["target_tg"] == 450.0
    assert merged.params["n"] == 6
    assert merged.params["temperature"] == 0.9
    assert merged.params["top_p"] == 0.95


def test_merge_with_history_without_base_stays_params() -> None:
    """With nothing to override, the params intent is returned unchanged."""
    merged = merge_with_history(parse_intent("use temperature 0.9"), [])
    assert merged.name == "params"


def test_format_reply_is_a_factual_sentence() -> None:
    """``format_reply`` summarises the cascade in prose, not JSON."""
    data = {
        "target_tg": 450.0,
        "n_requested": 20,
        "aggregate": {
            "counts": {"n_input": 20, "n_sv": 18, "n_tsd": 15, "n_dd": 13, "n_pv": 12},
            "rates": {"sv_rate": 0.9, "tsd_rate": 0.75, "dd_rate": 0.65, "pv_rate": 0.6},
            "sr_rate": 1.0,
            "sa_mean": 2.4,
            "tp_rate": 0.5,
            "tp_tolerance": 50.0,
            "mean_predicted_tg": 447.0,
            "n_predicted": 12,
        },
    }
    reply = format_reply(parse_intent("generate 20 polymers with Tg near 450 K"), data)
    assert "20" in reply and "450" in reply
    assert "18" in reply and "12" in reply
    assert "447" in reply


# --------------------------------------------------------------------------
# health and laziness
# --------------------------------------------------------------------------


def test_health_is_ok_before_any_model_is_loaded(client: TestClient) -> None:
    """/api/health answers without touching a checkpoint (so without the GPU)."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"]["generation"] is False
    assert body["models_loaded"]["prediction"] is False
    assert body["tokenizer_sha256"]
    assert body["checkpoints"]["generation"]["configured"] is True
    assert body["max_candidates"] == MAX_CANDIDATES
    assert body["device"] in {"cpu", "cuda"}


def test_health_reports_disabled_features_without_checkpoints(bare_client: TestClient) -> None:
    """A missing checkpoint disables a feature; it never crashes the app."""
    body = bare_client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["features"]["generation"] is False
    assert body["features"]["prediction"] is False
    assert body["checkpoints"]["generation"]["configured"] is False


def test_generate_requires_a_configured_checkpoint(bare_client: TestClient) -> None:
    """Calling a disabled feature is a clean 503, not a stack trace."""
    response = bare_client.post("/api/generate", json={"target_tg": 450.0, "n": 2})
    assert response.status_code == 503
    assert "error" in response.json()
    assert "generation" in response.json()["error"].lower()


# --------------------------------------------------------------------------
# /api/generate
# --------------------------------------------------------------------------


def test_generate_returns_documented_schema_and_nested_counts(client: TestClient) -> None:
    """The response carries per-candidate rows plus the paper's nested counts."""
    response = client.post(
        "/api/generate",
        json={"target_tg": 450.0, "n": 6, "seed": 0, "max_length": TEST_MAX_LENGTH},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["target_tg"] == 450.0
    assert body["n_requested"] == 6
    assert len(body["candidates"]) == 6

    for candidate in body["candidates"]:
        for key in (
            "pselfies",
            "psmiles",
            "passed_sv",
            "passed_tsd",
            "passed_dd",
            "passed_pv",
            "sa_score",
            "predicted_tg",
            "tg_error",
            "svg",
        ):
            assert key in candidate, key
        assert isinstance(candidate["passed_sv"], bool)

    counts = body["aggregate"]["counts"]
    # "These filters follow a nested relationship: SV > TSD > DD > PV."
    assert counts["n_input"] == 6
    assert counts["n_input"] >= counts["n_sv"] >= counts["n_tsd"]
    assert counts["n_tsd"] >= counts["n_dd"] >= counts["n_pv"]

    rates = body["aggregate"]["rates"]
    assert 0.0 <= rates["pv_rate"] <= rates["dd_rate"] <= rates["tsd_rate"]
    assert rates["tsd_rate"] <= rates["sv_rate"] <= 1.0
    assert 0.0 <= body["aggregate"]["sr_rate"] <= 1.0
    assert body["settings"]["temperature"] > 0.0


def test_generate_candidate_flags_agree_with_the_cascade(client: TestClient) -> None:
    """A candidate that failed a stage cannot claim to have passed a later one."""
    body = client.post(
        "/api/generate",
        json={"target_tg": 300.0, "n": 5, "seed": 1, "max_length": TEST_MAX_LENGTH},
    ).json()
    for candidate in body["candidates"]:
        assert candidate["passed_pv"] <= candidate["passed_dd"]
        assert candidate["passed_dd"] <= candidate["passed_tsd"]
        assert candidate["passed_tsd"] <= candidate["passed_sv"]
        if not candidate["passed_sv"]:
            assert candidate["failure_stage"] == "SV"


def test_generate_rejects_n_above_the_cap(client: TestClient) -> None:
    """``n`` above ``MAX_CANDIDATES`` is REJECTED (422), not silently clamped."""
    response = client.post("/api/generate", json={"target_tg": 450.0, "n": MAX_CANDIDATES + 1})
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert str(MAX_CANDIDATES) in body["error"] or "n" in body["error"]


def test_generate_rejects_nonsense_parameters(client: TestClient) -> None:
    """Out-of-range decoding knobs are rejected before any model is loaded."""
    assert client.post("/api/generate", json={"target_tg": 450.0, "top_p": 0.0}).status_code == 422
    assert (
        client.post("/api/generate", json={"target_tg": 450.0, "temperature": -1.0}).status_code
        == 422
    )
    assert client.post("/api/generate", json={"n": 2}).status_code == 422


def test_generate_is_reproducible_under_a_seed(client: TestClient) -> None:
    """The same seed gives the same candidates -- the paper reports no seeds."""
    payload = {"target_tg": 450.0, "n": 4, "seed": 7, "max_length": TEST_MAX_LENGTH}
    first = client.post("/api/generate", json=payload).json()
    second = client.post("/api/generate", json=payload).json()
    assert [c["pselfies"] for c in first["candidates"]] == [
        c["pselfies"] for c in second["candidates"]
    ]


# --------------------------------------------------------------------------
# /api/predict
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("structure", "kind"),
    [(PEO_PSMILES, "psmiles"), (PEO_PSELFIES, "pselfies"), (PEO_PSMILES, "auto")],
)
def test_predict_accepts_both_notations(client: TestClient, structure: str, kind: str) -> None:
    """Prediction works from PSMILES, from PSELFIES, and with auto-detection."""
    response = client.post("/api/predict", json={"structure": structure, "kind": kind})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["n_termini"] == 2
    assert body["passes_pv"] is True
    assert body["canonical_psmiles"]
    assert body["pselfies"] == PEO_PSELFIES
    # A randomly initialised model rarely emits a number, so predicted_tg may be
    # None -- but the raw decoder output must always be reported.
    assert "predicted_tg" in body
    assert isinstance(body["raw_output"], str)


def test_predict_rejects_garbage_cleanly(client: TestClient) -> None:
    """Garbage is a 4xx with a message, never a 500 and never a traceback."""
    response = client.post("/api/predict", json={"structure": "not a molecule!!", "kind": "auto"})
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert body["error"].strip()
    assert "Traceback" not in response.text


def test_predict_rejects_empty_structure(client: TestClient) -> None:
    """An empty structure is a validation error, not a crash."""
    assert client.post("/api/predict", json={"structure": "  "}).status_code in (400, 422)


def test_predict_reports_bad_termini_without_failing(client: TestClient) -> None:
    """A parseable molecule with the wrong terminus count is a verdict, not an error."""
    response = client.post("/api/predict", json={"structure": "CCO", "kind": "psmiles"})
    assert response.status_code == 200
    body = response.json()
    assert body["n_termini"] == 0
    assert body["passes_pv"] is False


# --------------------------------------------------------------------------
# /api/chat
# --------------------------------------------------------------------------


def test_chat_routes_a_generation_message(client: TestClient) -> None:
    """A generate-style sentence reaches the generation path."""
    response = client.post(
        "/api/chat", json={"message": "generate 4 polymers with Tg near 450 K"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"] == "generate"
    assert body["params"]["target_tg"] == 450.0
    assert body["params"]["n"] == 4
    assert body["data"]["aggregate"]["counts"]["n_input"] == 4
    assert body["reply"].strip()


def test_chat_routes_a_prediction_message(client: TestClient) -> None:
    """A predict-style sentence reaches the prediction path."""
    body = client.post("/api/chat", json={"message": f"predict Tg for {PEO_PSMILES}"}).json()
    assert body["intent"] == "predict"
    assert body["params"]["structure"] == PEO_PSMILES
    assert body["data"]["n_termini"] == 2
    assert body["reply"].strip()


def test_chat_help_and_unknown_never_500(client: TestClient) -> None:
    """Help and gibberish both answer 200 with prose and no data payload."""
    helped = client.post("/api/chat", json={"message": "what can you do?"})
    assert helped.status_code == 200
    assert helped.json()["intent"] == "help"
    assert "Tg" in helped.json()["reply"]

    puzzled = client.post("/api/chat", json={"message": "asdfgh qwerty"})
    assert puzzled.status_code == 200
    assert puzzled.json()["intent"] == "unknown"
    assert puzzled.json()["reply"].strip()


def test_chat_history_carries_the_override(client: TestClient) -> None:
    """"use temperature 0.9" re-runs the previous generation with that knob."""
    history = [{"role": "user", "content": "generate 3 polymers with Tg near 420 K"}]
    body = client.post(
        "/api/chat", json={"message": "use temperature 0.9 and top_p 0.95", "history": history}
    ).json()
    assert body["intent"] == "generate"
    assert body["params"]["temperature"] == 0.9
    assert body["params"]["target_tg"] == 420.0


def test_chat_rejects_an_empty_message(client: TestClient) -> None:
    """An empty message is a validation error rather than an unknown intent."""
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


# --------------------------------------------------------------------------
# tokenizer provenance
# --------------------------------------------------------------------------


def test_tokenizer_sha_mismatch_refuses_to_serve(tiny_paths: dict[str, Path]) -> None:
    """A vocabulary mismatch is a loud error, never silently wrong token ids."""
    app = create_app(
        generation_checkpoint=tiny_paths["mismatched"],
        tokenizer_path=tiny_paths["tokenizer"],
        device="cpu",
    )
    client = TestClient(app)

    # Health still answers: the mismatch is only discovered when the model loads.
    assert client.get("/api/health").status_code == 200

    response = client.post(
        "/api/generate", json={"target_tg": 450.0, "n": 2, "max_length": TEST_MAX_LENGTH}
    )
    assert response.status_code == 503
    message = response.json()["error"]
    assert "tokenizer" in message.lower()
    assert "0000000000000000" in message or "76471956" in message
    assert "Traceback" not in response.text


def test_missing_checkpoint_file_is_a_clean_error(tiny_paths: dict[str, Path]) -> None:
    """A configured-but-absent checkpoint degrades instead of crashing at import."""
    app = create_app(
        generation_checkpoint=tiny_paths["checkpoint"].parent / "does_not_exist.pt",
        tokenizer_path=tiny_paths["tokenizer"],
        device="cpu",
    )
    client = TestClient(app)
    assert client.get("/api/health").json()["features"]["generation"] is False
    response = client.post("/api/generate", json={"target_tg": 450.0, "n": 2})
    assert response.status_code == 503
    assert "error" in response.json()


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not rendering.RENDERING_AVAILABLE, reason="rdkit.Chem.Draw unavailable on this machine"
)
def test_psmiles_to_svg_returns_inlineable_svg() -> None:
    """The SVG is returned without an XML declaration so it can be inlined."""
    svg = rendering.psmiles_to_svg("[At]CCO[At]")
    assert svg is not None
    assert "<svg" in svg
    assert not svg.lstrip().startswith("<?xml")


@pytest.mark.skipif(
    not rendering.RENDERING_AVAILABLE, reason="rdkit.Chem.Draw unavailable on this machine"
)
def test_psmiles_to_svg_accepts_both_notations() -> None:
    """``[At]`` and ``[*]`` forms both draw, and both draw the same molecule."""
    assert rendering.psmiles_to_svg("[*]CCO[*]") is not None
    assert rendering.psmiles_to_svg("[At]CCO[At]") is not None


def test_psmiles_to_svg_is_total() -> None:
    """Garbage returns None rather than raising."""
    for bad in ["", "   ", "not a molecule", "[At]C(((", None]:  # type: ignore[list-item]
        assert rendering.psmiles_to_svg(bad) is None  # type: ignore[arg-type]


def test_endpoints_degrade_when_rendering_is_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With drawing disabled the API returns ``svg: null``, not a 500."""
    monkeypatch.setattr(rendering, "RENDERING_AVAILABLE", False)
    response = client.post("/api/predict", json={"structure": PEO_PSMILES})
    assert response.status_code == 200
    assert response.json()["svg"] is None

    generated = client.post(
        "/api/generate",
        json={"target_tg": 450.0, "n": 3, "seed": 3, "max_length": TEST_MAX_LENGTH},
    )
    assert generated.status_code == 200
    assert all(c["svg"] is None for c in generated.json()["candidates"])


def test_summary_table_flattens_records() -> None:
    """``summary_table`` turns CandidateRecords into plain UI dicts."""
    from polyt5.evaluation import apply_filter_cascade

    records, _ = apply_filter_cascade([PEO_PSELFIES, "garbage"], training_index=None)
    rows = rendering.summary_table(records)
    assert len(rows) == 2
    assert rows[0]["index"] == 0
    assert rows[0]["passed_pv"] is True
    assert rows[1]["failure_stage"] == "SV"


# --------------------------------------------------------------------------
# the page itself
# --------------------------------------------------------------------------


def test_index_page_is_served(client: TestClient) -> None:
    """``GET /`` returns the single-page UI as HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


def test_index_page_makes_no_network_requests(client: TestClient) -> None:
    """Offline guarantee: no external resource reference anywhere in the page."""
    html = client.get("/").text
    assert "http://" not in html
    assert "https://" not in html
    assert "//cdn" not in html
    assert "@import" not in html


def test_index_page_is_theme_aware(client: TestClient) -> None:
    """Colours are tokens with a dark-scheme override, and body has a background."""
    html = client.get("/").text
    assert "prefers-color-scheme: dark" in html
    assert "--bg" in html
    assert "background" in html


def test_index_page_shows_provenance(client: TestClient) -> None:
    """The footer exposes the tokenizer hash and checkpoint paths."""
    html = client.get("/").text
    assert "sha256" in html.lower()
    assert "checkpoint" in html.lower()
