"""Offline checks for the Vertex AI (EU) transport of PubFit Match.

Run with:  python test_vertex.py

Covers the Vertex URL and Bearer header for text generation, the EU location
allowlist, the fallback to the Gemini API key when no Vertex credentials are
configured, the cached OAuth token, both branches of the embeddings
self-check, and what /health reports. No network and no real credentials:
urlopen and the token refresh are replaced by fakes, and the service account
is a throwaway key generated on the fly.
"""
import io
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PUBLIC_BASE_URL", "https://app.pubfit.ai")
sys.argv = ["app.py"]
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np  # noqa: E402

import vertex_ai  # noqa: E402
from recommender import GeminiLLM  # noqa: E402
from vector_store import EmbeddingService, VectorStore  # noqa: E402

FAILURES = []
PROJECT = "pubfit-1a5ad"
EU_GENERATE_URL = ("https://aiplatform.eu.rep.googleapis.com/v1/projects/pubfit-1a5ad/"
                   "locations/eu/publishers/google/models/gemini-3.5-flash-lite:generateContent")
EU_PREDICT_URL = ("https://aiplatform.eu.rep.googleapis.com/v1/projects/pubfit-1a5ad/"
                  "locations/eu/publishers/google/models/gemini-embedding-001:predict")
NO_CREDS_FILE = os.path.join(tempfile.gettempdir(), "pubfit-no-such-vertex-key.json")


def check(label, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + label + ("" if condition else f"   -> {detail}"))
    if not condition:
        FAILURES.append(label)


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, body):
        self._raw = json.dumps(body).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeUrlopen:
    """Records every request; answers from a queue (the last answer repeats)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)

    def body(self, i=-1):
        return json.loads(self.requests[i].data.decode())


@contextmanager
def fake_network(fake):
    original = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        yield fake
    finally:
        urllib.request.urlopen = original


@contextmanager
def no_sleep():
    original = time.sleep
    time.sleep = lambda s: None
    try:
        yield
    finally:
        time.sleep = original


class FakeCredentials:
    """Mimics google.oauth2 credentials: a token that is valid until expiry."""

    def __init__(self):
        self.token = None
        self.expiry = None
        self.refreshes = 0

    @property
    def valid(self):
        return self.token is not None and self.expiry > datetime.now(timezone.utc)

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


@contextmanager
def fake_token_refresh():
    """Patch the real service-account refresh so no call reaches oauth2.googleapis.com."""
    from google.oauth2 import service_account
    calls = []

    def refresh(self, request):
        calls.append(request)
        self.token = f"sa-token-{len(calls)}"
        self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

    original = service_account.Credentials.refresh
    service_account.Credentials.refresh = refresh
    try:
        yield calls
    finally:
        service_account.Credentials.refresh = original


class LogCapture(logging.Handler):
    def __init__(self, *names):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self.loggers = [logging.getLogger(n) for n in names]

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        for lg in self.loggers:
            lg.addHandler(self)
        return self

    def __exit__(self, *exc):
        for lg in self.loggers:
            lg.removeHandler(self)

    def messages(self, level):
        return [r.getMessage() for r in self.records if r.levelno == level]


def make_service_account_file(directory, project_id=PROJECT):
    """Write a throwaway service-account JSON with a freshly generated RSA key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    info = {
        "type": "service_account", "project_id": project_id, "private_key_id": "test",
        "private_key": pem, "client_email": f"match-vertex@{project_id}.iam.gserviceaccount.com",
        "client_id": "1", "token_uri": "https://oauth2.googleapis.com/token",
    }
    path = os.path.join(directory, "vertex-ai.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return path, info


def vector_with_cosine(v, cosine, seed=0):
    """A vector whose cosine similarity with v is exactly `cosine`."""
    v = np.asarray(v, dtype=np.float64)
    unit = v / np.linalg.norm(v)
    w = np.random.default_rng(seed).standard_normal(v.shape)
    w -= (w @ unit) * unit
    w /= np.linalg.norm(w)
    return cosine * unit + np.sqrt(1.0 - cosine ** 2) * w


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_location_allowlist(sa_path):
    for loc in ["eu", "europe-west1", "europe-west3", "europe-west4", "europe-west8",
                "europe-west9", "europe-north1", "europe-central2", "europe-southwest1", " EU "]:
        try:
            ok = vertex_ai.validate_location(loc) in vertex_ai.EU_VERTEX_LOCATIONS
        except vertex_ai.VertexConfigError:
            ok = False
        check(f"allowlist: {loc.strip()!r} accepted", ok)

    for loc in ["global", "us", "us-central1", "europe-west2", "europe-west6",
                "asia-northeast1", ""]:
        try:
            vertex_ai.validate_location(loc)
            rejected = False
        except vertex_ai.VertexConfigError:
            rejected = True
        check(f"allowlist: {loc!r} rejected", rejected)

    try:
        vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": sa_path, "VERTEX_LOCATION": "global"})
        refused = False
    except vertex_ai.VertexConfigError:
        refused = True
    check("allowlist: valid credentials + global location refuse to load", refused)


def test_no_credentials_falls_back():
    with LogCapture("vertex_ai") as logs:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": NO_CREDS_FILE})
    check("fallback: no credentials -> no Vertex", auth is None)
    warnings = logs.messages(logging.WARNING)
    check("fallback: exactly one WARNING", len(warnings) == 1, warnings)
    check("fallback: WARNING names the API-key fallback",
          warnings and "Gemini API key" in warnings[0], warnings)

    with LogCapture("vertex_ai") as logs:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": NO_CREDS_FILE,
                                           "VERTEX_CREDENTIALS_JSON": "{not json"})
    check("fallback: malformed inline JSON -> no Vertex", auth is None)
    check("fallback: malformed inline JSON logged as ERROR", logs.messages(logging.ERROR))

    llm = GeminiLLM(api_key="test-key")
    fake = FakeUrlopen({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    with fake_network(fake):
        text = llm.create_message("sys", "user")
    req = fake.requests[0]
    check("fallback: LLM backend is api_key", llm.backend == "api_key", llm.backend)
    check("fallback: LLM calls the Gemini API with the key",
          req.full_url.startswith("https://generativelanguage.googleapis.com/v1beta/models/"
                                  "gemini-3.5-flash-lite:generateContent?key=test-key"), req.full_url)
    check("fallback: no Bearer header on the API-key path", req.get_header("Authorization") is None)
    check("fallback: API-key body unchanged (snake_case, thinkingLevel low)",
          "system_instruction" in fake.body() and
          fake.body()["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low", fake.body())
    check("fallback: response parsed", text == "ok", text)

    try:
        GeminiLLM()
        refused = False
    except ValueError:
        refused = True
    check("fallback: neither Vertex nor key -> GeminiLLM refuses", refused)


def test_vertex_generate_content():
    creds = FakeCredentials()
    auth = vertex_ai.VertexAuth(creds, PROJECT, "eu")
    llm = GeminiLLM(api_key="unused-key", vertex=auth)
    answer = {"candidates": [{"content": {"parts": [
        {"text": "internal reasoning", "thought": True},
        {"text": '{"rankings": []}'},
    ]}}]}
    fake = FakeUrlopen(answer)
    with fake_network(fake):
        text = llm.create_message("You rank journals.", "Abstract text")
    req = fake.requests[0]
    body = fake.body()
    check("vertex llm: backend is vertex", llm.backend == "vertex", llm.backend)
    check("vertex llm: EU multi-region generateContent URL", req.full_url == EU_GENERATE_URL, req.full_url)
    check("vertex llm: no API key in the URL", "key=" not in req.full_url)
    check("vertex llm: POST", req.get_method() == "POST")
    check("vertex llm: Bearer token header", req.get_header("Authorization") == "Bearer token-1",
          req.get_header("Authorization"))
    check("vertex llm: JSON content type", req.get_header("Content-type") == "application/json")
    check("vertex llm: systemInstruction (camelCase)",
          body.get("systemInstruction") == {"parts": [{"text": "You rank journals."}]}, body.keys())
    check("vertex llm: user content",
          body.get("contents") == [{"role": "user", "parts": [{"text": "Abstract text"}]}])
    gen = body.get("generationConfig", {})
    check("vertex llm: maxOutputTokens 65536", gen.get("maxOutputTokens") == 65536, gen)
    check("vertex llm: thinkingLevel LOW", gen.get("thinkingConfig") == {"thinkingLevel": "LOW"}, gen)
    check("vertex llm: no temperature, no thinkingBudget",
          "temperature" not in gen and "thinkingBudget" not in gen.get("thinkingConfig", {}), gen)
    check("vertex llm: thought parts dropped from the answer", text == '{"rankings": []}', text)

    regional = vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "europe-west4")
    check("vertex llm: regional EU host",
          regional.model_url("gemini-3.5-flash-lite", "generateContent") ==
          "https://europe-west4-aiplatform.googleapis.com/v1/projects/pubfit-1a5ad/locations/"
          "europe-west4/publishers/google/models/gemini-3.5-flash-lite:generateContent")


def test_vertex_retry_keeps_behaviour():
    creds = FakeCredentials()
    llm = GeminiLLM(vertex=vertex_ai.VertexAuth(creds, PROJECT, "eu"))
    overloaded = HTTPError(EU_GENERATE_URL, 503, "Service Unavailable", {}, io.BytesIO(b""))
    fake = FakeUrlopen(overloaded, {"candidates": [{"content": {"parts": [{"text": "late"}]}}]})
    with fake_network(fake), no_sleep():
        text = llm.create_message("s", "u")
    check("vertex retry: 503 retried then answered", len(fake.requests) == 2 and text == "late",
          (len(fake.requests), text))
    check("vertex retry: token reused across attempts", creds.refreshes == 1, creds.refreshes)

    forbidden = HTTPError(EU_GENERATE_URL, 403, "Forbidden", {}, io.BytesIO(b""))
    fake = FakeUrlopen(forbidden)
    with fake_network(fake), no_sleep():
        try:
            llm.create_message("s", "u")
            raised = False
        except HTTPError as e:
            raised = e.code == 403
    check("vertex retry: 403 fails fast", raised and len(fake.requests) == 1, len(fake.requests))


def test_token_cache_and_refresh(sa_path, sa_info):
    with fake_token_refresh() as refreshes:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": sa_path})
        check("token: credentials loaded from the file", auth is not None)
        if auth is None:
            return
        check("token: project from the credentials file", auth.project_id == PROJECT, auth.project_id)
        check("token: default location eu", auth.location == "eu", auth.location)
        first = auth.headers()["Authorization"]
        second = auth.headers()["Authorization"]
        check("token: Bearer from the service account", first == "Bearer sa-token-1", first)
        check("token: cached while valid", second == first and len(refreshes) == 1, len(refreshes))
        auth._credentials.expiry = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)
        third = auth.headers()["Authorization"]
        check("token: refreshed once expired", third == "Bearer sa-token-2" and len(refreshes) == 2,
              (third, len(refreshes)))
        check("token: cloud-platform scope requested",
              vertex_ai.CLOUD_PLATFORM_SCOPE in (auth._credentials.scopes or []), auth._credentials.scopes)

    override = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": sa_path,
                                           "VERTEX_PROJECT_ID": "other-project",
                                           "VERTEX_LOCATION": "europe-west1"})
    check("token: VERTEX_PROJECT_ID overrides the file",
          override is not None and override.project_id == "other-project")
    check("token: VERTEX_LOCATION honoured", override is not None and override.location == "europe-west1")

    inline = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": NO_CREDS_FILE,
                                         "VERTEX_CREDENTIALS_JSON": json.dumps(sa_info)})
    check("token: inline VERTEX_CREDENTIALS_JSON accepted", inline is not None)


def _tiny_store(directory):
    """Three journals, the first with empty text (as the build script would skip it)."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((3, 768)).astype(np.float32)
    npz = os.path.join(directory, "emb.npz")
    np.savez_compressed(npz, embeddings=vectors, ids=np.array([1, 2, 3], dtype=np.int32))
    jsonl = os.path.join(directory, "input.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for jid, text in [(1, "   "), (2, "Journal: Second | Scope: retina"), (3, "Journal: Third")]:
            f.write(json.dumps({"id": jid, "title": f"J{jid}", "text": text}) + "\n")
    store = VectorStore()
    store.load(npz, jsonl)
    return store, jsonl, vectors


def _predict_answer(values):
    return {"predictions": [{"embeddings": {"values": list(map(float, values)),
                                            "statistics": {"truncated": False, "token_count": 9}}}]}


def test_embedding_self_check(directory):
    store, jsonl, raw = _tiny_store(directory)
    stored = raw[1]  # journal 2 is the first with text

    # Accept: Vertex reproduces the stored vector (a scaled copy: cosine 1.0).
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    auth = vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "eu")
    fake = FakeUrlopen(_predict_answer(stored * 0.6))
    with fake_network(fake):
        accepted = svc.enable_vertex(auth, store, jsonl)
        svc.embed_query("a new abstract")
    body = fake.body(0)
    check("self-check accept: switched to Vertex", accepted and svc.backend == "vertex", svc.backend)
    check("self-check accept: predict on the EU multi-region",
          fake.requests[0].full_url == EU_PREDICT_URL, fake.requests[0].full_url)
    check("self-check accept: Bearer header", fake.requests[0].get_header("Authorization") == "Bearer token-1")
    check("self-check accept: sample is the first journal with text",
          body["instances"] == [{"content": "Journal: Second | Scope: retina"}], body)
    check("self-check accept: 768 dims in parameters, no task_type",
          body["parameters"] == {"outputDimensionality": 768}, body)
    check("self-check accept: result recorded",
          svc.vertex_self_check == {"accepted": True, "journal_id": 2, "cosine": 1.0}, svc.vertex_self_check)
    check("self-check accept: queries go to Vertex",
          len(fake.requests) == 2 and fake.requests[1].full_url == EU_PREDICT_URL
          and fake.body(1)["instances"] == [{"content": "a new abstract"}])

    # Reject: close but not close enough (cosine 0.98 < 0.995).
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(vector_with_cosine(stored, 0.98)),
                       {"embedding": {"values": [0.1] * 768}})
    with fake_network(fake), LogCapture("vector_store") as logs:
        accepted = svc.enable_vertex(vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "eu"), store, jsonl)
        svc.embed_query("a new abstract")
    warnings = logs.messages(logging.WARNING)
    check("self-check reject: stays on the API key", not accepted and svc.backend == "api_key", svc.backend)
    check("self-check reject: WARNING carries the cosine",
          any("cosine=0.980000" in w for w in warnings), warnings)
    check("self-check reject: cosine recorded",
          svc.vertex_self_check.get("cosine") == 0.98 and svc.vertex_self_check.get("accepted") is False,
          svc.vertex_self_check)
    check("self-check reject: queries go to the Gemini API",
          fake.requests[1].full_url.startswith("https://generativelanguage.googleapis.com/")
          and "key=test-key" in fake.requests[1].full_url, fake.requests[1].full_url)

    # Reject: Vertex call fails (e.g. the service account lacks the role).
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(HTTPError(EU_PREDICT_URL, 403, "Forbidden", {}, io.BytesIO(b"")))
    with fake_network(fake), LogCapture("vector_store"):
        accepted = svc.enable_vertex(vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "eu"), store, jsonl)
    check("self-check error: stays on the API key", not accepted and svc.backend == "api_key")
    check("self-check error: HTTP status recorded",
          svc.vertex_self_check == {"accepted": False, "error": "HTTPError", "http_status": 403},
          svc.vertex_self_check)

    # Reject: wrong dimensionality (e.g. the full 3072-d vector).
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(np.ones(3072)))
    with fake_network(fake), LogCapture("vector_store"):
        accepted = svc.enable_vertex(vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "eu"), store, jsonl)
    check("self-check dims: 3072 vs 768 rejected", not accepted and svc.backend == "api_key")

    # Optional task type is sent and validated by the same check.
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(stored))
    with fake_network(fake):
        svc.enable_vertex(vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "eu"), store, jsonl,
                          task_type="retrieval_query")
    check("self-check task type: sent upper-case in the instance",
          fake.body(0)["instances"][0].get("task_type") == "RETRIEVAL_QUERY", fake.body(0))


def test_app_wiring_and_health(sa_path):
    """initialize() picks the transport and /health reports it."""
    try:
        import app as app_module
        from fastapi.testclient import TestClient
    except ImportError as e:
        check(f"wiring: app importable ({e})", False)
        return

    data = Path(HERE) / "data"
    app_module.EMBEDDINGS_FILE = data / "journal_embeddings.npz"
    app_module.JOURNALS_FILE = data / "journal_database_final.json"
    app_module.JOURNALS_JSONL = data / "journal_embeddings_input.jsonl"
    app_module.JOURNALS_BASE = data / "journal_database.json"
    app_module.META_FILE = data / "embedding_metadata.json"
    args = SimpleNamespace(embedding_provider="gemini", embedding_key=None, llm="gemini", llm_key=None)
    client = TestClient(app_module.app)
    saved_env = {k: os.environ.get(k) for k in
                 ("VERTEX_CREDENTIALS_FILE", "VERTEX_CREDENTIALS_JSON", "VERTEX_LOCATION", "GEMINI_API_KEY")}
    try:
        os.environ.pop("VERTEX_CREDENTIALS_JSON", None)
        os.environ.pop("VERTEX_LOCATION", None)
        os.environ["GEMINI_API_KEY"] = "test-key"

        # 1) No Vertex key uploaded yet: everything stays on the API key.
        os.environ["VERTEX_CREDENTIALS_FILE"] = NO_CREDS_FILE
        app_module.initialize(args)
        backends = client.get("/health").json().get("ai_backends")
        check("wiring: without Vertex creds both on api_key",
              backends == {"llm": "api_key", "embeddings": "api_key"}, backends)

        # 2) Vertex key present and the real stored vector reproduced.
        os.environ["VERTEX_CREDENTIALS_FILE"] = sa_path
        first = next(json.loads(l) for l in open(app_module.JOURNALS_JSONL, encoding="utf-8") if l.strip())
        npz = np.load(app_module.EMBEDDINGS_FILE)
        stored = npz["embeddings"][list(npz["ids"]).index(first["id"])]
        fake = FakeUrlopen(_predict_answer(stored))
        with fake_token_refresh(), fake_network(fake):
            app_module.initialize(args)
        backends = client.get("/health").json().get("ai_backends")
        check("wiring: self-check used the real first journal text",
              fake.body(0)["instances"] == [{"content": first["text"]}])
        check("wiring: with Vertex creds both on vertex",
              backends.get("llm") == "vertex" and backends.get("embeddings") == "vertex", backends)
        check("wiring: /health reports eu and the self-check",
              backends.get("vertex_location") == "eu"
              and backends.get("embedding_self_check", {}).get("accepted") is True, backends)
        check("wiring: /health leaks no project id or key",
              PROJECT not in json.dumps(backends) and "key" not in json.dumps(backends).lower(), backends)

        # 3) Non-EU location: the service refuses to start.
        os.environ["VERTEX_LOCATION"] = "us-central1"
        try:
            app_module.initialize(args)
            refused = False
        except vertex_ai.VertexConfigError:
            refused = True
        check("wiring: non-EU VERTEX_LOCATION stops startup", refused)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    logging.getLogger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp:
        sa_path, sa_info = make_service_account_file(tmp)
        test_location_allowlist(sa_path)
        test_no_credentials_falls_back()
        test_vertex_generate_content()
        test_vertex_retry_keeps_behaviour()
        test_token_cache_and_refresh(sa_path, sa_info)
        test_embedding_self_check(tmp)
        test_app_wiring_and_health(sa_path)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
