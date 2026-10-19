"""Offline checks for the Vertex AI (EU) transport of PubFit Match.

Run with:  python test_vertex.py

Covers the Vertex URL and Bearer header for text generation, the EU location
allowlist, the fail-closed LLM (no Vertex credentials -> heuristic mode, never
the API key), credential hygiene (no Firebase key, nothing secret logged), the
cached OAuth token, the embeddings self-check (accept / temporary API-key
fallback), the Vertex corpus rebuild in 05_generate_embeddings.py, and what
/health reports. No network and no real credentials: urlopen and the token
refresh are replaced by fakes, and the service account is a throwaway key
generated on the fly.
"""
import importlib.util
import io
import json
import logging
import os
import re
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
KEY_ID = "0123456789abcdef0123456789abcdef01234567"
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


def eu_auth():
    return vertex_ai.VertexAuth(FakeCredentials(), PROJECT, "eu")


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
        self._levels = [lg.level for lg in self.loggers]
        for lg in self.loggers:
            lg.addHandler(self)
            lg.setLevel(logging.DEBUG)  # see INFO lines even when the root is at WARNING
        return self

    def __exit__(self, *exc):
        for lg, level in zip(self.loggers, self._levels):
            lg.removeHandler(self)
            lg.setLevel(level)

    def messages(self, level=None):
        return [r.getMessage() for r in self.records if level is None or r.levelno == level]


def make_service_account_file(directory, project_id=PROJECT, client_email=None, name="vertex-ai.json"):
    """Write a throwaway service-account JSON with a freshly generated RSA key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    info = {
        "type": "service_account", "project_id": project_id, "private_key_id": KEY_ID,
        "private_key": pem,
        "client_email": client_email or f"match-vertex@{project_id}.iam.gserviceaccount.com",
        "client_id": "1", "token_uri": "https://oauth2.googleapis.com/token",
    }
    path = os.path.join(directory, name)
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


def _predict_answer(values):
    return {"predictions": [{"embeddings": {"values": list(map(float, values)),
                                            "statistics": {"truncated": False, "token_count": 9}}}]}


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

    check("allowlist: eu maps to the fixed EU REP host",
          vertex_ai.endpoint_host("eu") == "aiplatform.eu.rep.googleapis.com")
    try:
        vertex_ai.endpoint_host("global")
        host_refused = False
    except vertex_ai.VertexConfigError:
        host_refused = True
    check("allowlist: no host is ever built for global", host_refused)


def test_llm_is_vertex_only():
    creds = FakeCredentials()
    llm = GeminiLLM(vertex=vertex_ai.VertexAuth(creds, PROJECT, "eu"))
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

    try:
        GeminiLLM(None)
        refused = False
    except ValueError:
        refused = True
    check("vertex llm: no Vertex credentials -> GeminiLLM refuses (no API-key path)", refused)
    try:
        GeminiLLM(api_key="test-key")
        refused = False
    except TypeError:
        refused = True
    check("vertex llm: an API key is not accepted at all", refused)
    source = Path(HERE, "recommender.py").read_text(encoding="utf-8")
    gemini_cls = source[source.index("class GeminiLLM"):source.index("def _brief_reason")]
    check("vertex llm: no generativelanguage URL left in GeminiLLM",
          "generativelanguage" not in gemini_cls)


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


def test_credentials(tmp, sa_path, sa_info):
    with LogCapture("vertex_ai") as logs:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": NO_CREDS_FILE})
    check("creds: absent -> None", auth is None)
    check("creds: absent -> loader stays quiet (the caller logs the consequence)",
          not logs.messages(logging.ERROR) and not logs.messages(logging.WARNING), logs.messages())

    with LogCapture("vertex_ai") as logs:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": NO_CREDS_FILE,
                                           "VERTEX_CREDENTIALS_JSON": '{"private_key": "SECRET'})
    check("creds: malformed inline JSON -> None + ERROR", auth is None and logs.messages(logging.ERROR))
    check("creds: malformed JSON content never echoed", "SECRET" not in " ".join(logs.messages()))

    fb_path, _ = make_service_account_file(
        tmp, client_email=f"firebase-adminsdk-abc12@{PROJECT}.iam.gserviceaccount.com",
        name="firebase-like.json")
    with LogCapture("vertex_ai") as logs:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": fb_path})
    check("creds: Firebase Admin service account refused", auth is None and logs.messages(logging.ERROR))
    with LogCapture("vertex_ai"):
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": sa_path,
                                           "FIREBASE_ADMIN_SDK_FILE": sa_path})
    check("creds: the Firebase Admin key file refused as Vertex key", auth is None)

    with fake_token_refresh() as refreshes, LogCapture("vertex_ai") as logs:
        auth = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": sa_path})
        check("creds: loaded from the file", auth is not None)
        if auth is None:
            return
        logged = " ".join(logs.messages())
        check("creds: log shows at most 8 chars of private_key_id",
              KEY_ID[:8] in logged and KEY_ID[:9] not in logged, logged)
        check("creds: private key never logged",
              "PRIVATE KEY" not in logged and sa_info["private_key"][40:80] not in logged)
        check("creds: project from the credentials file", auth.project_id == PROJECT, auth.project_id)
        check("creds: default location eu", auth.location == "eu", auth.location)
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
    check("creds: VERTEX_PROJECT_ID overrides the file",
          override is not None and override.project_id == "other-project")
    check("creds: VERTEX_LOCATION honoured", override is not None and override.location == "europe-west1")

    inline = vertex_ai.load_vertex_auth({"VERTEX_CREDENTIALS_FILE": NO_CREDS_FILE,
                                         "VERTEX_CREDENTIALS_JSON": json.dumps(sa_info)})
    check("creds: inline VERTEX_CREDENTIALS_JSON accepted", inline is not None)


def _tiny_store(directory, with_journal_1=True):
    """Three journals; journal 1 is the fixed self-check text."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((3, 768)).astype(np.float32)
    ids = [1, 2, 3] if with_journal_1 else [4, 2, 3]
    tag = "a" if with_journal_1 else "b"
    npz = os.path.join(directory, f"emb_{tag}.npz")
    np.savez_compressed(npz, embeddings=vectors, ids=np.array(ids, dtype=np.int32))
    jsonl = os.path.join(directory, f"input_{tag}.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for jid in ids:
            f.write(json.dumps({"id": jid, "title": f"J{jid}", "text": f"Journal: J{jid} | Scope: x"}) + "\n")
    store = VectorStore()
    store.load(npz, jsonl)
    return store, jsonl, vectors


def test_embedding_self_check(directory):
    store, jsonl, raw = _tiny_store(directory)
    stored = raw[0]  # journal 1

    # Accept: Vertex reproduces the stored vector (a scaled copy: cosine 1.0).
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(stored * 0.6))
    with fake_network(fake):
        accepted = svc.enable_vertex(eu_auth(), store, jsonl)
        svc.embed_query("a new abstract")
    body = fake.body(0)
    check("self-check accept: switched to Vertex", accepted and svc.backend == "vertex", svc.backend)
    check("self-check accept: predict on the EU multi-region",
          fake.requests[0].full_url == EU_PREDICT_URL, fake.requests[0].full_url)
    check("self-check accept: Bearer header", fake.requests[0].get_header("Authorization") == "Bearer token-1")
    check("self-check accept: fixed text is journal 1",
          body["instances"] == [{"content": "Journal: J1 | Scope: x"}], body)
    check("self-check accept: 768 dims requested, no task_type",
          body["parameters"] == {"outputDimensionality": 768} and "task_type" not in body["instances"][0],
          body)
    check("self-check accept: result recorded",
          svc.vertex_self_check == {"accepted": True, "journal_id": 1, "cosine": 1.0}, svc.vertex_self_check)
    check("self-check accept: queries go to Vertex",
          len(fake.requests) == 2 and fake.requests[1].full_url == EU_PREDICT_URL
          and fake.body(1)["instances"] == [{"content": "a new abstract"}])

    # Reject: close but under the threshold (0.998 < 0.999) -> temporary API-key fallback.
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(vector_with_cosine(stored, 0.998)),
                       {"embedding": {"values": [0.1] * 768}})
    with fake_network(fake), LogCapture("vector_store") as logs:
        accepted = svc.enable_vertex(eu_auth(), store, jsonl)
        svc.embed_query("a new abstract")
    errors = logs.messages(logging.ERROR)
    check("self-check reject: api_key_fallback", not accepted and svc.backend == "api_key_fallback",
          svc.backend)
    check("self-check reject: ERROR carries the cosine", any("cosine=0.998000" in e for e in errors), errors)
    check("self-check reject: cosine recorded",
          svc.vertex_self_check.get("cosine") == 0.998 and svc.vertex_self_check.get("accepted") is False,
          svc.vertex_self_check)
    check("self-check reject: queries take the Gemini API key embedContent",
          fake.requests[1].full_url.startswith("https://generativelanguage.googleapis.com/v1beta/models/"
                                               "gemini-embedding-001:embedContent?key=test-key"),
          fake.requests[1].full_url)

    # Reject: Vertex call fails (e.g. the service account lacks the permission).
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(HTTPError(EU_PREDICT_URL, 403, "Forbidden", {}, io.BytesIO(b"")))
    with fake_network(fake), LogCapture("vector_store") as logs:
        accepted = svc.enable_vertex(eu_auth(), store, jsonl)
    check("self-check error: api_key_fallback + ERROR",
          not accepted and svc.backend == "api_key_fallback" and logs.messages(logging.ERROR))
    check("self-check error: HTTP status recorded",
          svc.vertex_self_check == {"accepted": False, "error": "HTTPError", "http_status": 403},
          svc.vertex_self_check)

    # Reject: the model's default 3072 dimensions instead of 768.
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(np.ones(3072)))
    with fake_network(fake), LogCapture("vector_store"):
        accepted = svc.enable_vertex(eu_auth(), store, jsonl)
    check("self-check dims: 3072 vs 768 rejected", not accepted and svc.backend == "api_key_fallback")

    # No Vertex credentials at all: same temporary fallback, logged at ERROR, no call made.
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    fake = FakeUrlopen(_predict_answer(stored))
    with fake_network(fake), LogCapture("vector_store") as logs:
        accepted = svc.enable_vertex(None, store, jsonl)
    check("self-check no creds: api_key_fallback + ERROR, no request",
          not accepted and svc.backend == "api_key_fallback" and logs.messages(logging.ERROR)
          and not fake.requests)

    # The fixed journal missing from the corpus: no sample, fallback.
    store_b, jsonl_b, _ = _tiny_store(directory, with_journal_1=False)
    svc = EmbeddingService("gemini", "gemini-embedding-001", api_key="test-key")
    with fake_network(FakeUrlopen(_predict_answer(stored))), LogCapture("vector_store"):
        accepted = svc.enable_vertex(eu_auth(), store_b, jsonl_b)
    check("self-check: fixed journal absent -> fallback",
          not accepted and svc.vertex_self_check == {"accepted": False, "error": "no_sample"})


def test_corpus_rebuild_script():
    spec = importlib.util.spec_from_file_location("gen_embeddings", os.path.join(HERE, "05_generate_embeddings.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rate_limited = HTTPError(EU_PREDICT_URL, 429, "Too Many Requests", {}, io.BytesIO(b""))
    fake = FakeUrlopen(_predict_answer([0.1] * 768), rate_limited, _predict_answer([0.2] * 768))
    with fake_network(fake), no_sleep(), LogCapture("gen_embeddings", "__main__"):
        vectors = gen.embed_vertex(["first journal", "second journal"], eu_auth())
    check("rebuild: one vector per text, 768 dims",
          len(vectors) == 2 and all(len(v) == 768 for v in vectors), [len(v) for v in vectors])
    check("rebuild: one text per predict call, 429 retried", len(fake.requests) == 3, len(fake.requests))
    check("rebuild: same request as the runtime (EU predict, 768, no task_type)",
          all(r.full_url == EU_PREDICT_URL for r in fake.requests)
          and fake.body(0) == {"instances": [{"content": "first journal"}],
                               "parameters": {"outputDimensionality": 768}}, fake.body(0))


def test_only_predict_permission_methods():
    """Every Vertex call is generateContent or predict (aiplatform.endpoints.predict)."""
    methods = set()
    for name in ("recommender.py", "vector_store.py", "vertex_ai.py", "05_generate_embeddings.py", "app.py"):
        source = Path(HERE, name).read_text(encoding="utf-8")
        methods |= set(re.findall(r'model_url\([^()]*,\s*"(\w+)"\)', source))
    check("permissions: only generateContent and predict are called",
          methods == {"generateContent", "predict"}, methods)


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

        # 1) No Vertex key uploaded yet: LLM off (heuristic), embeddings on the fallback.
        os.environ["VERTEX_CREDENTIALS_FILE"] = NO_CREDS_FILE
        with LogCapture("journal-api") as logs:
            app_module.initialize(args)
        backends = client.get("/health").json().get("ai_backends")
        check("wiring: without Vertex creds the LLM is disabled, even with GEMINI_API_KEY set",
              app_module.engine.llm is None and backends.get("llm") == "disabled", backends)
        check("wiring: without Vertex creds embeddings report api_key_fallback",
              backends.get("embeddings") == "api_key_fallback", backends)
        check("wiring: LLM disabled logged once at ERROR",
              sum("LLM disabled" in m for m in logs.messages(logging.ERROR)) == 1, logs.messages())

        # 2) Vertex key present and the real stored vector of journal 1 reproduced.
        os.environ["VERTEX_CREDENTIALS_FILE"] = sa_path
        first = next(json.loads(l) for l in open(app_module.JOURNALS_JSONL, encoding="utf-8") if l.strip())
        npz = np.load(app_module.EMBEDDINGS_FILE)
        stored = npz["embeddings"][list(npz["ids"]).index(first["id"])]
        fake = FakeUrlopen(_predict_answer(stored))
        with fake_token_refresh(), fake_network(fake):
            app_module.initialize(args)
        backends = client.get("/health").json().get("ai_backends")
        check("wiring: self-check used journal 1's real text",
              first["id"] == 1 and fake.body(0)["instances"] == [{"content": first["text"]}])
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
        test_llm_is_vertex_only()
        test_vertex_retry_keeps_behaviour()
        test_credentials(tmp, sa_path, sa_info)
        test_embedding_self_check(tmp)
        test_corpus_rebuild_script()
        test_only_predict_permission_methods()
        test_app_wiring_and_health(sa_path)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
