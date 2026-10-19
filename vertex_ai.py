"""
Vertex AI (Gemini Enterprise Agent Platform) access for PubFit Match.

The LLM re-ranking runs only on Vertex AI in an EU location; the query
embeddings run there too once their startup self-check passes. Vertex is
authenticated with a dedicated service account, never the Firebase Admin one:

  VERTEX_CREDENTIALS_FILE    path to the service-account JSON
                             (default: /etc/secrets/vertex-ai.json, a Render Secret File)
  VERTEX_CREDENTIALS_JSON    the same JSON inline (used when the file is absent)
  VERTEX_PROJECT_ID          GCP project (default: project_id inside the credentials)
  VERTEX_LOCATION            text generation; default "eu" (EU multi-region)
  VERTEX_EMBEDDING_LOCATION  query/corpus embeddings; default "europe-west4"
                             (Netherlands, EU region)

Both locations accept EU locations only; the global endpoint never is. They
differ because gemini-embedding-001 is not served on the `eu` multi-region
(predict and embedContent answer 404 "Publisher model ... not found", checked
2026-10-19) while europe-west4 serves it. `eu` passes the allowlist for
VERTEX_EMBEDDING_LOCATION but then the embeddings self-check fails and queries
stay on the temporary API-key fallback.

No usable credentials -> load_vertex_auth() returns None and the caller
decides (LLM: heuristic mode; embeddings: temporary API-key fallback). A
non-EU VERTEX_LOCATION or VERTEX_EMBEDDING_LOCATION raises VertexConfigError,
so the service refuses to start rather than send data outside the EU.

Only two Vertex methods are called, publishers.models.generateContent and
publishers.models.predict, both covered by the aiplatform.endpoints.predict
permission. The OAuth2 token comes from the service account's own key, which
needs no IAM permission.

google-auth (and requests, used for the token refresh) are installed
transitively by firebase-admin; they are imported lazily.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_CREDENTIALS_FILE = "/etc/secrets/vertex-ai.json"
DEFAULT_LOCATION = "eu"
# gemini-embedding-001 is served in EU regions but not on the `eu` multi-region.
DEFAULT_EMBEDDING_LOCATION = "europe-west4"

# EU multi-region endpoint (Representational State Endpoint). Fixed in code: the
# environment can only choose another EU location, never another host.
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
EU_MULTI_REGION_HOST = "aiplatform.eu.rep.googleapis.com"

# Vertex AI locations whose ML processing stays inside the EU: the "eu" multi-region and
# the Vertex AI regions located in EU member states. europe-west2 (London) and
# europe-west6 (Zurich) are in Europe but outside the EU, so they are excluded.
# Same list as reviewpro-backend app/config.py EU_VERTEX_LOCATIONS.
EU_VERTEX_LOCATIONS = frozenset({
    "eu",
    "europe-west1",       # Belgium
    "europe-west3",       # Frankfurt
    "europe-west4",       # Netherlands
    "europe-west8",       # Milan
    "europe-west9",       # Paris
    "europe-north1",      # Finland
    "europe-central2",    # Warsaw
    "europe-southwest1",  # Madrid
})

# Where the Firebase Admin key lives (app.py). It must never double as the Vertex key.
_FIREBASE_KEY_PATHS = ("/etc/secrets/firebase-admin.json",)
_FIREBASE_SA_PREFIX = "firebase-adminsdk-"


class VertexConfigError(ValueError):
    """Vertex AI is configured in a way that must not be used (e.g. non-EU location)."""


def validate_location(location: str, name: str = "Vertex location") -> str:
    """Return the normalised location, or raise if it is not an EU location.

    `name` is the setting being checked (VERTEX_LOCATION, VERTEX_EMBEDDING_LOCATION),
    only used in the error message.
    """
    loc = (location or "").strip().lower()
    if loc not in EU_VERTEX_LOCATIONS:
        raise VertexConfigError(
            f"{name} {location!r} does not keep processing in the EU "
            "(global, US and non-EU regions such as europe-west2 London or "
            "europe-west6 Zurich are rejected). Use 'eu' (EU multi-region) or one of: "
            f"{', '.join(sorted(EU_VERTEX_LOCATIONS - {'eu'}))}."
        )
    return loc


def endpoint_host(location: str) -> str:
    """Host for an EU location: the EU multi-region REP or an EU regional endpoint."""
    loc = validate_location(location)
    return EU_MULTI_REGION_HOST if loc == "eu" else f"{loc}-aiplatform.googleapis.com"


class VertexAuth:
    """Project, locations and a cached OAuth2 token for Vertex AI REST calls.

    `location` serves text generation, `embedding_location` the embeddings
    (see the module docstring for why they differ). The credentials object is
    created once and its access token is reused until it expires; the lock
    keeps concurrent request threads from refreshing it at the same time.
    """

    def __init__(self, credentials, project_id: str, location: str,
                 embedding_location: str = DEFAULT_EMBEDDING_LOCATION):
        self._credentials = credentials
        self.project_id = project_id
        self.location = validate_location(location, "VERTEX_LOCATION")
        self.embedding_location = validate_location(embedding_location, "VERTEX_EMBEDDING_LOCATION")
        self._lock = threading.Lock()

    def model_url(self, model: str, method: str, location: Optional[str] = None) -> str:
        """REST URL of a publisher model method in `location` (default: self.location).

        The host follows the location: the EU multi-region REP host for `eu`,
        `{region}-aiplatform.googleapis.com` for an EU region. Non-EU raises.
        """
        loc = validate_location(location) if location is not None else self.location
        return (
            f"https://{endpoint_host(loc)}/v1/projects/{self.project_id}"
            f"/locations/{loc}/publishers/google/models/{model}:{method}"
        )

    def access_token(self) -> str:
        with self._lock:
            # `valid` is False when there is no token yet or it is about to expire.
            if not self._credentials.valid:
                from google.auth.transport.requests import Request as _AuthRequest
                self._credentials.refresh(_AuthRequest())
            return self._credentials.token

    def headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token()}",
        }


def _firebase_key_paths(env) -> set:
    paths = {p for p in (env.get("FIREBASE_ADMIN_SDK_FILE"), *_FIREBASE_KEY_PATHS) if p}
    return {os.path.normcase(os.path.abspath(p)) for p in paths}


def _read_service_account_info(env) -> tuple[Optional[dict], str]:
    """Return (service-account info, source label) or (None, "") when absent/unusable."""
    path = env.get("VERTEX_CREDENTIALS_FILE") or DEFAULT_CREDENTIALS_FILE
    if os.path.normcase(os.path.abspath(path)) in _firebase_key_paths(env):
        log.error("VERTEX_CREDENTIALS_FILE points at the Firebase Admin key; Vertex AI "
                  "needs its own service account. Vertex AI disabled.")
        return None, ""
    if Path(path).is_file():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f), f"file {path}"
        except (OSError, ValueError) as e:
            # Exception type only: never echo anything read from a credentials file.
            log.error(f"Vertex credentials file '{path}' present but unreadable "
                      f"({type(e).__name__}).")
    inline = env.get("VERTEX_CREDENTIALS_JSON")
    if inline:
        try:
            return json.loads(inline), "VERTEX_CREDENTIALS_JSON"
        except ValueError as e:
            log.error(f"VERTEX_CREDENTIALS_JSON is not valid JSON ({type(e).__name__}).")
    return None, ""


def load_vertex_auth(env=None) -> Optional[VertexAuth]:
    """Build VertexAuth from the environment, or None when no usable credentials.

    Raises VertexConfigError when VERTEX_LOCATION or VERTEX_EMBEDDING_LOCATION
    is not an EU location, before the credentials are even read. Logs ERROR for
    credentials that are present but unusable; silence when they are simply
    absent (each caller logs what that means for it).
    """
    env = os.environ if env is None else env
    location = validate_location(env.get("VERTEX_LOCATION") or DEFAULT_LOCATION,
                                 "VERTEX_LOCATION")
    embedding_location = validate_location(
        env.get("VERTEX_EMBEDDING_LOCATION") or DEFAULT_EMBEDDING_LOCATION,
        "VERTEX_EMBEDDING_LOCATION")

    info, source = _read_service_account_info(env)
    if info is None:
        return None
    if not isinstance(info, dict):
        log.error(f"Vertex credentials from {source} are not a JSON object. Vertex AI disabled.")
        return None

    key_id = str(info.get("private_key_id") or "")[:8] or "n/a"
    if str(info.get("client_email") or "").startswith(_FIREBASE_SA_PREFIX):
        log.error(f"Vertex credentials from {source} (key id {key_id}…) belong to the "
                  "Firebase Admin service account; Vertex AI needs its own. Vertex AI disabled.")
        return None

    try:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=[CLOUD_PLATFORM_SCOPE])
    except Exception as e:  # ImportError, malformed key, wrong credential type
        log.error(f"Vertex credentials from {source} (key id {key_id}…) could not be "
                  f"loaded ({type(e).__name__}). Vertex AI disabled.")
        return None

    project_id = (env.get("VERTEX_PROJECT_ID") or info.get("project_id") or "").strip()
    if not project_id:
        log.error(f"Vertex credentials from {source} carry no project_id and "
                  "VERTEX_PROJECT_ID is unset. Vertex AI disabled.")
        return None

    auth = VertexAuth(credentials, project_id, location, embedding_location)
    log.info(f"Vertex AI enabled: credentials from {source} (key id {key_id}…), "
             f"location={location}, endpoint={endpoint_host(location)}, "
             f"embedding_location={embedding_location}, "
             f"embedding_endpoint={endpoint_host(embedding_location)}")
    if embedding_location == "eu":
        log.warning("VERTEX_EMBEDDING_LOCATION=eu: gemini-embedding-001 is not served on the "
                    "EU multi-region (404), so the embeddings self-check will fail. Use an EU "
                    f"region such as {DEFAULT_EMBEDDING_LOCATION}.")
    return auth
