"""
Vertex AI (Gemini Enterprise Agent Platform) access for PubFit Match.

Both the LLM re-ranking and the query embeddings can run on Vertex AI in an EU
location instead of the Gemini Developer API (generativelanguage.googleapis.com,
API key). Vertex is authenticated with a dedicated service account, kept
separate from the Firebase Admin credentials:

  VERTEX_CREDENTIALS_FILE   path to the service-account JSON
                            (default: /etc/secrets/vertex-ai.json, a Render Secret File)
  VERTEX_CREDENTIALS_JSON   the same JSON inline (used when the file is absent)
  VERTEX_PROJECT_ID         GCP project (default: project_id inside the credentials)
  VERTEX_LOCATION           default "eu" (EU multi-region); must be an EU location

No credentials -> load_vertex_auth() returns None and the callers keep the
Gemini API key path. A non-EU VERTEX_LOCATION is a configuration error and
raises VertexConfigError, so the service refuses to start rather than send
data outside the EU.

google-auth (and requests, used for the token refresh) are installed
transitively by firebase-admin; they are imported lazily so a missing package
degrades to the API-key path instead of breaking the import of this module.
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

# Multi-region locations are served from their own hostname; regional ones follow
# the "{location}-aiplatform.googleapis.com" pattern.
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
_MULTI_REGION_HOSTS = {"eu": "aiplatform.eu.rep.googleapis.com"}


class VertexConfigError(ValueError):
    """Vertex AI is configured in a way that must not be used (e.g. non-EU location)."""


def validate_location(location: str) -> str:
    """Return the normalised location, or raise if it is not an EU location."""
    loc = (location or "").strip().lower()
    if loc not in EU_VERTEX_LOCATIONS:
        raise VertexConfigError(
            f"VERTEX_LOCATION {location!r} does not keep processing in the EU "
            "(global, US and non-EU regions such as europe-west2 London or "
            "europe-west6 Zurich are rejected). Use 'eu' (EU multi-region) or one of: "
            f"{', '.join(sorted(EU_VERTEX_LOCATIONS - {'eu'}))}."
        )
    return loc


def endpoint_host(location: str) -> str:
    return _MULTI_REGION_HOSTS.get(location, f"{location}-aiplatform.googleapis.com")


class VertexAuth:
    """Project, location and a cached OAuth2 token for Vertex AI REST calls.

    The credentials object is created once and its access token is reused
    until it expires; the lock keeps concurrent request threads from
    refreshing it at the same time.
    """

    backend = "vertex"

    def __init__(self, credentials, project_id: str, location: str):
        self._credentials = credentials
        self.project_id = project_id
        self.location = validate_location(location)
        self._lock = threading.Lock()

    def model_url(self, model: str, method: str) -> str:
        return (
            f"https://{endpoint_host(self.location)}/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/google/models/{model}:{method}"
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

    def describe(self) -> dict:
        """Non-secret summary for logs and /health."""
        return {"location": self.location, "host": endpoint_host(self.location)}


def _read_service_account_info(env) -> tuple[Optional[dict], str]:
    """Return (service-account info, source label) or (None, "") when absent."""
    path = env.get("VERTEX_CREDENTIALS_FILE") or DEFAULT_CREDENTIALS_FILE
    if Path(path).is_file():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f), f"file {path}"
        except (OSError, ValueError) as e:
            log.error(f"Vertex credentials file '{path}' present but unreadable: {e}")
    inline = env.get("VERTEX_CREDENTIALS_JSON")
    if inline:
        try:
            return json.loads(inline), "VERTEX_CREDENTIALS_JSON"
        except ValueError as e:
            log.error(f"VERTEX_CREDENTIALS_JSON is not valid JSON: {e}")
    return None, ""


def load_vertex_auth(env=None) -> Optional[VertexAuth]:
    """Build VertexAuth from the environment.

    Returns None (after one WARNING) when no usable Vertex credentials are
    configured, so callers fall back to the Gemini API key. Raises
    VertexConfigError when VERTEX_LOCATION is not an EU location.
    """
    env = os.environ if env is None else env
    location = validate_location(env.get("VERTEX_LOCATION") or DEFAULT_LOCATION)

    info, source = _read_service_account_info(env)
    if info is None:
        log.warning(
            "Vertex AI credentials not found (VERTEX_CREDENTIALS_FILE / "
            "VERTEX_CREDENTIALS_JSON). Falling back to the Gemini API key "
            "(generativelanguage.googleapis.com): no EU data residency."
        )
        return None

    try:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=[CLOUD_PLATFORM_SCOPE])
    except Exception as e:  # ImportError, malformed key, wrong credential type
        log.error(f"Vertex credentials from {source} could not be loaded: {e}. "
                  "Falling back to the Gemini API key: no EU data residency.")
        return None

    project_id = (env.get("VERTEX_PROJECT_ID") or info.get("project_id") or "").strip()
    if not project_id:
        log.error(f"Vertex credentials from {source} carry no project_id and "
                  "VERTEX_PROJECT_ID is unset. Falling back to the Gemini API key.")
        return None

    auth = VertexAuth(credentials, project_id, location)
    log.info(f"Vertex AI enabled: credentials from {source}, location={location}, "
             f"endpoint={endpoint_host(location)}")
    return auth
