"""
Vector Store — Local numpy-based vector similarity search.
No external database required. Good for up to ~100k records.

For production scale, swap this for Pinecone / Weaviate / pgvector / Qdrant.
The interface stays the same — just replace the VectorStore class.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector store with cosine similarity search.
    Loads journal embeddings + metadata for fast lookup.
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self.ids: Optional[np.ndarray] = None          # (N,) int32
        self.id_to_idx: dict = {}                       # journal_id → array index
        self.journals: dict = {}                        # journal_id → full metadata
        self.dimension: int = 0
        self.provider: str = ""
        self.model: str = ""
        self._loaded = False

    def load(self, embeddings_path: str, journals_path: str, meta_path: str = None):
        """Load embeddings and journal metadata from disk."""
        # Load embeddings
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"]  # (N, D)
        self.ids = data["ids"]                # (N,)
        self.dimension = self.embeddings.shape[1]

        # Normalize for cosine similarity (so dot product = cosine sim)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms

        # Build index
        self.id_to_idx = {int(jid): idx for idx, jid in enumerate(self.ids)}

        # Load journal metadata
        journals_p = Path(journals_path)
        if journals_p.suffix == ".jsonl":
            with open(journals_p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        j = json.loads(line)
                        self.journals[j["id"]] = j
        else:
            with open(journals_p, encoding="utf-8") as f:
                data = json.load(f)
            for j in data:
                self.journals[j["id"]] = j

        # Load embedding metadata if available
        if meta_path and Path(meta_path).exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.provider = meta.get("provider", "unknown")
            self.model = meta.get("model", "unknown")

        self._loaded = True
        log.info(
            f"VectorStore loaded: {len(self.ids)} embeddings ({self.dimension}d), "
            f"{len(self.journals)} journals, provider={self.provider}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[dict]:
        """
        Find the top_k most similar journals to the query embedding.
        Returns list of {"id": int, "score": float, "journal": dict}.
        """
        if not self._loaded:
            raise RuntimeError("VectorStore not loaded. Call load() first.")

        # Normalize query
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Cosine similarity via dot product (both normalized)
        scores = (self.embeddings @ query.T).flatten()

        # Top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            jid = int(self.ids[idx])
            journal = self.journals.get(jid, {})
            results.append({
                "id": jid,
                "score": float(scores[idx]),
                "journal": journal,
            })

        return results

    def get_journal(self, journal_id: int) -> Optional[dict]:
        """Get a single journal by ID."""
        return self.journals.get(journal_id)

    def get_stats(self) -> dict:
        """Return store statistics."""
        return {
            "total_embeddings": len(self.ids) if self.ids is not None else 0,
            "total_journals": len(self.journals),
            "dimensions": self.dimension,
            "provider": self.provider,
            "model": self.model,
        }


class EmbeddingService:
    """
    Generates query embeddings at runtime using the same provider
    that was used to generate the journal embeddings.
    """

    def __init__(self, provider: str, model: str, api_key: str = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._local_model = None

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        if self.provider == "openai":
            return self._embed_openai(text)
        elif self.provider == "cohere":
            return self._embed_cohere(text)
        elif self.provider == "voyage":
            return self._embed_voyage(text)
        elif self.provider == "gemini":
            return self._embed_gemini(text)
        elif self.provider == "local":
            return self._embed_local(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _embed_openai(self, text: str) -> list[float]:
        from urllib.request import urlopen, Request
        import json as _json

        url = "https://api.openai.com/v1/embeddings"
        payload = _json.dumps({"input": [text], "model": self.model}).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode())
        return result["data"][0]["embedding"]

    def _embed_cohere(self, text: str) -> list[float]:
        from urllib.request import urlopen, Request
        import json as _json

        url = "https://api.cohere.ai/v1/embed"
        payload = _json.dumps({
            "texts": [text],
            "model": self.model,
            "input_type": "search_query",
            "truncate": "END",
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode())
        return result["embeddings"][0]

    def _embed_voyage(self, text: str) -> list[float]:
        from urllib.request import urlopen, Request
        import json as _json

        url = "https://api.voyageai.com/v1/embeddings"
        payload = _json.dumps({
            "input": [text],
            "model": self.model,
            "input_type": "query",
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode())
        return result["data"][0]["embedding"]

    def _embed_gemini(self, text: str) -> list[float]:
        from urllib.request import urlopen, Request
        import json as _json

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        payload = _json.dumps({
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": 768,
        }).encode()
        headers = {"Content-Type": "application/json"}
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode())
        return result["embedding"]["values"]

    def _embed_local(self, text: str) -> list[float]:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.model)
        embedding = self._local_model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()
