#!/usr/bin/env python3
"""
Embedding Generation Script
=============================
Generates vector embeddings for all journal records.
Supports three providers (choose one):

  1. OpenAI  (text-embedding-3-small) — fastest, cheapest cloud option (~$0.02 for 9k journals)
  2. Cohere  (embed-english-v3.0)     — good free tier (100 calls/min)
  3. Local   (all-MiniLM-L6-v2)       — free, runs on CPU, no API key needed

Usage:
    python 05_generate_embeddings.py --provider openai --api-key sk-...
    python 05_generate_embeddings.py --provider cohere --api-key ...
    python 05_generate_embeddings.py --provider local
    python 05_generate_embeddings.py --provider local --model all-mpnet-base-v2

Output:
    journal_embeddings.npz  — numpy arrays (vectors + IDs)
    embedding_metadata.json — config used for generation
"""

import json
import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE = "journal_embeddings_input.jsonl"
FALLBACK_INPUT = "journal_database_final.json"
FALLBACK_INPUT_2 = "journal_database.json"
OUTPUT_NPZ = "journal_embeddings.npz"
OUTPUT_META = "embedding_metadata.json"
PROGRESS_FILE = "embedding_progress.json"

BATCH_SIZE_OPENAI = 100
BATCH_SIZE_COHERE = 96
BATCH_SIZE_LOCAL = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _build_text_from_record(j):
    """Build embedding text from whatever fields are available."""
    parts = []
    title = j.get("title", "")
    if title:
        parts.append(f"Journal: {title}")
    scope = j.get("aims_scope", "")
    if scope:
        parts.append(f"Scope: {scope}")
    subjects = j.get("subject_categories", [])
    if subjects:
        parts.append(f"Subjects: {', '.join(subjects[:10])}")
    topics = j.get("top_topics", [])
    if topics:
        parts.append(f"Topics: {', '.join(topics[:10])}")
    keywords = j.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords[:10])}")
    publisher = j.get("publisher", "")
    if publisher:
        parts.append(f"Publisher: {publisher}")
    abbr = j.get("nlm_abbreviation", "")
    if abbr:
        parts.append(f"Abbreviation: {abbr}")
    return " | ".join(parts) if parts else title


def load_records():
    """Load journal records with embedding text."""
    records = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif os.path.exists(FALLBACK_INPUT):
        log.info(f"Using fallback: {FALLBACK_INPUT}")
        with open(FALLBACK_INPUT, encoding="utf-8") as f:
            data = json.load(f)
        for j in data:
            text = j.get("embedding_text", "")
            if not text:
                text = _build_text_from_record(j)
            records.append({
                "id": j["id"],
                "title": j.get("title", ""),
                "text": text,
                "issn": j.get("electronic_issn") or j.get("print_issn", ""),
            })
    elif os.path.exists(FALLBACK_INPUT_2):
        log.info(f"Using base database: {FALLBACK_INPUT_2}")
        log.info(f"(Tip: run the enrichment pipeline first for better results)")
        with open(FALLBACK_INPUT_2, encoding="utf-8") as f:
            data = json.load(f)
        skipped = 0
        for j in data:
            if j.get("doaj_withdrawn"):
                skipped += 1
                continue
            text = _build_text_from_record(j)
            records.append({
                "id": j["id"],
                "title": j.get("title", ""),
                "text": text,
                "issn": j.get("electronic_issn") or j.get("print_issn", ""),
            })
        if skipped:
            log.info(f"Skipped {skipped} withdrawn journals")
    else:
        log.error(f"No input file found!")
        log.error(f"Make sure one of these files exists in the current folder:")
        log.error(f"  - {INPUT_FILE}")
        log.error(f"  - {FALLBACK_INPUT}")
        log.error(f"  - {FALLBACK_INPUT_2}")
        sys.exit(1)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER: OpenAI
# ═══════════════════════════════════════════════════════════════════════════════
def embed_openai(texts, api_key, model="text-embedding-3-small"):
    """Generate embeddings using OpenAI API."""
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE_OPENAI):
        batch = texts[i:i + BATCH_SIZE_OPENAI]
        payload = json.dumps({"input": batch, "model": model}).encode()
        
        for attempt in range(3):
            try:
                req = Request(url, data=payload, headers=headers, method="POST")
                with urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                
                # Sort by index to maintain order
                sorted_data = sorted(result["data"], key=lambda x: x["index"])
                batch_embs = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_embs)
                
                usage = result.get("usage", {})
                if (i // BATCH_SIZE_OPENAI) % 10 == 0:
                    log.info(f"  Batch {i//BATCH_SIZE_OPENAI + 1}: {len(all_embeddings)}/{len(texts)} "
                             f"(tokens: {usage.get('total_tokens', '?')})")
                break
                
            except HTTPError as e:
                if e.code == 429:
                    wait = 10 * (attempt + 1)
                    log.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    body = e.read().decode() if hasattr(e, 'read') else str(e)
                    log.error(f"OpenAI API error {e.code}: {body}")
                    raise
            except Exception as e:
                log.warning(f"Error: {e}, retrying...")
                time.sleep(5)
        
        time.sleep(0.1)  # Small delay between batches
    
    return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER: Cohere
# ═══════════════════════════════════════════════════════════════════════════════
def embed_cohere(texts, api_key, model="embed-english-v3.0"):
    """Generate embeddings using Cohere API."""
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    
    url = "https://api.cohere.ai/v1/embed"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE_COHERE):
        batch = texts[i:i + BATCH_SIZE_COHERE]
        payload = json.dumps({
            "texts": batch,
            "model": model,
            "input_type": "search_document",
            "truncate": "END",
        }).encode()
        
        for attempt in range(3):
            try:
                req = Request(url, data=payload, headers=headers, method="POST")
                with urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                
                all_embeddings.extend(result["embeddings"])
                if (i // BATCH_SIZE_COHERE) % 10 == 0:
                    log.info(f"  Batch {i//BATCH_SIZE_COHERE + 1}: {len(all_embeddings)}/{len(texts)}")
                break
                
            except HTTPError as e:
                if e.code == 429:
                    time.sleep(60)  # Cohere free tier: 100 calls/min
                else:
                    raise
        
        time.sleep(0.6)  # Respect rate limits
    
    return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER: Local (sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════════════
def embed_local(texts, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings locally using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run:")
        log.error("  pip install sentence-transformers")
        sys.exit(1)
    
    log.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    log.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE_LOCAL,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    return embeddings.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER: Gemini (text-embedding-004) — free, uses your existing Gemini key
# ═══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE_GEMINI = 50

def embed_gemini(texts, api_key, model="gemini-embedding-001", dimensions=768):
    """Generate embeddings using Gemini API."""
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError

    log.info(f"Using {dimensions}-dimensional embeddings (reduces file size for GitHub)")
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE_GEMINI):
        batch = texts[i:i + BATCH_SIZE_GEMINI]

        # Gemini embeds one text at a time via embedContent, or batch via batchEmbedContents
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"
        requests_list = []
        for text in batch:
            requests_list.append({
                "model": f"models/{model}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": dimensions,
            })
        payload = json.dumps({"requests": requests_list}).encode()
        headers = {"Content-Type": "application/json"}

        for attempt in range(5):
            try:
                req = Request(url, data=payload, headers=headers, method="POST")
                with urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())

                batch_embs = [item["values"] for item in result["embeddings"]]
                all_embeddings.extend(batch_embs)

                if (i // BATCH_SIZE_GEMINI) % 10 == 0:
                    log.info(f"  Batch {i//BATCH_SIZE_GEMINI + 1}: {len(all_embeddings)}/{len(texts)}")
                break
            except HTTPError as e:
                if e.code == 429:
                    wait = (attempt + 1) * 15
                    log.warning(f"Rate limited, waiting {wait}s... ({attempt+1}/5)")
                    time.sleep(wait)
                else:
                    body = e.read().decode() if hasattr(e, 'read') else str(e)
                    log.error(f"Gemini API error {e.code}: {body}")
                    raise
            except Exception as e:
                log.warning(f"Error: {e}, retrying...")
                time.sleep(5)

        time.sleep(0.05)  # Small delay between batches

    return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER: Anthropic (use Claude's embedding via Voyage)
# ═══════════════════════════════════════════════════════════════════════════════
def embed_voyage(texts, api_key, model="voyage-3-lite"):
    """Generate embeddings using Voyage AI (recommended by Anthropic)."""
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    
    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    batch_size = 72
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = json.dumps({
            "input": batch,
            "model": model,
            "input_type": "document",
        }).encode()
        
        for attempt in range(3):
            try:
                req = Request(url, data=payload, headers=headers, method="POST")
                with urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                
                sorted_data = sorted(result["data"], key=lambda x: x["index"])
                batch_embs = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_embs)
                
                if (i // batch_size) % 10 == 0:
                    log.info(f"  Batch {i//batch_size + 1}: {len(all_embeddings)}/{len(texts)}")
                break
            except HTTPError as e:
                if e.code == 429:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise
        
        time.sleep(0.1)
    
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate journal embeddings")
    parser.add_argument("--provider", choices=["openai", "cohere", "local", "voyage", "gemini"],
                        default="local", help="Embedding provider")
    parser.add_argument("--api-key", help="API key (not needed for local)")
    parser.add_argument("--model", help="Override default model name")
    args = parser.parse_args()
    
    # Load records
    records = load_records()
    log.info(f"Loaded {len(records)} journal records")
    
    # Prepare texts
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    
    # Filter out empty texts
    valid = [(i, t) for i, t in zip(ids, texts) if t.strip()]
    ids = [v[0] for v in valid]
    texts = [v[1] for v in valid]
    log.info(f"Generating embeddings for {len(texts)} non-empty records")
    
    # Generate embeddings
    provider = args.provider
    start = time.time()
    
    if provider == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OpenAI API key required. Use --api-key or set OPENAI_API_KEY")
            sys.exit(1)
        model = args.model or "text-embedding-3-small"
        log.info(f"Using OpenAI {model}")
        embeddings = embed_openai(texts, api_key, model)
        dim = len(embeddings[0])
    
    elif provider == "cohere":
        api_key = args.api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            log.error("Cohere API key required. Use --api-key or set COHERE_API_KEY")
            sys.exit(1)
        model = args.model or "embed-english-v3.0"
        log.info(f"Using Cohere {model}")
        embeddings = embed_cohere(texts, api_key, model)
        dim = len(embeddings[0])
    
    elif provider == "voyage":
        api_key = args.api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            log.error("Voyage API key required. Use --api-key or set VOYAGE_API_KEY")
            sys.exit(1)
        model = args.model or "voyage-3-lite"
        log.info(f"Using Voyage {model}")
        embeddings = embed_voyage(texts, api_key, model)
        dim = len(embeddings[0])
    
    elif provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log.error("Gemini API key required. Use --api-key or set GEMINI_API_KEY")
            sys.exit(1)
        model = args.model or "gemini-embedding-001"
        log.info(f"Using Gemini {model}")
        embeddings = embed_gemini(texts, api_key, model)
        dim = len(embeddings[0])
    
    elif provider == "local":
        model = args.model or "all-MiniLM-L6-v2"
        log.info(f"Using local model: {model}")
        embeddings = embed_local(texts, model)
        dim = len(embeddings[0])
    
    elapsed = time.time() - start
    log.info(f"Generated {len(embeddings)} embeddings ({dim}d) in {elapsed:.1f}s")
    
    # Save as numpy
    emb_array = np.array(embeddings, dtype=np.float32)
    id_array = np.array(ids, dtype=np.int32)
    
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=emb_array,
        ids=id_array,
    )
    file_size = os.path.getsize(OUTPUT_NPZ) / 1024 / 1024
    log.info(f"Saved: {OUTPUT_NPZ} ({file_size:.1f} MB)")
    
    # Save metadata
    meta = {
        "provider": provider,
        "model": model if provider != "local" else (args.model or "all-MiniLM-L6-v2"),
        "dimensions": dim,
        "num_embeddings": len(embeddings),
        "input_file": INPUT_FILE if os.path.exists(INPUT_FILE) else FALLBACK_INPUT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved: {OUTPUT_META}")
    
    log.info("Done! Next: start the backend API with `python app.py`")


if __name__ == "__main__":
    main()
