# Journal Database Enrichment Pipeline

## Overview

This pipeline enriches the base journal database (9,300+ journals from PubMed/NLM + DOAJ) with metadata from three free APIs. All APIs are free, require no API keys, and have generous rate limits.

## Architecture

```
journal_database.json (base: title, ISSN, indexing status)
        │
        ▼
┌──────────────────────┐
│ Step 1: DOAJ API     │  → Aims & scope, subjects, APC, OA details, license
│ ~90 min              │    (Best source for OA journals)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 2: OpenAlex API │  → Topics, impact metrics, h-index, publisher, country
│ ~25 min (polite)     │    (Best source for citation metrics & topic classification)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 3: CrossRef API │  → Subject areas, publisher, activity status, DOI counts
│ ~15 min (polite)     │    (Good gap-filler, validates activity)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 4: Merge/Export │  → Deduplicate, score, export JSON + JSONL + XLSX
│ <1 min               │
└──────────────────────┘
           │
           ▼
  journal_database_final.json      ← Production database
  journal_embeddings_input.jsonl   ← Ready for vector embedding
  journal_database_enriched.xlsx   ← Excel with color-coded completeness
  enrichment_report.txt            ← Data quality summary
```

## Quick Start

```bash
# 1. Set up
cd enrichment/
cp ../journal_database.json .

# 2. (Optional) Set your email for faster API access
#    Edit 02_enrich_openalex.py and 03_enrich_crossref.py:
#    CONTACT_EMAIL = "your-real-email@example.com"

# 3. Run full pipeline
python run_enrichment.py

# Or check status / run individual steps:
python run_enrichment.py --status
python run_enrichment.py --step 2    # Resume from step 2
python run_enrichment.py --reset     # Clear progress and start fresh
```

## Requirements

- Python 3.8+
- openpyxl (for Excel export in step 4): `pip install openpyxl`
- No other dependencies — uses only stdlib for API calls

## Resume Support

Every step saves progress to a `*_progress.json` file after every 100-200 journals. If the script is interrupted (Ctrl+C, network failure, laptop sleep), just run it again — it picks up exactly where it left off.

## What Each API Provides

| Field                  | DOAJ | OpenAlex | CrossRef |
|------------------------|------|----------|----------|
| Aims & scope text      | ✅   |          |          |
| Subject categories     | ✅   | ✅       | ✅       |
| Topic classification   |      | ✅       |          |
| APC amount             | ✅   | ✅       |          |
| OA model               | ✅   | ✅       |          |
| Impact proxy (IF-like) |      | ✅       |          |
| H-index                |      | ✅       |          |
| Publisher               | ✅   | ✅       | ✅       |
| Country                |      | ✅       |          |
| Homepage URL           |      | ✅       |          |
| License info           | ✅   |          |          |
| Review process type    | ✅   |          |          |
| Publication time       | ✅   |          |          |
| Activity / DOI status  |      |          | ✅       |

## Rate Limits

| API      | Without email    | With email (polite pool) |
|----------|------------------|--------------------------|
| DOAJ     | ~2 req/s         | ~2 req/s (no change)     |
| OpenAlex | 1 req/s          | 10 req/s                 |
| CrossRef | ~5 req/s         | ~50 req/s                |

**Tip:** Adding your email to OpenAlex and CrossRef scripts gives a ~10x speedup. These APIs use the email to route you to a faster "polite pool" — they don't send you anything.

## Output Files

### `journal_database_final.json`
Complete enriched database. Each record contains:
```json
{
  "id": 1,
  "title": "The Lancet",
  "publisher": "Elsevier",
  "electronic_issn": "1474-547X",
  "indexed_pubmed": true,
  "in_doaj": false,
  "aims_scope": "The Lancet publishes original research...",
  "subject_categories": ["Medicine", "Health Sciences", ...],
  "fields": ["Medicine", "Biochemistry"],
  "top_topics": ["Clinical Medicine", "Public Health", ...],
  "oa_model": "Hybrid",
  "apc_display": "USD 6300",
  "impact_proxy": "Q1 (High)",
  "h_index": 872,
  "completeness_score": 95,
  "embedding_text": "Journal: The Lancet | Scope: ... | Subjects: ...",
  ...
}
```

### `journal_embeddings_input.jsonl`
One JSON object per line, optimized for embedding generation:
```jsonl
{"id":1,"title":"The Lancet","text":"Journal: The Lancet | Scope: ... | Subjects: ...","issn":"1474-547X",...}
```

Use this with any embedding model (OpenAI `text-embedding-3-small`, Cohere `embed-v3`, or open-source like `all-MiniLM-L6-v2`) to generate vectors for semantic search.

## After Enrichment: Setting Up Semantic Search

### Option A: Simple (Claude API only, no vector DB)
The React prototype already works this way — it sends the abstract to Claude and gets recommendations based on Claude's knowledge. Good enough for MVP with <2000 journals.

### Option B: Production (Embeddings + Vector DB + LLM re-ranking)
```
1. Generate embeddings:
   - Load journal_embeddings_input.jsonl
   - Call embedding API for each record's "text" field
   - Store vectors in Pinecone / Weaviate / pgvector / Qdrant

2. At query time:
   - Embed user's abstract
   - Vector search → top 20 candidates
   - Filter by constraints (indexing, OA, APC)
   - LLM re-rank top 5 → explain top 3

3. Cost estimate:
   - Embedding 9k journals: ~$0.02 (OpenAI) or free (open-source)
   - Per query: ~$0.001 (embedding) + ~$0.01 (LLM re-ranking)
```

## Troubleshooting

**"Rate limited" warnings**: Normal. The scripts automatically back off and retry.

**Script hangs**: Some university/corporate networks block API calls. Try from a different network or use a VPN.

**"Not found" for many journals**: Expected. Not all journals are in all APIs. DOAJ only indexes OA journals. OpenAlex has the broadest coverage (~250k sources).

**Resuming after error**: Just run the same script again. Progress is saved automatically.
