#!/usr/bin/env python3
"""
CrossRef API Enrichment Script
================================
Queries CrossRef to enrich journal records with:
- Full journal title and abbreviation
- Publisher name
- Subject areas
- ISSN validation
- Deposit/publication activity status

CrossRef API docs: https://api.crossref.org/swagger-ui/index.html
Rate limit: ~50 req/s with polite pool (with mailto)
No API key required.
"""

import json
import time
import os
import sys
import logging
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE = "enriched_openalex.json"  # Output from step 2
OUTPUT_FILE = "enriched_crossref.json"
PROGRESS_FILE = "crossref_progress.json"
LOG_FILE = "crossref_enrichment.log"

CROSSREF_BASE = "https://api.crossref.org"
CONTACT_EMAIL = "info.reviewpro@gmail.com"  # ← CHANGE THIS for polite pool

RATE_LIMIT_DELAY = 0.1  # seconds between requests (polite pool is generous)
MAX_RETRIES = 3
RETRY_DELAY = 3

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def api_get(url, retries=MAX_RETRIES):
    headers = {"User-Agent": f"JournalRecommenderMVP/1.0 (mailto:{CONTACT_EMAIL})"}
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                wait = RETRY_DELAY * (attempt + 2)
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {e.code} (attempt {attempt+1})")
                time.sleep(RETRY_DELAY)
        except (URLError, TimeoutError) as e:
            log.warning(f"Network error: {e} (attempt {attempt+1})")
            time.sleep(RETRY_DELAY)
    return None


def query_crossref_journal(issn):
    """Look up a journal in CrossRef by ISSN."""
    url = f"{CROSSREF_BASE}/journals/{quote(issn)}"
    data = api_get(url)
    if data and data.get("status") == "ok":
        return data.get("message", {})
    return None


def extract_crossref_metadata(record):
    """Extract structured metadata from a CrossRef journal record."""
    if not record:
        return None
    
    title = record.get("title", "")
    publisher = record.get("publisher", "")
    subjects = record.get("subjects", [])
    subject_list = [s.get("name", "") for s in subjects if s.get("name")]
    
    # ISSNs
    issns = record.get("ISSN", [])
    issn_types = record.get("issn-type", [])
    
    # Coverage / activity
    coverage = record.get("coverage", {})
    total_dois = record.get("total-dois", 0)
    current_dois = record.get("current-dois", 0)
    
    # Flags
    flags = record.get("flags", {})
    deposits_articles = flags.get("deposits-articles-current", False)
    
    # Breakdowns
    dois_by_year = record.get("breakdowns", {}).get("dois-by-issued-year", [])
    recent_years = sorted(dois_by_year, key=lambda x: x[0], reverse=True)[:3] if dois_by_year else []
    
    is_active = deposits_articles or (current_dois > 0)
    
    return {
        "crossref_title": title,
        "crossref_publisher": publisher,
        "crossref_subjects": subject_list,
        "crossref_issns": issns,
        "total_dois": total_dois,
        "current_dois": current_dois,
        "is_active": is_active,
        "recent_year_counts": recent_years,
        "coverage_references": coverage.get("references-current", 0),
        "coverage_abstracts": coverage.get("abstracts-current", 0),
    }


def main():
    log.info("=" * 60)
    log.info("CrossRef Enrichment Pipeline Starting")
    log.info("=" * 60)
    
    # Find input
    actual_input = INPUT_FILE
    for fallback in [INPUT_FILE, "enriched_openalex.json", "enriched_doaj.json", "journal_database.json"]:
        if os.path.exists(fallback):
            actual_input = fallback
            break
    
    journals = load_json(actual_input)
    log.info(f"Loaded {len(journals)} journals from {actual_input}")
    
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        progress = load_json(PROGRESS_FILE)
        log.info(f"Resuming: {len(progress)} already processed")
    
    # Only process journals still missing subject info
    to_process = []
    for j in journals:
        jid = str(j["id"])
        if jid in progress:
            continue
        issn = j.get("electronic_issn") or j.get("print_issn", "")
        if not issn:
            continue
        # Prioritize journals missing data
        has_subjects = bool(j.get("subject_categories"))
        has_publisher = bool(j.get("publisher"))
        to_process.append((j, not has_subjects or not has_publisher))
    
    # Sort: journals missing data first
    to_process.sort(key=lambda x: x[1], reverse=True)
    journals_to_query = [j for j, _ in to_process]
    
    log.info(f"Journals to process: {len(journals_to_query)}")
    
    success_count = 0
    not_found_count = 0
    
    for idx, journal in enumerate(journals_to_query):
        jid = str(journal["id"])
        
        result = None
        for test_issn in [journal.get("electronic_issn"), journal.get("print_issn")]:
            if test_issn:
                result = query_crossref_journal(test_issn)
                if result:
                    break
        
        if result:
            metadata = extract_crossref_metadata(result)
            if metadata:
                progress[jid] = {"status": "found", "metadata": metadata}
                success_count += 1
            else:
                progress[jid] = {"status": "parse_error"}
        else:
            progress[jid] = {"status": "not_found"}
            not_found_count += 1
        
        if (idx + 1) % 200 == 0:
            save_json(progress, PROGRESS_FILE)
            log.info(f"[{idx+1}/{len(journals_to_query)}] {success_count} found, {not_found_count} not found")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    save_json(progress, PROGRESS_FILE)
    
    # ─── Merge ───
    enriched = []
    for journal in journals:
        jid = str(journal["id"])
        entry = dict(journal)
        
        if jid in progress and progress[jid]["status"] == "found":
            meta = progress[jid]["metadata"]
            
            # Subjects: merge with existing
            existing = set(entry.get("subject_categories", []))
            crossref_subj = set(meta.get("crossref_subjects", []))
            entry["subject_categories"] = list(existing | crossref_subj)
            
            # Publisher: fill if missing
            if not entry.get("publisher") and meta.get("crossref_publisher"):
                entry["publisher"] = meta["crossref_publisher"]
            
            # Activity flag
            entry["is_active"] = meta.get("is_active", True)
            entry["total_dois"] = meta.get("total_dois", 0)
            
            entry["enrichment_source"] = entry.get("enrichment_source", "") + "+crossref"
        
        enriched.append(entry)
    
    save_json(enriched, OUTPUT_FILE)
    
    log.info("=" * 60)
    log.info("CROSSREF ENRICHMENT COMPLETE")
    log.info(f"  Found: {success_count}  |  Not found: {not_found_count}")
    log.info(f"  Output: {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
