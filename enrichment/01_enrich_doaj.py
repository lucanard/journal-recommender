#!/usr/bin/env python3
"""
DOAJ API Enrichment Script
===========================
Queries the DOAJ API to enrich journal records with:
- Aims & scope (editorial description)
- Subject categories
- APC amount and currency
- OA model details
- Publisher
- Article types / license info
- Language

DOAJ API docs: https://doaj.org/api/docs
Rate limit: ~2 requests/second (be polite)
No API key required.
"""

import json
import time
import os
import sys
import logging
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE = "journal_database.json"
OUTPUT_FILE = "enriched_doaj.json"
PROGRESS_FILE = "doaj_progress.json"
LOG_FILE = "doaj_enrichment.log"

DOAJ_API_BASE = "https://doaj.org/api"
RATE_LIMIT_DELAY = 0.6  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

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
    """Make a GET request with retry logic."""
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "JournalRecommenderMVP/1.0"})
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
                log.warning(f"HTTP {e.code} for {url} (attempt {attempt+1})")
                time.sleep(RETRY_DELAY)
        except (URLError, TimeoutError) as e:
            log.warning(f"Network error: {e} (attempt {attempt+1})")
            time.sleep(RETRY_DELAY)
    return None


def query_doaj_by_issn(issn):
    """
    Search DOAJ for a journal by ISSN.
    Returns the full journal record or None.
    """
    # DOAJ search endpoint
    url = f"{DOAJ_API_BASE}/search/journals/issn%3A{quote(issn)}?pageSize=1"
    data = api_get(url)
    
    if data and data.get("results") and len(data["results"]) > 0:
        return data["results"][0]
    
    # Fallback: try the journal lookup endpoint directly
    url2 = f"{DOAJ_API_BASE}/journals/issn/{quote(issn)}"
    data2 = api_get(url2)
    return data2


def extract_doaj_metadata(doaj_record):
    """
    Extract structured metadata from a DOAJ API journal record.
    """
    if not doaj_record:
        return None
    
    bibjson = doaj_record.get("bibjson", {})
    
    # Aims & scope
    aims_scope = bibjson.get("editorial", {}).get("review_process", "")
    # Some journals have it under "description" or in aims_scope field
    description = bibjson.get("description", "")
    
    # Subject categories
    subjects = []
    for subj in bibjson.get("subject", []):
        scheme = subj.get("scheme", "")
        term = subj.get("term", "")
        code = subj.get("code", "")
        subjects.append({
            "scheme": scheme,
            "term": term,
            "code": code
        })
    
    # APC information
    apc = bibjson.get("apc", {})
    has_apc = apc.get("has_apc", False)
    apc_info = []
    if has_apc and "max" in apc:
        for entry in apc.get("max", []):
            apc_info.append({
                "currency": entry.get("currency", ""),
                "price": entry.get("price", 0)
            })
    
    # OA details
    oa_start = bibjson.get("oa_start", {})
    license_info = []
    for lic in bibjson.get("license", []):
        license_info.append({
            "type": lic.get("type", ""),
            "url": lic.get("url", ""),
            "embedded": lic.get("embedded", False)
        })
    
    # Publisher
    publisher = bibjson.get("publisher", {}).get("name", "")
    
    # Language
    languages = bibjson.get("language", [])
    
    # Keywords / alternate titles
    alt_titles = bibjson.get("alternative_title", "")
    keywords_list = bibjson.get("keywords", [])
    
    # Identifiers
    issns = []
    for ident in bibjson.get("identifier", []):
        issns.append({
            "type": ident.get("type", ""),
            "id": ident.get("id", "")
        })
    
    # Preservation / archiving
    preservation = bibjson.get("preservation", {})
    
    # Review process
    review_process = bibjson.get("editorial", {}).get("review_process", [])
    review_url = bibjson.get("editorial", {}).get("review_url", "")
    
    # Plagiarism detection
    plagiarism = bibjson.get("plagiarism", {})
    
    # Publication time
    pub_time_weeks = bibjson.get("publication_time_weeks", None)
    
    # Article stats
    article_stats = bibjson.get("article", {})
    
    return {
        "doaj_id": doaj_record.get("id", ""),
        "description": description,
        "aims_scope_text": description,  # Primary field for semantic matching
        "subjects": subjects,
        "subject_terms": [s["term"] for s in subjects if s.get("term")],
        "has_apc": has_apc,
        "apc_info": apc_info,
        "apc_display": format_apc(has_apc, apc_info),
        "publisher": publisher,
        "languages": languages,
        "license_info": license_info,
        "oa_model": "Full OA",  # All DOAJ journals are OA
        "keywords": keywords_list,
        "alt_title": alt_titles,
        "review_process": review_process,
        "review_url": review_url,
        "plagiarism_detection": plagiarism.get("detection", False),
        "publication_time_weeks": pub_time_weeks,
        "preservation": preservation.get("service", []),
        "issns": issns,
    }


def format_apc(has_apc, apc_info):
    """Format APC into a readable string."""
    if not has_apc:
        return "Free (No APC)"
    if not apc_info:
        return "APC charged (amount unknown)"
    parts = []
    for entry in apc_info:
        currency = entry.get("currency", "USD")
        price = entry.get("price", 0)
        parts.append(f"{currency} {price}")
    return " / ".join(parts)


def main():
    log.info("=" * 60)
    log.info("DOAJ Enrichment Pipeline Starting")
    log.info("=" * 60)
    
    # Load journal database
    if not os.path.exists(INPUT_FILE):
        log.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)
    
    journals = load_json(INPUT_FILE)
    log.info(f"Loaded {len(journals)} journals from {INPUT_FILE}")
    
    # Load progress (for resume capability)
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        progress = load_json(PROGRESS_FILE)
        log.info(f"Resuming from previous run: {len(progress)} already processed")
    
    # Filter to journals that are in DOAJ or might be found
    # We query ALL journals since some PubMed journals may also be in DOAJ
    to_process = []
    for j in journals:
        issn = j.get("electronic_issn") or j.get("print_issn", "")
        if not issn:
            continue
        jid = str(j["id"])
        if jid in progress:
            continue
        to_process.append(j)
    
    log.info(f"Journals to process: {len(to_process)}")
    
    # Process
    success_count = 0
    not_found_count = 0
    error_count = 0
    
    for idx, journal in enumerate(to_process):
        jid = str(journal["id"])
        issn = journal.get("electronic_issn") or journal.get("print_issn", "")
        title = journal.get("title", "Unknown")
        
        # Try electronic ISSN first, then print
        result = None
        for test_issn in [journal.get("electronic_issn"), journal.get("print_issn")]:
            if test_issn:
                result = query_doaj_by_issn(test_issn)
                if result:
                    break
        
        if result:
            metadata = extract_doaj_metadata(result)
            if metadata:
                progress[jid] = {
                    "status": "found",
                    "issn_queried": issn,
                    "metadata": metadata
                }
                success_count += 1
                log.info(f"[{idx+1}/{len(to_process)}] ✓ {title} ({issn}) — {len(metadata.get('subject_terms', []))} subjects")
            else:
                progress[jid] = {"status": "parse_error", "issn_queried": issn}
                error_count += 1
        else:
            progress[jid] = {"status": "not_found", "issn_queried": issn}
            not_found_count += 1
            if (idx + 1) % 50 == 0:
                log.info(f"[{idx+1}/{len(to_process)}] — {title} ({issn}) not in DOAJ")
        
        # Save progress every 100 journals
        if (idx + 1) % 100 == 0:
            save_json(progress, PROGRESS_FILE)
            log.info(f"--- Progress saved: {success_count} found, {not_found_count} not found, {error_count} errors ---")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Final save
    save_json(progress, PROGRESS_FILE)
    
    # ─── Merge results back into database ───
    enriched = []
    for journal in journals:
        jid = str(journal["id"])
        entry = dict(journal)  # copy
        
        if jid in progress and progress[jid]["status"] == "found":
            meta = progress[jid]["metadata"]
            entry["aims_scope"] = meta.get("aims_scope_text", "")
            entry["subject_categories"] = meta.get("subject_terms", [])
            entry["subjects_full"] = meta.get("subjects", [])
            entry["has_apc"] = meta.get("has_apc", None)
            entry["apc_display"] = meta.get("apc_display", "")
            entry["apc_info"] = meta.get("apc_info", [])
            entry["oa_model"] = meta.get("oa_model", "")
            entry["languages"] = meta.get("languages", [])
            entry["keywords"] = meta.get("keywords", [])
            entry["license_info"] = meta.get("license_info", [])
            entry["review_process"] = meta.get("review_process", [])
            entry["publication_time_weeks"] = meta.get("publication_time_weeks")
            entry["doaj_id"] = meta.get("doaj_id", "")
            entry["enrichment_source"] = "doaj"
        
        enriched.append(entry)
    
    save_json(enriched, OUTPUT_FILE)
    
    # ─── Summary ───
    log.info("=" * 60)
    log.info("DOAJ ENRICHMENT COMPLETE")
    log.info(f"  Found in DOAJ:     {success_count}")
    log.info(f"  Not in DOAJ:       {not_found_count}")
    log.info(f"  Errors:            {error_count}")
    log.info(f"  Output saved to:   {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
