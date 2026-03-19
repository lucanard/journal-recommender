#!/usr/bin/env python3
"""
OpenAlex API Enrichment Script
================================
Queries OpenAlex to enrich journal records with:
- Scope / topic classification (hierarchical: domain > field > subfield > topic)
- Citation metrics (h-index, impact factor proxy, works count)
- Publisher information
- Open Access status and model
- Country of origin
- Related concepts / subjects

OpenAlex API docs: https://docs.openalex.org/
Rate limit: 10 requests/second (with polite pool), 100k/day
No API key required, but adding email gives you the polite pool.
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
INPUT_FILE = "enriched_doaj.json"  # Output from step 1 (or journal_database.json if skipping DOAJ)
OUTPUT_FILE = "enriched_openalex.json"
PROGRESS_FILE = "openalex_progress.json"
LOG_FILE = "openalex_enrichment.log"

OPENALEX_BASE = "https://api.openalex.org"
# Add your email for the polite pool (10 req/s instead of 1 req/s)
CONTACT_EMAIL = "your-email@example.com"  # ← CHANGE THIS

RATE_LIMIT_DELAY = 0.15  # 0.15s = ~6 req/s (safe margin for polite pool)
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
    """Make a GET request with retry logic."""
    if CONTACT_EMAIL and CONTACT_EMAIL != "your-email@example.com":
        sep = "&" if "?" in url else "?"
        url += f"{sep}mailto={quote(CONTACT_EMAIL)}"
    
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


def query_openalex_by_issn(issn):
    """
    Look up a journal (source) in OpenAlex by ISSN.
    Returns the source record or None.
    """
    url = f"{OPENALEX_BASE}/sources/issn:{quote(issn)}"
    return api_get(url)


def extract_openalex_metadata(record):
    """Extract structured metadata from an OpenAlex source record."""
    if not record:
        return None
    
    openalex_id = record.get("id", "")
    
    # ─── Topics / Subjects (hierarchical) ───
    topics = []
    for topic in record.get("topics", []):
        topics.append({
            "name": topic.get("display_name", ""),
            "count": topic.get("count", 0),
            "subfield": topic.get("subfield", {}).get("display_name", ""),
            "field": topic.get("field", {}).get("display_name", ""),
            "domain": topic.get("domain", {}).get("display_name", ""),
        })
    
    # Top-level subject areas (deduplicated)
    domains = list(set(t["domain"] for t in topics if t.get("domain")))
    fields = list(set(t["field"] for t in topics if t.get("field")))
    subfields = list(set(t["subfield"] for t in topics if t.get("subfield")))
    topic_names = [t["name"] for t in topics[:20]]  # Top 20 topics
    
    # ─── Scope description (from topics) ───
    # OpenAlex doesn't have a text description, but we can synthesize one
    scope_from_topics = ""
    if fields:
        scope_from_topics = f"Covers topics in: {', '.join(fields[:5])}. "
    if subfields:
        scope_from_topics += f"Key subfields: {', '.join(subfields[:8])}."
    
    # ─── Metrics ───
    summary_stats = record.get("summary_stats", {})
    counts_by_year = record.get("counts_by_year", [])
    
    h_index = summary_stats.get("h_index", None)
    i10_index = summary_stats.get("i10_index", None)
    two_yr_mean_citedness = summary_stats.get("2yr_mean_citedness", None)
    
    works_count = record.get("works_count", 0)
    cited_by_count = record.get("cited_by_count", 0)
    
    # Recent publication volume (last 2 years)
    recent_works = sum(
        y.get("works_count", 0) for y in counts_by_year[:2]
    )
    
    # ─── Publisher & OA ───
    publisher = record.get("host_organization_name", "")
    raw_lineage = record.get("host_organization_lineage", [])
    publisher_lineage = []
    for org in raw_lineage:
        if isinstance(org, dict):
            publisher_lineage.append(org.get("display_name", ""))
        elif isinstance(org, str):
            publisher_lineage.append(org)
    
    is_oa = record.get("is_oa", False)
    apc_usd = record.get("apc_usd", None)
    
    # ─── Type & country ───
    source_type = record.get("type", "")
    country_code = record.get("country_code", "")
    homepage = record.get("homepage_url", "")
    
    # ─── Impact quartile estimate ───
    impact_proxy = "Unknown"
    if two_yr_mean_citedness is not None:
        if two_yr_mean_citedness >= 5.0:
            impact_proxy = "Q1 (High)"
        elif two_yr_mean_citedness >= 2.0:
            impact_proxy = "Q1-Q2"
        elif two_yr_mean_citedness >= 1.0:
            impact_proxy = "Q2-Q3"
        elif two_yr_mean_citedness >= 0.3:
            impact_proxy = "Q3-Q4"
        else:
            impact_proxy = "Q4"
    
    return {
        "openalex_id": openalex_id,
        "scope_from_topics": scope_from_topics,
        "domains": domains,
        "fields": fields,
        "subfields": subfields,
        "top_topics": topic_names,
        "all_topics": topics[:30],
        "h_index": h_index,
        "i10_index": i10_index,
        "two_yr_mean_citedness": two_yr_mean_citedness,
        "impact_proxy": impact_proxy,
        "works_count": works_count,
        "cited_by_count": cited_by_count,
        "recent_works_2yr": recent_works,
        "publisher": publisher,
        "publisher_lineage": publisher_lineage,
        "is_oa": is_oa,
        "apc_usd": apc_usd,
        "source_type": source_type,
        "country_code": country_code,
        "homepage": homepage,
    }


def main():
    log.info("=" * 60)
    log.info("OpenAlex Enrichment Pipeline Starting")
    log.info("=" * 60)
    
    if CONTACT_EMAIL == "info.reviewpro@gmail.com":
        log.warning("No contact email set! Using slow pool (1 req/s).")
        log.warning("Set CONTACT_EMAIL in the script for 10x faster processing.")
        global RATE_LIMIT_DELAY
        RATE_LIMIT_DELAY = 1.1
    
    # Load journal database
    if not os.path.exists(INPUT_FILE):
        # Try fallback
        fallback = "journal_database.json"
        if os.path.exists(fallback):
            log.info(f"Using fallback input: {fallback}")
            actual_input = fallback
        else:
            log.error(f"Input file not found: {INPUT_FILE}")
            sys.exit(1)
    else:
        actual_input = INPUT_FILE
    
    journals = load_json(actual_input)
    log.info(f"Loaded {len(journals)} journals from {actual_input}")
    
    # Load progress
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        progress = load_json(PROGRESS_FILE)
        log.info(f"Resuming: {len(progress)} already processed")
    
    # Filter
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
    
    # Estimate time
    est_minutes = len(to_process) * RATE_LIMIT_DELAY / 60
    log.info(f"Estimated time: {est_minutes:.0f} minutes")
    
    success_count = 0
    not_found_count = 0
    
    for idx, journal in enumerate(to_process):
        jid = str(journal["id"])
        title = journal.get("title", "Unknown")
        
        result = None
        for test_issn in [journal.get("electronic_issn"), journal.get("print_issn")]:
            if test_issn:
                result = query_openalex_by_issn(test_issn)
                if result:
                    break
        
        if result:
            metadata = extract_openalex_metadata(result)
            if metadata:
                progress[jid] = {"status": "found", "metadata": metadata}
                success_count += 1
                if (idx + 1) % 20 == 0:
                    fields = metadata.get("fields", [])
                    log.info(f"[{idx+1}/{len(to_process)}] ✓ {title} — {', '.join(fields[:3])}")
            else:
                progress[jid] = {"status": "parse_error"}
        else:
            progress[jid] = {"status": "not_found"}
            not_found_count += 1
        
        # Save progress every 200
        if (idx + 1) % 200 == 0:
            save_json(progress, PROGRESS_FILE)
            log.info(f"--- Progress saved: {success_count} found, {not_found_count} not found ---")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    save_json(progress, PROGRESS_FILE)
    
    # ─── Merge results ───
    enriched = []
    for journal in journals:
        jid = str(journal["id"])
        entry = dict(journal)
        
        if jid in progress and progress[jid]["status"] == "found":
            meta = progress[jid]["metadata"]
            
            # Merge scope: prefer DOAJ description if exists, supplement with OpenAlex topics
            existing_scope = entry.get("aims_scope", "")
            openalex_scope = meta.get("scope_from_topics", "")
            if not existing_scope and openalex_scope:
                entry["aims_scope"] = openalex_scope
            elif existing_scope and openalex_scope:
                entry["aims_scope_openalex"] = openalex_scope
            
            # Subject categories: merge
            existing_subjects = entry.get("subject_categories", [])
            openalex_fields = meta.get("fields", [])
            openalex_subfields = meta.get("subfields", [])
            merged_subjects = list(set(existing_subjects + openalex_fields + openalex_subfields))
            entry["subject_categories"] = merged_subjects
            
            entry["openalex_id"] = meta.get("openalex_id", "")
            entry["domains"] = meta.get("domains", [])
            entry["fields"] = meta.get("fields", [])
            entry["subfields"] = meta.get("subfields", [])
            entry["top_topics"] = meta.get("top_topics", [])
            entry["h_index"] = meta.get("h_index")
            entry["two_yr_mean_citedness"] = meta.get("two_yr_mean_citedness")
            entry["impact_proxy"] = meta.get("impact_proxy", "Unknown")
            entry["works_count"] = meta.get("works_count", 0)
            entry["cited_by_count"] = meta.get("cited_by_count", 0)
            entry["recent_works_2yr"] = meta.get("recent_works_2yr", 0)
            entry["country_code"] = meta.get("country_code", "")
            entry["homepage"] = meta.get("homepage", "")
            
            # APC: prefer DOAJ, fall back to OpenAlex
            if not entry.get("apc_display") and meta.get("apc_usd") is not None:
                entry["apc_display"] = f"USD {meta['apc_usd']}"
                entry["has_apc"] = meta["apc_usd"] > 0
            
            # OA model: refine
            if not entry.get("oa_model"):
                entry["oa_model"] = "Full OA" if meta.get("is_oa") else "Subscription/Hybrid"
            
            # Publisher: fill if missing
            if not entry.get("publisher") and meta.get("publisher"):
                entry["publisher"] = meta["publisher"]
            
            entry["enrichment_source"] = entry.get("enrichment_source", "") + "+openalex"
        
        enriched.append(entry)
    
    save_json(enriched, OUTPUT_FILE)
    
    # ─── Summary ───
    has_scope = sum(1 for e in enriched if e.get("aims_scope"))
    has_subjects = sum(1 for e in enriched if e.get("subject_categories"))
    has_impact = sum(1 for e in enriched if e.get("impact_proxy") and e["impact_proxy"] != "Unknown")
    has_apc = sum(1 for e in enriched if e.get("apc_display"))
    
    log.info("=" * 60)
    log.info("OPENALEX ENRICHMENT COMPLETE")
    log.info(f"  Found in OpenAlex:   {success_count}")
    log.info(f"  Not found:           {not_found_count}")
    log.info(f"  ---")
    log.info(f"  Journals with scope:     {has_scope}/{len(enriched)}")
    log.info(f"  Journals with subjects:  {has_subjects}/{len(enriched)}")
    log.info(f"  Journals with impact:    {has_impact}/{len(enriched)}")
    log.info(f"  Journals with APC info:  {has_apc}/{len(enriched)}")
    log.info(f"  Output saved to:         {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
