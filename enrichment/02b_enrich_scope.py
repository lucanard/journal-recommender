#!/usr/bin/env python3
"""
OpenAlex Deep Scope Enrichment
================================
Fetches recent works (articles) from each journal via OpenAlex to build:
- Editorial keywords (what the journal actually publishes)
- Research focus areas (from article topics/concepts)
- A richer, synthesized scope description for better AI tailoring

This runs AFTER steps 1-3. It uses the openalex_id from step 2.

OpenAlex API: https://docs.openalex.org/
Rate limit: 10 req/s with polite pool (with mailto)

Usage:
    python 02b_enrich_scope.py
    python 02b_enrich_scope.py --limit 500    # Process only first 500 journals
    python 02b_enrich_scope.py --works 30     # Fetch 30 recent works per journal (default: 50)
"""

import json
import time
import os
import sys
import logging
import argparse
from collections import Counter
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE = "enriched_openalex.json"       # Must have openalex_id from step 2
OUTPUT_FILE = "enriched_scope.json"
PROGRESS_FILE = "scope_progress.json"
LOG_FILE = "scope_enrichment.log"

OPENALEX_BASE = "https://api.openalex.org"
CONTACT_EMAIL = "info.reviewpro@gmail.com"  # ← CHANGE THIS for 10x faster access

RATE_LIMIT_DELAY = 0.15      # 0.15s = ~6 req/s (polite pool)
MAX_RETRIES = 3
RETRY_DELAY = 3
WORKS_PER_JOURNAL = 50       # How many recent works to analyze

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
    if CONTACT_EMAIL and CONTACT_EMAIL != "your-email@example.com":
        sep = "&" if "?" in url else "?"
        url += f"{sep}mailto={quote(CONTACT_EMAIL)}"

    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "JournalRecommenderMVP/1.0"})
            with urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                wait = RETRY_DELAY * (attempt + 2)
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {e.code} for {url[:80]}... (attempt {attempt+1})")
                time.sleep(RETRY_DELAY)
        except (URLError, TimeoutError) as e:
            log.warning(f"Network error: {e} (attempt {attempt+1})")
            time.sleep(RETRY_DELAY)
    return None


def fetch_recent_works(openalex_id, num_works=50):
    """
    Fetch the most recent works from a journal/source in OpenAlex.
    Returns a list of work records with keywords, topics, and concepts.
    """
    # Extract the source ID (e.g., "S12345" from "https://openalex.org/S12345")
    source_id = openalex_id.split("/")[-1] if "/" in openalex_id else openalex_id

    url = (
        f"{OPENALEX_BASE}/works?"
        f"filter=primary_location.source.id:{source_id}"
        f"&sort=publication_date:desc"
        f"&per_page={min(num_works, 50)}"
        f"&select=id,title,keywords,topics,concepts,publication_date,type"
    )

    data = api_get(url)
    if not data:
        return []

    works = data.get("results", [])

    # If we need more than 50, fetch page 2
    if num_works > 50 and len(works) >= 50:
        cursor = data.get("meta", {}).get("next_cursor")
        if cursor:
            url2 = (
                f"{OPENALEX_BASE}/works?"
                f"filter=primary_location.source.id:{source_id}"
                f"&sort=publication_date:desc"
                f"&per_page={min(num_works - 50, 50)}"
                f"&cursor={cursor}"
                f"&select=id,title,keywords,topics,concepts,publication_date,type"
            )
            data2 = api_get(url2)
            if data2:
                works.extend(data2.get("results", []))

    return works


def extract_editorial_profile(works, existing_topics=None):
    """
    Analyze recent works to build a rich editorial profile.

    Returns:
        dict with editorial_keywords, research_focus, article_types,
        recent_themes, and a synthesized detailed_scope.
    """
    if not works:
        return None

    # ─── Collect keywords from all works ───
    keyword_counter = Counter()
    for work in works:
        # Keywords field (newer OpenAlex)
        for kw in work.get("keywords", []):
            if isinstance(kw, dict):
                term = kw.get("display_name", kw.get("keyword", ""))
                score = kw.get("score", 0.5)
            else:
                term = str(kw)
                score = 0.5
            if term and len(term) > 2:
                keyword_counter[term.lower()] += score

    # ─── Collect concepts (older OpenAlex field, but still useful) ───
    concept_counter = Counter()
    for work in works:
        for concept in work.get("concepts", []):
            name = concept.get("display_name", "")
            score = concept.get("score", 0)
            level = concept.get("level", 0)
            # Prefer specific concepts (level 2-4), not too broad (level 0-1)
            if name and level >= 1 and score > 0.3:
                concept_counter[name] += score

    # ─── Collect topics ───
    topic_counter = Counter()
    subfield_counter = Counter()
    field_counter = Counter()
    for work in works:
        for topic in work.get("topics", []):
            tname = topic.get("display_name", "")
            if tname:
                topic_counter[tname] += 1
            sf = topic.get("subfield", {})
            if isinstance(sf, dict):
                sfname = sf.get("display_name", "")
            else:
                sfname = ""
            if sfname:
                subfield_counter[sfname] += 1
            fi = topic.get("field", {})
            if isinstance(fi, dict):
                fname = fi.get("display_name", "")
            else:
                fname = ""
            if fname:
                field_counter[fname] += 1

    # ─── Collect article types ───
    type_counter = Counter()
    for work in works:
        wtype = work.get("type", "")
        if wtype:
            type_counter[wtype] += 1

    # ─── Build editorial keywords (top 30, deduplicated) ───
    # Merge keywords and concepts, preferring keywords
    all_terms = Counter()
    for term, score in keyword_counter.items():
        all_terms[term] += score * 2  # Weight keywords higher
    for term, score in concept_counter.items():
        all_terms[term.lower()] += score

    # Remove overly generic terms
    generic_terms = {
        "science", "research", "study", "analysis", "method", "result",
        "model", "system", "review", "article", "paper", "data",
        "effect", "approach", "human", "animal", "cell", "patient",
        "medicine", "biology", "chemistry", "physics", "engineering",
    }
    editorial_keywords = [
        term for term, _ in all_terms.most_common(50)
        if term not in generic_terms and len(term) > 3
    ][:30]

    # ─── Build research focus areas (from topics) ───
    top_topics = [t for t, _ in topic_counter.most_common(15)]
    top_subfields = [sf for sf, _ in subfield_counter.most_common(8)]
    top_fields = [f for f, _ in field_counter.most_common(5)]

    # ─── Build article type distribution ───
    total_works = len(works)
    article_types = {
        atype: round(count / total_works * 100)
        for atype, count in type_counter.most_common(5)
    }

    # ─── Synthesize a detailed scope description ───
    scope_parts = []

    if top_fields:
        scope_parts.append(f"Publishes research primarily in {', '.join(top_fields[:3])}")

    if top_subfields:
        scope_parts.append(
            f"with focus areas including {', '.join(top_subfields[:5])}"
        )

    if editorial_keywords[:10]:
        scope_parts.append(
            f"Frequently published topics include {', '.join(editorial_keywords[:10])}"
        )

    if article_types:
        main_type = list(article_types.keys())[0] if article_types else "article"
        scope_parts.append(
            f"Primarily publishes {main_type}s"
        )

    detailed_scope = ". ".join(scope_parts) + "." if scope_parts else ""

    # ─── Date range of analyzed works ───
    dates = [w.get("publication_date", "") for w in works if w.get("publication_date")]
    date_range = ""
    if dates:
        dates_sorted = sorted(dates)
        date_range = f"{dates_sorted[0]} to {dates_sorted[-1]}"

    return {
        "editorial_keywords": editorial_keywords,
        "research_focus_topics": top_topics,
        "research_focus_subfields": top_subfields,
        "research_focus_fields": top_fields,
        "article_type_distribution": article_types,
        "detailed_scope": detailed_scope,
        "works_analyzed": len(works),
        "analysis_date_range": date_range,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep scope enrichment via OpenAlex works")
    parser.add_argument("--limit", type=int, help="Process only this many journals")
    parser.add_argument("--works", type=int, default=WORKS_PER_JOURNAL, help="Recent works to fetch per journal")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("DEEP SCOPE ENRICHMENT — OpenAlex Recent Works")
    log.info("=" * 60)

    if CONTACT_EMAIL == "your-email@example.com":
        log.warning("No contact email set! Using slow pool (1 req/s).")
        log.warning("Set CONTACT_EMAIL for 10x faster processing.")
        global RATE_LIMIT_DELAY
        RATE_LIMIT_DELAY = 1.1

    # Find input
    actual_input = None
    for candidate in [INPUT_FILE, "enriched_crossref.json", "enriched_doaj.json", "journal_database_final.json", "journal_database.json"]:
        if os.path.exists(candidate):
            actual_input = candidate
            break

    if not actual_input:
        log.error("No input file found!")
        sys.exit(1)

    journals = load_json(actual_input)
    log.info(f"Loaded {len(journals)} journals from {actual_input}")

    # Load progress
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        progress = load_json(PROGRESS_FILE)
        log.info(f"Resuming: {len(progress)} already processed")

    # Filter: only journals with an openalex_id
    to_process = []
    skipped_no_id = 0
    for j in journals:
        jid = str(j["id"])
        if jid in progress:
            continue
        openalex_id = j.get("openalex_id", "")
        if not openalex_id:
            skipped_no_id += 1
            continue
        to_process.append(j)

    if args.limit:
        to_process = to_process[:args.limit]

    log.info(f"Journals to process: {len(to_process)} (skipped {skipped_no_id} without OpenAlex ID)")
    est_minutes = len(to_process) * RATE_LIMIT_DELAY * 1.5 / 60  # ~1.5 requests per journal
    log.info(f"Estimated time: {est_minutes:.0f} minutes")

    success_count = 0
    empty_count = 0

    for idx, journal in enumerate(to_process):
        jid = str(journal["id"])
        title = journal.get("title", "Unknown")
        openalex_id = journal.get("openalex_id", "")

        # Fetch recent works
        works = fetch_recent_works(openalex_id, num_works=args.works)

        if works:
            profile = extract_editorial_profile(
                works,
                existing_topics=journal.get("top_topics", [])
            )
            if profile and profile.get("editorial_keywords"):
                progress[jid] = {
                    "status": "found",
                    "metadata": profile
                }
                success_count += 1
                if (idx + 1) % 20 == 0:
                    kws = ", ".join(profile["editorial_keywords"][:5])
                    log.info(f"[{idx+1}/{len(to_process)}] ✓ {title} — {profile['works_analyzed']} works, keywords: {kws}")
            else:
                progress[jid] = {"status": "no_keywords"}
                empty_count += 1
        else:
            progress[jid] = {"status": "no_works"}
            empty_count += 1

        # Save progress every 100
        if (idx + 1) % 100 == 0:
            save_json(progress, PROGRESS_FILE)
            log.info(f"--- Progress saved: {success_count} enriched, {empty_count} empty ---")

        time.sleep(RATE_LIMIT_DELAY)

    save_json(progress, PROGRESS_FILE)

    # ─── Merge results ───
    enriched = []
    for journal in journals:
        jid = str(journal["id"])
        entry = dict(journal)

        if jid in progress and progress[jid]["status"] == "found":
            meta = progress[jid]["metadata"]

            entry["editorial_keywords"] = meta.get("editorial_keywords", [])
            entry["research_focus_topics"] = meta.get("research_focus_topics", [])
            entry["research_focus_subfields"] = meta.get("research_focus_subfields", [])
            entry["article_type_distribution"] = meta.get("article_type_distribution", {})
            entry["works_analyzed"] = meta.get("works_analyzed", 0)

            # Enhance aims_scope with detailed_scope if current one is poor
            existing_scope = entry.get("aims_scope", "")
            detailed_scope = meta.get("detailed_scope", "")
            if detailed_scope:
                if not existing_scope or len(existing_scope) < 50:
                    # Replace weak scope with synthesized one
                    entry["aims_scope"] = detailed_scope
                else:
                    # Append editorial focus to existing scope
                    entry["aims_scope_extended"] = detailed_scope

            entry["enrichment_source"] = entry.get("enrichment_source", "") + "+scope"

        enriched.append(entry)

    save_json(enriched, OUTPUT_FILE)

    # ─── Summary ───
    has_keywords = sum(1 for e in enriched if e.get("editorial_keywords"))
    has_extended = sum(1 for e in enriched if e.get("aims_scope_extended"))
    has_scope = sum(1 for e in enriched if e.get("aims_scope") and len(e.get("aims_scope", "")) > 50)

    log.info("=" * 60)
    log.info("DEEP SCOPE ENRICHMENT COMPLETE")
    log.info(f"  Enriched with keywords:    {success_count}")
    log.info(f"  No works / no keywords:    {empty_count}")
    log.info(f"  ---")
    log.info(f"  Journals with editorial keywords: {has_keywords}/{len(enriched)}")
    log.info(f"  Journals with extended scope:     {has_extended}/{len(enriched)}")
    log.info(f"  Journals with good scope (>50ch): {has_scope}/{len(enriched)}")
    log.info(f"  Output saved to: {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
