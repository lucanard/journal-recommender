"""
PubFit Benchmark v4 — External Tool Comparison
===============================================
Extends v3 by adding two real-world external tools:

  5. JANE       — Journal/Author Name Estimator (biosemantics.org)
                  PubMed/MEDLINE-based, gold-standard academic tool since 2007.
                  Uses Apache Lucene MoreLikeThis on 10 years of PubMed abstracts.
                  Free, no API key needed. HTML scraping of suggest.php.
                  ** Best suited for biomedical/health papers only **

  6. B!SON      — Bibliometric and Semantic Open Access recommender (TIB/SLUB)
                  Open-source, publisher-independent, OA journals only (DOAJ).
                  Combines semantic similarity + bibliometric methods.
                  Free REST API, no key needed.

All 4 methods from v3 are retained:
  1. PubFit AI        — Gemini embeddings + LLM re-ranking
  2. PubFit Keywords  — Weighted keyword matching
  3. TF-IDF           — Cosine similarity (simulates Elsevier Journal Finder)
  4. BM25             — Lexical matching (standard IR baseline)

New CLI flags:
  --skip-external    Skip JANE + B!SON (use if offline or for speed)
  --skip-baselines   Skip TF-IDF + BM25

Important notes on external tools:
  - JANE:  biomedical-only. Papers from non-medical journals will score low.
           Adds ~3 s/paper (respectful rate limiting).
  - B!SON: OA journals only. If the ground-truth journal is not OA, Hit@K = 0.
           Both tools are queried with abstract only (no title, to match PubFit).

Usage:
    # Step 1: Build journal index (one-time)
    python benchmark_v4.py --api-url https://journal-recommender.onrender.com --build-index

    # Step 2: Run full benchmark (all 6 methods)
    python benchmark_v4.py --api-url https://journal-recommender.onrender.com --n-papers 100

    # Run without external tools (faster, same as v3)
    python benchmark_v4.py --api-url https://journal-recommender.onrender.com --n-papers 100 --skip-external

Requires:
    pip install requests pandas tabulate scikit-learn rank-bm25 numpy beautifulsoup4 lxml
"""

import argparse, json, time, random, re, sys, os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

try:
    import requests
    import numpy as np
    import pandas as pd
    from tabulate import tabulate
except ImportError:
    print("pip install requests pandas tabulate numpy scikit-learn rank-bm25 beautifulsoup4 lxml")
    sys.exit(1)

OPENALEX_API = "https://api.openalex.org"
CONTACT_EMAIL = "benchmark@pubfit.ai"
DATA_DIR = "benchmark_data"
JOURNAL_INDEX = os.path.join(DATA_DIR, "journal_index.json")
JOURNAL_FULL  = os.path.join(DATA_DIR, "journal_full.json")

# External tool endpoints
# JANE: form POSTs to suggestions.php (confirmed from source: mi-erasmusmc/JANE/JaneClient/index.php)
JANE_URL  = "https://jane.biosemantics.org/suggestions.php"
# B!SON: confirmed from Swagger UI at service.tib.eu/bison/api/schema/swagger-ui
BISON_URL = "https://service.tib.eu/bison/api/public/v1/search"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD JOURNAL INDEX (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def build_journal_index(api_url):
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Connecting to {api_url}...")
    total = 0
    try:
        r = requests.get(f"{api_url}/health", timeout=60)
        h = r.json()
        vs = h.get("vector_store", {})
        total = (vs.get("journals") or vs.get("total_journals") or
                 vs.get("journal_count") or vs.get("count") or 0)
        if total == 0:
            try:
                r2 = requests.get(f"{api_url}/stats", timeout=30)
                total = r2.json().get("total_journals", 0)
            except Exception:
                pass
        print(f"  Database reports {total} journals")
    except Exception as e:
        print(f"  Warning: health check failed ({e}) — continuing anyway")

    journals = {}
    search_terms = list("abcdefghijklmnopqrstuvwxyz0123456789")
    search_terms += [
        "journal", "review", "research", "science", "medicine", "international",
        "clinical", "applied", "european", "american", "british", "world",
        "public", "health", "social", "environmental", "biology", "chemistry",
        "physics", "engineering", "computer", "education", "economic", "nature",
        "cell", "brain", "cancer", "heart", "plant", "food", "water", "energy",
        "plos", "bmc", "frontiers", "mdpi", "springer", "elsevier", "wiley",
        "taylor", "oxford", "cambridge", "lancet", "annals", "archives",
        "proceedings", "transactions", "letters", "reports", "advances",
        "current", "modern", "new", "open", "peer", "scientific", "academic",
    ]

    print(f"\nCollecting journals via {len(search_terms)} search queries...")
    for i, term in enumerate(search_terms):
        try:
            r = requests.get(f"{api_url}/journals/search",
                             params={"q": term, "limit": 50}, timeout=15)
            if r.status_code == 200:
                for j in r.json().get("results", []):
                    jid = j.get("id")
                    if jid and jid not in journals:
                        journals[jid] = j
            time.sleep(0.15)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(search_terms)} queries → {len(journals)} unique journals")

    pct = f" ({len(journals)/total*100:.0f}% of database)" if total > 0 else ""
    print(f"\n  Collected {len(journals)} journals{pct}")

    issn_to_id = {}
    for jid, j in journals.items():
        for field in ["issn", "electronic_issn", "print_issn"]:
            issn = (j.get(field) or "").replace("-", "").strip()
            if issn and len(issn) == 8:
                issn_to_id[issn] = jid

    index_data = {
        "journals": journals,
        "issn_to_id": issn_to_id,
        "total_in_db": total if total > 0 else len(journals),
        "collected": len(journals),
        "timestamp": datetime.now().isoformat(),
    }
    with open(JOURNAL_INDEX, "w") as f:
        json.dump(index_data, f)
    print(f"  Saved index with {len(issn_to_id)} ISSNs → {JOURNAL_INDEX}")

    print(f"\nFetching full details for {len(journals)} journals (for baselines)...")
    full = {}
    for i, jid in enumerate(journals):
        try:
            r = requests.get(f"{api_url}/journals/{jid}", timeout=10)
            if r.status_code == 200:
                full[jid] = r.json()
            time.sleep(0.08)
        except Exception:
            full[jid] = journals[jid]
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(journals)} details fetched...")

    with open(JOURNAL_FULL, "w") as f:
        json.dump(full, f)
    print(f"  Saved full records → {JOURNAL_FULL}")

    return index_data, full


def load_journal_index():
    if not os.path.exists(JOURNAL_INDEX): return None
    with open(JOURNAL_INDEX) as f: return json.load(f)

def load_journal_full():
    if not os.path.exists(JOURNAL_FULL): return None
    with open(JOURNAL_FULL) as f: return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: FETCH TEST PAPERS (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_abstract(inverted_index):
    if not inverted_index: return ""
    positions = []
    for word, poslist in inverted_index.items():
        for pos in poslist:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def fetch_papers_from_database_journals(journal_index, n_papers, year_range=(2022, 2025)):
    journals   = journal_index["journals"]
    issn_to_id = journal_index["issn_to_id"]

    journal_issns = []
    seen = set()
    for jid, j in journals.items():
        title = j.get("title", "")
        for field in ["issn", "electronic_issn", "print_issn"]:
            issn = (j.get(field) or "").strip()
            if issn and len(issn.replace("-", "")) == 8 and jid not in seen:
                journal_issns.append({
                    "id": jid, "title": title, "issn": issn,
                    "issn_clean": issn.replace("-", ""),
                })
                seen.add(jid)
                break

    print(f"  {len(journal_issns)} journals have valid ISSNs")
    sample_size = min(len(journal_issns), n_papers * 3)
    sampled = random.sample(journal_issns, sample_size)

    papers  = []
    headers = {"User-Agent": f"PubFitBenchmark/4.0 (mailto:{CONTACT_EMAIL})"}
    tried   = 0

    for jinfo in sampled:
        if len(papers) >= n_papers: break
        tried += 1

        issn_formatted = jinfo["issn"]
        if "-" not in issn_formatted and len(issn_formatted) == 8:
            issn_formatted = issn_formatted[:4] + "-" + issn_formatted[4:]

        try:
            params = {
                "filter": (
                    f"primary_location.source.issn:{issn_formatted},"
                    f"type:article,"
                    f"from_publication_date:{year_range[0]}-01-01,"
                    f"to_publication_date:{year_range[1]}-12-31"
                ),
                "select": "id,doi,title,abstract_inverted_index,primary_location,publication_year",
                "sort": "cited_by_count:desc",
                "per_page": 5,
                "page": 1,
                "mailto": CONTACT_EMAIL,
            }
            r = requests.get(f"{OPENALEX_API}/works", params=params,
                             headers=headers, timeout=20)
            if r.status_code != 200: continue

            for work in r.json().get("results", []):
                abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))
                if len(abstract) < 100: continue

                loc    = work.get("primary_location", {}) or {}
                source = loc.get("source", {}) or {}
                journal_name = source.get("display_name", "") or jinfo["title"]

                papers.append({
                    "abstract":         abstract,
                    "journal_name":     journal_name,
                    "journal_issn":     jinfo["issn"],
                    "journal_id_in_db": jinfo["id"],
                    "openalex_id":      work.get("id", ""),
                    "doi":              work.get("doi", ""),
                    "title":            work.get("title", ""),
                    "year":             work.get("publication_year", ""),
                })
                break

        except Exception:
            pass

        if tried % 20 == 0:
            print(f"  Tried {tried} journals → {len(papers)} papers collected")
        time.sleep(0.15)

    random.shuffle(papers)
    print(f"  Final: {len(papers)} test papers from distinct journals ✓")
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL BASELINES (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def build_journal_scope_text(journal):
    parts = [
        journal.get("title", ""),
        journal.get("aims_scope", "") or "",
        journal.get("aims_scope_extended", "") or "",
        " ".join(journal.get("subject_categories", []) or []),
        " ".join(journal.get("top_topics", []) or []),
        " ".join(journal.get("editorial_keywords", []) or []),
        " ".join(journal.get("fields", []) or []),
        " ".join(journal.get("subfields", []) or []),
        journal.get("publisher", "") or "",
    ]
    return " ".join(p for p in parts if p).lower()


class TFIDFBaseline:
    def __init__(self, journals_dict):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self.journals    = journals_dict
        self.journal_ids = list(journals_dict.keys())
        self.cos_sim     = cosine_similarity
        scope_texts = [build_journal_scope_text(journals_dict[jid]) for jid in self.journal_ids]
        print("  Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                          stop_words="english", min_df=2, sublinear_tf=True)
        self.journal_matrix = self.vectorizer.fit_transform(scope_texts)
        print(f"  TF-IDF: {self.journal_matrix.shape[0]} journals × {self.journal_matrix.shape[1]} features")

    def recommend(self, abstract, top_k=10):
        qv   = self.vectorizer.transform([abstract.lower()])
        sims = self.cos_sim(qv, self.journal_matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [{"journal_name": self.journals[self.journal_ids[i]].get("title", ""),
                 "issn": (self.journals[self.journal_ids[i]].get("electronic_issn") or
                          self.journals[self.journal_ids[i]].get("print_issn", "")),
                 "score": float(sims[i])} for i in top_idx]


class BM25Baseline:
    def __init__(self, journals_dict):
        from rank_bm25 import BM25Okapi
        self.journals    = journals_dict
        self.journal_ids = list(journals_dict.keys())
        print("  Building BM25 index...")
        tokenized = [build_journal_scope_text(journals_dict[jid]).split()
                     for jid in self.journal_ids]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25: {len(self.journal_ids)} journals indexed")

    def recommend(self, abstract, top_k=10):
        scores  = self.bm25.get_scores(abstract.lower().split())
        top_idx = scores.argsort()[::-1][:top_k]
        return [{"journal_name": self.journals[self.journal_ids[i]].get("title", ""),
                 "issn": (self.journals[self.journal_ids[i]].get("electronic_issn") or
                          self.journals[self.journal_ids[i]].get("print_issn", "")),
                 "score": float(scores[i])} for i in top_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERNAL TOOL 1: JANE
# ═══════════════════════════════════════════════════════════════════════════════

def call_jane(abstract, top_k=10, timeout=25):
    """
    Query JANE (jane.biosemantics.org) for journal recommendations.

    Endpoint: POST https://jane.biosemantics.org/suggestions.php
    Confirmed from JANE source code (mi-erasmusmc/JANE, JaneClient/index.php):
      <form name='form' action="suggestions.php" method="post">
    
    Required POST fields:
      text          — the abstract/title text
      findJournals  — submit button name, value "Find journals"

    JANE scrapes PubMed abstracts using Lucene MoreLikeThis and returns an
    HTML table with columns: Confidence bar | Journal name | Article Influence | Articles.
    Returns journal names only (no ISSNs) — matching falls back to fuzzy name match.

    Rate limit: caller should wait ≥3 s — JANE is a small academic server.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "beautifulsoup4 not installed: pip install beautifulsoup4 lxml"}

    try:
        resp = requests.post(
            JANE_URL,
            data={
                "text": abstract[:4000],
                "findJournals": "Find journals",   # name of the submit button
                "languageCount": "0",              # required hidden field
            },
            headers={
                "User-Agent": f"PubFitBenchmark/4.0 (mailto:{CONTACT_EMAIL})",
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://jane.biosemantics.org/index.php",
            },
            timeout=timeout,
            allow_redirects=True,
        )

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}

        soup = BeautifulSoup(resp.text, "lxml")
        recommendations = []
        seen = set()

        # JANE renders results as a table with one journal per row.
        # The journal name is in a <td> with class "journal" (or just the 2nd td).
        # Confidence is a <td> containing a coloured bar div — not a plain number.
        # Screenshot shows columns: [confidence bar] | [Journal name + badges] | [AI score] | [Articles]
        # We target the journal name cell specifically.

        # Strategy A: <td class="journal"> (most reliable if class is set)
        for td in soup.find_all("td", class_="journal"):
            # The first <a> inside is the journal link
            link = td.find("a")
            name = (link.get_text(strip=True) if link else td.get_text(separator=" ", strip=True))
            # Strip badge text like "Medline-indexed", "PMC", "High-quality open access"
            name = re.sub(r'\s*(Medline-indexed|PMC|High-quality open access|DOAJ)\s*', '', name, flags=re.I).strip()
            if name and name not in seen and len(name) > 3:
                seen.add(name)
                recommendations.append({"journal_name": name, "issn": "", "score": 0.0})
                if len(recommendations) >= top_k:
                    break

        # Strategy B: generic table rows — journal name is in the 2nd <td> (after confidence bar)
        if not recommendations:
            for row in soup.select("table tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                # Skip header rows
                if cells[0].find("th") or row.find("th"):
                    continue
                name_cell = cells[1] if len(cells) >= 2 else cells[0]
                link = name_cell.find("a")
                name = (link.get_text(strip=True) if link
                        else name_cell.get_text(separator=" ", strip=True))
                name = re.sub(r'\s*(Medline-indexed|PMC|High-quality open access|DOAJ)\s*', '', name, flags=re.I)
                name = name.strip()
                if not name or len(name) < 4:
                    continue
                # Skip obvious non-journal strings
                if name.lower() in ("journal", "confidence", "article influence", "articles"):
                    continue
                if name not in seen:
                    seen.add(name)
                    recommendations.append({"journal_name": name, "issn": "", "score": 0.0})
                    if len(recommendations) >= top_k:
                        break

        if not recommendations:
            # Debug: save a snippet of what JANE returned
            snippet = resp.text[:500].replace("\n", " ")
            return {"error": f"JANE returned 0 results (page snippet: {snippet[:200]})"}

        return {"recommendations": recommendations}

    except requests.exceptions.Timeout:
        return {"error": "JANE timeout"}
    except Exception as e:
        return {"error": f"JANE error: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERNAL TOOL 2: B!SON
# ═══════════════════════════════════════════════════════════════════════════════

def call_bison(abstract, top_k=10, timeout=30):
    """
    Query B!SON (service.tib.eu/bison) for OA journal recommendations.

    Endpoint: POST https://service.tib.eu/bison/api/public/v1/search
    Confirmed from Swagger UI at service.tib.eu/bison/api/schema/swagger-ui
    (visible in screenshot: POST /bison/api/public/v1/search under 'public' section)

    Request body (JSON):
      {"title": "...", "abstract": "...", "references": []}

    B!SON only recommends DOAJ-listed open-access journals.
    If the ground-truth journal is not OA, Hit@K will be 0 by definition.
    """
    try:
        resp = requests.post(
            BISON_URL,
            json={"title": "", "abstract": abstract[:4000], "references": []},
            headers={
                "User-Agent": f"PubFitBenchmark/4.0 (mailto:{CONTACT_EMAIL})",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()

        # B!SON response is a list of journal objects directly, or wrapped in a key.
        # Each journal object has: issn, eissn, title, score (and other metadata).
        results_raw = (data if isinstance(data, list)
                       else data.get("results") or data.get("journals")
                       or data.get("recommendations") or data.get("data") or [])

        recommendations = []
        for item in results_raw[:top_k]:
            name  = (item.get("title") or item.get("journal_title") or item.get("name") or "")
            issn  = (item.get("issn") or item.get("print_issn") or "")
            eissn = (item.get("eissn") or item.get("electronic_issn") or "")
            best_issn = eissn or issn
            score = float(item.get("score") or item.get("similarity_score") or item.get("relevance") or 0.0)
            if name or best_issn:
                recommendations.append({"journal_name": name, "issn": best_issn, "score": score})

        if not recommendations:
            snippet = str(data)[:300]
            return {"error": f"B!SON returned 0 usable results. Response: {snippet}"}

        return {"recommendations": recommendations}

    except requests.exceptions.Timeout:
        return {"error": "B!SON timeout"}
    except Exception as e:
        return {"error": f"B!SON error: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL API CALLERS (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def call_recommend(api_url, abstract, num_results=10, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(f"{api_url}/recommend", json={
                "abstract": abstract[:5000], "num_results": num_results,
                "article_type": "Original Research", "discipline": "Any",
                "language_preference": "auto",
            }, timeout=120)
            if r.status_code == 429:
                time.sleep((attempt + 1) * 30); continue
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < retries - 1: time.sleep(10)
            else: return {"error": "Timeout"}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Failed"}


def call_keyword_search(api_url, abstract, retries=2):
    for attempt in range(retries):
        try:
            r = requests.post(f"{api_url}/keyword-search",
                              json={"abstract": abstract[:5000], "limit": 10}, timeout=30)
            if r.status_code != 200: return {"error": f"HTTP {r.status_code}"}
            return r.json()
        except Exception as e:
            if attempt < retries - 1: time.sleep(5)
            else: return {"error": str(e)}
    return {"error": "Failed"}


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHING (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_name(name):
    if not name: return ""
    name = name.lower().strip()
    name = re.sub(r'^the\s+', '', name)
    name = re.sub(r'\s*\(.*?\)\s*', ' ', name)
    name = re.sub(r'\s*[:–—-]\s+.*$', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def is_journal_match(actual_name, actual_issn, recommended):
    rec_name = recommended.get("journal_name", "")
    rec_issn = recommended.get("issn", "")
    if actual_issn and rec_issn:
        a = actual_issn.replace("-", "").strip()
        r = rec_issn.replace("-", "").strip()
        if a and r and a == r: return True
    a_n = normalize_name(actual_name)
    r_n = normalize_name(rec_name)
    if a_n and r_n:
        if a_n == r_n: return True
        if len(a_n) > 5 and (a_n in r_n or r_n in a_n): return True
        a_t = set(a_n.split()); r_t = set(r_n.split())
        if len(a_t) >= 2 and len(r_t) >= 2:
            if len(a_t & r_t) / max(len(a_t), len(r_t)) >= 0.8: return True
    return False


def find_rank(actual_name, actual_issn, recommendations):
    for i, rec in enumerate(recommendations, 1):
        if is_journal_match(actual_name, actual_issn, rec): return i
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(ranks, k_values=[1, 3, 5, 10]):
    n = len(ranks)
    if n == 0: return {}
    m = {}
    for k in k_values:
        m[f"Hit@{k}"] = sum(1 for r in ranks if 0 < r <= k) / n
    m["MRR"]       = sum(1.0/r if r > 0 else 0.0 for r in ranks) / n
    found          = [r for r in ranks if r > 0]
    m["found_rate"] = len(found) / n
    m["avg_rank"]   = sum(found) / len(found) if found else 0
    m["n"]          = n
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(api_url, papers, journal_full, num_results=10,
                  skip_baselines=False, skip_external=False):

    print(f"\n{'='*70}")
    print(f"  PUBFIT BENCHMARK v4  —  {len(papers)} papers")
    print(f"  Top-{num_results} per query  |  API: {api_url}")
    if not skip_external:
        print(f"  External tools: JANE + B!SON  (adds ~5-6 s/paper)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # ── Internal baselines ──────────────────────────────────────────────────
    tfidf = bm25 = None
    if journal_full and not skip_baselines:
        try:
            tfidf = TFIDFBaseline(journal_full)
        except ImportError:
            print("  ⚠ Install scikit-learn for TF-IDF: pip install scikit-learn")
        try:
            bm25 = BM25Baseline(journal_full)
        except ImportError:
            print("  ⚠ Install rank-bm25 for BM25: pip install rank-bm25")
    elif not journal_full:
        print("  ⚠ No full journal data — run --build-index first for baselines\n")

    # ── External tool availability ──────────────────────────────────────────
    jane_available = bison_available = False
    if not skip_external:
        try:
            from bs4 import BeautifulSoup
            jane_available = True
            print("  ✓ BeautifulSoup found — JANE enabled")
        except ImportError:
            print("  ⚠ Install beautifulsoup4 for JANE: pip install beautifulsoup4 lxml")

        # Quick B!SON ping
        try:
            ping = requests.get("https://service.tib.eu/bison/", timeout=10)
            bison_available = True
            print("  ✓ B!SON service reachable")
        except Exception as e:
            print(f"  ⚠ B!SON service unreachable ({e}) — will skip B!SON")

    # ── Method registry ─────────────────────────────────────────────────────
    methods = {"PubFit AI": [], "PubFit Keywords": []}
    if tfidf:           methods["TF-IDF (≈ Elsevier)"]  = []
    if bm25:            methods["BM25 (IR baseline)"]    = []
    if jane_available:  methods["JANE (PubMed)"]         = []
    if bison_available: methods["B!SON (OA only)"]       = []

    details = []
    errors  = 0

    # ── Per-paper loop ───────────────────────────────────────────────────────
    for i, paper in enumerate(papers):
        pct   = (i + 1) / len(papers) * 100
        short = (paper.get("title") or "")[:40]
        print(f"\n  [{i+1}/{len(papers)}] ({pct:.0f}%) {short}")

        row = {
            "idx":            i + 1,
            "actual_journal": paper["journal_name"],
            "actual_issn":    paper["journal_issn"],
            "doi":            paper.get("doi", ""),
            "year":           paper.get("year", ""),
            "ai_rank": -1, "kw_rank": -1, "tfidf_rank": -1,
            "bm25_rank": -1, "jane_rank": -1, "bison_rank": -1,
        }

        # 1. PubFit AI
        res = call_recommend(api_url, paper["abstract"], num_results)
        if "error" in res:
            errors += 1
            print(f"    ✗ AI error: {res['error']}")
        else:
            recs = res.get("recommendations", [])
            rank = find_rank(paper["journal_name"], paper["journal_issn"], recs)
            methods["PubFit AI"].append(rank)
            row["ai_rank"] = rank
            print(f"    AI: {'✓ rank '+str(rank) if rank else '✗ not in top-'+str(num_results)}"
                  f"  |  actual: {paper['journal_name'][:40]}")

        # 2. PubFit Keywords
        kw = call_keyword_search(api_url, paper["abstract"])
        if "error" not in kw:
            kw_recs = [{"journal_name": r.get("title", ""), "issn": r.get("issn", "")}
                       for r in kw.get("results", [])]
            kw_rank = find_rank(paper["journal_name"], paper["journal_issn"], kw_recs)
            methods["PubFit Keywords"].append(kw_rank)
            row["kw_rank"] = kw_rank

        # 3. TF-IDF
        if tfidf:
            tr = tfidf.recommend(paper["abstract"], top_k=num_results)
            tfidf_rank = find_rank(paper["journal_name"], paper["journal_issn"], tr)
            methods["TF-IDF (≈ Elsevier)"].append(tfidf_rank)
            row["tfidf_rank"] = tfidf_rank

        # 4. BM25
        if bm25:
            br = bm25.recommend(paper["abstract"], top_k=num_results)
            bm25_rank = find_rank(paper["journal_name"], paper["journal_issn"], br)
            methods["BM25 (IR baseline)"].append(bm25_rank)
            row["bm25_rank"] = bm25_rank

        # 5. JANE
        if jane_available:
            time.sleep(3.0)   # respectful rate limit — JANE is a small academic service
            jr = call_jane(paper["abstract"], top_k=num_results)
            if "error" in jr:
                print(f"    ✗ JANE error: {jr['error']}")
                methods["JANE (PubMed)"].append(0)
                row["jane_rank"] = -2  # -2 = API error (distinguished from "not found")
            else:
                jane_recs = jr.get("recommendations", [])
                jane_rank = find_rank(paper["journal_name"], paper["journal_issn"], jane_recs)
                methods["JANE (PubMed)"].append(jane_rank)
                row["jane_rank"] = jane_rank
                print(f"    JANE: {'✓ rank '+str(jane_rank) if jane_rank else '✗ not found'}"
                      f"  ({len(jane_recs)} results returned)")

        # 6. B!SON
        if bison_available:
            time.sleep(1.0)
            bisonr = call_bison(paper["abstract"], top_k=num_results)
            if "error" in bisonr:
                print(f"    ✗ B!SON error: {bisonr['error']}")
                methods["B!SON (OA only)"].append(0)
                row["bison_rank"] = -2
            else:
                bison_recs = bisonr.get("recommendations", [])
                bison_rank = find_rank(paper["journal_name"], paper["journal_issn"], bison_recs)
                methods["B!SON (OA only)"].append(bison_rank)
                row["bison_rank"] = bison_rank
                print(f"    B!SON: {'✓ rank '+str(bison_rank) if bison_rank else '✗ not found'}"
                      f"  ({len(bison_recs)} results returned)")

        details.append(row)
        time.sleep(1.5)

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS  ({len(papers)} papers, {errors} AI errors)")
    idx = load_journal_index() or {}
    db_size = idx.get("total_in_db", idx.get("collected", "?"))
    print(f"  Ground-truth journals all in PubFit's {db_size}-journal database")
    if not skip_external:
        print(f"  ⚠ JANE:  biomedical papers only (PubMed-indexed journals)")
        print(f"  ⚠ B!SON: OA journals only (non-OA ground truth → always 0)")
    print(f"{'='*70}\n")

    rows = []
    for name, ranks in methods.items():
        if not ranks: continue
        m = compute_metrics(ranks)
        rows.append({
            "Method":  name,
            "N":       m["n"],
            "Hit@1":   f"{m['Hit@1']:.1%}",
            "Hit@3":   f"{m['Hit@3']:.1%}",
            "Hit@5":   f"{m['Hit@5']:.1%}",
            "Hit@10":  f"{m['Hit@10']:.1%}",
            "MRR":     f"{m['MRR']:.4f}",
            "Found":   f"{m['found_rate']:.1%}",
        })
    print(tabulate(rows, headers="keys", tablefmt="grid"))

    # Pairwise comparison vs PubFit AI
    ai_m = compute_metrics(methods.get("PubFit AI", []))
    for baseline_name in [
        "TF-IDF (≈ Elsevier)", "BM25 (IR baseline)",
        "JANE (PubMed)", "B!SON (OA only)", "PubFit Keywords"
    ]:
        bl_ranks = methods.get(baseline_name, [])
        if not bl_ranks: continue
        bl_m = compute_metrics(bl_ranks)
        print(f"\n  PubFit AI  vs  {baseline_name}:")
        for k in [1, 3, 5, 10]:
            ai_v  = ai_m.get(f"Hit@{k}", 0)
            bl_v  = bl_m.get(f"Hit@{k}", 0)
            delta = ai_v - bl_v
            rel   = (delta / bl_v * 100) if bl_v > 0 else (float('inf') if delta > 0 else 0)
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"    Hit@{k}:  {ai_v:.1%}  vs  {bl_v:.1%}"
                  f"   ({arrow} {abs(delta):.1%},  {'+' if rel >= 0 else ''}{rel:.0f}% relative)")

    return methods, details


def save_results(methods, details, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(details)
    csv_path = os.path.join(output_dir, f"benchmark_v4_{ts}.csv")
    df.to_csv(csv_path, index=False)

    summary = {name: compute_metrics(ranks) for name, ranks in methods.items() if ranks}
    summary["_meta"] = {"timestamp": ts, "n_papers": len(details), "version": "v4"}
    json_path = os.path.join(output_dir, f"benchmark_v4_summary_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  CSV:   {csv_path}")
    print(f"  JSON:  {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PubFit Benchmark v4 — includes JANE and B!SON external tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build journal index first (one-time, ~15 min):
  python benchmark_v4.py --api-url https://journal-recommender.onrender.com --build-index

  # Full benchmark — all 6 methods (takes ~6 s/paper):
  python benchmark_v4.py --api-url https://journal-recommender.onrender.com --n-papers 100

  # Quick run — no external tools (same as v3):
  python benchmark_v4.py --api-url https://journal-recommender.onrender.com --n-papers 100 --skip-external

  # Biomedical-focused run (JANE works best here):
  python benchmark_v4.py --api-url https://journal-recommender.onrender.com --n-papers 50

Install deps:
  pip install requests pandas tabulate scikit-learn rank-bm25 numpy beautifulsoup4 lxml
        """)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--n-papers", type=int, default=100)
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip TF-IDF + BM25")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip JANE + B!SON (faster, same as v3)")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--year-start", type=int, default=2022)
    parser.add_argument("--year-end",   type=int, default=2025)
    args = parser.parse_args()

    if args.build_index:
        idx, full = build_journal_index(args.api_url)
        if args.n_papers == 100:   # default — index-only run
            print(f"\n✓ Index built. Now run without --build-index to benchmark.")
            return

    journal_index = load_journal_index()
    if not journal_index:
        print("No journal index found. Run with --build-index first.")
        sys.exit(1)
    print(f"Journal index: {journal_index['collected']} journals, "
          f"{len(journal_index['issn_to_id'])} ISSNs")

    journal_full = load_journal_full()
    if journal_full:
        print(f"Full journal data: {len(journal_full)} records")

    print(f"\nChecking API...")
    try:
        r = requests.get(f"{args.api_url}/health", timeout=60)
        if r.status_code == 200:
            h = r.json()
            print(f"  Status: {h.get('status')} | "
                  f"Journals: {h.get('vector_store',{}).get('journals','?')} | "
                  f"LLM: {'on' if h.get('llm_enabled') else 'off'}")
    except Exception as e:
        print(f"  API not responding ({e}) — may need cold start")

    print(f"\nFetching {args.n_papers} test papers from journals in the database...")
    papers = fetch_papers_from_database_journals(
        journal_index, args.n_papers,
        year_range=(args.year_start, args.year_end))

    if not papers:
        print("No papers fetched."); sys.exit(1)

    methods, details = run_benchmark(
        args.api_url, papers, journal_full,
        num_results=args.num_results,
        skip_baselines=args.skip_baselines,
        skip_external=args.skip_external)

    save_results(methods, details, args.output_dir)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK v4 COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
