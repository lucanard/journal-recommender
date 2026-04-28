"""
PubFit references-ablation benchmark
====================================
Quantifies the lift the references signal adds on top of abstract-only matching.

Methodology
-----------
1. Sample N real papers from OpenAlex, drawn from journals that exist in our DB.
2. For each paper, extract:
   - The abstract (from OpenAlex's inverted index)
   - The references list (resolved to journal names via OpenAlex)
   - The true journal of publication (the label)
3. Hit POST /recommend twice for each paper:
   - CONTROL:    references=""
   - TREATMENT:  references=<formatted bibliography>
4. Compute Hit@K and MRR for each condition; report deltas.
5. Stratify by self-citation status (does the paper cite its own target journal?)
   so we can separate "free win from self-citation" from "true topical lift".

Notes
-----
- All API calls cached on disk under ./bench_cache/ — re-runs are nearly free.
- Resumable: writes after every paper, so a kill -9 won't lose progress.
- Stratification matters because real submissions often DO self-cite the
  target journal — so the with-self-cite number is the realistic field number,
  and the no-self-cite number is the hardest test of the topical signal.

Usage
-----
    # Local backend
    python benchmark_references.py --api http://localhost:8000 --n-papers 100

    # Deployed backend
    python benchmark_references.py --api https://journal-recommender.onrender.com --n-papers 50

    # Resume an interrupted run (just rerun the same command)
    python benchmark_references.py --api http://localhost:8000 --n-papers 100

Dependencies: requests, numpy. No extra installs beyond what PubFit already uses.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import requests

log = logging.getLogger("bench_refs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OPENALEX = "https://api.openalex.org"
CACHE_DIR = Path("./bench_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─── tiny on-disk cache ──────────────────────────────────────────────────────
def _cache_get(key: str):
    p = CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_set(key: str, val):
    p = CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".json")
    p.write_text(json.dumps(val))


# ─── OpenAlex helpers ────────────────────────────────────────────────────────
def fetch_url(url: str, max_retries=3):
    """GET with disk cache and retry-on-429."""
    cached = _cache_get(url)
    if cached is not None:
        return cached
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "PubFit-bench/1.0 (mailto:hello@pubfit.ai)"})
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            _cache_set(url, data)
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                log.warning(f"Fetch failed: {url[:80]} → {e}")
                return None
            time.sleep(1 + attempt)
    return None


def reconstruct_abstract(inv_idx: dict) -> str:
    """OpenAlex stores abstracts as inverted indexes — flip back to text."""
    if not inv_idx:
        return ""
    pairs = []
    for word, positions in inv_idx.items():
        for p in positions:
            pairs.append((p, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def sample_papers(target_journals: list, n_papers: int, year_min: int = 2022) -> list:
    """
    Sample papers from journals we know exist in our DB.
    Returns list of {paper_id, abstract, true_journal, references}.
    """
    out = []
    seen_ids = set()
    random.shuffle(target_journals)

    for jname in target_journals:
        if len(out) >= n_papers:
            break
        # Search OpenAlex for recent papers in this journal
        url = f"{OPENALEX}/works?search={quote_plus(jname)}&filter=from_publication_date:{year_min}-01-01,has_abstract:true,type:article&per-page=10&sort=cited_by_count:desc"
        data = fetch_url(url)
        if not data or not data.get("results"):
            continue
        for w in data["results"]:
            if len(out) >= n_papers:
                break
            wid = w.get("id", "")
            if wid in seen_ids:
                continue
            seen_ids.add(wid)
            # Resolve actual journal of publication (the label)
            host = (w.get("primary_location") or {}).get("source") or {}
            actual_journal = host.get("display_name", "")
            if not actual_journal:
                continue
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 100:
                continue
            ref_ids = w.get("referenced_works", []) or []
            if len(ref_ids) < 5:
                continue  # too few refs to be meaningful
            out.append({
                "paper_id": wid,
                "abstract": abstract,
                "true_journal": actual_journal,
                "ref_work_ids": ref_ids[:50],  # cap to keep budget reasonable
                "queried_journal": jname,
            })
            log.info(f"  [{len(out)}/{n_papers}] {actual_journal[:50]:50} | refs={len(ref_ids)}")
    return out


def format_references_string(ref_work_ids: list) -> tuple:
    """
    Resolve each reference work-id to a {title, journal, year} via OpenAlex,
    then format as a Vancouver-style bibliography that PubFit's parser can read.
    Returns (formatted_string, list_of_journal_names_cited).
    """
    lines = []
    cited_journals = []
    for i, wid in enumerate(ref_work_ids, 1):
        # OpenAlex IDs are full URLs — extract the W-id
        slug = wid.rsplit("/", 1)[-1]
        url = f"{OPENALEX}/works/{slug}"
        d = fetch_url(url)
        if not d:
            continue
        title = (d.get("title") or "").replace(".", "").strip()
        host = (d.get("primary_location") or {}).get("source") or {}
        jname = host.get("display_name", "")
        year = d.get("publication_year", "")
        if not jname:
            continue
        cited_journals.append(jname)
        # Vancouver-ish: "1. Title. Journal Name. 2023"
        title_short = title[:100]
        lines.append(f"{i}. {title_short}. {jname}. {year}.")
    return "\n".join(lines), cited_journals


def call_recommend(api_base: str, abstract: str, references: str, n_results=10) -> list:
    """Hit the PubFit /recommend endpoint, return list of journal names in rank order."""
    payload = {
        "abstract": abstract,
        "references": references,
        "num_results": n_results,
        "language_preference": "en",
    }
    try:
        r = requests.post(f"{api_base}/recommend", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return [rec.get("journal_name", "") for rec in data.get("recommendations", [])]
    except Exception as e:
        log.warning(f"  /recommend failed: {e}")
        return []


# ─── Metrics ────────────────────────────────────────────────────────────────
def journal_match(predicted: str, truth: str) -> bool:
    """Loose journal-name match (case-insensitive, ignoring punctuation)."""
    def norm(s):
        return "".join(c.lower() for c in (s or "") if c.isalnum())
    p, t = norm(predicted), norm(truth)
    if not p or not t:
        return False
    return p == t or p in t or t in p


def hit_at_k(ranked: list, truth: str, k: int) -> int:
    return int(any(journal_match(p, truth) for p in ranked[:k]))


def reciprocal_rank(ranked: list, truth: str) -> float:
    for i, p in enumerate(ranked, 1):
        if journal_match(p, truth):
            return 1.0 / i
    return 0.0


def summarize(rows: list, condition_key: str) -> dict:
    """Aggregate Hit@K and MRR over the rows for a given condition."""
    out = {}
    for k in (1, 3, 5, 10):
        out[f"Hit@{k}"] = float(np.mean([r[f"{condition_key}_hit{k}"] for r in rows]))
    out["MRR"] = float(np.mean([r[f"{condition_key}_mrr"] for r in rows]))
    out["n"] = len(rows)
    return out


def paired_bootstrap_ci(rows, key_a, key_b, metric, n_boot=1000, ci=0.95):
    """95% CI on the paired difference (treatment - control) via bootstrap."""
    a = np.array([r[f"{key_a}_{metric}"] for r in rows])
    b = np.array([r[f"{key_b}_{metric}"] for r in rows])
    diffs = b - a
    n = len(diffs)
    boots = np.array([diffs[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1-ci)/2, 1-(1-ci)/2])
    return float(diffs.mean()), float(lo), float(hi)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True, help="PubFit API base URL")
    p.add_argument("--n-papers", type=int, default=100)
    p.add_argument("--journals-file", default="data/journals.jsonl",
                   help="Used to sample which journals to draw papers from")
    p.add_argument("--out", default="bench_refs_results.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # 1) Pick which journals to sample papers from (from our DB)
    journals_path = Path(args.journals_file)
    if not journals_path.exists():
        log.error(f"Journals file not found: {journals_path}")
        log.error("Pass --journals-file pointing to the same .jsonl your VectorStore uses.")
        sys.exit(1)
    journals = []
    with open(journals_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                t = j.get("title")
                if t:
                    journals.append(t)
            except Exception:
                continue
    log.info(f"Loaded {len(journals)} journal titles from DB")
    if len(journals) < 50:
        log.warning("Few journals in DB — sample diversity will suffer.")

    # 2) Resume support: load any prior results
    out_path = Path(args.out)
    rows = []
    seen_papers = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # Cast numeric fields
                for k in r:
                    if k.startswith(("ctrl_", "trt_")) or k in ("ref_count", "self_cites"):
                        try: r[k] = float(r[k])
                        except: pass
                rows.append(r)
                seen_papers.add(r["paper_id"])
        log.info(f"Resuming with {len(rows)} previously-completed papers")

    # 3) Sample papers
    needed = args.n_papers - len(rows)
    if needed > 0:
        log.info(f"Sampling {needed} new papers from OpenAlex")
        new_papers = sample_papers(journals, needed, year_min=2022)
    else:
        new_papers = []
        log.info("All papers already done — skipping sample step")

    # 4) Run benchmark on each new paper
    fieldnames = [
        "paper_id", "true_journal", "ref_count", "self_cites",
        "ctrl_hit1", "ctrl_hit3", "ctrl_hit5", "ctrl_hit10", "ctrl_mrr",
        "trt_hit1",  "trt_hit3",  "trt_hit5",  "trt_hit10",  "trt_mrr",
    ]
    write_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, paper in enumerate(new_papers, 1):
            if paper["paper_id"] in seen_papers:
                continue
            log.info(f"[{i}/{len(new_papers)}] {paper['true_journal'][:50]}")

            # Resolve references
            refs_str, cited_journals = format_references_string(paper["ref_work_ids"])
            if not cited_journals:
                log.info("  skipping — no resolvable references")
                continue
            self_cites = sum(1 for j in cited_journals if journal_match(j, paper["true_journal"]))

            # Run both conditions
            ranked_ctrl = call_recommend(args.api, paper["abstract"], references="")
            time.sleep(0.3)  # be nice to the API
            ranked_trt = call_recommend(args.api, paper["abstract"], references=refs_str)
            time.sleep(0.3)

            if not ranked_ctrl and not ranked_trt:
                log.warning("  both calls failed — skipping")
                continue

            row = {
                "paper_id": paper["paper_id"],
                "true_journal": paper["true_journal"],
                "ref_count": len(cited_journals),
                "self_cites": self_cites,
                "ctrl_hit1":  hit_at_k(ranked_ctrl, paper["true_journal"], 1),
                "ctrl_hit3":  hit_at_k(ranked_ctrl, paper["true_journal"], 3),
                "ctrl_hit5":  hit_at_k(ranked_ctrl, paper["true_journal"], 5),
                "ctrl_hit10": hit_at_k(ranked_ctrl, paper["true_journal"], 10),
                "ctrl_mrr":   reciprocal_rank(ranked_ctrl, paper["true_journal"]),
                "trt_hit1":   hit_at_k(ranked_trt, paper["true_journal"], 1),
                "trt_hit3":   hit_at_k(ranked_trt, paper["true_journal"], 3),
                "trt_hit5":   hit_at_k(ranked_trt, paper["true_journal"], 5),
                "trt_hit10":  hit_at_k(ranked_trt, paper["true_journal"], 10),
                "trt_mrr":    reciprocal_rank(ranked_trt, paper["true_journal"]),
            }
            writer.writerow(row); fout.flush()
            rows.append(row)

    # 5) Report
    if not rows:
        log.error("No rows — nothing to report.")
        return

    print("\n" + "=" * 70)
    print(f"PubFit references-ablation benchmark — n={len(rows)}")
    print("=" * 70)

    def fmt_summary(label, summary):
        print(f"\n  {label}:")
        print(f"    Hit@1  = {summary['Hit@1']:.3f}")
        print(f"    Hit@3  = {summary['Hit@3']:.3f}")
        print(f"    Hit@5  = {summary['Hit@5']:.3f}")
        print(f"    Hit@10 = {summary['Hit@10']:.3f}")
        print(f"    MRR    = {summary['MRR']:.3f}")

    fmt_summary("CONTROL (abstract only)", summarize(rows, "ctrl"))
    fmt_summary("TREATMENT (abstract + references)", summarize(rows, "trt"))

    print("\n  PAIRED BOOTSTRAP DIFFERENCE (treatment − control), 95% CI:")
    for k in (1, 3, 5, 10):
        d, lo, hi = paired_bootstrap_ci(rows, "ctrl", "trt", f"hit{k}")
        sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
        print(f"    {sig} ΔHit@{k:<2} = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    d, lo, hi = paired_bootstrap_ci(rows, "ctrl", "trt", "mrr")
    sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
    print(f"    {sig} ΔMRR    = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # Stratify by self-citation
    no_self = [r for r in rows if r["self_cites"] == 0]
    yes_self = [r for r in rows if r["self_cites"] > 0]
    print(f"\n  STRATIFIED BY SELF-CITATION:")
    print(f"    Papers that cite their target journal: {len(yes_self)} / {len(rows)} ({len(yes_self)/len(rows):.0%})")
    if no_self:
        print("\n    On papers that do NOT self-cite the target journal (hardest test):")
        fmt_summary("      CONTROL", summarize(no_self, "ctrl"))
        fmt_summary("      TREATMENT", summarize(no_self, "trt"))
    if yes_self:
        print("\n    On papers that DO self-cite the target journal (most realistic):")
        fmt_summary("      CONTROL", summarize(yes_self, "ctrl"))
        fmt_summary("      TREATMENT", summarize(yes_self, "trt"))

    print(f"\n  Full results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
