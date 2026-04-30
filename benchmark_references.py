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


def call_recommend(api_base: str, abstract: str, references: str, n_results=10) -> tuple:
    """Hit /recommend, return (list_of_names, list_of_ids, true_journal_id_if_resolvable)."""
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
        recs = data.get("recommendations", [])
        names = [rec.get("journal_name", "") for rec in recs]
        ids = [rec.get("journal_id", 0) for rec in recs]
        return names, ids
    except Exception as e:
        log.warning(f"  /recommend failed: {e}")
        return [], []


# ─── Cluster similarity (optional — needs embeddings) ───────────────────────
class JournalSimilarity:
    """
    Resolves journal names → IDs and computes embedding cosine similarity
    between journals, so we can score whether the recommended journals are
    in the SAME SCOPE CLUSTER as the true journal — not just the exact match.

    Loaded lazily; if files aren't available, every method returns sentinel
    values that the caller treats as "cluster metrics unavailable".
    """
    def __init__(self, journals_path=None, embeddings_path=None):
        self.available = False
        self.id_to_idx = {}
        self.id_to_subjects = {}
        self.name_to_id = {}        # normalized name → id
        self.embeddings = None
        if not journals_path or not embeddings_path:
            return
        try:
            jp = Path(journals_path); ep = Path(embeddings_path)
            if not jp.exists() or not ep.exists():
                log.warning(f"Cluster files not found ({jp.exists()=}, {ep.exists()=}) — cluster metrics disabled")
                return
            data = np.load(ep)
            self.embeddings = data["embeddings"].astype(np.float32)
            ids = data["ids"]
            self.id_to_idx = {int(jid): i for i, jid in enumerate(ids)}
            # L2-normalize so dot product = cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.embeddings = self.embeddings / norms
            with open(jp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        j = json.loads(line)
                        jid = j.get("id")
                        if jid is None: continue
                        title = j.get("title") or ""
                        self.name_to_id[self._norm(title)] = jid
                        self.id_to_subjects[jid] = set(j.get("subject_categories", []) or [])
                    except Exception: continue
            self.available = True
            log.info(f"Cluster metrics enabled: {len(self.id_to_idx)} embeddings, {len(self.name_to_id)} journals")
        except Exception as e:
            log.warning(f"Cluster setup failed: {e} — cluster metrics disabled")

    @staticmethod
    def _norm(s):
        return "".join(c.lower() for c in (s or "") if c.isalnum())

    def resolve_id(self, journal_name: str):
        """Best-effort: find a journal_id given a fuzzy journal name (e.g. from OpenAlex)."""
        if not self.available or not journal_name:
            return None
        n = self._norm(journal_name)
        if n in self.name_to_id:
            return self.name_to_id[n]
        # Substring fallback (e.g. "Hepatology" → "Hepatology (Baltimore, Md.)")
        for k, jid in self.name_to_id.items():
            if n and (n in k or k in n) and abs(len(n) - len(k)) < 30:
                return jid
        return None

    def cosine(self, jid_a: int, jid_b: int) -> float:
        if not self.available: return 0.0
        a = self.id_to_idx.get(int(jid_a)); b = self.id_to_idx.get(int(jid_b))
        if a is None or b is None: return 0.0
        return float(np.dot(self.embeddings[a], self.embeddings[b]))

    def shares_subject(self, jid_a: int, jid_b: int) -> bool:
        if not self.available: return False
        sa = self.id_to_subjects.get(int(jid_a), set())
        sb = self.id_to_subjects.get(int(jid_b), set())
        return len(sa & sb) > 0


def cluster_hit_at_k(predicted_ids, truth_id, sim: JournalSimilarity,
                      k: int, threshold: float = 0.75) -> int:
    """
    Cluster-aware Hit@K: the recommendation counts as a hit if the true journal
    OR any journal whose embedding cosine similarity to it ≥ threshold appears
    in the top-K.
    """
    if not sim.available or truth_id is None:
        return -1  # sentinel for "unavailable"
    for jid in predicted_ids[:k]:
        if jid == truth_id:
            return 1
        if sim.cosine(jid, truth_id) >= threshold:
            return 1
    return 0


def cluster_reciprocal_rank(predicted_ids, truth_id, sim: JournalSimilarity,
                             threshold: float = 0.75) -> float:
    """Reciprocal rank using cluster matching (sentinel −1 if unavailable)."""
    if not sim.available or truth_id is None:
        return -1.0
    for i, jid in enumerate(predicted_ids, 1):
        if jid == truth_id or sim.cosine(jid, truth_id) >= threshold:
            return 1.0 / i
    return 0.0


def subject_hit_at_k(predicted_ids, truth_id, sim: JournalSimilarity, k: int) -> int:
    """Subject-overlap Hit@K: any predicted journal sharing ≥1 subject category."""
    if not sim.available or truth_id is None:
        return -1
    for jid in predicted_ids[:k]:
        if jid == truth_id or sim.shares_subject(jid, truth_id):
            return 1
    return 0


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


# ─── New framing metrics — what really matters under the survival/cohesion lens
# (Aligned with the agreed framing: don't lose the target from top-10, improve
# the surrounding cohort. These supplement Hit@K rather than replacing it.)

def topk_cohesion(predicted_ids, sim: "JournalSimilarity", k: int = 10) -> float:
    """
    Mean pairwise cosine similarity within the top-K recommendations.
    Higher = more topically uniform "neighborhood of options" surfaced to the user.
    Returns -1 if cluster scoring isn't available.
    """
    if not sim.available:
        return -1.0
    ids = [jid for jid in predicted_ids[:k] if sim.id_to_idx.get(int(jid)) is not None]
    if len(ids) < 2:
        return 0.0
    sims = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sims.append(sim.cosine(ids[i], ids[j]))
    return float(np.mean(sims))


def plausible_alternatives_rate(predicted_ids, truth_id, sim: "JournalSimilarity",
                                  k: int = 10) -> float:
    """
    Fraction of top-K predictions that share ≥1 subject category with the target.
    Higher = more of the recommendations are plausible alternative submission targets.
    Returns -1 if unavailable.
    """
    if not sim.available or truth_id is None:
        return -1.0
    top = predicted_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for jid in top if jid == truth_id or sim.shares_subject(jid, truth_id))
    return float(hits) / len(top)


def survives(predicted_ids, truth_id, sim: "JournalSimilarity", k: int = 10) -> int:
    """
    Did the target journal stay in the top-K? (Boolean, 1 if yes, 0 if no.)
    Uses ID match if available, otherwise -1 sentinel for "unmeasurable".
    """
    if not sim.available or truth_id is None:
        return -1
    return int(truth_id in predicted_ids[:k])


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
                   help="Used to sample journals AND for cluster metrics")
    p.add_argument("--embeddings-file", default=None,
                   help="Optional: .npz with journal embeddings (enables cluster metrics)")
    p.add_argument("--cluster-threshold", type=float, default=0.75,
                   help="Cosine sim threshold for cluster-Hit@K (default 0.75)")
    p.add_argument("--out", default="bench_refs_results.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # Set up cluster-similarity helper (optional — gracefully disabled if files missing)
    sim = JournalSimilarity(args.journals_file, args.embeddings_file)
    if not sim.available:
        log.info("Cluster metrics disabled (pass --embeddings-file to enable)")

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
        "paper_id", "true_journal", "true_journal_id", "ref_count", "self_cites",
        "ctrl_hit1", "ctrl_hit3", "ctrl_hit5", "ctrl_hit10", "ctrl_mrr",
        "trt_hit1",  "trt_hit3",  "trt_hit5",  "trt_hit10",  "trt_mrr",
        # Cluster-aware metrics (-1 means "unavailable" — no embeddings file)
        "ctrl_chit1","ctrl_chit3","ctrl_chit5","ctrl_chit10","ctrl_cmrr",
        "trt_chit1", "trt_chit3", "trt_chit5", "trt_chit10", "trt_cmrr",
        # Subject-overlap metrics (broader cluster definition)
        "ctrl_shit3","trt_shit3","ctrl_shit10","trt_shit10",
        # NEW framing metrics (survival, cohesion, plausible alternatives)
        "ctrl_survive10","trt_survive10",
        "ctrl_cohesion10","trt_cohesion10",
        "ctrl_alt10","trt_alt10",
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
            names_ctrl, ids_ctrl = call_recommend(args.api, paper["abstract"], references="")
            time.sleep(0.3)
            names_trt,  ids_trt  = call_recommend(args.api, paper["abstract"], references=refs_str)
            time.sleep(0.3)

            if not names_ctrl and not names_trt:
                log.warning("  both calls failed — skipping")
                continue

            true_jid = sim.resolve_id(paper["true_journal"]) if sim.available else None
            thr = args.cluster_threshold

            row = {
                "paper_id": paper["paper_id"],
                "true_journal": paper["true_journal"],
                "true_journal_id": true_jid if true_jid is not None else "",
                "ref_count": len(cited_journals),
                "self_cites": self_cites,
                "ctrl_hit1":  hit_at_k(names_ctrl, paper["true_journal"], 1),
                "ctrl_hit3":  hit_at_k(names_ctrl, paper["true_journal"], 3),
                "ctrl_hit5":  hit_at_k(names_ctrl, paper["true_journal"], 5),
                "ctrl_hit10": hit_at_k(names_ctrl, paper["true_journal"], 10),
                "ctrl_mrr":   reciprocal_rank(names_ctrl, paper["true_journal"]),
                "trt_hit1":   hit_at_k(names_trt, paper["true_journal"], 1),
                "trt_hit3":   hit_at_k(names_trt, paper["true_journal"], 3),
                "trt_hit5":   hit_at_k(names_trt, paper["true_journal"], 5),
                "trt_hit10":  hit_at_k(names_trt, paper["true_journal"], 10),
                "trt_mrr":    reciprocal_rank(names_trt, paper["true_journal"]),
                # Cluster metrics (sentinel −1 if unavailable)
                "ctrl_chit1":  cluster_hit_at_k(ids_ctrl, true_jid, sim, 1, thr),
                "ctrl_chit3":  cluster_hit_at_k(ids_ctrl, true_jid, sim, 3, thr),
                "ctrl_chit5":  cluster_hit_at_k(ids_ctrl, true_jid, sim, 5, thr),
                "ctrl_chit10": cluster_hit_at_k(ids_ctrl, true_jid, sim, 10, thr),
                "ctrl_cmrr":   cluster_reciprocal_rank(ids_ctrl, true_jid, sim, thr),
                "trt_chit1":   cluster_hit_at_k(ids_trt, true_jid, sim, 1, thr),
                "trt_chit3":   cluster_hit_at_k(ids_trt, true_jid, sim, 3, thr),
                "trt_chit5":   cluster_hit_at_k(ids_trt, true_jid, sim, 5, thr),
                "trt_chit10":  cluster_hit_at_k(ids_trt, true_jid, sim, 10, thr),
                "trt_cmrr":    cluster_reciprocal_rank(ids_trt, true_jid, sim, thr),
                # Subject-overlap (broader cluster definition)
                "ctrl_shit3":  subject_hit_at_k(ids_ctrl, true_jid, sim, 3),
                "trt_shit3":   subject_hit_at_k(ids_trt, true_jid, sim, 3),
                "ctrl_shit10": subject_hit_at_k(ids_ctrl, true_jid, sim, 10),
                "trt_shit10":  subject_hit_at_k(ids_trt, true_jid, sim, 10),
                # NEW framing metrics — what we actually optimize for
                "ctrl_survive10":   survives(ids_ctrl, true_jid, sim, 10),
                "trt_survive10":    survives(ids_trt,  true_jid, sim, 10),
                "ctrl_cohesion10":  topk_cohesion(ids_ctrl, sim, 10),
                "trt_cohesion10":   topk_cohesion(ids_trt,  sim, 10),
                "ctrl_alt10":       plausible_alternatives_rate(ids_ctrl, true_jid, sim, 10),
                "trt_alt10":        plausible_alternatives_rate(ids_trt,  true_jid, sim, 10),
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

    # ── CLUSTER-AWARE REPORT ────────────────────────────────────────────────
    # Only run if cluster metrics are available (sentinel −1 = unavailable)
    has_cluster = any(r.get("ctrl_chit3", -1) != -1 for r in rows)
    if not has_cluster:
        print("\n  (Cluster-aware metrics not computed — pass --embeddings-file to enable)")
        return

    # Filter to rows where cluster scoring was actually possible
    cluster_rows = [r for r in rows if r.get("ctrl_chit3", -1) != -1
                    and r.get("trt_chit3", -1) != -1]
    if not cluster_rows:
        print("\n  No rows with resolvable true-journal IDs — cluster report skipped.")
        return

    print("\n" + "=" * 70)
    print(f"CLUSTER-AWARE METRICS (n={len(cluster_rows)} of {len(rows)} resolved)")
    print(f"  Cluster definition: cosine similarity ≥ {args.cluster_threshold} to true journal")
    print("=" * 70)
    print("  Counts a hit if the recommended journal IS the target OR a")
    print("  topically similar journal (same scope cluster). This rewards the")
    print("  algorithm for surfacing valid alternative submission targets, not")
    print("  just the exact title the author happened to choose.\n")

    def fmt_cluster_summary(label, summary, prefix="c"):
        print(f"  {label}:")
        labels = ['Hit@1','Hit@3','Hit@5','Hit@10','MRR']
        for L in labels:
            print(f"    cluster-{L:6s} = {summary[L]:.3f}")
        print()

    # Cluster summarizer — same shape as `summarize` but reads chitN/cmrr
    def summarize_cluster(rs, key):
        out = {}
        for k in (1, 3, 5, 10):
            out[f"Hit@{k}"] = float(np.mean([r[f"{key}_chit{k}"] for r in rs]))
        out["MRR"] = float(np.mean([r[f"{key}_cmrr"] for r in rs]))
        return out

    fmt_cluster_summary("CONTROL  (abstract only)",      summarize_cluster(cluster_rows, "ctrl"))
    fmt_cluster_summary("TREATMENT (abstract + references)", summarize_cluster(cluster_rows, "trt"))

    print("  PAIRED BOOTSTRAP DIFFERENCE (cluster metrics):")
    for k in (1, 3, 5, 10):
        d, lo, hi = paired_bootstrap_ci(cluster_rows, "ctrl", "trt", f"chit{k}")
        sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
        print(f"    {sig} Δcluster-Hit@{k:<2} = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    d, lo, hi = paired_bootstrap_ci(cluster_rows, "ctrl", "trt", "cmrr")
    sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
    print(f"    {sig} Δcluster-MRR    = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # Subject-overlap (broader: any shared subject category)
    print("\n  SUBJECT-OVERLAP HIT@K (broader cluster definition):")
    print("  ('hit' if any predicted journal shares ≥1 subject category with target)")
    sub_rows = [r for r in rows if r.get("ctrl_shit3", -1) != -1]
    if sub_rows:
        for k in (3, 10):
            cs = float(np.mean([r[f"ctrl_shit{k}"] for r in sub_rows]))
            ts = float(np.mean([r[f"trt_shit{k}"] for r in sub_rows]))
            d, lo, hi = paired_bootstrap_ci(sub_rows, "ctrl", "trt", f"shit{k}")
            sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
            print(f"    Hit@{k:<2}: ctrl={cs:.3f}  trt={ts:.3f}  Δ={d:+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")

    # Re-stratify by self-citation under the cluster-aware lens
    no_self_c = [r for r in cluster_rows if r["self_cites"] == 0]
    yes_self_c = [r for r in cluster_rows if r["self_cites"] > 0]
    if no_self_c:
        print(f"\n  No-self-cite stratum (n={len(no_self_c)}) — purest topical signal:")
        for k in (3, 10):
            d, lo, hi = paired_bootstrap_ci(no_self_c, "ctrl", "trt", f"chit{k}")
            sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
            print(f"    {sig} Δcluster-Hit@{k:<2} = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("=" * 70)

    # ── FRAMING METRICS (what we actually optimize for) ─────────────────────
    framing_rows = [r for r in rows if r.get("ctrl_survive10", -1) != -1
                    and r.get("trt_survive10", -1) != -1]
    if not framing_rows:
        return

    print("\n" + "=" * 70)
    print(f"FRAMING METRICS — what we actually optimize for (n={len(framing_rows)})")
    print("=" * 70)
    print("  Survival rate: did the target stay in top-10 of BOTH conditions?")
    print("    Higher is better. Worst failure mode = treatment kicks target out.")
    print("  Top-10 cohesion: mean pairwise similarity within top-10.")
    print("    Higher = recommendations form a tighter neighborhood of options.")
    print("  Plausible alternatives: fraction of top-10 sharing a subject category.")
    print("    Higher = more of the recommendations are valid submission targets.\n")

    def m(rs, key):
        return float(np.mean([r[key] for r in rs]))

    survive_ctrl = m(framing_rows, "ctrl_survive10")
    survive_trt = m(framing_rows, "trt_survive10")
    coh_ctrl = m(framing_rows, "ctrl_cohesion10")
    coh_trt = m(framing_rows, "trt_cohesion10")
    alt_ctrl = m(framing_rows, "ctrl_alt10")
    alt_trt = m(framing_rows, "trt_alt10")

    # Joint-survival (target in top-10 in both): use AND
    joint = sum(1 for r in framing_rows if r["ctrl_survive10"] == 1 and r["trt_survive10"] == 1) / len(framing_rows)
    only_ctrl = sum(1 for r in framing_rows if r["ctrl_survive10"] == 1 and r["trt_survive10"] == 0) / len(framing_rows)
    only_trt = sum(1 for r in framing_rows if r["ctrl_survive10"] == 0 and r["trt_survive10"] == 1) / len(framing_rows)

    print(f"  TARGET SURVIVAL @ top-10:")
    print(f"    survives in control:   {survive_ctrl:.3f}")
    print(f"    survives in treatment: {survive_trt:.3f}")
    d, lo, hi = paired_bootstrap_ci(framing_rows, "ctrl", "trt", "survive10")
    sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
    print(f"    {sig} Δsurvival = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"    survives in BOTH:        {joint:.3f}")
    print(f"    only in control (lost):  {only_ctrl:.3f}   ← treatment regressions")
    print(f"    only in treatment (won): {only_trt:.3f}    ← treatment recoveries")

    print(f"\n  TOP-10 COHESION (mean pairwise similarity):")
    print(f"    control:   {coh_ctrl:.3f}")
    print(f"    treatment: {coh_trt:.3f}")
    d, lo, hi = paired_bootstrap_ci(framing_rows, "ctrl", "trt", "cohesion10")
    sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
    print(f"    {sig} Δcohesion = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print(f"\n  PLAUSIBLE-ALTERNATIVES RATE (fraction of top-10 sharing a subject):")
    print(f"    control:   {alt_ctrl:.3f}")
    print(f"    treatment: {alt_trt:.3f}")
    d, lo, hi = paired_bootstrap_ci(framing_rows, "ctrl", "trt", "alt10")
    sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
    print(f"    {sig} Δalternatives = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
