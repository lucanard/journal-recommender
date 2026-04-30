"""
Recommendation Engine v1.1
===========================
Pipeline: Embed all sections → Vector search → Constraint filter → LLM re-rank with full context
"""

import json, logging, re, time
from typing import Optional
from dataclasses import dataclass, field, asdict
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class UserConstraints:
    article_type: str = "Original Research"
    discipline: str = ""
    indexing_required: list[str] = field(default_factory=list)
    indexing_any: bool = False
    oa_preference: str = "Any"
    apc_free_only: bool = False
    max_apc: Optional[float] = None
    min_impact_factor: Optional[float] = None
    target_impact: Optional[str] = None
    keywords: str = ""
    language: str = "en"
    # Manuscript sections (optional, improve matching quality)
    introduction_conclusion: str = ""
    methods: str = ""
    results_summary: str = ""
    # User's reference list — strong topical/editor-fit signal
    references: str = ""


@dataclass
class Recommendation:
    rank: int
    journal_id: int
    journal_name: str
    publisher: str
    issn: str
    oa_model: str
    apc_estimate: str
    indexing: list[str]
    fit: str
    score: float
    reasons: list[str]
    concern: str
    subjects: list[str] = field(default_factory=list)
    impact_proxy: str = ""
    impact_factor: str = ""
    acceptance_rate: str = ""
    review_time: str = ""
    homepage: str = ""
    cited_count: int = 0          # times this journal appears in user's references
    citation_boost: float = 1.0   # total multiplicative boost applied (1.0 = none)
    topical_score: float = 0.0    # cosine similarity to user's reference-cluster centroid (0..1)
    cluster_size: int = 0         # how many cited journals contributed to the centroid


# ═══════════════════════════════════════════════════════════════════════════════
# Reference parsing & journal matching
# ═══════════════════════════════════════════════════════════════════════════════

# Words to ignore when comparing journal-name token sequences
_JOURNAL_STOPWORDS = {"of", "the", "and", "in", "on", "for", "to", "a", "an", "&"}


def _normalize_journal_tokens(name: str) -> list:
    """Lowercase, strip punctuation/stopwords, return list of word tokens."""
    if not name:
        return []
    n = name.lower().strip()
    n = re.sub(r"\([^)]*\)", " ", n)          # remove parentheticals e.g. "(London)"
    n = re.sub(r"[.,;:&/\-]", " ", n)         # punctuation → space
    n = re.sub(r"[^a-z0-9\s]", "", n)         # drop other punctuation
    n = re.sub(r"\s+", " ", n).strip()
    return [t for t in n.split() if t and t not in _JOURNAL_STOPWORDS]


def _looks_like_journal(text: str) -> bool:
    """Quick sanity filter — rejects titles, author lists, URLs."""
    if not text or len(text) < 3 or len(text) > 100:
        return False
    if "://" in text or "@" in text:
        return False
    words = text.split()
    if len(words) > 12:           # too long to be a journal name
        return False
    cap_starts = sum(1 for w in words if w and w[0].isupper())
    return cap_starts >= max(1, len(words) // 2)


def _strip_title_prefix(name: str) -> str:
    """
    If a sentence-break appears inside the captured journal name, keep only the
    part after it. Two heuristics:
      (a) A period is a sentence end if preceded by ≥5 consecutive lowercase
          letters (handles "...polyphenols. Analytical Chemistry").
      (b) A period followed by a single-letter abbreviation (e.g. "J.") signals
          the start of an abbreviation chain (handles short titles like
          "Wine. J. Agric. Food Chem.").
    Abbreviation periods like "Mol.", "Biol.", "Chem.", "Agric." are preserved.
    """
    if not name:
        return name
    # (a) long lowercase word + period + Capital
    parts = re.split(r"(?<=[a-z]{5})\.\s+(?=[A-Z])", name)
    if len(parts) > 1:
        return parts[-1].strip().rstrip(".,;:")
    # (b) any 3+ char word + period + (single-capital + period) — abbrev chain start
    # ("Wine. J.", "SVM. J.") but NOT abbreviation chains themselves ("J. M.", "Mol. B.")
    parts = re.split(r"(?<=[A-Za-z]{3})\.\s+(?=[A-Z]\.\s)", name)
    if len(parts) > 1:
        return parts[-1].strip().rstrip(".,;:")
    return name


def _extract_journal_from_reference(line: str):
    """
    Extract (journal_name, issn) from a single reference line.
    Returns (None, None) if nothing looks like a journal.
    """
    line = line.strip()
    if len(line) < 15:
        return None, None
    # Strip leading numbering: "1.", "[1]", "(1)", "1)"
    line = re.sub(r"^[\[\(]?\s*\d{1,3}\s*[\]\)\.]?\s*", "", line)

    # ISSN — high-confidence anchor (require explicit "ISSN" prefix to avoid
    # matching page ranges like "4500-4510" which share the XXXX-XXXX shape).
    issn = None
    m = re.search(r"\bISSN[\s:]*?(\d{4}-\d{3}[\dX])\b", line, re.IGNORECASE)
    if m:
        issn = m.group(1)

    # Year (1900-2030) — anchor for journal-name extraction
    year_match = re.search(r"\b(19\d{2}|20[0-3]\d)\b", line)

    # Pattern A — italic-marked: *Journal Name* or _Journal Name_
    m = re.search(r"[*_]([A-Z][^*_]{2,80})[*_]", line)
    if m:
        cand = _strip_title_prefix(m.group(1).strip(" .,"))
        if _looks_like_journal(cand):
            return cand, issn

    # Pattern B — Vancouver/AMA: ". J Mol Biol. 2023" — period-bounded chunk before year
    m = re.search(r"\.\s+([A-Z][A-Za-z\.\s&\-]{2,80}?)\.\s+(?:19\d{2}|20[0-3]\d)\b", line)
    if m:
        cand = _strip_title_prefix(m.group(1).strip(" .,"))
        if _looks_like_journal(cand):
            return cand, issn

    # Pattern C — APA: ". Journal Name, 12(3), 100-120"
    m = re.search(r"\.\s+([A-Z][A-Za-z\.\s&\-]{2,80}?),\s+\d+\s*[\(,]", line)
    if m:
        cand = _strip_title_prefix(m.group(1).strip(" .,"))
        if _looks_like_journal(cand):
            return cand, issn

    # Pattern D — fallback: chunk before year (works when title was already stripped)
    if year_match:
        before = line[: year_match.start()].rstrip(" .,;")
        chunks = re.split(r"\.\s+", before)
        if chunks:
            cand = _strip_title_prefix(chunks[-1].strip(" .,"))
            if _looks_like_journal(cand):
                return cand, issn

    return None, issn


def _split_references(refs_text: str) -> list:
    """Split a references blob into individual entries."""
    if not refs_text or not refs_text.strip():
        return []
    txt = refs_text.strip()
    # Split on newlines that are followed by a numbering marker:
    #   "1. ", "12. ", "[3]", "(4)", "5) " — handles mixed styles together.
    marker = r"(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[\.\)])\s+"
    parts = re.split(rf"\n(?=\s*{marker})", txt)
    if len(parts) < 2:
        # No numbering found — fall back to blank-line then single-line split
        parts = re.split(r"\n\s*\n|\n", txt)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 15:
            out.append(p)
    return out


def parse_references(refs_text: str) -> list:
    """
    Parse free-text references into [{journal_name, issn?}, ...].
    Returns empty list if nothing extractable.
    """
    entries = _split_references(refs_text)
    out = []
    for entry in entries:
        name, issn = _extract_journal_from_reference(entry)
        if name or issn:
            out.append({"journal_name": name, "issn": issn})
    return out


def _build_journal_index(journals: dict) -> dict:
    """
    Build lookup indices for fast matching:
      - by_issn:   issn -> journal_id
      - by_first_token: first_token -> [(journal_id, tokens), ...]
    """
    by_issn = {}
    by_first_token = {}
    for jid, j in journals.items():
        # ISSN variants
        for key in ("issn", "electronic_issn", "print_issn"):
            v = j.get(key)
            if v and isinstance(v, str):
                by_issn[v.strip().upper()] = jid
        # Tokenize title
        title = j.get("title") or ""
        toks = _normalize_journal_tokens(title)
        if toks:
            by_first_token.setdefault(toks[0][:4], []).append((jid, toks))
    return {"by_issn": by_issn, "by_first_token": by_first_token}


def _tokens_compatible(cited: list, full: list) -> bool:
    """
    True if `cited` tokens look like an abbreviation of `full` tokens.
    Each cited token must be a prefix of (or equal to) the corresponding full token.
    Allows full to have at most 2 extra trailing tokens.
    """
    if not cited or not full:
        return False
    if len(cited) > len(full):
        return False
    if len(full) - len(cited) > 2:
        return False
    for ct, ft in zip(cited, full):
        if ct == ft:
            continue
        if ft.startswith(ct) or ct.startswith(ft):
            continue
        return False
    return True


def match_cited_journal(cited_name: str, cited_issn, journal_index: dict):
    """Return journal_id (or None) for a single parsed reference."""
    if cited_issn:
        jid = journal_index["by_issn"].get(cited_issn.strip().upper())
        if jid is not None:
            return jid
    if not cited_name:
        return None
    cited_toks = _normalize_journal_tokens(cited_name)
    if not cited_toks:
        return None
    bucket_key = cited_toks[0][:4]
    candidates = journal_index["by_first_token"].get(bucket_key, [])
    # Also check buckets where the full token starts with the cited prefix
    # (e.g. cited "j" → bucket "j", but full could be "jour")
    for k, lst in journal_index["by_first_token"].items():
        if k != bucket_key and (k.startswith(cited_toks[0]) or cited_toks[0].startswith(k)):
            candidates = candidates + lst
    best = None
    for jid, full_toks in candidates:
        if _tokens_compatible(cited_toks, full_toks):
            # Prefer tighter length match
            penalty = abs(len(full_toks) - len(cited_toks))
            if best is None or penalty < best[1]:
                best = (jid, penalty)
    return best[0] if best else None


def build_citation_map(refs_text: str, journal_index: dict):
    """
    Parse references and match them to journals in the database.
    Returns (citation_counts, parsed_count, matched_count).
      citation_counts: {journal_id: count}
    """
    parsed = parse_references(refs_text)
    counts = {}
    matched = 0
    for entry in parsed:
        jid = match_cited_journal(entry.get("journal_name"), entry.get("issn"), journal_index)
        if jid is not None:
            counts[jid] = counts.get(jid, 0) + 1
            matched += 1
    return counts, len(parsed), matched


def build_reference_centroid(citation_counts: dict, store) -> tuple:
    """
    Build a weighted centroid from cited journals' embeddings.

    Each cited journal contributes its embedding vector, weighted by its citation
    count (capped — see CITE_WEIGHT_CAP — to prevent one heavily-cited journal
    from dominating the centroid). The result is L2-normalized so cosine
    similarity reduces to a dot product downstream.

    Parameters
    ----------
    citation_counts : {journal_id: count}  — output of build_citation_map
    store           : the VectorStore (must expose .embeddings, .id_to_idx)

    Returns
    -------
    (centroid_unit_vector, cluster_size)  — or (None, 0) if no embeddings found.
    """
    CITE_WEIGHT_CAP = 5  # weight per journal saturates at 5 cites
    if not citation_counts:
        return None, 0
    if store is None or not hasattr(store, "embeddings") or store.embeddings is None:
        return None, 0

    vecs, weights = [], []
    for jid, cites in citation_counts.items():
        idx_pos = store.id_to_idx.get(int(jid))
        if idx_pos is None:
            continue
        vecs.append(store.embeddings[idx_pos])
        weights.append(min(cites, CITE_WEIGHT_CAP))

    if not vecs:
        return None, 0

    V = np.asarray(vecs, dtype=np.float32)              # (k, D)
    w = np.asarray(weights, dtype=np.float32)[:, None]  # (k, 1)
    centroid = (V * w).sum(axis=0) / max(w.sum(), 1e-9) # (D,)
    n = float(np.linalg.norm(centroid))
    if n < 1e-9:
        return None, 0
    return centroid / n, len(vecs)


# ═══════════════════════════════════════════════════════════════════════════════
# Recommendation engine
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendationEngine:

    def __init__(self, vector_store, embedding_service, llm_client=None):
        self.store = vector_store
        self.embedder = embedding_service
        self.llm = llm_client
        self._journal_index = None  # built lazily on first use

    def _get_journal_index(self):
        """Lazy-build the journal lookup index used for citation matching."""
        if self._journal_index is None:
            self._journal_index = _build_journal_index(self.store.journals)
            log.info(
                f"Built journal index for citation matching: "
                f"{len(self._journal_index['by_issn'])} ISSN keys, "
                f"{len(self._journal_index['by_first_token'])} title-token buckets"
            )
        return self._journal_index

    def recommend(self, abstract, constraints, num_results=3, candidate_pool=0):
        timing = {}

        # Step 0: Embed (abstract + optional sections combined)
        t0 = time.time()
        query_text = self._build_query_text(abstract, constraints)
        query_embedding = self.embedder.embed_query(query_text)
        timing["embed_ms"] = int((time.time() - t0) * 1000)

        # Step 0b: Citation-graph query displacement
        # ──────────────────────────────────────────
        # Build a centroid from the embeddings of journals the user cites, then
        # shift the query embedding partway toward it BEFORE retrieval. This
        # gives the references signal differential influence (not a uniform
        # boost across the cluster) and shapes both retrieval and ranking with
        # one principled operation. Set DISPLACEMENT_BETA = 0.0 to disable.
        t0 = time.time()
        cite_counts = {}
        refs_parsed = 0
        refs_matched = 0
        cluster_size = 0
        centroid = None
        DISPLACEMENT_BETA = 0.20  # 0.0 = pure abstract, 1.0 = pure citations

        if constraints.references and constraints.references.strip():
            try:
                idx = self._get_journal_index()
                cite_counts, refs_parsed, refs_matched = build_citation_map(
                    constraints.references, idx
                )
                if cite_counts:
                    centroid, cluster_size = build_reference_centroid(cite_counts, self.store)
                    if centroid is not None and DISPLACEMENT_BETA > 0:
                        # Normalize the abstract query so the interpolation is dimensionally clean
                        q = np.asarray(query_embedding, dtype=np.float32)
                        qn = float(np.linalg.norm(q))
                        if qn > 1e-9:
                            q_unit = q / qn
                            displaced = (1.0 - DISPLACEMENT_BETA) * q_unit + DISPLACEMENT_BETA * centroid
                            dn = float(np.linalg.norm(displaced))
                            if dn > 1e-9:
                                displaced = displaced / dn
                                query_embedding = displaced.tolist()
                                log.info(
                                    f"Query displaced toward {cluster_size}-journal citation cluster "
                                    f"(β={DISPLACEMENT_BETA}, refs_parsed={refs_parsed}, refs_matched={refs_matched})"
                                )
            except Exception as e:
                log.warning(f"Query displacement failed (non-fatal): {e}")
        timing["citation_ms"] = int((time.time() - t0) * 1000)

        # Step 1: Vector search (with possibly-displaced query)
        t0 = time.time()
        search_k = candidate_pool if candidate_pool > 0 else len(self.store.ids)
        raw_results = self.store.search(query_embedding, top_k=search_k)
        timing["search_ms"] = int((time.time() - t0) * 1000)

        # Step 1b: Minimum similarity filter — discard very low matches
        MIN_SIMILARITY = 0.10  # Below this, the match is essentially random
        if raw_results and raw_results[0]["score"] < MIN_SIMILARITY:
            log.warning(f"Best match score {raw_results[0]['score']:.4f} below threshold {MIN_SIMILARITY}. Abstract may not be meaningful.")
            return {
                "recommendations": [],
                "analysis_summary": "No meaningful matches found. The abstract may be too short, too general, or not contain recognizable scientific content. Please paste a complete research abstract.",
                "timing": timing,
                "candidates_searched": len(raw_results),
                "candidates_after_filter": 0,
                "constraints_relaxed": False,
            }

        # Step 2: Constraint filter
        t0 = time.time()
        filtered, stats = self._apply_constraints(raw_results, constraints)
        timing["filter_ms"] = int((time.time() - t0) * 1000)
        log.info(f"Pipeline: {len(raw_results)} → {len(filtered)} after filtering")

        # Step 2b: Language boost — prioritize journals matching abstract language
        if constraints.language and constraints.language not in ("en", "any", ""):
            lang_map = {
                "zh": ["chinese", "zh", "mandarin"],
                "ar": ["arabic", "ar"],
                "es": ["spanish", "es", "español"],
                "fr": ["french", "fr", "français"],
                "de": ["german", "de", "deutsch"],
                "pt": ["portuguese", "pt", "português"],
                "ja": ["japanese", "ja"],
                "ko": ["korean", "ko"],
            }
            lang_terms = lang_map.get(constraints.language, [constraints.language])
            for r in filtered:
                j = r["journal"]
                j_langs = [l.lower() for l in j.get("languages", [])]
                if any(term in lang for term in lang_terms for lang in j_langs):
                    r["score"] = r["score"] * 1.5  # 50% boost for language match
                    r["language_match"] = True
            # Re-sort by boosted score
            filtered.sort(key=lambda x: x["score"], reverse=True)
            log.info(f"Language boost applied for '{constraints.language}'")

        relaxed = False
        if len(filtered) == 0 and len(raw_results) > 0:
            log.info("All filtered out — relaxing constraints")
            filtered = raw_results
            relaxed = True

        # Step 2c: Annotate candidates with citation/topical info (no boost)
        # ─────────────────────────────────────────────────────────────────
        # The query displacement (Step 0b) has already shaped the ranking.
        # Here we just attach explanatory fields so the UI and LLM rerank
        # can show "Cited Nx" / "Topically aligned" reasons.
        if cite_counts and centroid is not None:
            try:
                for r in filtered:
                    jid = r["journal"].get("id")
                    cites = cite_counts.get(jid, 0)
                    r["cited_count"] = cites
                    r["cluster_size"] = cluster_size
                    r["citation_boost"] = 1.0  # no longer used; kept for response stability
                    # Topical similarity to centroid — purely informational now
                    topical_sim = 0.0
                    pos = self.store.id_to_idx.get(int(jid)) if jid is not None else None
                    if pos is not None:
                        topical_sim = max(0.0, float(np.dot(self.store.embeddings[pos], centroid)))
                    r["topical_score"] = round(topical_sim, 4)
            except Exception as e:
                log.warning(f"Annotation failed (non-fatal): {e}")

        # Step 3: LLM re-rank (10 candidates for better selection)
        t0 = time.time()
        if self.llm and len(filtered) > 0:
            recs, summary = self._llm_rerank(abstract, constraints, filtered[:10], num_results)
        else:
            recs = self._heuristic_rank(filtered, num_results)
            summary = self._generate_summary_heuristic(abstract, recs)
        timing["rerank_ms"] = int((time.time() - t0) * 1000)

        if relaxed:
            parts = []
            for key, label in [("removed_by_indexing", "indexing"), ("removed_by_oa", "OA"), ("removed_by_apc", "APC"), ("removed_by_apc_free", "APC-free"), ("removed_by_impact", "impact"), ("removed_by_min_if", "min IF")]:
                if stats.get(key, 0) > 0: parts.append(f"{label} removed {stats[key]}")
            warning = "No journals matched ALL constraints"
            if parts: warning += f" ({'; '.join(parts)})"
            warning += ". Showing best matches without filters."
            summary = warning + " " + summary

        return {
            "recommendations": [asdict(r) for r in recs],
            "analysis_summary": summary,
            "timing": timing,
            "candidates_searched": len(raw_results),
            "candidates_after_filter": len(filtered) if not relaxed else 0,
            "constraints_relaxed": relaxed,
            "references_parsed": refs_parsed,
            "references_matched": refs_matched,
            "unique_journals_cited": len(cite_counts),
            "topical_cluster_size": cluster_size,
        }

    def _build_query_text(self, abstract, constraints):
        """Combine all manuscript sections into one rich query for embedding."""
        parts = [abstract]

        if constraints.keywords:
            parts.append(f"Keywords: {constraints.keywords}")
        if constraints.discipline:
            parts.append(f"Discipline: {constraints.discipline}")
        if constraints.article_type:
            parts.append(f"Article type: {constraints.article_type}")

        # Additional sections — truncate to keep embedding focused
        if constraints.introduction_conclusion:
            parts.append(f"Contribution and context: {constraints.introduction_conclusion[:1000]}")
        if constraints.methods:
            parts.append(f"Methodology: {constraints.methods[:800]}")
        if constraints.results_summary:
            parts.append(f"Key findings: {constraints.results_summary[:800]}")

        return " ".join(parts)

    def _apply_constraints(self, results, constraints):
        filtered = []
        stats = {"removed_by_indexing": 0, "removed_by_oa": 0, "removed_by_apc": 0, "removed_by_apc_free": 0, "removed_by_impact": 0, "removed_by_min_if": 0}

        for r in results:
            j = r["journal"]

            if constraints.indexing_required:
                jidx = set()
                if j.get("indexed_pubmed"): jidx.update(["PubMed/MEDLINE", "PubMed"])
                if j.get("in_doaj"): jidx.add("DOAJ")
                for n in j.get("indexing", []): jidx.add(n)
                if not set(constraints.indexing_required).issubset(jidx):
                    stats["removed_by_indexing"] += 1; continue

            if constraints.indexing_any:
                # Must be indexed somewhere — PubMed, DOAJ, or any listed indexing
                has_any_index = j.get("indexed_pubmed") or j.get("in_doaj") or bool(j.get("indexing"))
                if not has_any_index:
                    stats["removed_by_indexing"] += 1; continue

            if constraints.oa_preference == "Open Access Only":
                oa = (j.get("oa_model") or "").lower()
                if not (j.get("in_doaj") or j.get("is_oa") or "full oa" in oa):
                    stats["removed_by_oa"] += 1; continue

            if constraints.apc_free_only:
                apc_d = (j.get("apc_display") or "").lower()
                if j.get("has_apc") is True:
                    stats["removed_by_apc_free"] += 1; continue
                if apc_d and "free" not in apc_d and "no apc" not in apc_d:
                    if re.search(r'\d{3,}', apc_d):
                        stats["removed_by_apc_free"] += 1; continue

            if constraints.max_apc is not None:
                v = self._parse_apc(j.get("apc_display", ""))
                if v is not None and v > constraints.max_apc:
                    stats["removed_by_apc"] += 1; continue

            if constraints.target_impact:
                imp = j.get("impact_proxy", "Unknown")
                if imp != "Unknown" and not self._impact_matches(imp, constraints.target_impact):
                    stats["removed_by_impact"] += 1; continue

            if constraints.min_impact_factor is not None:
                jif = j.get("two_yr_mean_citedness")
                if jif is not None and jif < constraints.min_impact_factor:
                    stats["removed_by_min_if"] += 1; continue

            filtered.append(r)
        return filtered, stats

    def _parse_apc(self, s):
        if not s: return None
        if "free" in s.lower() or "no apc" in s.lower(): return 0.0
        nums = re.findall(r'[\d,]+\.?\d*', s)
        if nums:
            try: return float(nums[-1].replace(",", ""))
            except: pass
        return None

    def _impact_matches(self, ji, target):
        levels = {"Q1 (High)": 4, "Q1-Q2": 3, "Q2-Q3": 2, "Q3-Q4": 1, "Q4": 0}
        return levels.get(ji, -1) >= levels.get(target, -1)

    def _heuristic_rank(self, candidates, num_results):
        scored = [(r["score"] * 0.7 + r["journal"].get("completeness_score", 10) / 100 * 0.3, r) for r in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        recs = []
        for rank, (score, r) in enumerate(scored[:num_results], 1):
            j = r["journal"]
            idx = (["PubMed"] if j.get("indexed_pubmed") else []) + (["DOAJ"] if j.get("in_doaj") else [])
            cited = r.get("cited_count", 0)
            topical = r.get("topical_score", 0.0)
            reasons = [f"Similarity: {r['score']:.3f}", f"Subjects: {', '.join(j.get('subject_categories', [])[:3]) or 'N/A'}"]
            if cited > 0:
                reasons.insert(0, f"Cited {cited}× in your references")
            elif topical >= 0.5:
                reasons.insert(0, f"Topically aligned with your references (sim={topical:.2f})")
            recs.append(Recommendation(
                rank=rank, journal_id=j.get("id", 0), journal_name=j.get("title", "?"), publisher=j.get("publisher", "?"),
                issn=j.get("issn", j.get("electronic_issn", "")), oa_model=j.get("oa_model", "?"),
                apc_estimate=j.get("apc_display", "?"), indexing=idx,
                fit="High" if score > 0.6 else "Medium", score=round(score, 4),
                reasons=reasons,
                concern="Heuristic scoring — enable LLM for better results.",
                subjects=j.get("subject_categories", [])[:5], impact_proxy=j.get("impact_proxy", ""),
                impact_factor=str(j.get("two_yr_mean_citedness", "?")),
                acceptance_rate="N/A", review_time="N/A", homepage=j.get("homepage", ""),
                cited_count=cited, citation_boost=r.get("citation_boost", 1.0),
                topical_score=topical, cluster_size=0,
            ))
        return recs

    def _generate_summary_heuristic(self, abstract, recs):
        if not recs: return "No matching journals found."
        return f"Top match: {recs[0].journal_name}. Enable LLM for detailed analysis."

    def _llm_rerank(self, abstract, constraints, candidates, num_results):
        # Build candidate descriptions
        cands = []
        for i, r in enumerate(candidates, 1):
            j = r["journal"]
            d = {
                "num": i, "title": j.get("title", ""), "publisher": j.get("publisher", ""),
                "subjects": j.get("subject_categories", [])[:8],
                "aims_scope": (j.get("aims_scope", "") or "")[:500],
                "oa_model": j.get("oa_model", ""), "apc": j.get("apc_display", ""),
                "impact": j.get("impact_proxy", ""),
                "impact_factor_approx": j.get("two_yr_mean_citedness", "Unknown"),
                "h_index": j.get("h_index", "Unknown"),
                "indexing": [], "similarity": round(r["score"], 4),
                "cited_in_user_refs": r.get("cited_count", 0),
                "topical_alignment": round(r.get("topical_score", 0.0), 3),
            }
            if j.get("indexed_pubmed"): d["indexing"].append("PubMed")
            if j.get("in_doaj"): d["indexing"].append("DOAJ")
            cands.append(d)

        system_prompt = """You are an expert academic journal recommendation system.

TASK:
1. Re-rank candidates by scope fit, audience, methodology match
2. 2-4 reasons per journal explaining fit
3. One concern per journal
4. Provide CURRENT data for each field

RULES — APC:
- CURRENT 2024-2025 rates only. APCs have risen significantly.
- If unsure: "Approximately $X (verify with journal)"
- HYBRID journals: state BOTH routes. "Free via subscription; $X for OA route."
- APC-free request: hybrid OK (subscription route). Pure Gold OA = NOT acceptable.

RULES — IMPACT FACTOR:
- Use most recent IF (2023/2024 JCR preferred)
- If min IF specified, only recommend journals meeting it

RULES — ADDITIONAL DATA:
- acceptance_rate: approximate (e.g. "20-25%") or "Not publicly available"
- review_time: time to first decision (e.g. "4-6 weeks") or "Not publicly available"

RULES — MANUSCRIPT SECTIONS:
- The user may provide Introduction/Conclusion, Methods, and Results in addition to the abstract
- Use ALL provided sections to assess fit. Methods help match methodology-focused journals. Results help match journals that publish similar evidence types.
- Weigh scope fit (abstract + intro/conclusion) highest, then methodology fit, then evidence type.

RULES — REFERENCES SIGNAL (two related fields):
- "cited_in_user_refs": how many times the user explicitly cited THIS journal.
- "topical_alignment" (0..1): cosine similarity to the centroid of journals the user cites — a measure of how well this candidate fits the user's overall topical neighborhood, EVEN if it was not cited directly.
- The intent is to surface a NEIGHBORHOOD of related journals, not just one. Treat both signals as positive evidence; do NOT use them to manufacture a winner.
- Strong positive: topical_alignment ≥ 0.5 AND/OR cited_in_user_refs ≥ 2 — mention this in reasons (e.g. "Topically aligned with the user's reference cluster" or "Cited 4× in user's references").
- Mild positive: topical_alignment 0.3–0.5 — worth noting briefly.
- Do NOT promote a candidate purely on these signals if scope clearly mismatches the abstract.
- Do NOT collapse the top results onto a single journal even when one has high cited_in_user_refs; keep diversity within the topical cluster so the user has real options to choose from.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "recommendations": [
    {
      "candidate_num": 1,
      "journal": "Journal Name",
      "publisher": "Publisher",
      "oa_model": "Hybrid",
      "apc_estimate": "Free via subscription; $4,500 for OA (2024)",
      "impact_factor": "5.2",
      "acceptance_rate": "20-25%",
      "review_time": "4-6 weeks",
      "indexing": ["PubMed"],
      "fit": "High",
      "reasons": ["Scope match reason", "Methodology fit reason", "Audience reason"],
      "concern": "One concern"
    }
  ],
  "analysis_summary": "Brief summary of research topic and why these journals fit"
}"""

        # Constraint descriptions
        apc_text = "No limit"
        if constraints.apc_free_only:
            apc_text = "FREE ONLY — only free-to-publish journals. Hybrid OK (subscription route). Gold OA with mandatory APC = NOT acceptable."
        elif constraints.max_apc:
            apc_text = f"Max ${constraints.max_apc}"

        if_text = "No minimum"
        if constraints.min_impact_factor:
            if_text = f"Minimum IF >= {constraints.min_impact_factor}"

        # Build manuscript context from all provided sections
        manuscript_sections = f"ABSTRACT:\n{abstract}"
        if constraints.introduction_conclusion:
            manuscript_sections += f"\n\nINTRODUCTION / CONCLUSION:\n{constraints.introduction_conclusion[:2000]}"
        if constraints.methods:
            manuscript_sections += f"\n\nMATERIALS & METHODS:\n{constraints.methods[:2000]}"
        if constraints.results_summary:
            manuscript_sections += f"\n\nKEY RESULTS:\n{constraints.results_summary[:2000]}"

        user_msg = f"""{manuscript_sections}

CONSTRAINTS:
- Article type: {constraints.article_type}
- Discipline: {constraints.discipline or 'Not specified'}
- Indexing: {', '.join(constraints.indexing_required) or 'None'}
- OA: {constraints.oa_preference}
- APC: {apc_text}
- Impact Factor: {if_text}
{f'- Keywords: {constraints.keywords}' if constraints.keywords else ''}

CANDIDATES:
{json.dumps(cands, indent=2)}

Select the top {num_results}. For EVERY journal: state publishing model (especially if hybrid), current APC, IF, acceptance rate, review time.{f' Focus on {constraints.discipline} journals.' if constraints.discipline else ''}"""

        try:
            response = self.llm.create_message(system_prompt, user_msg)
            cleaned = response.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = cleaned.strip()
            if not cleaned.startswith('{'):
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match: cleaned = match.group(0)

            log.info(f"LLM response (first 200): {cleaned[:200]}")
            parsed = json.loads(cleaned)

            recs = []
            for rank, rec in enumerate(parsed.get("recommendations", [])[:num_results], 1):
                cn = rec.get("candidate_num", rank)
                cj = candidates[cn - 1]["journal"] if 1 <= cn <= len(candidates) else {}
                cr = candidates[cn - 1] if 1 <= cn <= len(candidates) else {}
                recs.append(Recommendation(
                    rank=rank,
                    journal_id=cj.get("id", 0),
                    journal_name=rec.get("journal", cj.get("title", "")),
                    publisher=rec.get("publisher", cj.get("publisher", "")),
                    issn=cj.get("issn", cj.get("electronic_issn", "")),
                    oa_model=rec.get("oa_model", cj.get("oa_model", "")),
                    apc_estimate=rec.get("apc_estimate", cj.get("apc_display", "")),
                    indexing=rec.get("indexing", []),
                    fit=rec.get("fit", "Medium"),
                    score=candidates[cn - 1]["score"] if 1 <= cn <= len(candidates) else 0,
                    reasons=rec.get("reasons", []),
                    concern=rec.get("concern", ""),
                    subjects=cj.get("subject_categories", [])[:5],
                    impact_proxy=cj.get("impact_proxy", ""),
                    impact_factor=str(rec.get("impact_factor", cj.get("two_yr_mean_citedness", "?"))),
                    acceptance_rate=rec.get("acceptance_rate", "N/A"),
                    review_time=rec.get("review_time", "N/A"),
                    homepage=cj.get("homepage", ""),
                    cited_count=cr.get("cited_count", 0) if isinstance(cr, dict) else 0,
                    citation_boost=cr.get("citation_boost", 1.0) if isinstance(cr, dict) else 1.0,
                    topical_score=cr.get("topical_score", 0.0) if isinstance(cr, dict) else 0.0,
                    cluster_size=cr.get("cluster_size", 0) if isinstance(cr, dict) else 0,
                ))

            # Post-filter APC-free
            if constraints.apc_free_only:
                clean = []
                for r in recs:
                    al = (r.apc_estimate or "").lower()
                    ol = (r.oa_model or "").lower()
                    bad = False
                    if any(w in al for w in ["$", "usd", "eur", "gbp"]):
                        if re.search(r'\d{2,}', al):
                            if "free" not in al and "subscription" not in al: bad = True
                    if "gold" in ol and "free" not in al and "hybrid" not in ol: bad = True
                    if bad:
                        log.warning(f"APC post-filter: removing {r.journal_name}")
                    else:
                        clean.append(r)
                if clean:
                    recs = clean
                    for i, r in enumerate(recs, 1): r.rank = i
                else:
                    for r in recs:
                        r.concern = f"WARNING: may charge APC ({r.apc_estimate}). " + r.concern

            return recs, parsed.get("analysis_summary", "")

        except Exception as e:
            log.error(f"LLM re-ranking failed: {e}")
            try: log.error(f"Raw response: {response[:500]}")
            except: pass
            log.info("Falling back to heuristic")
            return self._heuristic_rank(candidates, num_results), self._generate_summary_heuristic(abstract, [])


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Clients
# ═══════════════════════════════════════════════════════════════════════════════

class AnthropicLLM:
    def __init__(self, api_key, model="claude-sonnet-4-20250514"):
        self.api_key = api_key; self.model = model
    def create_message(self, system, user_msg):
        from urllib.request import urlopen, Request
        payload = json.dumps({"model": self.model, "max_tokens": 4000, "system": system, "messages": [{"role": "user", "content": user_msg}]}).encode()
        req = Request("https://api.anthropic.com/v1/messages", data=payload, headers={"Content-Type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, method="POST")
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        return " ".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")


class OpenAILLM:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.api_key = api_key; self.model = model
    def create_message(self, system, user_msg):
        from urllib.request import urlopen, Request
        payload = json.dumps({"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 4000, "temperature": 0.3}).encode()
        req = Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, method="POST")
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"]


class GeminiLLM:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        self.api_key = api_key; self.model = model
    def create_message(self, system, user_msg):
        import time as _time
        from urllib.request import urlopen, Request
        from urllib.error import HTTPError
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 65536, "thinkingConfig": {"thinkingBudget": 1024}}
        }).encode()
        for attempt in range(5):
            try:
                req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(req, timeout=90) as resp:
                    result = json.loads(resp.read().decode())
                cands = result.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    return " ".join(p.get("text", "") for p in parts if "thought" not in p and p.get("text"))
                return ""
            except HTTPError as e:
                if e.code == 429:
                    w = (attempt + 1) * 15
                    log.warning(f"Gemini rate limited. Waiting {w}s... ({attempt+1}/5)")
                    _time.sleep(w)
                else: raise
        raise Exception("Gemini rate limit: failed after 5 retries.")
