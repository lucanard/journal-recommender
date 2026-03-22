"""
Recommendation Engine v1.1
===========================
Pipeline: Embed all sections → Vector search → Constraint filter → LLM re-rank with full context
"""

import json, logging, re, time
from typing import Optional
from dataclasses import dataclass, field, asdict

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
    # Manuscript sections (optional, improve matching quality)
    introduction_conclusion: str = ""
    methods: str = ""
    results_summary: str = ""


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


class RecommendationEngine:

    def __init__(self, vector_store, embedding_service, llm_client=None):
        self.store = vector_store
        self.embedder = embedding_service
        self.llm = llm_client

    def recommend(self, abstract, constraints, num_results=3, candidate_pool=0):
        timing = {}

        # Step 0: Embed (abstract + optional sections combined)
        t0 = time.time()
        query_text = self._build_query_text(abstract, constraints)
        query_embedding = self.embedder.embed_query(query_text)
        timing["embed_ms"] = int((time.time() - t0) * 1000)

        # Step 1: Vector search (all journals)
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

        relaxed = False
        if len(filtered) == 0 and len(raw_results) > 0:
            log.info("All filtered out — relaxing constraints")
            filtered = raw_results
            relaxed = True

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
            recs.append(Recommendation(
                rank=rank, journal_id=j.get("id", 0), journal_name=j.get("title", "?"), publisher=j.get("publisher", "?"),
                issn=j.get("issn", j.get("electronic_issn", "")), oa_model=j.get("oa_model", "?"),
                apc_estimate=j.get("apc_display", "?"), indexing=idx,
                fit="High" if score > 0.6 else "Medium", score=round(score, 4),
                reasons=[f"Similarity: {r['score']:.3f}", f"Subjects: {', '.join(j.get('subject_categories', [])[:3]) or 'N/A'}"],
                concern="Heuristic scoring — enable LLM for better results.",
                subjects=j.get("subject_categories", [])[:5], impact_proxy=j.get("impact_proxy", ""),
                impact_factor=str(j.get("two_yr_mean_citedness", "?")),
                acceptance_rate="N/A", review_time="N/A", homepage=j.get("homepage", ""),
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
