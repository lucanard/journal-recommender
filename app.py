"""
Journal Recommendation API v1.1
================================
Usage: python app.py --data-dir data --llm gemini --llm-key YOUR_KEY
"""

import os, sys, json, logging, argparse, re
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator
    from typing import Optional
    import uvicorn
except ImportError:
    print("pip install fastapi uvicorn pydantic"); sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("pip install numpy"); sys.exit(1)

from vector_store import VectorStore, EmbeddingService
from recommender import RecommendationEngine, UserConstraints, AnthropicLLM, OpenAILLM, GeminiLLM

try:
    from report_generator import generate_report, DOCX_AVAILABLE
except ImportError:
    DOCX_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("journal-api")

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
EMBEDDINGS_FILE = DATA_DIR / "journal_embeddings.npz"
JOURNALS_FILE = DATA_DIR / "journal_database_final.json"
JOURNALS_JSONL = DATA_DIR / "journal_embeddings_input.jsonl"
JOURNALS_BASE = DATA_DIR / "journal_database.json"
META_FILE = DATA_DIR / "embedding_metadata.json"

DISCIPLINES = [
    "Any",
    "Agricultural Sciences",
    "Biochemistry & Molecular Biology",
    "Biomedical Engineering",
    "Chemistry",
    "Clinical Medicine",
    "Computer Science & AI",
    "Earth & Environmental Sciences",
    "Ecology & Evolution",
    "Economics & Business",
    "Education",
    "Engineering",
    "Epidemiology & Public Health",
    "Food Science & Nutrition",
    "Genetics & Genomics",
    "Humanities & Social Sciences",
    "Immunology",
    "Law",
    "Materials Science",
    "Mathematics & Statistics",
    "Microbiology",
    "Neuroscience",
    "Nursing & Health Professions",
    "Pharmacology & Toxicology",
    "Physics & Astronomy",
    "Psychology & Psychiatry",
    "Veterinary Science",
    "Other",
]


def _clean_text(v):
    """Remove line breaks, tabs, and extra spaces from pasted text."""
    if isinstance(v, str):
        v = v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
        v = re.sub(r"\s+", " ", v).strip()
    return v


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    # ─── Manuscript content ───
    abstract: str = Field(..., min_length=50, description="Research abstract (required, min 50 chars). Line breaks cleaned automatically.")
    keywords: str = Field("", description="Optional comma-separated keywords.")
    introduction_conclusion: str = Field("", description="Optional: paste the last paragraph of your Introduction and/or your Conclusion. Helps match the contribution and audience.")
    methods: str = Field("", description="Optional: paste your Materials & Methods section. Helps match methodology-specific journals.")
    results_summary: str = Field("", description="Optional: paste key results or findings. Helps match journals that publish similar types of evidence.")

    # ─── Manuscript metadata ───
    article_type: str = Field("Original Research", description="Original Research | Review Article | Case Report | Short Communication | Systematic Review | Meta-Analysis | Letter/Correspondence")
    discipline: str = Field("Any", description=f"Research discipline. Options: {', '.join(DISCIPLINES)}")

    # ─── Journal filters ───
    indexing_required: list[str] = Field(default_factory=list, description="Required indexes: PubMed/MEDLINE, DOAJ, Scopus, Web of Science. Empty [] = no filter.")
    oa_preference: str = Field("Any", description="Any | Open Access Only | Hybrid")
    apc_free_only: bool = Field(False, description="true = only free-to-publish journals. Hybrid OK via subscription route.")
    max_apc: float | None = Field(None, description="Max APC in USD. null = no limit.")
    min_impact_factor: float | None = Field(None, description="Minimum Impact Factor (e.g. 2.0, 5.0). null = no filter.")
    target_impact: str | None = Field(None, description="Q1 (High) | Q1-Q2 | Q2-Q3 | Q3-Q4. null = no filter.")
    num_results: int = Field(3, ge=1, le=10, description="Number of recommendations (1-10).")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "abstract": "Paste your abstract here. Must be at least 50 characters.",
                "keywords": "",
                "introduction_conclusion": "",
                "methods": "",
                "results_summary": "",
                "article_type": "Original Research",
                "discipline": "Any",
                "indexing_required": [],
                "oa_preference": "Any",
                "apc_free_only": False,
                "max_apc": None,
                "min_impact_factor": None,
                "target_impact": None,
                "num_results": 3
            }]
        }
    }

    @field_validator("abstract", "introduction_conclusion", "methods", "results_summary", "keywords", mode="before")
    @classmethod
    def clean_text_fields(cls, v):
        return _clean_text(v)


class JournalRecommendation(BaseModel):
    rank: int
    journal_id: int = 0
    journal_name: str
    publisher: str
    issn: str
    oa_model: str
    apc_estimate: str
    impact_factor: Optional[str] = ""
    acceptance_rate: Optional[str] = ""
    review_time: Optional[str] = ""
    indexing: list[str]
    fit: str
    score: float
    reasons: list[str]
    concern: str
    subjects: list[str] = []
    impact_proxy: Optional[str] = ""
    homepage: Optional[str] = ""


class RecommendResponse(BaseModel):
    recommendations: list[JournalRecommendation]
    analysis_summary: str
    timing: dict
    candidates_searched: int
    candidates_after_filter: int
    constraints_relaxed: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: dict
    llm_enabled: bool


class FeedbackRequest(BaseModel):
    recommendation_id: int
    journal_name: str
    feedback: str
    selected: bool = False
    comment: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Journal Recommendation API",
    description="""Intelligent journal matching for researchers.

## Quick start
1. Click **POST /recommend** → **Try it out**
2. Replace the abstract with your own
3. Optionally paste your Introduction/Conclusion, Methods, and Results
4. Set discipline and filters as needed
5. Click **Execute** → scroll down for results

## Manuscript inputs
| Field | Purpose | Required? |
|-------|---------|-----------|
| **abstract** | Overall scope matching | Yes |
| **keywords** | Specific term matching | No |
| **introduction_conclusion** | Contribution & audience matching | No |
| **methods** | Methodology-specific journals | No |
| **results_summary** | Evidence type matching | No |
| **discipline** | Narrows to your research field | No |

## Each result includes
**oa_model** (Hybrid journals explain both routes) · **apc_estimate** (2024-2025 rates) · **impact_factor** · **acceptance_rate** · **review_time** · **reasons** · **concern**
""",
    version="1.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

store = VectorStore()
embedder = None
engine = None
feedback_log = []


# ═══════════════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════════════

def initialize(args):
    global store, embedder, engine

    journals_path = None
    for candidate in [JOURNALS_FILE, JOURNALS_JSONL, JOURNALS_BASE]:
        if candidate.exists():
            journals_path = str(candidate)
            break
    if not journals_path:
        log.error(f"No journal data found in {DATA_DIR}"); sys.exit(1)

    log.info(f"Using journal data: {journals_path}")

    if EMBEDDINGS_FILE.exists():
        meta_path = str(META_FILE) if META_FILE.exists() else None
        store.load(str(EMBEDDINGS_FILE), journals_path, meta_path)
    else:
        log.warning("No embeddings — DEGRADED MODE")
        if journals_path.endswith(".jsonl"):
            with open(journals_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        j = json.loads(line); store.journals[j["id"]] = j
        else:
            with open(journals_path, encoding="utf-8") as f:
                for j in json.load(f):
                    if not j.get("doaj_withdrawn"): store.journals[j["id"]] = j
        store._loaded = True

    emb_provider = args.embedding_provider or os.environ.get("EMBEDDING_PROVIDER", "local")
    emb_key = args.embedding_key or os.environ.get("EMBEDDING_API_KEY", "")
    emb_model = store.model or {"openai": "text-embedding-3-small", "cohere": "embed-english-v3.0", "voyage": "voyage-3-lite", "gemini": "gemini-embedding-001", "local": "all-MiniLM-L6-v2"}.get(emb_provider, "all-MiniLM-L6-v2")
    if emb_provider != "local" and not emb_key:
        emb_key = os.environ.get(f"{emb_provider.upper()}_API_KEY", "")
    embedder = EmbeddingService(provider=emb_provider, model=emb_model, api_key=emb_key)
    log.info(f"Embedding: {emb_provider} ({emb_model})")

    llm_provider = args.llm or os.environ.get("LLM_PROVIDER", "none")
    llm_key = args.llm_key or os.environ.get("LLM_API_KEY", "")
    llm_client = None
    if llm_provider == "anthropic":
        llm_key = llm_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if llm_key: llm_client = AnthropicLLM(api_key=llm_key); log.info("LLM: Anthropic Claude")
    elif llm_provider == "openai":
        llm_key = llm_key or os.environ.get("OPENAI_API_KEY", "")
        if llm_key: llm_client = OpenAILLM(api_key=llm_key); log.info("LLM: OpenAI")
    elif llm_provider == "gemini":
        llm_key = llm_key or os.environ.get("GEMINI_API_KEY", "")
        if llm_key: llm_client = GeminiLLM(api_key=llm_key); log.info("LLM: Google Gemini")
    if not llm_client and llm_provider != "none":
        log.warning(f"{llm_provider} key not set — LLM disabled")
    elif llm_provider == "none":
        log.info("LLM: disabled")

    engine = RecommendationEngine(store, embedder, llm_client)
    log.info("Engine initialized")


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health_check():
    """System status."""
    return HealthResponse(status="ok" if store.is_loaded else "degraded", version="1.1.0", vector_store=store.get_stats(), llm_enabled=engine.llm is not None if engine else False)


@app.post("/recommend",
    response_model=RecommendResponse,
    openapi_extra={
        "requestBody": {
            "content": {"application/json": {
                "schema": RecommendRequest.model_json_schema(),
                "example": {
                    "abstract": "Paste your abstract here. Must be at least 50 characters.",
                    "keywords": "",
                    "introduction_conclusion": "",
                    "methods": "",
                    "results_summary": "",
                    "article_type": "Original Research",
                    "discipline": "Any",
                    "indexing_required": [],
                    "oa_preference": "Any",
                    "apc_free_only": False,
                    "max_apc": None,
                    "min_impact_factor": None,
                    "target_impact": None,
                    "num_results": 3
                }
            }},
            "required": True
        }
    }
)
async def recommend(raw_request: Request):
    """Get journal recommendations for your manuscript."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Read raw body and fix line breaks
    body = await raw_request.body()
    text = body.decode("utf-8")
    fixed = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    try:
        data = json.loads(fixed)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")

    try:
        request = RecommendRequest(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    # Clean placeholders
    PH = {"string", ""}
    cleaned_indexing = [idx for idx in request.indexing_required if idx.lower() not in PH]
    cleaned_oa = request.oa_preference if request.oa_preference not in PH else "Any"
    cleaned_impact = request.target_impact
    if cleaned_impact and cleaned_impact.lower() in PH: cleaned_impact = None
    cleaned_apc = request.max_apc
    if cleaned_apc is not None and cleaned_apc <= 0: cleaned_apc = None
    cleaned_min_if = request.min_impact_factor
    if cleaned_min_if is not None and cleaned_min_if <= 0: cleaned_min_if = None
    cleaned_discipline = request.discipline if request.discipline not in PH and request.discipline != "Any" else ""

    constraints = UserConstraints(
        article_type=request.article_type,
        discipline=cleaned_discipline,
        indexing_required=cleaned_indexing,
        oa_preference=cleaned_oa,
        max_apc=cleaned_apc,
        target_impact=cleaned_impact,
        keywords=request.keywords,
        apc_free_only=request.apc_free_only,
        min_impact_factor=cleaned_min_if,
        introduction_conclusion=request.introduction_conclusion,
        methods=request.methods,
        results_summary=request.results_summary,
    )

    try:
        result = engine.recommend(abstract=request.abstract, constraints=constraints, num_results=request.num_results)
    except Exception as e:
        log.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
    return result


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Submit feedback."""
    feedback_log.append({"recommendation_id": request.recommendation_id, "journal_name": request.journal_name, "feedback": request.feedback, "selected": request.selected, "comment": request.comment})
    return {"status": "recorded", "total_feedback": len(feedback_log)}


class ReportRequest(BaseModel):
    results: dict = Field(..., description="The full recommendation response object")
    abstract: str = Field(..., description="The original abstract")
    constraints: dict = Field(default_factory=dict, description="Constraints used in the search")


@app.post("/report")
def generate_report_endpoint(request: ReportRequest):
    """Generate a downloadable Word document summarizing the results."""
    if not DOCX_AVAILABLE:
        raise HTTPException(status_code=501, detail="Report generation requires python-docx. Install with: pip install python-docx")
    try:
        buffer = generate_report(
            results=request.results,
            abstract=request.abstract,
            constraints=request.constraints,
        )
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=journal_recommendations.docx"},
        )
    except Exception as e:
        log.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


class TailorRequest(BaseModel):
    abstract: str = Field(..., min_length=50, description="The manuscript abstract")
    keywords: str = Field("", description="Comma-separated keywords")
    introduction_conclusion: str = Field("", description="Intro/conclusion text")
    methods: str = Field("", description="Methods section text")
    results_summary: str = Field("", description="Key results text")
    article_type: str = Field("Original Research", description="Type of manuscript")
    journal_id: int = Field(..., description="Journal ID from the database")
    journal_name: str = Field("", description="Journal name (fallback if ID not found)")

    @field_validator("abstract", "introduction_conclusion", "methods", "results_summary", "keywords", mode="before")
    @classmethod
    def clean_text_fields(cls, v):
        return _clean_text(v)


@app.post("/tailor")
async def tailor_manuscript(raw_request: Request):
    """Get AI suggestions on how to tailor a manuscript for a specific journal."""
    if not engine or not engine.llm:
        raise HTTPException(status_code=503, detail="LLM not available. Tailoring requires an LLM provider.")

    body = await raw_request.body()
    text = body.decode("utf-8")
    fixed = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    try:
        data = json.loads(fixed)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")

    try:
        request = TailorRequest(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    # Get journal data
    journal = store.get_journal(request.journal_id)
    if not journal:
        raise HTTPException(status_code=404, detail=f"Journal ID {request.journal_id} not found")

    # Build journal profile
    journal_profile = {
        "title": journal.get("title", ""),
        "publisher": journal.get("publisher", ""),
        "aims_scope": (journal.get("aims_scope", "") or "")[:2000],
        "aims_scope_extended": (journal.get("aims_scope_extended", "") or "")[:1000],
        "subjects": journal.get("subject_categories", [])[:10],
        "topics": journal.get("top_topics", [])[:10],
        "editorial_keywords": journal.get("editorial_keywords", [])[:20],
        "research_focus_subfields": journal.get("research_focus_subfields", [])[:8],
        "article_type_distribution": journal.get("article_type_distribution", {}),
        "oa_model": journal.get("oa_model", ""),
        "impact_proxy": journal.get("impact_proxy", ""),
        "impact_factor": journal.get("two_yr_mean_citedness", "Unknown"),
        "indexed_pubmed": journal.get("indexed_pubmed", False),
    }

    # Build manuscript context
    manuscript = f"ABSTRACT:\n{request.abstract}"
    if request.keywords:
        manuscript += f"\n\nKEYWORDS: {request.keywords}"
    if request.introduction_conclusion:
        manuscript += f"\n\nINTRODUCTION / CONCLUSION:\n{request.introduction_conclusion[:2000]}"
    if request.methods:
        manuscript += f"\n\nMATERIALS & METHODS:\n{request.methods[:2000]}"
    if request.results_summary:
        manuscript += f"\n\nKEY RESULTS:\n{request.results_summary[:2000]}"

    system_prompt = """You are an expert academic publishing advisor. A researcher has selected a target journal and wants to know how to tailor their manuscript for the best chance of acceptance.

Analyze the manuscript content against the journal's aims, scope, audience, subject areas, and — critically — the journal's EDITORIAL KEYWORDS. Editorial keywords are extracted from the journal's recent publications and show exactly what topics the journal is actively publishing. Use them to give specific, actionable advice.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "journal_name": "...",
  "overall_fit_assessment": "A 2-3 sentence honest assessment of how well the manuscript fits this journal and what the main gap is, if any.",
  "framing": {
    "current": "How the manuscript currently frames its contribution (1-2 sentences)",
    "suggested": "How to reframe it for this journal's audience (2-3 sentences)",
    "opening_angle": "A suggested angle for the introduction's opening paragraph"
  },
  "title_suggestions": [
    "A suggested title variation that better fits this journal's style",
    "Another alternative title"
  ],
  "keywords_to_add": ["keyword1", "keyword2", "keyword3"],
  "keywords_to_avoid": ["keyword1", "keyword2"],
  "structure": {
    "recommended_sections": ["Introduction", "..."],
    "section_notes": "Any notes on section emphasis, length, or ordering for this journal",
    "estimated_word_count": "Recommended word count range for this article type at this journal"
  },
  "methodology_emphasis": {
    "highlight": ["What aspects of methods to emphasize for this journal"],
    "downplay": ["What aspects are less relevant to this journal's audience"]
  },
  "results_presentation": {
    "suggestions": ["How to present results for this journal's audience"],
    "figures_tables": "Guidance on what types of figures/tables this journal prefers"
  },
  "discussion_angles": [
    "Key point to address in the discussion for this journal's audience",
    "Another important angle"
  ],
  "terminology": {
    "use": ["Term or phrase that aligns with this journal's language"],
    "avoid": ["Term or phrase that doesn't fit this journal's audience"]
  },
  "reviewer_concerns": [
    "A likely concern a reviewer at this journal would raise",
    "Another potential concern and how to preemptively address it"
  ],
  "references_strategy": "What types of references to prioritize (e.g., cite papers from this journal, emphasize certain methodologies)",
  "cover_letter_tips": [
    "Key point to make in the cover letter for this specific journal",
    "Another tip"
  ]
}"""

    user_msg = f"""MANUSCRIPT:
{manuscript}

Article type: {request.article_type}

TARGET JOURNAL:
{json.dumps(journal_profile, indent=2)}

Provide specific, actionable tailoring advice for this exact manuscript-journal combination. Reference the journal's actual scope, subjects, and editorial keywords. The editorial_keywords field shows what the journal actually publishes based on analysis of recent articles — use these to guide terminology, framing, and emphasis. Don't give generic advice — every suggestion should be specific to THIS journal."""

    try:
        import time as _time
        t0 = _time.time()
        response = engine.llm.create_message(system_prompt, user_msg)
        elapsed_ms = int((_time.time() - t0) * 1000)

        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        if not cleaned.startswith('{'):
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match: cleaned = match.group(0)

        parsed = json.loads(cleaned)
        parsed["timing_ms"] = elapsed_ms
        parsed["journal_id"] = request.journal_id
        return parsed

    except json.JSONDecodeError as e:
        log.error(f"Tailor LLM response not valid JSON: {e}")
        log.error(f"Raw response: {response[:500] if response else 'empty'}")
        raise HTTPException(status_code=500, detail="AI response could not be parsed. Please try again.")
    except Exception as e:
        log.error(f"Tailoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tailoring failed: {str(e)}")


@app.get("/journals/search")
def search_journals(q: str, limit: int = 10):
    """Search journals by name."""
    if not store.journals: raise HTTPException(status_code=503, detail="No journals loaded")
    q_lower = q.lower()
    results = []
    for j in store.journals.values():
        if q_lower in j.get("title", "").lower():
            results.append({"id": j.get("id"), "title": j.get("title"), "publisher": j.get("publisher"), "issn": j.get("issn", j.get("electronic_issn", "")), "indexed_pubmed": j.get("indexed_pubmed", False), "in_doaj": j.get("in_doaj", False)})
            if len(results) >= limit: break
    return {"results": results, "total": len(results)}


@app.get("/journals/{journal_id}")
def get_journal(journal_id: int):
    """Get full journal details."""
    journal = store.get_journal(journal_id)
    if not journal: raise HTTPException(status_code=404, detail="Journal not found")
    return journal


@app.get("/stats")
def get_stats():
    """Database statistics."""
    journals = list(store.journals.values())
    return {"total_journals": len(journals), "indexed_pubmed": sum(1 for j in journals if j.get("indexed_pubmed")), "in_doaj": sum(1 for j in journals if j.get("in_doaj")), "has_scope": sum(1 for j in journals if j.get("aims_scope")), "has_apc_info": sum(1 for j in journals if j.get("apc_display")), "vector_store": store.get_stats(), "feedback_collected": len(feedback_log)}


@app.get("/disciplines")
def get_disciplines():
    """List available discipline options."""
    return {"disciplines": DISCIPLINES}


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the frontend UI."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found. Place index.html next to app.py.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Journal Recommendation API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--embedding-provider", choices=["openai", "cohere", "local", "voyage", "gemini"])
    p.add_argument("--embedding-key")
    p.add_argument("--llm", choices=["anthropic", "openai", "gemini", "none"], default="none")
    p.add_argument("--llm-key")
    p.add_argument("--data-dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        EMBEDDINGS_FILE = DATA_DIR / "journal_embeddings.npz"
        JOURNALS_FILE = DATA_DIR / "journal_database_final.json"
        JOURNALS_JSONL = DATA_DIR / "journal_embeddings_input.jsonl"
        JOURNALS_BASE = DATA_DIR / "journal_database.json"
        META_FILE = DATA_DIR / "embedding_metadata.json"

    print(r"""
       ╔═══════════════════════════════════════════════╗
       ║   AI Journal Recommendation API  v1.1         ║
       ╚═══════════════════════════════════════════════╝
    """)
    initialize(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
