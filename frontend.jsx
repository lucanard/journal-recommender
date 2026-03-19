import { useState, useEffect, useRef } from "react";

// ─── Config ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

const ARTICLE_TYPES = [
  "Original Research", "Review Article", "Case Report",
  "Short Communication", "Systematic Review", "Meta-Analysis",
  "Letter/Correspondence",
];
const INDEXING_OPTIONS = ["PubMed/MEDLINE", "DOAJ", "Scopus", "Web of Science"];
const OA_OPTIONS = ["Any", "Open Access Only", "Hybrid"];
const IMPACT_OPTIONS = ["Any", "Q1 (High)", "Q1-Q2", "Q2-Q3", "Q3-Q4"];

const DISCIPLINES = [
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
];

// ─── Color system — warm research library aesthetic ──────────────────────────
const C = {
  bg: "#FAFAF7",
  surface: "#FFFFFF",
  surfaceAlt: "#F5F3EE",
  border: "#E8E4DB",
  borderFocus: "#B8956A",
  accent: "#8B6941",
  accentLight: "#C9A87440",
  accentText: "#6B4D2D",
  green: "#3D7A4A",
  greenBg: "#EDF5EF",
  yellow: "#A67B28",
  yellowBg: "#FBF5E6",
  red: "#B84233",
  redBg: "#FBEEEC",
  text: "#2C2417",
  textMid: "#5C5040",
  textMuted: "#938A79",
  textDim: "#B8B0A0",
};

const DISPLAY = "'Newsreader', 'Georgia', 'Times New Roman', serif";
const BODY = "'Source Sans 3', 'Source Sans Pro', 'Segoe UI', sans-serif";
const MONO = "'IBM Plex Mono', 'Menlo', monospace";

export default function JournalRecommender() {
  const [view, setView] = useState("input");
  // Core fields
  const [abstract, setAbstract] = useState("");
  const [keywords, setKeywords] = useState("");
  const [articleType, setArticleType] = useState("Original Research");
  const [discipline, setDiscipline] = useState("Any");
  // Additional manuscript sections
  const [introConclusion, setIntroConclusion] = useState("");
  const [methods, setMethods] = useState("");
  const [resultsSummary, setResultsSummary] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Filters
  const [indexing, setIndexing] = useState([]);
  const [oaPref, setOaPref] = useState("Any");
  const [apcFreeOnly, setApcFreeOnly] = useState(false);
  const [maxAPC, setMaxAPC] = useState("");
  const [minIF, setMinIF] = useState("");
  const [targetImpact, setTargetImpact] = useState("Any");
  const [numResults, setNumResults] = useState(3);
  // State
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState({});
  const [apiStatus, setApiStatus] = useState(null);
  const [loadPhase, setLoadPhase] = useState(0);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setApiStatus(d))
      .catch(() => setApiStatus(null));
  }, []);

  const toggleIdx = (opt) =>
    setIndexing(prev => prev.includes(opt) ? prev.filter(i => i !== opt) : [...prev, opt]);

  const filledSections = [introConclusion, methods, resultsSummary].filter(s => s.trim().length > 0).length;

  const handleSubmit = async () => {
    if (abstract.trim().length < 50) {
      setError("Abstract must be at least 50 characters.");
      return;
    }
    setError("");
    setView("loading");
    setLoadPhase(0);

    const phases = [500, 1500, 3000];
    phases.forEach((ms, i) => setTimeout(() => setLoadPhase(i + 1), ms));

    try {
      const body = {
        abstract: abstract.trim(),
        keywords: keywords.trim(),
        introduction_conclusion: introConclusion.trim(),
        methods: methods.trim(),
        results_summary: resultsSummary.trim(),
        article_type: articleType,
        discipline,
        indexing_required: indexing,
        oa_preference: oaPref,
        apc_free_only: apcFreeOnly,
        max_apc: maxAPC ? parseFloat(maxAPC) : null,
        min_impact_factor: minIF ? parseFloat(minIF) : null,
        target_impact: targetImpact !== "Any" ? targetImpact : null,
        num_results: numResults,
      };

      const resp = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `API error: ${resp.status}`);
      }

      const data = await resp.json();
      setResults(data);
      setView("results");
    } catch (err) {
      setError(err.message || "Failed to connect to the recommendation API.");
      setView("input");
    }
  };

  const sendFeedback = async (rec, type) => {
    setFeedback(prev => ({ ...prev, [rec.rank]: type }));
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recommendation_id: rec.rank,
          journal_name: rec.journal_name,
          feedback: type,
        }),
      });
    } catch {}
  };

  const handleReset = () => {
    setView("input");
    setResults(null);
    setFeedback({});
  };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text }}>
      <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Source+Sans+3:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />

      {/* ── Header ── */}
      <header style={{
        borderBottom: `1px solid ${C.border}`,
        background: C.surface,
        padding: "18px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "12px" }}>
          <h1 style={{
            fontFamily: DISPLAY, fontSize: "22px", fontWeight: 500,
            margin: 0, color: C.accentText, letterSpacing: "-0.5px",
          }}>
            Journal Recommender
          </h1>
          <span style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim }}>
            v1.1
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {apiStatus && (
            <span style={{
              fontFamily: MONO, fontSize: "11px",
              color: apiStatus.status === "ok" ? C.green : C.yellow,
              display: "flex", alignItems: "center", gap: "5px",
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: "50%",
                background: apiStatus.status === "ok" ? C.green : C.yellow,
              }}/>
              {apiStatus.vector_store?.total_journals?.toLocaleString() || "?"} journals
              {apiStatus.llm_enabled ? " · LLM" : ""}
            </span>
          )}
          {view === "results" && (
            <button onClick={handleReset}
              style={{
                background: "transparent", border: `1px solid ${C.border}`,
                borderRadius: "6px", padding: "6px 14px", cursor: "pointer",
                fontFamily: BODY, fontSize: "13px", color: C.textMid,
              }}>
              ← New search
            </button>
          )}
        </div>
      </header>

      <div style={{ maxWidth: 760, margin: "0 auto", padding: "32px 24px" }}>
        {/* ── Error ── */}
        {error && (
          <div style={{
            background: C.redBg, border: `1px solid ${C.red}30`, borderRadius: "8px",
            padding: "12px 16px", marginBottom: "20px", fontFamily: BODY, fontSize: "13px", color: C.red,
          }}>
            {error}
          </div>
        )}

        {/* ══ INPUT VIEW ══ */}
        {view === "input" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            {!apiStatus && (
              <div style={{
                background: C.yellowBg, border: `1px solid ${C.yellow}30`,
                borderRadius: "8px", padding: "14px 18px", marginBottom: "24px",
                fontFamily: BODY, fontSize: "13px", color: C.yellow, lineHeight: 1.6,
              }}>
                <strong>API not connected.</strong> Start the backend with{" "}
                <code style={{ fontFamily: MONO, fontSize: "12px" }}>python app.py --data-dir data</code>{" "}
                to enable semantic search.
              </div>
            )}

            {/* ── 01 Abstract ── */}
            <SectionLabel num="01" label="Abstract or Draft" />
            <textarea
              value={abstract} onChange={e => setAbstract(e.target.value)}
              placeholder="Paste your manuscript abstract here. The more detail, the better the match..."
              style={{
                width: "100%", minHeight: 180, background: C.surface,
                border: `1.5px solid ${C.border}`, borderRadius: "10px",
                fontFamily: BODY, fontSize: "14px", lineHeight: 1.75,
                color: C.text, padding: "16px 18px", resize: "vertical",
                outline: "none", transition: "border-color 0.2s", boxSizing: "border-box",
              }}
              onFocus={e => e.target.style.borderColor = C.borderFocus}
              onBlur={e => e.target.style.borderColor = C.border}
            />
            <div style={{ fontFamily: MONO, fontSize: "11px", color: abstract.length >= 50 ? C.green : C.textDim, marginTop: "6px", textAlign: "right" }}>
              {abstract.length} characters {abstract.length >= 50 ? "✓" : "· minimum 50"}
            </div>

            {/* ── 02 Keywords ── */}
            <div style={{ marginTop: "24px" }}>
              <SectionLabel num="02" label="Keywords" optional />
              <input
                type="text" value={keywords} onChange={e => setKeywords(e.target.value)}
                placeholder="e.g., machine learning, drug discovery, cohort study"
                style={{
                  width: "100%", background: C.surface, border: `1.5px solid ${C.border}`,
                  borderRadius: "8px", fontFamily: BODY, fontSize: "14px",
                  color: C.text, padding: "11px 16px", outline: "none",
                  transition: "border-color 0.2s", boxSizing: "border-box",
                }}
                onFocus={e => e.target.style.borderColor = C.borderFocus}
                onBlur={e => e.target.style.borderColor = C.border}
              />
            </div>

            {/* ── 03 Additional manuscript sections (collapsible) ── */}
            <div style={{ marginTop: "24px" }}>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  padding: 0, display: "flex", alignItems: "center", gap: "8px",
                  width: "100%",
                }}
              >
                <span style={{ fontFamily: MONO, fontSize: "12px", color: C.accent, fontWeight: 500 }}>03</span>
                <span style={{ fontFamily: DISPLAY, fontSize: "17px", fontWeight: 500, color: C.text }}>
                  Strengthen your match
                </span>
                <span style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim }}>optional</span>
                {filledSections > 0 && (
                  <span style={{
                    fontFamily: MONO, fontSize: "10px", color: C.green,
                    background: C.greenBg, padding: "2px 8px", borderRadius: "10px",
                    border: `1px solid ${C.green}30`,
                  }}>
                    {filledSections} added
                  </span>
                )}
                <span style={{
                  marginLeft: "auto", fontFamily: MONO, fontSize: "13px", color: C.textDim,
                  transform: showAdvanced ? "rotate(90deg)" : "rotate(0deg)",
                  transition: "transform 0.2s",
                  display: "inline-block",
                }}>
                  ›
                </span>
              </button>

              {showAdvanced && (
                <div style={{
                  marginTop: "14px", padding: "20px",
                  background: C.surfaceAlt, borderRadius: "12px",
                  border: `1px solid ${C.border}`,
                  display: "flex", flexDirection: "column", gap: "18px",
                  animation: "fadeUp 0.25s ease",
                }}>
                  <div style={{ fontFamily: BODY, fontSize: "12.5px", color: C.textMuted, lineHeight: 1.6 }}>
                    Providing additional sections helps the engine match methodology-specific journals
                    and better assess audience fit. Each field is independent — add whichever you have.
                  </div>

                  <div>
                    <label style={{ fontFamily: BODY, fontSize: "12px", fontWeight: 600, color: C.textMuted, textTransform: "uppercase", letterSpacing: "0.4px" }}>
                      Introduction / Conclusion
                    </label>
                    <div style={{ fontFamily: BODY, fontSize: "11.5px", color: C.textDim, marginBottom: "6px", marginTop: "2px" }}>
                      Last paragraph of your Introduction and/or your Conclusion
                    </div>
                    <textarea
                      value={introConclusion} onChange={e => setIntroConclusion(e.target.value)}
                      placeholder="Paste the final paragraph of your Introduction and/or Conclusion..."
                      style={{
                        width: "100%", minHeight: 90, background: C.surface,
                        border: `1.5px solid ${C.border}`, borderRadius: "8px",
                        fontFamily: BODY, fontSize: "13px", lineHeight: 1.7,
                        color: C.text, padding: "12px 14px", resize: "vertical",
                        outline: "none", transition: "border-color 0.2s", boxSizing: "border-box",
                      }}
                      onFocus={e => e.target.style.borderColor = C.borderFocus}
                      onBlur={e => e.target.style.borderColor = C.border}
                    />
                  </div>

                  <div>
                    <label style={{ fontFamily: BODY, fontSize: "12px", fontWeight: 600, color: C.textMuted, textTransform: "uppercase", letterSpacing: "0.4px" }}>
                      Materials & Methods
                    </label>
                    <div style={{ fontFamily: BODY, fontSize: "11.5px", color: C.textDim, marginBottom: "6px", marginTop: "2px" }}>
                      Helps match methodology-focused journals
                    </div>
                    <textarea
                      value={methods} onChange={e => setMethods(e.target.value)}
                      placeholder="Paste your Materials & Methods section..."
                      style={{
                        width: "100%", minHeight: 90, background: C.surface,
                        border: `1.5px solid ${C.border}`, borderRadius: "8px",
                        fontFamily: BODY, fontSize: "13px", lineHeight: 1.7,
                        color: C.text, padding: "12px 14px", resize: "vertical",
                        outline: "none", transition: "border-color 0.2s", boxSizing: "border-box",
                      }}
                      onFocus={e => e.target.style.borderColor = C.borderFocus}
                      onBlur={e => e.target.style.borderColor = C.border}
                    />
                  </div>

                  <div>
                    <label style={{ fontFamily: BODY, fontSize: "12px", fontWeight: 600, color: C.textMuted, textTransform: "uppercase", letterSpacing: "0.4px" }}>
                      Key Results / Findings
                    </label>
                    <div style={{ fontFamily: BODY, fontSize: "11.5px", color: C.textDim, marginBottom: "6px", marginTop: "2px" }}>
                      Helps match journals that publish similar types of evidence
                    </div>
                    <textarea
                      value={resultsSummary} onChange={e => setResultsSummary(e.target.value)}
                      placeholder="Paste key results or findings..."
                      style={{
                        width: "100%", minHeight: 90, background: C.surface,
                        border: `1.5px solid ${C.border}`, borderRadius: "8px",
                        fontFamily: BODY, fontSize: "13px", lineHeight: 1.7,
                        color: C.text, padding: "12px 14px", resize: "vertical",
                        outline: "none", transition: "border-color 0.2s", boxSizing: "border-box",
                      }}
                      onFocus={e => e.target.style.borderColor = C.borderFocus}
                      onBlur={e => e.target.style.borderColor = C.border}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* ── 04 Constraints ── */}
            <div style={{ marginTop: "28px" }}>
              <SectionLabel num="04" label="Constraints & Preferences" />
              <div style={{
                background: C.surfaceAlt, borderRadius: "12px", padding: "20px 22px",
                display: "grid", gridTemplateColumns: "1fr 1fr", gap: "18px",
                border: `1px solid ${C.border}`,
              }}>
                <FilterField label="Article Type">
                  <StyledSelect value={articleType} onChange={e => setArticleType(e.target.value)} options={ARTICLE_TYPES} />
                </FilterField>
                <FilterField label="Discipline">
                  <StyledSelect value={discipline} onChange={e => setDiscipline(e.target.value)} options={DISCIPLINES} />
                </FilterField>
                <FilterField label="Open Access">
                  <StyledSelect value={oaPref} onChange={e => setOaPref(e.target.value)} options={OA_OPTIONS} />
                </FilterField>
                <FilterField label="Target Impact">
                  <StyledSelect value={targetImpact} onChange={e => setTargetImpact(e.target.value)} options={IMPACT_OPTIONS} />
                </FilterField>
                <FilterField label="Maximum APC (USD)">
                  <input type="number" value={maxAPC} onChange={e => setMaxAPC(e.target.value)}
                    placeholder="No limit" min="0"
                    style={{
                      width: "100%", background: C.surface, border: `1.5px solid ${C.border}`,
                      borderRadius: "6px", fontFamily: BODY, fontSize: "13px",
                      color: C.text, padding: "8px 12px", outline: "none", boxSizing: "border-box",
                    }} />
                </FilterField>
                <FilterField label="Minimum Impact Factor">
                  <input type="number" value={minIF} onChange={e => setMinIF(e.target.value)}
                    placeholder="No minimum" min="0" step="0.1"
                    style={{
                      width: "100%", background: C.surface, border: `1.5px solid ${C.border}`,
                      borderRadius: "6px", fontFamily: BODY, fontSize: "13px",
                      color: C.text, padding: "8px 12px", outline: "none", boxSizing: "border-box",
                    }} />
                </FilterField>

                {/* APC-free toggle */}
                <div style={{ gridColumn: "1 / -1" }}>
                  <button
                    onClick={() => setApcFreeOnly(!apcFreeOnly)}
                    style={{
                      display: "flex", alignItems: "center", gap: "10px",
                      background: apcFreeOnly ? C.greenBg : "transparent",
                      border: `1.5px solid ${apcFreeOnly ? C.green : C.border}`,
                      borderRadius: "8px", padding: "10px 16px", cursor: "pointer",
                      width: "100%", transition: "all 0.15s",
                    }}
                  >
                    <span style={{
                      width: 18, height: 18, borderRadius: "4px",
                      border: `2px solid ${apcFreeOnly ? C.green : C.textDim}`,
                      background: apcFreeOnly ? C.green : "transparent",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      transition: "all 0.15s", flexShrink: 0,
                    }}>
                      {apcFreeOnly && <span style={{ color: "#fff", fontSize: "12px", fontWeight: 700, lineHeight: 1 }}>✓</span>}
                    </span>
                    <div style={{ textAlign: "left" }}>
                      <div style={{ fontFamily: BODY, fontSize: "13px", fontWeight: 600, color: apcFreeOnly ? C.green : C.textMid }}>
                        Free to publish only
                      </div>
                      <div style={{ fontFamily: BODY, fontSize: "11.5px", color: C.textDim, marginTop: "1px" }}>
                        Excludes journals with mandatory APCs. Hybrid journals are still included (publish free via subscription route).
                      </div>
                    </div>
                  </button>
                </div>

                {/* Indexing */}
                <div style={{ gridColumn: "1 / -1" }}>
                  <FilterField label="Required Indexing">
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                      {INDEXING_OPTIONS.map(opt => (
                        <button key={opt} onClick={() => toggleIdx(opt)}
                          style={{
                            background: indexing.includes(opt) ? C.accentLight : C.surface,
                            border: `1.5px solid ${indexing.includes(opt) ? C.accent : C.border}`,
                            color: indexing.includes(opt) ? C.accentText : C.textMuted,
                            padding: "5px 14px", borderRadius: "20px", cursor: "pointer",
                            fontFamily: BODY, fontSize: "13px", fontWeight: indexing.includes(opt) ? 600 : 400,
                            transition: "all 0.15s",
                          }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  </FilterField>
                </div>

                {/* Number of results */}
                <div style={{ gridColumn: "1 / -1" }}>
                  <FilterField label={`Number of results: ${numResults}`}>
                    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <span style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim }}>1</span>
                      <input
                        type="range" min={1} max={10} value={numResults}
                        onChange={e => setNumResults(parseInt(e.target.value))}
                        style={{
                          flex: 1, height: "4px", appearance: "none", WebkitAppearance: "none",
                          background: `linear-gradient(to right, ${C.accent} 0%, ${C.accent} ${(numResults - 1) / 9 * 100}%, ${C.border} ${(numResults - 1) / 9 * 100}%, ${C.border} 100%)`,
                          borderRadius: "2px", outline: "none", cursor: "pointer",
                        }}
                      />
                      <span style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim }}>10</span>
                    </div>
                  </FilterField>
                </div>
              </div>
            </div>

            {/* ── Submit ── */}
            <button onClick={handleSubmit}
              style={{
                width: "100%", marginTop: "28px", padding: "15px",
                background: C.accent, border: "none", borderRadius: "10px",
                color: "#FFFCF7", fontFamily: DISPLAY, fontSize: "16px", fontWeight: 500,
                cursor: "pointer", letterSpacing: "0.2px",
                transition: "background 0.2s, transform 0.1s",
              }}
              onMouseEnter={e => e.target.style.background = C.accentText}
              onMouseLeave={e => e.target.style.background = C.accent}
              onMouseDown={e => e.target.style.transform = "scale(0.99)"}
              onMouseUp={e => e.target.style.transform = "scale(1)"}>
              Find Matching Journals
            </button>
            <p style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim, textAlign: "center", marginTop: "10px" }}>
              Advisory tool · Results require independent verification
            </p>
          </div>
        )}

        {/* ══ LOADING VIEW ══ */}
        {view === "loading" && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 350, gap: "28px" }}>
            <div style={{ position: "relative", width: 56, height: 56 }}>
              <div style={{
                width: 56, height: 56, borderRadius: "50%",
                border: `3px solid ${C.border}`, borderTopColor: C.accent,
                animation: "spin 0.9s linear infinite",
              }} />
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: DISPLAY, fontSize: "18px", color: C.text, marginBottom: "12px" }}>
                Analyzing your manuscript
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "center" }}>
                {["Embedding abstract", "Searching journal database", "Filtering by constraints", "Ranking & generating explanations"].map((phase, i) => (
                  <div key={i} style={{
                    fontFamily: BODY, fontSize: "13px",
                    color: loadPhase > i ? C.green : loadPhase === i ? C.accent : C.textDim,
                    display: "flex", alignItems: "center", gap: "8px",
                    transition: "color 0.3s",
                  }}>
                    <span style={{ fontFamily: MONO, fontSize: "11px" }}>
                      {loadPhase > i ? "✓" : loadPhase === i ? "›" : "·"}
                    </span>
                    {phase}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══ RESULTS VIEW ══ */}
        {view === "results" && results && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            {/* Timing bar */}
            {results.timing && (
              <div style={{
                display: "flex", gap: "16px", marginBottom: "20px",
                fontFamily: MONO, fontSize: "11px", color: C.textDim,
                flexWrap: "wrap",
              }}>
                {results.timing.embed_ms != null && <span>embed: {results.timing.embed_ms}ms</span>}
                {results.timing.search_ms != null && <span>search: {results.timing.search_ms}ms</span>}
                {results.timing.filter_ms != null && <span>filter: {results.timing.filter_ms}ms</span>}
                {results.timing.rerank_ms != null && <span>rerank: {results.timing.rerank_ms}ms</span>}
                <span style={{ marginLeft: "auto" }}>
                  {results.candidates_searched} → {results.candidates_after_filter} → {results.recommendations?.length} shown
                </span>
              </div>
            )}

            {/* Constraint relaxation warning */}
            {results.constraints_relaxed && (
              <div style={{
                background: C.yellowBg, border: `1px solid ${C.yellow}30`,
                borderRadius: "8px", padding: "12px 16px", marginBottom: "16px",
                fontFamily: BODY, fontSize: "13px", color: C.yellow, lineHeight: 1.6,
              }}>
                <strong>Filters relaxed:</strong> No journals matched all your constraints.
                Showing best matches without filters. Try loosening some requirements.
              </div>
            )}

            {/* Summary */}
            {results.analysis_summary && (
              <div style={{
                background: C.surfaceAlt, border: `1px solid ${C.border}`,
                borderRadius: "10px", padding: "18px 22px", marginBottom: "24px",
                fontFamily: BODY, fontSize: "14px", lineHeight: 1.7, color: C.textMid,
              }}>
                <span style={{ fontFamily: MONO, fontSize: "11px", color: C.accent, fontWeight: 500 }}>
                  ANALYSIS ·{" "}
                </span>
                {results.analysis_summary}
              </div>
            )}

            {/* Journal cards */}
            {results.recommendations?.map((rec, idx) => (
              <JournalCard key={idx} rec={rec} fb={feedback[rec.rank]} onFb={type => sendFeedback(rec, type)} />
            ))}

            {(!results.recommendations || results.recommendations.length === 0) && (
              <div style={{
                textAlign: "center", padding: "48px 24px",
                fontFamily: BODY, fontSize: "15px", color: C.textMuted,
              }}>
                No journals matched your constraints. Try relaxing some filters.
              </div>
            )}

            {/* Disclaimer */}
            <div style={{
              marginTop: "28px", padding: "14px 18px",
              background: C.yellowBg, border: `1px solid ${C.yellow}25`,
              borderRadius: "8px", fontFamily: BODY, fontSize: "12px",
              color: C.yellow, lineHeight: 1.65,
            }}>
              <strong>Disclaimer:</strong> This is an advisory tool. Recommendations are AI-generated
              and should be verified independently. Always confirm scope, APC, and submission guidelines
              with the journal directly.
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
        ::selection { background: ${C.accentLight}; }
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none; appearance: none;
          width: 16px; height: 16px; border-radius: 50%;
          background: ${C.accent}; cursor: pointer; border: 2px solid ${C.surface};
          box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        }
        input[type="range"]::-moz-range-thumb {
          width: 16px; height: 16px; border-radius: 50%;
          background: ${C.accent}; cursor: pointer; border: 2px solid ${C.surface};
          box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        }
      `}</style>
    </div>
  );
}


// ─── Sub-components ──────────────────────────────────────────────────────────

function SectionLabel({ num, label, optional }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "10px" }}>
      <span style={{ fontFamily: MONO, fontSize: "12px", color: C.accent, fontWeight: 500 }}>{num}</span>
      <span style={{ fontFamily: DISPLAY, fontSize: "17px", fontWeight: 500, color: C.text }}>{label}</span>
      {optional && <span style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim }}>optional</span>}
    </div>
  );
}

function FilterField({ label, children }) {
  return (
    <div>
      <div style={{ fontFamily: BODY, fontSize: "12px", fontWeight: 600, color: C.textMuted, marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.4px" }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function StyledSelect({ value, onChange, options }) {
  return (
    <select value={value} onChange={onChange} style={{
      width: "100%", background: C.surface, border: `1.5px solid ${C.border}`,
      borderRadius: "6px", fontFamily: BODY, fontSize: "13px",
      color: C.text, padding: "8px 12px", outline: "none", cursor: "pointer",
      boxSizing: "border-box",
    }}>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function JournalCard({ rec, fb, onFb }) {
  const fitMap = { "High": { c: C.green, bg: C.greenBg }, "Medium": { c: C.yellow, bg: C.yellowBg }, "Low": { c: C.red, bg: C.redBg } };
  const fit = fitMap[rec.fit] || fitMap["Medium"];

  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: "12px", padding: "22px 24px", marginBottom: "16px",
      transition: "box-shadow 0.2s",
    }}
    onMouseEnter={e => e.currentTarget.style.boxShadow = "0 2px 12px rgba(0,0,0,0.06)"}
    onMouseLeave={e => e.currentTarget.style.boxShadow = "none"}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "14px", gap: "12px" }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "10px", flexWrap: "wrap" }}>
            <span style={{ fontFamily: MONO, fontSize: "13px", fontWeight: 500, color: C.textDim }}>
              #{rec.rank}
            </span>
            <span style={{ fontFamily: DISPLAY, fontSize: "18px", fontWeight: 500, color: C.text, letterSpacing: "-0.3px" }}>
              {rec.journal_name}
            </span>
          </div>
          <div style={{ fontFamily: BODY, fontSize: "13px", color: C.textMuted, marginTop: "3px" }}>
            {rec.publisher}
            {rec.homepage && (
              <a href={rec.homepage} target="_blank" rel="noopener noreferrer"
                style={{ marginLeft: "8px", color: C.accent, fontSize: "12px", textDecoration: "none" }}>
                Visit →
              </a>
            )}
          </div>
        </div>
        <span style={{
          background: fit.bg, color: fit.c, padding: "4px 12px",
          borderRadius: "20px", fontFamily: MONO, fontSize: "12px", fontWeight: 500,
          border: `1px solid ${fit.c}30`, whiteSpace: "nowrap",
        }}>
          {rec.fit} Fit
        </span>
      </div>

      {/* Badges */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "7px", marginBottom: "16px" }}>
        <Badge label={rec.oa_model} accent />
        {rec.apc_estimate && <Badge label={rec.apc_estimate === "Free" || rec.apc_estimate === "N/A" ? "No APC" : `APC: ${rec.apc_estimate}`} />}
        {rec.impact_factor && rec.impact_factor !== "?" && rec.impact_factor !== "Unknown" && (
          <Badge label={`IF: ${rec.impact_factor}`} />
        )}
        {rec.indexing?.map(ix => <Badge key={ix} label={ix} green />)}
        {rec.impact_proxy && rec.impact_proxy !== "Unknown" && <Badge label={rec.impact_proxy} />}
        {rec.acceptance_rate && rec.acceptance_rate !== "N/A" && rec.acceptance_rate !== "Not publicly available" && (
          <Badge label={`Accept: ${rec.acceptance_rate}`} />
        )}
        {rec.review_time && rec.review_time !== "N/A" && rec.review_time !== "Not publicly available" && (
          <Badge label={`Review: ${rec.review_time}`} />
        )}
        {rec.score > 0 && <Badge label={`Score: ${rec.score.toFixed(3)}`} mono />}
      </div>

      {/* Reasons */}
      <div style={{ marginBottom: "14px" }}>
        <div style={{ fontFamily: MONO, fontSize: "11px", color: C.accent, fontWeight: 500, marginBottom: "8px", letterSpacing: "0.3px" }}>
          WHY IT MATCHES
        </div>
        {rec.reasons?.map((r, i) => (
          <div key={i} style={{
            fontFamily: BODY, fontSize: "13.5px", color: C.textMid,
            lineHeight: 1.65, paddingLeft: "18px", position: "relative", marginBottom: "5px",
          }}>
            <span style={{ position: "absolute", left: 0, color: C.green, fontSize: "14px", fontWeight: 600 }}>✓</span>
            {r}
          </div>
        ))}
      </div>

      {/* Concern */}
      {rec.concern && (
        <div style={{
          fontFamily: BODY, fontSize: "13px", color: C.yellow,
          background: C.yellowBg, padding: "10px 14px", borderRadius: "8px",
          borderLeft: `3px solid ${C.yellow}60`, marginBottom: "14px", lineHeight: 1.6,
        }}>
          <span style={{ fontWeight: 600 }}>Note: </span>{rec.concern}
        </div>
      )}

      {/* Subjects */}
      {rec.subjects && rec.subjects.length > 0 && (
        <div style={{ marginBottom: "14px" }}>
          <div style={{ fontFamily: MONO, fontSize: "10px", color: C.textDim, marginBottom: "5px", letterSpacing: "0.3px" }}>SUBJECTS</div>
          <div style={{ fontFamily: BODY, fontSize: "12px", color: C.textMuted, lineHeight: 1.6 }}>
            {rec.subjects.join(" · ")}
          </div>
        </div>
      )}

      {/* Feedback */}
      <div style={{
        display: "flex", alignItems: "center", gap: "10px",
        borderTop: `1px solid ${C.border}`, paddingTop: "12px",
      }}>
        <span style={{ fontFamily: MONO, fontSize: "11px", color: C.textDim }}>Helpful?</span>
        <FbButton active={fb === "up"} onClick={() => onFb("up")}>👍</FbButton>
        <FbButton active={fb === "down"} onClick={() => onFb("down")}>👎</FbButton>
        {fb && <span style={{ fontFamily: BODY, fontSize: "12px", color: C.green }}>Recorded — thank you</span>}
      </div>
    </div>
  );
}

function Badge({ label, accent, green, mono }) {
  const bg = accent ? C.accentLight : green ? C.greenBg : C.surfaceAlt;
  const color = accent ? C.accentText : green ? C.green : C.textMuted;
  return (
    <span style={{
      background: bg, color, padding: "3px 10px", borderRadius: "4px",
      fontFamily: mono ? MONO : BODY,
      fontSize: "11.5px", fontWeight: 500, border: `1px solid ${color}20`,
    }}>
      {label}
    </span>
  );
}

function FbButton({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      background: active ? C.accentLight : "transparent",
      border: `1.5px solid ${active ? C.accent : C.border}`,
      borderRadius: "6px", padding: "4px 10px", cursor: "pointer",
      fontSize: "14px", transition: "all 0.15s",
    }}>
      {children}
    </button>
  );
}
