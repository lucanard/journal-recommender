"""Server-rendered journal directory — the crawlable half of the site.

The app itself is a hash-routed SPA: every screen lives behind `#/...`, which a
search engine sees as one single URL. Nothing in it can rank. This module serves
the same journal data as plain HTML on real paths, so the 9,246 records become
9,246 indexable pages plus a few hundred subject hubs:

    /journals                      directory, paginated and searchable
    /journals/subject/{slug}       every journal in one subject category
    /journal/{slug}                one journal
    /sitemap.xml, /robots.txt      so a crawler can find all of the above

No JavaScript is required to read any of it. Each page carries its own title,
meta description, canonical URL, Open Graph tags and schema.org JSON-LD, and
links onward to the recommender, which is where a visitor converts.

The index is built once from the in-memory journal store and cached; rendering
a page is string formatting over a dict lookup.
"""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import date, timezone
from typing import Optional

PAGE_SIZE = 50
# A subject only gets its own page when there is enough behind it to be worth
# crawling. Thin pages over a handful of journals invite a soft-404.
SUBJECT_MIN_JOURNALS = 20
# How many sibling journals to link from a detail page.
RELATED_COUNT = 8

BRAND = "PubFit"
NAVY = "#1E3A8A"
TEAL = "#14B8A6"


def slugify(text: str, fallback: str = "") -> str:
    """URL slug from a title. Returns `fallback` when nothing survives.

    Titles in the database run from "Blood advances" to
    "مجله دانشگاه علوم پزشکی خراسان شمالی"; transliteration would be guesswork,
    so a non-Latin title falls back to its ISSN instead of being dropped.
    """
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)[:80].strip("-")
    # Lowercase the fallback too: URLs are case-sensitive and an ISSN ends in an
    # uppercase X often enough (2476-695X) to produce links that 404.
    return slug or fallback.lower()


def clean_publisher(raw: str) -> str:
    """Pull a publisher name out of an NLM catalogue blob.

    440 records carry the full catalogue line in `publisher`, e.g.
    "Nat Mater. ISSN:1476-1122 (Print) ; ... London, UK : Nature Pub. Group".
    Rendered as-is it swamps the page title and the listing rows. The name sits
    after the final " : " (the place-of-publication separator); if there is no
    such separator, fall back to the segment before the ISSN block.
    """
    text = (raw or "").strip()
    if "ISSN:" not in text:
        return text
    tail = text.rsplit(" : ", 1)
    if len(tail) == 2 and tail[1].strip():
        return tail[1].strip().rstrip(".").strip()
    return text.split(". ISSN:", 1)[0].strip()


def _issn_of(journal: dict) -> str:
    return (journal.get("electronic_issn") or journal.get("print_issn") or "").strip()


class ExplorerIndex:
    """Slug ↔ journal lookups, plus the groupings the directory pages need."""

    def __init__(self, journals: dict):
        self.by_slug: dict[str, dict] = {}
        self.slug_of_id: dict[int, str] = {}
        self.ordered: list[dict] = []          # alphabetical, titled journals only
        self.by_letter: dict[str, list[dict]] = {}
        self.by_subject: dict[str, list[dict]] = {}
        self.subject_of_slug: dict[str, str] = {}
        self.subjects: list[tuple[str, str, int]] = []   # (name, slug, count)

        for journal in journals.values():
            title = (journal.get("title") or "").strip()
            if not title:
                continue
            issn = _issn_of(journal)
            base = slugify(title, fallback=f"journal-{issn}" if issn else "")
            if not base:
                continue
            # Five title slugs collide across the corpus ("cifra" alone covers
            # eleven journals). The ISSN disambiguates; without one, fall back to
            # the record id so a page is never silently shadowed.
            slug = base
            if slug in self.by_slug:
                slug = (f"{base}-{issn}" if issn else f"{base}-{journal.get('id')}").lower()
                if slug in self.by_slug:
                    slug = f"{base}-{journal.get('id')}"
            self.by_slug[slug] = journal
            self.slug_of_id[journal.get("id")] = slug

        self.ordered = sorted(self.by_slug.values(), key=lambda j: (j.get("title") or "").lower())
        # The unfiltered hub is the strongest page on the site, so it should link
        # to journals a reader recognises. Alphabetical order put titles starting
        # with punctuation and digits on page one; prominence puts the
        # best-documented, most-cited journals there instead. A-Z browsing keeps
        # the alphabetical ordering.
        self.by_prominence = sorted(
            self.by_slug.values(),
            key=lambda j: (-(j.get("h_index") or 0), -(j.get("works_count") or 0),
                           (j.get("title") or "").lower()),
        )

        for journal in self.ordered:
            # A-Z only. `str.isalpha()` is true for Cyrillic, Arabic and CJK too,
            # which produced 124 near-empty "letter" pages — crawl budget spent on
            # thin content. Everything outside A-Z lands in the "#" bucket.
            first = (journal.get("title") or "")[:1].upper()
            self.by_letter.setdefault(first if "A" <= first <= "Z" else "#", []).append(journal)
            for subject in journal.get("subject_categories") or []:
                self.by_subject.setdefault(subject, []).append(journal)
        for members in self.by_subject.values():
            members.sort(key=lambda j: (-(j.get("h_index") or 0), -(j.get("works_count") or 0),
                                        (j.get("title") or "").lower()))

        subject_slugs: dict[str, str] = {}
        for subject, members in self.by_subject.items():
            if len(members) < SUBJECT_MIN_JOURNALS:
                continue
            slug = slugify(subject)
            if not slug or slug in subject_slugs:
                continue
            subject_slugs[slug] = subject
            self.subject_of_slug[slug] = subject
        self.subjects = sorted(
            ((name, slug, len(self.by_subject[name])) for slug, name in subject_slugs.items()),
            key=lambda row: row[0].lower(),
        )

    @property
    def letters(self) -> list[str]:
        return sorted(k for k in self.by_letter if k != "#") + (["#"] if "#" in self.by_letter else [])

    def search(self, query: str, limit: int = 200) -> list[dict]:
        """Substring match over title, publisher and ISSN — same rule as the API."""
        q = (query or "").lower().strip()
        if not q:
            return []
        hits = []
        for journal in self.ordered:
            haystack = " ".join([
                journal.get("title") or "", journal.get("publisher") or "",
                journal.get("electronic_issn") or "", journal.get("print_issn") or "",
                journal.get("nlm_abbreviation") or "",
            ]).lower()
            if q in haystack:
                hits.append(journal)
                if len(hits) >= limit:
                    break
        return hits

    def related(self, journal: dict, count: int = RELATED_COUNT) -> list[dict]:
        """Journals sharing this one's narrowest subject — the useful internal links.

        Subjects are ordered broad-to-narrow inconsistently across records, so
        "narrowest" is approximated by the smallest bucket the journal sits in.
        """
        buckets = [
            self.by_subject[s] for s in (journal.get("subject_categories") or [])
            if s in self.by_subject and len(self.by_subject[s]) > 1
        ]
        if not buckets:
            return []
        pool = min(buckets, key=len)
        siblings = [j for j in pool if j.get("id") != journal.get("id")]
        # Prefer well-documented neighbours: a link to a near-empty record wastes
        # the crawl and the reader's click.
        siblings.sort(key=lambda j: (-(j.get("completeness_score") or 0), (j.get("title") or "").lower()))
        return siblings[:count]


# ── Rendering ────────────────────────────────────────────────────────────────

def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _truncate(text: str, limit: int = 155) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


CSS = """
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1f2937;background:#f9fafb}
a{color:%(navy)s}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
header.site{background:#fff;border-bottom:1px solid #e5e7eb}
header.site .wrap{display:flex;align-items:center;justify-content:space-between;height:64px;gap:16px}
.logo{display:flex;align-items:center;gap:8px;font-weight:700;font-size:20px;color:%(navy)s;text-decoration:none}
.logo span.mark{width:32px;height:32px;border-radius:8px;background:%(navy)s;color:#fff;display:grid;place-items:center;font-size:16px}
header.site nav a{margin-left:20px;font-size:14px;color:#4b5563;text-decoration:none}
header.site nav a:hover{color:%(navy)s}
.cta{display:inline-block;background:%(teal)s;color:#fff!important;padding:10px 18px;border-radius:8px;font-weight:600;text-decoration:none}
.cta:hover{background:#0F9B8E}
main{padding:32px 0 56px}
h1{font-size:32px;line-height:1.25;color:%(navy)s;margin:0 0 8px}
h2{font-size:20px;color:%(navy)s;margin:32px 0 12px}
.lede{color:#4b5563;margin:0 0 24px;max-width:70ch}
.crumbs{font-size:13px;color:#6b7280;margin:0 0 16px}
.crumbs a{color:#6b7280}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 20px}
.badge{font-size:12px;font-weight:700;padding:4px 10px;border-radius:999px;border:1px solid #e5e7eb;background:#fff;color:#374151}
.badge.q1{background:#d1fae5;color:#065f46;border-color:#a7f3d0}
.badge.q2{background:#dbeafe;color:#1e40af;border-color:#bfdbfe}
.badge.q3{background:#fef3c7;color:#92400e;border-color:#fde68a}
.badge.q4{background:#f3f4f6;color:#374151}
.badge.oa{background:%(teal)s1a;color:#0F9B8E;border-color:#99f6e4}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin:0 0 16px}
table.facts{width:100%%;border-collapse:collapse;font-size:15px}
table.facts th{text-align:left;font-weight:600;color:#6b7280;padding:8px 16px 8px 0;vertical-align:top;width:34%%;font-size:14px}
table.facts td{padding:8px 0;vertical-align:top;border-bottom:1px solid #f3f4f6}
table.facts tr:last-child td,table.facts tr:last-child th{border-bottom:0}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.metric{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px}
.metric .n{font-size:22px;font-weight:700;color:%(navy)s}
.metric .l{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}
ul.jlist{list-style:none;padding:0;margin:0}
ul.jlist li{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin:0 0 10px}
ul.jlist a.title{font-weight:600;font-size:17px;text-decoration:none}
ul.jlist .meta{font-size:13px;color:#6b7280;margin-top:2px}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 0}
.tags a,.tags span{font-size:12px;padding:3px 9px;border-radius:6px;background:#f3f4f6;color:#4b5563;text-decoration:none}
.tags a:hover{background:#e5e7eb}
.letters{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 20px}
.letters a{padding:5px 11px;border:1px solid #e5e7eb;border-radius:6px;background:#fff;text-decoration:none;font-size:14px;font-weight:600}
.letters a.on{background:%(navy)s;color:#fff;border-color:%(navy)s}
form.find{display:flex;gap:8px;margin:0 0 24px;max-width:520px}
form.find input{flex:1;padding:11px 14px;border:1px solid #d1d5db;border-radius:8px;font-size:15px}
form.find button{padding:11px 20px;border:0;border-radius:8px;background:%(navy)s;color:#fff;font-weight:600;cursor:pointer}
.pager{display:flex;gap:10px;align-items:center;margin:24px 0 0;font-size:14px}
.pager a{padding:8px 14px;border:1px solid #e5e7eb;border-radius:8px;background:#fff;text-decoration:none}
.convert{background:%(navy)s;color:#fff;border-radius:12px;padding:24px;margin:32px 0}
.convert h2{color:#fff;margin:0 0 8px}
.convert p{margin:0 0 16px;color:#c7d2fe}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px 24px}
.cols a{font-size:14px;text-decoration:none;padding:3px 0;display:block}
footer.site{background:%(navy)s;color:#cbd5e1;font-size:14px;padding:32px 0;margin-top:48px}
footer.site a{color:#fff}
.src{font-size:13px;color:#6b7280;margin-top:24px}
@media(max-width:640px){h1{font-size:26px}header.site nav a{margin-left:12px}table.facts th{width:45%%}}
""" % {"navy": NAVY, "teal": TEAL}


def _shell(*, title: str, description: str, canonical: str, body: str,
           base_url: str, head_extra: str = "") -> str:
    """One page, complete with the metadata a crawler reads."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_esc(title)}</title>
<meta name="description" content="{_esc(description)}"/>
<link rel="canonical" href="{_esc(canonical)}"/>
<meta property="og:type" content="website"/>
<meta property="og:title" content="{_esc(title)}"/>
<meta property="og:description" content="{_esc(description)}"/>
<meta property="og:url" content="{_esc(canonical)}"/>
<meta property="og:site_name" content="{BRAND}"/>
<meta name="twitter:card" content="summary"/>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%231E3A8A'/><text x='16' y='23' text-anchor='middle' fill='white' font-size='18' font-weight='bold' font-family='sans-serif'>P</text></svg>"/>
<style>{CSS}</style>
{head_extra}
</head>
<body>
<header class="site"><div class="wrap">
  <a class="logo" href="{base_url}/"><span class="mark">P</span>{BRAND}</a>
  <nav>
    <a href="{base_url}/journals">Journals</a>
    <a href="{base_url}/#/features">Features</a>
    <a href="{base_url}/#/pricing">Pricing</a>
    <a class="cta" href="{base_url}/#/dashboard">Match my paper</a>
  </nav>
</div></header>
<main><div class="wrap">
{body}
</div></main>
<footer class="site"><div class="wrap">
  <p><strong>{BRAND}</strong> — AI-powered journal matching for researchers.</p>
  <p><a href="{base_url}/journals">Journal directory</a> ·
     <a href="{base_url}/#/pricing">Pricing</a> ·
     <a href="{base_url}/#/privacy">Privacy</a> ·
     <a href="{base_url}/#/terms">Terms</a></p>
</div></footer>
</body>
</html>"""


def _convert_block(base_url: str, journal_title: Optional[str] = None) -> str:
    if journal_title:
        heading = f"Is your manuscript a fit for {_esc(journal_title)}?"
        sub = ("Paste your abstract and see how this journal ranks against 9,300+ others — "
               "with the reasoning behind each match.")
    else:
        heading = "Not sure which of these fits your paper?"
        sub = ("Paste your abstract and get a ranked shortlist across 9,300+ journals, "
               "with the reasoning behind every match.")
    return (f'<div class="convert"><h2>{heading}</h2><p>{sub}</p>'
            f'<a class="cta" href="{base_url}/#/dashboard">Find matching journals — free</a></div>')


def _quartile_class(impact_proxy: str) -> str:
    q = (impact_proxy or "").lower()
    for key in ("q1", "q2", "q3", "q4"):
        if key in q:
            return key
    return ""


def _badges(journal: dict) -> str:
    out = []
    proxy = journal.get("impact_proxy")
    if proxy:
        out.append(f'<span class="badge {_quartile_class(proxy)}">{_esc(proxy)}</span>')
    oa = journal.get("oa_model")
    if oa:
        out.append(f'<span class="badge oa">{_esc(oa)}</span>')
    if journal.get("indexed_pubmed"):
        out.append('<span class="badge">PubMed/MEDLINE</span>')
    if journal.get("in_doaj"):
        out.append('<span class="badge">DOAJ</span>')
    return f'<div class="badges">{"".join(out)}</div>' if out else ""


def _journal_line(journal: dict, index: ExplorerIndex, base_url: str) -> str:
    slug = index.slug_of_id.get(journal.get("id"), "")
    bits = [b for b in (clean_publisher(journal.get("publisher")), journal.get("impact_proxy"),
                        journal.get("oa_model")) if b]
    issn = _issn_of(journal)
    if issn:
        bits.append(f"ISSN {issn}")
    return (f'<li><a class="title" href="{base_url}/journal/{_esc(slug)}">{_esc(journal.get("title"))}</a>'
            f'<div class="meta">{_esc(" · ".join(bits))}</div></li>')


def _pager(base_path: str, page: int, total_pages: int, extra: str = "") -> str:
    if total_pages <= 1:
        return ""
    sep = "&" if extra else ""
    links = []
    if page > 1:
        prev_q = f"?{extra}{sep}page={page-1}" if page > 2 or extra else ""
        links.append(f'<a rel="prev" href="{base_path}{prev_q}">← Previous</a>')
    links.append(f"<span>Page {page} of {total_pages}</span>")
    if page < total_pages:
        links.append(f'<a rel="next" href="{base_path}?{extra}{sep}page={page+1}">Next →</a>')
    return f'<div class="pager">{"".join(links)}</div>'


def render_directory(index: ExplorerIndex, base_url: str, *, page: int = 1,
                     letter: str = "", query: str = "") -> str:
    """The hub: search, A–Z, subject facets and a paginated list."""
    if query:
        pool = index.search(query)
        heading = f'Journals matching “{_esc(query)}”'
        lede = f"{len(pool)} journal{'s' if len(pool) != 1 else ''} matched your search."
        title = f"Search: {query} — Journal Directory | {BRAND}"
        desc = f"Journals matching “{query}” — scope, open access model, APC, indexing and citation metrics."
        base_path, extra = f"{base_url}/journals", f"q={html.escape(query, quote=True).replace(' ', '+')}"
        canonical = f"{base_url}/journals"
    elif letter:
        pool = index.by_letter.get(letter, [])
        heading = f"Journals starting with “{_esc(letter)}”"
        lede = f"{len(pool)} journals indexed under {_esc(letter)}."
        title = f"Journals starting with {letter} — Directory | {BRAND}"
        desc = (f"Every journal in the {BRAND} directory starting with {letter}: scope, "
                f"open access model, APC, indexing and citation metrics.")
        base_path, extra = f"{base_url}/journals", f"letter={_esc(letter)}"
        canonical = f"{base_url}/journals?letter={_esc(letter)}"
    else:
        pool = index.by_prominence
        heading = "Journal Directory"
        lede = (f"Browse {len(pool):,} academic journals — scope, open access model, article "
                f"processing charges, indexing and citation metrics, from public sources.")
        title = f"Journal Directory — {len(pool):,} academic journals | {BRAND}"
        desc = (f"Browse {len(pool):,} academic journals by name or subject. Scope, open access "
                f"model, APC, PubMed and DOAJ indexing, quartile and citation metrics.")
        base_path, extra = f"{base_url}/journals", ""
        canonical = f"{base_url}/journals"

    total_pages = max(1, (len(pool) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    if page > 1:
        canonical = f"{canonical}{'&' if '?' in canonical else '?'}page={page}"
    window = pool[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    letters = "".join(
        f'<a class="{"on" if L == letter else ""}" href="{base_url}/journals?letter={_esc(L)}">{_esc(L)}</a>'
        for L in index.letters
    )
    subjects = "".join(
        f'<a href="{base_url}/journals/subject/{_esc(slug)}">{_esc(name)} ({count:,})</a>'
        for name, slug, count in index.subjects
    )
    listing = "".join(_journal_line(j, index, base_url) for j in window) or \
        '<li>No journals matched. Try a different spelling, or browse the A–Z above.</li>'

    body = f"""
<h1>{heading}</h1>
<p class="lede">{lede}</p>
<form class="find" action="{base_url}/journals" method="get">
  <input type="search" name="q" value="{_esc(query)}" placeholder="Journal name, publisher or ISSN" aria-label="Search journals"/>
  <button type="submit">Search</button>
</form>
<div class="letters">{letters}</div>
<ul class="jlist">{listing}</ul>
{_pager(base_path, page, total_pages, extra)}
{_convert_block(base_url)}
<h2>Browse by subject</h2>
<div class="cols">{subjects}</div>
<p class="src">Journal metadata is compiled from OpenAlex, DOAJ, Crossref and the NLM catalogue.
Figures reflect the most recent enrichment run and are provided for orientation, not as a
substitute for the publisher's own guidance.</p>
"""
    return _shell(title=title, description=desc, canonical=canonical, body=body, base_url=base_url)


def render_subject(index: ExplorerIndex, subject_slug: str, base_url: str, *, page: int = 1) -> Optional[str]:
    subject = index.subject_of_slug.get(subject_slug)
    if not subject:
        return None
    pool = index.by_subject.get(subject, [])
    total_pages = max(1, (len(pool) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    window = pool[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    oa_count = sum(1 for j in pool if j.get("in_doaj") or "full oa" in (j.get("oa_model") or "").lower())
    pubmed_count = sum(1 for j in pool if j.get("indexed_pubmed"))

    canonical = f"{base_url}/journals/subject/{_esc(subject_slug)}"
    if page > 1:
        canonical += f"?page={page}"
    title = f"{subject} journals — {len(pool):,} titles | {BRAND}"
    desc = (f"{len(pool):,} journals in {subject}, {oa_count:,} of them open access. "
            f"Compare scope, APC, indexing and citation metrics.")

    listing = "".join(_journal_line(j, index, base_url) for j in window)
    siblings = "".join(
        f'<a href="{base_url}/journals/subject/{_esc(s)}">{_esc(n)} ({c:,})</a>'
        for n, s, c in index.subjects if s != subject_slug
    )
    body = f"""
<p class="crumbs"><a href="{base_url}/journals">Journal directory</a> → {_esc(subject)}</p>
<h1>{_esc(subject)} journals</h1>
<p class="lede">{len(pool):,} journals classified under {_esc(subject)} — {oa_count:,} open access,
{pubmed_count:,} indexed in PubMed/MEDLINE.</p>
<ul class="jlist">{listing}</ul>
{_pager(f"{base_url}/journals/subject/{_esc(subject_slug)}", page, total_pages)}
{_convert_block(base_url)}
<h2>Other subjects</h2>
<div class="cols">{siblings}</div>
"""
    return _shell(title=title, description=desc, canonical=canonical, body=body, base_url=base_url)


def _facts_table(journal: dict, index: ExplorerIndex, base_url: str) -> str:
    rows: list[tuple[str, str]] = []

    def add(label, value):
        if value:
            rows.append((label, value))

    add("Publisher", _esc(clean_publisher(journal.get("publisher"))))
    issn_bits = []
    if journal.get("electronic_issn"):
        issn_bits.append(f'{_esc(journal["electronic_issn"])} (electronic)')
    if journal.get("print_issn"):
        issn_bits.append(f'{_esc(journal["print_issn"])} (print)')
    add("ISSN", "<br/>".join(issn_bits))
    add("Abbreviation", _esc(journal.get("nlm_abbreviation")))
    add("Access model", _esc(journal.get("oa_model")))

    apc = journal.get("apc_display")
    if journal.get("has_apc") is False and not apc:
        apc = "No article processing charge"
    add("Article processing charge", _esc(apc))

    licenses = journal.get("license_info") or []
    if licenses:
        parts = []
        for lic in licenses:
            name = _esc(lic.get("type"))
            url = lic.get("url")
            parts.append(f'<a href="{_esc(url)}" rel="nofollow noopener">{name}</a>' if url else name)
        add("Licence", ", ".join(p for p in parts if p))

    review = journal.get("review_process") or []
    add("Peer review", _esc(", ".join(review)) if isinstance(review, list) else _esc(review))

    weeks = journal.get("publication_time_weeks")
    if weeks:
        add("Typical time to publication", f"{_esc(weeks)} weeks")

    languages = journal.get("languages") or []
    add("Languages", _esc(", ".join(languages)))
    add("Country", _esc(journal.get("country_code")))

    indexed = []
    if journal.get("indexed_pubmed"):
        indexed.append("PubMed/MEDLINE")
    if journal.get("in_doaj"):
        indexed.append("DOAJ")
    add("Indexed in", _esc(", ".join(indexed)) or "Not recorded in our sources")

    homepage = journal.get("homepage")
    if homepage:
        add("Journal homepage", f'<a href="{_esc(homepage)}" rel="nofollow noopener">{_esc(homepage)}</a>')

    cells = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f'<table class="facts">{cells}</table>'


def _metrics_block(journal: dict) -> str:
    tiles = []

    def tile(number, label, hint=""):
        tiles.append(f'<div class="metric"><div class="n">{_esc(number)}</div>'
                     f'<div class="l" title="{_esc(hint)}">{_esc(label)}</div></div>')

    if journal.get("h_index") is not None:
        tile(f'{journal["h_index"]:,}', "h-index", "OpenAlex")
    citedness = journal.get("two_yr_mean_citedness")
    if citedness is not None:
        # Deliberately not called an Impact Factor: this is OpenAlex's 2-year
        # mean citedness and we hold no Clarivate JIF licence.
        tile(f"{citedness:.2f}", "2-yr citation rate", "Mean citations per article over 2 years (OpenAlex)")
    if journal.get("works_count"):
        tile(f'{journal["works_count"]:,}', "works indexed")
    if journal.get("recent_works_2yr"):
        tile(f'{journal["recent_works_2yr"]:,}', "published (2 yrs)")
    if journal.get("cited_by_count"):
        tile(f'{journal["cited_by_count"]:,}', "total citations")
    if not tiles:
        return ""
    return f'<h2>Metrics</h2><div class="metrics">{"".join(tiles)}</div>'


def _json_ld(journal: dict, canonical: str) -> str:
    import json as _json
    data = {
        "@context": "https://schema.org",
        "@type": "Periodical",
        "name": journal.get("title"),
        "url": canonical,
    }
    issn = [v for v in (journal.get("electronic_issn"), journal.get("print_issn")) if v]
    if issn:
        data["issn"] = issn
    if clean_publisher(journal.get("publisher")):
        data["publisher"] = {"@type": "Organization", "name": clean_publisher(journal["publisher"])}
    if journal.get("aims_scope"):
        data["description"] = _truncate(journal["aims_scope"], 300)
    if journal.get("languages"):
        data["inLanguage"] = journal["languages"]
    if journal.get("homepage"):
        data["sameAs"] = journal["homepage"]
    payload = _json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/ld+json">{payload}</script>'


def render_journal(index: ExplorerIndex, slug: str, base_url: str) -> Optional[str]:
    journal = index.by_slug.get(slug)
    if not journal:
        return None

    title_text = journal.get("title") or "Journal"
    canonical = f"{base_url}/journal/{slug}"
    scope = (journal.get("aims_scope") or "").strip()
    publisher = clean_publisher(journal.get("publisher"))

    # A search result shows roughly 60 characters of title. Keep the journal's
    # own name first and intact, cap the handful of very long ones, and name the
    # publisher only when there is room left for it.
    page_title = _truncate(title_text, 70)
    if publisher and len(page_title) + len(publisher) <= 48:
        page_title += f" ({publisher})"
    page_title += f" — scope, APC & indexing | {BRAND}"

    desc_source = scope or (
        f"{title_text}"
        + (f", published by {publisher}. " if publisher else ". ")
        + "Access model, article processing charge, indexing and citation metrics."
    )
    description = _truncate(desc_source, 155)

    subject_links = "".join(
        f'<a href="{base_url}/journals/subject/{_esc(slugify(s))}">{_esc(s)}</a>'
        if slugify(s) in index.subject_of_slug else f"<span>{_esc(s)}</span>"
        for s in (journal.get("subject_categories") or [])[:14]
    )
    topics = "".join(f"<span>{_esc(t)}</span>" for t in (journal.get("top_topics") or [])[:12])

    related = index.related(journal)
    related_html = ""
    if related:
        related_html = (
            f'<h2>Similar journals</h2><ul class="jlist">'
            + "".join(_journal_line(j, index, base_url) for j in related)
            + "</ul>"
        )

    scope_html = f'<h2>Aims and scope</h2><div class="card"><p>{_esc(scope)}</p></div>' if scope else ""
    subjects_html = f'<h2>Subject areas</h2><div class="tags">{subject_links}</div>' if subject_links else ""
    topics_html = f'<h2>Frequent topics</h2><div class="tags">{topics}</div>' if topics else ""

    body = f"""
<p class="crumbs"><a href="{base_url}/journals">Journal directory</a> → {_esc(title_text)}</p>
<h1>{_esc(title_text)}</h1>
<p class="lede">{_esc(publisher)}</p>
{_badges(journal)}
<h2>At a glance</h2>
<div class="card">{_facts_table(journal, index, base_url)}</div>
{scope_html}
{_metrics_block(journal)}
{subjects_html}
{topics_html}
{_convert_block(base_url, title_text)}
{related_html}
<p class="src">Compiled from OpenAlex, DOAJ, Crossref and the NLM catalogue. {BRAND} is independent
of this journal and its publisher. Always confirm charges, licensing and scope with the publisher
before submitting.</p>
"""
    return _shell(title=page_title, description=description, canonical=canonical,
                  body=body, base_url=base_url, head_extra=_json_ld(journal, canonical))


def render_sitemap(index: ExplorerIndex, base_url: str) -> str:
    """One sitemap covering the hub, every subject page and every journal.

    Well inside the 50,000-URL / 50 MB limit at ~10k URLs, so no index file.
    """
    today = date.today().isoformat()
    urls = [(f"{base_url}/", "1.0"), (f"{base_url}/journals", "0.9")]
    urls += [(f"{base_url}/journals?letter={letter}", "0.5") for letter in index.letters]
    urls += [(f"{base_url}/journals/subject/{slug}", "0.7") for _, slug, _ in index.subjects]
    urls += [(f"{base_url}/journal/{slug}", "0.6") for slug in index.by_slug]

    entries = "".join(
        f"<url><loc>{_esc(loc)}</loc><lastmod>{today}</lastmod><priority>{priority}</priority></url>"
        for loc, priority in urls
    )
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"{entries}</urlset>")


def render_robots(base_url: str) -> str:
    return (
        "User-agent: *\n"
        "Allow: /\n"
        # The SPA's API surface has nothing to index and answers POST only.
        "Disallow: /stripe-webhook\n"
        "Disallow: /stripe-debug\n"
        "Disallow: /docs\n"
        "Disallow: /redoc\n"
        f"\nSitemap: {base_url}/sitemap.xml\n"
    )
