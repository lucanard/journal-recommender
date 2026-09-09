"""Checks for the server-rendered journal directory.

Run with:  python test_explorer.py

The directory exists to be crawled, so these assert on the things a crawler
reads and a user clicks: one canonical URL per page, a title and description
that are not boilerplate, no broken internal link, no duplicate slug, and a
sitemap that is valid XML covering every page. They also pin the two failure
modes that would be invisible in a browser — a slug collision silently
shadowing a journal, and the explorer routes shadowing the JSON API.

Rendering runs against the real journal database in data/, so a data change
that breaks a page fails here rather than in production.
"""
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import explorer  # noqa: E402

BASE = "https://reviewpro.io"
FAILURES = []


def check(label, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + label + ("" if condition else f"   -> {detail}"))
    if not condition:
        FAILURES.append(label)


def load_journals():
    path = os.path.join(HERE, "data", "journal_database_final.json")
    with open(path, encoding="utf-8") as f:
        return {j["id"]: j for j in json.load(f) if not j.get("doaj_withdrawn")}


class WellFormed(HTMLParser):
    """Minimal nesting check — catches an unclosed tag from a broken template."""

    VOID = {"meta", "link", "br", "img", "input", "hr", "source", "area", "base"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
        elif self.stack[-1] != tag:
            self.errors.append(f"expected </{self.stack[-1]}>, got </{tag}>")
        else:
            self.stack.pop()

    @property
    def ok(self):
        return not self.errors and not self.stack


def test_slugs(index, journals):
    check("slugs: every titled journal is reachable",
          len(index.by_slug) == len(journals),
          f"{len(index.by_slug)} slugs for {len(journals)} journals")
    check("slugs: no journal shadows another",
          len(set(index.slug_of_id.values())) == len(index.by_slug))
    check("slugs: none are empty", all(s.strip() for s in index.by_slug))
    check("slugs: url-safe", all(re.fullmatch(r"[a-z0-9-]+", s) for s in index.by_slug),
          next((s for s in index.by_slug if not re.fullmatch(r"[a-z0-9-]+", s)), ""))

    # Non-Latin titles cannot be transliterated, but they still get a page.
    persian = [j for j in journals.values()
               if (j.get("title") or "").startswith("مجله دانشگاه علوم پزشکی خراسان")]
    if persian:
        slug = index.slug_of_id.get(persian[0]["id"])
        check("slugs: a non-Latin title falls back to its ISSN",
              bool(slug) and slug.startswith("journal-"), slug)


def test_publisher_cleaning():
    blob = ("Nat Mater. ISSN:1476-1122 (Print) ; 1476-4660 (Electronic) ; "
            "1476-1122 (Linking). London, UK : Nature Pub. Group")
    check("publisher: NLM catalogue blob reduced to the name",
          explorer.clean_publisher(blob) == "Nature Pub. Group",
          explorer.clean_publisher(blob))
    check("publisher: an ordinary name is left alone",
          explorer.clean_publisher("American Society of Hematology") == "American Society of Hematology")
    check("publisher: no separator falls back to the leading segment",
          explorer.clean_publisher("Foo Bar. ISSN:1234-5678 (Print).") == "Foo Bar",
          explorer.clean_publisher("Foo Bar. ISSN:1234-5678 (Print)."))
    check("publisher: empty stays empty", explorer.clean_publisher(None) == "")


def test_letters(index):
    expected = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["#"]
    check("letters: A-Z plus one catch-all bucket", index.letters == expected, index.letters)
    check("letters: nothing is lost between the buckets",
          sum(len(v) for v in index.by_letter.values()) == len(index.by_slug))


def test_subject_pages(index):
    check("subjects: none are thin",
          all(count >= explorer.SUBJECT_MIN_JOURNALS for _, _, count in index.subjects),
          min((c for _, _, c in index.subjects), default=0))
    check("subjects: slugs are unique",
          len({slug for _, slug, _ in index.subjects}) == len(index.subjects))
    check("subjects: an unknown slug renders nothing",
          explorer.render_subject(index, "not-a-subject", BASE) is None)


def test_every_page_renders(index):
    """Every journal page, not a sample: one bad record is one 500 in the index."""
    failures = []
    for slug in index.by_slug:
        try:
            page = explorer.render_journal(index, slug, BASE)
            if not page or "<h1>" not in page:
                failures.append((slug, "no heading"))
        except Exception as exc:  # noqa: BLE001 - the point is to catch anything
            failures.append((slug, repr(exc)))
    check(f"render: all {len(index.by_slug)} journal pages build", not failures, failures[:3])

    subject_failures = []
    for _, slug, _ in index.subjects:
        try:
            if not explorer.render_subject(index, slug, BASE):
                subject_failures.append(slug)
        except Exception as exc:  # noqa: BLE001
            subject_failures.append((slug, repr(exc)))
    check(f"render: all {len(index.subjects)} subject pages build", not subject_failures,
          subject_failures[:3])


def test_page_metadata(index):
    random.seed(11)
    sample = random.sample(list(index.by_slug), 250)
    missing_canonical, missing_desc, wrong_canonical, malformed, boilerplate = [], [], [], [], set()

    for slug in sample:
        page = explorer.render_journal(index, slug, BASE)
        canonical = re.search(r'<link rel="canonical" href="([^"]+)"', page)
        description = re.search(r'<meta name="description" content="([^"]*)"', page)
        if not canonical:
            missing_canonical.append(slug)
        elif canonical.group(1) != f"{BASE}/journal/{slug}":
            wrong_canonical.append((slug, canonical.group(1)))
        if not description or len(description.group(1)) < 30:
            missing_desc.append(slug)
        else:
            boilerplate.add(description.group(1))
        parser = WellFormed()
        parser.feed(page)
        if not parser.ok:
            malformed.append((slug, parser.errors[:2], parser.stack[-2:]))

    check("meta: every page has a canonical", not missing_canonical, missing_canonical[:3])
    check("meta: the canonical points at the page itself", not wrong_canonical, wrong_canonical[:3])
    check("meta: every page has a real description", not missing_desc, missing_desc[:3])
    check("meta: descriptions are not one shared boilerplate",
          len(boilerplate) > len(sample) * 0.8, f"{len(boilerplate)} distinct of {len(sample)}")
    check("meta: pages are well-formed", not malformed, malformed[:2])

    page = explorer.render_journal(index, sample[0], BASE)
    check("meta: schema.org data is present and parses",
          _json_ld_of(page) is not None and _json_ld_of(page).get("@type") == "Periodical")


def _json_ld_of(page):
    block = re.search(r'<script type="application/ld\+json">(.*?)</script>', page, re.S)
    if not block:
        return None
    try:
        return json.loads(block.group(1))
    except json.JSONDecodeError:
        return None


def test_internal_links(index):
    """A crawled link that 404s wastes crawl budget and looks like a dead site."""
    random.seed(12)
    broken = []
    for slug in random.sample(list(index.by_slug), 200):
        page = explorer.render_journal(index, slug, BASE)
        for href in re.findall(r'href="([^"]+)"', page):
            if href.startswith(f"{BASE}/journal/") and href.rsplit("/", 1)[1] not in index.by_slug:
                broken.append(href)
            elif href.startswith(f"{BASE}/journals/subject/") and \
                    href.rsplit("/", 1)[1] not in index.subject_of_slug:
                broken.append(href)
    check("links: no broken internal link on journal pages", not broken, broken[:3])

    directory = explorer.render_directory(index, BASE)
    subject_hrefs = re.findall(rf'{re.escape(BASE)}/journals/subject/([a-z0-9-]+)', directory)
    check("links: every subject linked from the hub exists",
          all(s in index.subject_of_slug for s in subject_hrefs))
    check("links: the hub reaches the recommender", f"{BASE}/#/dashboard" in directory)


def test_escaping(index):
    """Titles carry quotes and ampersands; none of it may break out of the markup."""
    hostile = {
        1: {"id": 1, "title": 'Journal of "Quotes" & <script>alert(1)</script>',
            "publisher": "A & B <b>Press</b>", "electronic_issn": "1234-5678",
            "aims_scope": 'Scope with <img src=x onerror="alert(1)"> and "quotes".',
            "subject_categories": ["Medicine"]},
    }
    small = explorer.ExplorerIndex(hostile)
    slug = list(small.by_slug)[0]
    page = explorer.render_journal(small, slug, BASE)
    check("escaping: no script tag survives from the data",
          "<script>alert(1)</script>" not in page)
    check("escaping: no attribute injection survives",
          'onerror="alert(1)"' not in page)
    check("escaping: the text is still shown, escaped",
          "&lt;script&gt;" in page and "&amp;" in page)
    ld = _json_ld_of(page)
    check("escaping: JSON-LD stays parseable with hostile input", ld is not None)


def test_sitemap_and_robots(index):
    xml = explorer.render_sitemap(index, BASE)
    root = ET.fromstring(xml)
    ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    locs = [u.find(f"{ns}loc").text for u in root]
    check("sitemap: valid XML", root.tag == f"{ns}urlset")
    check("sitemap: no duplicate URLs", len(locs) == len(set(locs)),
          f"{len(locs) - len(set(locs))} duplicates")
    check("sitemap: every journal page is listed",
          all(f"{BASE}/journal/{slug}" in set(locs) for slug in index.by_slug))
    check("sitemap: every subject page is listed",
          all(f"{BASE}/journals/subject/{slug}" in set(locs) for _, slug, _ in index.subjects))
    check("sitemap: under the 50,000-URL limit", len(locs) < 50000, len(locs))
    check("sitemap: every URL is absolute", all(l.startswith(BASE) for l in locs))

    robots = explorer.render_robots(BASE)
    check("robots: points at the sitemap", f"Sitemap: {BASE}/sitemap.xml" in robots)
    check("robots: does not block the directory", "Disallow: /journals" not in robots)


def test_routes_do_not_shadow_the_api():
    """The explorer lives alongside the JSON API; neither may swallow the other."""
    try:
        import app as app_module
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"SKIP  routes: {exc}")
        return

    app_module.store.journals = load_journals()
    app_module.store._loaded = True
    app_module._explorer_index = None
    app_module._sitemap_cache = {}
    client = TestClient(app_module.app)

    check("routes: JSON journal search still answers",
          client.get("/journals/search?q=blood").status_code == 200)
    check("routes: JSON journal detail still answers",
          client.get("/journals/1").status_code == 200)
    check("routes: JSON detail is still JSON",
          client.get("/journals/1").headers["content-type"].startswith("application/json"))

    directory = client.get("/journals")
    check("routes: the directory renders HTML", directory.status_code == 200
          and directory.headers["content-type"].startswith("text/html"))
    check("routes: an unknown journal is a 404, not a 500",
          client.get("/journal/no-such-journal-anywhere").status_code == 404)
    check("routes: an unknown subject is a 404",
          client.get("/journals/subject/no-such-subject").status_code == 404)
    check("routes: sitemap is served as XML",
          client.get("/sitemap.xml").headers["content-type"].startswith("application/xml"))
    check("routes: robots.txt is served", client.get("/robots.txt").status_code == 200)

    # Without PUBLIC_BASE_URL the origin has to come from the request, or every
    # canonical on a second domain points at the wrong site.
    os.environ.pop("PUBLIC_BASE_URL", None)
    app_module._sitemap_cache = {}
    page = client.get("/journal/nature-materials",
                      headers={"Host": "reviewpro.io", "X-Forwarded-Proto": "https"})
    check("routes: canonical follows the request origin",
          'href="https://reviewpro.io/journal/nature-materials"' in page.text,
          page.text[:200] if page.status_code != 200 else "canonical not found")

    os.environ["PUBLIC_BASE_URL"] = "https://configured.example"
    app_module._sitemap_cache = {}
    configured = client.get("/robots.txt", headers={"Host": "reviewpro.io"})
    check("routes: PUBLIC_BASE_URL overrides the request origin",
          "https://configured.example/sitemap.xml" in configured.text, configured.text)
    os.environ.pop("PUBLIC_BASE_URL", None)


def main():
    journals = load_journals()
    index = explorer.ExplorerIndex(journals)
    print(f"Indexed {len(index.by_slug)} journals, {len(index.subjects)} subject pages\n")

    test_slugs(index, journals)
    test_publisher_cleaning()
    test_letters(index)
    test_subject_pages(index)
    test_every_page_renders(index)
    test_page_metadata(index)
    test_internal_links(index)
    test_escaping(index)
    test_sitemap_and_robots(index)
    test_routes_do_not_shadow_the_api()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
