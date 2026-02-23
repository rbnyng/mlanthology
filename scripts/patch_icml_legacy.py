#!/usr/bin/env python3
"""Patch ICML legacy records with data scraped from icml.cc conference pages.

For years 2004-2011 the icml.cc site hosts proceedings pages with inline
abstracts and/or PDF links.  This script:

  1. Fetches each conference proceedings page
  2. Parses out title → {abstract, pdf_url} mappings via regex
  3. Matches to existing legacy records by normalised title
  4. Fills in missing abstract and pdf_url fields (never overwrites)
  5. Optionally derives pdf_url from ACM DOI for remaining gaps
  6. Optionally scrapes 2003 abstracts from the AAAI Library (sequentially,
     since concurrent access triggers rate limiting)
  7. Optionally enriches 2005/2006 abstracts from OpenAlex (batch DOI lookup)
  8. Writes back to data/legacy/icml-legacy.jsonl.gz atomically

For 2004 and 2007 the abstracts live on individual sub-pages that are
fetched concurrently.  For 2005 only PDF links are on the index page
(no abstract sub-pages exist).  For 2006, icml.cc has PDF links only
(pointing to a now-dead Oregon State mirror); abstracts come from OpenAlex.

Run once (all years + ACM DOI pdf derivation + AAAI 2003 abstracts + OpenAlex):
    python scripts/patch_icml_legacy.py

Single year, dry-run to see what would change:
    python scripts/patch_icml_legacy.py --year 2009 --dry-run

Skip the ACM DOI → pdf_url derivation step:
    python scripts/patch_icml_legacy.py --no-doi-pdf

Skip the AAAI 2003 abstract scraping step:
    python scripts/patch_icml_legacy.py --no-aaai

Skip the OpenAlex 2005/2006 abstract enrichment step:
    python scripts/patch_icml_legacy.py --no-openalex
"""

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy, normalize_title
from adapters.http import fetch_with_retry
from adapters.openalex import fetch_abstracts_by_doi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conference page definitions
# ---------------------------------------------------------------------------

CONF_PAGES: dict[str, dict] = {
    "2004": {
        "url": "https://icml.cc/Conferences/2004/proceedings.html",
        "base": "https://icml.cc/Conferences/2004/",
    },
    "2005": {
        "url": "https://icml.cc/Conferences/2005/proceedings.html",
        "base": "https://icml.cc/Conferences/2005/",
    },
    "2007": {
        "url": "https://icml.cc/Conferences/2007/proceedings.html",
        "base": "https://icml.cc/Conferences/2007/",
    },
    "2008": {
        "url": "https://icml.cc/Conferences/2008/abstracts.shtml.html",
        "base": "https://icml.cc/Conferences/2008/",
    },
    "2009": {
        "url": "https://icml.cc/Conferences/2009/abstracts.html",
        "base": "https://icml.cc/Conferences/2009/",
    },
    "2010": {
        "url": "https://icml.cc/Conferences/2010/abstracts.html",
        "base": "https://icml.cc/Conferences/2010/",
    },
    "2011": {
        "url": "https://icml.cc/Conferences/2011/papers.php.html",
        "base": "https://icml.cc/2011/",
    },
}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _strip_tags(s: str) -> str:
    """Remove all HTML tags and decode entities."""
    return unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def _clean(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Year-specific parsers
# Each returns a list of dicts with keys:
#   title, paper_id, abstract, pdf_url
# ---------------------------------------------------------------------------

def parse_2004_index(html: str, base: str) -> list[dict]:
    """Parse 2004 proceedings index.

    The index is a table of papers, each row group following the pattern:
      <tr class="proc_2004_title"><td ...>TITLE</td></tr>
      <tr><td ...><a href="proceedings/abstracts/{id}.htm">[Abstract]</a>
                  <a href="proceedings/papers/{id}.{pdf|ps}">[Paper]</a></td></tr>
      <tr class="proc_2004_authors">...</tr>

    Returns entries with abstract_url set but abstract empty (fetched separately).
    """
    entries = []
    pattern = re.compile(
        r'class="proc_2004_title"[^>]*>\s*<td[^>]*>\s*(.*?)\s*</td>'   # 1: title HTML
        r'.*?'
        r'<a href="(proceedings/abstracts/(\d+)\.htm)">[^<]*</a>'        # 2: abs path, 3: id
        r'(?:\s*<a href="(proceedings/papers/\d+\.[a-z]+)">[^<]*</a>)?', # 4: pdf path (optional)
        re.DOTALL,
    )
    for m in pattern.finditer(html):
        title = _clean(_strip_tags(m.group(1)))
        abs_path = m.group(2)
        paper_id = m.group(3)
        pdf_path = m.group(4) or ""
        entries.append({
            "title": title,
            "paper_id": paper_id,
            "abstract_url": base + abs_path,
            "pdf_url": (base + pdf_path) if pdf_path else "",
            "abstract": "",
        })
    return entries


def _fetch_2004_abstract(abstract_url: str) -> str:
    """Fetch a 2004 per-paper abstract page.

    Page structure (minimal HTML, no doctype):
      <table><tr><th>TITLE</th></tr>
              <tr><td>AUTHOR - <i>AFF</i><br>...</td></tr>
              <tr><td>ABSTRACT TEXT</td></tr></table>
    """
    try:
        resp = fetch_with_retry(abstract_url, return_none_on_404=True)
        if resp is None:
            return ""
        # Take content of last <td> (the abstract)
        tds = re.findall(r"<td>(.*?)</td>", resp.text, re.DOTALL)
        if tds:
            return _clean(_strip_tags(tds[-1]))
    except Exception as e:
        logger.warning(f"Failed to fetch 2004 abstract {abstract_url}: {e}")
    return ""


def parse_2004(html: str, base: str) -> list[dict]:
    """Full 2004 scrape: index page + concurrent abstract sub-page fetches."""
    entries = parse_2004_index(html, base)
    logger.info(f"    Fetching {len(entries)} abstract sub-pages in parallel…")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_2004_abstract, e["abstract_url"]): i
                   for i, e in enumerate(entries) if e.get("abstract_url")}
        for future in as_completed(futures):
            idx = futures[future]
            entries[idx]["abstract"] = future.result()
    return entries


def parse_2005(html: str, base: str) -> list[dict]:
    """Parse 2005 proceedings index (PDF links only, no abstracts).

    Per-paper structure:
      <tr class="proc_2005_link">
        <td><a href="proceedings/papers/{NNN}_{Word}_{Author}.pdf">TITLE</a></td>
      </tr>
      <tr class="proc_2005_authors"><td>AUTHOR1, AUTHOR2</td></tr>
    """
    entries = []
    pattern = re.compile(
        r'class="proc_2005_link"[^>]*>\s*<td>'
        r'<a href="(proceedings/papers/[^"]+\.pdf)">(.*?)</a>',
        re.DOTALL,
    )
    for m in pattern.finditer(html):
        pdf_path = m.group(1)
        title = _clean(_strip_tags(m.group(2)))
        if title:
            entries.append({
                "title": title,
                "paper_id": "",
                "abstract": "",
                "pdf_url": base + pdf_path,
            })
    return entries


def parse_2007_index(html: str, base: str) -> list[dict]:
    """Parse 2007 proceedings index (Sphinx-rendered HTML).

    Per-paper structure:
      <div class="section" id="slug">
        <h2>TITLE</h2>
        <p><a href="proceedings/abstracts/{id}.htm">[Abstract]</a>
           <a href="proceedings/papers/{id}.pdf">[Paper]</a></p>
        <blockquote><p>AUTHOR - AFF</p>...</blockquote>
      </div>

    Returns entries with abstract_url set but abstract empty (fetched separately).
    """
    entries = []
    pattern = re.compile(
        r"<h2>(.*?)</h2>\s*"
        r'<p><a[^>]+href="(proceedings/abstracts/(\d+)\.htm)"[^>]*>[^<]*</a>'
        r'(?:\s*<a[^>]+href="(proceedings/papers/\d+\.pdf)"[^>]*>[^<]*</a>)?',
        re.DOTALL,
    )
    for m in pattern.finditer(html):
        title = _clean(_strip_tags(m.group(1)))
        abs_path = m.group(2)
        paper_id = m.group(3)
        pdf_path = m.group(4) or ""
        entries.append({
            "title": title,
            "paper_id": paper_id,
            "abstract_url": base + abs_path,
            "pdf_url": (base + pdf_path) if pdf_path else "",
            "abstract": "",
        })
    return entries


def parse_2007(html: str, base: str) -> list[dict]:
    """Full 2007 scrape: index page + concurrent abstract sub-page fetches.

    The 2007 abstract sub-pages use the same table structure as 2004.
    """
    entries = parse_2007_index(html, base)
    logger.info(f"    Fetching {len(entries)} abstract sub-pages in parallel…")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_2004_abstract, e["abstract_url"]): i
                   for i, e in enumerate(entries) if e.get("abstract_url")}
        for future in as_completed(futures):
            idx = futures[future]
            entries[idx]["abstract"] = future.result()
    return entries


def parse_2008(html: str, base: str) -> list[dict]:
    """Parse 2008 abstracts page.

    Per-paper structure (separated by <hr/>):
      <a name="{id}"></a>
      <p>paper ID: {id}</p>
      <h3>TITLE</h3>
      <p><i>AUTHORS</i><br></p>
      <p>ABSTRACT TEXT</p>
      <p>[<a href="papers/{id}.pdf">Full paper</a>] ...</p>
    """
    entries = []
    for block in re.split(r"<hr\s*/?>", html):
        id_m = re.search(r'<a name="(\d+)"></a>', block)
        if not id_m:
            continue
        paper_id = id_m.group(1)

        title_m = re.search(r"<h3>(.*?)</h3>", block, re.DOTALL)
        if not title_m:
            continue
        title = _clean(_strip_tags(title_m.group(1)))

        # Abstract: collect <p>...</p> blocks between the end of the authors
        # italic block and the [Full paper] link.
        authors_end = re.search(r"</i>", block)
        pdf_start = re.search(r'\[<a href="papers/', block)
        abstract = ""
        if authors_end and pdf_start:
            middle = block[authors_end.end():pdf_start.start()]
            parts = []
            for pb in re.findall(r"<p>(.*?)</p>", middle, re.DOTALL):
                text = _clean(_strip_tags(pb))
                if text and not text.startswith("paper ID"):
                    parts.append(text)
            abstract = " ".join(parts)

        pdf_m = re.search(r'href="(papers/\d+\.pdf)"', block)
        pdf_url = (base + pdf_m.group(1)) if pdf_m else ""

        if title:
            entries.append({
                "title": title, "paper_id": paper_id,
                "abstract": abstract, "pdf_url": pdf_url,
            })
    return entries


def parse_2009(html: str, base: str) -> list[dict]:
    """Parse 2009 abstracts page.

    Per-paper structure (separated by <hr/>):
      <h3><a name="{id}"></a>TITLE</h3>
         <p><i>AUTHORS</i></p>
        <p>paper ID: {id} </p>
          <p>ABSTRACT TEXT</p>
               [<a href="papers/{id}.pdf">Full paper</a>] ...
    """
    entries = []
    for block in re.split(r"<hr\s*/?>", html):
        id_m = re.search(r'<a name="(\d+)"></a>', block)
        if not id_m:
            continue
        paper_id = id_m.group(1)

        title_m = re.search(r'<h3><a name="\d+"></a>(.*?)</h3>', block, re.DOTALL)
        if not title_m:
            continue
        title = _clean(_strip_tags(title_m.group(1)))

        # Abstract: <p>...</p> block(s) after the "paper ID:" paragraph
        # and before the [Full paper] link (which is NOT wrapped in <p>)
        paper_id_end = re.search(r"paper ID:.*?</p>", block, re.DOTALL)
        pdf_start = re.search(r'\[<a href="papers/', block)
        abstract = ""
        if paper_id_end and pdf_start:
            middle = block[paper_id_end.end():pdf_start.start()]
            parts = []
            for pb in re.findall(r"<p>(.*?)</p>", middle, re.DOTALL):
                text = _clean(_strip_tags(pb))
                if text:
                    parts.append(text)
            abstract = " ".join(parts)

        pdf_m = re.search(r'href="(papers/\d+\.pdf)"', block)
        pdf_url = (base + pdf_m.group(1)) if pdf_m else ""

        if title:
            entries.append({
                "title": title, "paper_id": paper_id,
                "abstract": abstract, "pdf_url": pdf_url,
            })
    return entries


def parse_2010(html: str, base: str) -> list[dict]:
    """Parse 2010 abstracts page.

    Per-paper structure:
      <a name="{id}"></a>
      <p>Paper ID: {id}</p>
      <h3>TITLE</h3>
      <p><em>AUTHORS</em></p>
      <p class="abstracts">ABSTRACT</p>
      <p class="discussion">[<a href="papers/{id}.pdf">Full Paper</a>] ...
    """
    entries = []
    pattern = re.compile(
        r'<a name="(\d+)"></a>.*?'          # 1: paper_id
        r'<h3>(.*?)</h3>.*?'                # 2: title
        r'class="abstracts">(.*?)</p>.*?'   # 3: abstract
        r'href="(papers/\d+\.pdf)"',        # 4: pdf path
        re.DOTALL,
    )
    for m in pattern.finditer(html):
        entries.append({
            "title": _clean(_strip_tags(m.group(2))),
            "paper_id": m.group(1),
            "abstract": _clean(_strip_tags(m.group(3))),
            "pdf_url": base + m.group(4),
        })
    return entries


def parse_2011(html: str, base: str) -> list[dict]:
    """Parse 2011 papers page.

    Per-paper structure:
      <a name='{id}'><h3 style='...'>TITLE </h3>
      <span class='name'>AUTHORS</span>
      <p style='...'><span style='font-weight:bold;'>Abstract:</span>ABSTRACT </p>
      <p>[<a href='papers/{id}_icmlpaper.pdf'>download</a>] ...
    """
    entries = []
    pattern = re.compile(
        r"<a name='(\d+)'><h3[^>]*>(.*?)</h3>\s*"  # 1: id, 2: title
        r"<span class='name'>(.*?)</span>\s*"        # 3: authors
        r"<p[^>]*><span[^>]*>Abstract:</span>"       # abstract label
        r"(.*?)</p>\s*"                               # 4: abstract
        r"<p>\[<a href='(papers/\d+_icmlpaper\.pdf)'", # 5: pdf path
        re.DOTALL,
    )
    for m in pattern.finditer(html):
        entries.append({
            "title": _clean(_strip_tags(m.group(2))),
            "paper_id": m.group(1),
            "abstract": _clean(_strip_tags(m.group(4))),
            "pdf_url": base + m.group(5),
        })
    return entries


PARSERS = {
    "2004": parse_2004,
    "2005": parse_2005,
    "2007": parse_2007,
    "2008": parse_2008,
    "2009": parse_2009,
    "2010": parse_2010,
    "2011": parse_2011,
}


# ---------------------------------------------------------------------------
# ACM DOI → pdf_url derivation
# ---------------------------------------------------------------------------

# Years published by ACM Press (10.1145 DOIs) that benefit from this fallback.
ACM_YEARS = {"2003", "2004", "2005", "2006", "2007", "2008", "2009"}


def patch_acm_doi_pdfs(
    papers: list[dict],
    *,
    years: set[str] = ACM_YEARS,
    dry_run: bool = False,
) -> int:
    """Derive pdf_url from ACM DOI for papers that have a DOI but no pdf_url.

    Constructs https://dl.acm.org/doi/pdf/{doi} — these resolve in browsers
    (Cloudflare passes real users) even though they 403 programmatic scrapers.

    Returns the number of pdf_urls added.
    """
    added = 0
    for p in papers:
        if p.get("year") not in years:
            continue
        doi = p.get("doi", "")
        if not doi.startswith("10.1145/"):
            continue
        if p.get("pdf_url"):
            continue
        if not dry_run:
            p["pdf_url"] = f"https://dl.acm.org/doi/pdf/{doi}"
        added += 1
    return added


# ---------------------------------------------------------------------------
# AAAI Library 2003 abstract scraping
# ---------------------------------------------------------------------------

_AAAI_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


def _fetch_aaai_abstract(aaai_url: str) -> str:
    """Fetch one AAAI Library page and return its abstract text.

    The old ``http://www.aaai.org/Library/ICML/2003/icml03-NNN.php`` URLs
    redirect to ``https://aaai.org/papers/ICML03-NNN-slug/``.  The abstract
    lives in the ``<main>`` element after the label "Abstract:".

    AAAI rate-limits concurrent requests, so callers must serialise calls
    (see :func:`patch_aaai_2003`).
    """
    try:
        resp = fetch_with_retry(
            aaai_url,
            headers={"User-Agent": _AAAI_UA},
            return_none_on_404=True,
        )
        if resp is None:
            return ""
        main_m = re.search(r"<main[^>]*>(.*?)</main>", resp.text, re.DOTALL)
        if not main_m:
            return ""
        main_text = re.sub(r"<[^>]+>", " ", main_m.group(1))
        main_text = re.sub(r"\s+", " ", main_text).strip()
        abs_m = re.search(
            r"Abstract[:\s]+(.+?)(?=Topics:|Downloads?:|$)",
            main_text,
            re.DOTALL | re.IGNORECASE,
        )
        if not abs_m:
            return ""
        return _clean(abs_m.group(1))
    except Exception as e:
        logger.warning(f"  AAAI fetch failed for {aaai_url}: {e}")
        return ""


def patch_aaai_2003(
    papers: list[dict],
    *,
    delay: float = 1.0,
    dry_run: bool = False,
) -> int:
    """Scrape AAAI Library pages for ICML 2003 abstracts, sequentially.

    Derives the AAAI URL from each paper's ``venue_url`` field
    (``http://www.aaai.org/Library/ICML/2003/icml03-NNN.php``), rewrites it
    to use ``https://aaai.org/``, then fetches pages one at a time with
    *delay* seconds between requests to avoid triggering rate limiting.

    Returns the number of abstracts added (or that would be added in dry-run).
    """
    targets = [
        (i, p)
        for i, p in enumerate(papers)
        if p.get("year") == "2003"
        and not p.get("abstract")
        and p.get("venue_url", "").startswith("http")
        and "icml03-" in p.get("venue_url", "")
    ]
    if not targets:
        logger.info("  No 2003 papers need AAAI abstract scraping.")
        return 0

    logger.info(
        f"  Fetching {len(targets)} AAAI Library pages sequentially "
        f"(~{len(targets) * delay:.0f}s)…"
    )

    added = 0
    for n, (idx, paper) in enumerate(targets, 1):
        # Rewrite venue_url from www.aaai.org to aaai.org with https
        raw_url = paper["venue_url"]
        aaai_url = re.sub(r"^https?://(?:www\.)?(?:ed\.)?aaai\.org", "https://aaai.org", raw_url)

        if not dry_run:
            abstract = _fetch_aaai_abstract(aaai_url)
            if abstract:
                papers[idx]["abstract"] = abstract
                if papers[idx].get("source") == "dblp":
                    papers[idx]["source"] = "dblp+aaai"
                added += 1
                logger.debug(f"  [{n}/{len(targets)}] +abstract: {paper['title'][:60]}")
            else:
                logger.debug(f"  [{n}/{len(targets)}] no abstract: {aaai_url}")
            if n < len(targets):
                time.sleep(delay)
        else:
            added += 1  # assume we'd get an abstract for each target

        if n % 20 == 0 or n == len(targets):
            logger.info(f"  Progress: {n}/{len(targets)} fetched, {added} abstracts so far")

    return added


# ---------------------------------------------------------------------------
# OpenAlex abstract enrichment for 2005 / 2006
# ---------------------------------------------------------------------------

# Years that lack inline abstracts from icml.cc but have DOIs and are covered
# by OpenAlex.
OPENALEX_YEARS = {"2005", "2006"}


def patch_openalex_abstracts(
    papers: list[dict],
    *,
    years: set[str] = OPENALEX_YEARS,
    dry_run: bool = False,
) -> int:
    """Enrich abstracts for ICML 2005/2006 via the OpenAlex batch API.

    Looks up papers that have a DOI but no abstract using the OpenAlex
    ``/works`` endpoint (batch filter by DOI).  Only fills empty abstract
    fields — never overwrites existing data.

    Returns the number of abstracts added (or that would be added in dry-run).
    """
    targets = [
        (i, p)
        for i, p in enumerate(papers)
        if p.get("year") in years
        and p.get("doi")
        and not p.get("abstract")
    ]
    if not targets:
        logger.info("  No papers need OpenAlex abstract enrichment.")
        return 0

    year_counts = {}
    for _, p in targets:
        year_counts[p["year"]] = year_counts.get(p["year"], 0) + 1
    logger.info(
        f"  Fetching OpenAlex abstracts for {len(targets)} papers "
        f"({', '.join(f'{y}: {n}' for y, n in sorted(year_counts.items()))})…"
    )

    if dry_run:
        return len(targets)

    dois = [p["doi"] for _, p in targets]
    doi_to_abstract = fetch_abstracts_by_doi(dois)

    added = 0
    for idx, paper in targets:
        abstract = doi_to_abstract.get(paper["doi"], "")
        if abstract:
            papers[idx]["abstract"] = abstract
            if papers[idx].get("source") == "dblp":
                papers[idx]["source"] = "dblp+openalex"
            added += 1
            logger.debug(f"  +abstract (OpenAlex): {paper['title'][:60]}")

    logger.info(f"  OpenAlex: {added}/{len(targets)} abstracts retrieved")
    return added


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def fetch_year(year: str) -> list[dict]:
    """Fetch the conference page for *year* and return parsed entries."""
    conf = CONF_PAGES[year]
    logger.info(f"    GET {conf['url']}")
    resp = fetch_with_retry(conf["url"])
    return PARSERS[year](resp.text, conf["base"])


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_year(
    papers: list[dict],
    year: str,
    entries: list[dict],
    *,
    dry_run: bool = False,
) -> dict:
    """Apply *entries* scraped from icml.cc to legacy *papers* for *year*.

    Returns stats: matched, abstract_added, pdf_added.
    Only fills empty fields — never overwrites existing data.
    """
    # Build normalised-title index over legacy records for this year
    idx_by_title: dict[str, int] = {}
    for i, p in enumerate(papers):
        if p.get("year") == year:
            key = normalize_title(p["title"])
            if key not in idx_by_title:
                idx_by_title[key] = i

    matched = abstract_added = pdf_added = 0

    for entry in entries:
        norm = normalize_title(entry["title"])
        legacy_idx = idx_by_title.get(norm)
        if legacy_idx is None:
            logger.debug(f"  No match for: {entry['title']!r}")
            continue
        matched += 1
        p = papers[legacy_idx]

        if not dry_run:
            if entry.get("abstract") and not p.get("abstract"):
                p["abstract"] = entry["abstract"]
                if p.get("source") == "dblp":
                    p["source"] = "dblp+icmlcc"
                abstract_added += 1

            if entry.get("pdf_url") and not p.get("pdf_url"):
                p["pdf_url"] = entry["pdf_url"]
                pdf_added += 1
        else:
            if entry.get("abstract") and not p.get("abstract"):
                abstract_added += 1
            if entry.get("pdf_url") and not p.get("pdf_url"):
                pdf_added += 1

    return {"matched": matched, "abstract_added": abstract_added, "pdf_added": pdf_added}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patch ICML legacy records with data from icml.cc proceedings pages"
    )
    ap.add_argument(
        "--year", choices=sorted(CONF_PAGES) + ["2003"],
        help="Process a single year (default: all); '2003' only runs the AAAI scraping step",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything",
    )
    ap.add_argument(
        "--no-doi-pdf", action="store_true",
        help="Skip the ACM DOI → pdf_url derivation step",
    )
    ap.add_argument(
        "--no-aaai", action="store_true",
        help="Skip the AAAI Library 2003 abstract scraping step",
    )
    ap.add_argument(
        "--aaai-delay", type=float, default=1.0,
        help="Seconds between AAAI Library requests (default: 1.0)",
    )
    ap.add_argument(
        "--no-openalex", action="store_true",
        help="Skip the OpenAlex 2005/2006 abstract enrichment step",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    legacy_path = LEGACY_DIR / "icml-legacy.jsonl.gz"
    if not legacy_path.exists():
        logger.error(f"Legacy file not found: {legacy_path}")
        sys.exit(1)

    papers = read_legacy(legacy_path)
    logger.info(f"Loaded {len(papers)} ICML legacy records from {legacy_path.name}")

    # "2003" has no CONF_PAGES entry — handled solely by the AAAI scraping step below.
    if args.year is None:
        years = sorted(CONF_PAGES)
    elif args.year in CONF_PAGES:
        years = [args.year]
    else:
        years = []  # e.g. --year 2003 skips the icml.cc loop

    grand_matched = grand_abs = grand_pdf = 0

    for year in years:
        yr_papers = [p for p in papers if p.get("year") == year]
        before_abs = sum(1 for p in yr_papers if p.get("abstract"))
        before_pdf = sum(1 for p in yr_papers if p.get("pdf_url"))
        logger.info(
            f"\n--- ICML {year} ---  "
            f"({len(yr_papers)} papers, {before_abs} abstracts, {before_pdf} PDFs)"
        )

        try:
            entries = fetch_year(year)
        except Exception as e:
            logger.error(f"  Failed to scrape ICML {year}: {e}")
            continue

        with_abs = sum(1 for e in entries if e.get("abstract"))
        with_pdf = sum(1 for e in entries if e.get("pdf_url"))
        logger.info(f"  Scraped {len(entries)} entries ({with_abs} have abstract, {with_pdf} have PDF)")

        stats = patch_year(papers, year, entries, dry_run=args.dry_run)

        if args.dry_run:
            after_abs = before_abs + stats["abstract_added"]
            after_pdf = before_pdf + stats["pdf_added"]
        else:
            after_abs = sum(1 for p in papers if p.get("year") == year and p.get("abstract"))
            after_pdf = sum(1 for p in papers if p.get("year") == year and p.get("pdf_url"))

        mode = "[DRY RUN] " if args.dry_run else ""
        logger.info(
            f"  {mode}Matched {stats['matched']}/{len(entries)}  "
            f"+{stats['abstract_added']} abstracts  +{stats['pdf_added']} PDFs"
        )
        logger.info(
            f"  {'Would be' if args.dry_run else 'Now'}: "
            f"{after_abs} abstracts ({after_abs - before_abs:+d}), "
            f"{after_pdf} PDFs ({after_pdf - before_pdf:+d})"
        )
        grand_matched += stats["matched"]
        grand_abs += stats["abstract_added"]
        grand_pdf += stats["pdf_added"]

    # ACM DOI → pdf_url derivation (runs regardless of --year filter)
    if not args.no_doi_pdf:
        doi_years = ACM_YEARS if not args.year else {args.year}
        doi_years = doi_years & ACM_YEARS  # only ACM years
        if doi_years:
            n_doi = patch_acm_doi_pdfs(papers, years=doi_years, dry_run=args.dry_run)
            mode = "[DRY RUN] " if args.dry_run else ""
            logger.info(f"\n--- ACM DOI → pdf_url ---")
            logger.info(f"  {mode}+{n_doi} pdf_urls derived from DOI (dl.acm.org)")
            grand_pdf += n_doi

    # AAAI Library 2003 abstract scraping (always runs unless filtered out)
    run_aaai = not args.no_aaai and (args.year is None or args.year == "2003")
    if run_aaai:
        logger.info("\n--- AAAI Library 2003 abstracts ---")
        before_aaai = sum(1 for p in papers if p.get("year") == "2003" and p.get("abstract"))
        n_aaai = patch_aaai_2003(papers, delay=args.aaai_delay, dry_run=args.dry_run)
        after_aaai = before_aaai + n_aaai if args.dry_run else sum(
            1 for p in papers if p.get("year") == "2003" and p.get("abstract")
        )
        mode = "[DRY RUN] " if args.dry_run else ""
        logger.info(
            f"  {mode}+{n_aaai} abstracts  "
            f"({before_aaai} → {after_aaai} / "
            f"{sum(1 for p in papers if p.get('year') == '2003')} papers)"
        )
        grand_abs += n_aaai

    # OpenAlex abstract enrichment for 2005/2006 (runs unless --no-openalex)
    run_openalex = not args.no_openalex and (
        args.year is None or args.year in OPENALEX_YEARS
    )
    if run_openalex:
        openalex_years = OPENALEX_YEARS if not args.year else {args.year} & OPENALEX_YEARS
        if openalex_years:
            logger.info("\n--- OpenAlex 2005/2006 abstracts ---")
            before_oa: dict[str, int] = {
                y: sum(1 for p in papers if p.get("year") == y and p.get("abstract"))
                for y in sorted(openalex_years)
            }
            n_oa = patch_openalex_abstracts(
                papers, years=openalex_years, dry_run=args.dry_run
            )
            mode = "[DRY RUN] " if args.dry_run else ""
            for y in sorted(openalex_years):
                after_y = before_oa[y] + n_oa if args.dry_run else sum(
                    1 for p in papers if p.get("year") == y and p.get("abstract")
                )
                total_y = sum(1 for p in papers if p.get("year") == y)
                logger.info(
                    f"  {mode}ICML {y}: {before_oa[y]} → {after_y} abstracts / {total_y}"
                )
            grand_abs += n_oa

    print()
    if not args.dry_run:
        write_legacy(legacy_path, papers, atomic=True)
        logger.info(f"Written back to {legacy_path.name}")

    logger.info(
        f"\nTotal across {len(years)} year(s): "
        f"{grand_matched} matched, +{grand_abs} abstracts, +{grand_pdf} PDFs"
    )


if __name__ == "__main__":
    main()
