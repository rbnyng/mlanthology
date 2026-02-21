#!/usr/bin/env python3
"""Patch IJCAI legacy records with data scraped from ijcai.org proceedings pages.

Covers under-populated years that all have machine-readable proceedings at
ijcai.org but differ in how the data is sourced:

  1977 — pdf_url filled by scraping multi-volume proceedings index
          (/proceedings/1977-1, /proceedings/1977-2) and title-matching.

  2007 — abstract scraped from /Abstract/07/NNN; paper number derived from
          pdf_url (ijcai.org/Proceedings/07/Papers/NNN.pdf).

  2009 — same as 2007 with year code 09.

  2011 — venue_url is a DOI (10.5591/.../IJCAI11-NNN); pdf_url and abstract_url
          derived from the paper number embedded in the DOI.

  2013 — venue_url points to a dead AAAI OCS page; proceedings index scraped
          from ijcai.org/proceedings/2013 to recover title→(pdf, abstract) map.

  2015 — venue_url is already http://ijcai.org/Abstract/15/NNN; pdf_url derived
          inline, abstract fetched from venue_url.

  2016 — same as 2015 with year code 16.

  2017-2022 — pdf_url derived from DOI (10.24963/ijcai.YEAR/N → proceedings
          URL with 4-digit zero-padded paper number).  Metadata-only; no
          abstract scraping needed (coverage already ≥99.9%).

  2024 — venue_url is https://www.ijcai.org/proceedings/2024/NNN; DOI
          (10.24963/ijcai.2024/NNN) and pdf_url derived inline.  Abstracts
          are left for enrich_crossref.py which has full Crossref coverage.

Note: other pre-2007 years (1979, 1981, 1983, 1985, 1991, 1993, 1997, 2001)
were investigated but the ijcai.org proceedings indexes either don't have
individual paper PDFs (1979, 2001) or the missing DBLP papers don't appear
on the index at all (the rest — likely workshop/invited content not digitized
separately by IJCAI).

Coverage gains (approximate):
  1977:  +2 pdf_urls (from proceedings index title-match)
  1981:  52% → 86% PDF (via PDF enumeration + OCR title-match)
  1983:  86% → 86% PDF (enumeration found 0 new; gap is non-digitised papers)
  1993:  56% → 57% PDF (enumeration found 1 new; gap is non-digitised papers)
  1997:  76% → 76% PDF (enumeration found 0 new; gap is non-digitised papers)
  2007:  ~4% → ~99% abstract (via /Abstract/ pages)
  2009:  ~5% → ~99% abstract (via /Abstract/ pages)
  2011:  2% → ~99% PDF,  4% → ~99% abstract
  2013:  8% → ~99% PDF,  1% → ~99% abstract
  2015:  0% → ~99% PDF,  0% → ~99% abstract
  2016:  0% → ~99% PDF,  0% → ~99% abstract
  2017-2022: +114/104/97/146/152/24 pdf_urls derived from DOI
  2024:  already 100% PDF; 0% → 100% DOI (unlocks Crossref abstract fetch)

Usage:
    python scripts/patch_ijcai_legacy.py

    # Dry-run (counts only, no writes):
    python scripts/patch_ijcai_legacy.py --dry-run

    # Single year:
    python scripts/patch_ijcai_legacy.py --year 2024

    # Skip abstract scraping (PDF/DOI derivation only):
    python scripts/patch_ijcai_legacy.py --no-abstracts
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

import requests

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy, normalize_title
from adapters.http import fetch_with_retry

logger = logging.getLogger(__name__)

BASE = "https://www.ijcai.org"

# maximum concurrent abstract page fetches per year
_WORKERS = 12



def _strip_tags(s: str) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def scrape_proceedings_index(year: str) -> list[dict]:
    """Fetch the ijcai.org proceedings index for *year* and return paper entries.

    Each entry dict has: title, pdf_url, abstract_url.
    Only entries that have both a PDF link and an Abstract link are returned.
    """
    url = f"{BASE}/proceedings/{year}"
    logger.info(f"  GET {url}")
    resp = fetch_with_retry(url)
    html = resp.text

    year2 = year[-2:]  # e.g. "11", "13"
    entries = []

    # Match <p> blocks that contain both a PDF link and an Abstract link.
    # The title is the link text of the PDF <a> element.
    pattern = re.compile(
        r"<p>"
        r'<a\s+href="([^"]*Proceedings/' + year2 + r'/Papers/\d+\.pdf)">'
        r"(.*?)"           # title HTML
        r"</a>"
        r".*?"             # / PAGE<br /><i>authors</i><br />[doi comment]
        r'<a\s+href="([^"]*Abstract/' + year2 + r'/\d+)">Abstract</a>'
        r"\s*</p>",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(html):
        pdf_path = m.group(1)
        title = _clean(_strip_tags(m.group(2)))
        abstract_path = m.group(3)

        if not title:
            continue

        pdf_url = pdf_path if pdf_path.startswith("http") else BASE + pdf_path
        abstract_url = abstract_path if abstract_path.startswith("http") else BASE + abstract_path

        entries.append({
            "title": title,
            "pdf_url": pdf_url,
            "abstract_url": abstract_url,
        })

    logger.info(f"  Parsed {len(entries)} entries from {year} proceedings index")
    return entries


def _fetch_abstract(abstract_url: str) -> str:
    """Fetch one ijcai.org abstract page and return the abstract text.

    Two page formats exist:

    Pre-2013 (e.g. 2011):
      <div class="pabstract">ABSTRACT TEXT</div>

    2013+ (embedded XHTML inside the Drupal content block):
      <div class="content">
        ...
        <p>TITLE / PAGE<br/>AUTHORS<br/><a href="...">PDF</a></p>
        <p>ABSTRACT PARAGRAPH 1</p>
        <p>ABSTRACT PARAGRAPH 2</p>
        ...
      </div>
    """
    try:
        resp = fetch_with_retry(abstract_url, return_none_on_404=True)
        if resp is None:
            return ""
        html = resp.text

        # Format 1: <div class="pabstract">...</div>  (pre-2013)
        m = re.search(r'<div\s+class="pabstract">(.*?)</div>', html, re.DOTALL)
        if m:
            return _clean(_strip_tags(m.group(1)))

        # Format 2: embedded XHTML in <div class="content"> (2013+)
        m = re.search(r'<div\s+class="content">(.*?)(?=</div>)', html, re.DOTALL)
        if not m:
            return ""
        inner = m.group(1)

        # Skip the first <p> (title/authors/PDF block); collect rest as abstract
        paras = re.findall(r"<p>(.*?)</p>", inner, re.DOTALL)
        abstract_parts = []
        for para in paras[1:]:
            text = _clean(_strip_tags(para))
            if text:
                abstract_parts.append(text)

        return " ".join(abstract_parts)

    except Exception as e:
        logger.warning(f"  Failed to fetch abstract {abstract_url}: {e}")
        return ""


def _fetch_abstracts_concurrent(
    items: list[tuple[int, str]],  # [(legacy_idx, abstract_url), ...]
) -> dict[int, str]:
    """Fetch abstract pages concurrently. Returns {legacy_idx: abstract_text}."""
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_abstract, url): idx
            for idx, url in items
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            abstract = future.result()
            if abstract:
                results[idx] = abstract
            done += 1
            if done % 50 == 0 or done == len(items):
                logger.info(f"    {done}/{len(items)} abstract pages fetched ({len(results)} non-empty)")
    return results


def _abstract_url_from_venue_url(venue_url: str, year2: str) -> str:
    """Extract abstract URL from venue_url for years 2015/2016.

    venue_url is http://ijcai.org/Abstract/YY/NNN or similar.
    Returns the canonical https://www.ijcai.org/Abstract/YY/NNN form.
    """
    m = re.search(r"/Abstract/" + year2 + r"/(\d+)", venue_url, re.IGNORECASE)
    if not m:
        return ""
    return f"{BASE}/Abstract/{year2}/{m.group(1)}"


def _pdf_url_from_venue_url(venue_url: str, year2: str) -> str:
    """Derive PDF URL from venue_url for years 2015/2016."""
    m = re.search(r"/Abstract/" + year2 + r"/(\d+)", venue_url, re.IGNORECASE)
    if not m:
        return ""
    return f"{BASE}/Proceedings/{year2}/Papers/{m.group(1)}.pdf"


def _pdf_url_from_doi(doi: str) -> str:
    """Derive PDF URL from IJCAI 2011 DOI (10.5591/.../IJCAI11-NNN)."""
    m = re.search(r"IJCAI11-(\d+)", doi, re.IGNORECASE)
    if not m:
        return ""
    num = m.group(1).zfill(3)
    return f"{BASE}/Proceedings/11/Papers/{num}.pdf"


def _abstract_url_from_doi(doi: str) -> str:
    """Derive abstract URL from IJCAI 2011 DOI."""
    m = re.search(r"IJCAI11-(\d+)", doi, re.IGNORECASE)
    if not m:
        return ""
    num = m.group(1).zfill(3)
    return f"{BASE}/Abstract/11/{num}"


def patch_year_from_venue_url(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,
    dry_run: bool = False,
) -> dict:
    """Patch 2015 / 2016 using venue_url for both pdf_url and abstract fetching."""
    year2 = year[-2:]

    # Identify papers needing work
    needs_pdf = [
        (i, p) for i, p in enumerate(papers)
        if p.get("year") == year
        and not p.get("pdf_url")
        and re.search(r"/Abstract/" + year2 + r"/\d+", p.get("venue_url", ""), re.IGNORECASE)
    ]
    needs_abstract = [
        (i, p) for i, p in enumerate(papers)
        if p.get("year") == year
        and not p.get("abstract")
        and re.search(r"/Abstract/" + year2 + r"/\d+", p.get("venue_url", ""), re.IGNORECASE)
    ] if scrape_abstracts else []

    pdf_added = 0
    if not dry_run:
        for i, p in needs_pdf:
            pdf = _pdf_url_from_venue_url(p["venue_url"], year2)
            if pdf:
                papers[i]["pdf_url"] = pdf
                pdf_added += 1
    else:
        pdf_added = sum(
            1 for _, p in needs_pdf
            if _pdf_url_from_venue_url(p.get("venue_url", ""), year2)
        )

    abstract_added = 0
    if scrape_abstracts and needs_abstract:
        logger.info(f"  Fetching {len(needs_abstract)} abstract pages for {year}…")
        if not dry_run:
            items = [
                (i, _abstract_url_from_venue_url(p["venue_url"], year2))
                for i, p in needs_abstract
                if _abstract_url_from_venue_url(p.get("venue_url", ""), year2)
            ]
            results = _fetch_abstracts_concurrent(items)
            for idx, abstract in results.items():
                papers[idx]["abstract"] = abstract
                abstract_added += 1
        else:
            abstract_added = len(needs_abstract)

    return {"pdf_added": pdf_added, "abstract_added": abstract_added}


def patch_year_from_doi(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,
    dry_run: bool = False,
) -> dict:
    """Patch 2011 using DOI for pdf_url and abstract URL derivation."""
    needs_pdf = [
        (i, p) for i, p in enumerate(papers)
        if p.get("year") == year
        and not p.get("pdf_url")
        and re.search(r"IJCAI11-\d+", p.get("doi", ""), re.IGNORECASE)
    ]
    needs_abstract = [
        (i, p) for i, p in enumerate(papers)
        if p.get("year") == year
        and not p.get("abstract")
        and re.search(r"IJCAI11-\d+", p.get("doi", ""), re.IGNORECASE)
    ] if scrape_abstracts else []

    pdf_added = 0
    if not dry_run:
        for i, p in needs_pdf:
            pdf = _pdf_url_from_doi(p["doi"])
            if pdf:
                papers[i]["pdf_url"] = pdf
                pdf_added += 1
    else:
        pdf_added = sum(1 for _, p in needs_pdf if _pdf_url_from_doi(p.get("doi", "")))

    abstract_added = 0
    if scrape_abstracts and needs_abstract:
        logger.info(f"  Fetching {len(needs_abstract)} abstract pages for {year}…")
        if not dry_run:
            items = [
                (i, _abstract_url_from_doi(p["doi"]))
                for i, p in needs_abstract
                if _abstract_url_from_doi(p.get("doi", ""))
            ]
            results = _fetch_abstracts_concurrent(items)
            for idx, abstract in results.items():
                papers[idx]["abstract"] = abstract
                abstract_added += 1
        else:
            abstract_added = len(needs_abstract)

    return {"pdf_added": pdf_added, "abstract_added": abstract_added}


def patch_year_from_index(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,
    dry_run: bool = False,
) -> dict:
    """Patch year by scraping the proceedings index then abstract pages.

    Used for 2013 where venue_url points to a dead AAAI OCS page.
    """
    entries = scrape_proceedings_index(year)
    if not entries:
        logger.warning(f"  No entries scraped for {year}, skipping")
        return {"pdf_added": 0, "abstract_added": 0}

    # Build title → legacy index
    title_idx: dict[str, int] = {}
    for i, p in enumerate(papers):
        if p.get("year") == year:
            key = normalize_title(p["title"])
            if key not in title_idx:
                title_idx[key] = i

    pdf_added = 0
    abstract_targets: list[tuple[int, str]] = []
    matched = 0

    for entry in entries:
        norm = normalize_title(entry["title"])
        legacy_i = title_idx.get(norm)
        if legacy_i is None:
            logger.debug(f"  No match: {entry['title']!r}")
            continue
        matched += 1
        p = papers[legacy_i]

        if not p.get("pdf_url"):
            if not dry_run:
                papers[legacy_i]["pdf_url"] = entry["pdf_url"]
            pdf_added += 1

        if scrape_abstracts and not p.get("abstract") and entry.get("abstract_url"):
            abstract_targets.append((legacy_i, entry["abstract_url"]))

    logger.info(f"  Matched {matched}/{len(entries)} entries from index, +{pdf_added} PDFs")

    abstract_added = 0
    if scrape_abstracts and abstract_targets:
        logger.info(f"  Fetching {len(abstract_targets)} abstract pages for {year}…")
        if not dry_run:
            results = _fetch_abstracts_concurrent(abstract_targets)
            for idx, abstract in results.items():
                papers[idx]["abstract"] = abstract
                abstract_added += 1
        else:
            abstract_added = len(abstract_targets)

    return {"pdf_added": pdf_added, "abstract_added": abstract_added}


# 2007/2009: pdf_url → abstract URL
# pdf_url:  .../Proceedings/YY/Papers/NNN.pdf  →  .../Abstract/YY/NNN
_PDF_URL_PAPER_RE = re.compile(
    r"ijcai\.org/Proceedings/(\d{2})/Papers/(\d+)\.pdf",
    re.IGNORECASE,
)


def _abstract_url_from_pdf_url(pdf_url: str) -> str:
    """Derive abstract page URL from the old-style IJCAI pdf_url."""
    m = _PDF_URL_PAPER_RE.search(pdf_url)
    if not m:
        return ""
    year2, num = m.group(1), m.group(2)
    return f"{BASE}/Abstract/{year2}/{num}"


def patch_year_from_pdf_url(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,
    dry_run: bool = False,
) -> dict:
    """Patch 2007/2009 by deriving abstract URL from pdf_url and scraping."""
    needs_abstract = [
        (i, p) for i, p in enumerate(papers)
        if p.get("year") == year
        and p.get("authors")  # skip front matter
        and not p.get("abstract")
        and _abstract_url_from_pdf_url(p.get("pdf_url", ""))
    ] if scrape_abstracts else []

    abstract_added = 0
    if scrape_abstracts and needs_abstract:
        logger.info(f"  Fetching {len(needs_abstract)} abstract pages for {year}…")
        if not dry_run:
            items = [
                (i, _abstract_url_from_pdf_url(p["pdf_url"]))
                for i, p in needs_abstract
            ]
            results = _fetch_abstracts_concurrent(items)
            for idx, abstract in results.items():
                papers[idx]["abstract"] = abstract
                abstract_added += 1
        else:
            abstract_added = len(needs_abstract)

    return {"abstract_added": abstract_added, "pdf_added": 0}


# Year → volume suffixes for multi-volume proceedings indexes
# (1979, 2001 excluded: no individual paper PDFs on ijcai.org)
_MULTI_VOLUME_YEARS: dict[str, list[str]] = {
    "1977": ["1", "2"],
    "1981": ["1", "2"],
    "1983": ["1", "2"],
    "1985": ["1", "2"],
    "1991": ["1", "2"],
    "1993": ["1", "2"],
    "1997": ["1", "2"],
}

_INDEX_PDF_RE = re.compile(
    r'<a\s+href="([^"]*Proceedings/[^"]*\.pdf)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _scrape_volume_index(year: str, vol_suffix: str) -> list[dict]:
    """Scrape a single volume proceedings index, returning title→pdf_url entries."""
    url = f"{BASE}/proceedings/{year}-{vol_suffix}"
    logger.info(f"  GET {url}")
    try:
        resp = fetch_with_retry(url, return_none_on_404=True)
        if resp is None:
            logger.warning(f"  404 for {url}")
            return []
    except Exception as e:
        logger.warning(f"  Failed to fetch {url}: {e}")
        return []

    entries = []
    for m in _INDEX_PDF_RE.finditer(resp.text):
        pdf_path = m.group(1)
        title = _clean(_strip_tags(m.group(2)))
        if not title:
            continue
        pdf_url = pdf_path if pdf_path.startswith("http") else BASE + "/" + pdf_path.lstrip("/")
        entries.append({"title": title, "pdf_url": pdf_url})

    return entries


def patch_year_from_index_volumes(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,  # accepted for interface compat; unused
    dry_run: bool = False,
) -> dict:
    """Fill missing pdf_urls by title-matching against multi-volume proceedings index."""
    vol_suffixes = _MULTI_VOLUME_YEARS.get(year, [])
    if not vol_suffixes:
        return {"pdf_added": 0, "abstract_added": 0}

    # Scrape all volume index pages
    all_entries: list[dict] = []
    for vol in vol_suffixes:
        entries = _scrape_volume_index(year, vol)
        all_entries.extend(entries)

    if not all_entries:
        logger.warning(f"  No entries scraped for {year}")
        return {"pdf_added": 0, "abstract_added": 0}

    logger.info(f"  Scraped {len(all_entries)} entries from {len(vol_suffixes)} volumes")

    # Build title → pdf_url map from scraped entries
    scraped_map: dict[str, str] = {}
    for entry in all_entries:
        key = normalize_title(entry["title"])
        if key:
            scraped_map[key] = entry["pdf_url"]

    # Match against legacy records missing pdf_url
    pdf_added = 0
    for i, p in enumerate(papers):
        if p.get("year") != year or not p.get("authors"):
            continue
        if p.get("pdf_url"):
            continue
        key = normalize_title(p.get("title", ""))
        pdf_url = scraped_map.get(key)
        if pdf_url:
            if not dry_run:
                papers[i]["pdf_url"] = pdf_url
            pdf_added += 1

    logger.info(f"  Matched {pdf_added} missing pdf_urls from proceedings index")
    return {"pdf_added": pdf_added, "abstract_added": 0}


# years where enumerating server PDFs + OCR title-matching  recovers more than
# the proceedings index links to.  HEAD requests stop after 5 consecutive 404s.
_ENUMERATE_YEARS: dict[str, list[str]] = {
    "1981": ["1", "2"],
    "1983": ["1", "2"],
    "1993": ["1", "2"],
    "1997": ["1", "2"],
}


def _enumerate_pdf_urls(year: str, vol: str, session: requests.Session) -> list[str]:
    """Return all valid PDF URLs for a given year/volume by enumeration."""
    year2 = year[2:]  # "1981" -> "81"
    urls: list[str] = []
    consecutive_miss = 0
    for n in range(1, 300):
        url = f"{BASE}/Proceedings/{year2}-{vol}/Papers/{n:03d}.pdf"
        try:
            resp = session.head(url, timeout=10)
            if resp.status_code == 200:
                urls.append(url)
                consecutive_miss = 0
            else:
                consecutive_miss += 1
                if consecutive_miss > 5:
                    break
        except Exception:
            consecutive_miss += 1
            if consecutive_miss > 5:
                break
    return urls


def patch_year_from_pdf_ocr(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,  # accepted for interface compat; unused
    dry_run: bool = False,
) -> dict:
    """Fill missing pdf_urls by enumerating server PDFs and OCR title-matching."""
    from scripts.utils import pdf_title_from_url, make_session

    vol_suffixes = _ENUMERATE_YEARS.get(year, [])
    if not vol_suffixes:
        return {"pdf_added": 0, "abstract_added": 0}

    session = make_session(timeout=15)

    # Collect already-known PDF paths for this year (normalise to just the
    # Proceedings/YY-V/Papers/NNN.pdf path component for comparison, since
    # existing URLs may use ijcai.org vs www.ijcai.org or http vs https).
    _path_re = re.compile(r"Proceedings/[^/]+/Papers/[^/]+\.pdf", re.IGNORECASE)
    known_paths: set[str] = set()
    for p in papers:
        if p.get("year") == year and p.get("pdf_url"):
            m = _path_re.search(p["pdf_url"])
            if m:
                known_paths.add(m.group(0).lower())

    # Enumerate all PDFs on server
    all_urls: list[str] = []
    for vol in vol_suffixes:
        urls = _enumerate_pdf_urls(year, vol, session)
        logger.info(f"  Enumerated {len(urls)} PDFs for {year}-{vol}")
        all_urls.extend(urls)

    # Filter to unlisted PDFs only
    unlisted: list[str] = []
    for url in all_urls:
        m = _path_re.search(url)
        path = m.group(0).lower() if m else url.lower()
        if path not in known_paths:
            unlisted.append(url)

    logger.info(f"  {len(unlisted)} unlisted PDFs to OCR title-match")

    if not unlisted or dry_run:
        return {"pdf_added": len(unlisted) if dry_run else 0, "abstract_added": 0}

    # Build title → list index for papers missing pdf_url
    missing_map: dict[str, int] = {}
    for i, p in enumerate(papers):
        if p.get("year") == year and p.get("authors") and not p.get("pdf_url"):
            key = normalize_title(p.get("title", ""))
            if key:
                missing_map[key] = i

    # OCR each unlisted PDF and try to match.
    # Old proceedings PDFs often have author names or artefacts appended to the
    # title line, so we try three strategies:
    #   1. Exact normalised match (fastest, most reliable).
    #   2. Check if a known missing title is a prefix of the OCR text.
    #   3. Check if the OCR text is a prefix of a known missing title.
    pdf_added = 0
    for url in unlisted:
        title = pdf_title_from_url(url, session=session)
        if not title:
            continue
        key = normalize_title(title)
        if not key or len(key) < 10:
            continue  # too short to reliably match

        # Strategy 1: exact match
        idx = missing_map.get(key)
        if idx is not None:
            papers[idx]["pdf_url"] = url
            pdf_added += 1
            del missing_map[key]
            continue

        # Strategy 2/3: prefix matching (handles author bleed-in and truncation)
        best_idx = None
        best_key = None
        best_len = 0
        for mk, mi in missing_map.items():
            # OCR text starts with the missing title
            if key.startswith(mk) and len(mk) > best_len:
                best_idx, best_key, best_len = mi, mk, len(mk)
            # Missing title starts with the OCR text (truncated OCR)
            elif mk.startswith(key) and len(key) > best_len and len(key) >= 20:
                best_idx, best_key, best_len = mi, mk, len(key)

        if best_idx is not None and best_len >= 15:
            papers[best_idx]["pdf_url"] = url
            pdf_added += 1
            del missing_map[best_key]
            continue

        # Strategy 4: word overlap — for cases where OCR mangles word order
        # or inserts artefacts. Require ≥80% of the missing title's words to
        # appear in the OCR text, and the missing title must be ≥4 words.
        ocr_words = set(key.split())
        best_idx = None
        best_key = None
        best_overlap = 0.0
        for mk, mi in missing_map.items():
            mk_words = set(mk.split())
            if len(mk_words) < 4:
                continue
            overlap = len(mk_words & ocr_words) / len(mk_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx, best_key = mi, mk

        if best_idx is not None and best_overlap >= 0.80:
            papers[best_idx]["pdf_url"] = url
            pdf_added += 1
            del missing_map[best_key]

    logger.info(f"  OCR matched {pdf_added} missing pdf_urls")
    return {"pdf_added": pdf_added, "abstract_added": 0}


# 2017-2022: DOI 10.24963/ijcai.YEAR/N → .../proceedings/YEAR/NNNN.pdf
_IJCAI_DOI_RE = re.compile(
    r"10\.24963/ijcai\.(\d{4})/(\d+)",
    re.IGNORECASE,
)


def _extract_ijcai_doi_parts(paper: dict) -> tuple[str, str] | None:
    """Extract (year, number) from an IJCAI DOI in doi or venue_url fields."""
    for field in ("doi", "venue_url"):
        val = paper.get(field, "")
        m = _IJCAI_DOI_RE.search(val)
        if m:
            return m.group(1), m.group(2)
    return None


def patch_year_derive_pdf(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,  # accepted for interface compat; unused
    dry_run: bool = False,
) -> dict:
    """Derive pdf_url from DOI for 2017-2022 IJCAI papers."""
    pdf_added = 0

    for i, p in enumerate(papers):
        if p.get("year") != year or not p.get("authors"):
            continue
        if p.get("pdf_url"):
            continue
        parts = _extract_ijcai_doi_parts(p)
        if not parts:
            continue
        paper_year, paper_num = parts
        pdf = f"https://www.ijcai.org/proceedings/{paper_year}/{int(paper_num):04d}.pdf"
        if not dry_run:
            papers[i]["pdf_url"] = pdf
        pdf_added += 1

    return {"pdf_added": pdf_added, "abstract_added": 0}


# 2024: venue_url .../proceedings/2024/NNN → DOI + pdf_url
_PROCEEDINGS_URL_RE = re.compile(
    r"https?://(?:www\.)?ijcai\.org/proceedings/(\d{4})/(\d+)",
    re.IGNORECASE,
)


def patch_year_2024(
    papers: list[dict],
    year: str,
    *,
    scrape_abstracts: bool = True,   # accepted for interface compat; unused
    dry_run: bool = False,
) -> dict:
    """Derive DOI and pdf_url for 2024 IJCAI papers from venue_url.

    Abstracts are intentionally left empty here — run enrich_crossref.py
    afterwards to fill them via the Crossref API (10.24963 prefix has ~100%
    coverage for 2024).
    """
    doi_added = pdf_added = 0

    for i, p in enumerate(papers):
        if p.get("year") != year or not p.get("authors"):
            continue
        venue_url = p.get("venue_url", "")
        m = _PROCEEDINGS_URL_RE.match(venue_url)
        if not m:
            continue
        paper_year, paper_num = m.group(1), m.group(2)

        doi = f"10.24963/ijcai.{paper_year}/{paper_num}"
        pdf = f"https://www.ijcai.org/proceedings/{paper_year}/{int(paper_num):04d}.pdf"

        if not dry_run:
            if not p.get("doi"):
                papers[i]["doi"] = doi
                doi_added += 1
            if not p.get("pdf_url"):
                papers[i]["pdf_url"] = pdf
                pdf_added += 1
        else:
            if not p.get("doi"):
                doi_added += 1
            if not p.get("pdf_url"):
                pdf_added += 1

    return {"doi_added": doi_added, "pdf_added": pdf_added, "abstract_added": 0}


PATCH_FNS = {
    "1977": patch_year_from_index_volumes,
    "1981": patch_year_from_pdf_ocr,
    "1983": patch_year_from_pdf_ocr,
    "1993": patch_year_from_pdf_ocr,
    "1997": patch_year_from_pdf_ocr,
    "2007": patch_year_from_pdf_url,
    "2009": patch_year_from_pdf_url,
    "2011": patch_year_from_doi,
    "2013": patch_year_from_index,
    "2015": patch_year_from_venue_url,
    "2016": patch_year_from_venue_url,
    "2017": patch_year_derive_pdf,
    "2018": patch_year_derive_pdf,
    "2019": patch_year_derive_pdf,
    "2020": patch_year_derive_pdf,
    "2021": patch_year_derive_pdf,
    "2022": patch_year_derive_pdf,
    "2024": patch_year_2024,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patch IJCAI legacy records from ijcai.org"
    )
    ap.add_argument(
        "--year", choices=sorted(PATCH_FNS),
        help="Process a single year (default: all)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing",
    )
    ap.add_argument(
        "--no-abstracts", action="store_true",
        help="Skip abstract scraping (PDF derivation only)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    legacy_path = LEGACY_DIR / "ijcai-legacy.jsonl.gz"
    if not legacy_path.exists():
        logger.error(f"Legacy file not found: {legacy_path}")
        sys.exit(1)

    papers = read_legacy(legacy_path)
    logger.info(f"Loaded {len(papers)} IJCAI legacy records")

    years = [args.year] if args.year else sorted(PATCH_FNS)
    grand_doi = grand_pdf = grand_abs = 0

    for year in years:
        yr_papers = [p for p in papers if p.get("year") == year and p.get("authors")]
        before_doi = sum(1 for p in yr_papers if p.get("doi"))
        before_pdf = sum(1 for p in yr_papers if p.get("pdf_url"))
        before_abs = sum(1 for p in yr_papers if p.get("abstract"))
        logger.info(
            f"\n--- IJCAI {year} ---  "
            f"({len(yr_papers)} papers, {before_doi} DOIs, {before_pdf} PDFs, {before_abs} abstracts)"
        )

        fn = PATCH_FNS[year]
        stats = fn(
            papers,
            year,
            scrape_abstracts=not args.no_abstracts,
            dry_run=args.dry_run,
        )

        mode = "[DRY RUN] " if args.dry_run else ""
        doi_added = stats.get("doi_added", 0)
        after_doi = before_doi + doi_added
        after_pdf = before_pdf + stats["pdf_added"]
        after_abs = before_abs + stats["abstract_added"]
        parts = []
        if doi_added:
            parts.append(f"+{doi_added} DOIs")
        parts += [f"+{stats['pdf_added']} PDFs", f"+{stats['abstract_added']} abstracts"]
        logger.info(f"  {mode}{', '.join(parts)}")
        logger.info(
            f"  {'Would be' if args.dry_run else 'Now'}: "
            f"{after_doi}/{len(yr_papers)} DOIs, "
            f"{after_pdf}/{len(yr_papers)} PDFs, "
            f"{after_abs}/{len(yr_papers)} abstracts"
        )
        grand_doi += doi_added
        grand_pdf += stats["pdf_added"]
        grand_abs += stats["abstract_added"]

    print()
    if not args.dry_run:
        write_legacy(legacy_path, papers, atomic=True)
        logger.info(f"Written back to {legacy_path.name}")

    logger.info(
        f"Total: +{grand_doi} DOIs, +{grand_pdf} PDFs, +{grand_abs} abstracts across {years}"
    )


if __name__ == "__main__":
    main()
