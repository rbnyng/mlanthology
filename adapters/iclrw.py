"""ICLR Workshops adapter: fetches workshop papers from OpenReview via raw HTTP.

Uses the OpenReview REST API directly (no openreview-py dependency) to avoid
fragile binary dependency chains (pylatexenc, pycryptodome, etc.).

Workshop coverage:
  - 2019-2023: OpenReview API v1 (api.openreview.net)
  - 2024+:     OpenReview API v2 (api2.openreview.net)

Workshop papers are identified by content.venueid matching the workshop's
venue ID (ICLR.cc/{year}/Workshop/{code}).  For v1, an additional venue-label
check filters out papers still under review ("Submitted to …").

Usage::
    from adapters.iclrw import fetch_year, YEARS
    papers = fetch_year("2024")   # returns list[dict] across all 2024 workshops
"""

import logging
import time
import urllib.parse
from pathlib import Path
from typing import Optional

from .common import make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json, parse_author_name
from .http import fetch_with_retry

logger = logging.getLogger(__name__)

ORAPI_V1 = "https://api.openreview.net"
ORAPI_V2 = "https://api2.openreview.net"

# Years available on OpenReview.  2019 is the first year workshops appeared.
# v2 API went live for ICLR 2024+; older years use v1.
YEARS = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]

VENUE_FULLNAME = "International Conference on Learning Representations Workshops"


def _api(year: str) -> str:
    return ORAPI_V2 if int(year) >= 2024 else ORAPI_V1


def _list_workshop_ids(year: str) -> list[tuple[str, str]]:
    """Return list of (venue_id, workshop_short_name) for all workshops in year.

    Uses depth-3 group IDs only: ICLR.cc/{year}/Workshop/{code}.

    v2 API supports ?prefix=; v1 API requires ?parent= instead.
    """
    base = _api(year)
    parent = f"ICLR.cc/{year}/Workshop"

    if int(year) >= 2024:
        # v2: prefix parameter (includes trailing slash to avoid partial matches)
        prefix = parent + "/"
        url = f"{base}/groups?prefix={urllib.parse.quote(prefix, safe='/:.')}&limit=500"
    else:
        # v1: parent parameter lists direct children
        url = f"{base}/groups?parent={urllib.parse.quote(parent, safe='/:.')}&limit=500"

    resp = fetch_with_retry(url, headers={"User-Agent": "mlanthology/1.0"}, max_retries=5)
    data = resp.json()
    groups = data.get("groups", [])
    results = []
    for g in groups:
        gid = g.get("id", "")
        if gid.count("/") == 3:  # ICLR.cc/YEAR/Workshop/CODE — exclude sub-groups
            code = gid.split("/")[-1]
            results.append((gid, code))
    return results


def _fetch_notes_v2(venue_id: str) -> list[dict]:
    """Fetch all notes for a v2 workshop venue."""
    all_notes = []
    offset = 0
    limit = 1000
    vid_enc = urllib.parse.quote(venue_id, safe="/:.")
    while True:
        url = (
            f"{ORAPI_V2}/notes"
            f"?content.venueid={vid_enc}"
            f"&limit={limit}&offset={offset}"
        )
        resp = fetch_with_retry(url, headers={"User-Agent": "mlanthology/1.0"}, max_retries=5)
        data = resp.json()
        batch = data.get("notes", [])
        all_notes.extend(batch)
        total = data.get("count", 0)
        offset += len(batch)
        if offset >= total or not batch:
            break
        time.sleep(0.5)
    return all_notes


def _fetch_notes_v1(venue_id: str) -> list[dict]:
    """Fetch notes for a v1 workshop venue, filtering to accepted-only."""
    all_notes = []
    offset = 0
    limit = 1000
    vid_enc = urllib.parse.quote(venue_id, safe="/:.")
    while True:
        url = (
            f"{ORAPI_V1}/notes"
            f"?content.venueid={vid_enc}"
            f"&limit={limit}&offset={offset}"
        )
        resp = fetch_with_retry(url, headers={"User-Agent": "mlanthology/1.0"}, max_retries=5)
        data = resp.json()
        batch = data.get("notes", [])
        # Filter out papers still under review ("Submitted to …")
        accepted = []
        for n in batch:
            venue_label = n.get("content", {}).get("venue", "")
            if "submitted" not in venue_label.lower() and venue_label != "":
                accepted.append(n)
        all_notes.extend(accepted)
        total = data.get("count", 0)
        offset += len(batch)
        if offset >= total or not batch:
            break
        time.sleep(0.5)
    return all_notes


def _parse_note_v2(note: dict, year: str, venue_name: str) -> Optional[dict]:
    """Parse a v2 API note dict into canonical paper dict."""
    c = note.get("content", {})
    title = c.get("title", {}).get("value", "")
    if not title:
        return None
    author_names = c.get("authors", {}).get("value", [])
    authors = [parse_author_name(n) for n in author_names]
    if not authors:
        return None
    abstract = c.get("abstract", {}).get("value", "")
    pdf_path = c.get("pdf", {}).get("value", "")
    if not pdf_path:
        pdf_url = ""
    elif pdf_path.startswith("http"):
        pdf_url = pdf_path
    else:
        pdf_url = f"https://openreview.net{pdf_path}"
    forum = note.get("forum", "")
    openreview_url = f"https://openreview.net/forum?id={forum}" if forum else ""
    bkey = make_bibtex_key(
        first_author_family=authors[0].get("family", ""),
        year=year,
        venue="iclrw",
        title=title,
    )
    return {
        "bibtex_key": bkey,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": "iclrw",
        "venue_name": venue_name,
        "volume": "",
        "pages": "",
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue_url": openreview_url,
        "openreview_url": openreview_url,
        "code_url": c.get("code", {}).get("value", ""),
        "source": "openreview",
        "source_id": forum,
    }


def _parse_note_v1(note: dict, year: str, venue_name: str) -> Optional[dict]:
    """Parse a v1 API note dict into canonical paper dict."""
    c = note.get("content", {})
    title = c.get("title", "")
    if not title:
        return None
    author_names = c.get("authors", [])
    authors = [parse_author_name(n) for n in author_names]
    if not authors:
        return None
    abstract = c.get("abstract", "")
    pdf_path = c.get("pdf", "")
    if not pdf_path:
        pdf_url = ""
    elif pdf_path.startswith("http"):
        pdf_url = pdf_path
    else:
        pdf_url = f"https://openreview.net{pdf_path}"
    forum = note.get("forum", "")
    openreview_url = f"https://openreview.net/forum?id={forum}" if forum else ""
    bkey = make_bibtex_key(
        first_author_family=authors[0].get("family", ""),
        year=year,
        venue="iclrw",
        title=title,
    )
    return {
        "bibtex_key": bkey,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": "iclrw",
        "venue_name": venue_name,
        "volume": "",
        "pages": "",
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue_url": openreview_url,
        "openreview_url": openreview_url,
        "code_url": c.get("code", ""),
        "source": "openreview",
        "source_id": forum,
    }


def fetch_year(year: str) -> list[dict]:
    """Fetch all ICLR workshop papers for a given year.

    Returns a flat list across all workshops for that year.
    Workshop name is embedded in venue_name as
    "ICLR {year} Workshop: {code}".
    """
    is_v2 = int(year) >= 2024
    logger.info(f"Listing ICLR {year} workshop venues (API {'v2' if is_v2 else 'v1'})...")

    workshop_ids = _list_workshop_ids(year)
    logger.info(f"  Found {len(workshop_ids)} workshops for {year}")

    all_papers: list[dict] = []
    all_bkeys: list[str] = []

    for venue_id, code in workshop_ids:
        workshop_name = f"ICLR {year} Workshops: {code}"
        logger.info(f"  Fetching {venue_id}...")
        try:
            if is_v2:
                notes = _fetch_notes_v2(venue_id)
                parse_fn = _parse_note_v2
            else:
                notes = _fetch_notes_v1(venue_id)
                parse_fn = _parse_note_v1

            for note in notes:
                paper = parse_fn(note, year, workshop_name)
                if paper is not None:
                    all_papers.append(paper)
                    all_bkeys.append(paper["bibtex_key"])

            logger.info(f"    -> {len(notes)} notes, {sum(1 for p in all_papers if p['venue_name'] == workshop_name)} parsed")
            time.sleep(1)  # polite delay between workshops
        except Exception as e:
            logger.warning(f"  Failed to fetch {venue_id}: {e}")

    # Resolve bibtex key collisions across all workshops
    resolved = resolve_bibtex_collisions(all_bkeys)
    for paper, key in zip(all_papers, resolved):
        paper["bibtex_key"] = key

    return all_papers


def fetch_all_years(years: Optional[list[str]] = None) -> dict[str, list[dict]]:
    """Fetch ICLR workshop papers for all specified years.

    Returns dict mapping year to list of papers.
    """
    if years is None:
        years = YEARS
    result = {}
    for year in years:
        papers = fetch_year(year)
        logger.info(f"ICLR {year} workshops: {len(papers)} total papers")
        result[year] = papers
    return result
