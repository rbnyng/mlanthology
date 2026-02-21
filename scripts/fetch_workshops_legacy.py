#!/usr/bin/env python3
"""One-time fetch of NeurIPS and ICML workshop papers from OpenReview.

Writes legacy JSONL files to data/legacy/:
  - neuripsw-legacy.jsonl.gz  (2019-2024)
  - icmlw-legacy.jsonl.gz     (2018-2024)

Usage:
    python -m scripts.fetch_workshops_legacy
    python -m scripts.fetch_workshops_legacy --venue neuripsw
    python -m scripts.fetch_workshops_legacy --venue icmlw
"""

import argparse
import logging
import time
import urllib.parse
from pathlib import Path

from adapters.common import (
    make_bibtex_key,
    normalize_paper,
    parse_author_name,
    resolve_bibtex_collisions,
)
from adapters.http import fetch_with_retry
from scripts.utils import LEGACY_DIR, write_legacy

logger = logging.getLogger(__name__)

ORAPI_V1 = "https://api.openreview.net"
ORAPI_V2 = "https://api2.openreview.net"

UA = {"User-Agent": "mlanthology/1.0"}

# ---------------------------------------------------------------------------
# Venue configuration
# ---------------------------------------------------------------------------

VENUES = {
    "neuripsw": {
        "conf_prefix": "NeurIPS.cc",
        "years": [str(y) for y in range(2019, 2025)],
        "v2_from": 2023,  # 2019-2022 papers on V1 only, 2023+ on V2
        "venue_slug": "neuripsw",
        "venue_name_tpl": "NeurIPS {year} Workshops: {code}",
    },
    "icmlw": {
        "conf_prefix": "ICML.cc",
        "years": [str(y) for y in range(2018, 2025)],
        "v2_from": 2023,  # 2018-2022 papers on V1 only, 2023+ on V2
        "venue_slug": "icmlw",
        "venue_name_tpl": "ICML {year} Workshops: {code}",
    },
}


# ---------------------------------------------------------------------------
# OpenReview API helpers
# ---------------------------------------------------------------------------

def _api_base(year: int, v2_from: int) -> str:
    return ORAPI_V2 if year >= v2_from else ORAPI_V1


def _is_v2(year: int, v2_from: int) -> bool:
    return year >= v2_from


def _list_workshop_ids(conf_prefix: str, year: str, v2_from: int) -> list[tuple[str, str]]:
    """Return list of (venue_id, workshop_code) for all workshops in a year.

    Always uses V2 API for group discovery (works for all years), even
    though paper notes may need V1 for older years.
    """
    parent = f"{conf_prefix}/{year}/Workshop"
    prefix = parent + "/"

    # Try V2 prefix first (works for all years)
    url = f"{ORAPI_V2}/groups?prefix={urllib.parse.quote(prefix, safe='/:.')}&limit=500"
    resp = fetch_with_retry(url, headers=UA, max_retries=6, return_none_on_404=True)

    if resp is None:
        # Fall back to V1 parent query
        url = f"{ORAPI_V1}/groups?parent={urllib.parse.quote(parent, safe='/:.')}&limit=500"
        resp = fetch_with_retry(url, headers=UA, max_retries=6)

    data = resp.json()
    results = []
    for g in data.get("groups", []):
        gid = g.get("id", "")
        if gid.count("/") == 3:  # Conf.cc/YEAR/Workshop/CODE
            code = gid.split("/")[-1]
            results.append((gid, code))
    return results


def _fetch_notes(venue_id: str, year: int, v2_from: int) -> list[dict]:
    """Fetch all notes for a workshop venue, handling pagination."""
    base = _api_base(year, v2_from)
    vid_enc = urllib.parse.quote(venue_id, safe="/:.&=")
    all_notes = []
    offset = 0
    while True:
        url = f"{base}/notes?content.venueid={vid_enc}&limit=1000&offset={offset}"
        resp = fetch_with_retry(url, headers=UA, max_retries=6, return_none_on_404=True)
        if resp is None:
            break
        data = resp.json()
        batch = data.get("notes", [])

        if not _is_v2(year, v2_from):
            # V1: filter out "Submitted to ..." papers
            batch = [
                n for n in batch
                if "submitted" not in n.get("content", {}).get("venue", "").lower()
                and n.get("content", {}).get("venue", "") != ""
            ]

        all_notes.extend(batch)
        total = data.get("count", 0)
        offset += len(data.get("notes", []))  # advance by raw count, not filtered
        if offset >= total or not data.get("notes"):
            break
        time.sleep(0.5)
    return all_notes


def _parse_note(note: dict, year: str, venue_slug: str, venue_name: str, v2: bool) -> dict | None:
    """Parse an OpenReview note into a canonical paper dict."""
    c = note.get("content", {})

    if v2:
        title = c.get("title", {}).get("value", "")
        author_names = c.get("authors", {}).get("value", [])
        abstract = c.get("abstract", {}).get("value", "")
        pdf_path = c.get("pdf", {}).get("value", "")
        code_url = c.get("code", {}).get("value", "")
    else:
        title = c.get("title", "")
        author_names = c.get("authors", [])
        abstract = c.get("abstract", "")
        pdf_path = c.get("pdf", "")
        code_url = c.get("code", "")

    if not title:
        return None
    authors = [parse_author_name(n) for n in author_names]
    if not authors:
        return None

    if not pdf_path:
        pdf_url = ""
    elif pdf_path.startswith("http"):
        pdf_url = pdf_path
    else:
        pdf_url = f"https://openreview.net{pdf_path}"

    if not code_url or not code_url.startswith("http"):
        code_url = ""

    forum = note.get("forum", "")
    openreview_url = f"https://openreview.net/forum?id={forum}" if forum else ""

    bkey = make_bibtex_key(
        first_author_family=authors[0].get("family", ""),
        year=year,
        venue=venue_slug,
        title=title,
    )

    return {
        "bibtex_key": bkey,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue_slug,
        "venue_name": venue_name,
        "volume": "",
        "pages": "",
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue_url": openreview_url,
        "doi": "",
        "openreview_url": openreview_url,
        "code_url": code_url,
        "source": "openreview",
        "source_id": forum,
    }


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------

def fetch_venue(venue_key: str) -> list[dict]:
    """Fetch all workshop papers for a venue across all years."""
    cfg = VENUES[venue_key]
    all_papers = []
    all_bkeys = []

    for year in cfg["years"]:
        yr = int(year)
        v2_from = cfg["v2_from"]
        v2 = _is_v2(yr, v2_from)

        logger.info(f"Listing {cfg['conf_prefix']} {year} workshops...")
        workshop_ids = _list_workshop_ids(cfg["conf_prefix"], year, v2_from)
        logger.info(f"  Found {len(workshop_ids)} workshops")

        year_count = 0
        for venue_id, code in workshop_ids:
            workshop_name = cfg["venue_name_tpl"].format(year=year, code=code)
            logger.info(f"  Fetching {venue_id}...")
            try:
                notes = _fetch_notes(venue_id, yr, v2_from)
                for note in notes:
                    paper = _parse_note(note, year, cfg["venue_slug"], workshop_name, v2)
                    if paper is not None:
                        all_papers.append(paper)
                        all_bkeys.append(paper["bibtex_key"])
                        year_count += 1
                if notes:
                    logger.info(f"    -> {len(notes)} papers")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"  Failed to fetch {venue_id}: {e}")

        logger.info(f"  {cfg['conf_prefix']} {year} workshops total: {year_count} papers")

    # Resolve bibtex key collisions across all years
    resolved = resolve_bibtex_collisions(all_bkeys)
    for paper, key in zip(all_papers, resolved):
        paper["bibtex_key"] = key

    # Normalize
    return [normalize_paper(p) for p in all_papers]


def main():
    parser = argparse.ArgumentParser(description="Fetch NeurIPS/ICML workshops from OpenReview")
    parser.add_argument("--venue", choices=["neuripsw", "icmlw"], help="Fetch a single venue")
    args = parser.parse_args()

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    venues_to_fetch = [args.venue] if args.venue else ["neuripsw", "icmlw"]

    for venue_key in venues_to_fetch:
        papers = fetch_venue(venue_key)
        out_path = LEGACY_DIR / f"{venue_key}-legacy.jsonl.gz"
        write_legacy(out_path, papers)
        logger.info(f"Wrote {len(papers)} papers to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
