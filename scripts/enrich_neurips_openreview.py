#!/usr/bin/env python3
"""Enrich NeurIPS data with OpenReview metadata, or fetch new years.

Two modes:
  - **Enrich** (2021-2024): Match existing NeurIPS papers by title to
    OpenReview records and fill in openreview_url (and code_url for 2021).
  - **Fetch** (2025+): Create full paper records from OpenReview when
    proceedings.neurips.cc has no data yet.

Usage:
    # Enrich existing years with OpenReview URLs
    python -m scripts.enrich_neurips_openreview --enrich

    # Fetch NeurIPS 2025 from OpenReview
    python -m scripts.enrich_neurips_openreview --fetch 2025

    # Do both
    python -m scripts.enrich_neurips_openreview --enrich --fetch 2025
"""

import argparse
import logging
import re
import time
import unicodedata
from pathlib import Path

import requests

from adapters.common import (
    make_bibtex_key,
    normalize_paper,
    parse_author_name,
    read_venue_json,
    resolve_bibtex_collisions,
    write_venue_json,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"

# (venue_id, api_version, track_label)
# track_label is used to tag papers from non-main tracks
NEURIPS_VENUES = {
    2025: [
        ("NeurIPS.cc/2025/Conference", "v2", "conference"),
        ("NeurIPS.cc/2025/Datasets_and_Benchmarks_Track", "v2", "datasets_and_benchmarks"),
        ("NeurIPS.cc/2025/Position_Paper_Track", "v2", "position"),
    ],
    2024: [
        ("NeurIPS.cc/2024/Conference", "v2", "conference"),
        ("NeurIPS.cc/2024/Datasets_and_Benchmarks_Track", "v2", "datasets_and_benchmarks"),
    ],
    2023: [
        ("NeurIPS.cc/2023/Conference", "v2", "conference"),
        ("NeurIPS.cc/2023/Track/Datasets_and_Benchmarks", "v2", "datasets_and_benchmarks"),
    ],
    2022: [
        ("NeurIPS.cc/2022/Conference", "v1", "conference"),
        ("NeurIPS.cc/2022/Track/Datasets_and_Benchmarks", "v1", "datasets_and_benchmarks"),
    ],
    2021: [
        ("NeurIPS.cc/2021/Conference", "v1", "conference"),
    ],
}

VENUE_NAME = "Advances in Neural Information Processing Systems"

API_BASES = {
    "v1": "https://api.openreview.net",
    "v2": "https://api2.openreview.net",
}


def _api_get(api_version: str, endpoint: str, params: dict) -> dict | None:
    """GET request to OpenReview API with retry/backoff."""
    base = API_BASES[api_version]
    url = f"{base}/{endpoint}"
    for attempt in range(6):
        if attempt > 0:
            wait = min(2 ** (attempt + 1), 60)
            logger.debug(f"  Retry {attempt}, waiting {wait}s...")
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, timeout=60, headers={
                "User-Agent": "ml-proceedings/1.0",
            })
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                logger.debug(f"  Rate limited (429)")
                continue
            else:
                logger.warning(f"  HTTP {r.status_code} from {url}")
                return None
        except requests.RequestException as e:
            logger.warning(f"  Request error: {e}")
            continue
    logger.error(f"  Failed after 6 retries: {url}")
    return None


def fetch_openreview_notes(venue_id: str, api_version: str) -> list[dict]:
    """Fetch all notes for a venue, handling pagination."""
    all_notes = []
    offset = 0
    while True:
        data = _api_get(api_version, "notes", {
            "content.venueid": venue_id,
            "limit": 1000,
            "offset": offset,
        })
        if not data or not data.get("notes"):
            break
        notes = data["notes"]
        all_notes.extend(notes)
        logger.info(f"    Fetched {len(all_notes)} notes so far...")
        if len(notes) < 1000:
            break
        offset += 1000
        time.sleep(1)  # be polite between pages
    return all_notes


def _is_accepted(note: dict, api_version: str) -> bool:
    """Check if a note represents an accepted paper (filter out 'Submitted')."""
    if api_version == "v2":
        venue_label = note.get("content", {}).get("venue", {}).get("value", "")
    else:
        venue_label = note.get("content", {}).get("venue", "")
    lower = venue_label.lower()
    return "submitted" not in lower and venue_label != ""


def _extract_field(content: dict, key: str, api_version: str):
    """Extract a field value, handling V1 vs V2 wrapping."""
    val = content.get(key)
    if val is None:
        return None
    if api_version == "v2" and isinstance(val, dict):
        return val.get("value")
    return val


def _normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching: lowercase, strip accents/punct."""
    # NFKD decompose, strip combining marks
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))
    title = title.lower()
    # strip everything except alphanumeric and space
    title = re.sub(r"[^a-z0-9 ]+", "", title)
    # collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _build_openreview_index(year: int) -> dict[str, dict]:
    """Fetch OpenReview notes for a year and build a title -> metadata index."""
    venues = NEURIPS_VENUES.get(year, [])
    index = {}  # normalized_title -> {forum_id, code_url}

    for venue_id, api_version, track in venues:
        logger.info(f"  Fetching {venue_id}...")
        notes = fetch_openreview_notes(venue_id, api_version)
        accepted = 0
        for note in notes:
            if not _is_accepted(note, api_version):
                continue
            accepted += 1
            content = note.get("content", {})
            title = _extract_field(content, "title", api_version) or ""
            norm_title = _normalize_title(title)
            if not norm_title:
                continue
            forum_id = note.get("forum", "")
            code = _extract_field(content, "code", api_version) or ""
            index[norm_title] = {
                "forum_id": forum_id,
                "openreview_url": f"https://openreview.net/forum?id={forum_id}" if forum_id else "",
                "code_url": code if code.startswith("http") else "",
            }
        logger.info(f"    {accepted} accepted papers from {venue_id}")
        time.sleep(2)  # pause between venue fetches

    return index


def enrich_year(year: int) -> None:
    """Enrich a single year of NeurIPS data with OpenReview URLs."""
    path = DATA_DIR / f"neurips-{year}.json.gz"
    if not path.exists():
        logger.warning(f"  No data file for NeurIPS {year}, skipping")
        return

    logger.info(f"Enriching NeurIPS {year}...")

    # Build OpenReview index
    or_index = _build_openreview_index(year)
    logger.info(f"  OpenReview index: {len(or_index)} papers")

    # Load existing data
    data = read_venue_json(path)
    papers = data["papers"]
    logger.info(f"  Existing papers: {len(papers)}")

    # Match and enrich
    matched = 0
    enriched_or = 0
    enriched_code = 0
    for paper in papers:
        norm_title = _normalize_title(paper.get("title", ""))
        or_data = or_index.get(norm_title)
        if not or_data:
            continue
        matched += 1
        if not paper.get("openreview_url") and or_data["openreview_url"]:
            paper["openreview_url"] = or_data["openreview_url"]
            enriched_or += 1
        if not paper.get("code_url") and or_data["code_url"]:
            paper["code_url"] = or_data["code_url"]
            enriched_code += 1

    logger.info(f"  Matched: {matched}/{len(papers)}")
    logger.info(f"  Enriched openreview_url: {enriched_or}")
    if enriched_code:
        logger.info(f"  Enriched code_url: {enriched_code}")

    # Write back
    write_venue_json("neurips", str(year), papers, DATA_DIR)


def _note_to_paper(note: dict, api_version: str, year: str) -> dict | None:
    """Convert an OpenReview note to our canonical paper dict."""
    content = note.get("content", {})

    title = _extract_field(content, "title", api_version) or ""
    if not title:
        return None

    author_names = _extract_field(content, "authors", api_version) or []
    authors = [parse_author_name(name) for name in author_names]
    if not authors:
        return None

    abstract = _extract_field(content, "abstract", api_version) or ""

    pdf_path = _extract_field(content, "pdf", api_version) or ""
    if not pdf_path:
        pdf_url = ""
    elif pdf_path.startswith("http"):
        pdf_url = pdf_path
    else:
        pdf_url = f"https://openreview.net{pdf_path}"

    forum_id = note.get("forum", "")
    openreview_url = f"https://openreview.net/forum?id={forum_id}" if forum_id else ""

    code_url = _extract_field(content, "code", api_version) or ""
    if not code_url.startswith("http"):
        code_url = ""

    first_author = authors[0]
    bkey = make_bibtex_key(
        first_author_family=first_author.get("family", ""),
        year=year,
        venue="neurips",
        title=title,
    )

    return {
        "bibtex_key": bkey,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": "neurips",
        "venue_name": VENUE_NAME,
        "volume": "",
        "pages": "",
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue_url": openreview_url,
        "doi": "",
        "openreview_url": openreview_url,
        "code_url": code_url,
        "source": "openreview",
        "source_id": forum_id,
    }


def fetch_year(year: int) -> None:
    """Fetch a full year of NeurIPS from OpenReview and write the data file."""
    venues = NEURIPS_VENUES.get(year)
    if not venues:
        logger.error(f"No OpenReview venue config for NeurIPS {year}")
        return

    logger.info(f"Fetching NeurIPS {year} from OpenReview...")

    all_papers = []
    bibtex_keys = []

    for venue_id, api_version, track in venues:
        logger.info(f"  Fetching {venue_id}...")
        notes = fetch_openreview_notes(venue_id, api_version)

        track_papers = 0
        for note in notes:
            if not _is_accepted(note, api_version):
                continue
            paper = _note_to_paper(note, api_version, str(year))
            if paper is None:
                continue
            all_papers.append(paper)
            bibtex_keys.append(paper["bibtex_key"])
            track_papers += 1

        logger.info(f"    {track_papers} accepted papers from {track}")
        time.sleep(2)

    # Resolve bibtex key collisions
    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(all_papers, resolved_keys):
        paper["bibtex_key"] = key

    # Normalize and write
    normalized = [normalize_paper(p) for p in all_papers]
    write_venue_json("neurips", str(year), normalized, DATA_DIR)
    logger.info(f"  Wrote {len(normalized)} papers to neurips-{year}.json.gz")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich NeurIPS data from OpenReview or fetch new years"
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Enrich existing NeurIPS 2021-2024 data with OpenReview URLs",
    )
    parser.add_argument(
        "--fetch", type=int, nargs="+", metavar="YEAR",
        help="Fetch full NeurIPS year(s) from OpenReview (e.g. --fetch 2025)",
    )
    args = parser.parse_args()

    if not args.enrich and not args.fetch:
        parser.print_help()
        return

    if args.fetch:
        for year in args.fetch:
            fetch_year(year)

    if args.enrich:
        for year in [2021, 2022, 2023, 2024]:
            enrich_year(year)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
