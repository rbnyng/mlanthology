"""NeurIPS adapter: scrapes proceedings from proceedings.neurips.cc.

Covers all NeurIPS/NIPS proceedings from 1987 to present.
Two data strategies:
- 1987-2019: Metadata.json files available per paper (structured JSON)
- 2020+: HTML scraping of paper detail pages
"""

import re
import logging
from html import unescape
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import (
    make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json,
    parse_author_name as _parse_author_name, strip_html as _strip_html,
)
from .http import fetch_with_retry as _fetch_with_retry
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

BASE_URL = "https://proceedings.neurips.cc"
VENUE_NAME = "Advances in Neural Information Processing Systems"

# All NeurIPS/NIPS years available on proceedings.neurips.cc
KNOWN_YEARS = [str(y) for y in range(1987, 2025)]


def _list_papers(year: str) -> list[dict]:
    """List all papers for a year from the proceedings index page.

    Returns list of dicts with keys: hash, title, authors_str, track.
    """
    url = f"{BASE_URL}/paper/{year}"
    resp = _fetch_with_retry(url)

    papers = []
    pattern = re.compile(
        r'<li\s+class="([^"]*)"[^>]*data-track="([^"]*)"[^>]*>\s*'
        r'<div class="paper-content">\s*'
        r'<a[^>]*href="/paper_files/paper/' + year + r'/hash/([a-f0-9]+)-Abstract[^"]*\.html"[^>]*>'
        r'([^<]+)</a>\s*'
        r'<span class="paper-authors">([^<]*)</span>',
        re.DOTALL,
    )

    for match in pattern.finditer(resp.text):
        css_class, track, hash_id, title, authors_str = match.groups()
        papers.append({
            "hash": hash_id,
            "title": unescape(title.strip()),
            "authors_str": unescape(authors_str.strip()),
            "track": track,
        })

    return papers


def _fetch_metadata_json(year: str, hash_id: str) -> Optional[dict]:
    """Fetch Metadata.json for a paper (available 1987-2019)."""
    url = f"{BASE_URL}/paper_files/paper/{year}/file/{hash_id}-Metadata.json"
    try:
        resp = _fetch_with_retry(url)
        return resp.json()
    except (FileNotFoundError, RuntimeError):
        return None


def _fetch_paper_detail(year: str, hash_id: str, track: str) -> Optional[dict]:
    """Fetch paper detail page and extract abstract + PDF URL (2020+)."""
    suffix = "Conference" if track == "conference" else "Datasets_and_Benchmarks_Track"
    url = f"{BASE_URL}/paper_files/paper/{year}/hash/{hash_id}-Abstract-{suffix}.html"
    try:
        resp = _fetch_with_retry(url)
    except (FileNotFoundError, RuntimeError):
        # Try without track suffix
        url = f"{BASE_URL}/paper_files/paper/{year}/hash/{hash_id}-Abstract.html"
        try:
            resp = _fetch_with_retry(url)
        except (FileNotFoundError, RuntimeError):
            return None

    abstract = ""
    abstract_match = re.search(
        r'class="paper-abstract"[^>]*>(.*?)</p>',
        resp.text,
        re.DOTALL,
    )
    if abstract_match:
        abstract = _strip_html(abstract_match.group(1)).strip()

    pdf_url = ""
    pdf_match = re.search(
        r"href='(/paper_files/paper/" + year + r"/file/" + hash_id + r"-Paper[^']*\.pdf)'",
        resp.text,
    )
    if pdf_match:
        pdf_url = f"{BASE_URL}{pdf_match.group(1)}"

    doi = ""
    doi_match = re.search(
        r'href="https?://doi\.org/([^"]+)"',
        resp.text,
    )
    if doi_match:
        doi = doi_match.group(1)

    pages = ""
    return {"abstract": abstract, "pdf_url": pdf_url, "pages": pages, "doi": doi}


def fetch_and_parse_paper(entry: dict, year: str, use_metadata_json: bool) -> Optional[dict]:
    """Fetch and parse a single paper, returning paper dict or None."""
    try:
        hash_id = entry["hash"]
        title = entry["title"]
        authors = [_parse_author_name(a) for a in entry["authors_str"].split(",")]
        if not authors:
            return None

        abstract = ""
        pdf_url = ""
        pages = ""
        doi = ""

        if use_metadata_json:
            meta = _fetch_metadata_json(year, hash_id)
            if meta:
                abstract = meta.get("abstract") or ""
                page_first = meta.get("page_first", "")
                page_last = meta.get("page_last", "")
                if page_first and page_last:
                    pages = f"{page_first}-{page_last}"
                # Construct PDF URL
                pdf_url = f"{BASE_URL}/paper_files/paper/{year}/file/{hash_id}-Paper.pdf"
            # Fall back to HTML detail page if metadata had no abstract
            if not abstract:
                detail = _fetch_paper_detail(year, hash_id, entry.get("track", ""))
                if detail:
                    if detail["abstract"]:
                        abstract = detail["abstract"]
                    if not pdf_url and detail["pdf_url"]:
                        pdf_url = detail["pdf_url"]
                    if detail.get("doi"):
                        doi = detail["doi"]
        else:
            detail = _fetch_paper_detail(year, hash_id, entry["track"])
            if detail:
                abstract = detail["abstract"]
                pdf_url = detail["pdf_url"]
                pages = detail.get("pages", "")
                doi = detail.get("doi", "")

        first_author = authors[0]
        bkey = make_bibtex_key(
            first_author_family=first_author.get("family", ""),
            year=year,
            venue="neurips",
            title=title,
        )

        # Construct venue URL — the plain -Abstract.html form works for all
        # years (1987-present); the track-suffixed forms (-Abstract-Conference.html
        # etc.) are only valid for 2021+ and only for the specific track.
        venue_url = f"{BASE_URL}/paper_files/paper/{year}/hash/{hash_id}-Abstract.html"

        return {
            "bibtex_key": bkey,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": "neurips",
            "venue_name": VENUE_NAME,
            "volume": "",
            "pages": pages,
            "doi": doi,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "venue_url": venue_url,
            "openreview_url": "",
            "code_url": "",
            "source": "neurips",
            "source_id": hash_id,
        }
    except Exception as e:
        logger.warning(f"Failed to process paper {entry.get('hash', 'unknown')}: {e}")
        return None


def process_year(year: str) -> list[dict]:
    """Process a single year of NeurIPS proceedings."""
    logger.info(f"Processing NeurIPS {year}")

    entries = _list_papers(year)
    if not entries:
        logger.warning(f"  No papers found for NeurIPS {year}")
        return []

    logger.info(f"  Found {len(entries)} papers on index page")

    use_metadata_json = int(year) <= 2019

    papers = []
    bibtex_keys = []

    # Fetch papers concurrently for much faster processing
    # Use 20 workers to balance speed with avoiding rate limits
    logger.info(f"  Fetching {len(entries)} papers concurrently...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(fetch_and_parse_paper, entry, year, use_metadata_json): entry
            for entry in entries
        }

        # Collect results as they complete
        processed = 0
        for future in as_completed(future_to_entry):
            processed += 1
            if processed % 100 == 0:
                logger.info(f"  Progress: {processed}/{len(entries)} papers")

            paper = future.result()
            if paper is not None:
                papers.append(paper)
                bibtex_keys.append(paper["bibtex_key"])

    # Resolve bibtex key collisions
    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    logger.info(f"  Processed {len(papers)} papers for NeurIPS {year}")
    return papers


def fetch_all(
    years: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
) -> dict[str, list[dict]]:
    """Fetch all specified NeurIPS years and write YAML output."""
    if years is None:
        years = KNOWN_YEARS

    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    for year in years:
        cache_key = f"neurips-{year}"
        if cache is not None and not should_fetch(cache, cache_key, year):
            logger.info(f"Skipping NeurIPS {year} — cached")
            continue

        venue_year = f"neurips-{year}"
        papers = process_year(year)

        if papers:
            all_papers[venue_year] = papers
            write_venue_json("neurips", year, [normalize_paper(p) for p in papers], output_dir)
            if cache is not None:
                mark_fetched(cache, cache_key)

    return all_papers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch NeurIPS proceedings")
    parser.add_argument("--year", type=str, help="Specific year to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all known years")
    parser.add_argument("--output", type=str, default="data/papers", help="Output directory")
    args = parser.parse_args()

    if args.year:
        papers = process_year(args.year)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_venue_json("neurips", args.year, [normalize_paper(p) for p in papers], output_dir)
        print(f"Wrote {len(papers)} papers to {out_path}")
    elif args.all:
        fetch_all(output_dir=Path(args.output))
    else:
        parser.print_help()
