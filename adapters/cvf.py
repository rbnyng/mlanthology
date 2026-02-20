"""CVF Open Access adapter: scrapes papers from openaccess.thecvf.com.

Covers:
- CVPR (2013-2025)
- ICCV (2013-2025, biennial)
- WACV (2020-2025)

Two-phase approach:
1. Listing page (CONF_YEAR?day=all) provides title, authors, PDF URL,
   supplemental URL, arXiv URL, and BibTeX for all papers.
2. Individual paper HTML pages provide abstracts (fetched in parallel).
"""

import re
import logging
from html import unescape
from pathlib import Path
from typing import Optional

from .common import (
    make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json,
    parse_author_name as _parse_author_name, strip_html as _strip_html,
)
from .http import fetch_with_retry as _fetch_with_retry, fetch_parallel
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

BASE_URL = "https://openaccess.thecvf.com"

# (conference, year) tuples
# CVPR: annual since 2013
# ICCV: biennial (odd years) since 2013
# WACV: annual since 2020
# Latest first so --quick picks the most recent year per conference
KNOWN_CONFERENCES = []
for y in range(2025, 2012, -1):
    KNOWN_CONFERENCES.append(("CVPR", str(y)))
for y in range(2025, 2012, -2):
    KNOWN_CONFERENCES.append(("ICCV", str(y)))
for y in range(2025, 2019, -1):
    KNOWN_CONFERENCES.append(("WACV", str(y)))

VENUE_NAMES = {
    "CVPR": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "ICCV": "Proceedings of the IEEE/CVF International Conference on Computer Vision",
    "WACV": "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision",
}


def _make_absolute_url(path: str) -> str:
    """Ensure a URL path is absolute."""
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{BASE_URL}{path}"


def _list_papers(conference: str, year: str) -> list[dict]:
    """List all papers from a CVF conference listing page.

    Returns list of dicts with: title, authors, pdf_url, supp_url,
    arxiv_url, bibtex, paper_html_path, pages.
    """
    url = f"{BASE_URL}/{conference}{year}?day=all"
    resp = _fetch_with_retry(url)
    html = resp.text

    papers = []

    blocks = re.split(r'<dt\s+class="ptitle">', html)

    for block in blocks[1:]:
        title_match = re.search(r'<a\s+href="([^"]*)"[^>]*>([^<]+)</a>', block)
        if not title_match:
            continue
        paper_html_path = title_match.group(1)
        if not paper_html_path.startswith("/") and not paper_html_path.startswith("http"):
            paper_html_path = f"/{paper_html_path}"
        title = unescape(title_match.group(2).strip())

        dd_match = re.search(r'<dd>(.*?)</dd>', block, re.DOTALL)
        if not dd_match:
            continue
        authors_block = dd_match.group(1)
        author_names = [
            name.strip() for name in
            re.findall(r'<a\s+[^>]*>([^<]+)</a>', authors_block)
            if name.strip() and not name.strip().startswith('[')
        ]
        if not author_names:
            continue

        authors = [_parse_author_name(name) for name in author_names]

        remaining = block[dd_match.end():]

        pdf_url = ""
        pdf_match = re.search(r'<a\s+href="([^"]*\.pdf)"[^>]*>\s*pdf\s*</a>', remaining)
        if pdf_match:
            pdf_url = _make_absolute_url(pdf_match.group(1))

        supp_url = ""
        supp_match = re.search(r'<a\s+href="([^"]*)"[^>]*>\s*supp\s*</a>', remaining)
        if supp_match:
            supp_url = _make_absolute_url(supp_match.group(1))

        arxiv_url = ""
        arxiv_match = re.search(r'<a\s+href="(https?://arxiv\.org/[^"]*)"', remaining)
        if arxiv_match:
            arxiv_url = arxiv_match.group(1)

        pages = ""
        pages_match = re.search(r'pages\s*=\s*\{([^}]+)\}', remaining)
        if pages_match:
            pages = pages_match.group(1).strip()

        papers.append({
            "title": title,
            "authors": authors,
            "pdf_url": pdf_url,
            "supp_url": supp_url,
            "arxiv_url": arxiv_url,
            "paper_html_path": paper_html_path,
            "pages": pages,
        })

    return papers


def _fetch_abstract(paper_html_path: str) -> str:
    """Fetch abstract from an individual paper's HTML page."""
    url = _make_absolute_url(paper_html_path)
    try:
        resp = _fetch_with_retry(url)
    except (FileNotFoundError, RuntimeError):
        return ""

    match = re.search(
        r'<div\s+id="abstract"[^>]*>(.*?)</div>',
        resp.text, re.DOTALL,
    )
    if match:
        return _strip_html(match.group(1))

    # fallback: text after "Abstract" heading
    match = re.search(
        r'(?:Abstract|ABSTRACT)\s*</?\w+[^>]*>\s*(.*?)(?:<div|<h[23]|<br\s*/?\s*><br)',
        resp.text, re.DOTALL,
    )
    if match:
        return _strip_html(match.group(1))

    return ""


def _fetch_abstracts_parallel(
    paper_html_paths: list[str],
    max_workers: int = 10,
) -> dict[str, str]:
    """Fetch abstracts for multiple papers in parallel."""
    return fetch_parallel(paper_html_paths, _fetch_abstract, max_workers=max_workers)


def process_conference_year(
    conference: str,
    year: str,
    fetch_abstracts: bool = True,
) -> list[dict]:
    """Process a single conference-year from CVF Open Access.

    Args:
        conference: Conference name (CVPR, ICCV, WACV).
        year: Year string.
        fetch_abstracts: If True, fetch abstracts from individual pages.
    """
    venue_slug = conference.lower()
    venue_name = VENUE_NAMES.get(conference, f"Proceedings of {conference}")

    logger.info(f"Processing {conference} {year}")

    entries = _list_papers(conference, year)
    if not entries:
        logger.warning(f"  No papers found for {conference} {year}")
        return []

    logger.info(f"  Found {len(entries)} papers on listing page")

    abstracts: dict[str, str] = {}
    if fetch_abstracts:
        paths = [e["paper_html_path"] for e in entries]
        logger.info(f"  Fetching abstracts in parallel ({len(paths)} papers)...")
        abstracts = _fetch_abstracts_parallel(paths)

    papers = []
    bibtex_keys = []

    for entry in entries:
        title = entry["title"]
        authors = entry["authors"]
        if not authors:
            continue

        first_author = authors[0]
        bkey = make_bibtex_key(
            first_author_family=first_author.get("family", ""),
            year=year,
            venue=venue_slug,
            title=title,
        )
        bibtex_keys.append(bkey)

        paper_path = entry["paper_html_path"]
        venue_url = _make_absolute_url(paper_path)

        paper = {
            "bibtex_key": bkey,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue_slug,
            "venue_name": venue_name,
            "volume": "",
            "pages": entry["pages"],
            "abstract": abstracts.get(paper_path, ""),
            "pdf_url": entry["pdf_url"],
            "venue_url": venue_url,
            "openreview_url": "",
            "arxiv_url": entry["arxiv_url"],
            "code_url": "",
            "source": "cvf",
            "source_id": paper_path,
        }
        papers.append(paper)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    logger.info(f"  Processed {len(papers)} papers for {conference} {year}")
    return papers


def fetch_all(
    conferences: Optional[list[tuple[str, str]]] = None,
    output_dir: Optional[Path] = None,
    fetch_abstracts: bool = True,
    cache: Optional[dict] = None,
) -> dict[str, list[dict]]:
    """Fetch all specified CVF conference-years and write YAML output.

    Args:
        conferences: List of (conference, year) tuples.
        output_dir: Output directory for YAML files.
        fetch_abstracts: If True (default), fetch abstracts from paper pages.
        cache: Cache dict for skip logic. None = no caching.
    """
    if conferences is None:
        conferences = KNOWN_CONFERENCES

    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    for conference, year in conferences:
        venue_slug = conference.lower()
        cache_key = f"{venue_slug}-{year}"

        if cache is not None and not should_fetch(cache, cache_key, year):
            logger.info(f"Skipping {conference} {year} — cached")
            continue

        papers = process_conference_year(
            conference, year, fetch_abstracts=fetch_abstracts,
        )

        if papers:
            venue_year = f"{venue_slug}-{year}"
            all_papers[venue_year] = papers
            write_venue_json(venue_slug, year, [normalize_paper(p) for p in papers], output_dir)
            if cache is not None:
                mark_fetched(cache, cache_key)

    return all_papers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch CVF Open Access proceedings")
    parser.add_argument("--conference", type=str, choices=["CVPR", "ICCV", "WACV"],
                        help="Specific conference to fetch")
    parser.add_argument("--year", type=str, help="Specific year to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all known conferences")
    parser.add_argument("--no-abstracts", action="store_true",
                        help="Skip fetching abstracts from paper detail pages")
    parser.add_argument("--output", type=str, default="data/papers", help="Output directory")
    args = parser.parse_args()

    if args.conference and args.year:
        papers = process_conference_year(
            args.conference, args.year,
            fetch_abstracts=not args.no_abstracts,
        )
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        venue_slug = args.conference.lower()
        out_path = write_venue_json(venue_slug, args.year, [normalize_paper(p) for p in papers], output_dir)
        print(f"Wrote {len(papers)} papers to {out_path}")
    elif args.all:
        fetch_all(
            output_dir=Path(args.output),
            fetch_abstracts=not args.no_abstracts,
        )
    else:
        parser.print_help()
