"""ECVA adapter: scrapes ECCV papers from ecva.net/papers.php.

ECCV is biennial on even years since 2018 (2018, 2020, 2022, 2024, 2026, …).
KNOWN_YEARS is computed dynamically from the conference cadence so no manual
update is needed when a new edition is published — the adapter simply tries
each year and skips ones that aren't on the index page yet.

Two-phase approach:
1. Index page (ecva.net/papers.php) provides title, authors, and PDF URL,
   grouped by year in collapsible accordion sections.
2. Individual paper pages provide abstracts (fetched in parallel).
"""

import re
import datetime
import logging
from html import unescape
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import (
    make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json,
    strip_html as _strip_html, parse_author_name,
)
from .http import fetch_with_retry as _fetch_with_retry
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ecva.net"
VENUE_NAME = "Proceedings of the European Conference on Computer Vision (ECCV)"

# ECCV first appeared on ecva.net in 2018 (even years, biennial)
_ECCV_START = 2018
_ECCV_STEP = 2


def _eccv_years(max_year: Optional[int] = None) -> list[str]:
    """Return all ECCV years from 2018 up through max_year (inclusive).

    Defaults to current calendar year + 1 so a just-published edition is
    picked up without any code change.  The adapter handles missing years
    gracefully (returns an empty list when the section isn't on the page yet).
    """
    if max_year is None:
        max_year = datetime.date.today().year + 1
    return [str(y) for y in range(_ECCV_START, max_year + 1, _ECCV_STEP)]


# Exported so fetch_all.py can show the count before running
KNOWN_YEARS = _eccv_years()


def _parse_author_name(name: str) -> dict:
    """Split a name into given/family components, stripping * and superscript markers.

    Handles:
    - Asterisk corresponding-author markers: "Smith*" -> "Smith"
    - Unicode superscript affiliation numbers: "Smith ¹²" -> "Smith"
      (U+00B2 ², U+00B3 ³, U+00B9 ¹, U+2070-U+2079 ⁰⁴⁵⁶⁷⁸⁹)
    """
    name = name.replace("*", "")
    name = re.sub(r"[\u00b2\u00b3\u00b9\u2070-\u2079]+", "", name).strip()
    return parse_author_name(name)


def _parse_authors(authors_raw: str) -> list[dict]:
    """Parse an author string, handling both page formats.

    2018 format (bibtex-style):  "Last, First and Last, First and ..."
    2020+ format (comma-separated): "First Last*, First Last ¹, ..."

    Detection: if the text before the first ' and ' contains a comma,
    it's almost certainly bibtex Last-First format.
    """
    first_and = authors_raw.find(" and ")
    if first_and > 0 and "," in authors_raw[:first_and]:
        # Bibtex format
        parts = re.split(r"\s+and\s+", authors_raw)
        authors = []
        for part in parts:
            part = part.strip()
            if "," in part:
                family, given = part.split(",", 1)
                authors.append({"given": given.strip(), "family": family.strip()})
            else:
                authors.append(_parse_author_name(part))
        return authors
    else:
        # Comma-separated format
        return [_parse_author_name(a) for a in authors_raw.split(",") if a.strip()]


def _list_papers(year: str) -> list[dict]:
    """Fetch ecva.net/papers.php and extract all paper entries for a year.

    Returns list of dicts with keys: detail_url, title, authors_raw,
    pdf_url, source_id.
    """
    url = f"{BASE_URL}/papers.php"
    resp = _fetch_with_retry(url)
    html = resp.text

    section_pattern = re.compile(
        rf"ECCV\s+{year}\s+Papers(.*?)(?=ECCV\s+\d{{4}}\s+Papers|$)",
        re.DOTALL | re.IGNORECASE,
    )
    m = section_pattern.search(html)
    if not m:
        logger.warning(f"No ECCV {year} section found on {url}")
        return []

    section = m.group(1)

    blocks = re.split(r'<dt\s+class="ptitle">', section, flags=re.IGNORECASE)[1:]

    papers = []
    for block in blocks:
        # match href with single-quoted, double-quoted, or unquoted paths
        # anchored on papers/eccv_ to avoid grabbing the Springer DOI link
        title_match = re.search(
            r'<a\s+href=(?:"(papers/eccv_[^"]+)"|\'(papers/eccv_[^\']+)\'|(papers/eccv_\S+))\s*>'
            r"\s*(.*?)\s*</a>",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not title_match:
            continue

        detail_rel = title_match.group(1) or title_match.group(2) or title_match.group(3)
        title = unescape(re.sub(r"\s+", " ", title_match.group(4)).strip())
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        detail_url = f"{BASE_URL}/{detail_rel.lstrip('/')}"

        dd_contents = re.findall(r"<dd>(.*?)</dd>", block, re.DOTALL)
        if not dd_contents:
            continue

        authors_raw = _strip_html(dd_contents[0]).strip()

        pdf_url = ""
        doi = ""
        if len(dd_contents) > 1:
            links_html = dd_contents[1]
            pdf_match = re.search(
                r"href=['\"]([^'\"]+(?<!-supp)\.pdf)['\"]", links_html
            )
            if pdf_match:
                pdf_rel = pdf_match.group(1)
                pdf_url = f"{BASE_URL}/{pdf_rel.lstrip('/')}"

            springer_match = re.search(
                r'href=["\']?(https://link\.springer\.com/chapter/(10\.[^"\'>\s]+))["\']?',
                links_html,
            )
            if springer_match:
                doi = springer_match.group(2)

        source_id = ""
        if pdf_url:
            fname_match = re.search(r"/(\d+)\.pdf$", pdf_url)
            if fname_match:
                source_id = fname_match.group(1)
            else:
                stem_match = re.search(r"/([^/]+)\.pdf$", pdf_url)
                if stem_match:
                    source_id = stem_match.group(1)

        papers.append({
            "detail_url": detail_url,
            "title": title,
            "authors_raw": authors_raw,
            "pdf_url": pdf_url,
            "doi": doi,
            "source_id": source_id,
        })

    return papers


def _fetch_abstract(detail_url: str) -> str:
    """Fetch an individual paper page and extract the abstract."""
    try:
        resp = _fetch_with_retry(detail_url)
    except (FileNotFoundError, RuntimeError):
        return ""

    m = re.search(r'<div\s+id="abstract">(.*?)</div>', resp.text, re.DOTALL)
    if not m:
        return ""

    abstract = _strip_html(m.group(1)).strip()
    abstract = abstract.strip('\u201c\u201d"').strip()
    return abstract


def process_year(year: str) -> list[dict]:
    """Fetch and parse all ECCV papers for a given year."""
    logger.info(f"Processing ECCV {year}")

    entries = _list_papers(year)
    if not entries:
        logger.warning(f"  No papers found for ECCV {year}")
        return []

    logger.info(f"  Found {len(entries)} papers, fetching abstracts...")

    papers = []
    bibtex_keys = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_entry = {
            executor.submit(_fetch_abstract, entry["detail_url"]): entry
            for entry in entries
        }

        processed = 0
        for future in as_completed(future_to_entry):
            processed += 1
            if processed % 100 == 0:
                logger.info(f"  Progress: {processed}/{len(entries)}")

            entry = future_to_entry[future]
            abstract = future.result()

            authors = _parse_authors(entry["authors_raw"])
            if not authors:
                continue

            first_author = authors[0]
            bkey = make_bibtex_key(
                first_author_family=first_author.get("family", ""),
                year=year,
                venue="eccv",
                title=entry["title"],
            )

            papers.append({
                "bibtex_key": bkey,
                "title": entry["title"],
                "authors": authors,
                "year": year,
                "venue": "eccv",
                "venue_name": VENUE_NAME,
                "volume": "",
                "pages": "",
                "abstract": abstract,
                "pdf_url": entry["pdf_url"],
                "venue_url": entry["detail_url"],
                "doi": entry.get("doi", ""),
                "openreview_url": "",
                "code_url": "",
                "source": "ecva",
                "source_id": entry["source_id"],
            })
            bibtex_keys.append(bkey)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    logger.info(f"  Processed {len(papers)} papers for ECCV {year}")
    return papers


def fetch_all(
    years: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
    max_year: Optional[int] = None,
) -> dict[str, list[dict]]:
    """Fetch all specified ECCV years and write YAML output.

    If years is None, generates the list dynamically from conference cadence
    up to max_year (defaults to current year + 1).  Missing years on the
    index page are silently skipped, so passing a future year is safe.
    """
    if years is None:
        years = _eccv_years(max_year)

    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    for year in years:
        cache_key = f"eccv-{year}"
        if cache is not None and not should_fetch(cache, cache_key, year):
            logger.info(f"Skipping ECCV {year} — cached")
            continue

        papers = process_year(year)

        if papers:
            all_papers[f"eccv-{year}"] = papers
            write_venue_json("eccv", year, [normalize_paper(p) for p in papers], output_dir)
            if cache is not None:
                mark_fetched(cache, cache_key)

    return all_papers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch ECCV proceedings from ECVA")
    parser.add_argument("--year", type=str, help="Specific year to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all known years")
    parser.add_argument("--max-year", type=int, default=None,
                        help="Upper bound year (inclusive) when using --all")
    parser.add_argument("--output", type=str, default="data/papers", help="Output directory")
    args = parser.parse_args()

    if args.year:
        papers = process_year(args.year)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_venue_json("eccv", args.year, [normalize_paper(p) for p in papers], output_dir)
        print(f"Wrote {len(papers)} papers to {out_path}")
    elif args.all:
        fetch_all(output_dir=Path(args.output), max_year=args.max_year)
    else:
        parser.print_help()
