"""JMLR adapter: scrapes papers from jmlr.org and data.mlr.press (DMLR).

Covers:
- JMLR main journal (volumes 1-26, 2000-2025)
- MLOSS papers (tagged subset of JMLR, detected from volume page markup)
- DMLR (data.mlr.press, volumes 1-2)

Two metadata strategies for JMLR:
- Volume 6+: BibTeX files available for clean structured metadata
- Volume 1-5: HTML volume page parsing (no .bib files available)

Abstracts require fetching individual paper pages (optional, slow).
DMLR abstracts are inline on volume pages (no extra requests).
"""

import re
import logging
from functools import partial
from pathlib import Path
from typing import Optional

from .common import (
    make_bibtex_key, resolve_bibtex_collisions, normalize_paper,
    write_venue_json, read_venue_json,
    parse_author_name as _parse_author_name,
    strip_html as _strip_html,
)
from .http import fetch_with_retry as _fetch_with_retry, fetch_parallel
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

BASE_URL = "https://jmlr.org"
DMLR_BASE_URL = "https://data.mlr.press"
VENUE_NAME = "Journal of Machine Learning Research"
DMLR_VENUE_NAME = "Journal of Data-centric Machine Learning Research"

# JMLR volumes: (volume_number, year)
# Volume 1 started in 2000. Some volumes span two years.
KNOWN_VOLUMES = [
    (1, "2000"), (2, "2001"), (3, "2002"), (4, "2003"), (5, "2004"),
    (6, "2005"), (7, "2006"), (8, "2007"), (9, "2008"), (10, "2009"),
    (11, "2010"), (12, "2011"), (13, "2012"), (14, "2013"), (15, "2014"),
    (16, "2015"), (17, "2016"), (18, "2017"), (19, "2018"), (20, "2019"),
    (21, "2020"), (22, "2021"), (23, "2022"), (24, "2023"), (25, "2024"),
    (26, "2025"),
]

# DMLR volumes
KNOWN_DMLR_VOLUMES = [
    (1, "2024"),
    (2, "2025"),
]

# BibTeX files available starting from volume 6
BIBTEX_MIN_VOLUME = 6


def _parse_bibtex(bib_text: str) -> Optional[dict]:
    """Parse a single BibTeX entry into a dict."""
    fields = {}
    for match in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}", bib_text, re.DOTALL):
        key, value = match.groups()
        fields[key.lower()] = value.strip()

    if not fields.get("title"):
        return None

    return fields


def _list_papers_from_volume(volume: int) -> list[dict]:
    """Parse JMLR volume listing page to extract paper entries.

    Returns list of dicts with: paper_id, title, authors_str, issue, pages,
    year, is_mloss, code_url, pdf_url.
    """
    url = f"{BASE_URL}/papers/v{volume}/"
    resp = _fetch_with_retry(url)
    html = resp.text

    dl_blocks = re.findall(r"<dl>(.*?)</dl>", html, re.DOTALL)

    papers = []
    for block in dl_blocks:
        title_match = re.search(r"<dt>\s*(.*?)(?:</dt>|\s*<dd>)", block, re.DOTALL)
        if not title_match:
            continue
        title = title_match.group(1)
        title = re.sub(r"&#\d+;", " ", title)
        title = _strip_html(title).strip()
        if not title:
            continue

        authors_match = re.search(
            r"<b>\s*<i>(.*?)</i>\s*</b>|<b>\s*<i>(.*?)</b>\s*</i>",
            block, re.DOTALL,
        )
        if not authors_match:
            continue
        authors_str = _strip_html(authors_match.group(1) or authors_match.group(2))

        # citation: handles various dash encodings across volume eras
        _dash = r"(?:&minus;|--?|[\u2212\u2013\u2014])"

        cite_match = re.search(
            r";\s*\((\w+)\):" + r"(\d+)\s*" + _dash + r"\s*(\d+),\s*(\d{4})\.",
            block,
        )
        if cite_match:
            issue = cite_match.group(1)
            page_start = cite_match.group(2)
            page_end = cite_match.group(3)
            year = cite_match.group(4)
        else:
            cite_match = re.search(
                r";\s*\d+\(\w+\):" + r"(\d+)\s*" + _dash + r"\s*(\d+),\s*(\d{4})\.",
                block,
            )
            if cite_match:
                issue = ""
                page_start = cite_match.group(1)
                page_end = cite_match.group(2)
                year = cite_match.group(3)
            else:
                continue

        pages = f"{page_start}-{page_end}" if page_start and page_end else ""

        abs_match = re.search(
            r"href=['\"]?\s*(?:https?://[^/]*jmlr\.org)?(?:/papers/v\d+/)?(\S+?)\.html\s*['\"]?\s*>\s*(?:\[)?abs",
            block, re.DOTALL,
        )
        if not abs_match:
            continue
        paper_id = abs_match.group(1)
        if "/" in paper_id:
            paper_id = paper_id.rsplit("/", 1)[-1]

        is_mloss = "Machine Learning Open Source Software" in block

        code_url = ""
        code_match = re.search(r'href=["\']([^"\']+)["\']>\s*code\s*</a>', block)
        if code_match:
            code_url = code_match.group(1)

        pdf_url = ""
        pdf_match = re.search(
            r"href=['\"]?\s*(\S+\.pdf)\s*['\"]?[^>]*>\s*(?:\[)?pdf", block, re.DOTALL
        )
        if pdf_match:
            pdf_path = pdf_match.group(1)
            if pdf_path.startswith("http"):
                pdf_url = pdf_path.replace("http://www.jmlr.org", "https://jmlr.org")
                pdf_url = pdf_url.replace("https://www.jmlr.org", "https://jmlr.org")
            elif pdf_path.startswith("/"):
                pdf_url = f"{BASE_URL}{pdf_path}"
            else:
                pdf_url = f"{BASE_URL}/papers/v{volume}/{pdf_path}"

        papers.append({
            "paper_id": paper_id,
            "title": title,
            "authors_str": authors_str,
            "issue": issue,
            "pages": pages,
            "year": year,
            "is_mloss": is_mloss,
            "code_url": code_url,
            "pdf_url": pdf_url,
        })

    return papers


def _fetch_abstract(volume: int, paper_id: str) -> str:
    """Fetch abstract from a JMLR paper detail page."""
    url = f"{BASE_URL}/papers/v{volume}/{paper_id}.html"
    try:
        resp = _fetch_with_retry(url)
    except (FileNotFoundError, RuntimeError):
        return ""

    match = re.search(
        r'<p\s+class="abstract">\s*(.*?)\s*</p>',
        resp.text, re.DOTALL,
    )
    if match:
        return _strip_html(match.group(1))

    match = re.search(
        r"<h3>\s*Abstract\s*</h3>\s*<p[^>]*>\s*(.*?)\s*</p>",
        resp.text, re.DOTALL,
    )
    if match:
        return _strip_html(match.group(1))

    match = re.search(
        r"<h3>\s*Abstract\s*</h3>\s*(.*?)(?:<h3>|<hr|<p\s*class|</div>|\[abs\])",
        resp.text, re.DOTALL,
    )
    if match:
        return _strip_html(match.group(1))

    return ""


def _fetch_abstracts_parallel(volume: int, paper_ids: list[str], max_workers: int = 8) -> dict[str, str]:
    """Fetch abstracts for multiple papers in parallel."""
    return fetch_parallel(
        paper_ids,
        partial(_fetch_abstract, volume),
        max_workers=max_workers,
        progress_interval=50,
    )


def process_volume(volume: int, fetch_abstracts: bool = True) -> list[dict]:
    """Process a single JMLR volume.

    Args:
        volume: Volume number (1-26).
        fetch_abstracts: If True, fetch abstract from each paper's detail page.
            Defaults to True. Abstracts are fetched in parallel.
    """
    logger.info(f"Processing JMLR volume {volume}")

    entries = _list_papers_from_volume(volume)
    if not entries:
        logger.warning(f"  No papers found for JMLR volume {volume}")
        return []

    logger.info(f"  Found {len(entries)} papers on volume page")

    abstracts: dict[str, str] = {}
    if fetch_abstracts:
        paper_ids = [e["paper_id"] for e in entries]
        logger.info(f"  Fetching abstracts in parallel ({len(paper_ids)} papers)...")
        abstracts = _fetch_abstracts_parallel(volume, paper_ids)

    papers = []
    bibtex_keys = []

    for entry in entries:
        paper_id = entry["paper_id"]
        title = entry["title"]
        year = entry["year"]

        authors = [_parse_author_name(a) for a in entry["authors_str"].split(",")]
        if not authors:
            continue

        venue_tag = "jmlr"
        venue_name = VENUE_NAME
        if entry["is_mloss"]:
            venue_tag = "mloss"
            venue_name = f"{VENUE_NAME} (MLOSS)"

        first_author = authors[0]
        bkey = make_bibtex_key(
            first_author_family=first_author.get("family", ""),
            year=year,
            venue="jmlr",
            title=title,
        )
        bibtex_keys.append(bkey)

        paper = {
            "bibtex_key": bkey,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue_tag,
            "venue_name": venue_name,
            "volume": str(volume),
            "number": entry.get("issue", ""),
            "pages": entry["pages"],
            "abstract": abstracts.get(paper_id, ""),
            "pdf_url": entry["pdf_url"],
            "venue_url": f"{BASE_URL}/papers/v{volume}/{paper_id}.html",
            "openreview_url": "",
            "code_url": entry["code_url"],
            "source": "jmlr",
            "source_id": paper_id,
        }
        papers.append(paper)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    mloss_count = sum(1 for p in papers if p["venue"] == "mloss")
    logger.info(f"  Processed {len(papers)} papers for JMLR v{volume}"
                + (f" ({mloss_count} MLOSS)" if mloss_count else ""))
    return papers


def _list_dmlr_papers(volume: int) -> list[dict]:
    """Parse DMLR volume listing page. Abstracts are inline."""
    vol_str = f"{volume:02d}"
    url = f"{DMLR_BASE_URL}/volumes/{vol_str}.html"
    resp = _fetch_with_retry(url)
    html = resp.text

    papers = []

    li_blocks = re.findall(
        r'<li\s+class="list-group-item">(.*?)</li>',
        html, re.DOTALL,
    )

    for block in li_blocks:
        title_match = re.search(r"<dt>\s*(.*?)\s*</dt>", block, re.DOTALL)
        if not title_match:
            continue
        title = _strip_html(title_match.group(1))
        if not title:
            continue

        author_names = re.findall(r"<b><i>([^<]+)</i></b>", block)
        authors = [_parse_author_name(name) for name in author_names]
        if not authors:
            continue

        cite_match = re.search(
            r"\((\d+)\):(\d+)\s*(?:&minus;|[-\u2212\u2013\u2014])\s*(\d+),\s*(\d{4})\.",
            block,
        )
        if not cite_match:
            continue
        issue = cite_match.group(1)
        pages = f"{cite_match.group(2)}-{cite_match.group(3)}"
        year = cite_match.group(4)

        abstract = ""
        abstract_match = re.search(
            r"<summary>\s*Abstract\s*</summary>\s*<I>\s*<p>(.*?)</p>",
            block, re.DOTALL,
        )
        if abstract_match:
            abstract = _strip_html(abstract_match.group(1))

        pdf_url = ""
        pdf_match = re.search(r'href="([^"]+)"\s*>\s*\[PDF\]', block)
        if pdf_match:
            pdf_path = pdf_match.group(1)
            if pdf_path.startswith("/"):
                pdf_url = f"{DMLR_BASE_URL}{pdf_path}"
            else:
                pdf_url = pdf_path

        paper_num = issue

        papers.append({
            "paper_id": f"v{vol_str}-{paper_num}",
            "title": title,
            "authors": authors,
            "issue": issue,
            "pages": pages,
            "year": year,
            "abstract": abstract,
            "pdf_url": pdf_url,
        })

    return papers


def process_dmlr_volume(volume: int) -> list[dict]:
    """Process a single DMLR volume."""
    logger.info(f"Processing DMLR volume {volume}")

    entries = _list_dmlr_papers(volume)
    if not entries:
        logger.warning(f"  No papers found for DMLR volume {volume}")
        return []

    logger.info(f"  Found {len(entries)} papers")

    papers = []
    bibtex_keys = []

    for entry in entries:
        first_author = entry["authors"][0]
        year = entry["year"]
        bkey = make_bibtex_key(
            first_author_family=first_author.get("family", ""),
            year=year,
            venue="dmlr",
            title=entry["title"],
        )
        bibtex_keys.append(bkey)

        paper = {
            "bibtex_key": bkey,
            "title": entry["title"],
            "authors": entry["authors"],
            "year": year,
            "venue": "dmlr",
            "venue_name": DMLR_VENUE_NAME,
            "volume": str(volume),
            "number": entry.get("issue", ""),
            "pages": entry["pages"],
            "abstract": entry["abstract"],
            "pdf_url": entry["pdf_url"],
            "venue_url": f"{DMLR_BASE_URL}/v{volume}/{entry['paper_id']}.html",
            "openreview_url": "",
            "code_url": "",
            "source": "dmlr",
            "source_id": entry["paper_id"],
        }
        papers.append(paper)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    logger.info(f"  Processed {len(papers)} papers for DMLR v{volume}")
    return papers


def fetch_all(
    volumes: Optional[list[tuple[int, str]]] = None,
    dmlr_volumes: Optional[list[tuple[int, str]]] = None,
    output_dir: Optional[Path] = None,
    fetch_abstracts: bool = True,
    cache: Optional[dict] = None,
) -> dict[str, list[dict]]:
    """Fetch all specified JMLR/DMLR volumes and write YAML output.

    Args:
        volumes: List of (volume_number, year) tuples for JMLR.
        dmlr_volumes: List of (volume_number, year) tuples for DMLR.
        output_dir: Output directory for YAML files.
        fetch_abstracts: If True (default), fetch abstracts in parallel.
        cache: Cache dict for skip logic. None = no caching.
    """
    if volumes is None:
        volumes = KNOWN_VOLUMES
    if dmlr_volumes is None:
        dmlr_volumes = KNOWN_DMLR_VOLUMES

    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    for vol_num, default_year in volumes:
        cache_key = f"jmlr-v{vol_num}"
        if cache is not None and not should_fetch(cache, cache_key, default_year):
            logger.info(f"Skipping JMLR v{vol_num} — cached")
            continue

        papers = process_volume(vol_num, fetch_abstracts=fetch_abstracts)
        if papers:
            years_in_vol = sorted(set(p["year"] for p in papers))
            if len(years_in_vol) == 1:
                venue_year = f"jmlr-{years_in_vol[0]}"
                all_papers[venue_year] = papers
                _write_venue(venue_year, "jmlr", years_in_vol[0], vol_num, papers, output_dir)
            else:
                for year in years_in_vol:
                    year_papers = [p for p in papers if p["year"] == year]
                    venue_year = f"jmlr-{year}"
                    if venue_year in all_papers:
                        all_papers[venue_year].extend(year_papers)
                    else:
                        all_papers[venue_year] = year_papers
                    _write_venue(venue_year, "jmlr", year, vol_num, year_papers, output_dir,
                                append=venue_year in all_papers and len(all_papers[venue_year]) > len(year_papers))

            if cache is not None:
                mark_fetched(cache, cache_key)

    for vol_num, default_year in dmlr_volumes:
        cache_key = f"dmlr-v{vol_num}"
        if cache is not None and not should_fetch(cache, cache_key, default_year):
            logger.info(f"Skipping DMLR v{vol_num} — cached")
            continue

        papers = process_dmlr_volume(vol_num)
        if papers:
            venue_year = f"dmlr-{default_year}"
            all_papers[venue_year] = papers
            _write_venue(venue_year, "dmlr", default_year, vol_num, papers, output_dir)
            if cache is not None:
                mark_fetched(cache, cache_key)

    return all_papers


def _write_venue(
    venue_year: str,
    venue: str,
    year: str,
    volume: int,
    papers: list[dict],
    output_dir: Path,
    append: bool = False,
) -> None:
    """Write a JSON file for a venue-year, optionally appending to existing."""
    out_path = output_dir / f"{venue_year}.json.gz"

    if append and out_path.exists():
        existing = read_venue_json(out_path)
        existing_papers = existing.get("papers", [])
        all_paper_data = existing_papers + [normalize_paper(p) for p in papers]
    else:
        all_paper_data = [normalize_paper(p) for p in papers]

    write_venue_json(venue, year, all_paper_data, output_dir, filename=venue_year)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch JMLR/DMLR proceedings")
    parser.add_argument("--volume", type=int, help="Specific JMLR volume to fetch")
    parser.add_argument("--dmlr-volume", type=int, help="Specific DMLR volume to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all known volumes")
    parser.add_argument("--no-abstracts", action="store_true",
                        help="Skip fetching abstracts from paper detail pages")
    parser.add_argument("--output", type=str, default="data/papers", help="Output directory")
    args = parser.parse_args()

    if args.volume:
        papers = process_volume(args.volume, fetch_abstracts=not args.no_abstracts)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        if papers:
            year = papers[0]["year"]
            out_path = write_venue_json("jmlr", year, [normalize_paper(p) for p in papers], output_dir)
            mloss_count = sum(1 for p in papers if p["venue"] == "mloss")
            print(f"Wrote {len(papers)} papers ({mloss_count} MLOSS) to {out_path}")
    elif args.dmlr_volume:
        papers = process_dmlr_volume(args.dmlr_volume)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        if papers:
            year = papers[0]["year"]
            out_path = write_venue_json("dmlr", year, [normalize_paper(p) for p in papers], output_dir)
            print(f"Wrote {len(papers)} papers to {out_path}")
    elif args.all:
        fetch_all(output_dir=Path(args.output), fetch_abstracts=not args.no_abstracts)
    else:
        parser.print_help()
