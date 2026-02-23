"""PMLR adapter: fetches proceedings metadata from mlresearch GitHub repos.

PMLR stores each volume as a Jekyll site in github.com/mlresearch/v{N}.
Paper metadata lives in _posts/*.md as YAML front matter.
Volume metadata lives in _config.yml.
"""

import re
import logging
from functools import partial
from pathlib import Path
from typing import Optional, Union

import yaml

from .common import make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json
from .http import fetch_with_retry as _fetch_with_retry, fetch_parallel
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com/mlresearch"


# Known PMLR volumes for major venues.
# Format: (volume_number, venue_shortname, year)
# This is the authoritative mapping — we can extend it or auto-discover.
KNOWN_VOLUMES = [
    # ICML
    (267, "ICML", "2025"),
    (235, "ICML", "2024"),
    (202, "ICML", "2023"),
    (162, "ICML", "2022"),
    (139, "ICML", "2021"),
    (119, "ICML", "2020"),
    (97, "ICML", "2019"),
    (80, "ICML", "2018"),
    (70, "ICML", "2017"),
    (48, "ICML", "2016"),
    (37, "ICML", "2015"),
    (32, "ICML", "2014"),
    (28, "ICML", "2013"),
    # AISTATS
    (258, "AISTATS", "2025"),
    (238, "AISTATS", "2024"),
    (206, "AISTATS", "2023"),
    (151, "AISTATS", "2022"),
    (130, "AISTATS", "2021"),
    (108, "AISTATS", "2020"),
    (89, "AISTATS", "2019"),
    # COLT
    (291, "COLT", "2025"),
    (247, "COLT", "2024"),
    (195, "COLT", "2023"),
    (178, "COLT", "2022"),
    (134, "COLT", "2021"),
    (125, "COLT", "2020"),
    (99, "COLT", "2019"),
    # AISTATS (legacy — fetched directly from PMLR; some years absent from DBLP)
    (84, "AISTATS", "2018"),
    (54, "AISTATS", "2017"),
    (51, "AISTATS", "2016"),
    (38, "AISTATS", "2015"),
    (33, "AISTATS", "2014"),
    (31, "AISTATS", "2013"),
    (22, "AISTATS", "2012"),
    (15, "AISTATS", "2011"),
    (9, "AISTATS", "2010"),
    (5, "AISTATS", "2009"),
    (2, "AISTATS", "2007"),
    # AISTATS reissue volumes (pre-PMLR proceedings, reissued 2022)
    ("r5", "AISTATS", "2005"),
    ("r4", "AISTATS", "2003"),
    ("r3", "AISTATS", "2001"),
    ("r2", "AISTATS", "1999"),
    ("r1", "AISTATS", "1997"),
    ("r0", "AISTATS", "1995"),
    # COLT (legacy — 2010 not in PMLR; 2011/2012 fetched directly from PMLR)
    (75, "COLT", "2018"),
    (65, "COLT", "2017"),
    (49, "COLT", "2016"),
    (40, "COLT", "2015"),
    (35, "COLT", "2014"),
    (30, "COLT", "2013"),
    (23, "COLT", "2012"),
    (19, "COLT", "2011"),
    # ACML (legacy — 2009 not on PMLR; 2010-2024 from PMLR)
    (260, "ACML", "2024"),
    (222, "ACML", "2023"),
    (189, "ACML", "2022"),
    (157, "ACML", "2021"),
    (129, "ACML", "2020"),
    (101, "ACML", "2019"),
    (95,  "ACML", "2018"),
    (77,  "ACML", "2017"),
    (63,  "ACML", "2016"),
    (45,  "ACML", "2015"),
    (39,  "ACML", "2014"),
    (29,  "ACML", "2013"),
    (25,  "ACML", "2012"),
    (20,  "ACML", "2011"),
    (13,  "ACML", "2010"),
    # UAI
    (286, "UAI", "2025"),
    (244, "UAI", "2024"),
    (216, "UAI", "2023"),
    (180, "UAI", "2022"),
    (161, "UAI", "2021"),
    (124, "UAI", "2020"),
    (115, "UAI", "2019"),
    # CoRL
    (305, "CoRL", "2025"),
    (270, "CoRL", "2024"),
    (229, "CoRL", "2023"),
    (205, "CoRL", "2022"),
    (164, "CoRL", "2021"),
    (155, "CoRL", "2020"),
    (100, "CoRL", "2019"),
    (87, "CoRL", "2018"),
    (78, "CoRL", "2017"),
    # MIDL
    (250, "MIDL", "2024"),
    (172, "MIDL", "2023"),
    # ALT (2017-2025 on PMLR; pre-2017 Springer LNAI via DBLP legacy)
    (272, "ALT", "2025"),
    (237, "ALT", "2024"),
    (201, "ALT", "2023"),
    (167, "ALT", "2022"),
    (132, "ALT", "2021"),
    (117, "ALT", "2020"),
    (98,  "ALT", "2019"),
    (83,  "ALT", "2018"),
    (76,  "ALT", "2017"),
]


def _volume_repo(volume: Union[int, str]) -> str:
    """Return the mlresearch GitHub repo slug for a volume identifier.

    Regular volumes (int or numeric str): 28 → 'v28'
    Reissue volumes (str starting with 'r'): 'r0' → 'r0'
    """
    s = str(volume)
    return s if s.startswith("r") else f"v{s}"


def parse_pmlr_post(content: str) -> Optional[dict]:
    """Parse YAML front matter from a PMLR _posts markdown file."""
    match = re.match(r"^---\n(.+?)\n---", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML: {e}")
        return None


def fetch_volume_config(volume: Union[int, str]) -> dict:
    """Fetch _config.yml for a PMLR volume."""
    url = f"{GITHUB_RAW}/{_volume_repo(volume)}/gh-pages/_config.yml"
    resp = _fetch_with_retry(url)
    return yaml.safe_load(resp.text)


def list_posts(volume: Union[int, str]) -> list[str]:
    """List all _posts/*.md files in a PMLR volume repo via GitHub API.

    Uses the Git Tree API instead of Contents API to handle repositories
    with more than 1000 files (GitHub Contents API limit).
    """
    repo = _volume_repo(volume)

    repo_url = f"{GITHUB_API}/repos/mlresearch/{repo}"
    repo_resp = _fetch_with_retry(repo_url)
    repo_data = repo_resp.json()
    default_branch = repo_data.get("default_branch", "gh-pages")

    branch_url = f"{GITHUB_API}/repos/mlresearch/{repo}/git/refs/heads/{default_branch}"
    branch_resp = _fetch_with_retry(branch_url)
    branch_data = branch_resp.json()
    sha = branch_data["object"]["sha"]

    # recursive tree avoids the 1000-item Contents API limit
    tree_url = f"{GITHUB_API}/repos/mlresearch/{repo}/git/trees/{sha}?recursive=1"
    tree_resp = _fetch_with_retry(tree_url)
    tree_data = tree_resp.json()

    posts = []
    for item in tree_data.get("tree", []):
        path = item.get("path", "")
        if path.startswith("_posts/") and path.endswith(".md"):
            posts.append(path.split("/")[-1])

    return posts


def fetch_post(volume: Union[int, str], filename: str) -> str:
    """Fetch raw content of a single post file."""
    url = f"{GITHUB_RAW}/{_volume_repo(volume)}/gh-pages/_posts/{filename}"
    resp = _fetch_with_retry(url)
    return resp.text


def fetch_and_parse_post(volume: Union[int, str], filename: str, venue: str, year: str, venue_name: str) -> Optional[dict]:
    """Fetch and parse a single post file, returning paper dict or None."""
    try:
        content = fetch_post(volume, filename)
        meta = parse_pmlr_post(content)
        if meta is None:
            return None

        authors = meta.get("author", [])
        if not authors:
            return None

        first_author = authors[0]
        title = meta.get("title", "")

        bkey = make_bibtex_key(
            first_author_family=first_author.get("family", ""),
            year=year,
            venue=venue.lower(),
            title=title,
        )

        openreview_id = meta.get("openreview", "")
        openreview_url = f"https://openreview.net/forum?id={openreview_id}" if openreview_id else ""

        paper_id = meta.get("id", "")
        vol_prefix = _volume_repo(volume)
        venue_url = f"https://proceedings.mlr.press/{vol_prefix}/{paper_id}.html" if paper_id else ""

        return {
            "bibtex_key": bkey,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue.lower(),
            "venue_name": venue_name,
            "volume": str(meta.get("volume", volume)),
            "pages": meta.get("page", ""),
            "abstract": meta.get("abstract", ""),
            "pdf_url": meta.get("pdf", ""),
            "venue_url": venue_url,
            "openreview_url": openreview_url,
            "code_url": "",
            "source": "pmlr",
            "source_id": f"{vol_prefix}/{paper_id}",
        }
    except Exception as e:
        logger.warning(f"Failed to process {filename}: {e}")
        return None


def process_volume(volume: Union[int, str], venue: str, year: str) -> list[dict]:
    """Process a single PMLR volume, returning a list of paper dicts."""
    logger.info(f"Processing PMLR {_volume_repo(volume)} ({venue} {year})")

    try:
        config = fetch_volume_config(volume)
    except Exception as e:
        logger.error(f"Failed to fetch config for v{volume}: {e}")
        config = {}

    venue_name = config.get("booktitle", f"Proceedings of {venue} {year}")
    editors = config.get("editor", [])

    try:
        post_files = list_posts(volume)
    except Exception as e:
        logger.error(f"Failed to list posts for v{volume}: {e}")
        return []

    logger.info(f"  Fetching {len(post_files)} papers concurrently...")
    _fetch = partial(fetch_and_parse_post, volume, venue=venue, year=year, venue_name=venue_name)
    results = fetch_parallel(
        post_files, _fetch, max_workers=20, default=None, progress_interval=100,
    )

    papers = [p for p in results.values() if p is not None]
    bibtex_keys = [p["bibtex_key"] for p in papers]

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    logger.info(f"  Processed {len(papers)} papers from {_volume_repo(volume)}")
    return papers


def fetch_all(
    volumes: Optional[list[tuple[int, str, str]]] = None,
    output_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
) -> dict[str, list[dict]]:
    """Fetch all specified PMLR volumes and write YAML output.

    Returns dict mapping venue-year to list of papers.
    """
    if volumes is None:
        volumes = KNOWN_VOLUMES

    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    for vol_num, venue, year in volumes:
        cache_key = f"pmlr-v{vol_num}"
        if cache is not None and not should_fetch(cache, cache_key, year):
            logger.info(f"Skipping PMLR v{vol_num} ({venue} {year}) — cached")
            continue

        venue_year = f"{venue.lower()}-{year}"
        papers = process_volume(vol_num, venue, year)

        if papers:
            all_papers[venue_year] = papers
            write_venue_json(venue.lower(), year, [normalize_paper(p) for p in papers], output_dir)
            if cache is not None:
                mark_fetched(cache, cache_key)

    return all_papers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Quick test: fetch a single small volume
    import argparse

    parser = argparse.ArgumentParser(description="Fetch PMLR proceedings")
    parser.add_argument("--volume", type=int, help="Specific volume to fetch")
    parser.add_argument("--venue", type=str, help="Venue shortname (e.g. ICML)")
    parser.add_argument("--year", type=str, help="Year")
    parser.add_argument("--all", action="store_true", help="Fetch all known volumes")
    parser.add_argument("--output", type=str, default="data/papers", help="Output directory")
    args = parser.parse_args()

    if args.volume and args.venue and args.year:
        papers = process_volume(args.volume, args.venue, args.year)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_venue_json(args.venue.lower(), args.year, [normalize_paper(p) for p in papers], output_dir)
        print(f"Wrote {len(papers)} papers to {out_path}")
    elif args.all:
        fetch_all(output_dir=Path(args.output))
    else:
        parser.print_help()
