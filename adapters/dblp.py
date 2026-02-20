"""DBLP backlog adapter: fills historical gaps in venue coverage.

Uses the DBLP search API with table-of-contents (toc) facet queries to fetch
bibliographic metadata for conference proceedings.  Intended as a fill-in
adapter: by default it skips any venue-year whose output file already
exists (produced by a higher-quality primary adapter).

No abstracts are available from DBLP.  The adapter produces records with the
standard schema but with empty abstract and pdf_url fields.

Key quirks handled:
- NeurIPS was rebranded "nips→neurips" starting in 2020 in DBLP filenames.
- ECCV is split into many volumes per year (up to 89 in 2024).
- Single-author papers: DBLP returns authors.author as a dict, not a list.
- Author disambiguation suffixes: DBLP appends " 0001" etc. to names — stripped.
- DBLP titles sometimes end with a period — stripped.

Usage (standalone):
    python adapters/dblp.py --venue cvpr --max-year 2012
    python adapters/dblp.py --all --max-year 2015
"""

import gzip
import json
import re
import time
import datetime
import logging
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .common import make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json, parse_author_name
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

DBLP_API = "https://dblp.org/search/publ/api"
DBLP_DB = "https://dblp.org/db"

_HEADERS = {
    "User-Agent": "mlanthology/1.0 (https://github.com/rbnyng/mlanthology; research use)"
}

# thread-safe rate limiter — DBLP enforces ~1 req/sec
_rate_lock = threading.Lock()
_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL = 1.5  # seconds between requests

# venues and their DBLP config (start = earliest reliable year)
DBLP_VENUES: dict[str, dict] = {
    # NeurIPS is intentionally excluded: neurips.py covers 1987-present with
    # abstracts and PDFs from proceedings.neurips.cc — strictly better data.
    "icml": {
        "key": "conf/icml",
        "start": 1980,
        "name": "International Conference on Machine Learning",
    },
    "iclr": {
        "key": "conf/iclr",
        "start": 2013,
        "name": "International Conference on Learning Representations",
    },
    "cvpr": {
        "key": "conf/cvpr",
        "start": 1983,
        "name": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    },
    "iccv": {
        "key": "conf/iccv",
        "start": 1987,
        "name": "IEEE/CVF International Conference on Computer Vision",
    },
    "eccv": {
        "key": "conf/eccv",
        "start": 1990,
        "name": "European Conference on Computer Vision",
    },
    "wacv": {
        "key": "conf/wacv",
        "start": 2012,
        "name": "IEEE/CVF Winter Conference on Applications of Computer Vision",
    },
    "aistats": {
        "key": "conf/aistats",
        "start": 2007,
        "name": "International Conference on Artificial Intelligence and Statistics",
    },
    "uai": {
        "key": "conf/uai",
        "start": 1985,
        "name": "Conference on Uncertainty in Artificial Intelligence",
    },
    "colt": {
        "key": "conf/colt",
        "start": 1988,
        "name": "Annual Conference on Computational Learning Theory",
    },
    "corl": {
        "key": "conf/corl",
        "start": 2017,
        "name": "Conference on Robot Learning",
    },
    "aaai": {
        "key": "conf/aaai",
        "start": 1980,
        "name": "AAAI Conference on Artificial Intelligence",
    },
    "ijcai": {
        "key": "conf/ijcai",
        "start": 1969,
        "name": "International Joint Conference on Artificial Intelligence",
    },
    "alt": {
        # 1990-2016 Springer LNAI; 2017+ on PMLR (fetched separately).
        # DBLP legacy is capped at max_year=2016 in build_alt_legacy.py.
        "key": "conf/alt",
        "start": 1990,
        "name": "International Conference on Algorithmic Learning Theory",
    },
    "ecml": {
        "key": "conf/ecml",
        "start": 1993,
        "name": "European Conference on Machine Learning",
    },
    "ecmlpkdd": {
        # From 2008 ECML merged with PKDD.  The shared conf/pkdd key also
        # contains workshop stems (pkdd{year}-w*) and satellite events
        # (ial{year}, maclean{year}, …).  The stem_filter keeps only the
        # numbered main-conference volumes pkdd{year}-1 … pkdd{year}-N.
        "key": "conf/pkdd",
        "start": 2008,
        "name": "European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases",
        "stem_filter": r"pkdd\d{4}-\d+$",
    },
    "iccvw": {
        "key": "conf/iccvw",
        "start": 2007,
        "name": "IEEE/CVF International Conference on Computer Vision Workshops",
    },
    "eccvw": {
        # Workshop volumes live under the same conf/eccv/ DBLP key as the main
        # conference.  Pre-2020 stems use eccv2018w1 (no hyphen); 2020+ use
        # eccv2020-w1.  The filter w\d matches both forms while excluding main
        # conference stems like eccv2020-1 and satellite events like eccv2004hci.
        "key": "conf/eccv",
        "start": 2010,
        "name": "European Conference on Computer Vision Workshops",
        "stem_filter": r"w\d",
    },
    "cvprw": {
        # Workshop volumes share the conf/cvpr DBLP key with the main conference.
        # Stems are cvprw{year} for almost all years; 2022 uses cvpr2022w.
        # The filter `w` matches all workshop stems while excluding main-conference
        # stems (cvpr2024, etc.) and the unrelated medsam2024 satellite.
        "key": "conf/cvpr",
        "start": 2003,
        "name": "IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops",
        "stem_filter": r"w",
        "venue_type": "workshop",
    },
}

# venues that mix main + workshop papers under one DBLP key
WORKSHOP_OVERRIDES: dict[str, dict] = {
    "cvpr": {
        "slug": "cvprw",
        "name": "IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops",
    },
    "iccv": {
        "slug": "iccvw",
        "name": "IEEE/CVF International Conference on Computer Vision Workshops",
    },
    "wacv": {
        "slug": "wacvw",
        "name": "IEEE/CVF Winter Conference on Applications of Computer Vision Workshops",
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch_with_retry(url: str, max_retries: int = 4) -> Optional[requests.Response]:
    """Fetch URL with exponential backoff, honoring DBLP rate limits.

    This is intentionally separate from http.fetch_with_retry() because DBLP
    requires a thread-safe global rate limiter (_rate_lock / _last_request_time)
    to stay under its 1 req/sec policy, and it honours the Retry-After header
    on 429 responses — pushing the global timestamp forward so sibling threads
    also back off.  Returns None on 404 or unrecoverable error.
    """
    global _last_request_time
    for attempt in range(max_retries):
        try:
            with _rate_lock:
                now = time.monotonic()
                wait = _MIN_REQUEST_INTERVAL - (now - _last_request_time)
                _last_request_time = max(now, _last_request_time + _MIN_REQUEST_INTERVAL)
            if wait > 0:
                time.sleep(wait)

            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else 2 ** (attempt + 1)
                    # push global rate limiter forward so other threads back off too
                    with _rate_lock:
                        _last_request_time = max(
                            _last_request_time,
                            time.monotonic() + wait,
                        )
                else:
                    wait = 2 ** (attempt + 1)
                logger.warning(f"HTTP {resp.status_code} on {url}, retry in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wait = 2 ** (attempt + 1)
            logger.warning(f"Connection/timeout error on {url}, retry in {wait}s")
            time.sleep(wait)
    logger.warning(f"Giving up on {url} after {max_retries} retries, skipping")
    return None


# ---------------------------------------------------------------------------
# Venue index discovery
# ---------------------------------------------------------------------------

def _discover_venue_stems(dblp_key: str, start_year: int = 1965) -> dict[str, list[str]]:
    """Fetch the DBLP venue index page and return {year: [stem, ...]} mapping.

    Each stem is a filename without extension, e.g. "cvpr2013" or
    "eccv2024-7".  Multi-volume proceedings (ECCV) produce multiple stems
    per year; they are all grouped under the same year key.

    Some venues (e.g. UAI) don't list year pages on their index — for those
    we fall back to probing individual year URLs directly.

    *start_year* controls the earliest year probed in the fallback path
    (default 1965, covering IJCAI which began in 1969).
    """
    url = f"{DBLP_DB}/{dblp_key}/"
    resp = _fetch_with_retry(url)
    if resp is None:
        logger.error(f"Could not fetch DBLP index for {dblp_key}: {url}")
        return {}

    # dict.fromkeys deduplicates while preserving order
    stems = list(dict.fromkeys(re.findall(
        rf'/db/{re.escape(dblp_key)}/([^/"]+)\.html',
        resp.text,
    )))

    by_year: dict[str, list[str]] = {}
    for stem in stems:
        year_match = re.search(r'(\d{4})', stem)
        if year_match:
            if not re.match(r'^[a-z]+\d{4}', stem):
                continue
            year = year_match.group(1)
        else:
            m2 = re.match(r'^[a-z]+(\d{2})(?:\D|$)', stem)
            if not m2:
                continue
            yy = int(m2.group(1))
            year = str(1900 + yy if yy >= 50 else 2000 + yy)
        by_year.setdefault(year, []).append(stem)

    # fallback: probe individual year URLs when index has no links (e.g. UAI)
    if not by_year:
        slug = dblp_key.rsplit("/", 1)[-1]  # "conf/uai" → "uai"
        logger.info(f"  Index page has no year links, probing {slug}YYYY.html...")
        for year in range(start_year, datetime.date.today().year + 2):
            if year < 2000:
                stem = f"{slug}{year % 100:02d}"
            else:
                stem = f"{slug}{year}"
            probe_url = f"{DBLP_DB}/{dblp_key}/{stem}.html"
            probe = _fetch_with_retry(probe_url)
            if probe is not None and probe.status_code == 200:
                by_year.setdefault(str(year), []).append(stem)

    return by_year


# ---------------------------------------------------------------------------
# Paper fetching via DBLP search API
# ---------------------------------------------------------------------------

def _fetch_papers_for_stem(stem: str, dblp_key: str) -> list[dict]:
    """Query DBLP search API for all papers in a single proceedings file.

    Uses pagination (f= offset) to retrieve more than 1000 results.
    """
    bht_path = f"db/{dblp_key}/{stem}.bht"
    all_hits: list[dict] = []
    offset = 0
    per_page = 1000

    while True:
        url = (
            f"{DBLP_API}"
            f"?q=toc:{bht_path}:&h={per_page}&f={offset}&format=json"
        )
        resp = _fetch_with_retry(url)
        if resp is None:
            break

        try:
            data = resp.json()
        except ValueError:
            logger.warning(f"Non-JSON response from DBLP for {bht_path}")
            break

        hits_data = data.get("result", {}).get("hits", {})
        total = int(hits_data.get("@total", 0))
        hit_list = hits_data.get("hit", [])
        if not hit_list:
            break

        all_hits.extend(hit_list)
        offset += len(hit_list)

        if offset >= total:
            break

        time.sleep(0.3)  # be polite between paginated requests

    return all_hits


# ---------------------------------------------------------------------------
# Author parsing
# ---------------------------------------------------------------------------

def _parse_author(author_data: dict | list) -> dict:
    """Parse a DBLP author entry, stripping disambiguation suffixes.

    DBLP appends " 0001", " 0002" etc. to distinguish same-name authors.
    """
    if isinstance(author_data, list):
        author_data = author_data[0] if author_data else {}

    name = author_data.get("text", "").strip()
    name = re.sub(r'\s+\d{4}$', '', name).strip()  # strip DBLP disambiguation suffix

    return parse_author_name(name)


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------

def process_venue_year(
    venue_slug: str,
    year: str,
    dblp_key: str,
    stems: list[str],
) -> list[dict]:
    """Fetch and parse all papers for a single venue-year.

    Handles multi-volume proceedings by fetching all stems and merging.
    """
    venue_name = DBLP_VENUES[venue_slug]["name"]

    if len(stems) > 1:
        logger.info(f"  {venue_slug.upper()} {year}: {len(stems)} volumes")

    all_hits: list[dict] = []
    for stem in sorted(stems):
        hits = _fetch_papers_for_stem(stem, dblp_key)
        all_hits.extend(hits)
        if len(stems) > 1:
            logger.info(f"    {stem}: {len(hits)} papers")
        if len(stems) > 1:
            time.sleep(0.5)  # extra pause between volumes

    if not all_hits:
        return []

    papers: list[dict] = []
    bibtex_keys: list[str] = []

    for hit in all_hits:
        info = hit.get("info", {})

        title = info.get("title", "").strip().rstrip(".")
        if not title:
            continue

        authors_blob = info.get("authors", {})
        author_raw = authors_blob.get("author", [])
        if isinstance(author_raw, dict):
            author_raw = [author_raw]

        authors = [_parse_author(a) for a in author_raw]
        if not authors:
            continue

        pages = info.get("pages", "")
        ee = info.get("ee", "")
        dblp_url = info.get("url", "")
        source_id = info.get("key", "")

        doi = info.get("doi", "")
        if not doi and ee.startswith("https://doi.org/"):
            doi = ee[len("https://doi.org/"):]

        # detect workshop papers via DBLP's venue field
        dblp_venue_raw = info.get("venue", "")
        if isinstance(dblp_venue_raw, list):
            dblp_venue = " ".join(str(v) for v in dblp_venue_raw).lower()
        else:
            dblp_venue = str(dblp_venue_raw).lower()
        if venue_slug in WORKSHOP_OVERRIDES and "workshop" in dblp_venue:
            paper_venue_slug = WORKSHOP_OVERRIDES[venue_slug]["slug"]
            paper_venue_name = WORKSHOP_OVERRIDES[venue_slug]["name"]
        else:
            paper_venue_slug = venue_slug
            paper_venue_name = venue_name

        first_author = authors[0]
        bkey = make_bibtex_key(
            first_author_family=first_author.get("family", ""),
            year=year,
            venue=paper_venue_slug,
            title=title,
        )

        papers.append({
            "bibtex_key": bkey,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": paper_venue_slug,
            "venue_name": paper_venue_name,
            "volume": "",
            "pages": pages,
            "abstract": "",
            "pdf_url": "",
            "venue_url": ee,
            "doi": doi,
            "openreview_url": "",
            "code_url": "",
            "source": "dblp",
            "source_id": source_id,
        })
        bibtex_keys.append(bkey)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    logger.info(f"  {venue_slug.upper()} {year}: {len(papers)} papers")
    return papers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _find_legacy_dir() -> Optional[Path]:
    """Locate the legacy data directory relative to this file."""
    legacy = Path(__file__).resolve().parent.parent / "data" / "legacy"
    return legacy if legacy.is_dir() else None


def fetch_all(
    venues: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
    max_year: Optional[int] = None,
    fill_only: bool = True,
    max_workers: int = 4,
    backlog: bool = False,
) -> dict[str, list[dict]]:
    """Fetch DBLP backlog for specified venues, writing output files.

    Args:
        venues: Venue slugs to process.  Defaults to all of DBLP_VENUES.
        output_dir: Directory to write files.  Defaults to data/backlog when
            backlog=True, data/papers otherwise.
        cache: Fetch cache dict.  None = no cache.
        max_year: Upper bound year (inclusive).  Defaults to current year.
        fill_only: If True (default), skip venue-years where the output file
            already exists.  This ensures primary adapters (which have
            abstracts) are never overwritten by DBLP's abstract-free records.
        max_workers: Number of concurrent year-fetching threads.
        backlog: If True, write gzipped JSONL to data/backlog/ instead of
            plain YAML to data/papers/.  Backlog files are committed once and
            read by build_content.py identically to data/papers/ files.
    """
    if venues is None:
        venues = list(DBLP_VENUES.keys())

    if output_dir is None:
        output_dir = Path("data/backlog") if backlog else Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_year is None:
        max_year = datetime.date.today().year

    legacy_dir = _find_legacy_dir()
    legacy_venues: set[str] = set()
    if legacy_dir:
        for f in legacy_dir.glob("*-legacy.jsonl.gz"):
            slug = f.stem.replace("-legacy.jsonl", "")
            legacy_venues.add(slug)
        if legacy_venues:
            logger.info(f"Legacy data found for: {', '.join(sorted(legacy_venues))}")

    jobs: list[tuple] = []
    for venue_slug in venues:
        if venue_slug not in DBLP_VENUES:
            logger.warning(f"Unknown venue slug '{venue_slug}', skipping")
            continue

        venue_info = DBLP_VENUES[venue_slug]
        dblp_key = venue_info["key"]
        start_year = venue_info["start"]

        if fill_only and venue_slug in legacy_venues:
            logger.info(
                f"Skipping {venue_slug.upper()}: covered by legacy data "
                f"(data/legacy/{venue_slug}-legacy.jsonl.gz)"
            )
            continue

        logger.info(f"Discovering {venue_slug.upper()} stems from DBLP ({dblp_key})...")
        stems_by_year = _discover_venue_stems(dblp_key, start_year=start_year)
        if not stems_by_year:
            logger.warning(f"  No stems found for {venue_slug.upper()}, skipping")
            continue

        stem_filter_pattern = venue_info.get("stem_filter", "")
        if stem_filter_pattern:
            sfre = re.compile(stem_filter_pattern)
            stems_by_year = {
                y: [s for s in stems if sfre.search(s)]
                for y, stems in stems_by_year.items()
            }
            stems_by_year = {y: s for y, s in stems_by_year.items() if s}

        target_years = sorted(
            y for y in stems_by_year
            if start_year <= int(y) <= max_year
        )
        logger.info(
            f"  Found {len(stems_by_year)} total years in DBLP, "
            f"{len(target_years)} within [{start_year}, {max_year}]"
        )

        for year in target_years:
            venue_year_key = f"{venue_slug}-{year}"
            ext = ".jsonl.gz" if backlog else ".json.gz"
            out_path = output_dir / f"{venue_year_key}{ext}"
            cache_key = f"dblp-{venue_year_key}"

            if fill_only and out_path.exists():
                logger.debug(f"  Skipping {venue_year_key}: output file exists")
                continue

            if cache is not None and not should_fetch(cache, cache_key, year):
                logger.info(f"  Skipping {venue_year_key}: cached")
                continue

            jobs.append((venue_slug, year, dblp_key, stems_by_year[year], out_path, cache_key))

    logger.info(f"Fetching {len(jobs)} venue-years with {max_workers} workers...")

    all_papers: dict[str, list[dict]] = {}
    write_lock = threading.Lock()

    def _fetch_job(venue_slug: str, year: str, dblp_key: str, stems: list[str],
                   out_path: Path, cache_key: str) -> tuple[str, int]:
        papers = process_venue_year(venue_slug, year, dblp_key, stems)
        if not papers:
            return f"{venue_slug}-{year}", 0

        venue_year_key = f"{venue_slug}-{year}"
        records = [normalize_paper(p) for p in papers]
        with write_lock:
            if backlog:
                with gzip.open(out_path, "wt", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"  Wrote {out_path}")
            else:
                write_venue_json(venue_slug, year, records, output_dir)
            all_papers[venue_year_key] = papers
            if cache is not None:
                mark_fetched(cache, cache_key)

        return venue_year_key, len(papers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_job, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                job = futures[future]
                logger.error(f"  {job[0]}-{job[1]} failed: {exc}")

    return all_papers


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Fetch DBLP backlog — fills historical gaps in venue coverage"
    )
    parser.add_argument(
        "--venue", type=str,
        choices=sorted(DBLP_VENUES.keys()),
        help="Single venue slug to fetch",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch all venues",
    )
    parser.add_argument(
        "--max-year", type=int, default=None,
        help="Upper bound year (inclusive). Useful for backlog runs, e.g. --max-year 2015",
    )
    parser.add_argument(
        "--no-fill-only", action="store_true",
        help="Overwrite existing output files (default: skip if file already exists)",
    )
    parser.add_argument(
        "--output", type=str, default="data/papers",
        help="Output directory for YAML files",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent year-fetching threads (default: 8)",
    )
    parser.add_argument(
        "--backlog", action="store_true",
        help="Write gzipped JSONL to data/backlog/ instead of YAML to data/papers/",
    )
    args = parser.parse_args()

    venues_to_run = None
    if args.venue:
        venues_to_run = [args.venue]
    elif args.all:
        venues_to_run = list(DBLP_VENUES.keys())
    else:
        parser.print_help()
        raise SystemExit(0)

    fetch_all(
        venues=venues_to_run,
        output_dir=Path(args.output) if args.output != "data/papers" else None,
        max_year=args.max_year,
        fill_only=not args.no_fill_only,
        max_workers=args.workers,
        backlog=args.backlog,
    )
