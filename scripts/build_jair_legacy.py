#!/usr/bin/env python3
"""One-time script to build legacy files for DBLP-indexed journals.

Journals publish volumes (not year-indexed proceedings), so the standard DBLP
adapter's year-based stem discovery doesn't apply.  This script:

  1. Discovers volume stems ({slug}1 … {slug}N) from the DBLP index page.
  2. Fetches papers per volume via the DBLP search API.
  3. Writes a single {venue}-legacy.jsonl.gz.

Supported journals:
  jair  — Journal of Artificial Intelligence Research (journals/jair)
  mlj   — Machine Learning (journals/ml)
  neco  — Neural Computation (journals/neco, capped at 2004)

Usage:
    python scripts/build_jair_legacy.py               # build all
    python scripts/build_jair_legacy.py --venue jair   # single venue
    python scripts/build_jair_legacy.py --venue neco

Then enrich with S2 and rebuild content:
    python scripts/enrich_legacy.py --s2-api-key KEY --venue jair
    python scripts/enrich_legacy.py --s2-api-key KEY --venue neco
    python scripts/build_content.py
"""

import argparse
import logging
import re
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, write_legacy

from adapters.dblp import (
    _fetch_with_retry,
    _fetch_papers_for_stem,
    _parse_author,
    DBLP_DB,
)
from adapters.common import make_bibtex_key, resolve_bibtex_collisions, normalize_paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Journal configurations: venue_slug -> {dblp_key, dblp_stem_prefix, name, max_year?}
# max_year caps paper inclusion by year (useful when a journal changed scope).
JOURNALS = {
    "jair": {
        "dblp_key": "journals/jair",
        "stem_prefix": "jair",
        "name": "Journal of Artificial Intelligence Research",
    },
    "mlj": {
        "dblp_key": "journals/ml",
        "stem_prefix": "ml",
        "name": "Machine Learning",
    },
    "neco": {
        "dblp_key": "journals/neco",
        "stem_prefix": "neco",
        "name": "Neural Computation",
        "max_year": 2004,  # shifted to neuroscience after ~2004
    },
    "ftml": {
        "dblp_key": "journals/ftml",
        "stem_prefix": "ftml",
        "name": "Foundations and Trends in Machine Learning",
    },
    "distill": {
        "dblp_key": "journals/distill",
        "stem_prefix": "distill",
        "name": "Distill",
    },
}


def discover_volumes(dblp_key: str, stem_prefix: str) -> list[str]:
    """Fetch the DBLP index page and extract volume stems.

    Falls back to probing sequential volume numbers if the index page
    returns a truncated response (DBLP occasionally does this).
    """
    url = f"{DBLP_DB}/{dblp_key}/"
    resp = _fetch_with_retry(url)
    if resp is None:
        logger.error(f"Could not fetch DBLP index: {url}")
        return []

    # DBLP index pages use either relative (/db/...) or absolute (https://dblp.org/db/...)
    # links depending on the venue.  Match both forms.
    stems = list(dict.fromkeys(re.findall(
        rf'(?:https://dblp\.org)?/db/{re.escape(dblp_key)}/({re.escape(stem_prefix)}\d+)\.html',
        resp.text,
    )))

    # Fallback: if the index page was truncated, probe sequential volume numbers.
    if not stems:
        logger.info("  Index page had no volume links, probing sequentially...")
        import time
        for vol in range(1, 500):
            stem = f"{stem_prefix}{vol}"
            probe_url = f"{DBLP_DB}/{dblp_key}/{stem}.html"
            probe = _fetch_with_retry(probe_url)
            if probe is None or probe.status_code == 404:
                break
            stems.append(stem)
            time.sleep(0.3)

    # Sort by volume number
    stems.sort(key=lambda s: int(re.search(r'\d+', s).group()))
    return stems


def process_volume(stem: str, dblp_key: str, venue_slug: str, venue_name: str) -> list[dict]:
    """Fetch and parse all papers for a single journal volume."""
    vol_num = re.search(r'\d+', stem).group()

    hits = _fetch_papers_for_stem(stem, dblp_key)
    if not hits:
        return []

    papers = []
    bibtex_keys = []

    for hit in hits:
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

        year = info.get("year", "")
        pages = info.get("pages", "")
        doi = info.get("doi", "")
        ee = info.get("ee", "")
        source_id = info.get("key", "")

        if not doi and ee.startswith("https://doi.org/"):
            doi = ee[len("https://doi.org/"):]

        bkey = make_bibtex_key(
            first_author_family=authors[0].get("family", ""),
            year=year,
            venue=venue_slug,
            title=title,
        )

        papers.append({
            "bibtex_key": bkey,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue_slug,
            "venue_name": venue_name,
            "volume": vol_num,
            "number": "",
            "pages": pages or "",
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

    logger.info(f"  {venue_slug.upper()} vol {vol_num}: {len(papers)} papers")
    return papers


def build_venue(venue_slug: str) -> None:
    """Build legacy file for a single journal venue."""
    conf = JOURNALS[venue_slug]
    dblp_key = conf["dblp_key"]
    stem_prefix = conf["stem_prefix"]
    venue_name = conf["name"]

    out_path = LEGACY_DIR / f"{venue_slug}-legacy.jsonl.gz"
    if out_path.exists():
        logger.info(f"{out_path.name} already exists — skipping (delete to rebuild)")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Discovering {venue_slug.upper()} volumes from DBLP ({dblp_key})...")
    stems = discover_volumes(dblp_key, stem_prefix)
    if not stems:
        logger.error("No volume stems found, aborting")
        return
    logger.info(f"Found {len(stems)} volumes: {stems[0]} … {stems[-1]}")

    max_year = conf.get("max_year")

    all_papers = []
    for stem in stems:
        papers = process_volume(stem, dblp_key, venue_slug, venue_name)
        if max_year:
            papers = [p for p in papers if int(p["year"]) <= max_year]
        all_papers.extend(normalize_paper(p) for p in papers)

    if not all_papers:
        logger.warning("No papers collected, skipping write")
        return

    write_legacy(out_path, all_papers)

    years = sorted(set(p["year"] for p in all_papers))
    logger.info(f"Wrote {len(all_papers)} {venue_slug.upper()} papers to {out_path.name}")
    logger.info(f"Year range: {years[0]}–{years[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build legacy files for DBLP-indexed journals (JAIR, MLJ, NECO)"
    )
    parser.add_argument(
        "--venue", type=str, choices=sorted(JOURNALS.keys()),
        help="Single venue to build (default: all)",
    )
    args = parser.parse_args()

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    venues = [args.venue] if args.venue else sorted(JOURNALS.keys())
    for venue_slug in venues:
        build_venue(venue_slug)

    print(
        "\nDone. Next steps:\n"
        + "".join(f"  python scripts/enrich_legacy.py --s2-api-key KEY --venue {v}\n" for v in venues)
        + "  python scripts/build_content.py"
    )


if __name__ == "__main__":
    main()
