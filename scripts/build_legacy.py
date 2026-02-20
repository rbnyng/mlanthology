#!/usr/bin/env python3
"""Build cached legacy data from DBLP + Semantic Scholar.

One-shot script that fetches historical proceedings metadata from DBLP,
enriches papers with abstracts from Semantic Scholar, and writes
compressed archives to data/legacy/.  This data is essentially frozen
(pre-2013 to pre-2019 depending on venue) and only needs to be rebuilt
if we discover errors.

The output files are gzipped JSONL (one JSON object per line), which
build_content.py already knows how to read via the backlog loading path.

Usage:
    # Build all legacy data (full run — takes a while due to S2 rate limits)
    python scripts/build_legacy.py

    # Single venue
    python scripts/build_legacy.py --venue icml

    # Skip Semantic Scholar enrichment (DBLP-only, no abstracts)
    python scripts/build_legacy.py --no-enrich

    # Use a Semantic Scholar API key for higher rate limits
    python scripts/build_legacy.py --s2-api-key YOUR_KEY

    # Dry run: show what would be fetched without actually fetching
    python scripts/build_legacy.py --dry-run

    # Rebuild a specific venue (overwrite existing legacy file)
    python scripts/build_legacy.py --venue eccv --force
"""

import argparse
import gzip
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `scripts` and `adapters` are
# importable when invoked as `python scripts/build_legacy.py`.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import ROOT, LEGACY_DIR, write_legacy

from adapters.dblp import (
    DBLP_VENUES,
    _discover_venue_stems,
    process_venue_year,
)
from adapters.common import normalize_paper, resolve_bibtex_collisions
from adapters.semantic_scholar import enrich_papers

logger = logging.getLogger(__name__)

# Per-venue cutoff: the year *before* the primary adapter starts.
# Legacy covers [DBLP start, cutoff] inclusive.  The primary adapter
# covers [cutoff+1, present].
#
# These are derived from the earliest year each primary adapter covers:
#   PMLR (ICML): 2013    → legacy 1980-2012
#   CVF (CVPR, ICCV): 2013  → legacy 1983-2012 / 1987-2012
#   ECVA (ECCV): 2018    → legacy 1990-2017
#   OpenReview (ICLR): 2018 → legacy 2013-2017
#   PMLR (AISTATS): 2019 → legacy 2007-2018
#   PMLR (UAI): 2019     → legacy 1985-2018
#   PMLR (COLT): 2019    → legacy 1988-2018
#   PMLR (CoRL): 2019    → legacy 2017-2018
#   PMLR (MIDL): 2023    → legacy: none (DBLP doesn't cover MIDL)
#   NeurIPS: excluded (neurips.py covers 1987-present with abstracts)
#   WACV: 2020 (CVF)     → legacy 2012-2019
LEGACY_CUTOFFS: dict[str, int] = {
    "icml": 2012,
    "iclr": 2017,
    "cvpr": 2012,
    "iccv": 2012,
    "eccv": 2017,
    "wacv": 2019,
    "aistats": 2018,
    "uai": 2018,
    "colt": 2018,
    "corl": 2018,
}


def build_venue_legacy(
    venue_slug: str,
    output_dir: Path,
    enrich: bool = True,
    api_key: str | None = None,
    title_fallback: bool = False,
    force: bool = False,
) -> int:
    """Build legacy data for a single venue.

    Returns the total number of papers written.
    """
    if venue_slug not in DBLP_VENUES:
        logger.error(f"Unknown venue: {venue_slug}")
        return 0

    cutoff = LEGACY_CUTOFFS.get(venue_slug)
    if cutoff is None:
        logger.info(f"No legacy cutoff for {venue_slug}, skipping")
        return 0

    venue_info = DBLP_VENUES[venue_slug]
    dblp_key = venue_info["key"]
    start_year = venue_info["start"]

    out_path = output_dir / f"{venue_slug}-legacy.jsonl.gz"
    if out_path.exists() and not force:
        logger.info(f"  {venue_slug}: legacy file already exists, skipping (use --force to rebuild)")
        return 0

    logger.info(f"Building legacy data for {venue_slug.upper()} ({start_year}-{cutoff})...")

    # Phase 1: Discover DBLP stems
    logger.info(f"  Discovering {venue_slug.upper()} stems from DBLP...")
    stems_by_year = _discover_venue_stems(dblp_key)
    if not stems_by_year:
        logger.warning(f"  No stems found for {venue_slug.upper()}")
        return 0

    target_years = sorted(
        y for y in stems_by_year
        if start_year <= int(y) <= cutoff
    )
    logger.info(f"  Found {len(target_years)} years in [{start_year}, {cutoff}]")

    if not target_years:
        logger.info(f"  No years to process for {venue_slug.upper()}")
        return 0

    # Phase 2: Fetch all papers from DBLP
    all_papers: list[dict] = []
    for year in target_years:
        stems = stems_by_year[year]
        papers = process_venue_year(venue_slug, year, dblp_key, stems)
        if papers:
            all_papers.extend(papers)

    if not all_papers:
        logger.warning(f"  No papers found for {venue_slug.upper()}")
        return 0

    logger.info(f"  DBLP: {len(all_papers)} total papers for {venue_slug.upper()}")

    # Phase 3: Enrich with Semantic Scholar abstracts
    if enrich:
        logger.info(f"  Enriching {len(all_papers)} papers with Semantic Scholar abstracts...")
        enrich_papers(all_papers, api_key=api_key, title_fallback=title_fallback)

    # Phase 4: Normalize, resolve key collisions, and write compressed output
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [normalize_paper(p) for p in all_papers]

    # Resolve bibtex key collisions (same author, year, venue, content word)
    old_keys = [r["bibtex_key"] for r in records]
    new_keys = resolve_bibtex_collisions(old_keys)
    for record, new_key in zip(records, new_keys):
        record["bibtex_key"] = new_key

    # Mark source as "dblp+s2" if enriched, "dblp" if not
    if enrich:
        for record in records:
            if record.get("abstract"):
                record["source"] = "dblp+s2"

    write_legacy(out_path, records)

    # Report stats
    with_abstract = sum(1 for r in records if r.get("abstract"))
    with_doi = sum(1 for r in records if r.get("doi"))
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"  Wrote {out_path.name}: {len(records)} papers, "
        f"{with_abstract} with abstracts, {with_doi} with DOIs, "
        f"{file_size_mb:.1f} MB compressed"
    )

    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Build cached legacy data from DBLP + Semantic Scholar"
    )
    parser.add_argument(
        "--venue", type=str,
        choices=list(LEGACY_CUTOFFS.keys()),
        help="Build legacy data for a single venue (default: all)",
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Skip Semantic Scholar enrichment (DBLP-only, no abstracts)",
    )
    parser.add_argument(
        "--s2-api-key", type=str, default=None,
        help="Semantic Scholar API key for higher rate limits",
    )
    parser.add_argument(
        "--title-fallback", action="store_true",
        help="Also try title-based search for papers without DOIs (slow)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing legacy files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without actually fetching",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: data/legacy/)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output) if args.output else LEGACY_DIR
    venues = [args.venue] if args.venue else list(LEGACY_CUTOFFS.keys())

    if args.dry_run:
        print("DRY RUN — would build legacy data for:")
        for v in venues:
            cutoff = LEGACY_CUTOFFS[v]
            start = DBLP_VENUES[v]["start"]
            out = output_dir / f"{v}-legacy.jsonl.gz"
            exists = " (exists, would skip)" if out.exists() and not args.force else ""
            print(f"  {v.upper()}: {start}-{cutoff}{exists}")
        print(f"\nEnrich with Semantic Scholar: {'no' if args.no_enrich else 'yes'}")
        print(f"Output directory: {output_dir}")
        return

    t_start = time.time()
    total_papers = 0

    for venue_slug in venues:
        count = build_venue_legacy(
            venue_slug=venue_slug,
            output_dir=output_dir,
            enrich=not args.no_enrich,
            api_key=args.s2_api_key,
            title_fallback=args.title_fallback,
            force=args.force,
        )
        total_papers += count

    elapsed = time.time() - t_start
    logger.info(
        f"\nDone. {total_papers} total papers across {len(venues)} venues "
        f"in {elapsed:.0f}s"
    )

    # Show summary of legacy files
    if output_dir.exists():
        total_size = 0
        for f in sorted(output_dir.glob("*.jsonl.gz")):
            size = f.stat().st_size
            total_size += size
            # Count lines
            with gzip.open(f, "rt") as gf:
                count = sum(1 for _ in gf)
            print(f"  {f.name}: {count} papers, {size / (1024 * 1024):.1f} MB")
        print(f"  Total: {total_size / (1024 * 1024):.1f} MB compressed")


if __name__ == "__main__":
    main()
