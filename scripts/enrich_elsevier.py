#!/usr/bin/env python3
"""Enrich legacy data with abstracts from the Elsevier ScienceDirect API.

Reads legacy JSONL.gz files, queries the Elsevier Full-Text Article API for
papers with 10.1016/ or 10.1006/ DOIs that are missing abstracts, and also
sets venue_url = https://doi.org/{doi} for any paper that has a DOI but no
URL at all.

Primarily targets ICML 1988-2002 (book chapters published by Morgan
Kaufmann/Elsevier), plus a handful of IJCAI and COLT records.

Requires an Elsevier API key:
  Register at https://dev.elsevier.com/
  Then set: export ELSEVIER_API_KEY=your_key_here

Usage:
    export ELSEVIER_API_KEY=your_key_here

    # Enrich ICML legacy data
    python scripts/enrich_elsevier.py --venue icml

    # Enrich all eligible venues
    python scripts/enrich_elsevier.py

    # Dry run: show what would be enriched without making changes
    python scripts/enrich_elsevier.py --dry-run

    # Limit to first N papers (for testing)
    python scripts/enrich_elsevier.py --venue icml --limit 5
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy
from adapters.elsevier import ELSEVIER_DOI_PREFIXES, enrich_papers

logger = logging.getLogger(__name__)

# Venues that have Elsevier DOIs in their no-URL records
ELIGIBLE_VENUES = ["icml", "ijcai", "colt"]


def _stats(papers: list[dict]) -> dict[str, int]:
    return {
        "total": len(papers),
        "abstract": sum(1 for p in papers if p.get("abstract", "").strip()),
        "venue_url": sum(1 for p in papers if p.get("venue_url", "").strip()),
        "elsevier_doi": sum(
            1 for p in papers
            if p.get("doi", "").strip()
            and any(p["doi"].strip().startswith(pfx) for pfx in ELSEVIER_DOI_PREFIXES)
        ),
    }


def enrich_file(path: Path, api_key: str, *, dry_run: bool = False, limit: int = 0) -> None:
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {path.name}...")

    papers = read_legacy(path)
    if not papers:
        logger.info("  Empty file, skipping")
        return

    before = _stats(papers)

    # Count what needs work
    needs_abstract = sum(
        1 for p in papers
        if p.get("doi", "").strip()
        and any(p["doi"].strip().startswith(pfx) for pfx in ELSEVIER_DOI_PREFIXES)
        and not p.get("abstract", "").strip()
    )
    needs_venue_url = sum(
        1 for p in papers
        if p.get("doi", "").strip()
        and not (p.get("venue_url") or p.get("pdf_url") or p.get("openreview_url"))
    )

    logger.info(
        f"  {before['total']} papers — "
        f"abstracts: {before['abstract']}/{before['total']}, "
        f"venue_url: {before['venue_url']}/{before['total']}, "
        f"Elsevier DOIs: {before['elsevier_doi']}"
    )
    logger.info(
        f"  Candidates: {needs_abstract} need abstract, "
        f"{needs_venue_url} need venue_url"
    )

    if needs_abstract == 0 and needs_venue_url == 0:
        logger.info("  Nothing to enrich, skipping")
        return

    if dry_run:
        logger.info("  DRY RUN — no changes made")
        return

    if limit > 0:
        # Only process the first N papers that need enrichment
        target_indices = []
        for i, p in enumerate(papers):
            doi = p.get("doi", "").strip()
            if not doi:
                continue
            needs = (
                any(doi.startswith(pfx) for pfx in ELSEVIER_DOI_PREFIXES)
                and not p.get("abstract", "").strip()
            ) or not (p.get("venue_url") or p.get("pdf_url") or p.get("openreview_url"))
            if needs:
                target_indices.append(i)
                if len(target_indices) >= limit:
                    break

        target = [papers[i] for i in target_indices]
        logger.info(f"  Limiting to {len(target)} papers (--limit {limit})")
        enriched_count = enrich_papers(target, api_key)
        for idx, paper_idx in enumerate(target_indices):
            papers[paper_idx] = target[idx]
    else:
        enriched_count = enrich_papers(papers, api_key)

    after = _stats(papers)

    logger.info(f"  Results for {path.stem.replace('-legacy', '').upper()}:")
    for field in ("abstract", "venue_url"):
        delta = after[field] - before[field]
        symbol = f"+{delta}" if delta > 0 else "no change"
        logger.info(f"    {field}: {before[field]} -> {after[field]} ({symbol})")

    if enriched_count > 0:
        write_legacy(path, papers)
        logger.info(f"  Written back to {path.name}")
    else:
        logger.info("  No changes to write")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich legacy data with Elsevier abstracts and DOI venue URLs"
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help=f"Enrich a single venue (e.g. 'icml'). Default: {ELIGIBLE_VENUES}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enriched without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N papers needing enrichment (for testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ.get("ELSEVIER_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.error("ELSEVIER_API_KEY environment variable not set")
        sys.exit(1)

    if args.venue:
        files = [LEGACY_DIR / f"{args.venue}-legacy.jsonl.gz"]
        if not files[0].exists():
            logger.error(f"File not found: {files[0]}")
            sys.exit(1)
    else:
        files = [LEGACY_DIR / f"{v}-legacy.jsonl.gz" for v in ELIGIBLE_VENUES]
        files = [f for f in files if f.exists()]

    if not files:
        logger.error("No eligible legacy files found")
        sys.exit(1)

    logger.info(f"Will enrich {len(files)} legacy file(s) via Elsevier")

    t_start = time.time()
    for path in files:
        enrich_file(path, api_key, dry_run=args.dry_run, limit=args.limit)

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    logger.info("\nFinal summary:")
    for path in files:
        papers = read_legacy(path)
        stats = _stats(papers)
        name = path.stem.replace("-legacy", "").upper()
        logger.info(
            f"  {name}: {stats['total']} papers — "
            f"abs: {stats['abstract']}, "
            f"venue_url: {stats['venue_url']}, "
            f"elsevier_doi: {stats['elsevier_doi']}"
        )


if __name__ == "__main__":
    main()
