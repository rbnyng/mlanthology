#!/usr/bin/env python3
"""Enrich legacy IEEE venue data with abstracts from the IEEE Xplore API.

Reads legacy JSONL.gz files for IEEE-published venues (CVPR, ICCV, WACV,
and their workshops), queries the IEEE Xplore Metadata API for papers that
have a 10.1109/ DOI but are missing an abstract, and writes back in place.

Free-tier limits: 200 requests/day, 10 requests/second — one DOI per
request.  At 200/day the ~12 000 legacy IEEE papers will take ~60 days of
daily runs to fully enrich.  The script is idempotent: each run skips
papers that already have abstracts, so running it daily accumulates
coverage automatically.

Venues are processed smallest-first by default (wacvw → wacv → iccvw →
cvprw → iccv → cvpr) so the daily quota goes to the venues closest to
full coverage first.

Requires a free IEEE Xplore API key:
  Register at https://developer.ieee.org/
  Then set: export IEEE_API_KEY=your_key_here

Usage:
    # Enrich all IEEE venues (respects 200 req/day quota)
    python scripts/enrich_ieee.py

    # Enrich a single venue
    python scripts/enrich_ieee.py --venue cvpr

    # Dry run: show what would be enriched without making changes
    python scripts/enrich_ieee.py --dry-run

    # Override daily limit (e.g. if your key has a higher quota)
    python scripts/enrich_ieee.py --daily-limit 500

    # Limit to first N papers per venue (for testing)
    python scripts/enrich_ieee.py --venue cvpr --limit 10
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
from adapters.ieee import (
    IEEE_VENUES,
    enrich_papers,
    _get_api_key,
    _is_ieee_doi,
)

logger = logging.getLogger(__name__)


def _stats(papers: list[dict]) -> dict[str, int]:
    """Count papers with each enrichable field populated."""
    return {
        "total": len(papers),
        "abstract": sum(1 for p in papers if p.get("abstract", "").strip()),
        "ieee_doi": sum(1 for p in papers if _is_ieee_doi(p.get("doi", ""))),
        "venue_url": sum(1 for p in papers if p.get("venue_url", "").strip()),
    }


def _ieee_missing(papers: list[dict]) -> int:
    return sum(
        1
        for p in papers
        if _is_ieee_doi(p.get("doi", "")) and not p.get("abstract", "").strip()
    )


def enrich_file(
    path: Path,
    api_key: str,
    *,
    dry_run: bool = False,
    limit: int = 0,
    daily_limit: int = 200,
) -> int:
    """Read, enrich via IEEE Xplore, and write back a single legacy file.

    Returns the number of API calls consumed.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {path.name}...")

    papers = read_legacy(path)
    if not papers:
        logger.info("  Empty file, skipping")
        return 0

    before = _stats(papers)
    missing = _ieee_missing(papers)

    logger.info(
        f"  {before['total']} papers — "
        f"abstracts: {before['abstract']}/{before['total']}, "
        f"IEEE DOIs: {before['ieee_doi']}/{before['total']}"
    )
    logger.info(f"  IEEE candidates missing abstract: {missing}")

    if missing == 0:
        logger.info("  Nothing to enrich, skipping")
        return 0

    if dry_run:
        logger.info("  DRY RUN — no changes made")
        return 0

    if limit > 0:
        # Enrich only the first `limit` IEEE papers missing abstracts.
        target_indices = []
        for i, p in enumerate(papers):
            doi = p.get("doi", "").strip()
            if doi and _is_ieee_doi(doi) and not p.get("abstract", "").strip():
                target_indices.append(i)
                if len(target_indices) >= limit:
                    break
        target = [papers[i] for i in target_indices]
        logger.info(f"  Limiting to {len(target)} papers (--limit {limit})")
        enriched_count = enrich_papers(
            target, api_key=api_key, daily_limit=min(limit, daily_limit)
        )
        for idx, paper_idx in enumerate(target_indices):
            papers[paper_idx] = target[idx]
        calls_used = len(target)
    else:
        enriched_count = enrich_papers(
            papers, api_key=api_key, daily_limit=daily_limit
        )
        calls_used = min(missing, daily_limit)

    after = _stats(papers)
    name = path.stem.replace("-legacy", "").upper()
    logger.info(f"  Results for {name}:")
    for field in ("abstract", "venue_url"):
        delta = after.get(field, 0) - before.get(field, 0)
        symbol = f"+{delta}" if delta > 0 else "no change"
        logger.info(
            f"    {field}: {before.get(field, 0)} -> {after.get(field, 0)} ({symbol})"
        )

    still_missing = _ieee_missing(papers)
    if still_missing:
        logger.info(f"  {still_missing} papers still need enrichment (run again tomorrow)")

    if enriched_count > 0:
        write_legacy(path, papers)
        logger.info(f"  Written back to {path.name}")
    else:
        logger.info("  No changes to write")

    return calls_used


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich legacy IEEE venue data with IEEE Xplore abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help=(
            "Enrich a single venue (e.g. 'cvpr', 'iccv', 'wacv', "
            "'cvprw', 'iccvw', 'wacvw'). "
            "Default: all IEEE venues, smallest-first."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="IEEE Xplore API key (overrides IEEE_API_KEY env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enriched without making changes.",
    )
    parser.add_argument(
        "--daily-limit",
        type=int,
        default=200,
        help="Maximum API calls for this run (default 200 = free-tier cap).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N IEEE papers per file (for testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = _get_api_key(args.api_key)
    if not api_key:
        logger.error(
            "No IEEE API key found.  Set the IEEE_API_KEY environment "
            "variable or pass --api-key.  Register at https://developer.ieee.org/"
        )
        sys.exit(1)

    if args.venue:
        files = [LEGACY_DIR / f"{args.venue}-legacy.jsonl.gz"]
        if not files[0].exists():
            logger.error(f"File not found: {files[0]}")
            sys.exit(1)
    else:
        # All IEEE venues, sorted smallest-missing-abstract-first
        files = [
            LEGACY_DIR / f"{v}-legacy.jsonl.gz"
            for v in IEEE_VENUES
            if (LEGACY_DIR / f"{v}-legacy.jsonl.gz").exists()
        ]

    if not files:
        logger.error("No legacy files found for IEEE venues")
        sys.exit(1)

    logger.info(
        f"Will process {len(files)} legacy file(s) — "
        f"daily limit: {args.daily_limit} API calls"
    )

    t_start = time.time()
    quota_remaining = args.daily_limit

    for path in files:
        if quota_remaining <= 0:
            logger.warning("Daily quota exhausted — stopping. Run again tomorrow.")
            break

        calls = enrich_file(
            path,
            api_key,
            dry_run=args.dry_run,
            limit=args.limit,
            daily_limit=quota_remaining,
        )
        quota_remaining -= calls
        if quota_remaining > 0:
            logger.info(f"  Quota remaining: {quota_remaining} calls")

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(
        f"API calls used: {args.daily_limit - quota_remaining}/{args.daily_limit}"
    )


if __name__ == "__main__":
    main()
