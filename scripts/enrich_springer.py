#!/usr/bin/env python3
"""Enrich legacy data with Springer Nature metadata (abstracts + venue URLs).

Reads a legacy JSONL.gz file, queries the Springer Nature Meta API for papers
that have a 10.1007/ DOI but are missing an abstract, and writes back in place.

Applicable venues (Springer LNCS/LNAI publications fetched via DBLP):
  ecml      — European Conference on Machine Learning (1993–2007)
  ecmlpkdd  — ECML-PKDD (2008–2025)
  alt       — Algorithmic Learning Theory (1990–2016)

Requires a free Springer Nature API key:
  Register at https://dev.springernature.com/
  Then set the environment variable: export SPRINGER_API_KEY=your_key_here

Usage:
    # Enrich ECML legacy (needs SPRINGER_API_KEY env var)
    python scripts/enrich_springer.py --venue ecml

    # Enrich all legacy files that have Springer DOIs
    python scripts/enrich_springer.py

    # Dry run: show what would be enriched without making changes
    python scripts/enrich_springer.py --venue ecml --dry-run

    # Limit to first N papers (for testing)
    python scripts/enrich_springer.py --venue ecml --limit 20
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy
from adapters.springer import enrich_papers, _get_api_key, _is_springer_doi

logger = logging.getLogger(__name__)

# Venues known to have Springer LNCS/LNAI DOIs (10.1007/ prefix).
# Other venues may also contain some Springer papers; --all handles those.
SPRINGER_VENUES = ["ecml", "ecmlpkdd", "alt"]


def _stats(papers: list[dict]) -> dict[str, int]:
    """Count papers with each enrichable field populated."""
    return {
        "total": len(papers),
        "abstract": sum(1 for p in papers if p.get("abstract", "").strip()),
        "doi": sum(1 for p in papers if p.get("doi", "").strip()),
        "springer_doi": sum(
            1 for p in papers if _is_springer_doi(p.get("doi", ""))
        ),
        "venue_url": sum(1 for p in papers if p.get("venue_url", "").strip()),
    }


def enrich_file(
    path: Path,
    api_key: str,
    *,
    dry_run: bool = False,
    limit: int = 0,
) -> None:
    """Read, enrich via Springer, and write back a single legacy file."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {path.name}...")

    papers = read_legacy(path)
    if not papers:
        logger.info("  Empty file, skipping")
        return

    before = _stats(papers)
    springer_without_abstract = sum(
        1
        for p in papers
        if _is_springer_doi(p.get("doi", "")) and not p.get("abstract", "").strip()
    )

    logger.info(
        f"  {before['total']} papers — "
        f"abstracts: {before['abstract']}/{before['total']}, "
        f"Springer DOIs: {before['springer_doi']}/{before['total']}"
    )
    logger.info(
        f"  Springer candidates missing abstract: {springer_without_abstract}"
    )

    if springer_without_abstract == 0:
        logger.info("  Nothing to enrich, skipping")
        return

    if dry_run:
        logger.info("  DRY RUN — no changes made")
        return

    if limit > 0:
        # Enrich only the first `limit` Springer papers missing abstracts.
        target_indices = []
        for i, p in enumerate(papers):
            doi = p.get("doi", "").strip()
            if doi and _is_springer_doi(doi) and not p.get("abstract", "").strip():
                target_indices.append(i)
                if len(target_indices) >= limit:
                    break
        target = [papers[i] for i in target_indices]
        logger.info(f"  Limiting to {len(target)} papers (--limit {limit})")
        enriched_count = enrich_papers(target, api_key=api_key)
        for idx, paper_idx in enumerate(target_indices):
            papers[paper_idx] = target[idx]
    else:
        enriched_count = enrich_papers(papers, api_key=api_key)

    after = _stats(papers)
    name = path.stem.replace("-legacy", "").upper()
    logger.info(f"  Results for {name}:")
    for field in ("abstract", "venue_url"):
        delta = after.get(field, 0) - before.get(field, 0)
        symbol = f"+{delta}" if delta > 0 else "no change"
        logger.info(
            f"    {field}: {before.get(field, 0)} -> {after.get(field, 0)} ({symbol})"
        )

    if enriched_count > 0:
        write_legacy(path, papers)
        logger.info(f"  Written back to {path.name}")
    else:
        logger.info("  No changes to write")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich legacy data with Springer Nature abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help=(
            "Enrich a single venue (e.g. 'ecml', 'ecmlpkdd', 'alt'). "
            "Default: all legacy files are scanned for Springer DOIs."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Springer Nature API key (overrides SPRINGER_API_KEY env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enriched without making changes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N Springer papers per file (for testing).",
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
            "No Springer API key found.  Set the SPRINGER_API_KEY environment "
            "variable or pass --api-key.  Register for a free key at "
            "https://dev.springernature.com/"
        )
        sys.exit(1)

    if args.venue:
        files = [LEGACY_DIR / f"{args.venue}-legacy.jsonl.gz"]
        if not files[0].exists():
            logger.error(f"File not found: {files[0]}")
            sys.exit(1)
    else:
        files = sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz"))

    if not files:
        logger.error("No legacy files found")
        sys.exit(1)

    logger.info(f"Will process {len(files)} legacy file(s) via Springer API")

    t_start = time.time()
    for path in files:
        enrich_file(path, api_key, dry_run=args.dry_run, limit=args.limit)

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
