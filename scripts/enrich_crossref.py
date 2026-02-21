#!/usr/bin/env python3
"""Enrich legacy data with Crossref metadata (abstracts + PDF URLs).

Reads a legacy JSONL.gz file, queries Crossref for papers that have a
DOI but are missing an abstract and/or PDF URL, and writes back in place.

Crossref has near-perfect coverage for AAAI papers (10.1609 prefix) —
testing showed 100% hit rate across 2010-2025.  No API key required.

Usage:
    # Enrich AAAI legacy data (default: all papers needing enrichment)
    python scripts/enrich_crossref.py --venue aaai

    # Dry run: see what would be enriched without making changes
    python scripts/enrich_crossref.py --venue aaai --dry-run

    # Enrich all legacy files
    python scripts/enrich_crossref.py

    # Limit to first N papers (for testing)
    python scripts/enrich_crossref.py --venue aaai --limit 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy
from adapters.crossref import enrich_papers, fetch_batch

logger = logging.getLogger(__name__)


def _stats(papers: list[dict]) -> dict[str, int]:
    """Count papers with each enrichable field populated."""
    return {
        "total": len(papers),
        "abstract": sum(1 for p in papers if p.get("abstract", "").strip()),
        "doi": sum(1 for p in papers if p.get("doi", "").strip()),
        "pdf_url": sum(1 for p in papers if p.get("pdf_url", "").strip()),
    }


def enrich_file(path: Path, *, dry_run: bool = False, limit: int = 0) -> None:
    """Read, enrich via Crossref, and write back a single legacy file."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {path.name}...")

    papers = read_legacy(path)
    if not papers:
        logger.info("  Empty file, skipping")
        return

    before = _stats(papers)
    needs_abstract = before["doi"] - sum(
        1 for p in papers
        if p.get("doi", "").strip() and p.get("abstract", "").strip()
    )
    needs_pdf = before["doi"] - sum(
        1 for p in papers
        if p.get("doi", "").strip() and p.get("pdf_url", "").strip()
    )

    logger.info(
        f"  {before['total']} papers — "
        f"abstracts: {before['abstract']}/{before['total']}, "
        f"DOIs: {before['doi']}/{before['total']}, "
        f"PDFs: {before['pdf_url']}/{before['total']}"
    )
    logger.info(
        f"  Crossref candidates: {needs_abstract} need abstract, "
        f"{needs_pdf} need PDF URL"
    )

    if needs_abstract == 0 and needs_pdf == 0:
        logger.info("  Nothing to enrich, skipping")
        return

    if dry_run:
        logger.info("  DRY RUN — no changes made")
        return

    # Optionally limit for testing
    target = papers
    if limit > 0:
        # Only enrich the first N papers that need it
        target_indices = []
        for i, p in enumerate(papers):
            doi = p.get("doi", "").strip()
            if doi and (not p.get("abstract", "").strip() or not p.get("pdf_url", "").strip()):
                target_indices.append(i)
                if len(target_indices) >= limit:
                    break
        target = [papers[i] for i in target_indices]
        logger.info(f"  Limiting to {len(target)} papers (--limit {limit})")

        # Enrich the subset, then copy results back
        enriched_count = enrich_papers(target)

        # Copy enriched fields back to full list
        for idx, paper_idx in enumerate(target_indices):
            papers[paper_idx] = target[idx]
    else:
        enriched_count = enrich_papers(papers)

    after = _stats(papers)

    logger.info(f"  Results for {path.stem.replace('-legacy', '').upper()}:")
    for field in ("abstract", "doi", "pdf_url"):
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
        description="Enrich legacy data with Crossref metadata (abstracts + PDF URLs)"
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help="Enrich a single venue (e.g. 'aaai'). Default: all legacy files.",
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

    logger.info(f"Will enrich {len(files)} legacy file(s) via Crossref")

    t_start = time.time()

    for path in files:
        enrich_file(path, dry_run=args.dry_run, limit=args.limit)

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Final summary
    logger.info("\nFinal summary:")
    for path in files:
        papers = read_legacy(path)
        stats = _stats(papers)
        name = path.stem.replace("-legacy", "").upper()
        logger.info(
            f"  {name}: {stats['total']} papers — "
            f"abs: {stats['abstract']}, "
            f"doi: {stats['doi']}, "
            f"pdf: {stats['pdf_url']}"
        )


if __name__ == "__main__":
    main()
