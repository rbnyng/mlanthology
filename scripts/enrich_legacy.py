#!/usr/bin/env python3
"""Enrich existing legacy data files with Semantic Scholar metadata.

Reads each *-legacy.jsonl.gz file, enriches papers with abstracts, DOIs,
and PDF URLs from Semantic Scholar, and writes back in place.  Only fills
empty fields — never overwrites existing data.

Two modes:
  - **Batch mode** (default): uses the DOI-batch endpoint for all files,
    optionally falling back to title search for papers without DOIs.
  - **Chunked mode** (``--chunked``): processes a single venue in chunks
    of N papers via title search, saving after each chunk.  Designed for
    long-running enrichment of large venues (IJCAI, AAAI) where you want
    to preserve partial progress.

Usage:
    # Batch: enrich all legacy files (DOI batch + title search fallback)
    python scripts/enrich_legacy.py --s2-api-key YOUR_KEY

    # Batch: single venue, DOI batch only (fast)
    python scripts/enrich_legacy.py --s2-api-key YOUR_KEY --venue cvpr --no-title-fallback

    # Chunked: title-search enrichment with periodic saves
    python scripts/enrich_legacy.py --s2-api-key YOUR_KEY --venue ijcai --chunked
    python scripts/enrich_legacy.py --s2-api-key YOUR_KEY --venue aaai --chunked --chunk-size 500
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, read_legacy, write_legacy

from adapters.semantic_scholar import enrich_papers, fetch_abstracts_by_title

logger = logging.getLogger(__name__)



def _stats(papers: list[dict]) -> dict[str, int]:
    return {
        "abstract": sum(1 for p in papers if p.get("abstract")),
        "doi": sum(1 for p in papers if p.get("doi")),
        "pdf_url": sum(1 for p in papers if p.get("pdf_url")),
    }


def enrich_file(path: Path, api_key: str, title_fallback: bool) -> None:
    """Read, enrich, and write back a single legacy file."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {path.name}...")

    papers = read_legacy(path)

    if not papers:
        logger.info("  Empty file, skipping")
        return

    before = _stats(papers)
    logger.info(
        f"  {len(papers)} papers — "
        f"abstracts: {before['abstract']}/{len(papers)}, "
        f"DOIs: {before['doi']}/{len(papers)}, "
        f"PDFs: {before['pdf_url']}/{len(papers)}"
    )

    enrich_papers(papers, api_key=api_key, title_fallback=title_fallback)

    for p in papers:
        if p.get("source") == "dblp":
            if p.get("abstract") or p.get("pdf_url"):
                p["source"] = "dblp+s2"

    after = _stats(papers)

    logger.info(f"  Results for {path.stem.replace('-legacy', '').upper()}:")
    for field in ("abstract", "doi", "pdf_url"):
        delta = after[field] - before[field]
        symbol = f"+{delta}" if delta > 0 else "no change"
        logger.info(f"    {field}: {before[field]} -> {after[field]} ({symbol})")

    write_legacy(path, papers)
    logger.info(f"  Written back to {path.name}")


def run_batch(args) -> None:
    """Run batch enrichment across one or all legacy files."""
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

    logger.info(f"Will enrich {len(files)} legacy file(s)")
    logger.info(f"Title fallback: {'disabled' if args.no_title_fallback else 'enabled'}")

    t_start = time.time()

    for path in files:
        enrich_file(path, args.s2_api_key, title_fallback=not args.no_title_fallback)

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed / 60:.1f} minutes")

    # Final summary
    logger.info("\nFinal summary:")
    for path in files:
        papers = read_legacy(path)
        stats = _stats(papers)
        name = path.stem.replace("-legacy", "").upper()
        logger.info(
            f"  {name}: {len(papers)} papers — "
            f"abs: {stats['abstract']}, "
            f"doi: {stats['doi']}, "
            f"pdf: {stats['pdf_url']}"
        )


def _commit_and_push(venue: str, enriched_so_far: int, total_searched: int, total_papers: int) -> None:
    """Git add, commit, and push the legacy file."""
    path = f"data/legacy/{venue}-legacy.jsonl.gz"
    try:
        subprocess.run(["git", "add", path], cwd=ROOT, check=True, capture_output=True)
        msg = (
            f"Enrich {venue.upper()} legacy: +{enriched_so_far} papers via S2 title search "
            f"({total_searched}/{total_papers} searched)"
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=ROOT, check=True, capture_output=True,
        )
        logger.info(f"  Committed ({enriched_so_far} enriched, {total_searched}/{total_papers} searched)")
    except subprocess.CalledProcessError as e:
        logger.warning(f"  Git operation failed: {e.stderr.decode()[:200] if e.stderr else e}")


def run_chunked(args) -> None:
    """Run chunked title-search enrichment for a single venue."""
    if not args.venue:
        logger.error("--venue is required for chunked mode")
        sys.exit(1)

    path = LEGACY_DIR / f"{args.venue}-legacy.jsonl.gz"
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    papers = read_legacy(path)
    logger.info(f"Loaded {len(papers)} papers from {path.name}")

    needs_search = []
    for i, p in enumerate(papers):
        if not p.get("doi") and p.get("title", "").strip():
            needs_search.append(i)

    logger.info(f"{len(needs_search)} papers need title search (no DOI)")

    if not needs_search:
        logger.info("Nothing to do")
        return

    total_enriched = 0
    total_searched = 0

    for chunk_start in range(0, len(needs_search), args.chunk_size):
        chunk_indices = needs_search[chunk_start:chunk_start + args.chunk_size]
        chunk_papers = [papers[i] for i in chunk_indices]
        chunk_num = chunk_start // args.chunk_size + 1
        total_chunks = (len(needs_search) + args.chunk_size - 1) // args.chunk_size

        logger.info(f"\n--- Chunk {chunk_num}/{total_chunks} ({len(chunk_indices)} papers) ---")

        results = fetch_abstracts_by_title(chunk_papers, api_key=args.s2_api_key)

        chunk_enriched = 0
        for subset_idx, result in results.items():
            paper_idx = chunk_indices[subset_idx]
            paper = papers[paper_idx]
            changed = False
            if result.get("abstract") and not paper.get("abstract"):
                paper["abstract"] = result["abstract"]
                changed = True
            if result.get("doi") and not paper.get("doi"):
                paper["doi"] = result["doi"]
                changed = True
            if result.get("pdf_url") and not paper.get("pdf_url"):
                paper["pdf_url"] = result["pdf_url"]
                changed = True
            if changed:
                chunk_enriched += 1

        total_enriched += chunk_enriched
        total_searched += len(chunk_indices)

        logger.info(f"  Chunk {chunk_num}: {chunk_enriched} enriched, {total_enriched} total")

        # Save after each chunk
        write_legacy(path, papers)
        _commit_and_push(args.venue, total_enriched, total_searched, len(needs_search))

    # Final stats
    total_abs = sum(1 for p in papers if p.get("abstract"))
    total_doi = sum(1 for p in papers if p.get("doi"))
    logger.info(f"\nDone. {total_enriched} papers enriched via title search.")
    logger.info(f"Final: {total_abs}/{len(papers)} abstracts, {total_doi}/{len(papers)} DOIs")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich legacy data with Semantic Scholar metadata"
    )
    parser.add_argument(
        "--s2-api-key",
        type=str,
        required=True,
        help="Semantic Scholar API key",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help="Enrich a single venue (e.g. 'cvpr'). Default: all (batch mode only).",
    )
    parser.add_argument(
        "--no-title-fallback",
        action="store_true",
        help="Skip title-based search (DOI batch only — much faster)",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked title-search mode (requires --venue)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Papers per chunk in chunked mode (default: 1000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.chunked:
        run_chunked(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
