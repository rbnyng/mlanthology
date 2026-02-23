#!/usr/bin/env python3
"""Fix multiple consecutive spaces in paper titles and abstracts.

This script normalizes whitespace in paper metadata:
  - Replaces multiple consecutive spaces with a single space
  - Handles tabs and other whitespace characters
  - Preserves intentional formatting (e.g., LaTeX commands)

Run on all venues:
    python scripts/fix_double_spaces.py

Single venue, dry-run to see what would change:
    python scripts/fix_double_spaces.py --venue icml --dry-run

Only process legacy data:
    python scripts/fix_double_spaces.py --legacy-only --dry-run
"""

import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, PAPERS_DIR

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces to single space."""
    if not text or not isinstance(text, str):
        return text
    # Replace multiple spaces with single space
    return re.sub(r' {2,}', ' ', text)


def process_legacy_papers(
    venue_name: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Process legacy JSONL.GZ files.

    Returns stats: {changed: int, titles_fixed: int, abstracts_fixed: int}
    """
    stats = {"changed": 0, "titles_fixed": 0, "abstracts_fixed": 0}

    # Get files to process
    if venue_name:
        pattern = f"{venue_name}-legacy.jsonl.gz"
        files = list(LEGACY_DIR.glob(pattern))
        if not files:
            logger.error(f"No legacy files matching: {pattern}")
            return stats
    else:
        files = sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz"))

    for legacy_path in files:
        # Read papers
        papers: list[dict] = []
        with gzip.open(legacy_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(json.loads(line))

        logger.info(f"\n{legacy_path.name}: {len(papers)} papers")

        # Process
        venue_titles_fixed = 0
        venue_abstracts_fixed = 0

        for paper in papers:
            # Fix title
            if paper.get("title"):
                original_title = paper["title"]
                normalized_title = normalize_whitespace(original_title)
                if normalized_title != original_title:
                    if not dry_run:
                        paper["title"] = normalized_title
                    venue_titles_fixed += 1

            # Fix abstract
            if paper.get("abstract"):
                original_abstract = paper["abstract"]
                normalized_abstract = normalize_whitespace(original_abstract)
                if normalized_abstract != original_abstract:
                    if not dry_run:
                        paper["abstract"] = normalized_abstract
                    venue_abstracts_fixed += 1

        # Write back if changes made
        if not dry_run and (venue_titles_fixed > 0 or venue_abstracts_fixed > 0):
            tmp_path = legacy_path.with_suffix(".jsonl.gz.tmp")
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                for p in papers:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            tmp_path.replace(legacy_path)

        mode = "[DRY RUN] " if dry_run else ""
        logger.info(
            f"  {mode}Fixed {venue_titles_fixed} titles, {venue_abstracts_fixed} abstracts"
        )

        stats["titles_fixed"] += venue_titles_fixed
        stats["abstracts_fixed"] += venue_abstracts_fixed
        if venue_titles_fixed > 0 or venue_abstracts_fixed > 0:
            stats["changed"] += 1

    return stats


def process_current_papers(
    venue_name: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Process current JSON.GZ files.

    Returns stats: {changed: int, titles_fixed: int, abstracts_fixed: int}
    """
    stats = {"changed": 0, "titles_fixed": 0, "abstracts_fixed": 0}

    # Get files to process
    if venue_name:
        pattern = f"{venue_name}.json.gz"
        files = list(PAPERS_DIR.glob(pattern))
        if not files:
            logger.error(f"No paper files matching: {pattern}")
            return stats
    else:
        files = sorted(PAPERS_DIR.glob("*.json.gz"))

    for papers_path in files:
        # Read papers
        with gzip.open(papers_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        papers = data if isinstance(data, list) else data.get("papers", [])
        logger.info(f"\n{papers_path.name}: {len(papers)} papers")

        # Process
        venue_titles_fixed = 0
        venue_abstracts_fixed = 0

        for paper in papers:
            # Fix title
            if paper.get("title"):
                original_title = paper["title"]
                normalized_title = normalize_whitespace(original_title)
                if normalized_title != original_title:
                    if not dry_run:
                        paper["title"] = normalized_title
                    venue_titles_fixed += 1

            # Fix abstract
            if paper.get("abstract"):
                original_abstract = paper["abstract"]
                normalized_abstract = normalize_whitespace(original_abstract)
                if normalized_abstract != original_abstract:
                    if not dry_run:
                        paper["abstract"] = normalized_abstract
                    venue_abstracts_fixed += 1

        # Write back if changes made
        if not dry_run and (venue_titles_fixed > 0 or venue_abstracts_fixed > 0):
            tmp_path = papers_path.with_suffix(".json.gz.tmp")
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                if isinstance(data, list):
                    json.dump(papers, f, ensure_ascii=False)
                else:
                    data["papers"] = papers
                    json.dump(data, f, ensure_ascii=False)
            tmp_path.replace(papers_path)

        mode = "[DRY RUN] " if dry_run else ""
        logger.info(
            f"  {mode}Fixed {venue_titles_fixed} titles, {venue_abstracts_fixed} abstracts"
        )

        stats["titles_fixed"] += venue_titles_fixed
        stats["abstracts_fixed"] += venue_abstracts_fixed
        if venue_titles_fixed > 0 or venue_abstracts_fixed > 0:
            stats["changed"] += 1

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fix multiple consecutive spaces in paper titles and abstracts"
    )
    ap.add_argument(
        "--venue",
        help="Process a single venue (e.g., 'icml', 'aistats'); omit for all",
    )
    ap.add_argument(
        "--legacy-only",
        action="store_true",
        help="Only process legacy data files",
    )
    ap.add_argument(
        "--current-only",
        action="store_true",
        help="Only process current paper files",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    legacy_stats = {"changed": 0, "titles_fixed": 0, "abstracts_fixed": 0}
    current_stats = {"changed": 0, "titles_fixed": 0, "abstracts_fixed": 0}

    # Process legacy
    if not args.current_only:
        logger.info("=== Processing Legacy Data ===")
        legacy_stats = process_legacy_papers(venue_name=args.venue, dry_run=args.dry_run)

    # Process current
    if not args.legacy_only:
        logger.info("\n=== Processing Current Data ===")
        current_stats = process_current_papers(
            venue_name=args.venue, dry_run=args.dry_run
        )

    # Summary
    mode = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"\n--- Summary ---")
    logger.info(
        f"  {mode}Legacy: {legacy_stats['changed']} files changed, "
        f"{legacy_stats['titles_fixed']} titles, {legacy_stats['abstracts_fixed']} abstracts"
    )
    logger.info(
        f"  {mode}Current: {current_stats['changed']} files changed, "
        f"{current_stats['titles_fixed']} titles, {current_stats['abstracts_fixed']} abstracts"
    )
    logger.info(
        f"  {mode}Total: {legacy_stats['titles_fixed'] + current_stats['titles_fixed']} titles, "
        f"{legacy_stats['abstracts_fixed'] + current_stats['abstracts_fixed']} abstracts fixed"
    )


if __name__ == "__main__":
    main()
