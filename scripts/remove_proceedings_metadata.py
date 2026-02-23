#!/usr/bin/env python3
"""Remove proceedings/conference metadata entries from legacy data.

Identifies and removes entries that are conference metadata rather than
actual research papers. These are characterized by:
- Generic titles containing "proceedings", "conference", "symposium", etc.
- No pages (field is empty or missing)
- No abstract (field is empty or missing)
- Often only conference organizers as "authors"

Run on all venues:
    python scripts/remove_proceedings_metadata.py

Single venue, dry-run:
    python scripts/remove_proceedings_metadata.py --venue aaai --dry-run
"""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy

logger = logging.getLogger(__name__)


def is_proceedings_entry(paper: dict) -> bool:
    """Check if paper is a proceedings/conference metadata entry.

    Heuristics:
    - Title contains generic conference keywords
    - No pages
    - No abstract
    """
    title = paper.get("title", "").lower()
    has_pages = bool(paper.get("pages"))
    has_abstract = bool(paper.get("abstract"))

    # Generic conference metadata keywords
    keywords = [
        "proceedings of the",
        "conference on",
        "symposium on",
        "workshop on",
        "IEEE International Conference",
        "European Conference",
        "International Workshop",
        "International Conference",
    ]

    is_generic = any(keyword in title for keyword in keywords)

    # Metadata entries have no pages and no abstract
    is_likely_metadata = is_generic and not has_pages and not has_abstract

    return is_likely_metadata


def process_venue(venue_name: str, dry_run: bool = False) -> dict:
    """Remove proceedings entries from a venue.

    Returns: {removed: int, kept: int}
    """
    legacy_path = LEGACY_DIR / f"{venue_name}-legacy.jsonl.gz"
    if not legacy_path.exists():
        logger.warning(f"File not found: {legacy_path.name}")
        return {"removed": 0, "kept": 0}

    papers = read_legacy(legacy_path)
    original_count = len(papers)

    # Filter out proceedings entries
    filtered_papers = [p for p in papers if not is_proceedings_entry(p)]
    removed_count = original_count - len(filtered_papers)

    logger.info(f"{venue_name}: {removed_count} removed, {len(filtered_papers)} kept")

    if not dry_run and removed_count > 0:
        write_legacy(legacy_path, filtered_papers, atomic=True)
        logger.info(f"  Written {len(filtered_papers)} papers to {legacy_path.name}")

    return {"removed": removed_count, "kept": len(filtered_papers)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remove proceedings/conference metadata entries from legacy data"
    )
    ap.add_argument(
        "--venue",
        help="Process a single venue (e.g., 'aaai', 'eccv'); omit for all",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Get venues to process
    if args.venue:
        venues = [args.venue]
    else:
        # Get all legacy venues
        venues = [f.stem.replace("-legacy.jsonl", "") for f in LEGACY_DIR.glob("*-legacy.jsonl.gz")]
        venues = sorted(venues)

    mode = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"{mode}Processing {len(venues)} venue(s)")

    grand_removed = 0
    grand_kept = 0

    for venue in venues:
        stats = process_venue(venue, dry_run=args.dry_run)
        grand_removed += stats["removed"]
        grand_kept += stats["kept"]

    print()
    logger.info(f"{mode}Summary:")
    logger.info(f"  Total removed: {grand_removed}")
    logger.info(f"  Total kept: {grand_kept}")
    logger.info(f"  Final dataset size: {grand_kept}")


if __name__ == "__main__":
    main()
