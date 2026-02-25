#!/usr/bin/env python3
"""Remove proceedings/conference metadata entries from legacy data.

Identifies and removes entries that are conference volume metadata rather than
actual research papers. These are characterized by:
- No pages (field is empty or missing)
- No abstract (field is empty or missing)
- Title contains proceedings-volume language (e.g. "Proceedings, Part I",
  "Proceedings - Florence, Italy") OR
  DBLP source_id is volume-level (e.g. conf/eccv/2016w1) with no author token

Run on all venues:
    python scripts/remove_proceedings_metadata.py

Single venue, dry-run:
    python scripts/remove_proceedings_metadata.py --venue eccv --dry-run
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

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy

logger = logging.getLogger(__name__)

# Matches proceedings-volume titles:
#   "Proceedings, Part I/II/III/IV..."
#   "Proceedings - Florence, Italy"
#   "Proceedings, Vol ..."
#   "Proceedings of the 35th ..."
#   "2016 Proceedings"
_VOL_TITLE_RE = re.compile(
    r"Proceedings[,\s]*(Part\s+[IVX\d]+|Vol|of\s+the\s+\d)"
    r"|Proceedings\s*[-\u2013]\s+\w"
    r"|\d{4}\s+Proceedings",
    re.IGNORECASE,
)

# DBLP volume-level source_id: conf/venue/YEAR or conf/venue/YEAR-suffix
# where the suffix contains only lowercase letters, digits, and hyphens.
# This excludes paper-level IDs like conf/aaai/AlshareefBSH22 (CamelCase author
# abbreviation) and disambiguation IDs like conf/aaai/0001W22 (uppercase W).
_VOL_SRC_RE = re.compile(r"^(?:conf|journals)/\w+/\d{4}[-a-z0-9]*$")


def is_proceedings_entry(paper: dict) -> bool:
    """Return True if this record is a proceedings volume, not a paper.

    Requires ALL of:
      1. No abstract
      2. No pages
      3. Volume-like title OR volume-level DBLP source_id
    """
    if paper.get("abstract", "").strip():
        return False
    if paper.get("pages", "").strip():
        return False

    title = paper.get("title", "")
    src = paper.get("source_id", "")

    return bool(_VOL_TITLE_RE.search(title)) or bool(_VOL_SRC_RE.match(src))


def process_venue(venue_name: str, dry_run: bool = False) -> dict:
    """Remove proceedings entries from a venue's legacy file.

    Returns: {removed: int, kept: int}
    """
    legacy_path = LEGACY_DIR / f"{venue_name}-legacy.jsonl.gz"
    if not legacy_path.exists():
        logger.warning(f"File not found: {legacy_path.name}")
        return {"removed": 0, "kept": 0}

    papers = read_legacy(legacy_path)
    original_count = len(papers)

    kept = []
    removed = []
    for p in papers:
        if is_proceedings_entry(p):
            removed.append(p)
        else:
            kept.append(p)

    if removed:
        logger.info(f"{venue_name}: removing {len(removed)} / {original_count}")
        for p in removed:
            logger.info(f"  REMOVE [{p.get('year', '?')}] {p['title']}")
    else:
        logger.info(f"{venue_name}: nothing to remove ({original_count} records)")

    if not dry_run and removed:
        write_legacy(legacy_path, kept, atomic=True)
        logger.info(f"  Written {len(kept)} papers to {legacy_path.name}")

    return {"removed": len(removed), "kept": len(kept)}


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

    if args.venue:
        venues = [args.venue]
    else:
        venues = sorted(
            f.stem.replace("-legacy.jsonl", "")
            for f in LEGACY_DIR.glob("*-legacy.jsonl.gz")
        )

    mode = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"{mode}Processing {len(venues)} venue(s)")

    grand_removed = 0
    grand_kept = 0

    for venue in venues:
        stats = process_venue(venue, dry_run=args.dry_run)
        grand_removed += stats["removed"]
        grand_kept += stats["kept"]

    logger.info(f"{mode}Summary: {grand_removed} removed, {grand_kept} kept")


if __name__ == "__main__":
    main()
