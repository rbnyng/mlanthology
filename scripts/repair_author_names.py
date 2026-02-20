#!/usr/bin/env python3
"""One-shot repair script: re-normalise author names in stored JSON data.

Applies the full author-name cleaning pipeline from common.py to every
paper in data/papers/*.json.gz, fixing:
- HTML entities in names (&#352; -> Š)
- Equal-contribution markers (*)
- Degree/pronoun annotations: (PhD), (He/Him)
- Parenthetical nicknames/former names: (Kate), (Basha)
- BibTeX accent commands (\\L -> Ł)
- Orphaned leading-hyphen family names (-Malvajerdi -> Sharifi-Malvajerdi)
- Punctuation-only / junk name fields (., (), {namdar...@uber.com)

Writes files back in-place only when changes are detected.
Run from the repository root:

    python -m scripts.repair_author_names
"""

import gzip
import json
import logging
import sys
from pathlib import Path

from scripts.utils import ROOT, PAPERS_DIR

from adapters.common import (
    _clean_raw_author_name,
    _fix_leading_hyphen_family,
    _fix_misplaced_initial,
    _fix_punctuation_only_fields,
    _fix_single_letter_family,
    read_venue_json,
    repair_mojibake,
    slugify_author,
)

logger = logging.getLogger(__name__)


def _normalize_author(author: dict) -> dict:
    """Apply the full cleaning pipeline to a single author dict."""
    cleaned = {
        "given": _clean_raw_author_name(repair_mojibake(author.get("given") or "")),
        "family": _clean_raw_author_name(repair_mojibake(author.get("family") or "")),
    }
    cleaned = _fix_misplaced_initial(cleaned)
    cleaned = _fix_single_letter_family(cleaned)
    cleaned = _fix_leading_hyphen_family(cleaned)
    cleaned = _fix_punctuation_only_fields(cleaned)
    cleaned["slug"] = slugify_author(cleaned)
    return cleaned


def repair_file(path: Path) -> int:
    """Repair author names in a single .json.gz file.  Returns count of changed papers."""
    data = read_venue_json(path)

    changed = 0
    for paper in data.get("papers", []):
        original_authors = paper.get("authors", [])
        new_authors = []
        for a in original_authors:
            repaired = _normalize_author(a)
            # Drop authors that ended up completely empty (junk entries)
            if repaired.get("given") or repaired.get("family"):
                new_authors.append(repaired)
        if new_authors != original_authors:
            paper["authors"] = new_authors
            changed += 1

    if changed:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return changed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = PAPERS_DIR
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    json_files = sorted(data_dir.glob("*.json.gz"))
    if not json_files:
        logger.warning("No data files found in %s", data_dir)
        return

    total_changed = 0
    for path in json_files:
        n = repair_file(path)
        if n:
            logger.info(f"  {path.name}: repaired {n} paper(s)")
            total_changed += n

    if total_changed:
        logger.info(f"Total: repaired authors in {total_changed} paper(s)")
    else:
        logger.info("No repairs needed — all author names are clean.")


if __name__ == "__main__":
    main()
