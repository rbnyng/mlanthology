#!/usr/bin/env python3
"""Repair broken CVF venue_url / pdf_url fields caused by inconsistent
author-name normalisation on openaccess.thecvf.com.

Background
----------
CVF's listing-page hrefs normalise diacritics via NFD + strip-combining
(e.g. Souček → Soucek), but for some conference-years the actual files on
disk were named by stripping non-ASCII bytes directly (Souček → Souek).
The two schemes agree for most characters (ö → o in both) but diverge for
characters whose NFD decomposition adds a base letter that is absent from the
direct-strip result (č = c + combining-caron → c vs ∅).

This script:
1. Scans data/papers/cvpr-*, iccv-*, wacv-*.json.gz for papers whose first
   author has a non-ASCII family name.
2. GETs the stored venue_url; if it returns 200 the record is already correct.
3. If it 404s, tries each alternative normalisation and updates the record
   (both venue_url and pdf_url) with the first slug that returns 200.
4. Writes changed files back in-place.

Usage::

    python scripts/repair_cvf_urls.py           # all CVF files
    python scripts/repair_cvf_urls.py --venue cvpr  # single venue
    python scripts/repair_cvf_urls.py --dry-run     # check only, no writes
    python scripts/repair_cvf_urls.py --limit 20    # first 20 candidates only
"""

import argparse
import gzip
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from adapters.common import read_venue_json, write_venue_json
from adapters.http import fetch_with_retry
from scripts.utils import PAPERS_DIR

logger = logging.getLogger(__name__)

CVF_PREFIXES = ("cvpr-", "iccv-", "wacv-")
_REQUEST_DELAY = 0.12  # seconds between HEAD/GET checks (~8 req/s)



def _author_slug_from_url(url: str) -> str:
    """Extract the author-name segment (first component after the last /)."""
    filename = url.rsplit("/", 1)[-1]
    return filename.split("_", 1)[0]


def _replace_author_slug(url: str, new_slug: str) -> str:
    """Swap the author segment in a CVF URL path."""
    dir_part, filename = url.rsplit("/", 1)
    rest = filename.split("_", 1)[1] if "_" in filename else filename
    return f"{dir_part}/{new_slug}_{rest}"


def _candidate_slugs(family: str) -> list[str]:
    """Return alternative author-name slugs to try, ordered by likelihood."""
    nfd = unicodedata.normalize("NFKD", family).encode("ascii", "ignore").decode("ascii")
    direct = re.sub(r"[^\x00-\x7F]", "", family)
    seen: list[str] = []
    for s in (nfd, direct):
        if s and s not in seen:
            seen.append(s)
    return seen


def _url_ok(url: str) -> bool:
    """Return True if *url* resolves to HTTP 200."""
    try:
        r = fetch_with_retry(url)
        return r.status_code == 200
    except Exception:
        return False



def repair_file(
    path: Path,
    *,
    dry_run: bool = False,
    limit: int = 0,
) -> tuple[int, int]:
    """Repair broken CVF URLs in *path*.

    Returns (candidates_checked, papers_fixed).
    """
    data = read_venue_json(path)
    papers = data.get("papers", [])

    # collect candidates: papers whose first-author family name contains
    # non-ASCII characters (so NFD and direct-strip may diverge).
    candidates = []
    for paper in papers:
        authors = paper.get("authors", [])
        if not authors:
            continue
        family = authors[0].get("family", "")
        if not family:
            continue
        if all(ord(c) < 128 for c in family):
            continue  # purely ASCII — both normalisations agree, skip
        venue_url = paper.get("venue_url", "")
        if not venue_url:
            continue
        candidates.append(paper)

    if not candidates:
        return 0, 0

    logger.info(
        f"\n{'=' * 60}\n{path.name}: {len(candidates)} candidate(s) with "
        f"non-ASCII first-author family names"
    )

    if limit > 0:
        candidates = candidates[:limit]
        logger.info(f"  Limited to first {len(candidates)} candidate(s)")

    checked = 0
    fixed = 0

    for paper in candidates:
        family = paper["authors"][0]["family"]
        venue_url = paper.get("venue_url", "")
        pdf_url = paper.get("pdf_url", "")
        current_slug = _author_slug_from_url(venue_url)

        checked += 1
        time.sleep(_REQUEST_DELAY)

        if _url_ok(venue_url):
            logger.debug(f"  OK  {paper['title'][:60]}")
            continue

        logger.info(f"  BROKEN [{current_slug!r}]: {paper['title'][:60]}")

        # Try alternative slugs
        fixed_venue = None
        fixed_pdf = None
        for slug in _candidate_slugs(family):
            if slug == current_slug:
                continue
            alt_venue = _replace_author_slug(venue_url, slug)
            time.sleep(_REQUEST_DELAY)
            if _url_ok(alt_venue):
                fixed_venue = alt_venue
                if pdf_url:
                    fixed_pdf = _replace_author_slug(pdf_url, slug)
                logger.info(f"    → fixed with slug {slug!r}")
                break
        else:
            logger.warning(
                f"    Could not find working URL for {paper['title'][:60]!r}"
            )
            continue

        if not dry_run:
            paper["venue_url"] = fixed_venue
            if fixed_pdf:
                paper["pdf_url"] = fixed_pdf
        fixed += 1

    if fixed > 0 and not dry_run:
        write_venue_json(
            data["venue"],
            data["year"],
            papers,
            path.parent,
            filename=path.name.replace(".json.gz", ""),
        )
        logger.info(f"  Wrote {path.name} ({fixed} fix(es))")
    elif fixed > 0:
        logger.info(f"  DRY RUN — would fix {fixed} paper(s)")

    return checked, fixed



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair broken CVF venue_url/pdf_url fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help="Limit to a single venue prefix, e.g. 'cvpr', 'iccv', 'wacv'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check URLs and report findings without writing any changes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N candidates per file (for testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    files: list[Path] = []
    for f in sorted(PAPERS_DIR.glob("*.json.gz")):
        if args.venue:
            if not f.name.startswith(f"{args.venue}-"):
                continue
        else:
            if not any(f.name.startswith(p) for p in CVF_PREFIXES):
                continue
        files.append(f)

    if not files:
        logger.error("No matching files found.")
        sys.exit(1)

    logger.info(f"Scanning {len(files)} CVF file(s)…")

    t0 = time.time()
    total_checked = total_fixed = 0
    for path in files:
        checked, fixed = repair_file(path, dry_run=args.dry_run, limit=args.limit)
        total_checked += checked
        total_fixed += fixed

    elapsed = time.time() - t0
    action = "would fix" if args.dry_run else "fixed"
    logger.info(
        f"\nDone in {elapsed:.0f}s — "
        f"checked {total_checked} candidate(s), {action} {total_fixed}"
    )


if __name__ == "__main__":
    main()
