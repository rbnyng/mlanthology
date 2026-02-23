#!/usr/bin/env python3
"""Derive missing pdf_url values for pre-OpenReview ICLR papers (2013–2017).

ICLR 2013–2016 papers were submitted to arXiv; their venue_url is an arXiv
abstract page (http://arxiv.org/abs/{id}).  The PDF is at the same ID under
the /pdf/ path.

ICLR 2017 used OpenReview but pdf_url was not populated; venue_url is
https://openreview.net/forum?id={id} and the PDF lives at the same host
under /pdf?id={id}.

Neither requires a network request — both are pure URL rewrites.

Usage:
    python scripts/patch_iclr_legacy.py

    # Dry-run (show counts, no writes):
    python scripts/patch_iclr_legacy.py --dry-run
"""

import argparse
import logging
import re
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy

logger = logging.getLogger(__name__)


def _arxiv_pdf_url(venue_url: str) -> str:
    """Convert an arXiv abstract URL to a PDF URL.

    http://arxiv.org/abs/1234.5678  →  https://arxiv.org/pdf/1234.5678
    """
    m = re.match(r"https?://(?:www\.)?arxiv\.org/abs/(.+)", venue_url)
    if not m:
        return ""
    return f"https://arxiv.org/pdf/{m.group(1)}"


def _openreview_pdf_url(venue_url: str) -> str:
    """Convert an OpenReview forum URL to a PDF URL.

    https://openreview.net/forum?id=XYZ  →  https://openreview.net/pdf?id=XYZ
    """
    m = re.match(r"(https?://openreview\.net)/forum(\?.*)", venue_url)
    if not m:
        return ""
    return f"{m.group(1)}/pdf{m.group(2)}"


def patch_iclr_pdf_urls(papers: list[dict], *, dry_run: bool = False) -> dict[str, int]:
    """Derive pdf_url from venue_url for ICLR 2013–2017 papers.

    Only fills empty pdf_url fields — never overwrites existing data.
    Returns counts: {arxiv_added, openreview_added}.
    """
    arxiv_added = openreview_added = 0

    for p in papers:
        if p.get("venue") != "iclr":
            continue
        if p.get("pdf_url"):
            continue

        venue_url = p.get("venue_url", "")

        if "arxiv.org/abs/" in venue_url:
            pdf = _arxiv_pdf_url(venue_url)
            if pdf:
                if not dry_run:
                    p["pdf_url"] = pdf
                arxiv_added += 1

        elif "openreview.net/forum" in venue_url:
            pdf = _openreview_pdf_url(venue_url)
            if pdf:
                if not dry_run:
                    p["pdf_url"] = pdf
                openreview_added += 1

    return {"arxiv_added": arxiv_added, "openreview_added": openreview_added}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive pdf_url for ICLR 2013–2017 from venue_url (arXiv / OpenReview)"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing anything")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    legacy_path = LEGACY_DIR / "iclr-legacy.jsonl.gz"
    if not legacy_path.exists():
        logger.error(f"Legacy file not found: {legacy_path}")
        sys.exit(1)

    papers = read_legacy(legacy_path)
    iclr = [p for p in papers if p.get("venue") == "iclr" and p.get("authors")]
    logger.info(f"Loaded {len(papers)} records ({len(iclr)} ICLR papers)")

    before_pdf = sum(1 for p in iclr if p.get("pdf_url"))
    logger.info(f"PDF coverage before: {before_pdf}/{len(iclr)}")

    stats = patch_iclr_pdf_urls(papers, dry_run=args.dry_run)
    mode = "[DRY RUN] " if args.dry_run else ""
    logger.info(
        f"{mode}+{stats['arxiv_added']} arXiv pdf_urls  "
        f"+{stats['openreview_added']} OpenReview pdf_urls"
    )

    after_pdf = before_pdf + stats["arxiv_added"] + stats["openreview_added"]
    logger.info(f"PDF coverage after:  {after_pdf}/{len(iclr)}")

    if not args.dry_run:
        write_legacy(legacy_path, papers, atomic=True)
        logger.info(f"Written back to {legacy_path.name}")


if __name__ == "__main__":
    main()
