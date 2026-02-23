#!/usr/bin/env python3
"""Enrich CVPRW legacy data with abstracts from CVF OpenAccess pages.

Fetches abstracts from openaccess.thecvf.com for CVPRW papers that have
venue URLs but no abstracts, using the same logic as the CVF adapter.
"""

import logging
import re
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, read_legacy, write_legacy
from adapters.http import fetch_with_retry
from adapters.common import strip_html

logger = logging.getLogger(__name__)


def fetch_abstract_and_pdf_from_cvf_page(venue_url: str) -> tuple[str, str]:
    """Fetch abstract and PDF URL from a CVF paper HTML page.

    Returns:
        (abstract_text, pdf_url) - empty strings if not found

    Tries for abstract:
    1. <div id="abstract">content</div>
    2. Fallback: text after "Abstract" heading

    Tries for PDF URL:
    1. <a href="...pdf">pdf</a>
    """
    # Convert http to https
    venue_url = venue_url.replace("http://openaccess", "https://openaccess")

    try:
        resp = fetch_with_retry(venue_url, timeout=30)
    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.debug(f"Failed to fetch {venue_url}: {e}")
        return "", ""

    abstract = ""
    pdf_url = ""

    # Extract abstract - Try primary: <div id="abstract">...</div>
    match = re.search(
        r'<div\s+id="abstract"[^>]*>(.*?)</div>',
        resp.text, re.DOTALL,
    )
    if match:
        abstract = strip_html(match.group(1))
    else:
        # Fallback: text after "Abstract" heading
        match = re.search(
            r'(?:Abstract|ABSTRACT)\s*</?\w+[^>]*>\s*(.*?)(?:<div|<h[23]|<br\s*/?\s*><br)',
            resp.text, re.DOTALL,
        )
        if match:
            abstract = strip_html(match.group(1))

    # Extract PDF URL - look for pdf link
    pdf_match = re.search(r'<a\s+href="([^"]*\.pdf)"[^>]*>\s*(?:pdf|PDF)\s*</a>', resp.text, re.IGNORECASE)
    if pdf_match:
        pdf_path = pdf_match.group(1)
        # Make absolute URL if needed
        if pdf_path.startswith("http"):
            pdf_url = pdf_path
        elif pdf_path.startswith("/"):
            # Extract domain from venue_url
            from urllib.parse import urlparse
            parsed = urlparse(venue_url)
            pdf_url = f"{parsed.scheme}://{parsed.netloc}{pdf_path}"
        else:
            # Relative path - make it absolute
            base_dir = "/".join(venue_url.split("/")[:-1])
            pdf_url = f"{base_dir}/{pdf_path}"

    return abstract, pdf_url


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich CVPRW with abstracts and PDF URLs from CVF OpenAccess pages"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    ap.add_argument(
        "--abstracts-only",
        action="store_true",
        help="Only fetch abstracts, skip PDF URLs",
    )
    ap.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers to fetch (for testing)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Read CVPRW legacy data
    legacy_path = LEGACY_DIR / "cvprw-legacy.jsonl.gz"
    logger.info(f"Reading {legacy_path.name}")
    papers = read_legacy(legacy_path)

    # Find papers needing abstracts or PDFs
    targets = []
    for i, p in enumerate(papers):
        if p.get("venue_url") and "openaccess.thecvf.com" in p.get("venue_url", ""):
            needs_abstract = not p.get("abstract")
            needs_pdf = not p.get("pdf_url")
            if needs_abstract or needs_pdf:
                targets.append((i, p, needs_abstract, needs_pdf))
                if args.max_papers and len(targets) >= args.max_papers:
                    break

    logger.info(f"Found {len(targets)} papers to enrich")

    # Fetch abstracts and/or PDFs
    abstracts_added = 0
    pdfs_added = 0
    for n, (idx, paper, needs_abstract, needs_pdf) in enumerate(targets, 1):
        venue_url = paper["venue_url"]
        abstract, pdf_url = fetch_abstract_and_pdf_from_cvf_page(venue_url)

        updated = False
        status_parts = []

        if needs_abstract and abstract:
            if not args.dry_run:
                papers[idx]["abstract"] = abstract
            abstracts_added += 1
            updated = True
            status_parts.append("✓abs")

        if needs_pdf and pdf_url and not args.abstracts_only:
            if not args.dry_run:
                papers[idx]["pdf_url"] = pdf_url
            pdfs_added += 1
            updated = True
            status_parts.append("✓pdf")

        if not updated:
            status_parts.append("✗")

        status = "/".join(status_parts)
        logger.info(f"  [{n}/{len(targets)}] {status} {paper['title'][:50]}")

        # Progress update
        if n % 50 == 0 or n == len(targets):
            logger.info(
                f"  Progress: {n}/{len(targets)}, "
                f"{abstracts_added} abstracts, {pdfs_added} PDFs added so far"
            )

    # Write back
    mode = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"\n{mode}Added {abstracts_added} abstracts, {pdfs_added} PDFs")

    if not args.dry_run and (abstracts_added > 0 or pdfs_added > 0):
        write_legacy(legacy_path, papers, atomic=True)
        logger.info(f"Wrote changes back to {legacy_path.name}")


if __name__ == "__main__":
    import argparse
    main()
