#!/usr/bin/env python3
"""Patch COLT 2008–2010 papers with PDF links from learningtheory.org.

The learningtheory.org website hosts individual paper PDFs for COLT 2008–2010.
This script scrapes the proceedings page for each year, extracts
(title → pdf_url) pairs, matches them against colt-legacy.jsonl.gz by
normalised title, and fills in any missing or dead pdf_url fields.

Pages:
  2008  https://www.learningtheory.org/colt2008/
  2009  https://www.learningtheory.org/colt2009/
  2010  https://www.learningtheory.org/colt2010/conference-website/papers.html

Usage::

    python scripts/patch_colt_learningtheory.py
    python scripts/patch_colt_learningtheory.py --dry-run
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, clean_html, fuzzy_lookup, make_session, normalize_title, read_legacy, write_legacy

logger = logging.getLogger(__name__)

SOURCES = [
    {"year": "2008", "url": "https://www.learningtheory.org/colt2008/",
     "base": "https://www.learningtheory.org/colt2008/"},
    {"year": "2009", "url": "https://www.learningtheory.org/colt2009/",
     "base": "https://www.learningtheory.org/colt2009/"},
    {"year": "2010", "url": "https://www.learningtheory.org/colt2010/conference-website/papers.html",
     "base": "https://www.learningtheory.org/colt2010/conference-website/"},
]

def _parse_papers(html: str, base: str) -> list[tuple[str, str]]:
    """Extract (title, absolute_pdf_url) pairs from a learningtheory.org page."""
    results = []
    for m in re.finditer(
        r'<a\s+href=["\']?(papers/[^"\'>\s]+\.pdf)["\']?[^>]*>(.*?)</a>',
        html,
        re.DOTALL | re.IGNORECASE,
    ):
        title = clean_html(m.group(2))
        if title:
            results.append((title, base + m.group(1)))
    return results


COLT_LEGACY_FILE = LEGACY_DIR / "colt-legacy.jsonl.gz"

# domains that are dead or fragile — always replace even if pdf_url is set.
_DEAD_HOSTS = {
    "colt2008.cs.helsinki.fi",
    "colt2010.haifa.il.ibm.com",
    "www.cs.mcgill.ca",  # colt2009 personal server
}


def _is_dead(url: str) -> bool:
    return urlparse(url).netloc in _DEAD_HOSTS


def run(*, dry_run: bool = False) -> None:
    papers = read_legacy(COLT_LEGACY_FILE)
    logger.info("Loaded %d COLT papers", len(papers))

    session = make_session(retries=3, backoff_factor=1.0)
    session.headers.update({"User-Agent": "mlanthology-enrichment/1.0 (research bot)"})

    # Fetch and parse all source pages → year_index
    year_index: dict[str, dict[str, str]] = {}

    for i, src in enumerate(SOURCES):
        yr = src["year"]
        logger.info("[%d/%d] Fetching COLT %s …", i + 1, len(SOURCES), yr)
        try:
            resp = session.get(src["url"], timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("  Failed: %s", exc)
            continue

        pairs = _parse_papers(resp.text, src["base"])
        logger.info("  %d papers parsed", len(pairs))

        index: dict[str, str] = {}
        for title, pdf_url in pairs:
            norm = normalize_title(title)
            index[norm] = pdf_url
        year_index[yr] = index

        if i < len(SOURCES) - 1:
            time.sleep(1.0)

    # Match and patch
    n_added = n_replaced = n_skipped_good = n_unmatched = 0

    for paper in papers:
        yr = paper.get("year", "")
        if yr not in year_index:
            continue

        existing = paper.get("pdf_url", "").strip()

        # Skip if already on learningtheory.org
        if existing and "learningtheory.org" in existing:
            n_skipped_good += 1
            continue

        # Skip if has a good non-dead URL (arxiv, springer, etc.)
        if existing and not _is_dead(existing):
            n_skipped_good += 1
            continue

        pdf_url = fuzzy_lookup(paper["title"], year_index[yr])
        if pdf_url is None:
            n_unmatched += 1
            logger.debug("  UNMATCHED [%s] %s", yr, paper["title"][:70])
            continue

        if existing:
            n_replaced += 1
            action = "REPLACE"
        else:
            n_added += 1
            action = "ADD"

        if dry_run:
            logger.debug("  DRY-RUN %s [%s] %s → %s", action, yr, paper["title"][:60], pdf_url)
        else:
            paper["pdf_url"] = pdf_url

    logger.info(
        "Results: %d added, %d replaced (dead URL), %d already good (skipped), %d unmatched",
        n_added, n_replaced, n_skipped_good, n_unmatched,
    )

    if dry_run:
        logger.info("Dry-run — nothing written")
        return

    if n_added + n_replaced == 0:
        logger.info("No changes — not writing file")
        return

    write_legacy(COLT_LEGACY_FILE, papers, atomic=True)
    logger.info("Written back to %s", COLT_LEGACY_FILE.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.dry_run else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
