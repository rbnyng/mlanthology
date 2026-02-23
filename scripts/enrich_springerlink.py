#!/usr/bin/env python3
"""Enrich legacy data with abstracts scraped from SpringerLink article pages.

The Springer Nature Meta API only covers LNCS/LNAI book chapters (10.1007/
prefix).  MLJ and similar journals also use the older 10.1023/ Kluwer prefix,
and old 10.1007/BF*/s* journal DOIs which the Meta API returns without
abstracts.

This script scrapes https://link.springer.com/article/{doi} directly, which
has full abstract text for all Springer/Kluwer journal articles.

Applicable venues:
  mlj  — Machine Learning Journal (Springer/Kluwer, 1986–present)

Usage::

    python scripts/enrich_springerlink.py --venue mlj
    python scripts/enrich_springerlink.py --venue mlj --dry-run
    python scripts/enrich_springerlink.py --venue mlj --limit 20
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, clean_html, make_session, read_legacy, write_legacy

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.6  # seconds between requests — polite crawl rate

_SESSION = make_session(retries=4, backoff_factor=1.0)
_SESSION.headers.update({
    "User-Agent": "mlanthology-enrichment/1.0 (research bot; https://github.com/rbnyng/mlanthology)",
    "Accept": "text/html,application/xhtml+xml",
})

_ABSTRACT_SECTION_RE = re.compile(
    r'<section\b[^>]*\bdata-title="Abstract"[^>]*>(.*?)</section>',
    re.DOTALL,
)


def fetch_abstract(doi: str) -> str | None:
    """Fetch the abstract for *doi* from the SpringerLink article page.

    Returns the abstract text, or None if the page has no abstract section
    (e.g. editorials, corrections) or the DOI cannot be resolved.
    """
    url = f"https://link.springer.com/article/{doi}"
    try:
        resp = _SESSION.get(url, timeout=25, allow_redirects=True)
    except Exception as exc:
        logger.warning("  HTTP error %s for %s: %s", doi, url, exc)
        return None

    if resp.status_code == 404:
        logger.debug("  404: %s", doi)
        return None
    if resp.status_code != 200:
        logger.warning("  HTTP %d for %s", resp.status_code, doi)
        return None

    m = _ABSTRACT_SECTION_RE.search(resp.text)
    if not m:
        logger.debug("  no abstract section: %s", doi)
        return None

    text = clean_html(m.group(1), replace_tags_with=" ")
    # strip leading "Abstract" word that Springer somtimes includes in the HTML
    if text.lower().startswith("abstract"):
        text = text[8:].lstrip()

    return text or None


_SPRINGER_DOI_RE = re.compile(r"^10\.(1007|1023)/")


def _is_springer_doi(doi: str) -> bool:
    return bool(_SPRINGER_DOI_RE.match(doi))


def enrich_file(
    path: Path,
    *,
    dry_run: bool = False,
    limit: int = 0,
) -> None:
    papers = read_legacy(path)
    total = len(papers)
    with_abstract = sum(1 for p in papers if p.get("abstract", "").strip())
    logger.info("Loaded %d papers, %d with abstract", total, with_abstract)

    candidates = [
        p for p in papers
        if _is_springer_doi(p.get("doi", ""))
        and not p.get("abstract", "").strip()
    ]
    logger.info("%d Springer candidates missing abstract", len(candidates))

    if not candidates:
        logger.info("Nothing to enrich, exiting")
        return

    if limit > 0:
        candidates = candidates[:limit]
        logger.info("Limiting to %d papers (--limit)", len(candidates))

    if dry_run:
        logger.info("Dry-run: would fetch %d SpringerLink pages", len(candidates))
        return

    n_added = n_nosection = 0
    last_req = 0.0

    for i, paper in enumerate(candidates, 1):
        doi = paper["doi"]

        # Rate limiting
        elapsed = time.time() - last_req
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        last_req = time.time()

        abstract = fetch_abstract(doi)

        if abstract:
            paper["abstract"] = abstract
            src = paper.get("source", "")
            paper["source"] = (src + "+springerlink") if src else "springerlink"
            n_added += 1
            if i % 50 == 0 or i == len(candidates):
                logger.info(
                    "  [%d/%d] +abstract: %s | %s…",
                    i, len(candidates), doi, abstract[:50],
                )
        else:
            n_nosection += 1

    logger.info(
        "Results: %d abstracts added, %d no-section (editorials/corrections)",
        n_added, n_nosection,
    )

    if n_added == 0:
        logger.info("No changes — not writing file")
        return

    write_legacy(path, papers, atomic=True)
    logger.info("Written back to %s", path.name)

    # Final coverage
    with_abstract_after = sum(1 for p in papers if p.get("abstract", "").strip())
    logger.info(
        "Coverage: %d/%d (%.1f%%)",
        with_abstract_after, total, with_abstract_after / total * 100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--venue",
        required=True,
        help="Venue name, e.g. 'mlj'. Looks for data/legacy/{venue}-legacy.jsonl.gz",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N candidates (for testing)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.dry_run else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    path = LEGACY_DIR / f"{args.venue}-legacy.jsonl.gz"
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    logger.info("Enriching %s from SpringerLink…", path.name)
    enrich_file(path, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
