#!/usr/bin/env python3
"""Enrich UAI legacy papers (1985–2013) with arXiv PDF links.

The UAI proceedings for 1985–2013 were bulk-uploaded to arXiv in 2012–2013
as open-access PDFs.  Each year's proceedings has a corresponding arXiv index
paper (itself an arXiv submission) that lists all papers with their arXiv IDs.

This script:
  1. Fetches all 29 arXiv index pages and parses out (title → arXiv ID) pairs.
  2. Matches those titles against UAI papers in uai-legacy.jsonl.gz using
     normalised-title exact match, then SequenceMatcher fallback (≥0.92).
  3. For matched papers in years 1985–2013:
       - If pdf_url is empty          → sets it to https://arxiv.org/pdf/{id}
       - If pdf_url is a non-arXiv URL → replaces it  (Springer / auai.org /
         personal pages are dead or paywalled; arXiv is stable + open-access)
       - If pdf_url already points to arxiv.org → skips (already good)

Usage::

    python scripts/enrich_uai_arxiv.py
    python scripts/enrich_uai_arxiv.py --dry-run
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

from scripts.utils import LEGACY_DIR, clean_html, fuzzy_lookup, make_session, normalize_title, read_legacy, write_legacy

logger = logging.getLogger(__name__)

INDEX_ARXIV_IDS = [
    # Early proceedings bulk-uploaded April 2013
    "1304.4182",  # UAI 1985
    "1304.3859",  # UAI 1986
    "1304.3857",  # UAI 1987
    "1304.3856",  # UAI 1990
    "1304.3855",  # UAI 1991
    "1304.3854",  # UAI 1992
    "1304.3853",  # UAI 1993
    "1304.3852",  # UAI 1994
    "1304.3851",  # UAI 1995
    "1304.3849",  # UAI 1996
    "1304.3848",  # UAI 1997
    "1304.3847",  # UAI 1998
    "1304.3846",  # UAI 1999
    "1304.3844",  # UAI 2000
    "1304.3843",  # UAI 2002
    "1304.3842",  # UAI 2003
    # Later uploads
    "1301.4604",  # UAI 2004 (alt)
    "1301.4606",  # UAI 2005
    "1301.4607",  # UAI 2001 (confirmed)
    "1301.4608",  # UAI 2006
    "1208.5154",  # UAI 2007
    "1208.5155",  # UAI 2008
    "1208.5159",  # UAI 2009 (alt)
    "1208.5160",  # UAI 2010 (alt)
    "1208.5161",  # UAI 2004 (confirmed)
    "1206.3959",  # UAI 2011 (alt)
    "1205.2596",  # UAI 2011
    "1205.2597",  # UAI 2010 (confirmed)
    "1309.7971",  # UAI 2013 (confirmed)
]


def _parse_index_page(html: str) -> list[tuple[str, str]]:
    """Return a list of (title, arxiv_id) pairs from an arXiv index HTML page.

    The page structure (consistent across all UAI index submissions) is::

        <!-- Paper Title -->
        <dl>
          <dt>
            <a href ="/abs/XXXX.YYYY" ...>arXiv:XXXX.YYYY</a>
            ...
          </dt>
          <dd>
            <div class='meta'>
              <div class='list-title mathjax'><span class='descriptor'>Title:</span>
                Paper Title
              </div>
              ...
            </div>
          </dd>
        </dl>

    We extract the arXiv ID from the ``<dt>`` anchor and the title from the
    ``list-title`` div, then pair them up in order.
    """
    # arXiv IDs: href="/abs/XXXX.YYYY" or href ="/abs/XXXX.YYYY"
    arxiv_ids = re.findall(r'href\s*=\s*["\s]?/abs/([0-9]{4}\.[0-9]+)', html)

    # Titles from list-title divs
    raw_titles = re.findall(
        r"<div class=['\"]list-title[^'\"]*['\"]>"
        r"<span class=['\"]descriptor['\"]>Title:</span>(.*?)</div>",
        html,
        re.DOTALL,
    )
    titles = [clean_html(t) for t in raw_titles]

    if len(arxiv_ids) != len(titles):
        logger.warning(
            "  ID/title count mismatch: %d IDs vs %d titles — pairing by min",
            len(arxiv_ids), len(titles),
        )

    return list(zip(arxiv_ids, titles))


ARXIV_HTML_BASE = "https://arxiv.org/html/"
INTER_REQUEST_DELAY = 1.0  # seconds — be polite to arXiv


def fetch_all_indices(session) -> dict[str, str]:
    """Fetch all 29 index pages and return a mapping normalized_title → arxiv_id."""
    title_to_id: dict[str, str] = {}
    total_papers = 0

    for i, idx_id in enumerate(INDEX_ARXIV_IDS):
        url = f"{ARXIV_HTML_BASE}{idx_id}"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("  Failed to fetch %s: %s", url, exc)
            continue

        pairs = _parse_index_page(resp.text)
        logger.info("  [%2d/%d] %s → %d papers parsed", i + 1, len(INDEX_ARXIV_IDS), idx_id, len(pairs))

        for arxiv_id, title in pairs:
            norm = normalize_title(title)
            if norm in title_to_id:
                # duplicate title across index pages — keep first occurence
                pass
            else:
                title_to_id[norm] = arxiv_id
        total_papers += len(pairs)

        if i < len(INDEX_ARXIV_IDS) - 1:
            time.sleep(INTER_REQUEST_DELAY)

    logger.info("  Fetched %d total papers from %d index pages", total_papers, len(INDEX_ARXIV_IDS))
    return title_to_id


ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
UAI_LEGACY_FILE = LEGACY_DIR / "uai-legacy.jsonl.gz"
# Only enrich papers from years covered by the arXiv bulk upload
COVERED_YEARS = {str(y) for y in range(1985, 2014)}


def enrich(*, dry_run: bool = False) -> None:
    papers = read_legacy(UAI_LEGACY_FILE)
    logger.info("Loaded %d UAI papers from %s", len(papers), UAI_LEGACY_FILE.name)

    session = make_session(retries=3, backoff_factor=1.0)
    session.headers.update({"User-Agent": "mlanthology-enrichment/1.0 (research bot)"})

    logger.info("Fetching arXiv index pages (%d total)…", len(INDEX_ARXIV_IDS))
    index = fetch_all_indices(session)
    logger.info("Index contains %d distinct normalised titles", len(index))

    n_added = 0       # pdf_url was empty → filled
    n_replaced = 0    # pdf_url was non-arXiv → replaced
    n_skipped = 0     # already arXiv → untouched
    n_unmatched = 0   # no arXiv ID found

    for paper in papers:
        if paper.get("year") not in COVERED_YEARS:
            continue

        arxiv_id = fuzzy_lookup(paper["title"], index)
        if arxiv_id is None:
            n_unmatched += 1
            continue

        new_url = f"{ARXIV_PDF_BASE}{arxiv_id}"
        existing = paper.get("pdf_url", "").strip()

        if "arxiv.org" in existing.lower():
            n_skipped += 1
            continue

        if not existing:
            action = "add"
            n_added += 1
        else:
            action = "replace"
            n_replaced += 1

        if dry_run:
            logger.debug(
                "  DRY-RUN [%s] %s | %s → %s",
                action, paper["year"], existing or "(none)", new_url,
            )
        else:
            paper["pdf_url"] = new_url

    logger.info(
        "Results: %d added, %d replaced, %d already-arXiv (skipped), %d unmatched",
        n_added, n_replaced, n_skipped, n_unmatched,
    )

    if dry_run:
        logger.info("Dry-run — no files written")
        return

    if n_added + n_replaced == 0:
        logger.info("No changes — not writing file")
        return

    write_legacy(UAI_LEGACY_FILE, papers, atomic=True)
    logger.info("Written back to %s", UAI_LEGACY_FILE.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse indices and report what would change without writing",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.dry_run else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    enrich(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
