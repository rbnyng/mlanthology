#!/usr/bin/env python3
"""Enrich legacy data with abstracts scraped from IEEE Xplore.

IEEE Xplore embeds full paper metadata as a JSON object in every article page:
  xplGlobal.document.metadata = { ... "abstract": "...", ... };

This makes extraction reliable and robust — it's JSON parsing, not fragile HTML
scraping.  The URL pattern is:
  https://ieeexplore.ieee.org/document/{articleNumber}

The article number is the last numeric segment of the IEEE DOI:
  10.1109/CVPR.1992.223244  →  223244

Applicable venues (IEEE 10.1109/ DOIs):
  cvpr   — CVPR legacy (pre-2013)
  iccv   — ICCV legacy (pre-2013)
  cvprw  — CVPR workshops legacy
  iccvw  — ICCV workshops legacy
  wacv   — WACV legacy

Usage::

    python scripts/enrich_ieeexplore.py --venue cvpr
    python scripts/enrich_ieeexplore.py --venue iccv
    python scripts/enrich_ieeexplore.py            # all legacy files
    python scripts/enrich_ieeexplore.py --dry-run
    python scripts/enrich_ieeexplore.py --venue cvpr --limit 20
"""

import argparse
import json
import logging
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, make_session, normalize_title, read_legacy, write_legacy

logger = logging.getLogger(__name__)

REQUEST_DELAY = 1.0  # seconds between requests

_METADATA_RE = re.compile(
    r"xplGlobal\.document\.metadata\s*=\s*(\{.*?\})\s*;",
    re.DOTALL,
)

# minimum title similarity to accept an abstract.
# protects against short DOI suffixes that collide with unrelated  documents.
TITLE_SIMILARITY_THRESHOLD = 0.80

_BLOCKED = False  # set True on 418 to abort early

_SESSION = make_session(retries=4, backoff_factor=1.5)
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})


def _article_number(doi: str) -> str | None:
    """Return the numeric article-number suffix of a 10.1109/ DOI, or None.

      10.1109/CVPR.1992.223244  →  '223244'
    """
    last = doi.rsplit(".", 1)[-1]
    return last if last.isdigit() else None


def fetch_metadata(doi: str) -> dict | None:
    """Fetch IEEE Xplore metadata for a DOI.

    Returns the parsed xplGlobal.document.metadata dict, or None.
    Sets the module-level _BLOCKED flag on HTTP 418 (bot detection).
    """
    global _BLOCKED
    num = _article_number(doi)
    if num is None:
        return None

    url = f"https://ieeexplore.ieee.org/document/{num}"
    try:
        resp = _SESSION.get(url, timeout=25, allow_redirects=True)
    except Exception as exc:
        logger.warning("  HTTP error for %s: %s", doi, exc)
        return None

    if resp.status_code == 418:
        logger.error("  IEEE Xplore returned 418 (bot detection) — aborting")
        _BLOCKED = True
        return None
    if resp.status_code == 404:
        logger.debug("  404: %s", doi)
        return None
    if resp.status_code != 200:
        logger.warning("  HTTP %d for %s", resp.status_code, doi)
        return None

    m = _METADATA_RE.search(resp.text)
    if not m:
        logger.debug("  no xplGlobal metadata: %s", doi)
        return None

    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        logger.warning("  metadata JSON parse error for %s: %s", doi, exc)
        return None


def fetch_abstract(doi: str, expected_title: str) -> str | None:
    """Fetch the abstract for an IEEE DOI, verifying the title matches.

    Short DOI suffixes (e.g. .250) can collide with unrelated IEEE documents.
    Title verification ensures we only accept abstracts from the right paper.

    Returns the abstract string, or None if not found / title mismatch / no abstract.
    """
    meta = fetch_metadata(doi)
    if meta is None:
        return None

    page_title = meta.get("title", "")
    sim = SequenceMatcher(
        None,
        normalize_title(expected_title),
        normalize_title(page_title),
        autojunk=False,
    ).ratio()
    if sim < TITLE_SIMILARITY_THRESHOLD:
        logger.warning(
            "  TITLE MISMATCH (sim=%.2f) for %s\n"
            "    expected: %s\n"
            "    got:      %s",
            sim, doi, expected_title[:70], page_title[:70],
        )
        return None

    abstract = meta.get("abstract", "").strip()
    return abstract or None


def enrich_file(
    path: Path,
    *,
    dry_run: bool = False,
    limit: int = 0,
) -> int:
    """Enrich a single legacy file via IEEE Xplore.  Returns abstracts added."""
    papers = read_legacy(path)
    total = len(papers)
    with_abstract = sum(1 for p in papers if p.get("abstract", "").strip())

    candidates = [
        p for p in papers
        if p.get("doi", "").startswith("10.1109/")
        and _article_number(p.get("doi", "")) is not None
        and not p.get("abstract", "").strip()
    ]

    logger.info(
        "%s: %d papers, %d with abstract, %d IEEE candidates missing abstract",
        path.name, total, with_abstract, len(candidates),
    )

    if not candidates:
        logger.info("  Nothing to enrich")
        return 0

    if limit > 0:
        candidates = candidates[:limit]
        logger.info("  Limiting to %d papers (--limit)", len(candidates))

    if dry_run:
        logger.info("  Dry-run: would fetch %d IEEE Xplore pages", len(candidates))
        return 0

    n_added = n_no_abstract = 0
    last_req = 0.0

    for i, paper in enumerate(candidates, 1):
        if _BLOCKED:
            logger.error("  Aborting: IEEE Xplore is blocking requests")
            break

        doi = paper["doi"]

        elapsed = time.time() - last_req
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        last_req = time.time()

        abstract = fetch_abstract(doi, paper["title"])

        if abstract:
            paper["abstract"] = abstract
            src = paper.get("source", "")
            paper["source"] = (src + "+ieeexplore") if src else "ieeexplore"
            n_added += 1
            if i % 50 == 0 or i == len(candidates):
                logger.info(
                    "  [%d/%d] +abstract %s | %s…",
                    i, len(candidates), doi, abstract[:55],
                )
        else:
            n_no_abstract += 1

    logger.info(
        "  Results: +%d abstracts, %d not found/no abstract/mismatch",
        n_added, n_no_abstract,
    )

    if n_added == 0:
        logger.info("  No changes — not writing file")
        return 0

    write_legacy(path, papers, atomic=True)
    logger.info("  Written back to %s", path.name)

    after = sum(1 for p in papers if p.get("abstract", "").strip())
    logger.info("  Coverage: %d/%d (%.1f%%)", after, total, after / total * 100)
    return n_added


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--venue",
        default=None,
        help="Venue name (e.g. 'cvpr', 'iccv'). Default: all legacy files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N candidates per file (for testing).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.venue:
        files = [LEGACY_DIR / f"{args.venue}-legacy.jsonl.gz"]
        if not files[0].exists():
            logger.error("File not found: %s", files[0])
            sys.exit(1)
    else:
        files = sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz"))

    t_start = time.time()
    total_added = 0
    for path in files:
        total_added += enrich_file(path, dry_run=args.dry_run, limit=args.limit)

    elapsed = time.time() - t_start
    action = "would add" if args.dry_run else "added"
    logger.info(
        "\nDone in %.0fs — %d abstracts %s across %d file(s)",
        elapsed, total_added, action, len(files),
    )


if __name__ == "__main__":
    main()
