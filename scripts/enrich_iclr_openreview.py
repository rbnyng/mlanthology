"""Enrich ICLR papers with missing abstracts from OpenReview and arXiv.

Three cases handled:

  - **iclr-2020** (papers/): All 825 papers have empty abstracts.  Each record
    has a ``pdf_url`` whose ``?id=`` parameter is the OpenReview blind-submission
    forum ID.  We query ``api.openreview.net/notes?forum=<id>`` and extract the
    abstract from the note that has one.

  - **iclr-legacy 2017**: Papers have ``venue_url`` pointing to OpenReview
    forums.  Same approach as above.

  - **iclr-legacy 2013–2016**: Papers have ``venue_url`` pointing to arXiv
    abstract pages.  We batch-query the arXiv API and fill in the abstract.

Usage::

    python -m scripts.enrich_iclr_openreview          # both legacy + 2020
    python -m scripts.enrich_iclr_openreview --legacy-only
    python -m scripts.enrich_iclr_openreview --2020-only
    python -m scripts.enrich_iclr_openreview --dry-run
"""

import argparse
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from adapters.common import read_venue_json, write_venue_json
from scripts.utils import LEGACY_DIR, PAPERS_DIR, make_session, read_legacy, write_legacy

logger = logging.getLogger(__name__)

OPENREVIEW_API = "https://api.openreview.net"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE = 20
INTER_REQUEST_DELAY = 0.6  # seconds between OpenReview calls
ARXIV_DELAY = 1.0          # seconds between arXiv batch calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_url_id(url: str) -> str | None:
    """Return the ``?id=`` or ``&id=`` value from an OpenReview URL."""
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def _extract_arxiv_id(url: str) -> str | None:
    """Return the arXiv paper ID from an ``arxiv.org/abs/`` URL."""
    m = re.search(r"arxiv\.org/abs/([^\s/?#]+)", url, re.IGNORECASE)
    return m.group(1) if m else None


def _fetch_openreview_abstract(session, forum_id: str) -> str | None:
    """Return the abstract for an OpenReview forum, or None if not found."""
    try:
        r = session.get(
            f"{OPENREVIEW_API}/notes",
            params={"forum": forum_id, "limit": 50},
        )
        if r.status_code != 200:
            logger.warning("OpenReview HTTP %s for forum %s", r.status_code, forum_id)
            return None
        for note in r.json().get("notes", []):
            abstract = note.get("content", {}).get("abstract", "").strip()
            if abstract:
                return abstract
        return None
    except Exception as exc:
        logger.warning("Error fetching forum %s: %s", forum_id, exc)
        return None


def _fetch_arxiv_abstracts(session, arxiv_ids: list[str]) -> dict[str, str]:
    """Batch-query the arXiv API; return {arxiv_id: abstract}."""
    result: dict[str, str] = {}
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for i in range(0, len(arxiv_ids), ARXIV_BATCH_SIZE):
        batch = arxiv_ids[i : i + ARXIV_BATCH_SIZE]
        try:
            r = session.get(ARXIV_API, params={"id_list": ",".join(batch)})
            if r.status_code != 200:
                logger.warning("arXiv HTTP %s for batch %s", r.status_code, batch[:3])
                continue
            root = ET.fromstring(r.text)
            for entry in root.findall("atom:entry", ns):
                # arXiv ID is in <id>http://arxiv.org/abs/XXXX.YYYY</id>
                id_el = entry.find("atom:id", ns)
                summary_el = entry.find("atom:summary", ns)
                if id_el is None or summary_el is None:
                    continue
                arxiv_id = _extract_arxiv_id(id_el.text or "")
                abstract = " ".join((summary_el.text or "").split())
                if arxiv_id and abstract:
                    result[arxiv_id] = abstract
        except Exception as exc:
            logger.warning("arXiv batch error: %s", exc)
        time.sleep(ARXIV_DELAY)

    return result


# ---------------------------------------------------------------------------
# Enrichment functions
# ---------------------------------------------------------------------------

def enrich_legacy(*, dry_run: bool = False) -> None:
    """Fill missing abstracts in iclr-legacy.jsonl.gz."""
    path = LEGACY_DIR / "iclr-legacy.jsonl.gz"
    papers = read_legacy(path)
    missing = [p for p in papers if not p.get("abstract", "").strip()]
    logger.info("iclr-legacy: %d total, %d missing abstracts", len(papers), len(missing))

    session = make_session(retries=3, backoff_factor=1.5)
    session.headers["User-Agent"] = "mlanthology-enrichment/1.0"

    enriched = 0

    # --- 2013–2016: arXiv ---
    arxiv_subset = [p for p in missing if int(p.get("year", 0)) <= 2016]
    if arxiv_subset:
        ids_needed = []
        arxiv_id_map: dict[str, str] = {}  # paper title → arxiv_id (for lookup later)
        for p in arxiv_subset:
            aid = _extract_arxiv_id(p.get("venue_url", "") or p.get("pdf_url", ""))
            if aid:
                ids_needed.append(aid)
                arxiv_id_map[p["title"]] = aid
            else:
                logger.debug("No arXiv ID for: %s", p["title"][:60])

        logger.info("Fetching %d arXiv abstracts (2013–2016)…", len(ids_needed))
        abstract_by_id = _fetch_arxiv_abstracts(session, ids_needed)
        logger.info("  Got %d/%d abstracts from arXiv", len(abstract_by_id), len(ids_needed))

        for p in arxiv_subset:
            aid = arxiv_id_map.get(p["title"])
            if not aid:
                continue
            abstract = abstract_by_id.get(aid)
            if abstract:
                logger.info("[%s] %s: found (%d chars)", p["year"], p["title"][:55], len(abstract))
                if not dry_run:
                    p["abstract"] = abstract
                enriched += 1
            else:
                logger.debug("[%s] %s: no abstract from arXiv", p["year"], p["title"][:55])

    # --- 2017: OpenReview ---
    or_subset = [p for p in missing if p.get("year") == "2017"]
    logger.info("Fetching %d OpenReview abstracts (2017)…", len(or_subset))
    for idx, p in enumerate(or_subset, 1):
        forum_id = _extract_url_id(p.get("venue_url", "") or p.get("pdf_url", ""))
        if not forum_id:
            logger.debug("No forum ID: %s", p["title"][:60])
            continue

        abstract = _fetch_openreview_abstract(session, forum_id)
        if abstract:
            logger.info("[2017] %s: found (%d chars)", p["title"][:55], len(abstract))
            if not dry_run:
                p["abstract"] = abstract
            enriched += 1
        else:
            logger.debug("[2017] %s: no abstract", p["title"][:55])

        if idx % 20 == 0:
            logger.info("  Progress: %d/%d", idx, len(or_subset))
        time.sleep(INTER_REQUEST_DELAY)

    logger.info("iclr-legacy: enriched %d/%d missing abstracts", enriched, len(missing))

    if dry_run:
        logger.info("Dry-run — not writing")
        return
    if enriched == 0:
        logger.info("No changes — not writing")
        return
    write_legacy(path, papers, atomic=True)
    logger.info("Written back to %s", path.name)


def enrich_iclr2020(*, dry_run: bool = False) -> None:
    """Fill missing abstracts in iclr-2020.json.gz via OpenReview."""
    path = PAPERS_DIR / "iclr-2020.json.gz"
    data = read_venue_json(path)
    papers = data["papers"]
    missing = [p for p in papers if not p.get("abstract", "").strip()]
    logger.info("iclr-2020: %d total, %d missing abstracts", len(papers), len(missing))

    session = make_session(retries=3, backoff_factor=1.5)
    session.headers["User-Agent"] = "mlanthology-enrichment/1.0"

    enriched = 0
    for idx, p in enumerate(missing, 1):
        # The blind-submission forum ID is in pdf_url, not venue_url/source_id
        forum_id = _extract_url_id(p.get("pdf_url", ""))
        if not forum_id:
            logger.debug("No forum ID in pdf_url: %s", p["title"][:60])
            continue

        abstract = _fetch_openreview_abstract(session, forum_id)
        if abstract:
            if not dry_run:
                p["abstract"] = abstract
            enriched += 1
        else:
            logger.debug("%s: no abstract", p["title"][:55])

        if idx % 50 == 0:
            logger.info("  Progress: %d/%d (enriched %d so far)", idx, len(missing), enriched)
        time.sleep(INTER_REQUEST_DELAY)

    logger.info("iclr-2020: enriched %d/%d papers", enriched, len(missing))

    if dry_run:
        logger.info("Dry-run — not writing")
        return
    if enriched == 0:
        logger.info("No changes — not writing")
        return
    write_venue_json("iclr", "2020", papers, PAPERS_DIR)
    logger.info("Written back to iclr-2020.json.gz")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files")
    parser.add_argument("--legacy-only", action="store_true",
                        help="Only process iclr-legacy.jsonl.gz")
    parser.add_argument("--2020-only", dest="year2020_only", action="store_true",
                        help="Only process iclr-2020.json.gz")
    args = parser.parse_args()

    if not args.year2020_only:
        enrich_legacy(dry_run=args.dry_run)
    if not args.legacy_only:
        enrich_iclr2020(dry_run=args.dry_run)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
