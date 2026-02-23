#!/usr/bin/env python3
"""Build ICLR Workshops legacy file from OpenReview.

ICLR workshops (2019-2025) live on OpenReview under venue IDs of the form
ICLR.cc/{year}/Workshop/{code}.  Papers have full abstracts and PDFs.

This script:
  1. Lists all workshop groups per year via the OpenReview groups API
  2. Fetches accepted papers from each workshop (venueid filter)
  3. Writes data/legacy/iclrw-legacy.jsonl.gz

There is no S2 enrichment step because OpenReview already provides
abstracts and PDFs directly.

Run once:
    python scripts/build_iclrw_legacy.py
    python scripts/build_content.py
"""

import logging
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, write_legacy

from adapters.iclrw import fetch_all_years, YEARS
from adapters.common import normalize_paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LEGACY_DIR / "iclrw-legacy.jsonl.gz"

    if out_path.exists():
        logger.info("iclrw-legacy.jsonl.gz already exists — delete to re-fetch")
        return

    logger.info(f"Building ICLR Workshops legacy for years: {YEARS}")
    by_year = fetch_all_years()

    all_papers = []
    for year in YEARS:
        papers = by_year.get(year, [])
        logger.info(f"  {year}: {len(papers)} papers")
        all_papers.extend(normalize_paper(p) for p in papers)

    if not all_papers:
        logger.warning("No papers found — check network / rate limits and retry")
        return

    write_legacy(out_path, all_papers)

    logger.info(f"Wrote {len(all_papers)} ICLR workshop papers to {out_path.name}")
    print(
        "\nDone. Run 'python scripts/build_content.py' to regenerate Hugo content."
    )


if __name__ == "__main__":
    main()
