#!/usr/bin/env python3
"""Patch pdf_url and venue_url for AISTATS 2013-2018 and COLT 2013-2018 legacy papers.

These years were originally fetched from DBLP (which has no direct PDF links for
PMLR proceedings) but the papers exist in PMLR and their PDFs are freely accessible.
This script fetches the PMLR metadata and matches it to existing legacy papers by
normalised title, then fills in pdf_url, venue_url, and abstract where missing.

Run once:
    python scripts/patch_pmlr_legacy.py
"""

import logging
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, read_legacy, write_legacy, normalize_title

from adapters.pmlr import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Confirmed PMLR volumes for the legacy years we want to patch.
# COLT 2010 is not in PMLR (published by Omnipress separately).
# COLT 2011/2012 (v19/v23) and AISTATS 2012 (v22) were absent from DBLP so
# were fetched directly from PMLR and appended to the legacy files; re-running
# this script against those years is harmless (matches will just overwrite with
# the same values).
VOLUMES = [
    # AISTATS
    (22, "aistats", "2012"),
    (31, "aistats", "2013"),
    (33, "aistats", "2014"),
    (38, "aistats", "2015"),
    (51, "aistats", "2016"),
    (54, "aistats", "2017"),
    (84, "aistats", "2018"),
    # COLT
    (19, "colt", "2011"),
    (23, "colt", "2012"),
    (30, "colt", "2013"),
    (35, "colt", "2014"),
    (40, "colt", "2015"),
    (49, "colt", "2016"),
    (65, "colt", "2017"),
    (75, "colt", "2018"),
]


def patch_venue(venue: str, vol_years: list[tuple[int, str]]) -> None:
    legacy_path = LEGACY_DIR / f"{venue}-legacy.jsonl.gz"
    logger.info(f"\n{'='*60}")
    logger.info(f"Patching {legacy_path.name}...")

    papers = read_legacy(legacy_path)

    # Index by normalised title for O(1) lookup
    title_index: dict[str, int] = {}
    for i, p in enumerate(papers):
        key = normalize_title(p["title"])
        if key not in title_index:      # keep first on collision
            title_index[key] = i

    total_matched = total_pdf = total_abs = total_url = 0

    for vol, year in sorted(vol_years, key=lambda x: x[1]):
        logger.info(f"  Fetching v{vol} ({venue.upper()} {year})...")
        try:
            pmlr_papers = process_volume(vol, venue, year)
        except Exception as e:
            logger.error(f"  v{vol}: failed — {e}")
            continue

        matched = pdf_added = abs_added = url_added = 0
        for pp in pmlr_papers:
            idx = title_index.get(normalize_title(pp.get("title", "")))
            if idx is None:
                continue
            matched += 1
            lp = papers[idx]

            if pp.get("pdf_url") and not lp.get("pdf_url"):
                lp["pdf_url"] = pp["pdf_url"]
                pdf_added += 1

            # PMLR page URL is more useful than a DOI redirect for these venues
            if pp.get("venue_url"):
                lp["venue_url"] = pp["venue_url"]
                url_added += 1

            if pp.get("abstract") and not lp.get("abstract"):
                lp["abstract"] = pp["abstract"]
                abs_added += 1

        logger.info(
            f"  {venue.upper()} {year} (v{vol}): "
            f"{matched}/{len(pmlr_papers)} matched  "
            f"+{pdf_added} PDFs  +{abs_added} abstracts  "
            f"{url_added} venue_urls updated"
        )
        total_matched += matched
        total_pdf += pdf_added
        total_abs += abs_added
        total_url += url_added

    # Write back atomically
    write_legacy(legacy_path, papers, atomic=True)

    logger.info(
        f"\n  {venue.upper()} total: {total_matched} matched, "
        f"+{total_pdf} PDFs, +{total_abs} abstracts, "
        f"{total_url} venue_urls → written to {legacy_path.name}"
    )


def main() -> None:
    by_venue: dict[str, list[tuple[int, str]]] = {}
    for vol, venue, year in VOLUMES:
        by_venue.setdefault(venue, []).append((vol, year))

    for venue, vol_years in by_venue.items():
        patch_venue(venue, vol_years)

    print("\nDone. Run 'python scripts/build_content.py' to regenerate Hugo content.")


if __name__ == "__main__":
    main()
