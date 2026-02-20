#!/usr/bin/env python3
"""Enrich CVF open-access papers with DOIs fetched from DBLP.

CVF-scraped papers (data/papers/cvpr-*.json.gz, iccv-*.json.gz, eccv-*.json.gz)
have PDF and abstract from CVF but no DOI.  DBLP carries the IEEE DOIs
for CVPR/ICCV and Springer DOIs for ECCV, which we want for BibTeX.

Strategy: fetch DBLP for each needed venue/year, match by normalised
title, patch the doi field in-place.

Run once:
    python scripts/enrich_cvf_dois.py
"""

import gzip
import json
import logging
import re
from pathlib import Path

from scripts.utils import ROOT, PAPERS_DIR, normalize_title

from adapters.common import read_venue_json
from adapters.dblp import _discover_venue_stems, process_venue_year, DBLP_VENUES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# For venues that share a DBLP key with workshops, exclude stems
# matching these patterns so we only get main-conference DOIs.
WORKSHOP_STEM_EXCLUDE = {
    "cvpr": r"w",       # cvprw stems contain "w"
    "eccv": r"w\d",     # eccvw stems look like eccv2024w1
}


def patch_venue(venue: str, years: list[str]) -> None:
    cfg = DBLP_VENUES[venue]
    dblp_key = cfg["key"]
    exclude_pat = WORKSHOP_STEM_EXCLUDE.get(venue)

    logger.info(f"\n{'='*60}")
    logger.info(f"Discovering DBLP stems for {venue.upper()}...")
    stems_by_year = _discover_venue_stems(dblp_key, start_year=min(int(y) for y in years))

    total_matched = total_filled = 0

    for year in sorted(years):
        gz_path = PAPERS_DIR / f"{venue}-{year}.json.gz"
        if not gz_path.exists():
            logger.warning(f"  {gz_path.name} not found, skipping")
            continue

        data = read_venue_json(gz_path)
        papers = data["papers"]

        need_doi = {normalize_title(p["title"]): i for i, p in enumerate(papers) if not p.get("doi")}
        if not need_doi:
            logger.info(f"  {venue.upper()} {year}: all {len(papers)} papers already have DOIs")
            continue

        year_stems = stems_by_year.get(year, [])
        if exclude_pat:
            year_stems = [s for s in year_stems if not re.search(exclude_pat, s)]
        if not year_stems:
            logger.warning(f"  {venue.upper()} {year}: no DBLP stems found")
            continue

        logger.info(
            f"  {venue.upper()} {year}: {len(need_doi)}/{len(papers)} need DOI, "
            f"fetching {len(year_stems)} DBLP stem(s)..."
        )
        dblp_papers = process_venue_year(venue, year, dblp_key, year_stems)

        doi_lookup = {normalize_title(p["title"]): p["doi"] for p in dblp_papers if p.get("doi")}

        filled = 0
        for norm_title, idx in need_doi.items():
            doi = doi_lookup.get(norm_title)
            if doi:
                papers[idx]["doi"] = doi
                filled += 1

        logger.info(
            f"  {venue.upper()} {year}: matched {filled}/{len(need_doi)} missing DOIs "
            f"(of {len(dblp_papers)} DBLP papers)"
        )
        total_matched += len(need_doi)
        total_filled += filled

        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n  {venue.upper()} total: filled {total_filled}/{total_matched} missing DOIs"
    )


def main() -> None:
    def years_for_venue(v: str) -> list[str]:
        return sorted(
            p.name.split("-", 1)[1].removesuffix(".json.gz")
            for p in PAPERS_DIR.glob(f"{v}-*.json.gz")
        )

    for venue in ["cvpr", "iccv", "eccv"]:
        years = years_for_venue(venue)
        if not years:
            logger.info(f"No data files found for {venue}, skipping")
            continue
        patch_venue(venue, years)

    print("\nDone. Run 'python scripts/build_content.py' to regenerate Hugo content.")


if __name__ == "__main__":
    main()
