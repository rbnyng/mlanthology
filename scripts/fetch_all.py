#!/usr/bin/env python3
"""Master fetch script: runs all adapters and optionally builds Hugo content.

Usage:
    python scripts/fetch_all.py                    # Fetch all + build content
    python scripts/fetch_all.py --source pmlr      # Fetch only PMLR + build
    python scripts/fetch_all.py --quick             # Fetch one volume per venue (for testing)
    python scripts/fetch_all.py --force             # Re-fetch everything, ignoring cache
    python scripts/fetch_all.py --no-build          # Fetch only, skip content build
    python scripts/fetch_all.py --build-only        # Skip fetching, just rebuild content

Note: Historical proceedings (pre-2013 to pre-2019 depending on venue) are
served from cached legacy data in data/legacy/.  This data is built once
via scripts/build_legacy.py and included automatically during content
generation.  The DBLP backlog adapter skips venue-years that have legacy
files, so there's no double-fetching.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `scripts` and `adapters` are
# importable when invoked as `python scripts/fetch_all.py`.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import ROOT

from adapters.pmlr import fetch_all as pmlr_fetch_all, KNOWN_VOLUMES as PMLR_VOLUMES
from adapters.openreview import fetch_all as openreview_fetch_all, KNOWN_VENUES as OPENREVIEW_VENUES
from adapters.neurips import fetch_all as neurips_fetch_all, KNOWN_YEARS as NEURIPS_YEARS
from adapters.jmlr import fetch_all as jmlr_fetch_all, KNOWN_VOLUMES as JMLR_VOLUMES, KNOWN_DMLR_VOLUMES
from adapters.cvf import fetch_all as cvf_fetch_all, KNOWN_CONFERENCES as CVF_CONFERENCES
from adapters.ecva import fetch_all as ecva_fetch_all, KNOWN_YEARS as ECVA_YEARS
from adapters.dblp import fetch_all as dblp_fetch_all, DBLP_VENUES
from adapters.cache import load_cache, save_cache
from scripts.build_content import build_all


def main():
    parser = argparse.ArgumentParser(description="Fetch ML proceedings metadata")
    parser.add_argument("--source", choices=["pmlr", "openreview", "neurips", "jmlr", "cvf", "ecva", "dblp", "all"], default="all",
                        help="Which source to fetch from")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fetch only the latest volume per venue")
    parser.add_argument("--force", action="store_true",
                        help="Force re-fetch all volumes, ignoring cache")
    parser.add_argument("--dblp-max-year", type=int, default=None,
                        help="Upper bound year for DBLP backlog (e.g. --dblp-max-year 2015)")
    parser.add_argument("--dblp-venue", type=str, choices=list(DBLP_VENUES.keys()),
                        help="Restrict DBLP backlog to a single venue")
    parser.add_argument("--output", type=str, default="data/papers",
                        help="Output directory for JSON data")
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument("--no-build", action="store_true",
                             help="Skip content build after fetching")
    build_group.add_argument("--build-only", action="store_true",
                             help="Skip fetching, just rebuild Hugo content")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = ROOT / args.output

    if not args.build_only:
        _fetch(args, output_dir)

    if not args.no_build:
        logging.info("Building Hugo content pages...")
        build_all()

    logging.info("Done.")


def _fetch(args, output_dir: Path) -> None:
    """Run all configured adapters."""
    cache = None if args.force else load_cache(ROOT)

    if args.source in ("pmlr", "all"):
        volumes = list(PMLR_VOLUMES)
        if args.quick:
            seen = set()
            quick_volumes = []
            for vol, venue, year in volumes:
                if venue not in seen:
                    seen.add(venue)
                    quick_volumes.append((vol, venue, year))
            volumes = quick_volumes

        logging.info(f"Fetching {len(volumes)} PMLR volumes...")
        pmlr_fetch_all(volumes=volumes, output_dir=output_dir, cache=cache)

    if args.source in ("openreview", "all"):
        venues = OPENREVIEW_VENUES
        if args.quick:
            # Take only the first (latest) entry per venue shortname
            seen = set()
            quick_venues = []
            for venue_id, venue, year, api_version in venues:
                if venue not in seen:
                    seen.add(venue)
                    quick_venues.append((venue_id, venue, year, api_version))
            venues = quick_venues

        logging.info(f"Fetching {len(venues)} OpenReview venues...")
        openreview_fetch_all(venues=venues, output_dir=output_dir, cache=cache)

    if args.source in ("neurips", "all"):
        years = NEURIPS_YEARS
        if args.quick:
            # Take only the latest year
            years = years[-1:]

        logging.info(f"Fetching {len(years)} NeurIPS years...")
        neurips_fetch_all(years=years, output_dir=output_dir, cache=cache)

    if args.source in ("jmlr", "all"):
        volumes = JMLR_VOLUMES
        dmlr_volumes = KNOWN_DMLR_VOLUMES
        if args.quick:
            # Take only the latest JMLR volume and latest DMLR volume
            volumes = volumes[-1:]
            dmlr_volumes = dmlr_volumes[-1:]

        logging.info(f"Fetching {len(volumes)} JMLR volumes + {len(dmlr_volumes)} DMLR volumes...")
        jmlr_fetch_all(volumes=volumes, dmlr_volumes=dmlr_volumes, output_dir=output_dir, cache=cache)

    if args.source in ("cvf", "all"):
        conferences = CVF_CONFERENCES
        if args.quick:
            # Take only the latest year per conference
            seen = set()
            quick_conferences = []
            for conf, year in conferences:
                if conf not in seen:
                    seen.add(conf)
                    quick_conferences.append((conf, year))
            conferences = quick_conferences

        logging.info(f"Fetching {len(conferences)} CVF conference-years...")
        cvf_fetch_all(conferences=conferences, output_dir=output_dir, cache=cache)

    if args.source in ("ecva", "all"):
        years = ECVA_YEARS
        if args.quick:
            # Take only the latest year
            years = years[-1:]

        logging.info(f"Fetching {len(years)} ECVA/ECCV years...")
        ecva_fetch_all(years=years, output_dir=output_dir, cache=cache)

    if args.source == "dblp":
        dblp_venues = [args.dblp_venue] if args.dblp_venue else list(DBLP_VENUES.keys())
        max_year = args.dblp_max_year  # None = current year (adapter default)

        logging.info(
            f"Running DBLP backlog for {len(dblp_venues)} venue(s)"
            + (f" up to {max_year}" if max_year else "")
            + " (fill-only: skips existing files)..."
        )
        dblp_fetch_all(
            venues=dblp_venues,
            output_dir=output_dir,
            cache=cache,
            max_year=max_year,
            fill_only=True,
        )

    if cache is not None:
        save_cache(ROOT, cache)
        logging.info(f"Cache updated ({len(cache['fetched'])} entries)")


if __name__ == "__main__":
    main()
