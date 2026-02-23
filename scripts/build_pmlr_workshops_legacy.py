#!/usr/bin/env python3
"""Build legacy files for PMLR workshop/small-conference venues.

Fetches proceedings from PMLR GitHub repos for venues that are too small
or infrequent to include in the main fetch pipeline, and writes them as
legacy JSONL.gz files.

Venues covered:
  MLHC     Machine Learning for Healthcare (2016-2025)
  L4DC     Learning for Dynamics and Control (2020-2025)
  CLeaR    Causal Learning and Reasoning (2022-2025)
  AutoML   International Conference on Automated Machine Learning (2016-2025)
  CHIL     Conference on Health, Inference, and Learning (2022-2025)
  PGM      Probabilistic Graphical Models (2016-2024)
  LoG      Learning on Graphs (2022-2025)
  CoLLAs   Conference on Lifelong Learning Agents (2022-2024)
  CPAL     Conference on Parsimony and Learning (2024-2025)
  ISIPTA   Imprecise Probabilities (2019-2025)

Usage:
    python scripts/build_pmlr_workshops_legacy.py
    python scripts/build_pmlr_workshops_legacy.py --venue mlhc
    python scripts/build_pmlr_workshops_legacy.py --venue l4dc --force
"""

import argparse
import logging
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, write_legacy

from adapters.pmlr import process_volume
from adapters.common import normalize_paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# (volume_number, venue_slug, year)
WORKSHOP_VOLUMES: dict[str, list[tuple[int, str, str]]] = {
    "mlhc": [
        (56,  "mlhc", "2016"),
        (68,  "mlhc", "2017"),
        (85,  "mlhc", "2018"),
        (106, "mlhc", "2019"),
        (126, "mlhc", "2020"),
        (149, "mlhc", "2021"),
        (182, "mlhc", "2022"),
        (219, "mlhc", "2023"),
        (252, "mlhc", "2024"),
        (298, "mlhc", "2025"),
    ],
    "l4dc": [
        (120, "l4dc", "2020"),
        (144, "l4dc", "2021"),
        (168, "l4dc", "2022"),
        (211, "l4dc", "2023"),
        (242, "l4dc", "2024"),
        (283, "l4dc", "2025"),
    ],
    "clear": [
        (177, "clear", "2022"),
        (213, "clear", "2023"),
        (236, "clear", "2024"),
        (275, "clear", "2025"),
    ],
    "automl": [
        (64,  "automl", "2016"),
        (188, "automl", "2022"),
        (224, "automl", "2023"),
        (256, "automl", "2024"),
        (293, "automl", "2025"),
    ],
    "chil": [
        (174, "chil", "2022"),
        (209, "chil", "2023"),
        (248, "chil", "2024"),
        (287, "chil", "2025"),
    ],
    "pgm": [
        (52,  "pgm", "2016"),
        (72,  "pgm", "2018"),
        (138, "pgm", "2020"),
        (186, "pgm", "2022"),
        (246, "pgm", "2024"),
    ],
    "log": [
        (198, "log", "2022"),
        (231, "log", "2023"),
        (269, "log", "2025"),
    ],
    "collas": [
        (199, "collas", "2022"),
        (232, "collas", "2023"),
        (274, "collas", "2024"),
    ],
    "cpal": [
        (234, "cpal", "2024"),
        (280, "cpal", "2025"),
    ],
    "isipta": [
        (103, "isipta", "2019"),
        (147, "isipta", "2021"),
        (215, "isipta", "2023"),
        (290, "isipta", "2025"),
    ],
}


def build_venue(venue_slug: str, force: bool = False) -> int:
    """Fetch all volumes for a venue and write a legacy JSONL.gz file.

    Returns the number of papers written.
    """
    volumes = WORKSHOP_VOLUMES[venue_slug]
    out_path = LEGACY_DIR / f"{venue_slug}-legacy.jsonl.gz"

    if out_path.exists() and not force:
        logger.info(f"{out_path.name} already exists — skipping (use --force to rebuild)")
        return 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Building {venue_slug.upper()} legacy from PMLR ({len(volumes)} volumes)...")

    all_papers = []
    for vol, venue, year in volumes:
        logger.info(f"  Fetching v{vol} ({venue_slug.upper()} {year})...")
        papers = process_volume(vol, venue, year)
        logger.info(f"    -> {len(papers)} papers")
        all_papers.extend(normalize_paper(p) for p in papers)

    if not all_papers:
        logger.warning(f"No papers found for {venue_slug.upper()}, skipping write")
        return 0

    write_legacy(out_path, all_papers)

    logger.info(f"Wrote {len(all_papers)} {venue_slug.upper()} papers to {out_path.name}")
    return len(all_papers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build legacy files for PMLR workshop/small-conference venues"
    )
    parser.add_argument(
        "--venue",
        choices=list(WORKSHOP_VOLUMES.keys()),
        help="Build legacy data for a single venue (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing legacy files",
    )
    args = parser.parse_args()

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    venues = [args.venue] if args.venue else list(WORKSHOP_VOLUMES.keys())
    total = 0

    for venue_slug in venues:
        count = build_venue(venue_slug, force=args.force)
        total += count

    print(f"\nDone. {total} total papers across {len(venues)} venue(s).")
    print("Next: python scripts/build_content.py")


if __name__ == "__main__":
    main()
