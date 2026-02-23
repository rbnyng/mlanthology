#!/usr/bin/env python3
"""One-time script to build ACML, ECML, and ECML-PKDD legacy files.

  ACML (2010-2024): PMLR volumes 13/20/25/29/39/45/63/77/95/101/
                    129/157/189/222/260 — full PDFs + abstracts.
  ECML (1993-2007): DBLP conf/ecml — Springer LNCS, needs S2 enrichment.
  ECML-PKDD (2008-2025): DBLP conf/pkdd main volumes only (stem_filter
                    pkdd{year}-N) — also Springer, needs S2 enrichment.

Run once, then enrich with S2 and rebuild content:
    python scripts/build_ecml_acml_legacy.py
    python scripts/enrich_legacy.py --s2-api-key KEY --venue ecml --no-title-fallback
    python scripts/enrich_legacy.py --s2-api-key KEY --venue ecmlpkdd --no-title-fallback
    python scripts/build_content.py
"""

import logging
import tempfile
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, read_legacy, write_legacy

from adapters.pmlr import process_volume
from adapters.dblp import fetch_all as dblp_fetch_all
from adapters.common import normalize_paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ACML_VOLUMES = [
    (13,  "acml", "2010"),
    (20,  "acml", "2011"),
    (25,  "acml", "2012"),
    (29,  "acml", "2013"),
    (39,  "acml", "2014"),
    (45,  "acml", "2015"),
    (63,  "acml", "2016"),
    (77,  "acml", "2017"),
    (95,  "acml", "2018"),
    (101, "acml", "2019"),
    (129, "acml", "2020"),
    (157, "acml", "2021"),
    (189, "acml", "2022"),
    (222, "acml", "2023"),
    (260, "acml", "2024"),
]


def build_acml() -> None:
    out_path = LEGACY_DIR / "acml-legacy.jsonl.gz"
    if out_path.exists():
        logger.info("acml-legacy.jsonl.gz already exists — skipping ACML fetch")
        return

    logger.info(f"\n{'='*60}")
    logger.info("Building ACML legacy from PMLR...")

    all_papers = []
    for vol, venue, year in ACML_VOLUMES:
        logger.info(f"  Fetching v{vol} (ACML {year})...")
        papers = process_volume(vol, venue, year)
        logger.info(f"    -> {len(papers)} papers")
        all_papers.extend(normalize_paper(p) for p in papers)

    write_legacy(out_path, all_papers)

    logger.info(f"Wrote {len(all_papers)} ACML papers to {out_path.name}")


def build_dblp_venue(slug: str, max_year: int | None = None) -> None:
    out_path = LEGACY_DIR / f"{slug}-legacy.jsonl.gz"
    if out_path.exists():
        logger.info(f"{out_path.name} already exists — skipping {slug.upper()} fetch")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Building {slug.upper()} legacy from DBLP...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dblp_fetch_all(
            venues=[slug],
            output_dir=tmp_path,
            fill_only=False,
            backlog=True,
            max_year=max_year,
        )
        gz_files = sorted(tmp_path.glob(f"{slug}-*.jsonl.gz"))
        logger.info(f"  Merging {len(gz_files)} year files...")
        all_papers = []
        for gz in gz_files:
            all_papers.extend(read_legacy(gz))
            logger.info(f"    merged {gz.name} ({len(all_papers)} total so far)")

    if not all_papers:
        logger.warning(f"No papers returned for {slug.upper()}, skipping write")
        return

    write_legacy(out_path, all_papers)

    logger.info(f"Wrote {len(all_papers)} {slug.upper()} papers to {out_path.name}")


def main() -> None:
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    build_acml()
    build_dblp_venue("ecml", max_year=2007)  # ECML merged into ECML-PKDD in 2008
    build_dblp_venue("ecmlpkdd")
    print(
        "\nDone. Next steps:\n"
        "  python scripts/enrich_legacy.py --s2-api-key KEY --venue ecml --no-title-fallback\n"
        "  python scripts/enrich_legacy.py --s2-api-key KEY --venue ecmlpkdd --no-title-fallback\n"
        "  python scripts/build_content.py"
    )


if __name__ == "__main__":
    main()
