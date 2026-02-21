#!/usr/bin/env python3
"""One-time script to build ALT legacy file.

  ALT (1990-2016): DBLP conf/alt — Springer LNAI, needs S2 enrichment.
  ALT (2017-2025): PMLR volumes 76/83/98/117/132/167/201/237/272
                   — fetched live into data/papers/ by the main pipeline.

Run once, then enrich with S2:
    python scripts/build_alt_legacy.py
    python scripts/enrich_legacy.py --s2-api-key KEY --venue alt --no-title-fallback
    python scripts/build_content.py
"""

import logging
import tempfile
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, read_legacy, write_legacy

from adapters.dblp import fetch_all as dblp_fetch_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_alt_legacy() -> None:
    out_path = LEGACY_DIR / "alt-legacy.jsonl.gz"
    if out_path.exists():
        logger.info("alt-legacy.jsonl.gz already exists — skipping ALT fetch")
        return

    logger.info(f"\n{'='*60}")
    logger.info("Building ALT legacy from DBLP (1990-2016)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dblp_fetch_all(
            venues=["alt"],
            output_dir=tmp_path,
            fill_only=False,
            backlog=True,
            max_year=2016,  # 2017+ is on PMLR, fetched into data/papers/
        )
        gz_files = sorted(tmp_path.glob("alt-*.jsonl.gz"))
        logger.info(f"  Merging {len(gz_files)} year files...")
        all_papers = []
        for gz in gz_files:
            all_papers.extend(read_legacy(gz))
            logger.info(f"    merged {gz.name} ({len(all_papers)} total so far)")

    if not all_papers:
        logger.warning("No papers returned for ALT, skipping write")
        return

    write_legacy(out_path, all_papers)

    logger.info(f"Wrote {len(all_papers)} ALT papers to {out_path.name}")


def main() -> None:
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    build_alt_legacy()
    print(
        "\nDone. Next steps:\n"
        "  python scripts/enrich_legacy.py --s2-api-key KEY --venue alt --no-title-fallback\n"
        "  python scripts/build_content.py"
    )


if __name__ == "__main__":
    main()
