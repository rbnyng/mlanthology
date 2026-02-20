#!/usr/bin/env python3
"""One-time script: split workshop papers out of mixed CVF legacy files,
and fetch ICCVW papers from DBLP to create iccvw-legacy.jsonl.gz.

Steps performed:
  1. cvpr-legacy.jsonl.gz  →  cvpr-legacy.jsonl.gz (main conf only)
                           +  cvprw-legacy.jsonl.gz (workshop papers)
  2. wacv-legacy.jsonl.gz  →  wacv-legacy.jsonl.gz (main conf only)
                           +  wacvw-legacy.jsonl.gz (workshop papers)
  3. Fetch conf/iccvw/ from DBLP  →  iccvw-legacy.jsonl.gz
     (~1795 papers, biennial 2007–2023; no abstracts, same as other DBLP legacy)

The venue field on every paper was already set correctly by enrich_workshops.py,
so steps 1–2 are pure file-system splits with no reclassification.

Run once, then rebuild Hugo content:
    python scripts/split_workshop_legacy.py
    python scripts/build_content.py
"""

import logging
import tempfile
from collections import defaultdict
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, read_legacy, write_legacy

from adapters.dblp import fetch_all as dblp_fetch_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def split_legacy_file(source_slug: str) -> dict[str, int]:
    """Partition a mixed legacy file into per-venue files.

    The source file is rewritten in-place to contain only main-conference
    papers.  Each workshop slug found gets its own new file.  All writes
    use a temp-then-rename pattern so a crash mid-run leaves files intact.

    Returns {venue_slug: paper_count} for every file written.
    """
    source_path = LEGACY_DIR / f"{source_slug}-legacy.jsonl.gz"
    if not source_path.exists():
        logger.warning(f"  {source_path.name} not found, skipping")
        return {}

    by_venue: dict[str, list[dict]] = defaultdict(list)
    for p in read_legacy(source_path):
        by_venue[p.get("venue", source_slug)].append(p)

    total = sum(len(ps) for ps in by_venue.values())
    logger.info(
        f"  {source_slug}-legacy: {total} papers → "
        + ", ".join(f"{v}: {len(ps)}" for v, ps in sorted(by_venue.items()))
    )

    counts: dict[str, int] = {}
    for venue_slug, papers in sorted(by_venue.items()):
        out_path = LEGACY_DIR / f"{venue_slug}-legacy.jsonl.gz"
        write_legacy(out_path, papers, atomic=True)
        counts[venue_slug] = len(papers)
        action = "rewrote" if venue_slug == source_slug else "created"
        logger.info(f"  {action} {out_path.name}  ({len(papers)} papers)")

    return counts


def fetch_iccvw_legacy() -> int:
    """Fetch all ICCVW years from DBLP and write iccvw-legacy.jsonl.gz.

    Uses a temporary directory so the legacy file is only created when the
    full fetch succeeds.  Skips silently if the file already exists.
    """
    out_path = LEGACY_DIR / "iccvw-legacy.jsonl.gz"
    if out_path.exists():
        logger.info(f"  {out_path.name} already exists — skipping DBLP fetch")
        return 0

    logger.info(
        "Fetching ICCVW from DBLP (conf/iccvw/, biennial 2007–present) — "
        "this takes a few minutes due to rate limiting…"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dblp_fetch_all(
            venues=["iccvw"],
            output_dir=tmp_path,
            fill_only=False,
            backlog=True,   # write gzipped JSONL per year
        )

        # Merge all per-year files into a single legacy file.
        all_papers: list[dict] = []
        for gz_file in sorted(tmp_path.glob("iccvw-*.jsonl.gz")):
            all_papers.extend(read_legacy(gz_file))
            logger.info(f"  merged {gz_file.name}")

    if not all_papers:
        logger.warning("No ICCVW papers returned from DBLP")
        return 0

    write_legacy(out_path, all_papers)

    logger.info(f"Wrote {len(all_papers)} ICCVW papers to {out_path.name}")
    return len(all_papers)


def main() -> None:
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    # Steps 1–2: split CVPR and WACV legacy files by venue field.
    total_split = 0
    for source_slug in ["cvpr", "wacv"]:
        logger.info(f"Splitting {source_slug}-legacy.jsonl.gz…")
        counts = split_legacy_file(source_slug)
        total_split += sum(counts.values())

    # Step 3: fetch ICCVW from DBLP.
    total_iccvw = fetch_iccvw_legacy()

    print(
        f"\nDone."
        f"\n  Split {total_split} papers "
        f"(cvpr→cvpr+cvprw, wacv→wacv+wacvw)"
        f"\n  ICCVW: {total_iccvw} papers fetched"
        f"\n\nRun 'python scripts/build_content.py' to regenerate Hugo content."
    )


if __name__ == "__main__":
    main()
