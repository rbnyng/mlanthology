#!/usr/bin/env python3
"""One-time script: reclassify workshop papers in legacy data and patch venue_name.

Workshop detection requires both signals to match, avoiding false positives
(e.g. a UAI paper that happens to have an ICCVW DOI from a cross-publication):
  1. DOI prefix  (e.g. 10.1109/CVPRW.)
  2. source_id prefix matching the stored venue (e.g. conf/cvpr/)

After reclassification:
  - Workshop papers get a dedicated venue slug (cvpr→cvprw, wacv→wacvw)
  - Bibtex keys are regenerated with the new venue slug
  - venue_name is patched for all papers (main conference and workshop alike)

Run once, then rebuild Hugo content:
    python scripts/enrich_workshops.py
    python scripts/build_content.py
"""

import json
import re
import sys
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, read_legacy, write_legacy

from adapters.common import make_bibtex_key, resolve_bibtex_collisions

# Venue names aligned with DBLP_VENUES in adapters/dblp.py.
# Used to patch venue_name for all legacy papers regardless of workshop status.
VENUE_NAMES: dict[str, str] = {
    "cvpr":    "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "cvprw":   "IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops",
    "iccv":    "IEEE/CVF International Conference on Computer Vision",
    "iccvw":   "IEEE/CVF International Conference on Computer Vision Workshops",
    "eccv":    "European Conference on Computer Vision",
    "eccvw":   "European Conference on Computer Vision Workshops",
    "wacv":    "IEEE/CVF Winter Conference on Applications of Computer Vision",
    "wacvw":   "IEEE/CVF Winter Conference on Applications of Computer Vision Workshops",
    "icml":    "International Conference on Machine Learning",
    "iclr":    "International Conference on Learning Representations",
    "aistats": "International Conference on Artificial Intelligence and Statistics",
    "uai":     "Conference on Uncertainty in Artificial Intelligence",
    "colt":    "Annual Conference on Computational Learning Theory",
    "corl":    "Conference on Robot Learning",
    "aaai":    "AAAI Conference on Artificial Intelligence",
    "ijcai":   "International Joint Conference on Artificial Intelligence",
}

# Workshop detection rules.
# A paper is reclassified only when BOTH doi_re and sid_prefix match,
# preventing false positives from cross-publication DOIs.
# (src_venue, doi_re, source_id_prefix, target_venue)
WORKSHOP_RULES = [
    ("cvpr", re.compile(r"10\.\d+/CVPRW\.", re.I), "conf/cvpr/", "cvprw"),
    ("wacv", re.compile(r"10\.\d+/WACVW\.", re.I), "conf/wacv/", "wacvw"),
    ("iccv", re.compile(r"10\.\d+/ICCVW\.", re.I), "conf/iccv/", "iccvw"),
]


def process_legacy_file(path: Path) -> tuple[int, int]:
    """Process one legacy JSONL.gz file in-place.

    Returns (total_papers, reclassified_count).
    """
    papers: list[dict] = read_legacy(path)

    updated: list[dict] = []
    # Track indices and candidate keys for reclassified papers so we can
    # run collision resolution across them as a batch.
    reclassified_indices: list[int] = []
    candidate_keys: list[str] = []

    for i, paper in enumerate(papers):
        p = dict(paper)
        venue = p.get("venue", "")
        doi = p.get("doi", "")
        source_id = p.get("source_id", "")

        # Patch venue_name for every paper in this file.
        if venue in VENUE_NAMES:
            p["venue_name"] = VENUE_NAMES[venue]

        # Check workshop rules; both DOI and source_id must match.
        for src_venue, doi_re, sid_prefix, target_venue in WORKSHOP_RULES:
            if (venue == src_venue
                    and doi_re.search(doi)
                    and source_id.startswith(sid_prefix)):
                p["venue"] = target_venue
                p["venue_name"] = VENUE_NAMES.get(target_venue, "")
                authors = p.get("authors", [])
                if authors:
                    candidate = make_bibtex_key(
                        first_author_family=authors[0].get("family", ""),
                        year=str(p.get("year", "")),
                        venue=target_venue,
                        title=p.get("title", ""),
                    )
                    reclassified_indices.append(i)
                    candidate_keys.append(candidate)
                break

        updated.append(p)

    # Resolve collisions among the newly generated workshop keys.
    if candidate_keys:
        resolved = resolve_bibtex_collisions(candidate_keys)
        for idx, new_key in zip(reclassified_indices, resolved):
            updated[idx]["bibtex_key"] = new_key

    write_legacy(path, updated)

    return len(papers), len(reclassified_indices)


def main() -> None:
    files = sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz"))
    if not files:
        print(f"No legacy files found in {LEGACY_DIR}", file=sys.stderr)
        sys.exit(1)

    total_papers = 0
    total_reclassified = 0

    for path in files:
        venue = path.stem.replace("-legacy", "")
        count, reclassified = process_legacy_file(path)
        total_papers += count
        total_reclassified += reclassified
        note = f"  → reclassified {reclassified} as workshop" if reclassified else ""
        print(f"{venue:12s}  {count:5d} papers{note}")

    print(f"\nTotal: {total_reclassified} workshop papers reclassified out of {total_papers}.")
    if total_reclassified:
        print("Run 'python scripts/build_content.py' to regenerate Hugo content.")


if __name__ == "__main__":
    main()
