#!/usr/bin/env python3
"""Report metadata coverage per venue across all data files.

Reads every file in data/papers/ and data/legacy/ and prints a table
showing abstract, DOI, and pdf_url coverage, plus the number of papers
that are candidates for OpenAlex enrichment:

  - DoiNoAbs:    has a DOI but no abstract → DOI-based OpenAlex enrichment
  - NoDOINoAbs:  no DOI and no abstract    → title-search enrichment

Usage::

    python scripts/check_venue_coverage.py
    python scripts/check_venue_coverage.py --papers-only
    python scripts/check_venue_coverage.py --legacy-only
    python scripts/check_venue_coverage.py --sort abstract   # sort by abstract coverage
    python scripts/check_venue_coverage.py --min-gap 50      # only venues with >50 missing abstracts
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

PAPERS_DIR = Path(_project_root) / "data" / "papers"
LEGACY_DIR = Path(_project_root) / "data" / "legacy"


def _has(paper: dict, key: str) -> bool:
    val = paper.get(key)
    if isinstance(val, str):
        return bool(val.strip())
    return bool(val)


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.0f}%" if total else "n/a"


def _read_venue_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _read_legacy(path: Path) -> list[dict]:
    papers: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def _tally(papers: list[dict], stats: dict) -> None:
    for p in papers:
        stats["total"] += 1
        has_abs = _has(p, "abstract")
        has_doi = _has(p, "doi")
        has_pdf = _has(p, "pdf_url")
        if has_abs:
            stats["abstract"] += 1
        if has_doi:
            stats["doi"] += 1
        if has_pdf:
            stats["pdf_url"] += 1
        if has_doi and not has_abs:
            stats["doi_no_abstract"] += 1
        if not has_doi and not has_abs and _has(p, "title"):
            stats["no_doi_no_abstract"] += 1
        year = str(p.get("year", ""))
        if year:
            stats["years"].add(year)


def collect(
    *,
    include_papers: bool = True,
    include_legacy: bool = True,
) -> dict[str, dict]:
    venue_stats: dict[str, dict] = defaultdict(
        lambda: {
            "total": 0,
            "abstract": 0,
            "doi": 0,
            "pdf_url": 0,
            "doi_no_abstract": 0,
            "no_doi_no_abstract": 0,
            "years": set(),
            "source": "papers",
        }
    )

    if include_papers:
        for path in sorted(PAPERS_DIR.glob("*.json.gz")):
            try:
                data = _read_venue_json(path)
            except Exception as exc:
                print(f"WARNING: could not read {path.name}: {exc}", file=sys.stderr)
                continue
            venue = data.get("venue", path.stem.rsplit("-", 1)[0])
            _tally(data.get("papers", []), venue_stats[venue])

    if include_legacy:
        for path in sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz")):
            venue = path.name.replace("-legacy.jsonl.gz", "")
            key = f"{venue} (legacy)"
            try:
                papers = _read_legacy(path)
            except Exception as exc:
                print(f"WARNING: could not read {path.name}: {exc}", file=sys.stderr)
                continue
            venue_stats[key]["source"] = "legacy"
            _tally(papers, venue_stats[key])

    return dict(venue_stats)


def print_report(
    stats: dict[str, dict],
    sort_by: str = "missing_abstract",
    min_gap: int = 0,
) -> None:
    def _sort_key(item: tuple[str, dict]):
        name, s = item
        group = 0 if s["source"] == "papers" else 1
        missing = s["total"] - s["abstract"]
        pct_abs = s["abstract"] / s["total"] if s["total"] else 1.0
        if sort_by == "abstract":
            return (group, pct_abs)
        if sort_by == "doi_candidates":
            return (group, -s["doi_no_abstract"])
        if sort_by == "title_candidates":
            return (group, -s["no_doi_no_abstract"])
        # default: sort by number of missing abstracts desc
        return (group, -missing)

    rows = [
        (venue, s)
        for venue, s in stats.items()
        if (s["total"] - s["abstract"]) >= min_gap or min_gap == 0
    ]
    rows.sort(key=_sort_key)

    hdr = (
        f"{'Venue':<25} {'Papers':>7} {'Abstr%':>7} {'DOI%':>6} {'PDF%':>6}"
        f" {'DoiNoAbs':>9} {'NoDOINoAbs':>11}  Years"
    )
    print(hdr)
    print("-" * len(hdr))

    for venue, s in rows:
        t = s["total"]
        years = sorted(s["years"])
        yr_range = f"{years[0]}–{years[-1]}" if years else "?"
        print(
            f"{venue:<25} {t:>7,} {_pct(s['abstract'], t):>7} {_pct(s['doi'], t):>6}"
            f" {_pct(s['pdf_url'], t):>6} {s['doi_no_abstract']:>9,}"
            f" {s['no_doi_no_abstract']:>11,}  {yr_range}"
        )

    # Totals
    total_p = sum(s["total"] for _, s in rows)
    total_a = sum(s["abstract"] for _, s in rows)
    total_d = sum(s["doi"] for _, s in rows)
    total_pdf = sum(s["pdf_url"] for _, s in rows)
    total_doi_c = sum(s["doi_no_abstract"] for _, s in rows)
    total_title_c = sum(s["no_doi_no_abstract"] for _, s in rows)
    print("-" * len(hdr))
    print(
        f"{'TOTAL':<25} {total_p:>7,} {_pct(total_a, total_p):>7} {_pct(total_d, total_p):>6}"
        f" {_pct(total_pdf, total_p):>6} {total_doi_c:>9,} {total_title_c:>11,}"
    )
    print()
    print("OpenAlex enrichment opportunities:")
    print(f"  DOI-based  (have DOI, missing abstract): {total_doi_c:>6,} papers")
    print(f"  Title-search (no DOI, no abstract):      {total_title_c:>6,} papers")
    print(f"  Total missing abstracts:                 {total_p - total_a:>6,} papers")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report venue metadata coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--papers-only", action="store_true", help="Skip legacy files")
    parser.add_argument("--legacy-only", action="store_true", help="Skip papers/ files")
    parser.add_argument(
        "--sort",
        choices=["abstract", "doi_candidates", "title_candidates", "missing_abstract"],
        default="missing_abstract",
        help="Sort order within each source group (default: missing_abstract desc)",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=0,
        metavar="N",
        help="Only show venues with at least N missing abstracts",
    )
    args = parser.parse_args()

    stats = collect(
        include_papers=not args.legacy_only,
        include_legacy=not args.papers_only,
    )
    print_report(stats, sort_by=args.sort, min_gap=args.min_gap)


if __name__ == "__main__":
    main()
