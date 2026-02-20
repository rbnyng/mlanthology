#!/usr/bin/env python3
"""Build Hugo content pages from adapter JSON output.

Reads data/papers/*.json.gz and generates hugo/content/{venue}/{year}/*.md
with year-based organization: icml/2024/*.md
"""

import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict

import yaml

# Ensure project root is on sys.path so `scripts` is importable when
# invoked as `python scripts/build_content.py` (Python only adds the
# script's own directory, not its parent).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import ROOT, LEGACY_DIR
from scripts.titlecase import smart_title_case
from scripts.data_loader import load_all_papers
from scripts.page_builders import (
    sanitize,
    normalize_name_case,
    author_display,
    ascii_letter,
    yaml_scalar,
    build_paper_page,
    build_venue_index,
    build_year_index,
    build_author_page,
)

from adapters.common import slugify_author

DATA_DIR = ROOT / "data" / "papers"
BACKLOG_DIR = ROOT / "data" / "backlog"
CONTENT_DIR = ROOT / "hugo" / "content"
STATIC_PAGES_DIR = ROOT / "hugo" / "static_pages"
AUTHORS_DIR = CONTENT_DIR / "authors"
VENUES_YAML = ROOT / "hugo" / "data" / "venues.yaml"

# prefer C-accelerated YAML loader when available
try:
    _yaml_Loader = yaml.CSafeLoader
except AttributeError:
    _yaml_Loader = yaml.SafeLoader

with open(VENUES_YAML, encoding="utf-8") as _f:
    _VENUES = yaml.load(_f, Loader=_yaml_Loader) or {}


# ---------------------------------------------------------------------------
# Parallel worker data and functions
# ---------------------------------------------------------------------------

# shared with worker processes via fork COW
_WORKER_VY_DATA: dict = {}      # (venue, year) -> list[dict]
_WORKER_AUTHORS: dict = {}       # slug -> display_name
_WORKER_AUTHOR_PAPERS: dict = {} # slug -> list[dict]


def _write_venue_year_worker(vy_key: tuple) -> int:
    """Write all paper pages for one venue-year. Runs in a worker process."""
    venue, year = vy_key
    papers = _WORKER_VY_DATA[vy_key]

    year_dir = CONTENT_DIR / venue / year
    year_dir.mkdir(parents=True, exist_ok=True)

    (year_dir / "_index.md").write_text(
        build_year_index(venue, year, len(papers), _VENUES), encoding="utf-8"
    )

    count = 0
    for paper in papers:
        paper_id = paper.get("bibtex_key", "").replace("/", "-")
        if not paper_id:
            continue
        (year_dir / f"{paper_id}.md").write_text(
            build_paper_page(paper, _VENUES, slugify_author), encoding="utf-8"
        )
        count += 1
    return count


def _write_author_letter_worker(letter_and_slugs: tuple) -> tuple:
    """Write all author leaf pages for one letter. Runs in a worker process."""
    letter, slugs = letter_and_slugs

    letter_dir = AUTHORS_DIR / letter.lower()
    letter_dir.mkdir(parents=True, exist_ok=True)

    # Write letter _index.md (section page)
    letter_index = (
        f"---\n"
        f"title: {yaml_scalar(letter)}\n"
        f"letter: {yaml_scalar(letter)}\n"
        f"author_count: {len(slugs)}\n"
        f"---\n"
    )
    letter_index_path = letter_dir / "_index.md"
    if not letter_index_path.exists() or letter_index_path.read_text(encoding="utf-8") != letter_index:
        letter_index_path.write_text(letter_index, encoding="utf-8")

    # Clean stale files/dirs within this letter
    expected_files = {f"{s}.md" for s in slugs}
    existing_files = set()
    stale_count = 0
    for item in letter_dir.iterdir():
        if item.is_file() and item.name != "_index.md":
            existing_files.add(item.name)
        elif item.is_dir():
            shutil.rmtree(item)
            stale_count += 1
    for stale_file in existing_files - expected_files:
        (letter_dir / stale_file).unlink()
        stale_count += 1

    # Write author leaf pages
    written = 0
    skipped = 0
    for slug in slugs:
        display_name = _WORKER_AUTHORS[slug]
        leaf_path = letter_dir / f"{slug}.md"
        content = build_author_page(slug, display_name, _WORKER_AUTHOR_PAPERS.get(slug, []))

        if leaf_path.exists():
            try:
                if leaf_path.read_text(encoding="utf-8") == content:
                    skipped += 1
                    continue
            except (OSError, UnicodeDecodeError):
                pass

        leaf_path.write_text(content, encoding="utf-8")
        written += 1

    return written, skipped, stale_count


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def _clean_venue_dirs(venue_slugs: set[str]) -> None:
    """Remove old venue content directories before rebuilding."""
    all_venue_slugs = venue_slugs | set(_VENUES.keys())
    for slug in all_venue_slugs:
        venue_dir = CONTENT_DIR / slug
        if venue_dir.exists():
            shutil.rmtree(venue_dir)
    legacy_venues = CONTENT_DIR / "venues"
    if legacy_venues.exists():
        shutil.rmtree(legacy_venues)


def build_all(sample: bool = False):
    """Process all data files and generate Hugo content with year-based organization.

    When sample=True, only a small subset of venues/years is built for fast
    UI iteration (~2-5k papers instead of 180k+).
    """
    t_start = time.time()

    json_files = sorted(DATA_DIR.glob("*.json.gz"))
    if not json_files:
        print(f"No data files found in {DATA_DIR}")
        sys.exit(1)

    all_papers = load_all_papers(DATA_DIR, BACKLOG_DIR, LEGACY_DIR)

    if sample:
        _SAMPLE_VENUES = {
            "iclr", "colt", "jmlr", "tmlr",
            "iclrw", "jair"
        }
        venue_years: dict[str, set[str]] = defaultdict(set)
        for p in all_papers:
            v = p.get("venue", "").lower()
            y = str(p.get("year", ""))
            if v and y:
                venue_years[v].add(y)
        sample_vy = set()
        for v in _SAMPLE_VENUES:
            if v in venue_years:
                latest = max(venue_years[v])
                sample_vy.add((v, latest))
        all_papers = [
            p for p in all_papers
            if (p.get("venue", "").lower(), str(p.get("year", ""))) in sample_vy
        ]
        print(f"  Sample mode: {len(all_papers)} papers from {len(sample_vy)} venue-years")

    _VENUE_ALIASES = {"ecml": "ecmlpkdd"}

    papers_by_venue_year = defaultdict(list)
    authors = {}  # slug -> display_name
    author_papers = defaultdict(list)  # slug -> list of paper info dicts

    for paper in all_papers:
        venue = paper.get("venue", "").lower()
        venue = _VENUE_ALIASES.get(venue, venue)
        year = str(paper.get("year", ""))
        if not venue or not year:
            continue
        paper["venue"] = venue
        papers_by_venue_year[(venue, year)].append(paper)

        paper_id = paper.get("bibtex_key", "").replace("/", "-")
        paper_title = smart_title_case(sanitize(paper.get("title", "")))
        paper_authors_raw = paper.get("authors", [])

        author_list = []
        for a in paper_authors_raw:
            s = a.get("slug") or slugify_author(a)
            g = normalize_name_case(a.get("given", "").strip())
            f = normalize_name_case(a.get("family", "").strip())
            name = f"{g} {f}".strip()
            author_list.append({"name": name, "slug": s})

        paper_info = {
            "title": paper_title,
            "venue": venue,
            "year": year,
            "id": paper_id,
            "authors": author_list,
        }

        for a in paper_authors_raw:
            slug = a.get("slug") or slugify_author(a)
            if slug not in authors:
                authors[slug] = author_display(a)
            author_papers[slug].append(paper_info)

    for slug in author_papers:
        author_papers[slug].sort(key=lambda p: (-int(p["year"]), p["title"]))

    t_loaded = time.time()
    print(f"  Data loaded in {t_loaded - t_start:.1f}s")

    venue_years: dict[str, set] = defaultdict(set)
    paper_counts: dict[tuple, int] = {}
    for (venue, year), papers in papers_by_venue_year.items():
        venue_years[venue].add(year)
        paper_counts[(venue, year)] = len(papers)

    venue_slugs = set(venue_years.keys())
    _clean_venue_dirs(venue_slugs)

    authors_by_letter: dict[str, list[str]] = defaultdict(list)
    for slug, display_name in authors.items():
        letter = ascii_letter(display_name[0]) if display_name else "#"
        if not letter:
            letter = "#"
        authors_by_letter[letter].append(slug)

    if sample:
        if AUTHORS_DIR.exists():
            shutil.rmtree(AUTHORS_DIR)
    AUTHORS_DIR.mkdir(parents=True, exist_ok=True)
    (AUTHORS_DIR / "_index.md").write_text('---\ntitle: "Authors"\n---\n', encoding="utf-8")

    expected_letters = {l.lower() for l in authors_by_letter}
    if AUTHORS_DIR.exists():
        for d in AUTHORS_DIR.iterdir():
            if d.is_dir():
                if len(d.name) == 1 and d.name.lower() not in expected_letters:
                    shutil.rmtree(d)
                elif len(d.name) > 1:
                    shutil.rmtree(d)

    global _WORKER_VY_DATA, _WORKER_AUTHORS, _WORKER_AUTHOR_PAPERS
    _WORKER_VY_DATA = dict(papers_by_venue_year)
    _WORKER_AUTHORS = authors
    _WORKER_AUTHOR_PAPERS = dict(author_papers)

    n_workers = min(os.cpu_count() or 1, 8)
    vy_keys = list(papers_by_venue_year.keys())
    letter_items = list(authors_by_letter.items())

    with mp.Pool(processes=n_workers) as pool:
        paper_async = pool.map_async(_write_venue_year_worker, vy_keys)
        author_async = pool.map_async(_write_author_letter_worker, letter_items)

        paper_counts_list = paper_async.get()
        total_papers = sum(paper_counts_list)

        author_results = author_async.get()

    total_written = sum(r[0] for r in author_results)
    total_skipped = sum(r[1] for r in author_results)
    total_stale = sum(r[2] for r in author_results)

    t_gen = time.time()
    print(f"Generated {total_papers} paper pages across {len(venue_years)} venues")
    print(f"Venues: {', '.join(sorted(venue_years.keys()))}")
    print(f"Authors: {len(authors)} total, {total_written} written, {total_skipped} unchanged, {total_stale} removed")
    print(f"Content generated in {t_gen - t_loaded:.1f}s ({n_workers} workers)")

    for venue, years_set in venue_years.items():
        venue_dir = CONTENT_DIR / venue
        (venue_dir / "_index.md").write_text(
            build_venue_index(venue, years_set, paper_counts, _VENUES), encoding="utf-8"
        )

    if STATIC_PAGES_DIR.exists():
        for src in STATIC_PAGES_DIR.rglob("*"):
            if src.is_file():
                dest = CONTENT_DIR / src.relative_to(STATIC_PAGES_DIR)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

    t_end = time.time()
    print(f"Total build time: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Hugo content pages")
    parser.add_argument(
        "--sample", action="store_true",
        help="Build only a small subset of venues/years for fast UI iteration",
    )
    args = parser.parse_args()
    build_all(sample=args.sample)
