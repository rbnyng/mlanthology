#!/usr/bin/env python3
"""Data validation: crawl all paper data and flag anomalies.

Loads every paper from data/papers/, data/backlog/, and data/legacy/
and runs heuristic checks for unusual or suspicious records.
Prints a report grouped by check, with per-file detail.

Usage:
    python -m scripts.validate_data              # full report
    python -m scripts.validate_data --summary    # counts only
    python -m scripts.validate_data --venue icml # one venue only
    python -m scripts.validate_data --primary    # skip legacy/backlog
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

# ── data loading ─────────────────────────────────────────────────────

PAPERS_DIR = ROOT / "data" / "papers"
BACKLOG_DIR = ROOT / "data" / "backlog"
LEGACY_DIR = ROOT / "data" / "legacy"
VENUES_YAML = ROOT / "hugo" / "data" / "venues.yaml"

try:
    _Loader = yaml.CSafeLoader
except AttributeError:
    _Loader = yaml.SafeLoader


def load_venues() -> set[str]:
    with open(VENUES_YAML) as f:
        data = yaml.load(f, Loader=_Loader) or {}
    return set(data.keys())


def iter_papers(
    venue_filter: str | None = None,
    primary_only: bool = False,
) -> list[tuple[str, dict, str]]:
    """Return (source_file, paper_dict, data_tier) tuples.

    data_tier is one of "primary", "backlog", or "legacy".
    """
    papers: list[tuple[str, dict, str]] = []

    for jf in sorted(PAPERS_DIR.glob("*.json.gz")):
        if venue_filter and not jf.stem.startswith(venue_filter + "-"):
            continue
        with gzip.open(jf, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("papers", []):
            papers.append((jf.name, p, "primary"))

    if primary_only:
        return papers

    for directory, tier in [(BACKLOG_DIR, "backlog"), (LEGACY_DIR, "legacy")]:
        if not directory.exists():
            continue
        for jf in sorted(directory.glob("*.jsonl.gz")):
            if venue_filter and not jf.stem.startswith(venue_filter + "-"):
                continue
            with gzip.open(jf, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        papers.append((jf.name, json.loads(line), tier))

    return papers


# ── checks ───────────────────────────────────────────────────────────

# Mojibake: double-encoded UTF-8 signature (Ã followed by high byte).
_MOJIBAKE_RE = re.compile(r"\xc3[\x80-\xbf]")

# HTML tags that are NOT just ML-domain inline tokens.
# We flag real structural HTML tags: div, span, table, a, img, etc.
# We skip sub/sup (legitimate math formatting), and we skip lone
# angle-bracket tokens common in NLP papers like <human>, <SEG>, <EOS>.
_STRUCTURAL_HTML_RE = re.compile(
    r"</?(?:div|span|table|tr|td|th|ul|ol|li|a|img|br|hr|font|style|script|link|meta|head|body|html)\b[^>]*>",
    re.IGNORECASE,
)

# HTML entities that should have been decoded
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")

REQUIRED_FIELDS = ["bibtex_key", "title", "authors", "year", "venue"]


class Finding:
    """A single data quality issue."""

    __slots__ = ("check", "source_file", "bibtex_key", "detail")

    def __init__(self, check: str, source_file: str, bibtex_key: str, detail: str):
        self.check = check
        self.source_file = source_file
        self.bibtex_key = bibtex_key
        self.detail = detail

    def __repr__(self) -> str:
        return f"  [{self.source_file}] {self.bibtex_key}: {self.detail}"


def validate(
    papers: list[tuple[str, dict, str]],
    known_venues: set[str],
) -> list[Finding]:
    """Run all checks and return a list of findings."""
    findings: list[Finding] = []
    global_keys: Counter[str] = Counter()
    title_by_venue_year: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    def flag(check: str, src: str, key: str, detail: str) -> None:
        findings.append(Finding(check, src, key, detail))

    for source_file, p, tier in papers:
        key = p.get("bibtex_key", "<missing>")
        is_legacy = tier in ("legacy", "backlog")

        # ── required fields ──────────────────────────────────────
        for field in REQUIRED_FIELDS:
            if field not in p:
                flag("missing_field", source_file, key, f"missing '{field}'")
            elif not p[field] and field != "authors":
                flag("empty_required_field", source_file, key, f"'{field}' is empty")

        # ── bibtex key format ────────────────────────────────────
        if "bibtex_key" in p:
            global_keys[key] += 1
            if not re.match(r"^[a-z0-9][a-z0-9-]*$", key):
                flag("bibtex_key_bad_chars", source_file, key,
                     f"key contains unexpected characters: {key!r}")
            # Check structure: should look like name+year+venue-word
            m = re.match(r"^([a-z]+)(\d{4})([a-z0-9]+)-(.+)$", key)
            if m:
                key_year = m.group(2)
                if str(p.get("year", "")) != key_year:
                    flag("bibtex_key_year_mismatch", source_file, key,
                         f"key year {key_year} != paper year {p.get('year')}")
            elif key != "<missing>":
                if not re.match(r"^[a-z]+\d{4}[a-z0-9]+-[a-z0-9-]+$", key):
                    flag("bibtex_key_format", source_file, key,
                         f"key doesn't match expected pattern: {key!r}")

        # ── title checks ─────────────────────────────────────────
        title = p.get("title", "")
        if title:
            if len(title) < 10:
                flag("short_title", source_file, key,
                     f"title only {len(title)} chars: {title!r}")

            if len(title) > 300:
                flag("long_title", source_file, key,
                     f"title is {len(title)} chars: {title[:80]}...")

            if _STRUCTURAL_HTML_RE.search(title):
                flag("html_in_title", source_file, key,
                     f"HTML tag in title: {title[:100]}")

            if _HTML_ENTITY_RE.search(title):
                flag("html_entity_in_title", source_file, key,
                     f"undecoded HTML entity: {title[:100]}")

            if _MOJIBAKE_RE.search(title):
                flag("mojibake_in_title", source_file, key,
                     f"possible mojibake: {title[:100]}")

            # All caps (ignoring digits/symbols): suggests missed normalization
            alpha = re.sub(r"[^a-zA-Z]", "", title)
            if len(alpha) > 10 and alpha == alpha.upper():
                flag("allcaps_title", source_file, key,
                     f"all-caps title: {title[:80]}")

            # Unbalanced LaTeX $ delimiters
            dollar_count = title.count("$")
            if dollar_count % 2 != 0:
                flag("unbalanced_latex", source_file, key,
                     f"odd number of $ ({dollar_count}): {title[:80]}")

            # Track titles for duplicate detection
            venue = p.get("venue", "")
            year = str(p.get("year", ""))
            norm_title = re.sub(r"[^a-z0-9]", "", title.lower())
            title_by_venue_year[f"{venue}-{year}"].append(
                (norm_title, key, title)
            )

        # ── author checks ────────────────────────────────────────
        authors = p.get("authors", [])
        if isinstance(authors, list):
            if len(authors) == 0:
                flag("no_authors", source_file, key, "paper has zero authors")

            if len(authors) > 100:
                flag("extreme_author_count", source_file, key,
                     f"paper has {len(authors)} authors")
            elif len(authors) > 50:
                flag("many_authors", source_file, key,
                     f"paper has {len(authors)} authors")

            slugs_in_paper: list[str] = []
            for i, author in enumerate(authors):
                given = author.get("given", "")
                family = author.get("family", "")
                slug = author.get("slug", "")

                if not family and not given:
                    flag("empty_author", source_file, key,
                         f"author[{i}] has no name at all")
                elif not family:
                    flag("empty_family_name", source_file, key,
                         f"author[{i}] missing family name: given={given!r}")

                # Slug checks only for primary data — legacy data
                # predates the slug system
                if not is_legacy:
                    if not slug:
                        flag("missing_slug", source_file, key,
                             f"author[{i}] missing slug: {given} {family}")
                    elif slug == "unknown":
                        flag("unknown_slug", source_file, key,
                             f"author[{i}] slug is 'unknown': {given} {family}")

                # HTML entity remnants in author names
                full = f"{given} {family}"
                if _HTML_ENTITY_RE.search(full):
                    flag("html_entity_in_author", source_file, key,
                         f"author[{i}] has HTML entity: {full}")

                if _MOJIBAKE_RE.search(full):
                    flag("mojibake_in_author", source_file, key,
                         f"author[{i}] possible mojibake: {full}")

                # Email-like patterns in names
                if "@" in full:
                    flag("email_in_author", source_file, key,
                         f"author[{i}] looks like email: {full}")

                # Digits in author names (unusual but not uncommon — skip)
                # Parenthetical annotations that weren't cleaned
                if re.search(
                    r"\((?:PhD|Dr\.|Jr\.|Sr\.|He/Him|She/Her|They/Them)\)",
                    full,
                    re.IGNORECASE,
                ):
                    flag("annotation_in_author", source_file, key,
                         f"author[{i}] has uncleaned annotation: {full}")

                # Asterisk (equal contribution marker)
                if "*" in full:
                    flag("asterisk_in_author", source_file, key,
                         f"author[{i}] has asterisk: {full}")

                # Suspiciously long single name part
                if len(given) > 60 or len(family) > 60:
                    flag("long_author_name", source_file, key,
                         f"author[{i}] suspiciously long: {full[:80]}")

                slugs_in_paper.append(slug)

            # Duplicate slugs within same paper (only when slugs exist)
            slug_counts = Counter(s for s in slugs_in_paper if s)
            for s, c in slug_counts.items():
                if c > 1:
                    flag("duplicate_author_in_paper", source_file, key,
                         f"author slug {s!r} appears {c} times")
        else:
            flag("authors_not_list", source_file, key,
                 f"authors field is {type(authors).__name__}, not list")

        # ── year checks ──────────────────────────────────────────
        year_str = str(p.get("year", ""))
        if year_str:
            try:
                year_int = int(year_str)
                if year_int < 1950:
                    flag("ancient_year", source_file, key,
                         f"year {year_int} is before 1950")
                if year_int > 2026:
                    flag("future_year", source_file, key,
                         f"year {year_int} is in the future")
            except ValueError:
                flag("invalid_year", source_file, key,
                     f"year is not a number: {year_str!r}")

            # Year should match filename for primary data files
            if source_file.endswith(".json.gz"):
                fname_year = source_file.replace(".json.gz", "").rsplit("-", 1)[-1]
                if year_str != fname_year:
                    flag("year_file_mismatch", source_file, key,
                         f"paper year {year_str} != filename year {fname_year}")

        # ── venue checks ─────────────────────────────────────────
        venue = p.get("venue", "")
        if venue and venue not in known_venues:
            flag("unknown_venue", source_file, key,
                 f"venue {venue!r} not in venues.yaml")

        # ── URL checks ───────────────────────────────────────────
        for url_field in ["pdf_url", "venue_url", "openreview_url", "code_url"]:
            url = p.get(url_field, "")
            if url:
                if not url.startswith("http://") and not url.startswith("https://"):
                    flag("bad_url", source_file, key,
                         f"{url_field} doesn't start with http: {url[:80]}")
                if " " in url:
                    flag("space_in_url", source_file, key,
                         f"{url_field} contains spaces: {url[:80]}")

        # pdf_url: flag non-PDF-looking links (skip download URLs and DOI links)
        pdf = p.get("pdf_url", "")
        if pdf:
            pdf_lower = pdf.lower()
            looks_like_pdf = (
                pdf_lower.endswith(".pdf")
                or "/pdf" in pdf_lower
                or "download/" in pdf_lower
                or "doi.org/" in pdf_lower
            )
            if not looks_like_pdf:
                flag("pdf_url_suspicious", source_file, key,
                     f"pdf_url doesn't look like a PDF link: {pdf[:80]}")

        # ── abstract checks ──────────────────────────────────────
        abstract = p.get("abstract", "")
        if abstract:
            if len(abstract) > 10000:
                flag("very_long_abstract", source_file, key,
                     f"abstract is {len(abstract)} chars")

            if _MOJIBAKE_RE.search(abstract):
                idx = abstract.index("\xc3")
                context = abstract[max(0, idx - 20):idx + 30]
                flag("mojibake_in_abstract", source_file, key,
                     f"possible mojibake in abstract: ...{context}...")

            if _STRUCTURAL_HTML_RE.search(abstract):
                match = _STRUCTURAL_HTML_RE.search(abstract)
                flag("html_in_abstract", source_file, key,
                     f"structural HTML in abstract: {match.group()}")

    # ── cross-paper checks ───────────────────────────────────────

    # Global bibtex key duplicates (across all files)
    for bkey, count in global_keys.items():
        if count > 1:
            flag("global_duplicate_key", "<cross-file>", bkey,
                 f"bibtex_key appears {count} times across all data")

    # Duplicate titles within same venue-year
    for vy, entries in title_by_venue_year.items():
        seen: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for norm, bkey, raw in entries:
            if len(norm) > 15:  # skip short titles to avoid false positives
                seen[norm].append((bkey, raw))
        for norm, dupes in seen.items():
            if len(dupes) > 1:
                keys_str = ", ".join(d[0] for d in dupes)
                flag("duplicate_title", f"<{vy}>", dupes[0][0],
                     f"title appears {len(dupes)} times: "
                     f"{dupes[0][1][:60]}... keys: {keys_str}")

    # ── per-file statistics anomalies ────────────────────────────
    file_counts: Counter[str] = Counter()
    for source_file, _, tier in papers:
        if tier == "primary":
            file_counts[source_file] += 1

    # Flag files with very few papers — might indicate a fetch failure
    for fname, count in sorted(file_counts.items()):
        if count < 5:
            flag("tiny_file", fname, "<file-level>",
                 f"only {count} papers in {fname}")

    return findings


# ── reporting ────────────────────────────────────────────────────────

_ERRORS = frozenset({
    "missing_field", "empty_required_field", "no_authors",
    "authors_not_list", "invalid_year", "bibtex_key_bad_chars",
    "global_duplicate_key", "empty_author",
})
_WARNINGS = frozenset({
    "html_in_title", "html_entity_in_title", "mojibake_in_title",
    "mojibake_in_author", "mojibake_in_abstract", "html_in_abstract",
    "html_entity_in_author", "email_in_author", "asterisk_in_author",
    "annotation_in_author", "bad_url", "space_in_url",
    "allcaps_title", "unbalanced_latex", "duplicate_title",
    "bibtex_key_year_mismatch", "year_file_mismatch",
    "unknown_venue", "unknown_slug", "empty_family_name",
    "missing_slug", "duplicate_author_in_paper", "pdf_url_suspicious",
})


def _severity(check: str) -> str:
    if check in _ERRORS:
        return "ERROR"
    if check in _WARNINGS:
        return " WARN"
    return " INFO"


def print_report(
    findings: list[Finding],
    *,
    summary_only: bool = False,
    total_papers: int = 0,
) -> None:
    """Print findings grouped by check type."""
    by_check: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_check[f.check].append(f)

    total_checks = len(by_check)
    total_issues = len(findings)

    print(f"\n{'='*70}")
    print("DATA VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"  Scanned {total_papers:,} papers")

    if total_issues == 0:
        print("  All checks passed — no issues found.")
        return

    errors = sum(1 for f in findings if _severity(f.check) == "ERROR")
    warnings = sum(1 for f in findings if _severity(f.check) == " WARN")
    infos = total_issues - errors - warnings
    print(f"  Found {total_issues:,} issues across {total_checks} check types")
    print(f"  ({errors} errors, {warnings} warnings, {infos} info)\n")

    # Print checks sorted by severity then count
    severity_order = {"ERROR": 0, " WARN": 1, " INFO": 2}
    for check, items in sorted(
        by_check.items(),
        key=lambda x: (severity_order.get(_severity(x[0]), 9), -len(x[1])),
    ):
        severity = _severity(check)
        print(f"[{severity}] {check}: {len(items)}")
        if not summary_only:
            for item in items[:5]:
                print(repr(item))
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
        print()


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ML proceedings data for anomalies."
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print counts only, no per-issue detail.",
    )
    parser.add_argument(
        "--venue", type=str, default=None,
        help="Only check files for this venue slug (e.g. icml).",
    )
    parser.add_argument(
        "--primary", action="store_true",
        help="Only check primary data (skip legacy and backlog).",
    )
    args = parser.parse_args()

    known_venues = load_venues()
    print(f"Loaded {len(known_venues)} venues from venues.yaml")

    papers = iter_papers(venue_filter=args.venue, primary_only=args.primary)
    tiers = Counter(tier for _, _, tier in papers)
    parts = [f"{tiers['primary']} primary"]
    if tiers["backlog"]:
        parts.append(f"{tiers['backlog']} backlog")
    if tiers["legacy"]:
        parts.append(f"{tiers['legacy']} legacy")
    print(f"Loaded {len(papers):,} papers ({', '.join(parts)})")

    findings = validate(papers, known_venues)
    print_report(findings, summary_only=args.summary, total_papers=len(papers))

    has_errors = any(_severity(f.check) == "ERROR" for f in findings)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
