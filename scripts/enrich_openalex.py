#!/usr/bin/env python3
"""Enrich venue data with abstracts (and optionally OA PDF URLs) from OpenAlex.

Covers both legacy (data/legacy/*.jsonl.gz) and papers (data/papers/*.json.gz).

By default only processes IEEE DOIs (10.1109/) in legacy files, and IEEE
papers-file venues (CVPR, ICCV, WACV, workshops).  Use --all-venues to
process every papers file regardless of DOI prefix or venue.

OpenAlex uses credit-based billing: list/filter queries cost 1 credit,
search queries cost 10 credits.  Free API keys get 10,000 credits/day ($1).

Usage::

    # Enrich all IEEE venues (legacy + papers files) — original behaviour
    python scripts/enrich_openalex.py

    # Enrich a single venue (any venue, not just IEEE)
    python scripts/enrich_openalex.py --venue colt

    # Enrich ALL venues in data/papers/ (non-IEEE too)
    python scripts/enrich_openalex.py --all-venues

    # Also fill missing pdf_url from OpenAlex open-access links
    python scripts/enrich_openalex.py --all-venues --oa-pdf

    # Dry run: query OpenAlex and show coverage stats without writing
    python scripts/enrich_openalex.py --all-venues --oa-pdf --dry-run

    # Provide API key directly (or set OPENALEX_API_KEY env var)
    python scripts/enrich_openalex.py --api-key YOUR_KEY

    # Limit to first N candidates per file (for testing)
    python scripts/enrich_openalex.py --venue colt --limit 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from adapters.common import read_venue_json, write_venue_json
from adapters.openalex import (
    CREDIT_COST_LIST,
    CREDIT_COST_SEARCH,
    _get_api_key,
    check_rate_limit,
    fetch_metadata_by_doi,
    fetch_metadata_by_title,
)
from scripts.utils import LEGACY_DIR, PAPERS_DIR, read_legacy, write_legacy

logger = logging.getLogger(__name__)

# legacy files are all IEEE (CVF); sorted smallest-missing-abstract first.
IEEE_VENUES = ["wacvw", "wacv", "iccvw", "cvprw", "iccv", "cvpr"]



def _is_ieee_doi(doi: str) -> bool:
    return doi.startswith("10.1109/")


def _candidates(papers: list[dict], *, ieee_only: bool, oa_pdf: bool) -> list[dict]:
    """Return papers that need enrichment.

    A paper qualifies when it has a DOI and is missing at least one of:
      - abstract (always checked)
      - pdf_url  (only checked when oa_pdf=True)

    When ieee_only=True only IEEE DOIs (10.1109/) are considered.
    """
    out = []
    for p in papers:
        doi = p.get("doi", "").strip()
        if not doi:
            continue
        if ieee_only and not _is_ieee_doi(doi):
            continue
        missing_abstract = not p.get("abstract", "").strip()
        missing_pdf = oa_pdf and not p.get("pdf_url", "").strip()
        if missing_abstract or missing_pdf:
            out.append(p)
    return out


def _apply_metadata(
    papers: list[dict],
    meta: dict[str, dict],
    *,
    oa_pdf: bool,
) -> tuple[int, int, int]:
    """Apply OpenAlex metadata to papers in-place.

    Fills abstract and (when oa_pdf=True) pdf_url.  Never overwrites
    existing values.  Returns (papers_changed, abstracts_added, pdfs_added).
    """
    lower_meta = {k.lower(): v for k, v in meta.items()}

    changed = abs_added = pdf_added = 0
    for paper in papers:
        doi = paper.get("doi", "").strip().lower()
        if doi not in lower_meta:
            continue
        m = lower_meta[doi]
        paper_changed = False

        if not paper.get("abstract", "").strip() and "abstract" in m:
            paper["abstract"] = m["abstract"]
            abs_added += 1
            paper_changed = True

        if oa_pdf and not paper.get("pdf_url", "").strip() and "oa_url" in m:
            paper["pdf_url"] = m["oa_url"]
            pdf_added += 1
            paper_changed = True

        if paper_changed:
            src = paper.get("source", "")
            paper["source"] = f"{src}+openalex" if src else "openalex"
            changed += 1

    return changed, abs_added, pdf_added


def _fetch_and_apply(
    papers: list[dict],
    candidates: list[dict],
    *,
    api_key: str | None,
    dry_run: bool,
    oa_pdf: bool,
    limit: int,
) -> int:
    """Fetch OpenAlex metadata and apply to *papers* in-place.

    Returns papers changed, or abstract-hit count when dry_run=True.
    """
    target = candidates[:limit] if limit > 0 else candidates
    if limit > 0:
        logger.info(f"  Limiting to {len(target)} papers (--limit {limit})")

    dois = [p["doi"] for p in target]
    t0 = time.time()
    meta = fetch_metadata_by_doi(dois, api_key=api_key)
    elapsed = time.time() - t0
    logger.info(
        f"  OpenAlex: {len(meta)}/{len(dois)} records retrieved "
        f"in {elapsed:.1f}s ({len(dois) / max(elapsed, 0.01):.0f} DOIs/s)"
    )

    if dry_run:
        _report_dry_run(meta, len(dois), oa_pdf=oa_pdf)
        return sum(1 for m in meta.values() if "abstract" in m)

    changed, abs_added, pdf_added = _apply_metadata(papers, meta, oa_pdf=oa_pdf)
    logger.info(
        f"  abstract: +{abs_added}"
        + (f", pdf_url: +{pdf_added}" if oa_pdf else "")
    )
    return changed


def _report_dry_run(meta: dict[str, dict], n_candidates: int, *, oa_pdf: bool) -> None:
    abs_hits = sum(1 for m in meta.values() if "abstract" in m)
    pdf_hits = sum(1 for m in meta.values() if "oa_url" in m)
    if oa_pdf:
        logger.info(
            f"  DRY RUN: {abs_hits}/{n_candidates} abstracts, "
            f"{pdf_hits}/{n_candidates} OA PDFs available on OpenAlex"
        )
    else:
        logger.info(
            f"  DRY RUN: {abs_hits}/{n_candidates} abstracts available on OpenAlex"
        )



def _title_search_candidates(papers: list[dict]) -> list[int]:
    """Return indices of papers missing abstract and having no DOI."""
    return [
        i for i, p in enumerate(papers)
        if not p.get("abstract", "").strip()
        and not p.get("doi", "").strip()
        and p.get("title", "").strip()
    ]


def _apply_title_metadata(
    papers: list[dict],
    meta: dict[int, dict],
    candidate_indices: list[int],
) -> tuple[int, int, int, int]:
    """Apply title-search metadata to papers in-place.

    Returns (papers_changed, abstracts_added, dois_added, pdfs_added).
    """
    changed = abs_added = doi_added = pdf_added = 0
    for subset_idx, m in meta.items():
        paper = papers[candidate_indices[subset_idx]]
        paper_changed = False

        if not paper.get("abstract", "").strip() and "abstract" in m:
            paper["abstract"] = m["abstract"]
            abs_added += 1
            paper_changed = True

        if not paper.get("doi", "").strip() and "doi" in m:
            paper["doi"] = m["doi"]
            doi_added += 1
            paper_changed = True

        if not paper.get("pdf_url", "").strip() and "oa_url" in m:
            paper["pdf_url"] = m["oa_url"]
            pdf_added += 1
            paper_changed = True

        if paper_changed:
            src = paper.get("source", "")
            paper["source"] = f"{src}+openalex" if src else "openalex"
            changed += 1

    return changed, abs_added, doi_added, pdf_added


def enrich_legacy_file(
    path: Path,
    *,
    api_key: str | None = None,
    dry_run: bool = False,
    ieee_only: bool = True,
    oa_pdf: bool = False,
    limit: int = 0,
) -> int:
    """Enrich a legacy JSONL.gz file via OpenAlex.  Returns papers enriched."""
    papers = read_legacy(path)
    if not papers:
        logger.info(f"{path.name}: empty, skipping")
        return 0

    total = len(papers)
    with_abstract = sum(1 for p in papers if p.get("abstract", "").strip())
    candidates = _candidates(papers, ieee_only=ieee_only, oa_pdf=oa_pdf)
    doi_label = "IEEE DOIs" if ieee_only else "DOIs"
    logger.info(
        f"\n{'=' * 60}\n{path.name}: {total} papers, "
        f"{with_abstract} with abstract, "
        f"{len(candidates)} {doi_label} needing enrichment"
    )

    if not candidates:
        logger.info("  Nothing to enrich, skipping")
        return 0

    changed = _fetch_and_apply(
        papers, candidates,
        api_key=api_key, dry_run=dry_run, oa_pdf=oa_pdf, limit=limit,
    )

    if not dry_run:
        if changed > 0:
            write_legacy(path, papers, atomic=True)
            logger.info(f"  Written back to {path.name}")
        else:
            logger.info("  No changes to write")
        remaining = len(_candidates(papers, ieee_only=ieee_only, oa_pdf=oa_pdf))
        if remaining:
            logger.info(f"  {remaining} papers still need enrichment")

    return changed


def enrich_papers_file(
    path: Path,
    *,
    api_key: str | None = None,
    dry_run: bool = False,
    ieee_only: bool = True,
    oa_pdf: bool = False,
    limit: int = 0,
) -> int:
    """Enrich a venue JSON.gz papers file via OpenAlex.  Returns papers enriched."""
    data = read_venue_json(path)
    papers = data.get("papers", [])
    if not papers:
        return 0

    candidates = _candidates(papers, ieee_only=ieee_only, oa_pdf=oa_pdf)
    if not candidates:
        return 0

    doi_label = "IEEE DOIs" if ieee_only else "DOIs"
    logger.info(
        f"\n{'=' * 60}\n{path.name}: {len(papers)} papers, "
        f"{len(candidates)} {doi_label} needing enrichment"
    )

    changed = _fetch_and_apply(
        papers, candidates,
        api_key=api_key, dry_run=dry_run, oa_pdf=oa_pdf, limit=limit,
    )

    if not dry_run and changed > 0:
        write_venue_json(
            data["venue"],
            data["year"],
            papers,
            path.parent,
            filename=path.name.replace(".json.gz", ""),
        )
        logger.info(f"  Written {path.name}")

    return changed



def _title_search_file(
    papers: list[dict],
    *,
    api_key: str | None,
    dry_run: bool,
    limit: int,
) -> int:
    """Run title-based OpenAlex search on papers missing abstract+DOI.

    Returns number of papers enriched.
    """
    indices = _title_search_candidates(papers)
    if not indices:
        return 0

    target_indices = indices[:limit] if limit > 0 else indices
    target_papers = [papers[i] for i in target_indices]

    logger.info(f"  Title search: {len(target_papers)} candidates")

    t0 = time.time()
    meta = fetch_metadata_by_title(target_papers, api_key=api_key)
    elapsed = time.time() - t0

    abs_hits = sum(1 for m in meta.values() if "abstract" in m)
    logger.info(
        f"  Title search: {abs_hits}/{len(target_papers)} abstracts found "
        f"in {elapsed:.1f}s"
    )

    if dry_run:
        return abs_hits

    changed, abs_added, doi_added, pdf_added = _apply_title_metadata(
        papers, meta, target_indices
    )
    logger.info(
        f"  Title search applied: abstract +{abs_added}, doi +{doi_added}, pdf +{pdf_added}"
    )
    return changed


def enrich_legacy_title_search(
    path: Path,
    *,
    api_key: str | None = None,
    dry_run: bool = False,
    limit: int = 0,
) -> int:
    """Enrich a legacy file via OpenAlex title search.  Returns papers enriched."""
    papers = read_legacy(path)
    if not papers:
        return 0

    indices = _title_search_candidates(papers)
    if not indices:
        return 0

    total = len(papers)
    with_abstract = sum(1 for p in papers if p.get("abstract", "").strip())
    logger.info(
        f"\n{'=' * 60}\n{path.name}: {total} papers, "
        f"{with_abstract} with abstract, "
        f"{len(indices)} without abstract or DOI (title search)"
    )

    changed = _title_search_file(
        papers, api_key=api_key, dry_run=dry_run, limit=limit,
    )

    if not dry_run and changed > 0:
        write_legacy(path, papers, atomic=True)
        logger.info(f"  Written back to {path.name}")

    return changed


def enrich_papers_title_search(
    path: Path,
    *,
    api_key: str | None = None,
    dry_run: bool = False,
    limit: int = 0,
) -> int:
    """Enrich a papers file via OpenAlex title search.  Returns papers enriched."""
    data = read_venue_json(path)
    papers = data.get("papers", [])
    if not papers:
        return 0

    indices = _title_search_candidates(papers)
    if not indices:
        return 0

    logger.info(
        f"\n{'=' * 60}\n{path.name}: {len(papers)} papers, "
        f"{len(indices)} without abstract or DOI (title search)"
    )

    changed = _title_search_file(
        papers, api_key=api_key, dry_run=dry_run, limit=limit,
    )

    if not dry_run and changed > 0:
        write_venue_json(
            data["venue"],
            data["year"],
            papers,
            path.parent,
            filename=path.name.replace(".json.gz", ""),
        )
        logger.info(f"  Written {path.name}")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich venue data with abstracts/OA PDFs from OpenAlex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help=(
            "Enrich a single venue by name (e.g. 'cvpr', 'colt', 'uai'). "
            "Works for any venue in data/papers/. "
            "Default: all IEEE venues."
        ),
    )
    parser.add_argument(
        "--all-venues",
        action="store_true",
        help=(
            "Process every file in data/papers/ regardless of venue or DOI "
            "prefix (implies --no-legacy)."
        ),
    )
    parser.add_argument(
        "--title-search",
        action="store_true",
        help=(
            "Use title-based OpenAlex search for papers missing both "
            "abstract and DOI.  Slower than DOI lookup but covers papers "
            "without DOIs (AAAI, IJCAI, ICML legacy, etc.)."
        ),
    )
    parser.add_argument(
        "--oa-pdf",
        action="store_true",
        help="Also fill missing pdf_url from OpenAlex open-access links.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "OpenAlex API key for higher rate limits "
            "(overrides OPENALEX_API_KEY env var)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Query OpenAlex and report coverage stats without writing any changes."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N candidates per file (for testing).",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Skip legacy (data/legacy/) files; only process data/papers/ files.",
    )
    parser.add_argument(
        "--no-papers",
        action="store_true",
        help="Skip data/papers/ files; only process legacy files.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = _get_api_key(args.api_key)
    if api_key:
        logger.info("OpenAlex API key loaded — using authenticated rate limits")
    else:
        logger.info(
            "No OpenAlex API key — using polite pool (10 req/s). "
            "Set OPENALEX_API_KEY env var or pass --api-key for higher limits."
        )

    # Check budget before starting
    budget = check_rate_limit(api_key)
    if budget:
        remaining = budget["credits_remaining"]
        limit = budget["credits_limit"]
        reset_h = budget["resets_in_seconds"] / 3600
        logger.info(
            f"OpenAlex budget: {remaining}/{limit} credits remaining "
            f"(resets in {reset_h:.1f}h)"
        )
        if remaining < 100:
            logger.warning(
                f"Very low credit budget ({remaining} credits). "
                f"Consider waiting {reset_h:.1f}h for reset."
            )

    # Determine which files to process
    if args.all_venues or args.title_search:
        ieee_only = False
        legacy_files = sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz"))
        papers_files = sorted(PAPERS_DIR.glob("*.json.gz"))
    else:
        # Specific venue or default IEEE venues
        venues = [args.venue] if args.venue else IEEE_VENUES
        # Use IEEE-only filter only for the built-in IEEE venues list
        ieee_only = not args.venue or args.venue in IEEE_VENUES

        legacy_files = []
        if not args.no_legacy:
            for v in venues:
                p = LEGACY_DIR / f"{v}-legacy.jsonl.gz"
                if p.exists():
                    legacy_files.append(p)
                elif args.venue:
                    # Non-IEEE venue requested — no legacy file is fine
                    logger.debug(f"No legacy file for {v}, skipping")

        papers_files = []
        if not args.no_papers:
            for v in venues:
                for p in sorted(PAPERS_DIR.glob(f"{v}-*.json.gz")):
                    papers_files.append(p)

    # Filter by venue if specified with --title-search
    if args.venue and (args.all_venues or args.title_search):
        legacy_files = [p for p in legacy_files if p.name.startswith(f"{args.venue}-")]
        papers_files = [p for p in papers_files if p.name.startswith(f"{args.venue}-")]

    if args.no_legacy:
        legacy_files = []
    if args.no_papers:
        papers_files = []

    if not legacy_files and not papers_files:
        logger.error("No files found to process")
        sys.exit(1)

    mode_label = "title search" if args.title_search else (
        "all DOIs" if not ieee_only else "IEEE DOIs only"
    )
    logger.info(
        f"Will process {len(legacy_files)} legacy file(s) "
        f"and {len(papers_files)} papers file(s)"
        f" [{mode_label}]"
        + (" [+OA PDF]" if args.oa_pdf else "")
    )

    t_start = time.time()
    total_enriched = 0

    if args.title_search:
        # Title-based search mode
        for path in legacy_files:
            total_enriched += enrich_legacy_title_search(
                path,
                api_key=api_key,
                dry_run=args.dry_run,
                limit=args.limit,
            )

        for path in papers_files:
            total_enriched += enrich_papers_title_search(
                path,
                api_key=api_key,
                dry_run=args.dry_run,
                limit=args.limit,
            )
    else:
        # DOI-based enrichment (original mode)
        for path in legacy_files:
            total_enriched += enrich_legacy_file(
                path,
                api_key=api_key,
                dry_run=args.dry_run,
                ieee_only=ieee_only,
                oa_pdf=args.oa_pdf,
                limit=args.limit,
            )

        for path in papers_files:
            total_enriched += enrich_papers_file(
                path,
                api_key=api_key,
                dry_run=args.dry_run,
                ieee_only=ieee_only,
                oa_pdf=args.oa_pdf,
                limit=args.limit,
            )

    elapsed = time.time() - t_start
    action = "would enrich" if args.dry_run else "enriched"
    logger.info(
        f"\nDone in {elapsed:.0f}s — "
        f"{total_enriched} papers {action} across all venues"
    )


if __name__ == "__main__":
    main()
