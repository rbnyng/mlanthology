"""Elsevier ScienceDirect API adapter: fetch abstracts by DOI.

Uses the Elsevier Full-Text Article API to retrieve abstracts for papers
published by Elsevier (primarily ICML 1988-2002 book chapters and some
IJCAI/COLT proceedings with 10.1016 or 10.1006 DOIs).

API details:
- Endpoint: GET https://api.elsevier.com/content/article/doi/{doi}
- Auth:     API key via query param or X-ELS-APIKey header; register at
  https://dev.elsevier.com/
  Pass as env var ELSEVIER_API_KEY or via the api_key argument.
- Rate limit: 10,000 requests/week (institutional key).
- Response: JSON full-text-retrieval-response with abstract in
  coredata['dc:description'].

Only processes papers whose DOI starts with "10.1016/" or "10.1006/"
(Elsevier prefixes).  Also sets venue_url = https://doi.org/{doi} for
any paper that has a DOI but no venue_url.

Usage:
    export ELSEVIER_API_KEY=your_key_here
    python scripts/enrich_elsevier.py --venue icml
    python scripts/enrich_elsevier.py          # all eligible venues
"""

import logging
import time
from typing import Optional

import requests

from .http import fetch_with_retry as _fetch_with_retry

logger = logging.getLogger(__name__)

ELSEVIER_FULLTEXT_API = "https://api.elsevier.com/content/article/doi"

# Elsevier DOI prefixes we know the API covers well
ELSEVIER_DOI_PREFIXES = ("10.1016/", "10.1006/")

# Conservative rate: 10 req/s leaves plenty of headroom under 10k/week
MIN_REQUEST_INTERVAL = 0.1  # seconds between requests


def fetch_by_doi(doi: str, api_key: str) -> Optional[dict]:
    """Fetch abstract for a single DOI from the Elsevier full-text API.

    Returns:
        Dict with "abstract" key (value may be None), or None if not found.
    """
    url = f"{ELSEVIER_FULLTEXT_API}/{requests.utils.quote(doi, safe='')}"
    params = {"apiKey": api_key, "httpAccept": "application/json"}

    resp = _fetch_with_retry(
        url, params=params, max_retries=4, return_none_on_404=True,
        rate_limit_codes=(429, 500, 502, 503, 504),
    )
    if resp is None:
        return None

    try:
        data = resp.json()
    except ValueError:
        return None

    core = data.get("full-text-retrieval-response", {}).get("coredata", {})
    abstract = core.get("dc:description", "") or ""
    abstract = abstract.strip()

    if abstract:
        return {"abstract": abstract}
    return None


def fetch_batch(
    papers: list[dict],
    api_key: str,
) -> dict[str, dict]:
    """Fetch abstracts for papers with Elsevier DOIs that are missing abstracts.

    Args:
        papers:  List of paper dicts with at least "doi" field.
        api_key: Elsevier API key.

    Returns:
        Dict mapping DOI -> {"abstract": str}.
    """
    to_fetch = [
        p["doi"].strip()
        for p in papers
        if p.get("doi", "").strip()
        and any(p["doi"].strip().startswith(pfx) for pfx in ELSEVIER_DOI_PREFIXES)
        and not p.get("abstract", "").strip()
    ]

    if not to_fetch:
        logger.info("  No papers need Elsevier enrichment")
        return {}

    logger.info(f"  Fetching {len(to_fetch)} papers from Elsevier...")

    results: dict[str, dict] = {}
    found = 0
    errors = 0
    t_start = time.time()
    last_request_time = 0.0

    for i, doi in enumerate(to_fetch):
        elapsed = time.time() - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        last_request_time = time.time()
        result = fetch_by_doi(doi, api_key)

        if result is not None:
            results[doi] = result
            found += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0 or i + 1 == len(to_fetch):
            elapsed_total = time.time() - t_start
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            logger.info(
                f"  Elsevier: {i + 1}/{len(to_fetch)} queried, "
                f"{found} found, {errors} errors "
                f"({rate:.1f} req/s)"
            )

    elapsed_total = time.time() - t_start
    logger.info(
        f"  Elsevier done: {found}/{len(to_fetch)} found "
        f"in {elapsed_total:.0f}s ({errors} errors)"
    )
    return results


def enrich_papers(papers: list[dict], api_key: str) -> int:
    """Enrich papers in-place with Elsevier abstract and venue_url from DOI.

    - Sets abstract from the Elsevier full-text API (10.1016/10.1006 DOIs).
    - Sets venue_url = https://doi.org/{doi} for any paper that has a DOI
      but no venue_url, pdf_url, or openreview_url.
    Only fills empty fields — never overwrites existing data.

    Args:
        papers:  List of paper dicts to enrich (modified in-place).
        api_key: Elsevier API key.

    Returns:
        Number of papers that were enriched (had at least one field filled).
    """
    results = fetch_batch(papers, api_key)

    enriched = 0
    for paper in papers:
        doi = paper.get("doi", "").strip()
        if not doi:
            continue

        changed = False

        # Set abstract from API result if available
        if doi in results:
            abstract = results[doi].get("abstract", "")
            if abstract and not paper.get("abstract", "").strip():
                paper["abstract"] = abstract
                changed = True

        # Set venue_url from DOI for papers with no URL at all
        has_url = (
            paper.get("venue_url", "")
            or paper.get("pdf_url", "")
            or paper.get("openreview_url", "")
        )
        if not has_url and not paper.get("venue_url", ""):
            paper["venue_url"] = f"https://doi.org/{doi}"
            changed = True

        if changed:
            src = paper.get("source", "")
            if src == "dblp":
                paper["source"] = "dblp+elsevier"
            elif src and "elsevier" not in src:
                paper["source"] = src + "+elsevier"
            enriched += 1

    logger.info(f"  Enriched {enriched} papers via Elsevier")
    return enriched
