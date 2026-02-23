"""Crossref enrichment: fetch abstracts and PDF URLs by DOI.

Uses the Crossref REST API to look up papers by DOI and retrieve
abstracts and full-text PDF links.  Designed as a complement to the
Semantic Scholar adapter — Crossref has excellent coverage of AAAI
papers (10.1609 prefix) that S2 may lack.

API details:
- Endpoint: GET https://api.crossref.org/works/{doi}
- No authentication required; "polite pool" gives higher rate limits
  when a mailto: is included in the User-Agent header.
- Rate limit: ~50 req/s in polite pool (with mailto)
- Returns metadata including abstract (in JATS XML) and PDF links
"""

import logging
import re
import time
from typing import Optional

import requests

from .http import fetch_with_retry as _fetch_with_retry

logger = logging.getLogger(__name__)

CROSSREF_API = "https://api.crossref.org/works"

_HEADERS = {
    "User-Agent": "mlanthology/1.0 (https://github.com/rbnyng/mlanthology; mailto:crossref@mlanthology.org)",
    "Accept": "application/json",
}

# polite request interval — Crossref polite pool is generous but we
# stay conservative to avoid hitting limits during large runs.
MIN_REQUEST_INTERVAL = 0.1  # 10 req/s baseline


def _strip_jats(text: str) -> str:
    """Strip JATS XML tags from a Crossref abstract, returning plain text."""
    text = re.sub(r"</?jats:[^>]+>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def fetch_by_doi(doi: str) -> Optional[dict]:
    """Fetch metadata for a single DOI from Crossref.

    Returns:
        Dict with "abstract" and "pdf_url" keys (values may be None),
        or None if the DOI was not found.
    """
    url = f"{CROSSREF_API}/{requests.utils.quote(doi, safe='')}"
    resp = _fetch_with_retry(
        url, headers=_HEADERS, max_retries=4, return_none_on_404=True,
        rate_limit_codes=(429, 500, 502, 503, 504),
    )
    if resp is None:
        return None

    try:
        msg = resp.json().get("message", {})
    except ValueError:
        return None

    # Abstract (may contain JATS XML tags)
    raw_abstract = msg.get("abstract", "")
    abstract = _strip_jats(raw_abstract) if raw_abstract else None

    # PDF link — prefer explicit application/pdf content-type, but also
    # match links whose URL path contains "/pdf" (e.g. ACM DL returns
    # content-type "unspecified" with URL https://dl.acm.org/doi/pdf/...)
    pdf_url = None
    fallback_url = None
    for link in msg.get("link", []):
        url = link.get("URL", "")
        content_type = link.get("content-type", "").lower()
        if "pdf" in content_type:
            pdf_url = url
            break
        if fallback_url is None and "/pdf" in url:
            fallback_url = url
    if pdf_url is None:
        pdf_url = fallback_url

    if abstract or pdf_url:
        return {"abstract": abstract, "pdf_url": pdf_url}
    return None


def fetch_batch(
    papers: list[dict],
    *,
    needs_abstract: bool = True,
    needs_pdf: bool = True,
) -> dict[str, dict]:
    """Fetch enrichment data for multiple papers by DOI.

    Only queries Crossref for papers that have a DOI and are missing
    the requested fields (abstract and/or pdf_url).

    Args:
        papers: List of paper dicts with at least "doi" field.
        needs_abstract: Include papers missing abstracts.
        needs_pdf: Include papers missing pdf_url.

    Returns:
        Dict mapping DOI -> {"abstract": str|None, "pdf_url": str|None}.
    """
    # Select papers that need enrichment
    to_fetch = []
    for p in papers:
        doi = p.get("doi", "").strip()
        if not doi:
            continue
        missing_abs = needs_abstract and not p.get("abstract", "").strip()
        missing_pdf = needs_pdf and not p.get("pdf_url", "").strip()
        if missing_abs or missing_pdf:
            to_fetch.append(doi)

    if not to_fetch:
        logger.info("  No papers need Crossref enrichment")
        return {}

    logger.info(f"  Fetching {len(to_fetch)} papers from Crossref...")

    results: dict[str, dict] = {}
    found = 0
    errors = 0
    t_start = time.time()
    last_request_time = 0.0

    for i, doi in enumerate(to_fetch):
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        last_request_time = time.time()
        result = fetch_by_doi(doi)

        if result is not None:
            results[doi] = result
            found += 1
        else:
            errors += 1

        # Progress logging
        if (i + 1) % 200 == 0 or i + 1 == len(to_fetch):
            elapsed_total = time.time() - t_start
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            logger.info(
                f"  Crossref: {i + 1}/{len(to_fetch)} queried, "
                f"{found} found, {errors} errors "
                f"({rate:.1f} req/s)"
            )

    elapsed_total = time.time() - t_start
    logger.info(
        f"  Crossref done: {found}/{len(to_fetch)} found "
        f"in {elapsed_total:.0f}s ({errors} errors)"
    )
    return results


def enrich_papers(papers: list[dict]) -> int:
    """Enrich papers in-place with Crossref metadata.

    Only fills empty fields — never overwrites existing data.

    Args:
        papers: List of paper dicts to enrich (modified in-place).

    Returns:
        Number of papers that were enriched (had at least one field filled).
    """
    results = fetch_batch(papers)

    enriched = 0
    for paper in papers:
        doi = paper.get("doi", "").strip()
        if not doi or doi not in results:
            continue

        result = results[doi]
        changed = False

        if result.get("abstract") and not paper.get("abstract", "").strip():
            paper["abstract"] = result["abstract"]
            changed = True
        if result.get("pdf_url") and not paper.get("pdf_url", "").strip():
            paper["pdf_url"] = result["pdf_url"]
            changed = True

        if changed:
            # Update source tag
            if paper.get("source") == "dblp":
                paper["source"] = "dblp+crossref"
            elif paper.get("source") == "dblp+s2":
                paper["source"] = "dblp+s2+crossref"
            enriched += 1

    logger.info(f"  Enriched {enriched} papers via Crossref")
    return enriched
