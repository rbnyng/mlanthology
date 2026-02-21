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

logger = logging.getLogger(__name__)

CROSSREF_API = "https://api.crossref.org/works"

_HEADERS = {
    "User-Agent": "mlanthology/1.0 (https://github.com/rbnyng/mlanthology; mailto:crossref@mlanthology.org)",
    "Accept": "application/json",
}

# Polite request interval — Crossref polite pool is generous but we
# stay conservative to avoid hitting limits during large runs.
MIN_REQUEST_INTERVAL = 0.1  # 10 req/s baseline


def _strip_jats(text: str) -> str:
    """Strip JATS XML tags from a Crossref abstract, returning plain text."""
    text = re.sub(r"</?jats:[^>]+>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _get_with_retry(
    url: str,
    headers: Optional[dict] = None,
    max_retries: int = 4,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """GET with exponential backoff.  Returns None on unrecoverable error."""
    hdrs = dict(headers or _HEADERS)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=hdrs, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Crossref 429 rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                logger.warning(f"Crossref HTTP {resp.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            logger.warning(f"Crossref HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Crossref connection error, retry in {wait}s")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Crossref timeout, retry in {wait}s")
            time.sleep(wait)

    logger.warning(f"Crossref giving up after {max_retries} retries")
    return None


def fetch_by_doi(doi: str) -> Optional[dict]:
    """Fetch metadata for a single DOI from Crossref.

    Returns:
        Dict with "abstract" and "pdf_url" keys (values may be None),
        or None if the DOI was not found.
    """
    url = f"{CROSSREF_API}/{requests.utils.quote(doi, safe='')}"
    resp = _get_with_retry(url)
    if resp is None:
        return None

    try:
        msg = resp.json().get("message", {})
    except ValueError:
        return None

    # Abstract (may contain JATS XML tags)
    raw_abstract = msg.get("abstract", "")
    abstract = _strip_jats(raw_abstract) if raw_abstract else None

    # PDF link — look for content-type application/pdf in the link array
    pdf_url = None
    for link in msg.get("link", []):
        if "pdf" in link.get("content-type", "").lower():
            pdf_url = link["URL"]
            break

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
