"""Semantic Scholar enrichment: batch-fetch abstracts for papers.

Uses the Semantic Scholar Academic Graph API batch endpoint to look up
papers by DOI and retrieve abstracts.  This adapter is not a standalone
fetcher — it enriches paper dicts produced by other adapters (primarily
DBLP) that lack abstracts.

API details:
- Endpoint: POST https://api.semanticscholar.org/graph/v1/paper/batch
- Max 500 paper IDs per request, 10 MB response limit
- Rate limit: 5000 requests/5 min (unauthenticated), 1 RPS (with key)
- Paper IDs can be DOIs prefixed with "DOI:", or title-based search
"""

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_HEADERS = {
    "User-Agent": "mlanthology/1.0 (https://github.com/rbnyng/mlanthology; research use)",
    "Content-Type": "application/json",
}

# Batch size for the S2 batch endpoint (max 500 per API docs).
BATCH_SIZE = 500

# Minimum interval between requests (seconds).  With an API key the
# rate limit is 1 RPS; we use 1.5s to avoid 429 thrashing from timing
# jitter (the cost of a 429 retry is ≥2s, so being slightly conservative
# is faster overall).
MIN_REQUEST_INTERVAL = 1.5


def _post_with_retry(
    url: str,
    json_body: dict,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    api_key: Optional[str] = None,
    max_retries: int = 4,
    timeout: int = 60,
) -> Optional[requests.Response]:
    """POST with exponential backoff.  Returns None on unrecoverable error."""
    hdrs = dict(headers or _HEADERS)
    if api_key:
        hdrs["x-api-key"] = api_key

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, json=json_body, params=params, headers=hdrs, timeout=timeout
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"S2 429 rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                logger.warning(f"S2 HTTP {resp.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            # 400 = bad request (e.g. too many IDs) — not retryable
            logger.warning(f"S2 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            logger.warning(f"S2 connection error, retry in {wait}s")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            logger.warning(f"S2 timeout, retry in {wait}s")
            time.sleep(wait)

    logger.warning(f"S2 giving up after {max_retries} retries")
    return None


def _get_with_retry(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    api_key: Optional[str] = None,
    max_retries: int = 4,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """GET with exponential backoff."""
    hdrs = dict(headers or _HEADERS)
    if api_key:
        hdrs["x-api-key"] = api_key

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"S2 429 rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                logger.warning(f"S2 HTTP {resp.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wait = 2 ** (attempt + 1)
            logger.warning(f"S2 connection/timeout error, retry in {wait}s")
            time.sleep(wait)

    return None


def fetch_abstracts_batch(
    papers: list[dict],
    api_key: Optional[str] = None,
) -> dict[str, dict]:
    """Fetch enrichment data for papers using DOI batch lookup.

    Args:
        papers: List of paper dicts with at least "doi" and "title" fields.
        api_key: Optional Semantic Scholar API key for higher rate limits.

    Returns:
        Dict mapping DOI -> {"abstract": str|None, "pdf_url": str|None}.
        Only papers where at least one field was found are included.
    """
    # Collect papers with DOIs
    doi_papers = [(p["doi"], p) for p in papers if p.get("doi")]
    if not doi_papers:
        logger.info("  No papers with DOIs to look up")
        return {}

    logger.info(f"  Looking up {len(doi_papers)} papers by DOI via Semantic Scholar...")

    results: dict[str, dict] = {}

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, len(doi_papers), BATCH_SIZE):
        batch = doi_papers[batch_start:batch_start + BATCH_SIZE]
        ids = [f"DOI:{doi}" for doi, _ in batch]

        resp = _post_with_retry(
            S2_BATCH_URL,
            json_body={"ids": ids},
            params={"fields": "abstract,openAccessPdf"},
            api_key=api_key,
        )

        if resp is None:
            logger.warning(
                f"  Batch {batch_start // BATCH_SIZE + 1} failed, skipping "
                f"{len(batch)} papers"
            )
            time.sleep(MIN_REQUEST_INTERVAL)
            continue

        try:
            data = resp.json()
        except ValueError:
            logger.warning("  Non-JSON response from S2 batch endpoint")
            time.sleep(MIN_REQUEST_INTERVAL)
            continue

        # Response is a list aligned with input IDs (None for not-found)
        found = 0
        for i, entry in enumerate(data):
            if entry is None:
                continue
            abstract = (entry.get("abstract") or "").strip() or None
            oa_pdf = entry.get("openAccessPdf")
            pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
            if abstract or pdf_url:
                doi = batch[i][0]
                results[doi] = {"abstract": abstract, "pdf_url": pdf_url}
                found += 1

        logger.info(
            f"  Batch {batch_start // BATCH_SIZE + 1}/"
            f"{(len(doi_papers) + BATCH_SIZE - 1) // BATCH_SIZE}: "
            f"{found}/{len(batch)} papers matched"
        )
        time.sleep(MIN_REQUEST_INTERVAL)

    logger.info(f"  Semantic Scholar DOI batch: {len(results)}/{len(doi_papers)} matched")
    return results


def fetch_abstracts_by_title(
    papers: list[dict],
    api_key: Optional[str] = None,
) -> dict[int, dict]:
    """Fallback: fetch enrichment data by title search for papers without DOIs.

    This is much slower (one request per paper) and less reliable than
    DOI batch lookup.  Use only for papers that have no DOI.

    Args:
        papers: List of paper dicts (must have "title" field).
        api_key: Optional S2 API key.

    Returns:
        Dict mapping paper list index -> {"abstract": str|None,
        "doi": str|None, "pdf_url": str|None}.
    """
    results: dict[int, dict] = {}
    searched = 0

    for i, paper in enumerate(papers):
        title = paper.get("title", "").strip()
        if not title:
            continue

        resp = _get_with_retry(
            S2_SEARCH_URL,
            params={
                "query": title,
                "fields": "title,abstract,externalIds,openAccessPdf",
                "limit": "1",
            },
            api_key=api_key,
        )
        searched += 1

        if resp is None:
            time.sleep(MIN_REQUEST_INTERVAL)
            continue

        try:
            data = resp.json()
        except ValueError:
            time.sleep(MIN_REQUEST_INTERVAL)
            continue

        hits = data.get("data", [])
        if not hits:
            time.sleep(MIN_REQUEST_INTERVAL)
            continue

        # Basic title match check (case-insensitive, strip punctuation)
        hit_title = hits[0].get("title", "")
        if _normalize_title(hit_title) == _normalize_title(title):
            hit = hits[0]
            abstract = (hit.get("abstract") or "").strip() or None
            ext_ids = hit.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            oa_pdf = hit.get("openAccessPdf")
            pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
            if abstract or doi or pdf_url:
                results[i] = {"abstract": abstract, "doi": doi, "pdf_url": pdf_url}

        time.sleep(MIN_REQUEST_INTERVAL)

        if searched % 50 == 0:
            logger.info(f"  Title search: {searched}/{len(papers)} searched, {len(results)} matched")

    logger.info(f"  Title search: {len(results)}/{searched} matched")
    return results


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching."""
    import re
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


def enrich_papers(
    papers: list[dict],
    api_key: Optional[str] = None,
    title_fallback: bool = False,
) -> list[dict]:
    """Enrich a list of papers with abstracts from Semantic Scholar.

    Modifies papers in-place and returns the same list.

    Args:
        papers: List of paper dicts to enrich.
        api_key: Optional S2 API key.
        title_fallback: If True, also try title-based search for papers
            without DOIs (much slower).

    Returns:
        The same list of papers, with abstracts filled in where found.
    """
    # Phase 1: batch DOI lookup
    batch_results = fetch_abstracts_batch(papers, api_key=api_key)

    enriched = 0
    for paper in papers:
        doi = paper.get("doi", "")
        if doi and doi in batch_results:
            result = batch_results[doi]
            changed = False
            if result.get("abstract") and not paper.get("abstract"):
                paper["abstract"] = result["abstract"]
                changed = True
            if result.get("pdf_url") and not paper.get("pdf_url"):
                paper["pdf_url"] = result["pdf_url"]
                changed = True
            if changed:
                enriched += 1

    logger.info(f"  Enriched {enriched} papers via DOI batch lookup")

    # Phase 2: title fallback for papers without DOIs
    if title_fallback:
        needs_search = [
            (i, p) for i, p in enumerate(papers)
            if not p.get("doi")
        ]
        if needs_search:
            logger.info(f"  {len(needs_search)} papers without DOI, trying title search...")
            indices = [i for i, _ in needs_search]
            subset = [p for _, p in needs_search]
            title_results = fetch_abstracts_by_title(subset, api_key=api_key)

            title_enriched = 0
            for subset_idx, result in title_results.items():
                paper_idx = indices[subset_idx]
                paper = papers[paper_idx]
                changed = False
                if result.get("abstract") and not paper.get("abstract"):
                    paper["abstract"] = result["abstract"]
                    changed = True
                if result.get("doi") and not paper.get("doi"):
                    paper["doi"] = result["doi"]
                    changed = True
                if result.get("pdf_url") and not paper.get("pdf_url"):
                    paper["pdf_url"] = result["pdf_url"]
                    changed = True
                if changed:
                    title_enriched += 1

            logger.info(f"  Enriched {title_enriched} more papers via title search")
            enriched += title_enriched

    total_abs = sum(1 for p in papers if p.get("abstract"))
    total_doi = sum(1 for p in papers if p.get("doi"))
    total_pdf = sum(1 for p in papers if p.get("pdf_url"))
    logger.info(
        f"  Final: {total_abs}/{len(papers)} abstracts, "
        f"{total_doi}/{len(papers)} DOIs, "
        f"{total_pdf}/{len(papers)} PDFs "
        f"({enriched} papers enriched)"
    )

    return papers
