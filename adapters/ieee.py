"""IEEE Xplore Metadata API adapter: fetch abstracts by DOI.

Uses the IEEE Xplore REST API to look up paper metadata for conference
papers published by IEEE (CVPR, ICCV, WACV, and their workshops).

API details:
- Endpoint: GET https://ieeexploreapi.ieee.org/api/v1/search/articles
- Query:    ?doi={doi}&apikey={key}
- Auth:     API key via query param; register at https://developer.ieee.org/
  Pass as env var IEEE_API_KEY or via the api_key argument.
- Free-tier limits: 200 requests/day, 10 requests/second.
- No batch endpoint: one DOI per request.
- Response: JSON with an "articles" array; each article may have an
  "abstract" field.

Note on scale: at 200 req/day the ~12 000 legacy IEEE papers across
CVPR, ICCV, and WACV will take ~60 days of daily runs to fully enrich.
The script is idempotent — each run skips papers that already have
abstracts and processes venues smallest-first so coverage grows steadily.

Only processes papers whose DOI starts with "10.1109/" (IEEE).

Usage:
    export IEEE_API_KEY=your_key_here
    python scripts/enrich_ieee.py --venue wacv
    python scripts/enrich_ieee.py          # all IEEE venues, quota-aware
"""

import logging
import time
from typing import Optional

import requests

from .common import get_api_key as _get_api_key_from

logger = logging.getLogger(__name__)

IEEE_API = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

_HEADERS = {
    "User-Agent": (
        "mlanthology/1.0 (https://github.com/rbnyng/mlanthology; research use)"
    ),
    "Accept": "application/json",
}

# free tier: 10 req/s → 0.15 s minimum, use 0.15 s (leaves margin).
# daily cap enforced by the caller.
MIN_REQUEST_INTERVAL = 0.15  # seconds between requests

# IEEE venues — sorted smallest-missing-abstract first to maximise
# coverage with the tight 200 req/day free-tier quota.
IEEE_VENUES = ["wacvw", "wacv", "iccvw", "cvprw", "iccv", "cvpr"]


def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return _get_api_key_from("IEEE_API_KEY", api_key)


def _is_ieee_doi(doi: str) -> bool:
    """Return True for DOIs published by IEEE (10.1109 prefix)."""
    return doi.startswith("10.1109/")


def fetch_by_doi(
    doi: str,
    api_key: str,
    *,
    max_retries: int = 4,
) -> Optional[dict]:
    """Fetch IEEE Xplore metadata for a single DOI.

    Args:
        doi: The DOI string (without any prefix).
        api_key: IEEE Xplore API key.
        max_retries: Maximum number of retry attempts on transient errors.

    Returns:
        Dict with keys:
            "abstract"   – abstract text string or None
            "ieee_url"   – canonical article URL on ieeexplore.ieee.org
        Returns None if the DOI was not found or a permanent error occurred.
    """
    params = {
        "doi": doi,
        "apikey": api_key,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                IEEE_API, params=params, headers=_HEADERS, timeout=30
            )

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    logger.warning(f"IEEE: invalid JSON for doi:{doi}")
                    return None

                articles = data.get("articles", [])
                if not articles:
                    return None

                article = articles[0]
                abstract = (article.get("abstract") or "").strip() or None

                # Prefer the API-provided URL; fall back to constructing one.
                ieee_url = (
                    article.get("html_url")
                    or article.get("abstract_url")
                    or f"https://ieeexplore.ieee.org/document/{article.get('article_number', '')}"
                )

                return {"abstract": abstract, "ieee_url": ieee_url}

            if resp.status_code in (401, 403):
                logger.error(
                    f"IEEE: auth error HTTP {resp.status_code} — "
                    "check IEEE_API_KEY (key may still be pending activation)"
                )
                return None  # no point retrying auth failures

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"IEEE: rate limited (429), waiting {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"IEEE: HTTP {resp.status_code}, retry in {wait}s"
                )
                time.sleep(wait)
                continue

            logger.warning(
                f"IEEE: unexpected HTTP {resp.status_code} for doi:{doi}"
            )
            return None

        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            logger.warning(f"IEEE: connection error, retry in {wait}s")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            logger.warning(f"IEEE: timeout, retry in {wait}s")
            time.sleep(wait)

    logger.warning(f"IEEE: giving up after {max_retries} retries for doi:{doi}")
    return None


def fetch_batch(
    papers: list[dict],
    *,
    api_key: Optional[str] = None,
    needs_abstract: bool = True,
    daily_limit: int = 200,
) -> dict[str, dict]:
    """Fetch IEEE Xplore metadata for multiple papers (one request per DOI).

    Filters to papers that (a) have a DOI with the 10.1109/ prefix and
    (b) are missing the requested fields.  Stops after `daily_limit`
    successful requests to avoid exceeding the free-tier quota.

    Args:
        papers: List of paper dicts with at least a "doi" field.
        api_key: IEEE Xplore API key (falls back to IEEE_API_KEY env var).
        needs_abstract: Only process papers missing abstracts.
        daily_limit: Maximum number of API calls to make (default 200).

    Returns:
        Dict mapping DOI -> {"abstract": str|None, "ieee_url": str}.
        Returns an empty dict if no API key is available.
    """
    key = _get_api_key(api_key)
    if not key:
        logger.warning(
            "IEEE: no API key provided — set the IEEE_API_KEY environment "
            "variable or pass api_key= explicitly.  "
            "Register at https://developer.ieee.org/"
        )
        return {}

    # Select papers that need enrichment
    to_fetch: list[str] = []
    for p in papers:
        doi = p.get("doi", "").strip()
        if not doi or not _is_ieee_doi(doi):
            continue
        if needs_abstract and not p.get("abstract", "").strip():
            to_fetch.append(doi)

    if not to_fetch:
        logger.info("  IEEE: no papers need enrichment")
        return {}

    capped = len(to_fetch) > daily_limit
    effective = min(len(to_fetch), daily_limit)
    logger.info(
        f"  IEEE: {len(to_fetch)} candidates, "
        f"processing {effective} (daily_limit={daily_limit})"
        + (" — will need more runs to cover all" if capped else "")
    )

    results: dict[str, dict] = {}
    found = 0
    errors = 0
    t_start = time.time()
    last_request_time = 0.0

    for i, doi in enumerate(to_fetch[:daily_limit]):
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        last_request_time = time.time()
        result = fetch_by_doi(doi, key)

        if result is not None:
            results[doi] = result
            if result.get("abstract"):
                found += 1
        else:
            errors += 1

        # Abort early if the API key appears invalid (many errors, zero hits)
        if i == 9 and found == 0 and errors >= 8:
            logger.error(
                "  IEEE: aborting — too many early errors "
                "(check API key; it may still be pending activation)"
            )
            break

        # Progress logging
        if (i + 1) % 50 == 0 or i + 1 == effective:
            elapsed_total = time.time() - t_start
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            logger.info(
                f"  IEEE: {i+1}/{effective} queried, "
                f"{found} with abstracts, {errors} errors "
                f"({rate:.1f} req/s)"
            )

    elapsed_total = time.time() - t_start
    remaining = len(to_fetch) - min(len(to_fetch), daily_limit)
    logger.info(
        f"  IEEE done: {found} abstracts from {len(results)} records "
        f"in {elapsed_total:.0f}s"
        + (f" — {remaining} papers still need enrichment (run again tomorrow)" if remaining else "")
    )
    return results


def enrich_papers(
    papers: list[dict],
    *,
    api_key: Optional[str] = None,
    daily_limit: int = 200,
) -> int:
    """Enrich papers in-place with IEEE Xplore abstract data.

    Only fills empty fields — never overwrites existing data.
    Only processes papers whose DOI has the 10.1109/ prefix.

    Args:
        papers: List of paper dicts to enrich (modified in-place).
        api_key: IEEE Xplore API key (falls back to IEEE_API_KEY env var).
        daily_limit: Maximum API calls to make this run.

    Returns:
        Number of papers that had at least one field filled in.
    """
    results = fetch_batch(papers, api_key=api_key, daily_limit=daily_limit)

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

        # Set venue_url to the IEEE article page if not already set.
        if result.get("ieee_url") and not paper.get("venue_url", "").strip():
            paper["venue_url"] = result["ieee_url"]
            changed = True

        if changed:
            src = paper.get("source", "")
            if src and "ieee" not in src:
                paper["source"] = f"{src}+ieee"
            elif not src:
                paper["source"] = "ieee"
            enriched += 1

    logger.info(f"  IEEE: enriched {enriched} papers")
    return enriched
