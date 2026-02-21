"""OpenAlex enrichment: fetch abstracts and OA PDF URLs by DOI via batch API.

OpenAlex provides freely-accessible paper metadata including abstracts
(stored as inverted index) and open-access PDF URLs for a wide range of
venues.  Their /works endpoint supports filtering by up to ~100 DOIs per
request.

API details:
- Endpoint: GET https://api.openalex.org/works
- Filter:   doi:https://doi.org/A|https://doi.org/B|...
- No authentication required; include mailto for polite pool.
- Rate limit: 10 req/s (polite pool); back off on 429.
- API key: pass via api_key= param or OPENALEX_API_KEY env var for
  higher rate limits.  Register at https://openalex.org/

Usage::

    from adapters.openalex import fetch_abstracts_by_doi, fetch_metadata_by_doi

    # Abstracts only (backward-compatible):
    abstracts = fetch_abstracts_by_doi(["10.1145/...", "10.1145/..."])
    # Returns {doi: abstract_text, ...}

    # Full metadata (abstract + OA PDF URL):
    meta = fetch_metadata_by_doi(["10.1145/...", "10.1145/..."])
    # Returns {doi: {"abstract": str, "oa_url": str}, ...}
    # Keys are only present when non-empty.
"""

import logging
import time
from typing import Optional

import requests

from .common import get_api_key as _get_api_key_from
from .http import fetch_with_retry as _fetch_with_retry

logger = logging.getLogger(__name__)

_OPENALEX_WORKS = "https://api.openalex.org/works"
_MAILTO = "mailto:openalex@mlanthology.org"

# Maximum DOIs per batch request (OpenAlex recommends ≤100).
BATCH_SIZE = 50

# seconds to wait between batches (polite-pool ceiling is  10 req/s).
BATCH_DELAY = 1.0
# Reduced delay when using an API key (higher rate limits).
BATCH_DELAY_AUTHENTICATED = 0.1


def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return _get_api_key_from("OPENALEX_API_KEY", api_key)


def _decode_inverted_index(aii: Optional[dict]) -> str:
    """Reconstruct abstract text from an OpenAlex abstract_inverted_index."""
    if not aii:
        return ""
    pos_word: dict[int, str] = {}
    for word, positions in aii.items():
        for pos in positions:
            pos_word[pos] = word
    return " ".join(pos_word[i] for i in sorted(pos_word))


def _fetch_batch(
    dois: list[str],
    *,
    api_key: Optional[str] = None,
    max_retries: int = 4,
) -> dict[str, dict]:
    """Fetch metadata for up to BATCH_SIZE DOIs from OpenAlex.

    Returns a dict mapping doi → {"abstract": str, "oa_url": str}.
    Keys are only present when non-empty.  DOIs not found in OpenAlex
    are omitted entirely.
    """
    filter_str = "|".join(f"https://doi.org/{d}" for d in dois)
    params: dict[str, str] = {
        "filter": f"doi:{filter_str}",
        "per-page": "200",
        "select": "doi,abstract_inverted_index,open_access",
        "mailto": _MAILTO,
    }
    if api_key:
        params["api_key"] = api_key

    resp = _fetch_with_retry(
        _OPENALEX_WORKS, params=params, max_retries=max_retries,
        return_none_on_404=True, rate_limit_codes=(429, 500, 502, 503, 504),
    )
    if resp is None:
        return {}

    result: dict[str, dict] = {}
    for item in resp.json().get("results", []):
        raw_doi = item.get("doi", "")
        doi = raw_doi.replace("https://doi.org/", "").strip()
        if not doi:
            continue
        meta: dict = {}
        abstract = _decode_inverted_index(item.get("abstract_inverted_index"))
        if abstract:
            meta["abstract"] = abstract
        oa = item.get("open_access") or {}
        oa_url = oa.get("oa_url") or ""
        if oa_url:
            meta["oa_url"] = oa_url
        if meta:
            result[doi] = meta
    return result


def fetch_metadata_by_doi(
    dois: list[str],
    *,
    api_key: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    batch_delay: Optional[float] = None,
) -> dict[str, dict]:
    """Fetch abstract and OA PDF URL for a list of DOIs from OpenAlex, in batches.

    Args:
        dois: List of DOI strings (without ``https://doi.org/`` prefix).
        api_key: Optional OpenAlex API key for higher rate limits.
            Falls back to OPENALEX_API_KEY env var.
        batch_size: Number of DOIs per API request (default 50).
        batch_delay: Seconds to sleep between batches.  Defaults to
            0.1 s when an API key is supplied, 1.0 s otherwise.

    Returns:
        Dict mapping doi → {"abstract": str, "oa_url": str}.
        Only keys with non-empty values are included in each inner dict.
    """
    if not dois:
        return {}

    key = _get_api_key(api_key)
    if batch_delay is None:
        batch_delay = BATCH_DELAY_AUTHENTICATED if key else BATCH_DELAY

    result: dict[str, dict] = {}
    total_batches = (len(dois) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(dois), batch_size), start=1):
        batch = dois[i : i + batch_size]
        logger.debug(f"  OpenAlex batch {batch_num}/{total_batches} ({len(batch)} DOIs)")
        result.update(_fetch_batch(batch, api_key=key))
        if i + batch_size < len(dois):
            time.sleep(batch_delay)

    return result


def fetch_abstracts_by_doi(
    dois: list[str],
    *,
    api_key: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    batch_delay: Optional[float] = None,
) -> dict[str, str]:
    """Fetch abstracts for a list of DOIs from OpenAlex, in batches.

    Thin wrapper around :func:`fetch_metadata_by_doi` that returns only
    the abstract strings, for backward compatibility.

    Returns:
        Dict mapping doi → abstract_text for every DOI that had a
        non-empty abstract in OpenAlex.
    """
    meta = fetch_metadata_by_doi(
        dois, api_key=api_key, batch_size=batch_size, batch_delay=batch_delay
    )
    return {doi: m["abstract"] for doi, m in meta.items() if "abstract" in m}
