"""Springer Nature Meta API adapter: fetch abstracts by DOI.

Uses the Springer Nature Meta/v2 REST API to look up paper metadata
for LNCS/LNAI book chapters.  Designed to enrich ECML, ALT, and
similar Springer-published venues whose records come from DBLP and
lack abstracts.

API details:
- Endpoint: GET https://api.springernature.com/meta/v2/json
- Auth: api_key query parameter (free key from https://dev.springernature.com/)
  Pass as env var SPRINGER_API_KEY or via the api_key argument.
- Free-tier limits: 500 requests/day, 100 requests/minute.
- Batching: the API supports OR queries, e.g.
    q=doi:10.1007/A OR doi:10.1007/B OR doi:10.1007/C
  With BATCH_SIZE=25 DOIs/request this gives ~12 500 DOIs/day on the free
  tier — enough to cover all ECML + ALT legacy papers in a single run.
- Response: JSON with a "records" array; each record may have an
  "abstract" field.  Records are matched back to DOIs via the
  "identifier" field (formatted as "doi:10.1007/...").

Only processes papers whose DOI starts with "10.1007/" (Springer).

Usage:
    export SPRINGER_API_KEY=your_key_here
    python scripts/enrich_springer.py --venue ecml
    python scripts/enrich_springer.py --venue ecmlpkdd
    python scripts/enrich_springer.py --venue alt
"""

import logging
import time
from typing import Optional

import requests

from .common import get_api_key as _get_api_key_from

logger = logging.getLogger(__name__)

SPRINGER_API = "https://api.springernature.com/meta/v2/json"

_HEADERS = {
    "User-Agent": (
        "mlanthology/1.0 (https://github.com/rbnyng/mlanthology; research use)"
    ),
    "Accept": "application/json",
}

# Free tier: 100 requests/minute → 1 req per 0.65 s leaves comfortable margin.
MIN_REQUEST_INTERVAL = 0.65  # seconds between requests

# number of DOIs to pack into one OR query.
# 25 keeps the query string short (~800 chars) and gives 12 500 DOIs/day on
# the free tier (500 req/day × 25 DOIs/req).
BATCH_SIZE = 25


def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return _get_api_key_from("SPRINGER_API_KEY", api_key)


def _is_springer_doi(doi: str) -> bool:
    """Return True for DOIs published by Springer Nature (10.1007 prefix)."""
    return doi.startswith("10.1007/")


def _parse_doi_from_identifier(identifier: str) -> str:
    """Strip the 'doi:' prefix from a Springer record identifier field."""
    if identifier.startswith("doi:"):
        return identifier[4:]
    return identifier


def _request_with_retry(
    params: dict,
    max_retries: int = 4,
    timeout: int = 90,
) -> Optional[dict]:
    """Send a single GET to the Springer Meta API and return the parsed JSON.

    Retries on 429 and 5xx with exponential back-off.  Returns None on
    permanent errors (401/403/404) or after exhausting retries.

    The default timeout is 90 s: OR queries over large/old LNCS volumes
    (e.g. ALT 1994, ECML 1998 BFb-style DOIs) can take 40-50 s to
    resolve server-side.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                SPRINGER_API, params=params, headers=_HEADERS, timeout=timeout
            )

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    logger.warning("Springer: invalid JSON in response")
                    return None

            if resp.status_code in (401, 403):
                logger.error(
                    f"Springer: auth error HTTP {resp.status_code} — "
                    "check SPRINGER_API_KEY"
                )
                return None  # no point retrying auth failures

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Springer: rate limited (429), waiting {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Springer: HTTP {resp.status_code}, retry in {wait}s"
                )
                time.sleep(wait)
                continue

            logger.warning(f"Springer: unexpected HTTP {resp.status_code}")
            return None

        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Springer: connection error, retry in {wait}s")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Springer: timeout, retry in {wait}s")
            time.sleep(wait)

    logger.warning(f"Springer: giving up after {max_retries} retries")
    return None


def fetch_by_doi(
    doi: str,
    api_key: str,
    *,
    max_retries: int = 4,
) -> Optional[dict]:
    """Fetch Springer metadata for a single DOI.

    Convenience wrapper around the OR-batch machinery for single lookups.

    Args:
        doi: The DOI string (without "doi:" prefix).
        api_key: Springer Nature API key.
        max_retries: Maximum number of retry attempts.

    Returns:
        Dict with keys:
            "abstract"     – abstract text string or None
            "springer_url" – canonical chapter URL on link.springer.com
        Returns None if the DOI was not found or a permanent error occurred.
    """
    results = fetch_dois([doi], api_key, max_retries=max_retries)
    return results.get(doi)


def fetch_dois(
    dois: list[str],
    api_key: str,
    *,
    max_retries: int = 4,
) -> dict[str, dict]:
    """Fetch Springer metadata for a batch of DOIs in a single OR query.

    Sends one API request for up to BATCH_SIZE DOIs using OR syntax:
        q=doi:A OR doi:B OR doi:C

    Records in the response are matched back to input DOIs via their
    "identifier" field.

    Args:
        dois: List of DOI strings (without "doi:" prefix).
        api_key: Springer Nature API key.
        max_retries: Maximum retry attempts per request.

    Returns:
        Dict mapping DOI -> {"abstract": str|None, "springer_url": str}
        for every DOI that returned a record.
    """
    if not dois:
        return {}

    q = " OR ".join(f"doi:{d}" for d in dois)
    params = {
        "q": q,
        "api_key": api_key,
        "p": str(len(dois)),  # page size = batch size so we get all results
    }

    data = _request_with_retry(params, max_retries=max_retries)
    if data is None:
        return {}

    # DOIs are case-insensitive (RFC 2141 / DOI Handbook).  The Springer API
    # normalises old BFb-style identifiers to mixed case (e.g. BFb0026703)
    # while DBLP stores them as all-uppercase (BFB0026703).  Build a
    # lowercase reverse-lookup so matching is case-insensitive.
    lower_to_input = {d.lower(): d for d in dois}

    results: dict[str, dict] = {}
    for record in data.get("records", []):
        raw_id = record.get("identifier", "")
        doi_from_api = _parse_doi_from_identifier(raw_id)
        if not doi_from_api:
            continue

        # Map back to the original casing used in our dataset.
        doi = lower_to_input.get(doi_from_api.lower(), doi_from_api)

        abstract = record.get("abstract", "").strip() or None
        springer_url = f"https://link.springer.com/chapter/{doi}"
        results[doi] = {"abstract": abstract, "springer_url": springer_url}

    return results


def fetch_batch(
    papers: list[dict],
    *,
    api_key: Optional[str] = None,
    needs_abstract: bool = True,
    batch_size: int = BATCH_SIZE,
) -> dict[str, dict]:
    """Fetch Springer metadata for multiple papers using batched OR queries.

    Filters to papers that (a) have a DOI with the 10.1007/ prefix and
    (b) are missing the requested fields.  Sends ceil(N / batch_size)
    API requests total instead of one per paper.

    Args:
        papers: List of paper dicts with at least a "doi" field.
        api_key: Springer Nature API key (falls back to SPRINGER_API_KEY).
        needs_abstract: Include papers missing abstracts.
        batch_size: Number of DOIs to pack into each OR query (default 25).

    Returns:
        Dict mapping DOI -> {"abstract": str|None, "springer_url": str}.
        Returns an empty dict if no API key is available.
    """
    key = _get_api_key(api_key)
    if not key:
        logger.warning(
            "Springer: no API key provided — set the SPRINGER_API_KEY "
            "environment variable or pass api_key= explicitly.  "
            "Register for a free key at https://dev.springernature.com/"
        )
        return {}

    # Select papers that need enrichment
    to_fetch: list[str] = []
    for p in papers:
        doi = p.get("doi", "").strip()
        if not doi or not _is_springer_doi(doi):
            continue
        if needs_abstract and not p.get("abstract", "").strip():
            to_fetch.append(doi)

    if not to_fetch:
        logger.info("  Springer: no papers need enrichment")
        return {}

    # Build batches
    batches = [
        to_fetch[i : i + batch_size] for i in range(0, len(to_fetch), batch_size)
    ]
    logger.info(
        f"  Springer: {len(to_fetch)} papers in {len(batches)} batches "
        f"(batch_size={batch_size})"
    )

    results: dict[str, dict] = {}
    abstracts_found = 0
    errors = 0
    t_start = time.time()
    last_request_time = 0.0

    for i, batch in enumerate(batches):
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        last_request_time = time.time()
        batch_results = fetch_dois(batch, key)

        if not batch_results and len(batch) > 0:
            errors += len(batch)
        else:
            for doi, rec in batch_results.items():
                results[doi] = rec
                if rec.get("abstract"):
                    abstracts_found += 1
            # DOIs in the batch with no record are "not found" (not errors)
            not_found = len(batch) - len(batch_results)
            if not_found:
                logger.debug(
                    f"  Springer: {not_found} DOIs returned no record in batch {i+1}"
                )

        # Abort early if the API key appears invalid
        if i == 0 and not batch_results and len(batch) >= min(5, len(to_fetch)):
            logger.error(
                "  Springer: first batch returned nothing — aborting "
                "(check API key or network)"
            )
            break

        # Progress logging every 10 batches or on the last one
        if (i + 1) % 10 == 0 or i + 1 == len(batches):
            elapsed_total = time.time() - t_start
            dois_done = min((i + 1) * batch_size, len(to_fetch))
            rate = dois_done / elapsed_total if elapsed_total > 0 else 0
            logger.info(
                f"  Springer: batch {i+1}/{len(batches)} done — "
                f"{len(results)} records retrieved, "
                f"{abstracts_found} with abstracts "
                f"({rate:.1f} DOIs/s)"
            )

    elapsed_total = time.time() - t_start
    logger.info(
        f"  Springer done: {len(results)}/{len(to_fetch)} records found, "
        f"{abstracts_found} abstracts, in {elapsed_total:.0f}s"
    )
    return results


def enrich_papers(papers: list[dict], *, api_key: Optional[str] = None) -> int:
    """Enrich papers in-place with Springer abstract data.

    Only fills empty fields — never overwrites existing data.
    Only processes papers whose DOI has the 10.1007/ prefix.

    Adds the springer_url as venue_url when the field is currently empty,
    giving users a direct link to the chapter landing page on SpringerLink.

    Args:
        papers: List of paper dicts to enrich (modified in-place).
        api_key: Springer Nature API key (falls back to SPRINGER_API_KEY).

    Returns:
        Number of papers that had at least one field filled in.
    """
    results = fetch_batch(papers, api_key=api_key)

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

        # Set venue_url to the Springer chapter page if not already set.
        if result.get("springer_url") and not paper.get("venue_url", "").strip():
            paper["venue_url"] = result["springer_url"]
            changed = True

        if changed:
            src = paper.get("source", "")
            if src == "dblp":
                paper["source"] = "dblp+springer"
            elif src and "springer" not in src:
                paper["source"] = f"{src}+springer"
            enriched += 1

    logger.info(f"  Springer: enriched {enriched} papers")
    return enriched
