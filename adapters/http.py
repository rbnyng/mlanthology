"""Shared HTTP utilities for all adapters.

Provides fetch_with_retry() for exponential backoff and fetch_parallel()
for concurrent item fetching with progress logging.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)


def fetch_with_retry(
    url: str,
    headers: Optional[dict] = None,
    max_retries: int = 3,
    timeout: int = 30,
    rate_limit_codes: tuple[int, ...] = (403, 429),
    return_none_on_404: bool = False,
) -> requests.Response:
    """Fetch URL with exponential backoff.

    Args:
        url: URL to fetch.
        headers: Optional request headers.
        max_retries: Number of retry attempts.
        timeout: Request timeout in seconds.
        rate_limit_codes: HTTP status codes that trigger backoff.
        return_none_on_404: If True, return None instead of raising on 404.

    Returns:
        requests.Response on success.

    Raises:
        FileNotFoundError: On 404 (unless return_none_on_404 is True).
        RuntimeError: After exhausting all retries.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                # requests defaults to ISO-8859-1 without explicit charset;
                # override to UTF-8 to avoid mojibake
                if (
                    resp.encoding
                    and resp.encoding.lower().replace("-", "") == "iso88591"
                    and "charset" not in resp.headers.get("Content-Type", "").lower()
                ):
                    resp.encoding = "utf-8"
                return resp
            if resp.status_code == 404:
                if return_none_on_404:
                    return None
                raise FileNotFoundError(f"Not found: {url}")
            if resp.status_code in rate_limit_codes:
                wait = 2 ** (attempt + 1)
                logger.warning(f"HTTP {resp.status_code} on {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Connection error on {url}, retry in {wait}s")
            time.sleep(wait)

    if return_none_on_404:
        logger.warning(f"Giving up on {url} after {max_retries} retries, skipping")
        return None
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries")


def fetch_parallel(
    keys: list,
    fn: Callable,
    *,
    max_workers: int = 10,
    default: Any = "",
    progress_interval: int = 100,
) -> dict:
    """Call fn(key) in parallel for each key, returning {key: result}.

    Failed calls log a warning and return *default* instead of raising.

    Args:
        keys: Items to process.
        fn: Callable that takes a single key and returns a result.
        max_workers: ThreadPoolExecutor concurrency.
        default: Value to use when fn raises an exception.
        progress_interval: Log progress every N completed items.
    """
    results: dict = {}
    total = len(keys)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(fn, k): k for k in keys}
        done = 0
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            done += 1
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning(f"  Failed to fetch {key}: {e}")
                results[key] = default
            if done % progress_interval == 0:
                logger.info(f"  Progress: {done}/{total}")

    return results
