"""Fetch cache: tracks which volumes have already been fetched.

Stores a JSON manifest at data/.fetch_cache.json with a set of cache keys.
Historical volumes (year < current year) are skipped when cached.
Current-year volumes are always re-fetched.
Use --force to bypass all caching.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = "data/.fetch_cache.json"


def load_cache(root: Path) -> dict:
    """Load cache manifest. Returns dict with 'fetched' set."""
    cache_path = root / DEFAULT_CACHE_PATH
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return {"fetched": set(data.get("fetched", []))}
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Corrupt cache file {cache_path}, starting fresh")
    return {"fetched": set()}


def save_cache(root: Path, cache: dict) -> None:
    """Save cache manifest to disk."""
    cache_path = root / DEFAULT_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "fetched": sorted(cache["fetched"]),
                "updated": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


def is_current_year(year: str) -> bool:
    """Check if a year represents the current or future year.

    Returns False for non-integer year strings rather than defaulting to
    True, which would cause unnecessary re-fetches of cached data.
    """
    try:
        return int(year) >= datetime.now().year
    except (ValueError, TypeError):
        logger.warning(f"Non-integer year '{year}' passed to is_current_year, treating as historical")
        return False


def should_fetch(cache: dict, cache_key: str, year: str) -> bool:
    """Decide whether a volume needs fetching.

    Returns True if:
    - The cache_key is not in the cache, OR
    - The year is the current year or later (always refresh current data)
    """
    if cache_key not in cache["fetched"]:
        return True
    if is_current_year(year):
        return True
    return False


def mark_fetched(cache: dict, cache_key: str) -> None:
    """Record a volume as successfully fetched."""
    cache["fetched"].add(cache_key)
