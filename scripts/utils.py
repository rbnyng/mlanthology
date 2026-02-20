"""Shared utilities for scripts.

Common I/O helpers (gzipped JSONL read/write), title normalisation,
HTTP session factory with retry, and project-root bootstrapping.
"""

import gzip
import json
import re
import sys
import tempfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is importable (idempotent).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standard directories
LEGACY_DIR = ROOT / "data" / "legacy"
PAPERS_DIR = ROOT / "data" / "papers"

# ---------------------------------------------------------------------------
# Gzipped JSONL I/O
# ---------------------------------------------------------------------------


def read_legacy(path: Path) -> list[dict]:
    """Read a gzipped JSONL file and return a list of dicts."""
    papers: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def write_legacy(path: Path, papers: list[dict], *, atomic: bool = False) -> None:
    """Write a list of dicts as gzipped JSONL.

    When *atomic* is True the file is written to a temporary neighbour
    first, then atomically renamed into place so a crash mid-write
    never leaves a half-written file.
    """
    if atomic:
        tmp = path.with_suffix(".jsonl.gz.tmp")
        _write_gz(tmp, papers)
        tmp.replace(path)
    else:
        _write_gz(path, papers)


def _write_gz(path: Path, papers: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Title normalisation (for matching across sources)
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Normalise a paper title for fuzzy matching.

    Lowercases, strips punctuation (keeping only alphanumeric + spaces),
    and collapses whitespace.  Used for cross-source title matching in
    enrichment and patching scripts.
    """
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# HTTP session with automatic retry / back-off
# ---------------------------------------------------------------------------


def make_session(
    *,
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    timeout: float = 30,
) -> requests.Session:
    """Create a :class:`requests.Session` with automatic retry.

    Parameters
    ----------
    retries : int
        Maximum number of retries per request.
    backoff_factor : float
        Multiplier for exponential back-off between retries.
    status_forcelist : tuple[int, ...]
        HTTP status codes that trigger a retry.
    timeout : float
        Default timeout in seconds (applied to every request via a
        transport adapter hook).
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET", "HEAD", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set a default timeout for all requests made via this session.
    # (requests doesn't natively support a session-level timeout, so we
    #  monkey-patch the send method.)
    _original_send = session.send

    def _send_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return _original_send(*args, **kwargs)

    session.send = _send_with_timeout  # type: ignore[assignment]
    return session
