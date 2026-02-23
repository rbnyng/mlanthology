"""Shared utilities for scripts.

Common I/O helpers (gzipped JSONL read/write), title normalisation,
HTML cleaning, fuzzy title lookup, HTTP session factory with retry,
and project-root bootstrapping.
"""

import gzip
import json
import re
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is importable (idempotent).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standard directories
LEGACY_DIR = ROOT / "data" / "legacy"
PAPERS_DIR = ROOT / "data" / "papers"


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



def normalize_title(title: str) -> str:
    """Normalise a paper title for fuzzy matching.

    Lowercases, strips punctuation (keeping only alphanumeric + spaces),
    and collapses whitespace.  Used for cross-source title matching in
    enrichment and patching scripts.
    """
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def fuzzy_lookup(
    title: str,
    index: dict[str, str],
    *,
    threshold: float = 0.92,
) -> str | None:
    """Look up *title* in *index* by normalised exact match, then fuzzy fallback.

    *index* maps ``normalize_title(title) → value``.  Returns the value for the
    best match above *threshold*, or ``None``.
    """
    norm = normalize_title(title)
    if norm in index:
        return index[norm]

    best_ratio = 0.0
    best_val = None
    for idx_norm, val in index.items():
        ratio = SequenceMatcher(None, norm, idx_norm, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_val = val

    return best_val if best_ratio >= threshold else None


_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&#39;": "'", "&quot;": '"', "&nbsp;": " ",
    "&#8211;": "\u2013", "&#8212;": "\u2014",
}


def clean_html(text: str, *, replace_tags_with: str = "") -> str:
    """Strip HTML tags and decode common entities, then collapse whitespace."""
    text = _TAG_RE.sub(replace_tags_with, text)
    for ent, ch in _ENTITY_MAP.items():
        text = text.replace(ent, ch)
    return " ".join(text.split()).strip()



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

    # set a default timeout for all requests made via this session.
    # (requests doesn't natively support a session-level timeout, so we
    #  monkey-patch the send method.)
    _original_send = session.send

    def _send_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return _original_send(*args, **kwargs)

    session.send = _send_with_timeout  # type: ignore[assignment]
    return session



def pdf_text_from_url(
    url: str,
    *,
    pages: str = "first",
    timeout: int = 30,
    session: requests.Session | None = None,
) -> str | None:
    """Download a PDF from *url* and extract its text content.

    Parameters
    ----------
    url : str
        URL of a PDF file.
    pages : str
        Which pages to extract.  ``"first"`` (default) returns only the
        first page — useful for title extraction.  ``"all"`` concatenates
        every page.  A single integer string (e.g. ``"3"``) extracts that
        one page (0-indexed).
    timeout : int
        HTTP request timeout in seconds.
    session : requests.Session | None
        Optional pre-configured session.  Falls back to a plain
        ``requests.get`` call when *None*.

    Returns
    -------
    str | None
        Extracted text, or *None* if the URL returned a non-200 status
        or the PDF could not be parsed.

    Requires
    --------
    ``pymupdf`` (``pip install pymupdf``).
    """
    import pymupdf  # lazy import — only needed when called

    try:
        if session is not None:
            resp = session.get(url, timeout=timeout)
        else:
            resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    try:
        doc = pymupdf.open(stream=resp.content, filetype="pdf")
    except Exception:
        return None

    try:
        if pages == "first":
            return doc[0].get_text() if len(doc) > 0 else ""
        if pages == "all":
            return "\n".join(page.get_text() for page in doc)
        # Single page by index
        idx = int(pages)
        if 0 <= idx < len(doc):
            return doc[idx].get_text()
        return ""
    finally:
        doc.close()


def pdf_title_from_url(url: str, **kwargs) -> str | None:
    """Extract and normalise the title from the first page of a remote PDF.

    A thin wrapper around :func:`pdf_text_from_url` that grabs the first
    non-blank line of the first page — which for conference proceedings
    is almost always the paper title.

    Returns *None* when the PDF is unreachable or unparseable.
    """
    text = pdf_text_from_url(url, pages="first", **kwargs)
    if text is None:
        return None
    # Title is typically the first non-blank line(s) before authors.
    # Heuristic: take lines until we hit a short line, an author-like
    # pattern (word + comma + word), or an affiliation keyword.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None

    # Fix spaced-out OCR artefact: "T H E  I N T E R P R E T A T I O N"
    # Detect lines where most "words" are single characters.
    fixed_lines: list[str] = []
    for ln in lines:
        tokens = ln.split()
        if len(tokens) > 4 and sum(1 for t in tokens if len(t) == 1) > len(tokens) * 0.6:
            # Collapse single-char runs: join chars, split on double-space gaps
            collapsed = re.sub(r"(?<=[A-Za-z]) (?=[A-Za-z](?:\s|$))", "", ln)
            collapsed = re.sub(r"\s{2,}", " ", collapsed).strip()
            fixed_lines.append(collapsed)
        else:
            fixed_lines.append(ln)
    lines = fixed_lines

    # For most proceedings PDFs the title is the very first line.
    # Multi-line titles are common; accumulate until we see a clear break.
    title_parts: list[str] = []
    for ln in lines:
        # Stop at author lines (contain "@", "University", "Institute", etc.)
        lower = ln.lower()
        if any(kw in lower for kw in ("@", "university", "institute", "laboratory",
                                       "department", "abstract", "e-mail")):
            break
        # Stop at lines that look like "Firstname Lastname" only (author name)
        # Heuristic: <=4 words, each capitalised, no lowercase mid-words
        words = ln.split()
        if 1 < len(words) <= 4 and all(w[0].isupper() for w in words if w):
            # Likely an author name — but could also be a short title.
            # Accept if we haven't accumulated anything yet (it IS the title).
            if title_parts:
                break
        title_parts.append(ln)
        # Most titles are 1-2 lines
        if len(title_parts) >= 3:
            break
    return " ".join(title_parts) if title_parts else None
