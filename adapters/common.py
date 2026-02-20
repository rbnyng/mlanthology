"""Common utilities shared across adapters."""

import gzip
import json
import logging
import re
import unicodedata
from html import unescape
from pathlib import Path
from typing import Optional

import yaml

# prefer C-accelerated YAML loader when available
try:
    _yaml_Loader = yaml.CSafeLoader
except AttributeError:
    _yaml_Loader = yaml.SafeLoader

logger = logging.getLogger(__name__)


# venue shortnames -> canonical slugs
VENUE_MAP = {
    "ICML": "icml",
    "AISTATS": "aistats",
    "CoLT": "colt",
    "COLT": "colt",
    "UAI": "uai",
    "CoRL": "corl",
    "MIDL": "midl",
    "ALT": "alt",
    "ACML": "acml",
    "CPAL": "cpal",
    "ICLR": "iclr",
    "ICLRW": "iclrw",
    "TMLR": "tmlr",
    "NeurIPS": "neurips",
    "JMLR": "jmlr",
    "MLOSS": "mloss",
    "DMLR": "dmlr",
    "CVPR": "cvpr",
    "ICCV": "iccv",
    "WACV": "wacv",
    "ECCV": "eccv",
    "AAAI": "aaai",
    "IJCAI": "ijcai",
    "JAIR": "jair",
    "MLJ": "mlj",
    "NECO": "neco",
    "FTML": "ftml",
    "Distill": "distill",
    "MLHC": "mlhc",
    "L4DC": "l4dc",
    "CLeaR": "clear",
    "AutoML": "automl",
    "CHIL": "chil",
    "PGM": "pgm",
    "LoG": "log",
    "CoLLAs": "collas",
    "CPAL": "cpal",
    "ISIPTA": "isipta",
}

_VENUES_YAML = Path(__file__).resolve().parent.parent / "hugo" / "data" / "venues.yaml"

def _load_venues() -> dict:
    try:
        with open(_VENUES_YAML) as f:
            return yaml.load(f, Loader=_yaml_Loader) or {}
    except FileNotFoundError:
        return {}

_VENUES = _load_venues()


def get_venue_type(venue_slug: str) -> str:
    """Classify a venue slug as conference, journal, or workshop."""
    return _VENUES.get(venue_slug, {}).get("type", "workshop")


def repair_mojibake(text: str) -> str:
    """Detect and repair double-encoded UTF-8 text.

    When UTF-8 bytes are misinterpreted as ISO-8859-1 and then
    re-encoded as UTF-8, characters like ``é`` become ``Ã©`` and
    ``ń`` becomes ``Å\\x84``.  This function reverses that
    transformation by encoding back to latin-1 and decoding as UTF-8.

    Returns the original text unchanged if no mojibake is detected
    or the round-trip fails.
    """
    if not text:
        return text
    # Quick check: mojibake from double-encoded UTF-8 always contains
    # characters in the U+00C2..U+00DF range (the latin-1 interpretation
    # of UTF-8 leading bytes for 2-byte sequences).
    if not any("\xc2" <= ch <= "\xdf" for ch in text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK Unified Ideograph characters."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_text(text: str) -> str:
    """Normalize unicode to ASCII for key generation.

    Uses NFKD decomposition for accented Latin characters, falling back
    to unidecode for scripts without decompositions (CJK, Cyrillic, etc.).
    """
    nfkd = unicodedata.normalize("NFKD", text)
    result = nfkd.encode("ascii", "ignore").decode("ascii")
    if not result.strip() and text.strip():
        try:
            from unidecode import unidecode
            result = unidecode(text)
        except ImportError:
            pass
    return result


def _romanize_cjk(text: str) -> str:
    """Romanize a CJK name part, collapsing syllables into one token.

    E.g. "良华" -> "lianghua", "振辉" -> "zhenhui".
    Spaces between romanized syllables are removed so CJK given/family
    names become single slug tokens, matching conventional romanization.
    """
    try:
        from unidecode import unidecode
        return unidecode(text).replace(" ", "").lower()
    except ImportError:
        return ""


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = normalize_text(text.lower())
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def slugify_author(author: dict) -> str:
    """Create a URL-safe slug for an author.

    Formats as "family given" (family name first) to match the display
    order used in author page URLs.  For CJK names, each name part is
    romanized as a single token (e.g. 良华 -> "lianghua").
    """
    given = author.get("given", "").strip()
    family = author.get("family", "").strip()
    # Romanize CJK name parts individually so syllables stay joined
    if _has_cjk(family):
        family = _romanize_cjk(family)
    if _has_cjk(given):
        given = _romanize_cjk(given)
    text = f"{family} {given}".strip()
    return slugify(text) or "unknown"


def _strip_latex(text: str) -> str:
    """Remove LaTeX markup, keeping content inside braces."""
    # repeatedly unwrap \command{content} to handle nesting
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # bare commands
    text = re.sub(r"[${}]", "", text)  # math delimiters & stray braces
    return text


def first_content_word(title: str) -> str:
    """Extract first content word from a paper title, skipping stopwords.

    Strips LaTeX markup and captures hyphenated compounds (e.g. "3D-VLA",
    "GPT-4", "R-CNN") as single tokens including digits, so titles like
    "$\\texttt{C2-DPO}$: ..." produce "c2dpo" rather than "texttt".
    """
    stopwords = {
        "a", "an", "the", "on", "in", "at", "of", "for", "to", "and", "or",
        "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "can", "could", "will", "would",
        "shall", "should", "may", "might", "must", "have", "has", "had",
        "not", "no", "nor", "but", "yet", "so", "if", "then", "than",
        "that", "this", "these", "those", "it", "its", "as", "into",
        "through", "about", "above", "below", "between", "under", "over",
        "after", "before", "during", "without", "toward", "towards",
        "how", "what", "when", "where", "which", "who", "whom", "why",
    }
    # Decode HTML entities first so &quot; becomes " (then stripped), not "quot"
    cleaned = _strip_latex(normalize_text(unescape(title)))
    tokens = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", cleaned)
    for token in tokens:
        slug = re.sub(r"[^a-z0-9]", "", token.lower())
        if len(slug) >= 2 and slug not in stopwords and re.search(r"[a-z]", slug):
            return slug
    # Fallback: first token containing at least one letter
    for token in tokens:
        slug = re.sub(r"[^a-z0-9]", "", token.lower())
        if re.search(r"[a-z]", slug):
            return slug
    return "paper"


def make_bibtex_key(
    first_author_family: str,
    year: str,
    venue: str,
    title: str,
) -> str:
    """Generate a human-readable bibtex key.

    Pattern: {lastname}{year}{venue}-{first_content_word}
    Example: liu2025icml-reward
    """
    lastname = normalize_text(first_author_family).lower()
    lastname = re.sub(r"[^a-z]", "", lastname)
    content_word = first_content_word(title)
    venue_slug = venue.lower()
    return f"{lastname}{year}{venue_slug}-{content_word}"


def resolve_bibtex_collisions(keys: list[str]) -> list[str]:
    """Append -a, -b suffixes to resolve duplicate bibtex keys.

    The first occurrence keeps the original key; subsequent duplicates
    get -a, -b, ... suffixes.
    """
    seen: dict[str, int] = {}
    result = []
    for key in keys:
        if key in seen:
            seen[key] += 1
            suffix = chr(ord("a") + seen[key] - 1)
            result.append(f"{key}-{suffix}")
        else:
            seen[key] = 0
            result.append(key)
    return result


# lowercase name particles that belong with the family name
_NAME_PARTICLES = frozenset({
    "van", "von", "de", "del", "della", "der", "den", "di", "du",
    "das", "dos", "do", "da", "le", "la", "el", "al", "bin", "ibn",
    "ten", "ter", "het",
})


def parse_author_name(name: str) -> dict:
    """Split a full name string into given/family components.

    Handles common patterns:
    - "First Last" -> {"given": "First", "family": "Last"}
    - "First Middle Last" -> {"given": "First Middle", "family": "Last"}
    - "Last" -> {"given": "", "family": "Last"}
    - "First van Last" -> {"given": "First", "family": "van Last"}
    - "First de la Last" -> {"given": "First", "family": "de la Last"}

    Name particles (van, von, de, etc.) are kept with the family name
    when they appear in lowercase before the final word.
    """
    name = name.strip()
    if not name:
        return {"given": "", "family": ""}
    parts = name.split()
    if len(parts) == 1:
        return {"given": "", "family": parts[0]}

    # scan for first lowercase particle — everything from there becomes family
    family_start = len(parts) - 1  # default: last word only
    for i in range(1, len(parts) - 1):
        if parts[i] in _NAME_PARTICLES:
            family_start = i
            break

    return {
        "given": " ".join(parts[:family_start]),
        "family": " ".join(parts[family_start:]),
    }


def _fix_misplaced_initial(author: dict) -> dict:
    """Move a leading single-letter initial from family to given name.

    Some upstream sources (especially PMLR YAML) misplace a middle initial
    in the family field, producing entries like:
        {"given": "David", "family": "A Clifton"}   -> "A Clifton, David"
        {"given": "Michael", "family": "A. Osborne"} -> "A. Osborne, Michael"

    This heuristic detects a family name starting with a single letter
    (optionally followed by a period) and a space, then moves that initial
    to the end of the given name.
    """
    family = author.get("family") or ""
    given = author.get("given") or ""
    # Match: single letter, optional period, space, then remaining family name
    m = re.match(r"^([A-Za-z]\.?)\s+(.+)$", family)
    if m:
        initial = m.group(1)
        real_family = m.group(2)
        new_given = f"{given} {initial}".strip() if given else initial
        return {"given": new_given, "family": real_family}
    return author


_SINGLE_LETTER_RE = re.compile(r"^[A-Za-z]\.?$")


# ---------------------------------------------------------------------------
# Pre-parse author name cleaning
# ---------------------------------------------------------------------------

# common BibTeX accent commands -> unicode
_BIBTEX_ACCENT_MAP = {
    r"\L": "\u0141",   # Ł
    r"\l": "\u0142",   # ł
    r"\O": "\u00D8",   # Ø
    r"\o": "\u00F8",   # ø
    r"\AE": "\u00C6",  # Æ
    r"\ae": "\u00E6",  # æ
    r"\AA": "\u00C5",  # Å
    r"\aa": "\u00E5",  # å
    r"\SS": "\u1E9E",  # ẞ
    r"\ss": "\u00DF",  # ß
    r"\DH": "\u00D0",  # Ð
    r"\dh": "\u00F0",  # ð
    r"\TH": "\u00DE",  # Þ
    r"\th": "\u00FE",  # þ
    r"\NG": "\u014A",  # Ŋ
    r"\ng": "\u014B",  # ŋ
    r"\i": "\u0131",   # ı (dotless i)
    r"\j": "\u0237",   # ȷ (dotless j)
}

# matches \Command at start of word where rest is already unicode
# e.g. "\Lącki" -> Ł + ącki
_BIBTEX_CMD_RE = re.compile(
    r"\\(" + "|".join(re.escape(k[1:]) for k in _BIBTEX_ACCENT_MAP) + r")(?=[A-Za-z\u0080-\uffff])"
)

# parenthetical annotations that are never part of a real name
_ANNOTATION_RE = re.compile(
    r"\s*\("
    r"(?:He/Him|She/Her|They/Them|PhD|Ph\.?D\.?|Dr\.?|Jr\.?|Sr\.?|M\.?D\.?|M\.?Sc\.?)"
    r"\)\s*",
    re.IGNORECASE,
)

# parenthetical nicknames / former names, e.g. "Jeong (Kate) Lee"
_PAREN_NAME_RE = re.compile(r"\s*\([A-Za-z\u0080-\uffff-]+\)\s*")


def _clean_raw_author_name(name: str) -> str:
    """Clean a raw author name string before parsing into given/family.

    Handles HTML entities, *, degree/pronoun annotations, parenthetical
    nicknames, residual BibTeX accent commands, and whitespace.
    """
    name = unescape(name)
    name = name.replace("*", "")
    name = _ANNOTATION_RE.sub(" ", name)
    name = _PAREN_NAME_RE.sub(" ", name)

    def _replace_bibtex(m: re.Match) -> str:
        return _BIBTEX_ACCENT_MAP.get(f"\\{m.group(1)}", m.group(0))
    name = _BIBTEX_CMD_RE.sub(_replace_bibtex, name)

    name = " ".join(name.split())
    return name


def _fix_single_letter_family(author: dict) -> dict:
    """Fix single-letter family names from Surname-First input formats.

    e.g. {"given": "Butakov I.", "family": "D."} -> {"given": "I. D.", "family": "Butakov"}

    Only triggers when given contains initials, so legitimate single-letter
    surnames like Weinan E are left alone.
    """
    family = author.get("family") or ""
    given = author.get("given") or ""

    if not _SINGLE_LETTER_RE.match(family) or not given:
        return author

    tokens = given.split()

    has_initials = any(_SINGLE_LETTER_RE.match(t) for t in tokens)
    if not has_initials:
        return author

    name_parts = [t for t in tokens if not _SINGLE_LETTER_RE.match(t)]
    initial_parts = [t for t in tokens if _SINGLE_LETTER_RE.match(t)]

    if not name_parts:
        return author

    new_family = " ".join(name_parts)
    new_given = " ".join(initial_parts + [family])

    return {"given": new_given, "family": new_family}


def _fix_leading_hyphen_family(author: dict) -> dict:
    """Rejoin an orphaned leading-hyphen family name with the last given-name token.

    Some sources split hyphenated surnames incorrectly, producing:
        {"given": "Saeed Sharifi", "family": "-Malvajerdi"}

    This function moves the hyphenated fragment back to the preceding token:
        {"given": "Saeed", "family": "Sharifi-Malvajerdi"}
    """
    family = author.get("family") or ""
    given = author.get("given") or ""
    if not family.startswith("-") or not given:
        return author
    tokens = given.split()
    # Attach the hyphenated suffix to the last given-name token
    new_family = tokens[-1] + family  # e.g. "Sharifi" + "-Malvajerdi"
    new_given = " ".join(tokens[:-1])
    return {"given": new_given, "family": new_family}


def _fix_misplaced_particle(author: dict) -> dict:
    """Move name particles from the end of given name to the start of family name.

    Many upstream sources (PMLR, CVF, DBLP) return names like:
        {"given": "Luc Van", "family": "Gool"}
    when the correct split is:
        {"given": "Luc", "family": "Van Gool"}

    This handles both lowercase and capitalized particles.
    """
    given = author.get("given") or ""
    family = author.get("family") or ""
    if not given or not family:
        return author

    given_words = given.split()
    if len(given_words) < 2:
        return author

    # Check trailing words of given name for particles (case-insensitive)
    # Walk backwards from the end to catch multi-word particles like "van der"
    particle_start = None
    for i in range(len(given_words) - 1, 0, -1):
        if given_words[i].lower() in _NAME_PARTICLES:
            particle_start = i
        else:
            break

    if particle_start is None:
        return author

    particle_and_family = " ".join(given_words[particle_start:]) + " " + family
    new_given = " ".join(given_words[:particle_start])

    return {"given": new_given, "family": particle_and_family}


def _fix_punctuation_only_fields(author: dict) -> dict:
    """Replace name fields that contain only punctuation with empty strings.

    Catches artifacts like family="()", given=".", family="{namdar", etc.
    A field with no letters is treated as empty.  Fields containing '@'
    (leaked email addresses) are also cleared.
    """
    given = author.get("given") or ""
    family = author.get("family") or ""

    def _is_junk(s: str) -> bool:
        if not s:
            return False
        # Contains email artifacts
        if "@" in s or s.startswith("{") or s.endswith("}"):
            return True
        # No alphabetic characters at all
        if not any(c.isalpha() for c in s):
            return True
        return False

    if _is_junk(given):
        given = ""

    if not family or _is_junk(family):
        # If family is empty/junk but given looks like a real name, try to
        # re-parse given as a full name (e.g. given="Lihua Xie", family="()")
        if given and any(c.isalpha() for c in given):
            return parse_author_name(given)
        if _is_junk(family):
            return {"given": "", "family": ""}

    return {"given": given, "family": family}


def parse_bibtex_authors(author_field: str) -> list[dict]:
    """Parse a BibTeX author field into a list of {given, family} dicts.

    BibTeX separates authors with ' and '.  Individual names can be
    "Last, First" or "First Last" format.
    """
    authors = []
    for name in re.split(r"\s+and\s+", author_field):
        name = name.strip()
        if not name:
            continue
        if "," in name:
            parts = name.split(",", 1)
            authors.append({"given": parts[1].strip(), "family": parts[0].strip()})
        else:
            authors.append(parse_author_name(name))
    return authors


def normalize_title_case(title: str) -> str:
    """Normalize obviously broken title casing in source data.

    Converts ALL-CAPS or all-lowercase titles to title case using
    Python's str.title(), which is a reasonable first pass for data
    that is clearly malformed.  Mixed-case titles (the vast majority)
    are left untouched.  Display-time smart title case is applied
    separately in build_content.py.
    """
    if not title:
        return title
    alpha = "".join(c for c in title if c.isalpha())
    if not alpha:
        return title
    if alpha.isupper() or alpha.islower():
        return title.title()
    return title


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def write_venue_json(
    venue: str,
    year: str,
    papers: list[dict],
    output_dir: Path,
    *,
    filename: Optional[str] = None,
) -> Path:
    """Write a venue-year gzipped JSON file in the canonical format.

    Args:
        venue: Venue slug (e.g. "icml").
        year: Year string.
        papers: List of paper dicts (already passed through normalize_paper).
        output_dir: Output directory.
        filename: Override output filename (without extension). Defaults
                  to "{venue}-{year}".

    Returns:
        Path to the written file.
    """
    fname = filename or f"{venue}-{year}"
    out_path = output_dir / f"{fname}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(
            {"venue": venue, "year": year, "papers": papers},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"  Wrote {out_path}")
    return out_path


def read_venue_json(path: Path) -> dict:
    """Read a venue-year gzipped JSON file.

    Returns the parsed dict with keys: venue, year, papers.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# code_url cleanup
# ---------------------------------------------------------------------------

# Markdown link: [![alt](img)](url) or [text](url)
_MD_LINK_RE = re.compile(r"\[(?:[^\]]*\])?[^\]]*\]\((https?://[^)]+)\)")


def clean_code_url(raw: str) -> str:
    """Extract a plain URL from a code_url field.

    OpenReview V1 API (ICLR 2018-2021) returns markdown badge markup
    like ``[![github](/images/github_icon.svg) repo](https://github.com/...)``.
    Some JMLR entries contain bare filenames or free-text descriptions.

    This function extracts the first ``https://`` URL from markdown links
    (preferring GitHub over Papers With Code), and strips leading whitespace
    from otherwise valid URLs.  Non-URL values are discarded.
    """
    if not raw:
        return ""
    raw = raw.strip()

    # Already a clean URL
    if raw.startswith("https://") or raw.startswith("http://"):
        # But might have multiple URLs separated by semicolons or text
        if " " not in raw and ";" not in raw:
            return raw

    # Extract all markdown link URLs
    urls = _MD_LINK_RE.findall(raw)
    if urls:
        # Prefer GitHub URLs over paperswithcode
        for url in urls:
            if "github.com" in url or "gitlab.com" in url:
                return url
        return urls[0]

    # Try to find a bare URL in free text
    bare = re.search(r"(https?://\S+)", raw)
    if bare:
        return bare.group(1).rstrip(".,;)")

    return ""


# required fields — missing one means a bug in the adapter
_REQUIRED_PAPER_FIELDS = ("bibtex_key", "title", "authors", "year")


def normalize_paper(paper: dict) -> dict:
    """Normalize a paper dict to the ML Anthology canonical schema.

    The canonical id is the bibtex_key, which doubles as the page URL slug.
    The original source-specific identifier is preserved in source_id.
    venue_type is auto-classified if not set explicitly by the adapter.

    Raises:
        KeyError: If a required field is missing from the paper dict.
    """
    missing = [f for f in _REQUIRED_PAPER_FIELDS if f not in paper]
    if missing:
        raise KeyError(
            f"Paper dict missing required fields {missing}. "
            f"Available keys: {sorted(paper.keys())}"
        )

    venue = paper.get("venue", "")

    # clean up text fields and run the author name fixup pipeline
    title = normalize_title_case(repair_mojibake(unescape(paper["title"])))
    _raw_authors = []
    for a in paper["authors"]:
        cleaned = {
            "given": _clean_raw_author_name(repair_mojibake(a.get("given") or "")),
            "family": _clean_raw_author_name(repair_mojibake(a.get("family") or "")),
        }
        cleaned = _fix_misplaced_initial(cleaned)
        cleaned = _fix_misplaced_particle(cleaned)
        cleaned = _fix_single_letter_family(cleaned)
        cleaned = _fix_leading_hyphen_family(cleaned)
        cleaned = _fix_punctuation_only_fields(cleaned)
        _raw_authors.append(cleaned)
    _raw_authors = [a for a in _raw_authors if a.get("given") or a.get("family")]
    authors = [{**a, "slug": slugify_author(a)} for a in _raw_authors]
    abstract = repair_mojibake(paper.get("abstract", ""))

    return {
        "bibtex_key": paper["bibtex_key"],
        "title": title,
        "authors": authors,
        "year": paper["year"],
        "venue": venue,
        "venue_name": paper.get("venue_name", ""),
        "venue_type": paper.get("venue_type", get_venue_type(venue)),
        "volume": paper.get("volume", ""),
        "number": paper.get("number", ""),
        "pages": paper.get("pages", ""),
        "abstract": abstract,
        "pdf_url": paper.get("pdf_url", ""),
        "venue_url": paper.get("venue_url", ""),
        "doi": paper.get("doi", ""),
        "openreview_url": paper.get("openreview_url", ""),
        "code_url": clean_code_url(paper.get("code_url", "")),
        "source": paper.get("source", ""),
        "source_id": paper.get("source_id", ""),
    }
