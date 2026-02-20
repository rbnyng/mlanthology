"""Hugo page serialization helpers.

Generates markdown+YAML front matter for paper pages, venue indices,
year indices, and author leaf pages.  Uses hand-serialized YAML for
~15x speedup over yaml.dump per paper.
"""

import re
import unicodedata

from scripts.titlecase import smart_title_case

# ---------------------------------------------------------------------------
# YAML quoting helper for hand-serialized front matter
# ---------------------------------------------------------------------------

_YAML_NEEDS_QUOTING = re.compile(r'[:\{\}\[\],&\*\?|>\'"%@`#!]|^[-?]')


def yaml_scalar(value: str) -> str:
    """Format a string as a safe YAML scalar value.

    Uses double-quoting when the value contains characters that are
    special in YAML.  Plain scalars are used otherwise.
    """
    if not value:
        return '""'
    if _YAML_NEEDS_QUOTING.search(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


# ---------------------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------------------

def _fix_latex_braces(s: str) -> str:
    """Insert missing braces around LaTeX command arguments."""
    _CMDS = (
        r"mathcal", r"mathbf", r"mathrm", r"mathtt", r"mathbb",
        r"mathfrak", r"mathsf",
        r"textrm", r"texttt", r"textbf", r"textit", r"textsf",
        r"bf", r"rm", r"it", r"tt", r"sf",
        r"sqrt",
    )
    pattern = r"\\(" + "|".join(_CMDS) + r")([A-Za-z0-9][A-Za-z0-9_-]*)"
    return re.sub(pattern, r"\\\1{\2}", s)


def sanitize(s: str) -> str:
    """Collapse whitespace and strip control characters for YAML front matter."""
    if not s:
        return ""
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    s = re.sub(r"[\n\t]+", " ", s)
    s = re.sub(r"\$\$(.+?)\$\$", r"$\1$", s)
    s = _fix_latex_braces(s)
    return s


# ---------------------------------------------------------------------------
# Author name helpers
# ---------------------------------------------------------------------------

def normalize_name_case(name: str) -> str:
    """Normalize author name casing.

    Converts all-uppercase or all-lowercase names to title case.
    Mixed-case names (e.g. LaRavia, McArthur, de Sá) are left untouched.
    """
    if not name:
        return name
    alpha = "".join(c for c in name if c.isalpha())
    if not alpha:
        return name
    if alpha.isupper() or alpha.islower():
        return name.title()
    return name


def author_display(author: dict) -> str:
    """Get full display name for an author."""
    given = normalize_name_case(author.get("given", "").strip())
    family = normalize_name_case(author.get("family", "").strip())
    if given:
        return f"{family}, {given}"
    return family


def ascii_letter(char: str) -> str:
    """Map a single character to its ASCII base letter for index grouping."""
    _LETTER_MAP = {
        "Đ": "D", "đ": "D",
        "Ł": "L", "ł": "L",
        "Ø": "O", "ø": "O",
        "Ħ": "H", "ħ": "H",
        "Ŧ": "T", "ŧ": "T",
        "Ŋ": "N", "ŋ": "N",
        "Ð": "D", "ð": "D",
        "Þ": "T", "þ": "T",
        "ß": "S",
    }
    upper = char.upper()
    if upper in _LETTER_MAP:
        return _LETTER_MAP[upper]
    nfkd = unicodedata.normalize("NFKD", upper)
    for c in nfkd:
        if c.isascii() and c.isalpha():
            return c.upper()
    return ""


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def build_paper_page(paper: dict, venues: dict, slugify_author_fn) -> str:
    """Generate Hugo markdown content for a single paper."""
    title = smart_title_case(sanitize(paper.get("title", "")))
    venue = paper.get("venue", "")
    venue_name = paper.get("venue_name") or venues.get(venue, {}).get("name", venue.upper())
    venue_type = paper.get("venue_type", "conference")
    year = str(paper.get("year", ""))
    volume = str(paper.get("volume", ""))
    number = str(paper.get("number", ""))
    pages = paper.get("pages", "")
    abstract = sanitize(paper.get("abstract", ""))
    bibtex_key = paper.get("bibtex_key", "")
    pdf_url = paper.get("pdf_url", "")
    venue_url = paper.get("venue_url", "")
    doi = paper.get("doi", "")
    openreview_url = paper.get("openreview_url", "")
    code_url = paper.get("code_url", "")
    source = paper.get("source", "")
    source_id = paper.get("source_id", "")

    authors = paper.get("authors", [])

    pa_lines = []
    for a in authors:
        given = normalize_name_case(a.get("given", ""))
        family = normalize_name_case(a.get("family", ""))
        slug = a.get("slug") or slugify_author_fn(a)
        display = f"{family}, {given}".strip(", ")
        letter = ascii_letter(display[0]) if display else "#"
        if not letter:
            letter = "#"
        pa_lines.append(f"- given: {yaml_scalar(given)}")
        pa_lines.append(f"  family: {yaml_scalar(family)}")
        pa_lines.append(f"  slug: {yaml_scalar(slug)}")
        pa_lines.append(f"  letter: {yaml_scalar(letter)}")
    paper_authors_block = "\n".join(pa_lines)

    lines = [
        "---",
        "type: venue",
        f"title: {yaml_scalar(title)}",
        f"bibtex_key: {yaml_scalar(bibtex_key)}",
        f"venue: {yaml_scalar(venue)}",
        f"venue_name: {yaml_scalar(venue_name)}",
        f"venue_type: {yaml_scalar(venue_type)}",
        f"year: {yaml_scalar(year)}",
        f"volume: {yaml_scalar(volume)}",
        f"number: {yaml_scalar(number)}",
        f"pages: {yaml_scalar(pages)}",
        f"paper_authors:",
        paper_authors_block,
        f"abstract: {yaml_scalar(abstract)}",
        f"pdf_url: {yaml_scalar(pdf_url)}",
        f"venue_url: {yaml_scalar(venue_url)}",
        f"doi: {yaml_scalar(doi)}",
        f"openreview_url: {yaml_scalar(openreview_url)}",
        f"code_url: {yaml_scalar(code_url)}",
        f"source: {yaml_scalar(source)}",
        f"source_id: {yaml_scalar(source_id)}",
        "---",
        "",
    ]
    return "\n".join(lines)


def build_venue_index(venue_slug: str, years: list, paper_counts: dict, venues: dict) -> str:
    """Generate _index.md for a venue listing all years."""
    venue_name = venues.get(venue_slug, {}).get("name", venue_slug.upper())
    years_sorted = sorted(years, reverse=True)
    total = sum(paper_counts.get((venue_slug, y), 0) for y in years)

    body_lines = [f"# {venue_name}", ""]
    for year in years_sorted:
        count = paper_counts.get((venue_slug, year), 0)
        body_lines.append(f"- [{year}](./{year}/) ({count} papers)")

    return (
        f"---\ntype: venue\ntitle: {yaml_scalar(venue_name)}\nvenue_slug: {yaml_scalar(venue_slug)}\npaper_count: {total}\n---\n\n"
        + "\n".join(body_lines) + "\n"
    )


def build_year_index(venue_slug: str, year: str, paper_count: int, venues: dict) -> str:
    """Generate _index.md for a venue-year page."""
    venue_display = venues.get(venue_slug, {}).get("short", venue_slug.upper())
    title = f"{venue_display} {year}"
    return f"---\ntype: venue\ntitle: {yaml_scalar(title)}\nvenue_slug: {yaml_scalar(venue_slug)}\nyear: {yaml_scalar(year)}\npaper_count: {paper_count}\n---\n"


def build_author_page(slug: str, display_name: str, papers: list[dict]) -> str:
    """Generate {slug}.md leaf page for an author."""
    letter = ascii_letter(display_name[0]) if display_name else ""
    if not letter:
        letter = "#"

    lines = [
        "---",
        f"title: {yaml_scalar(display_name)}",
        f"author_slug: {yaml_scalar(slug)}",
        f"letter: {yaml_scalar(letter)}",
        f"paper_count: {len(papers)}",
        "paper_paths:",
    ]
    for p in papers:
        path = f"/{p['venue']}/{p['year']}/{p['id']}"
        lines.append(f"- {yaml_scalar(path)}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)
