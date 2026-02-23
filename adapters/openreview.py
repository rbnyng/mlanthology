"""OpenReview adapter: fetches accepted papers via the OpenReview API.

Supports ICLR, TMLR, and other venues hosted on OpenReview.
- API V2 (api2.openreview.net) for ICLR 2024+, TMLR
- API V1 (api.openreview.net) for ICLR 2018-2023
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import openreview

from .common import (
    make_bibtex_key, resolve_bibtex_collisions, normalize_paper, write_venue_json,
    parse_author_name,
)
from .cache import should_fetch, mark_fetched

logger = logging.getLogger(__name__)

VENUE_FULLNAMES = {
    "ICLR": "International Conference on Learning Representations",
    "TMLR": "Transactions on Machine Learning Research",
}

# Format: (venue_id, venue_shortname, year, api_version)
# year="all" means a rolling venue — fetch all and split by publication year
KNOWN_VENUES = [
    ("ICLR.cc/2025/Conference", "ICLR", "2025", "v2"),
    ("ICLR.cc/2024/Conference", "ICLR", "2024", "v2"),
    ("ICLR.cc/2023/Conference", "ICLR", "2023", "v1_venueid"),
    ("ICLR.cc/2022/Conference", "ICLR", "2022", "v1_venueid"),
    ("ICLR.cc/2021/Conference", "ICLR", "2021", "v1_venueid"),
    ("ICLR.cc/2020/Conference", "ICLR", "2020", "v1_venueid"),
    ("ICLR.cc/2019/Conference", "ICLR", "2019", "v1_decision"),
    ("ICLR.cc/2018/Conference", "ICLR", "2018", "v1_decision"),
    ("TMLR", "TMLR", "all", "v2_journal"),
]

# Decision invitation patterns per year (for v1_decision years)
DECISION_INVITATIONS = {
    "2018": "Acceptance_Decision",
    "2019": "Meta_Review",
}


def _get_client_v2() -> openreview.api.OpenReviewClient:
    """Create an OpenReview API V2 client (guest access)."""
    return openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net"
    )


def _get_client_v1() -> openreview.Client:
    """Create an OpenReview API V1 client (guest access)."""
    return openreview.Client(
        baseurl="https://api.openreview.net"
    )


def _is_accepted_venue_label(venue_label: str) -> bool:
    """Check if a V1 venue label indicates acceptance (not submitted/rejected)."""
    lower = venue_label.lower()
    return "submitted" not in lower and venue_label != ""


def _extract_paper_v2(note, venue: str, year: str, venue_name: str) -> Optional[dict]:
    """Extract canonical paper dict from an API V2 Note object."""
    content = note.content
    if not content:
        return None

    title = content.get("title", {}).get("value", "")
    if not title:
        return None

    author_names = content.get("authors", {}).get("value", [])
    authors = [parse_author_name(name) for name in author_names]
    if not authors:
        return None

    abstract = content.get("abstract", {}).get("value", "")

    pdf_path = content.get("pdf", {}).get("value", "")
    if not pdf_path:
        pdf_url = ""
    elif pdf_path.startswith("http"):
        pdf_url = pdf_path
    else:
        pdf_url = f"https://openreview.net{pdf_path}"

    openreview_url = f"https://openreview.net/forum?id={note.forum}"
    source_id = note.forum

    first_author = authors[0]
    bkey = make_bibtex_key(
        first_author_family=first_author.get("family", ""),
        year=year,
        venue=venue.lower(),
        title=title,
    )

    return {
        "bibtex_key": bkey,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue.lower(),
        "venue_name": venue_name,
        "volume": "",
        "pages": "",
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue_url": openreview_url,
        "openreview_url": openreview_url,
        "code_url": content.get("code", {}).get("value", ""),
        "source": "openreview",
        "source_id": source_id,
    }


def _extract_paper_v1(note, venue: str, year: str, venue_name: str) -> Optional[dict]:
    """Extract canonical paper dict from an API V1 Note object."""
    content = note.content
    if not content:
        return None

    title = content.get("title", "")
    if not title:
        return None

    author_names = content.get("authors", [])
    authors = [parse_author_name(name) for name in author_names]
    if not authors:
        return None

    abstract = content.get("abstract", "")

    pdf_path = content.get("pdf", "")
    if not pdf_path:
        pdf_url = ""
    elif pdf_path.startswith("http"):
        pdf_url = pdf_path
    else:
        pdf_url = f"https://openreview.net{pdf_path}"

    openreview_url = f"https://openreview.net/forum?id={note.forum}"
    source_id = note.forum

    first_author = authors[0]
    bkey = make_bibtex_key(
        first_author_family=first_author.get("family", ""),
        year=year,
        venue=venue.lower(),
        title=title,
    )

    return {
        "bibtex_key": bkey,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue.lower(),
        "venue_name": venue_name,
        "volume": "",
        "pages": "",
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue_url": openreview_url,
        "openreview_url": openreview_url,
        "code_url": content.get("code", ""),
        "source": "openreview",
        "source_id": source_id,
    }


def _fetch_v2(venue_id: str, venue: str, year: str, venue_name: str) -> list[dict]:
    """Fetch accepted papers using API V2 (ICLR 2024+)."""
    client = _get_client_v2()
    notes = client.get_all_notes(content={"venueid": venue_id})

    papers = []
    bibtex_keys = []
    for note in notes:
        paper = _extract_paper_v2(note, venue, year, venue_name)
        if paper is None:
            continue
        bibtex_keys.append(paper["bibtex_key"])
        papers.append(paper)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    return papers


def _fetch_v1_venueid(venue_id: str, venue: str, year: str, venue_name: str) -> list[dict]:
    """Fetch accepted papers using V1 API with venueid content filter (ICLR 2020-2023).

    For these years, all submissions share the same venueid. Accepted papers are
    distinguished by venue labels like 'ICLR 2023 poster', while rejected ones
    show 'Submitted to ICLR 2023'.
    """
    client = _get_client_v1()
    notes = list(client.get_all_notes(content={"venueid": venue_id}))

    papers = []
    bibtex_keys = []
    skipped = 0
    for note in notes:
        venue_label = note.content.get("venue", "")
        if not _is_accepted_venue_label(venue_label):
            skipped += 1
            continue

        paper = _extract_paper_v1(note, venue, year, venue_name)
        if paper is None:
            continue
        bibtex_keys.append(paper["bibtex_key"])
        papers.append(paper)

    if skipped:
        logger.info(f"  Skipped {skipped} non-accepted papers")

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    return papers


def _fetch_v1_decision(venue_id: str, venue: str, year: str, venue_name: str) -> list[dict]:
    """Fetch accepted papers using V1 API with decision replies (ICLR 2018-2019).

    For these years, papers have no venueid or venue label. Instead, we fetch all
    Blind_Submission notes with their direct replies, then check for Accept
    decisions in the replies.
    """
    client = _get_client_v1()
    notes = list(client.get_all_notes(
        invitation=f"{venue_id}/-/Blind_Submission",
        details="directReplies",
    ))

    decision_type = DECISION_INVITATIONS.get(year, "Decision")

    papers = []
    bibtex_keys = []
    for note in notes:
        # Check decision in direct replies
        accepted = False
        for reply in note.details.get("directReplies", []):
            inv = reply.get("invitation", "")
            if decision_type in inv:
                content = reply.get("content", {})
                # Field name varies: 'decision' (2018) vs 'recommendation' (2019)
                decision = content.get("decision", content.get("recommendation", ""))
                if "Accept" in decision:
                    accepted = True
                break

        if not accepted:
            continue

        paper = _extract_paper_v1(note, venue, year, venue_name)
        if paper is None:
            continue
        bibtex_keys.append(paper["bibtex_key"])
        papers.append(paper)

    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    return papers


def _fetch_v2_journal(venue_id: str, venue: str, venue_name: str) -> dict[str, list[dict]]:
    """Fetch all accepted papers from a rolling journal, split by publication year.

    Returns dict mapping year string to list of papers.
    """
    client = _get_client_v2()
    notes = client.get_all_notes(content={"venueid": venue_id})

    # Group notes by publication year
    by_year: dict[str, list] = {}
    for note in notes:
        if note.pdate:
            year = str(datetime.datetime.fromtimestamp(note.pdate / 1000).year)
        elif note.cdate:
            year = str(datetime.datetime.fromtimestamp(note.cdate / 1000).year)
        else:
            continue
        by_year.setdefault(year, []).append(note)

    result = {}
    for year in sorted(by_year.keys()):
        papers = []
        bibtex_keys = []
        for note in by_year[year]:
            paper = _extract_paper_v2(note, venue, year, venue_name)
            if paper is None:
                continue
            bibtex_keys.append(paper["bibtex_key"])
            papers.append(paper)

        resolved_keys = resolve_bibtex_collisions(bibtex_keys)
        for paper, key in zip(papers, resolved_keys):
            paper["bibtex_key"] = key

        result[year] = papers
        logger.info(f"  {venue} {year}: {len(papers)} papers")

    return result


def fetch_venue(
    venue_id: str,
    venue: str,
    year: str,
    api_version: str = "v2",
) -> list[dict]:
    """Fetch all accepted papers for a single venue/year from OpenReview."""
    venue_name = VENUE_FULLNAMES.get(venue, venue)
    logger.info(f"Fetching {venue} {year} from OpenReview ({api_version}, venue_id={venue_id})")

    if api_version == "v2":
        papers = _fetch_v2(venue_id, venue, year, venue_name)
    elif api_version == "v1_venueid":
        papers = _fetch_v1_venueid(venue_id, venue, year, venue_name)
    elif api_version == "v1_decision":
        papers = _fetch_v1_decision(venue_id, venue, year, venue_name)
    else:
        raise ValueError(f"Unknown api_version: {api_version}")

    logger.info(f"  Fetched {len(papers)} accepted papers for {venue} {year}")
    return papers


def _write_venue_year(venue: str, year: str, papers: list[dict], output_dir: Path) -> None:
    """Write a single venue-year JSON file."""
    write_venue_json(venue.lower(), year, [normalize_paper(p) for p in papers], output_dir)


def fetch_all(
    venues: Optional[list[tuple[str, str, str, str]]] = None,
    output_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
) -> dict[str, list[dict]]:
    """Fetch all specified OpenReview venues and write YAML output.

    Returns dict mapping venue-year to list of papers.
    """
    if venues is None:
        venues = KNOWN_VENUES

    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = {}

    for venue_id, venue, year, api_version in venues:
        if api_version == "v2_journal":
            # Rolling journals always re-fetch (papers added continuously)
            cache_key = f"openreview-{venue.lower()}-journal"
            venue_name = VENUE_FULLNAMES.get(venue, venue)
            logger.info(f"Fetching {venue} (all years) from OpenReview (v2_journal)")
            by_year = _fetch_v2_journal(venue_id, venue, venue_name)
            for y, papers in by_year.items():
                if papers:
                    venue_year = f"{venue.lower()}-{y}"
                    all_papers[venue_year] = papers
                    _write_venue_year(venue, y, papers, output_dir)
            if cache is not None:
                mark_fetched(cache, cache_key)
        else:
            cache_key = f"openreview-{venue.lower()}-{year}"
            if cache is not None and not should_fetch(cache, cache_key, year):
                logger.info(f"Skipping {venue} {year} — cached")
                continue

            venue_year = f"{venue.lower()}-{year}"
            papers = fetch_venue(venue_id, venue, year, api_version)
            if papers:
                all_papers[venue_year] = papers
                _write_venue_year(venue, year, papers, output_dir)
                if cache is not None:
                    mark_fetched(cache, cache_key)

    return all_papers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch OpenReview proceedings")
    parser.add_argument("--venue-id", type=str, help="OpenReview venue ID (e.g. ICLR.cc/2024/Conference)")
    parser.add_argument("--venue", type=str, help="Venue shortname (e.g. ICLR)")
    parser.add_argument("--year", type=str, help="Year")
    parser.add_argument("--api-version", type=str, default="v2",
                        choices=["v2", "v1_venueid", "v1_decision"],
                        help="API version to use")
    parser.add_argument("--all", action="store_true", help="Fetch all known venues")
    parser.add_argument("--output", type=str, default="data/papers", help="Output directory")
    args = parser.parse_args()

    if args.venue_id and args.venue and args.year:
        papers = fetch_venue(args.venue_id, args.venue, args.year, args.api_version)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_venue_json(args.venue.lower(), args.year, [normalize_paper(p) for p in papers], output_dir)
        print(f"Wrote {len(papers)} papers to {out_path}")
    elif args.all:
        fetch_all(output_dir=Path(args.output))
    else:
        parser.print_help()
