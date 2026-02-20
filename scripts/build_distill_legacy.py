#!/usr/bin/env python3
"""One-time script to build legacy file for Distill (distill.pub).

Distill is not indexed by DBLP, so this scrapes distill.pub directly:
  1. Fetches the index page to discover article URLs.
  2. Fetches each article to extract embedded BibTeX metadata.
  3. Writes distill-legacy.jsonl.gz.

Then enrich with S2 and rebuild content:
    python scripts/enrich_legacy.py --s2-api-key KEY --venue distill
    python scripts/build_content.py
"""

import logging
import re
import time
from pathlib import Path

from scripts.utils import ROOT, LEGACY_DIR, write_legacy, make_session

from adapters.common import make_bibtex_key, resolve_bibtex_collisions, normalize_paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
BASE_URL = "https://distill.pub"

# Skip editorial/meta posts
_SKIP_SLUGS = {"distill-hiatus", "editorial-update"}


_session = make_session(retries=3, backoff_factor=1.0)


def fetch_page(url: str) -> str | None:
    """Fetch a page with retries (uses shared session)."""
    try:
        resp = _session.get(url)
        if resp.status_code == 200:
            return resp.text
        logger.warning(f"  HTTP {resp.status_code} for {url}")
    except Exception as e:
        logger.warning(f"  Request error for {url}: {e}")
    return None


def discover_articles() -> list[dict]:
    """Fetch the Distill index and extract article URLs + metadata."""
    html = fetch_page(BASE_URL)
    if not html:
        logger.error("Could not fetch Distill index page")
        return []

    # Articles are listed as links with pattern /{year}/{slug}
    # Extract from the index page listing
    articles = []
    seen = set()

    # Match article links: 2016/augmented-rnns, /2016/augmented-rnns, etc.
    for match in re.finditer(
        r'href="/?(?:https?://distill\.pub/)?(20\d{2})/([\w-]+)"',
        html,
    ):
        year, slug = match.group(1), match.group(2)
        if slug in _SKIP_SLUGS:
            continue
        key = f"{year}/{slug}"
        if key not in seen:
            seen.add(key)
            articles.append({
                "year": year,
                "slug": slug,
                "url": f"{BASE_URL}/{year}/{slug}/",
            })

    # Sort by year then slug
    articles.sort(key=lambda a: (a["year"], a["slug"]))
    return articles


def parse_bibtex(bibtex: str) -> dict:
    """Extract fields from a BibTeX entry string."""
    fields = {}
    for match in re.finditer(r'(\w+)\s*=\s*\{([^}]*)\}', bibtex):
        fields[match.group(1).lower()] = match.group(2).strip()
    return fields


def fetch_article_metadata(article: dict) -> dict | None:
    """Fetch a single Distill article page and extract metadata."""
    html = fetch_page(article["url"])
    if not html:
        return None

    # Extract BibTeX block — Distill embeds it in the page.
    # Match from @article{ to the closing } that follows a newline.
    bibtex_match = re.search(r'(@article\{.+?\n\})', html, re.DOTALL)
    if not bibtex_match:
        logger.warning(f"  No BibTeX found for {article['url']}")
        return None

    bibtex_str = bibtex_match.group(1)
    fields = parse_bibtex(bibtex_str)

    title = fields.get("title", "").strip()
    if not title:
        # Fallback: extract from <title> tag
        title_match = re.search(r'<title>(.*?)</title>', html)
        if title_match:
            title = title_match.group(1).strip()

    # Parse authors from BibTeX "author" field: "Last, First and Last, First"
    authors = []
    author_str = fields.get("author", "")
    if author_str:
        for auth in re.split(r'\s+and\s+', author_str):
            auth = auth.strip()
            if "," in auth:
                parts = auth.split(",", 1)
                family = parts[0].strip()
                given = parts[1].strip()
            else:
                parts = auth.rsplit(" ", 1)
                if len(parts) == 2:
                    given, family = parts
                else:
                    given, family = "", parts[0]
            authors.append({"given": given, "family": family})

    if not authors:
        logger.warning(f"  No authors found for {article['url']}")
        return None

    doi = fields.get("doi", "")
    year = fields.get("year", article["year"])

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "venue_url": article["url"],
    }


def main():
    out_path = LEGACY_DIR / "distill-legacy.jsonl.gz"
    if out_path.exists():
        logger.info(f"{out_path.name} already exists — skipping (delete to rebuild)")
        return

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering Distill articles...")
    articles = discover_articles()
    if not articles:
        logger.error("No articles found")
        return
    logger.info(f"Found {len(articles)} articles")

    papers = []
    bibtex_keys = []

    for i, article in enumerate(articles):
        logger.info(f"  [{i+1}/{len(articles)}] {article['year']}/{article['slug']}")
        meta = fetch_article_metadata(article)
        if not meta:
            continue

        bkey = make_bibtex_key(
            first_author_family=meta["authors"][0].get("family", ""),
            year=meta["year"],
            venue="distill",
            title=meta["title"],
        )

        paper = {
            "bibtex_key": bkey,
            "title": meta["title"],
            "authors": meta["authors"],
            "year": meta["year"],
            "venue": "distill",
            "venue_name": "Distill",
            "volume": "",
            "number": "",
            "pages": "",
            "abstract": "",
            "pdf_url": "",
            "venue_url": meta["venue_url"],
            "doi": meta["doi"],
            "openreview_url": "",
            "code_url": "",
            "source": "distill.pub",
            "source_id": meta["doi"],
        }
        papers.append(paper)
        bibtex_keys.append(bkey)
        time.sleep(0.5)  # be polite

    if not papers:
        logger.warning("No papers collected")
        return

    # Resolve BibTeX key collisions
    resolved_keys = resolve_bibtex_collisions(bibtex_keys)
    for paper, key in zip(papers, resolved_keys):
        paper["bibtex_key"] = key

    # Normalize and write
    write_legacy(out_path, [normalize_paper(p) for p in papers])

    years = sorted(set(p["year"] for p in papers))
    logger.info(f"\nWrote {len(papers)} Distill papers to {out_path.name}")
    logger.info(f"Year range: {years[0]}–{years[-1]}")

    print(
        "\nDone. Next steps:\n"
        "  python scripts/enrich_legacy.py --s2-api-key KEY --venue distill\n"
        "  python scripts/build_content.py"
    )


if __name__ == "__main__":
    main()
