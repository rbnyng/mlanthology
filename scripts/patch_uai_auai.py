#!/usr/bin/env python3
"""Patch UAI 2014–2018 papers with PDF links from auai.org proceedings pages.

The auai.org website hosts individual paper PDFs for UAI 2014–2018.  This
script scrapes the accepted-papers / proceedings page for each year, extracts
(title → pdf_url) pairs, matches them against uai-legacy.jsonl.gz by
normalised title, and fills in any missing pdf_url fields.

Pages (one per year):
  2014 https://www.auai.org/uai2014/acceptedPapers.shtml
  2015 https://www.auai.org/uai2015/proceedings.shtml
  2016 https://www.auai.org/uai2016/proceedings.php
  2017 https://www.auai.org/uai2017/accepted.php
  2018 https://www.auai.org/uai2018/accepted.php

Usage::

    python scripts/patch_uai_auai.py
    python scripts/patch_uai_auai.py --dry-run
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, clean_html, fuzzy_lookup, make_session, normalize_title, read_legacy, write_legacy

logger = logging.getLogger(__name__)

SOURCES = [
    {
        "year": "2014",
        "url": "https://www.auai.org/uai2014/acceptedPapers.shtml",
        "base": "https://www.auai.org/uai2014/",
        "format": "2014",
    },
    {
        "year": "2015",
        "url": "https://www.auai.org/uai2015/proceedings.shtml",
        "base": "https://www.auai.org/uai2015/",
        "format": "2015_2016",
    },
    {
        "year": "2016",
        "url": "https://www.auai.org/uai2016/proceedings.php",
        "base": "https://www.auai.org/uai2016/",
        "format": "2015_2016",
    },
    {
        "year": "2017",
        "url": "https://www.auai.org/uai2017/accepted.php",
        "base": "",  # absolute URLs in HTML
        "format": "2017_2018",
    },
    {
        "year": "2018",
        "url": "https://www.auai.org/uai2018/accepted.php",
        "base": "",  # absolute URLs in HTML
        "format": "2017_2018",
    },
]


def _parse_2014(html: str, base: str) -> list[tuple[str, str]]:
    """
    Format::

        <b>ID: 72</b>&nbsp-&nbsp
        <a ... href="proceedings/individuals/72.pdf">Download</a>
        <br/>Title: Sequential Bayesian Optimisation...<br/>
    """
    results = []
    for m in re.finditer(
        r'href="(proceedings/individuals/\d+\.pdf)"[^>]*>\s*Download\s*</a>'
        r"<br/>\s*Title:\s*([^<\r\n]+)",
        html,
    ):
        pdf_url = base + m.group(1)
        title = clean_html(m.group(2))
        if title:
            results.append((title, pdf_url))
    return results


def _parse_2015_2016(html: str, base: str) -> list[tuple[str, str]]:
    """
    Format (both years)::

        <tr>
          <td ...><b>ID: N</b> <a ... href="proceedings/papers/N.pdf">(pdf)</a></td>
          <td><div class="collapse" tabindex=1><b>Title</b> +<div>abstract</div>
          </div><i>Authors</i></td>
        </tr>
    """
    results = []
    for row in re.split(r"<tr[>\s]", html):
        pdf_m = re.search(r'href="(proceedings/papers/\d+\.pdf)"', row)
        title_m = re.search(
            r'class="collapse"[^>]*>\s*<b>(.*?)</b>', row, re.DOTALL
        )
        if pdf_m and title_m:
            pdf_url = base + pdf_m.group(1)
            title = clean_html(title_m.group(1))
            if title:
                results.append((title, pdf_url))
    return results


def _parse_2017_2018(html: str) -> list[tuple[str, str]]:
    """
    Format (both years)::

        <tr>
          <td ...><h5>ID: N<br/>
            <a ... href='http://auai.org/uai20XX/proceedings/papers/N.pdf'>link</a>
          </h5></td>
          <td><h4>Title</h4>Authors...</td>
        </tr>

    Award banners appear as a separate ``<h4>`` containing a ``<p>``, e.g.::

        <h4><p class="text-info">Best Student Paper...</p></h4>
        <h4>Actual Title</h4>
    """
    results = []
    for row in re.split(r"<tr[>\s]", html):
        pdf_m = re.search(
            r"href='(http://auai\.org/uai20\d\d/proceedings/papers/\d+\.pdf)'",
            row,
        )
        if not pdf_m:
            continue
        # Find the first <h4> that is NOT an award banner (no nested <p>)
        title = None
        for h4_content in re.findall(r"<h4>(.*?)</h4>", row, re.DOTALL):
            if "<p" not in h4_content and h4_content.strip():
                title = clean_html(h4_content)
                break
        if title:
            results.append((title, pdf_m.group(1)))
    return results


def parse_page(html: str, fmt: str, base: str) -> list[tuple[str, str]]:
    if fmt == "2014":
        return _parse_2014(html, base)
    if fmt == "2015_2016":
        return _parse_2015_2016(html, base)
    if fmt == "2017_2018":
        return _parse_2017_2018(html)
    raise ValueError(f"Unknown format: {fmt!r}")


UAI_LEGACY_FILE = LEGACY_DIR / "uai-legacy.jsonl.gz"


def run(*, dry_run: bool = False) -> None:
    papers = read_legacy(UAI_LEGACY_FILE)
    logger.info("Loaded %d UAI papers", len(papers))

    session = make_session(retries=3, backoff_factor=1.0)
    session.headers.update({"User-Agent": "mlanthology-enrichment/1.0 (research bot)"})

    # Fetch and parse all source pages
    # year → {normalised_title: pdf_url}
    year_index: dict[str, dict[str, str]] = {}

    for i, src in enumerate(SOURCES):
        yr = src["year"]
        logger.info("[%d/%d] Fetching UAI %s …", i + 1, len(SOURCES), yr)
        try:
            resp = session.get(src["url"], timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("  Failed: %s", exc)
            continue

        pairs = parse_page(resp.text, src["format"], src["base"])
        logger.info("  %d papers parsed", len(pairs))

        index: dict[str, str] = {}
        for title, pdf_url in pairs:
            norm = normalize_title(title)
            index[norm] = pdf_url
        year_index[yr] = index

        if i < len(SOURCES) - 1:
            time.sleep(1.0)

    # Match and patch
    n_added = n_skipped_has = n_unmatched = 0

    for paper in papers:
        yr = paper.get("year", "")
        if yr not in year_index:
            continue

        existing = paper.get("pdf_url", "").strip()
        if existing:
            n_skipped_has += 1
            continue

        pdf_url = fuzzy_lookup(paper["title"], year_index[yr])
        if pdf_url is None:
            n_unmatched += 1
            logger.debug("  UNMATCHED [%s] %s", yr, paper["title"][:70])
            continue

        n_added += 1
        if dry_run:
            logger.debug("  DRY-RUN [%s] %s → %s", yr, paper["title"][:60], pdf_url)
        else:
            paper["pdf_url"] = pdf_url

    logger.info(
        "Results: %d added, %d already had PDF (skipped), %d unmatched",
        n_added, n_skipped_has, n_unmatched,
    )

    if dry_run:
        logger.info("Dry-run — nothing written")
        return

    if n_added == 0:
        logger.info("No changes — not writing file")
        return

    write_legacy(UAI_LEGACY_FILE, papers, atomic=True)
    logger.info("Written back to %s", UAI_LEGACY_FILE.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.dry_run else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
