#!/usr/bin/env python3
"""Convert HTML formatting tags in abstracts to LaTeX.

This script converts common HTML tags used in mathematical notation to LaTeX:
  - <sub>text</sub> → $_{text}$
  - <sup>text</sup> → $^{text}$
  - <i>text</i> → \\textit{text}
  - <b>text</b> → \\textbf{text}
  - <em>text</em> → \\textit{text}
  - <strong>text</strong> → \\textbf{text}

Run on all venues:
    python scripts/patch_html_to_latex.py

Single venue, dry-run to see what would change:
    python scripts/patch_html_to_latex.py --venue icml --dry-run

Only process papers with a specific source:
    python scripts/patch_html_to_latex.py --source dblp+s2 --dry-run
"""

import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utils import LEGACY_DIR, PAPERS_DIR

logger = logging.getLogger(__name__)


def html_to_latex(text: str) -> str:
    """Convert common HTML formatting tags to LaTeX.

    Preserves nested tags and handles common mathematical notation.
    """
    if not text or not isinstance(text, str):
        return text

    # Convert subscripts: <sub>1</sub> → $_{1}$
    text = re.sub(r'<sub>(.*?)</sub>', r'$_{\1}$', text)

    # Convert superscripts: <sup>2</sup> → $^{2}$
    text = re.sub(r'<sup>(.*?)</sup>', r'$^{\1}$', text)

    # Convert emphasis/italic: <i>text</i>, <em>text</em> → \textit{text}
    text = re.sub(r'<(?:i|em)>(.*?)</(?:i|em)>', r'\\textit{\1}', text)

    # Convert bold: <b>text</b>, <strong>text</strong> → \textbf{text}
    text = re.sub(r'<(?:b|strong)>(.*?)</(?:b|strong)>', r'\\textbf{\1}', text)

    return text


def process_papers(
    papers: list[dict],
    *,
    source: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Convert HTML to LaTeX in abstracts.

    Returns stats: {changed: int, abstracts_with_html: int}
    """
    changed = 0
    abstracts_with_html = 0

    for paper in papers:
        abstract = paper.get("abstract", "")
        if not abstract or not isinstance(abstract, str):
            continue

        # Filter by source if specified
        if source and paper.get("source") != source:
            continue

        # Check if abstract has HTML tags
        if not re.search(r'<(?:sub|sup|i|b|em|strong)>', abstract):
            continue

        abstracts_with_html += 1
        new_abstract = html_to_latex(abstract)

        if new_abstract != abstract:
            if not dry_run:
                paper["abstract"] = new_abstract
            changed += 1
            logger.debug(f"  Changed: {paper.get('title', 'Unknown')[:60]}")

    return {"changed": changed, "abstracts_with_html": abstracts_with_html}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert HTML formatting tags in abstracts to LaTeX"
    )
    ap.add_argument(
        "--venue",
        help="Process a single venue (e.g., 'icml', 'ijcai'); omit for all",
    )
    ap.add_argument(
        "--source",
        help="Only process papers with this source (e.g., 'dblp+s2')",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Process legacy data
    legacy_files = []
    if args.venue:
        pattern = f"{args.venue}-legacy.jsonl.gz"
        matches = list(LEGACY_DIR.glob(pattern))
        if not matches:
            logger.error(f"No legacy files matching: {pattern}")
            sys.exit(1)
        legacy_files = matches
    else:
        legacy_files = sorted(LEGACY_DIR.glob("*-legacy.jsonl.gz"))

    grand_total = 0
    grand_changed = 0

    for legacy_path in legacy_files:
        # Read papers
        papers: list[dict] = []
        with gzip.open(legacy_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(json.loads(line))

        logger.info(f"\n{legacy_path.name}: {len(papers)} papers")

        # Process
        stats = process_papers(papers, source=args.source, dry_run=args.dry_run)

        mode = "[DRY RUN] " if args.dry_run else ""
        logger.info(
            f"  {mode}Found {stats['abstracts_with_html']} abstracts with HTML tags, "
            f"converting {stats['changed']}"
        )

        if not args.dry_run and stats["changed"] > 0:
            # Write back atomically
            tmp_path = legacy_path.with_suffix(".jsonl.gz.tmp")
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                for p in papers:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            tmp_path.replace(legacy_path)
            logger.info(f"  Written {stats['changed']} changes to {legacy_path.name}")

        grand_total += stats["abstracts_with_html"]
        grand_changed += stats["changed"]

    logger.info(
        f"\n--- Summary ---\n"
        f"Total abstracts with HTML tags: {grand_total}\n"
        f"Total converted: {grand_changed}"
    )


if __name__ == "__main__":
    main()
