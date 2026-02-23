"""Data loading for the Hugo content builder.

Loads papers from data/papers/*.json.gz, data/backlog/, and data/legacy/
into a unified list of paper dicts.
"""

from pathlib import Path

from adapters.common import read_venue_json
from scripts.utils import read_legacy

DATA_DIR: Path | None = None
BACKLOG_DIR: Path | None = None
LEGACY_DIR: Path | None = None


def _load_gzipped_jsonl(directory: Path) -> list[dict]:
    """Load all gzipped JSONL files from a directory."""
    papers: list[dict] = []
    if not directory.exists():
        return papers
    for gz_file in sorted(directory.glob("*.jsonl.gz")):
        papers.extend(read_legacy(gz_file))
    return papers


def load_all_papers(
    data_dir: Path,
    backlog_dir: Path,
    legacy_dir: Path,
) -> list[dict]:
    """Load all papers from gzipped JSON data files, backlog, and legacy archives."""
    papers: list[dict] = []
    for jf in sorted(data_dir.glob("*.json.gz")):
        data = read_venue_json(jf)
        if not data or "papers" not in data:
            continue
        papers.extend(data["papers"])

    json_count = len(list(data_dir.glob("*.json.gz")))

    # Load gzipped JSONL from backlog and legacy directories
    backlog_papers = _load_gzipped_jsonl(backlog_dir)
    legacy_papers = _load_gzipped_jsonl(legacy_dir)
    papers.extend(backlog_papers)
    papers.extend(legacy_papers)

    parts = [f"{json_count} data files"]
    if backlog_papers:
        parts.append(f"{len(backlog_papers)} backlog")
    if legacy_papers:
        parts.append(f"{len(legacy_papers)} legacy")
    print(f"  Loaded {len(papers)} papers from {', '.join(parts)}")
    return papers
