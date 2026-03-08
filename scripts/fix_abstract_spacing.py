#!/usr/bin/env python3
"""One-time migration: fix PDF text extraction spacing artifacts in abstracts.

Handles two categories:
1. Missing spaces after periods/commas (via regex in adapters.common)
2. Concatenated words from PDF extraction (via wordninja heuristic)

Usage:
    python scripts/fix_abstract_spacing.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

# ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.common import read_venue_json, write_venue_json, repair_abstract_spacing
from scripts.utils import read_legacy, write_legacy

try:
    import wordninja
except ImportError:
    wordninja = None
    print("WARNING: wordninja not installed, skipping concatenated-word fixes")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "papers"
LEGACY_DIR = REPO_ROOT / "data" / "legacy"
BACKLOG_DIR = REPO_ROOT / "data" / "backlog"
MISC_DIR = REPO_ROOT / "data" / "misc"

# Detect tokens with lowercase-to-uppercase transition that are NOT at
# a word boundary — these are potential concatenation artifacts
# e.g., "onEnglish", "superiorin" won't match this but we catch them differently
_LOWERCASE_UPPERCASE_RE = re.compile(r"[a-z][A-Z]")

# Common short words that get concatenated in PDF extraction
_SHORT_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                "has", "have", "in", "is", "it", "its", "of", "on", "or", "our",
                "the", "to", "was", "we", "with", "can", "do", "if", "no", "not",
                "so", "up", "but", "than", "that", "this", "also", "been", "both",
                "each", "into", "more", "much", "over", "such", "very", "when",
                "which", "while", "will"}


def _is_likely_method_name(token: str) -> bool:
    """Check if a token looks like a deliberate camelCase identifier."""
    # Count uppercase transitions
    transitions = len(re.findall(r"[a-z][A-Z]", token))
    # Method names tend to have multiple transitions or start with uppercase
    if token[0].isupper() and transitions >= 1:
        return True
    # Single transition at the start (e.g., "iPhone") - likely intentional
    if transitions == 1 and re.match(r"^[a-z]+[A-Z]", token):
        # Check if the lowercase prefix is a common short word
        m = re.match(r"^([a-z]+)[A-Z]", token)
        if m and m.group(1) in _SHORT_WORDS:
            return False  # "onEnglish" -> not a method name
        if len(m.group(1)) <= 2:
            return True  # "iPhone", "eTime" - likely intentional
    return False


def _fix_concatenated_words(text: str) -> str:
    """Use wordninja to fix concatenated words in abstract text."""
    if wordninja is None:
        return text

    tokens = text.split()
    fixed_tokens = []
    changed = False

    for token in tokens:
        fixed = _try_fix_token(token)
        if fixed != token:
            fixed_tokens.append(fixed)
            changed = True
        else:
            fixed_tokens.append(token)

    if changed:
        return " ".join(fixed_tokens)
    return text


def _try_fix_token(token: str) -> str:
    """Try to fix a single token that may be concatenated words."""
    # Handle hyphenated tokens: split on hyphens, fix each part, rejoin
    if "-" in token and any(c.isalpha() for c in token):
        parts = token.split("-")
        fixed_parts = [_try_fix_alpha_token(p) if p.isalpha() else p for p in parts]
        return "-".join(fixed_parts)

    if token.isalpha():
        return _try_fix_alpha_token(token)
    return token


def _try_fix_alpha_token(token: str) -> str:
    """Try to fix a purely alphabetic token."""
    if len(token) < 5 or not token.isalpha():
        return token

    # Skip PascalCase method names (e.g., TaskNorm, FeatureGan)
    if token[0].isupper() and _is_likely_method_name(token):
        return token

    # Skip intentional stylization where case transition is very early
    # e.g., "mEmorability", "rEwarding", "eBay"
    early_transition = re.match(r"^[a-z]{1,2}[A-Z]", token)
    if early_transition and early_transition.group(0)[:-1] not in _SHORT_WORDS:
        return token

    # Check for lowercase→uppercase transition (e.g., "onEnglish")
    has_case_transition = bool(_LOWERCASE_UPPERCASE_RE.search(token))

    # Only proceed with wordninja for tokens that have a case transition.
    # All-lowercase concatenations (e.g., "previoussingle") are too risky
    # to detect automatically — wordninja splits legitimate compound words
    # like "backscattered" or "furthermore".
    if not has_case_transition:
        return token

    # Try wordninja split
    parts = wordninja.split(token)
    if len(parts) <= 1 or len(parts) > 4:
        return token

    # Validate the split: recombined parts should equal the original
    recombined = "".join(parts)
    if recombined.lower() != token.lower():
        return token

    # Validate quality of split
    all_reasonable = all(len(p) >= 2 for p in parts)
    if not all_reasonable:
        return token

    # Build cased parts for validation
    pos = 0
    cased_parts = []
    for part in parts:
        cased_parts.append(token[pos:pos + len(part)])
        pos += len(part)

    # Reject if any part is all-uppercase (likely an acronym: KITTI, DINO, MALA)
    if any(len(cp) >= 2 and cp.isupper() for cp in cased_parts):
        return token

    # Reject if the token contains intentional mixed-case stylization
    # (multiple uppercase letters not at word boundaries)
    # e.g., "hashtAGs", "cUrriculum", "multImodal"
    upper_count = sum(1 for c in token if c.isupper())
    if upper_count >= 2:
        return token

    # Require exactly 2 parts: one common short word + one content word.
    # This ensures we're fixing "andGaussian" → "and Gaussian" but not
    # over-splitting "theMultimodal" → "the Multi modal".
    if len(parts) != 2:
        return token
    has_common = any(p.lower() in _SHORT_WORDS for p in parts)
    if not has_common:
        return token
    # The non-short part must be substantial (>= 4 chars)
    content_parts = [p for p in parts if p.lower() not in _SHORT_WORDS]
    if not all(len(p) >= 4 for p in content_parts):
        return token

    # Verify the case transition aligns with a split boundary
    boundaries = set()
    pos = 0
    for part in parts:
        boundaries.add(pos)
        pos += len(part)
    has_aligned_transition = False
    for i in range(len(token) - 1):
        if token[i].islower() and token[i + 1].isupper():
            if (i + 1) in boundaries:
                has_aligned_transition = True
                break
    if not has_aligned_transition:
        return token

    return " ".join(cased_parts)


def fix_abstract(text: str) -> str:
    """Apply all abstract spacing fixes."""
    if not text:
        return text
    # First pass: regex-based period/comma fixes
    text = repair_abstract_spacing(text)
    # Second pass: wordninja-based concatenated word fixes
    text = _fix_concatenated_words(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Fix abstract spacing artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report fixes without writing files")
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("*.json.gz"))
    print(f"Scanning {len(files)} venue files in {DATA_DIR}")

    total_fixed = 0
    total_papers = 0
    files_modified = 0

    for path in files:
        data = read_venue_json(path)
        venue = data["venue"]
        year = data["year"]
        papers = data["papers"]
        file_fixes = 0

        for paper in papers:
            total_papers += 1
            abstract = paper.get("abstract", "")
            if not abstract:
                continue

            fixed = fix_abstract(abstract)
            if fixed != abstract:
                file_fixes += 1
                if args.dry_run:
                    # Show a sample of what changed
                    key = paper.get("bibtex_key", "?")
                    # Find first difference
                    for i, (a, b) in enumerate(zip(abstract, fixed)):
                        if a != b:
                            start = max(0, i - 20)
                            end = min(len(abstract), i + 40)
                            print(f"  {key}: ...{abstract[start:end]}...")
                            print(f"  {' ' * len(key)}  ...{fixed[start:end]}...")
                            break
                else:
                    paper["abstract"] = fixed

        if file_fixes > 0:
            total_fixed += file_fixes
            files_modified += 1
            venue_label = f"{venue}-{year}"
            print(f"  {venue_label}: {file_fixes} abstracts fixed")

            if not args.dry_run:
                write_venue_json(venue, year, papers, DATA_DIR)

    # --- Legacy / backlog JSONL files ---
    for data_dir in [LEGACY_DIR, BACKLOG_DIR]:
        jsonl_files = sorted(data_dir.glob("*.jsonl.gz")) if data_dir.exists() else []
        if not jsonl_files:
            continue
        print(f"\nScanning {len(jsonl_files)} JSONL files in {data_dir}")
        for path in jsonl_files:
            papers = read_legacy(path)
            file_fixes = 0
            for paper in papers:
                total_papers += 1
                abstract = paper.get("abstract", "")
                if not abstract:
                    continue
                fixed = fix_abstract(abstract)
                if fixed != abstract:
                    file_fixes += 1
                    if not args.dry_run:
                        paper["abstract"] = fixed
            if file_fixes > 0:
                total_fixed += file_fixes
                files_modified += 1
                print(f"  {path.name}: {file_fixes} abstracts fixed")
                if not args.dry_run:
                    write_legacy(path, papers)

    # --- Misc JSON files ---
    misc_files = sorted(MISC_DIR.glob("*.json")) if MISC_DIR.exists() else []
    if misc_files:
        print(f"\nScanning {len(misc_files)} misc JSON files in {MISC_DIR}")
        for path in misc_files:
            import json as _json
            with open(path) as f:
                raw = _json.load(f)
            papers = raw if isinstance(raw, list) else raw.get("papers", [raw])
            file_fixes = 0
            for paper in papers:
                total_papers += 1
                abstract = paper.get("abstract", "")
                if not abstract:
                    continue
                fixed = fix_abstract(abstract)
                if fixed != abstract:
                    file_fixes += 1
                    if not args.dry_run:
                        paper["abstract"] = fixed
            if file_fixes > 0:
                total_fixed += file_fixes
                files_modified += 1
                print(f"  {path.name}: {file_fixes} abstracts fixed")
                if not args.dry_run:
                    with open(path, "w") as f:
                        _json.dump(raw, f, ensure_ascii=False, indent=2)

    action = "would fix" if args.dry_run else "fixed"
    print(f"\nDone: {action} {total_fixed} abstracts across {files_modified} files "
          f"({total_papers} papers scanned)")


if __name__ == "__main__":
    main()
