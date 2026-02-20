"""Tests for the data validation script."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_data import validate, Finding


def _make_paper(**overrides):
    """Create a minimal valid paper dict."""
    paper = {
        "bibtex_key": "smith2024icml-learning",
        "title": "Learning Representations for Downstream Tasks",
        "authors": [
            {"given": "John", "family": "Smith", "slug": "smith-john"},
        ],
        "year": "2024",
        "venue": "icml",
        "venue_type": "conference",
        "volume": "235",
        "pages": "1-10",
        "abstract": "We study representation learning.",
        "pdf_url": "https://example.com/paper.pdf",
        "venue_url": "https://example.com/paper",
        "doi": "",
        "openreview_url": "",
        "code_url": "",
        "source": "pmlr",
        "source_id": "v235/smith24a",
    }
    paper.update(overrides)
    return paper


KNOWN_VENUES = {"icml", "neurips", "iclr", "cvpr", "jmlr"}


class TestRequiredFields:
    def test_valid_paper_passes(self):
        papers = [("icml-2024.json.gz", _make_paper(), "primary")]
        findings = validate(papers, KNOWN_VENUES)
        # Only "tiny_file" is expected (1 paper in the file)
        checks = {f.check for f in findings}
        assert checks == set() or checks == {"tiny_file"}

    def test_missing_title(self):
        p = _make_paper()
        del p["title"]
        findings = validate([("test.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "missing_field" in checks

    def test_empty_venue(self):
        findings = validate(
            [("test.json.gz", _make_paper(venue=""), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "empty_required_field" in checks


class TestBibtexKey:
    def test_bad_chars(self):
        findings = validate(
            [("t.json.gz", _make_paper(bibtex_key="Smith2024!"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "bibtex_key_bad_chars" in checks

    def test_year_mismatch(self):
        findings = validate(
            [("t.json.gz", _make_paper(bibtex_key="smith2023icml-learning", year="2024"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "bibtex_key_year_mismatch" in checks

    def test_global_duplicates(self):
        p = _make_paper()
        papers = [
            ("a.json.gz", p, "primary"),
            ("b.json.gz", p, "primary"),
        ]
        findings = validate(papers, KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "global_duplicate_key" in checks


class TestTitleChecks:
    def test_short_title(self):
        findings = validate(
            [("t.json.gz", _make_paper(title="Hi"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "short_title" in checks

    def test_allcaps_title(self):
        findings = validate(
            [("t.json.gz", _make_paper(title="LEARNING REPRESENTATIONS FOR TASKS"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "allcaps_title" in checks

    def test_unbalanced_latex(self):
        findings = validate(
            [("t.json.gz", _make_paper(title="Using $x in the wild"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "unbalanced_latex" in checks

    def test_balanced_latex_ok(self):
        findings = validate(
            [("t.json.gz", _make_paper(title="Using $x$ in the wild and more"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "unbalanced_latex" not in checks


class TestAuthorChecks:
    def test_no_authors(self):
        findings = validate(
            [("t.json.gz", _make_paper(authors=[]), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "no_authors" in checks

    def test_missing_slug_primary(self):
        p = _make_paper(authors=[{"given": "John", "family": "Doe", "slug": ""}])
        findings = validate([("t.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "missing_slug" in checks

    def test_missing_slug_legacy_ignored(self):
        p = _make_paper(authors=[{"given": "John", "family": "Doe", "slug": ""}])
        findings = validate([("t.jsonl.gz", p, "legacy")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "missing_slug" not in checks

    def test_html_entity_in_author(self):
        p = _make_paper(authors=[{"given": "O&apos;Brien", "family": "Pat", "slug": "pat-obrien"}])
        findings = validate([("t.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "html_entity_in_author" in checks

    def test_asterisk_in_author(self):
        p = _make_paper(authors=[{"given": "John*", "family": "Doe", "slug": "doe-john"}])
        findings = validate([("t.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "asterisk_in_author" in checks

    def test_duplicate_author_slug(self):
        p = _make_paper(authors=[
            {"given": "John", "family": "Smith", "slug": "smith-john"},
            {"given": "John", "family": "Smith", "slug": "smith-john"},
        ])
        findings = validate([("t.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "duplicate_author_in_paper" in checks


class TestURLChecks:
    def test_bad_url(self):
        p = _make_paper(pdf_url="ftp://example.com/paper.pdf")
        findings = validate([("t.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "bad_url" in checks

    def test_space_in_url(self):
        p = _make_paper(pdf_url="https://example.com/my paper.pdf")
        findings = validate([("t.json.gz", p, "primary")], KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "space_in_url" in checks


class TestVenueChecks:
    def test_unknown_venue(self):
        findings = validate(
            [("t.json.gz", _make_paper(venue="fakevenue"), "primary")],
            KNOWN_VENUES,
        )
        checks = {f.check for f in findings}
        assert "unknown_venue" in checks


class TestDuplicateTitle:
    def test_duplicate_titles_flagged(self):
        p1 = _make_paper(bibtex_key="smith2024icml-learning")
        p2 = _make_paper(bibtex_key="jones2024icml-learning")
        papers = [
            ("icml-2024.json.gz", p1, "primary"),
            ("icml-2024.json.gz", p2, "primary"),
        ]
        findings = validate(papers, KNOWN_VENUES)
        checks = {f.check for f in findings}
        assert "duplicate_title" in checks
