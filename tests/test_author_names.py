"""Tests for author name cleaning and normalization."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.common import (
    _clean_raw_author_name,
    _fix_leading_hyphen_family,
    _fix_punctuation_only_fields,
    parse_author_name,
    slugify_author,
)


# ---------------------------------------------------------------------------
# _clean_raw_author_name: pre-parse string cleaning
# ---------------------------------------------------------------------------

class TestCleanRawAuthorName:
    def test_html_entities(self):
        assert _clean_raw_author_name("&#352;ingliar") == "\u0160ingliar"  # Šingliar

    def test_html_entity_multiple(self):
        assert _clean_raw_author_name("&#352;tefankovi&#269;") == "\u0160tefankovi\u010d"  # Štefankovič

    def test_asterisk_stripped(self):
        assert _clean_raw_author_name("Jonas Rauber*") == "Jonas Rauber"

    def test_asterisk_only(self):
        assert _clean_raw_author_name("*") == ""

    def test_phd_annotation(self):
        assert _clean_raw_author_name("Million Meshesha (PhD)") == "Million Meshesha"

    def test_he_him_annotation(self):
        assert _clean_raw_author_name("Marc Deisenroth (He/Him)") == "Marc Deisenroth"

    def test_she_her_annotation(self):
        assert _clean_raw_author_name("Jane Doe (She/Her)") == "Jane Doe"

    def test_nickname_stripped(self):
        assert _clean_raw_author_name("Jeong (Kate) Lee") == "Jeong Lee"

    def test_former_name_stripped(self):
        assert _clean_raw_author_name("Tali Dekel (Basha)") == "Tali Dekel"

    def test_bibtex_L(self):
        assert _clean_raw_author_name("\\Lącki") == "\u0141\u0105cki"  # Łącki

    def test_bibtex_O(self):
        assert _clean_raw_author_name("\\Oyvind") == "\u00D8yvind"  # Øyvind

    def test_normal_name_unchanged(self):
        assert _clean_raw_author_name("Ashish Vaswani") == "Ashish Vaswani"

    def test_empty(self):
        assert _clean_raw_author_name("") == ""

    def test_whitespace_collapsed(self):
        assert _clean_raw_author_name("  John   Doe  ") == "John Doe"

    def test_annotation_case_insensitive(self):
        assert _clean_raw_author_name("John Doe (phd)") == "John Doe"


# ---------------------------------------------------------------------------
# _fix_leading_hyphen_family: post-parse structural fix
# ---------------------------------------------------------------------------

class TestFixLeadingHyphenFamily:
    def test_orphaned_hyphen(self):
        result = _fix_leading_hyphen_family({
            "given": "Saeed Sharifi", "family": "-Malvajerdi"
        })
        assert result == {"given": "Saeed", "family": "Sharifi-Malvajerdi"}

    def test_no_hyphen_unchanged(self):
        author = {"given": "John", "family": "Doe"}
        assert _fix_leading_hyphen_family(author) == author

    def test_hyphen_inside_name_unchanged(self):
        author = {"given": "Wei", "family": "Chiu-Ma"}
        assert _fix_leading_hyphen_family(author) == author

    def test_empty_given_with_hyphen(self):
        """If given is empty, can't rejoin — leave unchanged."""
        author = {"given": "", "family": "-Malvajerdi"}
        assert _fix_leading_hyphen_family(author) == author


# ---------------------------------------------------------------------------
# _fix_punctuation_only_fields: post-parse junk detection
# ---------------------------------------------------------------------------

class TestFixPunctuationOnlyFields:
    def test_empty_parens_family(self):
        result = _fix_punctuation_only_fields({
            "given": "Lihua Xie", "family": "()"
        })
        assert result == {"given": "Lihua", "family": "Xie"}

    def test_dot_given(self):
        result = _fix_punctuation_only_fields({
            "given": ".", "family": "Deepanshi"
        })
        assert result == {"given": "", "family": "Deepanshi"}

    def test_email_in_family(self):
        result = _fix_punctuation_only_fields({
            "given": "", "family": "urtasun}@uber.com"
        })
        assert result == {"given": "", "family": ""}

    def test_curly_brace_family(self):
        result = _fix_punctuation_only_fields({
            "given": "Foo Bar", "family": "{baz"
        })
        assert result == {"given": "Foo", "family": "Bar"}

    def test_normal_name_unchanged(self):
        author = {"given": "John", "family": "Doe"}
        assert _fix_punctuation_only_fields(author) == author

    def test_both_junk(self):
        result = _fix_punctuation_only_fields({
            "given": "...", "family": "()"
        })
        assert result == {"given": "", "family": ""}


# ---------------------------------------------------------------------------
# Integration: full pipeline through normalize_paper's author chain
# ---------------------------------------------------------------------------

class TestAuthorNormalizationIntegration:
    """Test the cleaning pipeline end-to-end using parse + fixups."""

    def _normalize(self, given: str, family: str) -> dict:
        """Simulate the normalize_paper author pipeline."""
        from adapters.common import (
            _fix_misplaced_initial,
            _fix_single_letter_family,
            repair_mojibake,
        )
        cleaned = {
            "given": _clean_raw_author_name(repair_mojibake(given)),
            "family": _clean_raw_author_name(repair_mojibake(family)),
        }
        cleaned = _fix_misplaced_initial(cleaned)
        cleaned = _fix_single_letter_family(cleaned)
        cleaned = _fix_leading_hyphen_family(cleaned)
        cleaned = _fix_punctuation_only_fields(cleaned)
        cleaned["slug"] = slugify_author(cleaned)
        return cleaned

    def test_sharifi_malvajerdi(self):
        result = self._normalize("Saeed Sharifi", "-Malvajerdi")
        assert result["given"] == "Saeed"
        assert result["family"] == "Sharifi-Malvajerdi"
        assert result["slug"] == "sharifi-malvajerdi-saeed"

    def test_html_entity_singliar(self):
        result = self._normalize("Tom\u00e1\u0161", "&#352;ingliar")
        assert result["family"] == "\u0160ingliar"

    def test_asterisk_author(self):
        result = self._normalize("Wieland Brendel", "*")
        # * is stripped, family becomes empty, given re-parsed
        assert result["family"] == "Brendel"
        assert result["given"] == "Wieland"

    def test_phd_author(self):
        result = self._normalize("Million Meshesha", "(PhD)")
        assert result["family"] == "Meshesha"
        assert result["given"] == "Million"

    def test_bibtex_lacki(self):
        result = self._normalize("Jakub", "\\L\u0105cki")
        assert result["family"] == "\u0141\u0105cki"
        assert result["given"] == "Jakub"

    def test_kate_lee(self):
        result = self._normalize("Jeong", "(Kate) Lee")
        assert result["family"] == "Lee"
        assert result["given"] == "Jeong"

    def test_basha_dekel(self):
        result = self._normalize("Tali Dekel", "(Basha)")
        assert result["family"] == "Dekel"
        assert result["given"] == "Tali"

    def test_dot_deepanshi(self):
        result = self._normalize(".", "Deepanshi")
        assert result["family"] == "Deepanshi"
        assert result["given"] == ""

    def test_empty_parens_xie(self):
        result = self._normalize("Lihua Xie", "()")
        assert result["family"] == "Xie"
        assert result["given"] == "Lihua"
