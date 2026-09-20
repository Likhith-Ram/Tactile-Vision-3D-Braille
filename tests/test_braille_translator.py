"""Unit tests for Liblouis Braille Translation service."""

from app.models.schemas import BrailleTranslation
from app.services.braille_translator import BrailleTranslator


def test_braille_translator_grade_1_basic() -> None:
    """Verifies Grade 1 Braille ASCII translation for lowercase letters and numbers."""
    translator = BrailleTranslator()
    result = translator.translate("hello 123")

    assert isinstance(result, BrailleTranslation)
    # 'hello' -> 'hello', '123' -> '#abc'
    assert "hello" in result.grade_1_ascii
    assert "#abc" in result.grade_1_ascii
    assert len(result.grade_1_unicode) > 0


def test_braille_translator_grade_1_capitalization() -> None:
    """Verifies capitalization prefix handling (, for single cap, ,, for all-caps)."""
    translator = BrailleTranslator()

    # Single capital
    res_single = translator.translate("Hello")
    assert res_single.grade_1_ascii.startswith(",h")

    # All caps word
    res_all_caps = translator.translate("HELLO")
    assert ",," in res_all_caps.grade_1_ascii or ",h,e,l,l,o" in res_all_caps.grade_1_ascii


def test_braille_translator_grade_2_contractions() -> None:
    """Verifies Grade 2 contracted Braille ASCII translation."""
    translator = BrailleTranslator()
    # 'the' -> '!', 'and' -> '&', 'for' -> '=', 'with' -> ')'
    res = translator.translate("the and for with")
    assert isinstance(res, BrailleTranslation)
    assert len(res.grade_2_ascii) > 0
    # In UEB ASCII: 'the' is '!', 'and' is '&', 'for' is '=', 'with' is ')'
    assert "!" in res.grade_2_ascii
    assert "&" in res.grade_2_ascii


def test_braille_ascii_to_unicode_mapping() -> None:
    """Verifies Braille ASCII translates directly into U+2800 Braille Patterns block."""
    translator = BrailleTranslator()

    # 'a' in Braille is dot 1 -> \u2801
    # 'b' in Braille is dots 1-2 -> \u2803
    # 'c' in Braille is dots 1-4 -> \u2809
    unicode_out = translator.ascii_to_unicode("abc")
    assert unicode_out == "\u2801\u2803\u2809"

    # Space -> \u2800 (blank Braille cell)
    assert translator.ascii_to_unicode(" ") == "\u2800"


def test_braille_translator_empty_string() -> None:
    """Verifies empty string produces empty translation safely."""
    translator = BrailleTranslator()
    res = translator.translate("")
    assert res.grade_1_ascii == ""
    assert res.grade_2_ascii == ""
    assert res.grade_1_unicode == ""
    assert res.grade_2_unicode == ""
