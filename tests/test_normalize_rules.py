"""Every normalization rule, with expected text and spans."""

from __future__ import annotations

import pytest

from textblast.normalize import BARRIER

from .helpers import letters, pairs, run

# --------------------------------------------------------------------------- shared rules (latin and latin-broad)


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_case_folding_and_accents(profile: str) -> None:
    assert letters(profile, "PÁTER Noſter") == "pater noster"
    assert pairs(profile, "Á") == [("a", "Á")]
    # a combining accent belongs to its letter: the span covers both code points
    assert pairs(profile, "áb") == [("a", "á"), ("b", "b")]


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_ligatures_and_long_s(profile: str) -> None:
    # æ/œ/ę expand to ae/oe/ae (the broad profile then contracts them, see below)
    if profile == "latin":
        assert letters(profile, "æ œ ę") == "ae oe ae"
        assert pairs(profile, "æ") == [("a", "æ"), ("e", "æ")]
    assert letters(profile, "ſ ﬁ") == "s fi"


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_letter_folds(profile: str) -> None:
    """j → i, v → u (shared), w → u, y → i, k → c, upper and lower case, exact spans."""
    for source, target in (("j", "i"), ("J", "i"), ("v", "u"), ("V", "u"), ("w", "u"), ("W", "u"), ("y", "i"),
                           ("Y", "i"), ("k", "c"), ("K", "c")):
        assert pairs(profile, source) == [(target, source)]
    assert letters(profile, "Ludowicus") == "ludouicus"
    assert letters(profile, "Karolus kalendas") == "carolus calendas"


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_et_signs(profile: str) -> None:
    assert letters(profile, "pater&filius ⁊ sanctus") == "pater et filius et sanctus"
    assert pairs(profile, "&") == [("e", "&"), ("t", "&")]


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_numerals_punctuation_whitespace_separate_words(profile: str) -> None:
    doc, ctx = run(profile, "c. 12, de\tpenitentia")
    assert " ".join(doc.text.split()) == "c de penitentia"
    assert ctx.deleted["numeral"] == 2
    assert ctx.deleted["punctuation"] == 2
    assert ctx.deleted["whitespace"] == 3


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_soft_hyphen_disappears_without_splitting(profile: str) -> None:
    assert letters(profile, "domi­nus") == "dominus"


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_supplement_brackets(profile: str) -> None:
    """Round 5: < > inside a word disappear without breaking it; around a passage they separate words."""
    doc, ctx = run(profile, "do<mi>nus <et> pater")
    assert " ".join(doc.text.split()) == "dominus et pater"
    assert ctx.deleted["supplement_bracket"] == 2 and ctx.deleted["punctuation"] == 2
    assert not ctx.unsupported
    assert letters(profile, "x <some passage> z") == "x some passage z"


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_line_break_hyphen_and_dot_above(profile: str) -> None:
    doc, ctx = run(profile, "domi¬nus a˙b")
    assert " ".join(doc.text.split()) == "dominus a b"
    assert ctx.deleted["line_break_hyphen"] == 1 and ctx.deleted["punctuation"] == 1 and not ctx.unsupported


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_l_with_stroke(profile: str) -> None:
    """Round 5: a standalone ł means vel; attached to a word it is an abbreviation stroke on l."""
    assert letters(profile, "signa ł uerba") == "signa uel uerba"
    assert pairs(profile, "ł") == [("u", "ł"), ("e", "ł"), ("l", "ł")]
    assert letters(profile, "concił Ł") == "concil uel"


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_unresolved_abbreviation_marks_are_removed(profile: str) -> None:
    """Tilde and macron (precomposed or combining) usually mark abbreviations: the grapheme is removed and listed."""
    for text in ("dñs", "dñs", "dōs"):
        doc, ctx = run(profile, text)
        assert " ".join(doc.text.split()) == "d s"
        assert ctx.deleted["unresolved_abbreviation_mark"] == len(text) - 2
        assert sum(ctx.unsupported.values()) == 1


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_unsupported_letters_and_signs_are_listed(profile: str) -> None:
    doc, ctx = run(profile, "λόγος ꝑ °")
    assert doc.text.strip() == ""
    assert ctx.unsupported["ꝑ"] == 1 and ctx.unsupported["°"] == 1
    assert ctx.deleted["unsupported_letter"] == 6 and ctx.deleted["unsupported_sign"] == 1


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_declared_gaps_and_scaffolding_become_barriers(profile: str) -> None:
    text = "ante [lacuna] post [...] tum … et === PAGE 3 === fin"
    doc, ctx = run(profile, text)
    barriers = [(a, b) for c, a, b in zip(doc.text, doc.start.tolist(), doc.end.tolist(), strict=True)
                if c == BARRIER]
    assert [text[a:b] for a, b in barriers] == ["[lacuna]", "[...]", "…", "=== PAGE 3 ==="]
    assert ctx.deleted["declared_gap"] == len("[lacuna]") + len("[...]") + len("…")
    assert ctx.deleted["export_scaffolding"] == len("=== PAGE 3 ===")


def test_h_kept_in_latin() -> None:
    assert letters("latin", "Hic haec") == "hic haec"


# --------------------------------------------------------------------------- broad rules


def test_broad_deletes_h() -> None:
    doc, ctx = run("latin-broad", "Hic haec")
    assert " ".join(doc.text.split()) == "ic ec"
    assert ctx.deleted["spelling_h"] == 2


def test_broad_vowel_pairs() -> None:
    assert letters("latin-broad", "caelum poena æcclesia") == "celum pena ecclesia"
    assert pairs("latin-broad", "ae") == [("e", "ae")]
    # repeated until stable, as in the reference: a + oe → a + e → e
    assert letters("latin-broad", "aoe") == "e"
    # never across words
    assert letters("latin-broad", "ita est") == "ita est"


def test_broad_c_before_i_and_vowel() -> None:
    assert letters("latin-broad", "nuncius gracia tercio") == "nuntius gratia tertio"
    assert pairs("latin-broad", "cia")[0] == ("t", "c")
    assert letters("latin-broad", "dici") == "dici"  # i not followed by a vowel


def test_reference_example() -> None:
    """The example of normalization.md: "Hæc & uita" → "ec et uita"."""
    assert letters("latin-broad", "Hæc & uita") == "ec et uita"
