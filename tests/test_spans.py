"""Span maps: 1:1, 1:n, n:1, deletions, general edits, and composition across steps."""

from __future__ import annotations

import numpy as np
import pytest

from textblast.normalize import Spanned

from .helpers import pairs


def spans(doc: Spanned) -> list[tuple[str, int, int]]:
    return list(zip(doc.text, doc.start.tolist(), doc.end.tolist(), strict=True))


def test_identity() -> None:
    doc = Spanned.from_surface("abc")
    assert spans(doc) == [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)]
    assert doc.is_identity(3) and not doc.is_identity(4)


def test_keep_drops_characters() -> None:
    doc = Spanned.from_surface("ahb").keep(np.array([True, False, True]))
    assert spans(doc) == [("a", 0, 1), ("b", 2, 3)]


def test_merge_pairs_is_n_to_1() -> None:
    doc = Spanned.from_surface("xaey").merge_pairs(np.array([1]), "e")
    assert spans(doc) == [("x", 0, 1), ("e", 1, 3), ("y", 3, 4)]


def test_splice_same_length_keeps_positions() -> None:
    doc = Spanned.from_surface("abcd").splice([(1, 3, "XY")])
    assert spans(doc) == [("a", 0, 1), ("X", 1, 2), ("Y", 2, 3), ("d", 3, 4)]


def test_splice_expansion_and_contraction_share_the_span() -> None:
    expanded = Spanned.from_surface("aæb").splice([(1, 2, "ae")])
    assert spans(expanded) == [("a", 0, 1), ("a", 1, 2), ("e", 1, 2), ("b", 2, 3)]
    contracted = Spanned.from_surface("a[...]b").splice([(1, 6, "|")])
    assert spans(contracted) == [("a", 0, 1), ("|", 1, 6), ("b", 6, 7)]


def test_splice_rejects_overlaps() -> None:
    with pytest.raises(ValueError):
        Spanned.from_surface("abcd").splice([(0, 2, "x"), (1, 3, "y")])


def test_composition_through_several_steps() -> None:
    """æ → ae (1:n), then ae → e (n:1), then h deleted: spans still point at the surface."""
    assert pairs("latin-broad", "Hæc") == [("e", "æ"), ("c", "c")]
    assert pairs("latin-broad", "ahe") == [("e", "ahe")]  # h deleted, a + e contracted across it
    assert pairs("latin", "cæli") == [("c", "c"), ("a", "æ"), ("e", "æ"), ("l", "l"), ("i", "i")]
