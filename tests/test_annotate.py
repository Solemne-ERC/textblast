"""Piece annotations: doc_degree and reference sets (no engine)."""

from __future__ import annotations

import polars as pl
import pytest

from textblast.annotate import doc_degree, reference_overlap, reference_set_members


def test_doc_degree_counts_distinct_partner_documents() -> None:
    pieces = pl.DataFrame({"piece_id": [0, 1, 2, 3], "doc": [0, 1, 2, 0]})
    pairs = pl.DataFrame({"piece1": [0, 0, 0], "piece2": [1, 2, 2], "doc1": [0, 0, 0], "doc2": [1, 2, 2]})
    assert doc_degree(pieces, pairs).tolist() == [2, 1, 1, 0]


def test_reference_overlap_and_set_members() -> None:
    documents = pl.DataFrame({"doc_idx": [0, 1, 2], "doc_id": ["a", "b", "bible:gen"]})
    member = reference_set_members(documents, "bible", "doc_id LIKE 'bible:%'")
    assert member.tolist() == [False, False, True]
    pieces = pl.DataFrame({"piece_id": [0, 1, 2], "doc": [0, 0, 1], "start": [0, 50, 0], "end": [100, 80, 10]})
    frame = pl.DataFrame({"doc1": [0, 0, 0, 0], "start1": [10, 40, 20, 90], "end1": [30, 60, 25, 95],
                          "doc2": [2, 2, 2, 1], "start2": [0] * 4, "end2": [5] * 4})
    # piece 0: [10, 30) ∪ [20, 25) ∪ [40, 60) = 40 of 100; piece 1: [50, 60) = 10 of 30; doc 1: no Bible hits
    assert reference_overlap(pieces, frame, member).tolist() == pytest.approx([0.4, 1 / 3, 0.0])
    with pytest.raises(ValueError, match="reference set 'x'"):
        reference_set_members(documents, "x", "doc_id LIKE")
