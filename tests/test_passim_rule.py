"""The passim merge rule and passim's span merging (no engine)."""

from __future__ import annotations

import numpy as np
import polars as pl

from textblast.defrag import DefragOptions, defragment, passim_links
from textblast.export import merge_spans


def test_passim_cross_role_links() -> None:
    assert passim_links([(0, 100)], [(50, 150)], 0.9, 0.5) == [((0, 100), (50, 150))]  # 50 of 100: at 0.5
    assert passim_links([(0, 100)], [(51, 150)], 0.9, 0.5) == []  # 49 of 100
    assert passim_links([(0, 100)], [(10, 60), (20, 100)], 0.9, 0.5) == [((0, 100), (20, 100))]  # the best
    assert passim_links([], [(0, 100), (0, 95)], 0.9, 0.5) == []  # destination spans never link each other


def test_passim_same_role_links() -> None:
    # sorted (−length): (0, 100), (5, 95), (0, 50). (5, 95) lies inside (0, 100): linked. (0, 50): scanning back from
    # (5, 95), which it overlaps by 45 of 50 = 0.9: linked to it, not to (0, 100), and only once
    links = passim_links([(0, 100), (5, 95), (0, 50)], [], 0.9, 0.5)
    assert links == [((5, 95), (0, 100)), ((0, 50), (5, 95))]
    assert passim_links([(0, 100), (0, 89)], [], 0.9, 0.5) == [((0, 89), (0, 100))]
    assert passim_links([(0, 100), (20, 100)], [], 0.9, 0.5) == [((20, 100), (0, 100))]
    assert passim_links([(0, 100), (15, 110)], [], 0.9, 0.5) == []  # 85 of 95 < 0.9


def test_passim_rule_groups_raw_pieces() -> None:
    # doc 0 is doc1 of two hits: spans (0, 100) and (5, 95), linked by the same-role rule; doc 1 and doc 2 untouched
    frame = pl.DataFrame({
        "hsp_id": [0, 1], "doc1": [0, 0], "doc2": [1, 2], "start1": [0, 5], "end1": [100, 95],
        "start2": [0, 0], "end2": [100, 90], "enc_start1": [0, 5], "enc_end1": [100, 95], "enc_start2": [0, 0],
        "enc_end2": [100, 90], "pident": [90.0, 90.0], "length": [100, 90], "score": [9, 9],
        "bitscore": [1.0, 1.0], "evalue": [1e-5, 1e-5],
    })
    pieces, piece_map, pairs = defragment(frame, "hsp_id", DefragOptions(rule="passim"))
    doc0 = pieces.filter(pl.col("doc") == 0)
    assert doc0.height == 1 and (doc0["start"][0], doc0["end"][0]) == (0, 100) and doc0["n_raw_pieces"][0] == 2
    assert pairs.height == 2


def test_merge_spans_as_passim() -> None:
    start, end, back = merge_spans(np.array([50, 0, 10, 200]), np.array([60, 20, 55, 210]))
    assert start.tolist() == [0, 200] and end.tolist() == [60, 210] and back.tolist() == [0, 0, 0, 1]
    touching, _, _ = merge_spans(np.array([0, 20]), np.array([20, 30]))  # passim: begin < curEnd merges
    assert touching.tolist() == [0, 20]
