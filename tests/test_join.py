"""Joining nearby hits of a pair (no engine)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from textblast.join import JoinError, JoinOptions, crossed_neighbours, join_hits, join_map_for


def hits(rows: list[tuple[int, int, int, int, int, int]]) -> pl.DataFrame:
    """Hits from (doc1, doc2, start1, end1, start2, end2) in letters; surface offsets = 2 × letters."""
    frame = pl.DataFrame(rows, schema=["doc1", "doc2", "enc_start1", "enc_end1", "enc_start2", "enc_end2"],
                         orient="row")
    n = frame.height
    length = (frame["enc_end1"] - frame["enc_start1"]).to_numpy()
    return frame.with_columns(
        pl.Series("hsp_id", np.arange(n)), *[(pl.col(f"enc_{p}{k}") * 2).alias(f"{p}{k}")
                                              for p in ("start", "end") for k in (1, 2)],
        pl.Series("length", length), pl.Series("nident", length - 1), pl.lit(1).alias("mismatch"),
        pl.lit(0).alias("gapopen"), pl.lit(0).alias("gaps"), pl.Series("pident", 100 * (length - 1) / length),
        pl.Series("ppos", 100 * (length - 1) / length), pl.Series("score", 9 * length),
        pl.Series("bitscore", 4.0 * length), pl.Series("evalue", 10.0 ** -np.arange(1, n + 1)),
    )


def groups(frame: pl.DataFrame, options: JoinOptions) -> list[list[int]]:
    _, join_map, _ = join_hits(frame, options)
    return sorted(join_map.group_by("joined_id").agg(pl.col("hsp_id").sort())["hsp_id"].to_list())


def test_gap_limit_in_either_document() -> None:
    options = JoinOptions(enabled=True, gap=500)
    at = hits([(0, 1, 0, 100, 0, 100), (0, 1, 600, 700, 600, 700)])  # gap 500 in both
    assert groups(at, options) == [[0, 1]]
    over1 = hits([(0, 1, 0, 100, 0, 100), (0, 1, 601, 700, 600, 700)])  # 501 in document 1
    over2 = hits([(0, 1, 0, 100, 0, 100), (0, 1, 600, 700, 601, 700)])  # 501 in document 2
    assert groups(over1, options) == [[0], [1]] and groups(over2, options) == [[0], [1]]


def test_crossed_order_and_a_hit_between_are_not_joined() -> None:
    options = JoinOptions(enabled=True, gap=500)
    crossed = hits([(0, 1, 0, 100, 300, 400), (0, 1, 150, 250, 0, 100)])
    assert groups(crossed, options) == [[0], [1]]
    assert crossed_neighbours(crossed, 500) == 1
    # hit 2 lies between hits 0 and 1 in document 2 only: 0 and 1 are not consecutive there
    between = hits([(0, 1, 0, 100, 0, 100), (0, 1, 150, 250, 400, 500), (0, 1, 900, 950, 200, 300)])
    assert groups(between, options) == [[0], [1], [2]]  # 0–1 not consecutive in document 2; 1–2 crossed


def test_overlaps_chains_and_other_pairs() -> None:
    options = JoinOptions(enabled=True, gap=50)
    frame = hits([(0, 1, 0, 100, 0, 100), (0, 1, 90, 200, 95, 210), (0, 1, 230, 300, 240, 300),
                  (0, 2, 0, 100, 0, 100), (0, 1, 1000, 1100, 1000, 1100)])
    assert groups(frame, options) == [[0, 1, 2], [3], [4]]  # a chain of three; another pair; a far hit


def test_ratio_and_drift_guards() -> None:
    frame = hits([(0, 1, 0, 100, 0, 100), (0, 1, 500, 600, 150, 250)])  # gaps 400 and 50: an insertion
    assert groups(frame, JoinOptions(enabled=True)) == [[0, 1]]
    assert groups(frame, JoinOptions(enabled=True, min_gap_ratio=0.6)) == [[0], [1]]
    assert groups(frame, JoinOptions(enabled=True, max_drift=300)) == [[0], [1]]
    with pytest.raises(JoinError):
        join_hits(frame, JoinOptions(enabled=True, min_gap_ratio=1.5))


def test_joined_statistics_and_map() -> None:
    frame = hits([(0, 1, 0, 100, 0, 100), (0, 1, 150, 250, 160, 260), (0, 1, 2000, 2100, 2000, 2100)])
    joined, join_map, stats = join_hits(frame, JoinOptions(enabled=True))
    first = joined.row(0, named=True)
    assert (first["start1"], first["end1"], first["start2"], first["end2"]) == (0, 500, 0, 520)
    assert (first["enc_start1"], first["enc_end1"]) == (0, 250)
    assert first["length"] == 200 and first["nident"] == 198 and first["score"] == 1800
    assert first["pident"] == pytest.approx(99.0) and first["evalue"] == pytest.approx(1e-2)
    assert first["n_members"] == 2 and first["max_gap"] == 60 and first["first_hsp_id"] == 0
    assert joined.row(1, named=True)["n_members"] == 1 and joined.row(1, named=True)["max_gap"] is None
    assert join_map.sort("hsp_id")["joined_id"].to_list() == [0, 0, 1]
    assert stats == {"hits": 3, "joined_hits": 2, "hits_in_joints": 2}


@pytest.mark.parametrize("side", [1, 2])
def test_repeat_guard_at_just_under_and_over_half_in_either_document(side: int) -> None:
    """A phrase matched twice: in one document the hits overlap; joined only up to half of the shorter hit."""
    options = JoinOptions(enabled=True, gap=500)

    def pair(overlap: int) -> pl.DataFrame:  # hits of 20 letters; the second 300 letters on in the other document
        same = (0, 20, 20 - overlap, 40 - overlap)
        far = (0, 20, 320, 340)
        a, b = (same, far) if side == 1 else (far, same)
        return hits([(0, 1, a[0], a[1], b[0], b[1]), (0, 1, a[2], a[3], b[2], b[3])])

    assert groups(pair(10), options) == [[0, 1]]  # exactly half: joined
    assert groups(pair(9), options) == [[0, 1]]
    assert groups(pair(11), options) == [[0], [1]]  # more than half: one phrase twice, not joined
    assert groups(pair(20), options) == [[0], [1]]  # the same span twice
    assert groups(pair(20), JoinOptions(enabled=True, gap=500, max_overlap=None)) == [[0, 1]]  # guard off


def test_repeat_guard_uses_the_shorter_hit_and_contained_hits() -> None:
    options = JoinOptions(enabled=True, gap=500)
    # a long hit and a short one inside it in document 1 (overlap = all of the short one), far apart in document 2
    inside = hits([(0, 1, 0, 200, 0, 200), (0, 1, 50, 70, 400, 420)])
    assert groups(inside, options) == [[0], [1]]
    with pytest.raises(JoinError):
        join_hits(inside, JoinOptions(max_overlap=1.5))


def test_join_map_is_built_without_the_joined_view() -> None:
    frame = hits([(0, 1, 0, 100, 0, 100), (0, 1, 150, 250, 150, 250), (0, 1, 2000, 2100, 2000, 2100)])
    join_map, _ = join_map_for(frame, JoinOptions())  # joining off: still the annotation
    assert join_map.columns == ["hsp_id", "joined_id", "n_members"]
    assert join_map.sort("hsp_id")["n_members"].to_list() == [2, 2, 1]
    assert join_map["joined_id"][0] == join_map["joined_id"][1] != join_map["joined_id"][2]
    _, from_view, stats = join_hits(frame, JoinOptions(enabled=True))
    assert from_view.equals(join_map)
    assert stats == {"hits": 3, "joined_hits": 2, "hits_in_joints": 2}

