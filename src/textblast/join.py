"""Joining nearby hits of a document pair into one hit.

Two hits of the same pair are joined when they are consecutive in both documents (no other hit of the pair starts
between them), in the same order in both, and the gap between them is at most ``gap`` letters in both documents.
Gaps are measured in encoded letters, so spaces and punctuation do not count; overlapping neighbours (a negative gap)
are joined too, unless they are a repeat: by default no two hits are joined that overlap by more than half of the
shorter one in either document (``max_overlap``). Such hits are usually one phrase matched twice, not one borrowing.
Joining is transitive along a chain. Unit boundaries are ignored.

Joining is off by default: raw hits stay the unit. ``join_map`` is written on every
run as an annotation (raw hit → the group it would be joined into, and the group's size), so a reader sees which
hits continue each other. With ``enabled`` the joined hits are also built as a view (``hsps_joined``, one row per
group; a hit without a partner is its own group). Nothing is deleted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# A small allowance so that the ratio safeguard does not block joints whose gaps are both short.
RATIO_SLACK = 20


@dataclass
class JoinOptions:
    enabled: bool = False  # also build the joined view, tables/hsps_joined.parquet
    gap: int = 500  # letters, in both documents (about 600 characters)
    max_overlap: float | None = 0.5  # repeat guard: no joint if the hits overlap by more than this share of the
    #   shorter one in either document; None = off
    max_drift: int | None = None  # |gap1 − gap2| ≤ max_drift; off by default (drift did not separate cases)
    min_gap_ratio: float | None = None  # (shorter + 20) / (longer + 20) ≥ ratio; off by default (insertion guard)


class JoinError(ValueError):
    """Invalid joining options."""


def check_options(options: JoinOptions) -> None:
    if options.gap < 0:
        raise JoinError("join.gap must be ≥ 0 letters")
    if options.max_drift is not None and options.max_drift < 0:
        raise JoinError("join.max_drift must be ≥ 0 letters")
    if options.min_gap_ratio is not None and not 0 <= options.min_gap_ratio <= 1:
        raise JoinError("join.min_gap_ratio must lie between 0 and 1")
    if options.max_overlap is not None and not 0 <= options.max_overlap <= 1:
        raise JoinError("join.max_overlap must lie between 0 and 1")


def overlap_share(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """For each hit after the first: its overlap with the hit before it, as a share of the shorter of the two
    (0 for the first; spans ``[s, e)`` in one document)."""
    share = np.zeros(len(s), dtype=np.float64)
    if len(s) > 1:
        overlap = np.maximum(np.minimum(e[1:], e[:-1]) - np.maximum(s[1:], s[:-1]), 0)
        shorter = np.maximum(np.minimum(e[1:] - s[1:], e[:-1] - s[:-1]), 1)
        share[1:] = overlap / shorter
    return share


def joints(hits: pl.DataFrame, options: JoinOptions) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """``(order, joined, gaps)``: the hits sorted by pair and position in document 1, whether each is joined to the
    one before it, and the larger of the two gaps to it (letters; 0 for the first hit of a pair)."""
    order = (
        hits.select("hsp_id", "doc1", "doc2", "enc_start1", "enc_end1", "enc_start2", "enc_end2")
        .sort("doc1", "doc2", "enc_start2", "enc_end2", "hsp_id")
        .with_row_index("_rank2")
        .sort("doc1", "doc2", "enc_start1", "enc_end1", "hsp_id")
    )
    if order.height == 0:
        return order, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    doc1, doc2 = order["doc1"].to_numpy(), order["doc2"].to_numpy()
    rank2 = order["_rank2"].to_numpy().astype(np.int64)
    s1, e1 = order["enc_start1"].to_numpy().astype(np.int64), order["enc_end1"].to_numpy().astype(np.int64)
    s2, e2 = order["enc_start2"].to_numpy().astype(np.int64), order["enc_end2"].to_numpy().astype(np.int64)
    same_pair = np.zeros(len(order), dtype=bool)
    same_pair[1:] = (doc1[1:] == doc1[:-1]) & (doc2[1:] == doc2[:-1])
    gap1 = np.zeros(len(order), dtype=np.int64)
    gap2 = np.zeros(len(order), dtype=np.int64)
    gap1[1:] = s1[1:] - e1[:-1]
    gap2[1:] = s2[1:] - e2[:-1]
    # consecutive in document 2 as well, and after the previous hit there: the same order in both documents
    consecutive = np.zeros(len(order), dtype=bool)
    consecutive[1:] = rank2[1:] == rank2[:-1] + 1
    ok = same_pair & consecutive & (gap1 <= options.gap) & (gap2 <= options.gap)
    if options.max_drift is not None:
        ok &= np.abs(gap1 - gap2) <= options.max_drift
    if options.min_gap_ratio is not None:
        short = np.minimum(np.maximum(gap1, 0), np.maximum(gap2, 0))
        long = np.maximum(np.maximum(gap1, 0), np.maximum(gap2, 0))
        ok &= (short + RATIO_SLACK) >= options.min_gap_ratio * (long + RATIO_SLACK)
    if options.max_overlap is not None:
        ok &= (overlap_share(s1, e1) <= options.max_overlap) & (overlap_share(s2, e2) <= options.max_overlap)
    return order.drop("_rank2"), ok, np.where(same_pair, np.maximum(gap1, gap2), 0)


def join_map_for(hits: pl.DataFrame, options: JoinOptions) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(join_map, order)``: each raw hit's group (``joined_id``) and the group's size (``n_members``), and the
    hits in joining order with the gap to the hit before (``_gap``; the smallest integer when not joined)."""
    check_options(options)
    order, ok, gaps = joints(hits, options)
    joined_id = np.cumsum(~ok) - 1 if len(ok) else np.zeros(0, dtype=np.int64)
    joint_gap = np.where(ok, gaps, np.iinfo(np.int64).min)
    order = order.select("hsp_id").with_columns(
        pl.Series("joined_id", joined_id.astype(np.int64)), pl.Series("_gap", joint_gap)
    )
    join_map = order.select("hsp_id", "joined_id").with_columns(
        pl.len().over("joined_id").cast(pl.Int32).alias("n_members")).sort("hsp_id")
    return join_map, order


def join_hits(hits: pl.DataFrame, options: JoinOptions) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """``(hsps_joined, join_map, stats)`` for the raw hits (``hsps``)."""
    join_map, order = join_map_for(hits, options)
    members = hits.join(order, on="hsp_id")
    positives = (pl.col("ppos") * pl.col("length") / 100) if "ppos" in hits.columns else pl.col("nident")
    joined = (
        members.with_columns(positives.alias("_positives"))
        .group_by("joined_id")
        .agg(
            pl.col("doc1").first(), pl.col("doc2").first(),
            pl.col("start1").min(), pl.col("end1").max(), pl.col("start2").min(), pl.col("end2").max(),
            pl.col("enc_start1").min(), pl.col("enc_end1").max(), pl.col("enc_start2").min(),
            pl.col("enc_end2").max(),
            pl.col("length").sum(), pl.col("nident").sum(), pl.col("mismatch").sum(), pl.col("gapopen").sum(),
            pl.col("gaps").sum(), pl.col("score").sum(), pl.col("bitscore").sum(), pl.col("evalue").min(),
            pl.col("_positives").sum(), pl.len().alias("n_members"),
            pl.col("_gap").max().alias("max_gap"),
            pl.col("hsp_id").min().alias("first_hsp_id"),
        )
        .sort("joined_id")
    )
    joined = joined.with_columns(
        (100 * pl.col("nident") / pl.col("length")).cast(pl.Float32).alias("pident"),
        (100 * pl.col("_positives") / pl.col("length")).cast(pl.Float32).alias("ppos"),
        pl.when(pl.col("n_members") > 1).then(pl.col("max_gap")).otherwise(None).cast(pl.Int32).alias("max_gap"),
    ).drop("_positives")
    int32 = ["doc1", "doc2", "n_members"]
    joined = joined.with_columns([pl.col(c).cast(pl.Int32) for c in int32]).select(
        "joined_id", "doc1", "doc2", "start1", "end1", "start2", "end2", "enc_start1", "enc_end1", "enc_start2",
        "enc_end2", "length", "nident", "mismatch", "gapopen", "gaps", "pident", "ppos", "score", "bitscore",
        "evalue", "n_members", "max_gap", "first_hsp_id",
    )
    return joined, join_map, join_stats(join_map)


def join_stats(join_map: pl.DataFrame) -> dict[str, int]:
    groups = join_map["joined_id"].n_unique() if join_map.height else 0
    return {"hits": join_map.height, "joined_hits": groups,
            "hits_in_joints": int((join_map["n_members"] > 1).sum())}


def crossed_neighbours(hits: pl.DataFrame, gap: int) -> int:
    """Neighbours within ``gap`` letters in both documents whose order is crossed (a transposition): consecutive in
    document 1, with the second hit starting before the first in document 2."""
    order = hits.select("doc1", "doc2", "enc_start1", "enc_end1", "enc_start2", "enc_end2", "hsp_id").sort(
        "doc1", "doc2", "enc_start1", "enc_end1", "hsp_id")
    if order.height < 2:
        return 0
    d1, d2 = order["doc1"].to_numpy(), order["doc2"].to_numpy()
    s1, e1 = order["enc_start1"].to_numpy().astype(np.int64), order["enc_end1"].to_numpy().astype(np.int64)
    s2, e2 = order["enc_start2"].to_numpy().astype(np.int64), order["enc_end2"].to_numpy().astype(np.int64)
    same = (d1[1:] == d1[:-1]) & (d2[1:] == d2[:-1])
    near1 = (s1[1:] - e1[:-1]) <= gap
    crossed = s2[1:] < s2[:-1]
    near2 = (s2[:-1] - e2[1:]) <= gap  # the second hit ends shortly before the first starts in document 2
    return int((same & near1 & crossed & near2).sum())
