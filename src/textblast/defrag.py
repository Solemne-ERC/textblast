"""Defragmentation: hit sides → pieces (passages per document) → pairs of pieces .

Raw pieces are the distinct ``(doc, start, end)`` of the hit sides. A rule groups the raw pieces of each document;
a merged piece spans its group. Pairs link the pieces of the two sides of each hit; ``num_orig_links`` counts the
hits behind a pair. Every rule works on surface offsets.

Presets:
- ``etl``: ReceptionReader's rule (HPC-HD ETL, ``piece_id_mappings.ipynb``), reproduced exactly;
- ``node_similarity`` / ``node_similarity-compat``: textreuse-blast's clusterizer rule, bug fixed / with the bug;
- ``passim``: passim's ``linkSpans`` (``seriatim.py``), with ``src_overlap`` and ``dst_overlap``;
- a plugin by import path (``module:function``).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from .parallel import process_pool

PRESETS = ("etl", "node_similarity", "node_similarity-compat", "passim")


class DefragError(ValueError):
    """An unknown rule or a plugin returning an invalid grouping."""


@dataclass
class DefragOptions:
    rule: str = "etl"  # etl | node_similarity | node_similarity-compat | passim | module:function
    node_similarity: float = 0.9
    src_overlap: float = 0.9  # passim: same-role spans
    dst_overlap: float = 0.5  # passim: a source-side span and a destination-side span
    view: str = "hsps"  # hsps | clipped | joined


# --------------------------------------------------------------------------- raw pieces


def raw_pieces(hits: pl.DataFrame, id_col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(raw pieces, sides): distinct (doc, start, end) numbered in that order, and each hit side's raw piece."""
    sides = pl.concat([
        hits.select(pl.col(id_col), pl.lit(1, pl.Int8).alias("side"), pl.col("doc1").alias("doc"),
                    pl.col("start1").alias("start"), pl.col("end1").alias("end"),
                    pl.col("enc_start1").alias("enc_start"), pl.col("enc_end1").alias("enc_end")),
        hits.select(pl.col(id_col), pl.lit(2, pl.Int8).alias("side"), pl.col("doc2").alias("doc"),
                    pl.col("start2").alias("start"), pl.col("end2").alias("end"),
                    pl.col("enc_start2").alias("enc_start"), pl.col("enc_end2").alias("enc_end")),
    ])
    raw = (
        sides.group_by("doc", "start", "end")
        .agg(pl.col("enc_start").min(), pl.col("enc_end").max(), pl.len().alias("n_hsps"))
        .sort("doc", "start", "end")
        .with_row_index("raw_piece_id")
        .with_columns(pl.col("raw_piece_id").cast(pl.Int64), pl.col("n_hsps").cast(pl.Int32))
    )
    sides = sides.join(raw.select("doc", "start", "end", "raw_piece_id"), on=["doc", "start", "end"], how="left")
    return raw, sides.select(id_col, "side", "raw_piece_id", "doc", "start", "end")


# --------------------------------------------------------------------------- the ETL rule


def etl_mapping(doc: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """For raw pieces sorted by (doc, start, end): the index each piece maps to (ReceptionReader's rule).

    Candidates of piece i: earlier pieces of its document starting at or after ``start[i] − 180``, and i itself. The
    first candidate in (start, end) order with ``|Δstart| ≤ L`` and ``|Δend| ≤ L``,
    ``L = min(max(min(len_i, len_c) // 4, 10), 180)``, wins; its own index is recorded (not its group).

    Vectorized over "how far back" (k): the leftmost qualifying candidate is the one with the largest k.
    """
    n = len(start)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    start = start.astype(np.int64)
    end = end.astype(np.int64)
    big = int(end.max()) + 1000
    key = doc.astype(np.int64) * big + start
    first = np.searchsorted(key, key - 180, side="left")  # first candidate (same document, start ≥ start − 180)
    window = np.arange(n) - first  # number of earlier candidates
    length = end - start
    best = np.zeros(n, dtype=np.int64)
    by_window = np.argsort(-window, kind="stable")
    sorted_window = window[by_window]
    for k in range(1, int(window.max()) + 1):
        active = by_window[: np.searchsorted(-sorted_window, -k, side="right")]
        j = active - k
        limit = np.minimum(np.maximum(np.minimum(length[active], length[j]) // 4, 10), 180)
        ok = (np.abs(start[active] - start[j]) <= limit) & (np.abs(end[active] - end[j]) <= limit)
        best[active[ok]] = k  # larger k (further back, i.e. further left) overrides
    return np.arange(n) - best


# --------------------------------------------------------------------------- node_similarity


def node_similarity_groups(
    starts: list[int], ends: list[int], threshold: float, compat: bool
) -> tuple[list[int], list[tuple[int, int]]]:
    """textreuse-blast's greedy grouping of one document's nodes (sorted by start, duplicates kept).

    Returns the group of each node and each group's centroid span. ``compat`` keeps the original's no-op line
    (``overlap - n1[1]-n2[1]``), so a short piece nested at the start of a long one is merged with it.
    """
    n = len(starts)
    group = [-1] * n
    centroids: list[tuple[int, int]] = []
    for i in range(n):
        if group[i] >= 0:
            continue
        members = [i]
        gid = len(centroids)
        group[i] = gid
        si, ei = starts[i], ends[i]
        for j in range(i + 1, n):
            if group[j] >= 0:
                continue
            sj, ej = starts[j], ends[j]
            len_i, len_j = ei - si, ej - sj
            extra = min(len_i, len_j) * (1 - threshold)
            if sj - extra > si:
                break
            overlap = ei - sj
            if ej < ei and not compat:
                overlap -= ei - ej
            if overlap / max(len_i, len_j) > threshold:
                members.append(j)
                group[j] = gid
        centroids.append((int(sum(starts[m] for m in members) / len(members)),
                          int(sum(ends[m] for m in members) / len(members))))
    return group, centroids


def _node_similarity_doc(task: tuple[np.ndarray, np.ndarray, float, bool]) -> np.ndarray:
    """Centroid span per node of one document; mapping by span string as in the original (last assignment wins)."""
    start, end, threshold, compat = task
    group, centroids = node_similarity_groups(start.tolist(), end.tolist(), threshold, compat)
    by_span: dict[tuple[int, int], tuple[int, int]] = {}
    for s, e, g in zip(start.tolist(), end.tolist(), group, strict=True):
        by_span[(s, e)] = centroids[g]
    return np.array([by_span[(s, e)] for s, e in zip(start.tolist(), end.tolist(), strict=True)],
                    dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------- passim


def passim_links(s1: list[tuple[int, int]], s2: list[tuple[int, int]], src_overlap: float, dst_overlap: float
                 ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """passim's ``linkSpans`` for one document: ``s1`` are its spans as the first document of a pair (passim's
    ``uid``), ``s2`` as the second (``uid2``); both distinct. Returns links between spans.

    1. Each ``s1`` span is linked to the ``s2`` span overlapping it most, if the overlap is at least ``dst_overlap``
       of the ``s1`` span's length (ties: the earliest ``s2`` span, for determinism; passim's order is arbitrary).
    2. The ``s1`` spans left unlinked are sorted by (−length, begin, end). Each is linked to at most one span before
       it: scanning back from its neighbour, the first it overlaps by at least ``src_overlap`` of its own length.
    """
    links = []
    a2 = np.array(sorted(s2), dtype=np.int64).reshape(-1, 2)
    reach2 = np.maximum.accumulate(a2[:, 1]) if len(a2) else a2[:, 1]
    unsourced = []
    for begin, end in sorted(s1):
        length = end - begin
        if len(a2):
            lo = int(np.searchsorted(reach2, begin, side="right"))
            hi = int(np.searchsorted(a2[:, 0], end, side="left"))
            if hi > lo:
                window = a2[lo:hi]
                overlap = np.minimum(window[:, 1], end) - np.maximum(window[:, 0], begin)
                ratio = np.where(overlap > 0, overlap, 0) / length
                best = int(np.argmax(ratio))  # first maximum: the earliest span
                if ratio[best] >= dst_overlap:
                    links.append(((begin, end), (int(window[best, 0]), int(window[best, 1]))))
                    continue
        unsourced.append((begin, end))
    if len(unsourced) > 1:
        unsourced.sort(key=lambda s: (-(s[1] - s[0]), s[0], s[1]))
        u = np.array(unsourced, dtype=np.int64)
        by_start = np.argsort(u[:, 0], kind="stable")
        starts, ends = u[by_start, 0], u[by_start, 1]
        reach = np.maximum.accumulate(ends)
        for i in range(1, len(u)):
            begin, end = int(u[i, 0]), int(u[i, 1])
            lo = int(np.searchsorted(reach, begin, side="right"))
            hi = int(np.searchsorted(starts, end, side="left"))
            cand = by_start[lo:hi]
            cand = cand[cand < i]
            if not len(cand):
                continue
            overlap = np.minimum(u[cand, 1], end) - np.maximum(u[cand, 0], begin)
            ok = cand[overlap / (end - begin) >= src_overlap]
            if len(ok):
                j = int(ok.max())  # scanning back from i - 1, the first qualifying span
                links.append(((begin, end), (int(u[j, 0]), int(u[j, 1]))))
    return links


def _passim_doc(task: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]) -> np.ndarray:
    """Group (component of the links) of each raw piece of one document."""
    from .cluster import components

    start, end, role1, role2, ids, src_overlap, dst_overlap = task
    index = {(int(s), int(e)): k for k, (s, e) in enumerate(zip(start.tolist(), end.tolist(), strict=True))}
    s1 = [(int(s), int(e)) for s, e, r in zip(start.tolist(), end.tolist(), role1.tolist(), strict=True) if r]
    s2 = [(int(s), int(e)) for s, e, r in zip(start.tolist(), end.tolist(), role2.tolist(), strict=True) if r]
    links = passim_links(s1, s2, src_overlap, dst_overlap)
    a = np.array([index[x] for x, _ in links], dtype=np.int64)
    b = np.array([index[y] for _, y in links], dtype=np.int64)
    return ids[components(len(start), a, b)]


# --------------------------------------------------------------------------- plugins


def load_rule(path: str) -> Callable[[pl.DataFrame], Any]:
    module, _, name = path.partition(":")
    if not name:
        raise DefragError(f"defrag.rule {path!r}: expected 'module:function' or one of {', '.join(PRESETS)}")
    try:
        obj = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise DefragError(f"cannot load defrag rule {path!r}: {error}") from error
    return obj() if isinstance(obj, type) else obj


def _plugin_doc(task: tuple[str, pl.DataFrame]) -> pl.DataFrame:
    path, pieces = task
    result = load_rule(path)(pieces)
    if not isinstance(result, pl.DataFrame) or not {"raw_piece_id", "group"} <= set(result.columns):
        raise DefragError(f"defrag rule {path!r} must return a DataFrame with raw_piece_id and group")
    if result.height != pieces.height or set(result["raw_piece_id"]) != set(pieces["raw_piece_id"]):
        raise DefragError(f"defrag rule {path!r} must return one group per raw piece")
    return result


# --------------------------------------------------------------------------- pieces and pairs


def _map(func: Callable[[Any], Any], tasks: list[Any], workers: int) -> list[Any]:
    if workers > 1 and len(tasks) > 1:
        with process_pool(max_workers=workers) as pool:
            return list(pool.map(func, tasks, chunksize=max(1, len(tasks) // (workers * 4))))
    return [func(task) for task in tasks]


def _groups_etl(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    target = etl_mapping(raw["doc"].to_numpy(), raw["start"].to_numpy(), raw["end"].to_numpy())
    return raw.select("raw_piece_id", pl.Series("group", raw["raw_piece_id"].to_numpy()[target])), None


def _groups_node_similarity(
    sides: pl.DataFrame, id_col: str, threshold: float, compat: bool, workers: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    nodes = sides.sort("doc", "start", id_col)  # one node per hit side, by start, ties in hit order
    bounds = np.flatnonzero(np.diff(nodes["doc"].to_numpy())) + 1
    tasks = [(s, e, threshold, compat) for s, e in zip(np.split(nodes["start"].to_numpy(), bounds),
                                                       np.split(nodes["end"].to_numpy(), bounds), strict=True)]
    centroid = np.concatenate(_map(_node_similarity_doc, tasks, workers)) if tasks else np.zeros((0, 2), np.int64)
    mapped = nodes.select("raw_piece_id", "doc").with_columns(
        pl.Series("start", centroid[:, 0]), pl.Series("end", centroid[:, 1])
    ).unique("raw_piece_id", keep="last")
    # pieces are identified by their centroid span (string identity in the original)
    spans = mapped.select("doc", "start", "end").unique().sort("doc", "start", "end").with_row_index("group")
    groups = mapped.join(spans, on=["doc", "start", "end"]).select("raw_piece_id", "group")
    return groups, spans.select("group", "start", "end")


def _groups_passim(raw: pl.DataFrame, sides: pl.DataFrame, options: DefragOptions, workers: int
                   ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Raw pieces grouped by passim's links. passim keeps linked spans as separate nodes of its cluster graph;
    merging them instead gives the same connected components, so clusters are passim's."""
    roles = sides.group_by("raw_piece_id").agg((pl.col("side") == 1).any().alias("role1"),
                                               (pl.col("side") == 2).any().alias("role2"))
    rows = raw.join(roles, on="raw_piece_id").sort("doc", "start", "end")
    tasks = []
    for _, part in rows.group_by("doc", maintain_order=True):
        tasks.append((part["start"].to_numpy(), part["end"].to_numpy(), part["role1"].to_numpy(),
                      part["role2"].to_numpy(), part["raw_piece_id"].to_numpy(), options.src_overlap,
                      options.dst_overlap))
    group = np.concatenate(_map(_passim_doc, tasks, workers)) if tasks else np.zeros(0, dtype=np.int64)
    return rows.select("raw_piece_id", pl.Series("group", group)), None


def _groups_plugin(raw: pl.DataFrame, path: str, workers: int) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    columns = ["raw_piece_id", "start", "end", "enc_start", "enc_end", "n_hsps"]
    tasks = [(path, part.select(columns)) for _, part in raw.group_by("doc", maintain_order=True)]
    results = _map(_plugin_doc, tasks, workers)
    parts = []
    for (_, part), result in zip(raw.group_by("doc", maintain_order=True), results, strict=True):
        parts.append(result.with_columns(pl.lit(int(part["doc"][0])).alias("doc"),
                                         pl.col("group").cast(pl.String)))
    result = pl.concat(parts, how="diagonal_relaxed")
    keys = result.select("doc", "group").unique(maintain_order=True).with_row_index("gid")
    result = result.join(keys, on=["doc", "group"]).drop("group").rename({"gid": "group"})
    groups = result.select("raw_piece_id", "group")
    if {"start", "end"} <= set(result.columns):
        return groups, result.group_by("group").agg(pl.col("start").first(), pl.col("end").first())
    return groups, None


def defragment(
    hits: pl.DataFrame, id_col: str, options: DefragOptions, workers: int = 1
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(pieces, piece_map, pairs) for ``hits`` (columns ``id_col``, doc1/2, start1/2, end1/2, enc_*1/2, scores).

    A rule maps each raw piece to a group, optionally with the group's span (default: from the smallest start to the
    largest end of its raw pieces). ``enc_start``/``enc_end`` of a piece are those of its raw pieces (min/max); the
    caller may recompute them from the merged surface span.
    """
    raw, sides = raw_pieces(hits, id_col)
    if options.rule == "etl":
        groups, spans = _groups_etl(raw)
    elif options.rule in ("node_similarity", "node_similarity-compat"):
        groups, spans = _groups_node_similarity(sides, id_col, options.node_similarity,
                                                options.rule.endswith("compat"), workers)
    elif options.rule == "passim":
        groups, spans = _groups_passim(raw, sides, options, workers)
    else:
        groups, spans = _groups_plugin(raw, options.rule, workers)
    joined = raw.join(groups, on="raw_piece_id")
    pieces = joined.group_by("group").agg(
        pl.col("doc").first(), pl.col("start").min(), pl.col("end").max(), pl.col("enc_start").min(),
        pl.col("enc_end").max(), pl.len().alias("n_raw_pieces"), pl.col("n_hsps").sum(),
    )
    if spans is not None:
        pieces = pieces.drop("start", "end").join(spans, on="group")
    pieces = pieces.sort("doc", "start", "end", "group").with_row_index("piece_id")
    raw_to_piece = groups.join(pieces.select("group", "piece_id"), on="group").select("raw_piece_id", "piece_id")
    pieces = pieces.select(
        pl.col("piece_id").cast(pl.Int64), pl.col("doc").cast(pl.Int32), pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64), pl.col("enc_start").cast(pl.Int64), pl.col("enc_end").cast(pl.Int64),
        pl.col("n_raw_pieces").cast(pl.Int32), pl.col("n_hsps").cast(pl.Int32),
    )
    piece_map = (
        sides.join(raw_to_piece.with_columns(pl.col("piece_id").cast(pl.Int64)), on="raw_piece_id", how="left")
        .select(id_col, "side", "raw_piece_id", "piece_id")
        .sort(id_col, "side")
    )
    return pieces, piece_map, make_pairs(hits, id_col, piece_map)


def make_pairs(hits: pl.DataFrame, id_col: str, piece_map: pl.DataFrame) -> pl.DataFrame:
    """One row per (piece1, piece2) linked by at least one hit, with the hits' scores aggregated."""
    side1 = piece_map.filter(pl.col("side") == 1).select(id_col, pl.col("piece_id").alias("piece1"))
    side2 = piece_map.filter(pl.col("side") == 2).select(id_col, pl.col("piece_id").alias("piece2"))
    score_cols = [c for c in ("score", "parent_score") if c in hits.columns]
    bits_cols = [c for c in ("bitscore", "parent_bitscore") if c in hits.columns]
    eval_cols = [c for c in ("evalue", "parent_evalue") if c in hits.columns]
    linked = hits.select(id_col, "doc1", "doc2", "pident", "length", *score_cols[:1], *bits_cols[:1], *eval_cols[:1]
                         ).join(side1, on=id_col).join(side2, on=id_col)
    pairs = (
        linked.group_by("piece1", "piece2")
        .agg(
            pl.col("doc1").first(), pl.col("doc2").first(), pl.len().alias("num_orig_links"),
            pl.col(score_cols[0]).max().alias("score"), pl.col(bits_cols[0]).max().alias("bitscore"),
            pl.col(eval_cols[0]).min().alias("evalue"), pl.col("pident").mean().alias("mean_pident"),
            pl.col("length").sum().alias("total_length"),
        )
        .sort("piece1", "piece2")
        .with_row_index("pair_id")
    )
    return pairs.select(
        pl.col("pair_id").cast(pl.Int64), "piece1", "piece2", pl.col("doc1").cast(pl.Int32),
        pl.col("doc2").cast(pl.Int32), pl.col("num_orig_links").cast(pl.Int32), pl.col("score").cast(pl.Int32),
        pl.col("bitscore").cast(pl.Float32), pl.col("evalue").cast(pl.Float64),
        pl.col("mean_pident").cast(pl.Float32), pl.col("total_length").cast(pl.Int64),
    )
