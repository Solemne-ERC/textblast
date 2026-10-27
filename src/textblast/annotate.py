"""Annotations of pieces: unit overlaps, masked share, encoded ranges (plan section 8; Q33; round 4).

Units come from the store's ``units.parquet`` (``doc_idx, loc, start, length``, surface code points). They may
overlap or nest in general (not in the current corpus); every function here handles that.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .mapping import CorpusStore


def units_frame(store: CorpusStore) -> pl.DataFrame:
    """Units with ``end``, sorted by document and start."""
    return (
        store.units()
        .with_columns((pl.col("start") + pl.col("length")).alias("end"))
        .select(pl.col("doc_idx").cast(pl.Int32).alias("doc"), "loc", "start", "end")
        .sort("doc", "start", "end")
    )


def unit_overlaps(pieces: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """``piece_id, doc, loc, overlap_chars, piece_share, unit_share`` for every piece/unit overlap.

    A sorted interval join per document: the candidates of a piece are the units from the first one whose running
    maximum end passes the piece's start to the last one starting before the piece's end; candidates that do not
    overlap (possible only with nested units) are dropped.
    """
    schema = {"piece_id": pl.Int64, "doc": pl.Int32, "loc": pl.String, "overlap_chars": pl.Int64,
              "piece_share": pl.Float32, "unit_share": pl.Float32}
    parts = []
    by_doc = dict(units.group_by(["doc"]).__iter__()) if not units.is_empty() else {}
    for (d,), part in pieces.select("piece_id", "doc", "start", "end").group_by(["doc"]):
        if (d,) not in by_doc:
            continue
        u = by_doc[(d,)]
        u_start, u_end = u["start"].to_numpy(), u["end"].to_numpy()
        reach = np.maximum.accumulate(u_end)
        p_start, p_end = part["start"].to_numpy(), part["end"].to_numpy()
        lo = np.searchsorted(reach, p_start, side="right")
        hi = np.searchsorted(u_start, p_end, side="left")
        count = np.maximum(hi - lo, 0)
        if not count.sum():
            continue
        row = np.repeat(np.arange(len(p_start)), count)
        unit = np.concatenate([np.arange(a, b) for a, b in zip(lo[count > 0], hi[count > 0], strict=True)])
        overlap = np.minimum(p_end[row], u_end[unit]) - np.maximum(p_start[row], u_start[unit])
        keep = overlap > 0
        row, unit, overlap = row[keep], unit[keep], overlap[keep]
        parts.append(pl.DataFrame({
            "piece_id": part["piece_id"].to_numpy()[row], "doc": np.full(len(row), d, dtype=np.int32),
            "loc": np.asarray(u["loc"].to_list(), dtype=object)[unit],
            "overlap_chars": overlap.astype(np.int64),
            "piece_share": (overlap / (p_end[row] - p_start[row])).astype(np.float32),
            "unit_share": (overlap / (u_end[unit] - u_start[unit])).astype(np.float32),
        }, schema=schema))
    if not parts:
        return pl.DataFrame(schema=schema)
    return pl.concat(parts).sort("piece_id", "loc")


def unit_at(units: pl.DataFrame, doc: np.ndarray, pos: np.ndarray) -> list[str | None]:
    """The unit containing each position (the innermost, i.e. latest-starting, one), or None."""
    result: list[str | None] = [None] * len(pos)
    if units.is_empty() or len(pos) == 0:
        return result
    for (d,), part in units.group_by(["doc"]):
        mask = np.flatnonzero(doc == d)
        if not len(mask):
            continue
        starts, ends, locs = part["start"].to_numpy(), part["end"].to_numpy(), part["loc"].to_list()
        reach = np.maximum.accumulate(ends)  # for nested units: how far any unit up to here reaches
        k = np.searchsorted(starts, pos[mask], side="right") - 1
        inside = (k >= 0) & (ends[np.maximum(k, 0)] > pos[mask])
        for i, kk in zip(mask[inside].tolist(), k[inside].tolist(), strict=True):
            result[i] = locs[kk]
        # rare: the latest-starting unit ended before pos, but an earlier, longer one contains it
        nested = (k >= 0) & ~inside & (reach[np.maximum(k, 0)] > pos[mask])
        for i, kk in zip(mask[nested].tolist(), k[nested].tolist(), strict=True):
            while kk >= 0 and reach[kk] > pos[i]:
                if ends[kk] > pos[i]:
                    result[i] = locs[kk]
                    break
                kk -= 1
    return result


def zones(units: pl.DataFrame, doc: int) -> np.ndarray:
    """Sorted distinct unit boundaries (starts and ends) of one document; a letter's zone is the number of
    boundaries at or before its surface start, so two letters are in one unit stretch iff their zones are equal."""
    part = units.filter(pl.col("doc") == doc)
    return np.unique(np.concatenate([part["start"].to_numpy(), part["end"].to_numpy()]))


def masked_share(store: CorpusStore, doc: np.ndarray, enc_start: np.ndarray, enc_end: np.ndarray) -> np.ndarray:
    """Share of the encoded letters in each range that are masked (lowercase)."""
    symbols = np.asarray(store.symbols)
    masked = np.concatenate([[0], np.cumsum((symbols >= ord("a")) & (symbols <= ord("z")), dtype=np.int64)])
    base = store.offsets[doc.astype(np.int64)]
    a, b = base + enc_start, base + enc_end
    n = np.maximum(b - a, 1)
    return ((masked[b] - masked[a]) / n).astype(np.float32)


def encoded_range(store: CorpusStore, doc: np.ndarray, start: np.ndarray, end: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The encoded letters inside surface spans ``[start, end)``: first letter starting at or after ``start``, up to
    the last letter ending at or before ``end``."""
    enc_start = np.zeros(len(doc), dtype=np.int64)
    enc_end = np.zeros(len(doc), dtype=np.int64)
    for d in np.unique(doc).tolist():
        rows = np.flatnonzero(doc == d)
        a, b = int(store.offsets[d]), int(store.offsets[d + 1])
        starts = np.asarray(store.map_start[a:b], dtype=np.int64)
        ends = np.asarray(store.map_end[a:b], dtype=np.int64)
        enc_start[rows] = np.searchsorted(starts, start[rows], side="left")
        enc_end[rows] = np.searchsorted(ends, end[rows], side="right")
    return enc_start, enc_end


# --------------------------------------------------------------------------- formulae and reference sets (Q29)


def doc_degree(pieces: pl.DataFrame, pairs: pl.DataFrame) -> np.ndarray:
    """Per piece (in ``pieces`` order): the number of distinct documents it is linked to by its pairs."""
    links = pl.concat([
        pairs.select(pl.col("piece1").alias("piece_id"), pl.col("doc2").alias("partner")),
        pairs.select(pl.col("piece2").alias("piece_id"), pl.col("doc1").alias("partner")),
    ])
    degree = links.group_by("piece_id").agg(pl.col("partner").n_unique().alias("doc_degree"))
    return (pieces.select("piece_id").join(degree, on="piece_id", how="left")
            .fill_null(0)["doc_degree"].to_numpy().astype(np.int32))


def reference_overlap(pieces: pl.DataFrame, hits: pl.DataFrame, in_set: np.ndarray) -> np.ndarray:
    """Per piece: the share of its characters covered by hit sides of the same document whose partner document is
    in the reference set (``in_set``: a boolean per ``doc_idx``)."""
    regions = pl.concat([
        hits.select(pl.col("doc1").alias("doc"), pl.col("start1").alias("start"), pl.col("end1").alias("end"),
                    pl.col("doc2").alias("partner")),
        hits.select(pl.col("doc2").alias("doc"), pl.col("start2").alias("start"), pl.col("end2").alias("end"),
                    pl.col("doc1").alias("partner")),
    ])
    regions = regions.filter(pl.Series(in_set)[regions["partner"].to_numpy()]) if regions.height else regions
    share = np.zeros(pieces.height, dtype=np.float32)
    if regions.height == 0 or pieces.height == 0:
        return share
    by_doc = {key[0]: part for key, part in regions.sort("doc", "start", "end").group_by(["doc"])}
    rows = pieces.with_row_index("_row").select("_row", "doc", "start", "end")
    for (d,), part in rows.group_by(["doc"]):
        r = by_doc.get(d)
        if r is None:
            continue
        # union of the regions: running maximum of their ends splits them into disjoint stretches
        starts, ends = r["start"].to_numpy().astype(np.int64), r["end"].to_numpy().astype(np.int64)
        reach = np.maximum.accumulate(ends)
        new = np.ones(len(starts), dtype=bool)
        new[1:] = starts[1:] > reach[:-1]
        group = np.cumsum(new) - 1
        u_start = starts[new]
        u_end = np.zeros(group[-1] + 1, dtype=np.int64)
        np.maximum.at(u_end, group, ends)
        p_start, p_end = part["start"].to_numpy().astype(np.int64), part["end"].to_numpy().astype(np.int64)
        length = np.maximum(p_end - p_start, 1)
        covered = _covered(u_start, u_end, p_end) - _covered(u_start, u_end, p_start)
        share[part["_row"].to_numpy()] = (covered / length).astype(np.float32)
    return share


def _covered(u_start: np.ndarray, u_end: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Characters of the disjoint sorted stretches ``[u_start, u_end)`` that lie before each position ``x``."""
    before = np.concatenate([[0], np.cumsum(u_end - u_start)])  # covered characters before stretch k
    k = np.searchsorted(u_start, x, side="right")  # stretches starting at or before x
    last = np.maximum(k - 1, 0)
    inside = np.minimum(np.maximum(x - u_start[last], 0), u_end[last] - u_start[last])
    return before[last] + np.where(k > 0, inside, 0)


def reference_set_members(documents: pl.DataFrame, name: str, expression: str) -> np.ndarray:
    """A boolean per ``doc_idx``: whether the document belongs to the reference set (a SQL expression over the
    document fields, e.g. ``doc_id LIKE 'bible:%'``)."""
    try:
        chosen = documents.filter(pl.sql_expr(expression).fill_null(False))["doc_idx"].to_numpy()
    except Exception as error:  # polars raises several exception types for SQL
        raise ValueError(f"reference set {name!r}: invalid expression {expression!r}: {error}") from error
    member = np.zeros(int(documents["doc_idx"].max()) + 1 if documents.height else 0, dtype=bool)
    member[chosen.astype(np.int64)] = True
    return member
