"""Unit clipping: raw hits cut at every unit boundary they cross on either side (plan section 8; round 4).

``tables/hsps_clipped.parquet`` is an additional view; ``hsps`` is never changed. A hit is cut on its alignment
(``btop``): a new segment starts at the column whose letter, on either side, lies in another unit stretch than the
previous letter of that side. Segments with letters on one side only (pure gap columns across a boundary) align
nothing and are dropped (counted). Only hits whose first and last letters lie in different stretches are parsed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .align import Columns, column_stats, parse_btop, segment_starts
from .annotate import unit_at, units_frame
from .mapping import CorpusStore
from .parallel import process_pool

CLIP_SCHEMA = {
    "clip_id": pl.Int64, "hsp_id": pl.Int64, "doc1": pl.Int32, "doc2": pl.Int32,
    "start1": pl.Int32, "end1": pl.Int32, "start2": pl.Int32, "end2": pl.Int32,
    "enc_start1": pl.Int32, "enc_end1": pl.Int32, "enc_start2": pl.Int32, "enc_end2": pl.Int32,
    "loc1": pl.String, "loc2": pl.String, "col_start": pl.Int32, "col_end": pl.Int32,
    "length": pl.Int32, "nident": pl.Int32, "mismatch": pl.Int32, "gapopen": pl.Int32, "gaps": pl.Int32,
    "pident": pl.Float32, "parent_score": pl.Int32, "parent_bitscore": pl.Float32, "parent_evalue": pl.Float64,
}


class ClipError(RuntimeError):
    """Clipping is impossible (e.g. the run has no ``btop``)."""


def _boundaries(units: pl.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(d): np.unique(np.concatenate([part["start"].to_numpy(), part["end"].to_numpy()]))
        for (d,), part in units.group_by(["doc"])
    }


def _zone(store: CorpusStore, bounds: dict[int, np.ndarray], doc: np.ndarray, enc: np.ndarray) -> np.ndarray:
    """Zone (number of unit boundaries at or before the letter's surface start) of letter ``enc`` of ``doc``."""
    out = np.zeros(len(doc), dtype=np.int64)
    surface = np.asarray(store.map_start[store.offsets[doc.astype(np.int64)] + enc], dtype=np.int64)
    for d in np.unique(doc).tolist():
        b = bounds.get(int(d))
        if b is not None:
            rows = doc == d
            out[rows] = np.searchsorted(b, surface[rows], side="right")
    return out


# --------------------------------------------------------------------------- worker side

_WORKER: dict[str, Any] = {}


def _init_worker(run_dir: str) -> None:
    store = CorpusStore(run_dir)
    _WORKER["store"] = store
    _WORKER["bounds"] = _boundaries(units_frame(store))


def _letter_zones(doc: int, enc_start: int, enc_end: int) -> np.ndarray:
    store, bounds = _WORKER["store"], _WORKER["bounds"]
    b = bounds.get(doc)
    if b is None:
        return np.zeros(enc_end - enc_start, dtype=np.int64)
    base = int(store.offsets[doc])
    surface = np.asarray(store.map_start[base + enc_start : base + enc_end], dtype=np.int64)
    return np.searchsorted(b, surface, side="right")


def split_hit(cols: Columns, zone1: np.ndarray, zone2: np.ndarray, enc_start1: int, enc_start2: int
              ) -> tuple[list[dict[str, int]], int]:
    """Segments of one alignment (encoded ranges, column range, statistics) and the number dropped."""
    starts = segment_starts(cols, zone1, zone2)
    bounds = [*starts, cols.length]
    letters1 = np.concatenate([[0], np.cumsum(cols.adv1)])
    letters2 = np.concatenate([[0], np.cumsum(cols.adv2)])
    segments, dropped = [], 0
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        n1, n2 = int(letters1[b] - letters1[a]), int(letters2[b] - letters2[a])
        if n1 == 0 or n2 == 0:
            dropped += 1
            continue
        segments.append({
            "enc_start1": enc_start1 + int(letters1[a]), "enc_end1": enc_start1 + int(letters1[b]),
            "enc_start2": enc_start2 + int(letters2[a]), "enc_end2": enc_start2 + int(letters2[b]),
            "col_start": a, "col_end": b, **column_stats(cols.part(a, b)),
        })
    return segments, dropped


SEGMENT_COLUMNS = ("hsp_id", "enc_start1", "enc_end1", "enc_start2", "enc_end2", "col_start", "col_end",
                   "length", "nident", "mismatch", "gapopen", "gaps")


def _clip_chunk(frame: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Segments of the hits in ``frame`` as one compact table (cheap to send back from a worker)."""
    columns: dict[str, list[int]] = {name: [] for name in SEGMENT_COLUMNS}
    dropped = 0
    for hsp_id, doc1, doc2, s1, e1, s2, e2, btop, query_side in frame.iter_rows():
        cols = parse_btop(btop, query_side)
        segments, n = split_hit(cols, _letter_zones(doc1, s1, e1), _letter_zones(doc2, s2, e2), s1, s2)
        dropped += n
        for seg in segments:
            columns["hsp_id"].append(hsp_id)
            for name in SEGMENT_COLUMNS[1:]:
                columns[name].append(seg[name])
    return pl.DataFrame(columns, schema={name: pl.Int64 for name in SEGMENT_COLUMNS}), dropped


# --------------------------------------------------------------------------- main


def clip_hits(run_dir: Path, hsps: pl.DataFrame, workers: int = 1) -> tuple[pl.DataFrame, dict[str, int]]:
    """``hsps_clipped`` and counts (hits cut, segments, dropped one-sided segments)."""
    if "btop" not in hsps.columns:
        raise ClipError("unit clipping needs BLAST's btop, but this run was made with --no-btop "
                        "(blast.btop: false); rerun BLAST with btop to clip")
    store = CorpusStore(run_dir)
    units = units_frame(store)
    bounds = _boundaries(units)
    doc1, doc2 = hsps["doc1"].to_numpy(), hsps["doc2"].to_numpy()
    s1, e1 = hsps["enc_start1"].to_numpy().astype(np.int64), hsps["enc_end1"].to_numpy().astype(np.int64)
    s2, e2 = hsps["enc_start2"].to_numpy().astype(np.int64), hsps["enc_end2"].to_numpy().astype(np.int64)
    crosses = (_zone(store, bounds, doc1, s1) != _zone(store, bounds, doc1, e1 - 1)) | (
        _zone(store, bounds, doc2, s2) != _zone(store, bounds, doc2, e2 - 1)
    )

    whole = hsps.filter(pl.Series(~crosses)).select(
        "hsp_id", "enc_start1", "enc_end1", "enc_start2", "enc_end2",
        pl.lit(0).alias("col_start"), pl.col("length").alias("col_end"),
        "length", "nident", "mismatch", "gapopen", "gaps",
    )
    cut = hsps.filter(pl.Series(crosses)).select(
        "hsp_id", "doc1", "doc2", "enc_start1", "enc_end1", "enc_start2", "enc_end2", "btop", "query_side")
    n_chunks = max(1, workers * 8)
    size = -(-cut.height // n_chunks) if cut.height else 1
    chunks = [cut.slice(i, size) for i in range(0, cut.height, size)]
    if workers > 1 and cut.height > 1000:
        with process_pool(max_workers=workers, initializer=_init_worker, initargs=(str(run_dir),)) as pool:
            results = list(pool.map(_clip_chunk, chunks))
    else:
        _init_worker(str(run_dir))
        results = [_clip_chunk(chunk) for chunk in chunks]
    dropped = sum(n for _, n in results)
    segs = pl.concat([whole.with_columns(pl.all().cast(pl.Int64)), *[part for part, _ in results]])
    del cut, chunks, results

    parents = hsps.select("hsp_id", "doc1", "doc2", pl.col("score").alias("parent_score"),
                          pl.col("bitscore").alias("parent_bitscore"), pl.col("evalue").alias("parent_evalue"))
    segs = segs.join(parents, on="hsp_id")
    d1, d2 = segs["doc1"].to_numpy(), segs["doc2"].to_numpy()
    start1, end1 = store.to_surface(d1, segs["enc_start1"].to_numpy(), segs["enc_end1"].to_numpy())
    start2, end2 = store.to_surface(d2, segs["enc_start2"].to_numpy(), segs["enc_end2"].to_numpy())
    segs = segs.with_columns(
        pl.Series("start1", start1), pl.Series("end1", end1), pl.Series("start2", start2), pl.Series("end2", end2),
        pl.Series("loc1", unit_at(units, d1, start1), dtype=pl.String),
        pl.Series("loc2", unit_at(units, d2, start2), dtype=pl.String),
        (100.0 * pl.col("nident") / pl.col("length")).alias("pident"),
    )
    clipped = (
        segs.sort("doc1", "doc2", "start1", "start2", "hsp_id")
        .with_row_index("clip_id")
        .select([pl.col(name).cast(dtype) for name, dtype in CLIP_SCHEMA.items()])
    )
    stats = {"hits": hsps.height, "hits_cut": int(crosses.sum()), "segments": clipped.height,
             "dropped_one_sided": dropped}
    return clipped, stats
