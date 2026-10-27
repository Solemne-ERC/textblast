"""``textblast export``: a portable bundle and converters to other formats.

The core tables are the native format. Converters derive other formats from them; each declares the tables it needs
and refuses with a clear message when one is missing. Custom converters plug in by import path. All exports use the
hit filter that ``post`` applied (recorded in the manifest), so exports and ``pairs_filtered`` always agree.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import __version__
from .align import parse_btop
from .api import BUNDLE_MARKER, Run, RunNotAvailable, open_run
from .logs import Stage, format_bytes, format_count, get_logger
from .mapping import CorpusStore
from .parallel import process_pool
from .post import SCHEMAS, annotation_columns, kept_hits
from .prepare import read_manifest, write_manifest
from .sysinfo import default_workers

log = get_logger("export")

STORE_STATS = {"doc_idx", "doc_id", "n_chars", "n_letters", "n_encoded", "n_masked", "n_barriers", "deleted_share",
               "empty"}


class ExportError(RuntimeError):
    """An export cannot be written (missing table, unknown format, bad option)."""


@dataclass
class ExportOptions:
    formats: list[str] = field(default_factory=list)  # written by `run` (none by default)
    excerpts: str = "surface"  # surface | none
    offsets: str = "codepoints"  # codepoints | utf16
    fields: list[str] | None = None  # document metadata to carry (default: all input metadata)
    file_format: str = "parquet"  # pairs_flat: parquet | csv | jsonl
    view: str | None = None  # hits for etl / passim-pairwise: hsps | clipped | joined (default: the view post used)
    texts: bool = True  # bundle: include the texts
    boiler_field: str | None = None  # passim-clusters: links within one value of this field count for pboiler
    out_dir: str | None = None  # default: <run_dir>/exports
    converters: dict[str, str] = field(default_factory=dict)  # custom: name → "module:function"


@dataclass
class Context:
    run: Run
    run_dir: Path
    options: ExportOptions
    out_dir: Path
    hit_filter: str | None
    view: str
    workers: int

    def metadata_fields(self) -> list[str]:
        columns = self.run.documents.collect_schema().names()
        if self.options.fields is None:
            return [c for c in columns if c not in STORE_STATS]
        unknown = [f for f in self.options.fields if f not in columns]
        if unknown:
            raise ExportError(f"export.fields: unknown document fields {unknown} (available: {columns})")
        return list(self.options.fields)

    def offsets(self, doc: np.ndarray, values: np.ndarray) -> np.ndarray:
        if self.options.offsets == "utf16":
            return self.run.to_utf16(doc, values)
        return np.asarray(values, dtype=np.int64)


@dataclass
class Converter:
    func: Callable[[Context], tuple[Path, int]]  # → (output path, rows)
    needs: tuple[str, ...]  # tables


# --------------------------------------------------------------------------- bundle


def export_bundle(ctx: Context) -> tuple[Path, int]:
    target = ctx.out_dir / "bundle"
    tmp = target.with_name("bundle.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    copied = []
    for name in SCHEMAS:
        if name == "texts" and not ctx.options.texts:
            continue
        try:
            source = ctx.run._file(name)
        except RunNotAvailable:
            continue
        if source.is_file():
            shutil.copy2(source, tmp / f"{name}.parquet")
            copied.append(name)
    manifest = read_manifest(ctx.run_dir)
    stages = manifest.get("stages", {})
    info = {
        "bundle": 1,
        "textblast": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(ctx.run_dir.resolve()),
        "tables": copied,
        "schemas": {k: v for k, v in SCHEMAS.items() if k in copied},
        "corpus_fingerprint": stages.get("prepare", {}).get("fingerprint"),
        "prepare": {k: stages.get("prepare", {}).get(k) for k in ("profile", "config")},
        "blast": {k: stages.get("plan", {}).get(k) for k in ("settings", "engine", "pair_filter")},
        "post": stages.get("post", {}).get("settings"),
    }
    (tmp / BUNDLE_MARKER).write_text(json.dumps(info, indent=1, default=str) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    return target, len(copied)


# --------------------------------------------------------------------------- pairs_flat


def export_pairs_flat(ctx: Context) -> tuple[Path, int]:
    name = "pairs_filtered" if ctx.hit_filter else "pairs"
    pairs = ctx.run.table(name).collect()
    pieces = ctx.run.pieces.collect()
    docs = ctx.run.documents.collect()
    fields = ctx.metadata_fields()
    frame = pairs
    names = annotation_columns(pieces)
    try:
        clusters = ctx.run.clusters.collect()
    except RunNotAvailable:
        clusters = None
    for k in (1, 2):
        side = pieces.select(pl.col("piece_id").alias(f"piece{k}"), pl.col("start").alias(f"start{k}"),
                             pl.col("end").alias(f"end{k}"), *[pl.col(n).alias(f"{n}{k}") for n in names])
        if clusters is not None:
            side = side.join(clusters.rename({"piece_id": f"piece{k}", "cluster_id": f"cluster{k}"}),
                             on=f"piece{k}", how="left")
        meta = docs.select(pl.col("doc_idx").cast(pl.Int32).alias(f"doc{k}"), pl.col("doc_id").alias(f"doc_id{k}"),
                           *[pl.col(f).alias(f"{f}{k}") for f in fields])
        frame = frame.join(side, on=f"piece{k}", how="left").join(meta, on=f"doc{k}", how="left")
    try:
        overlaps = ctx.run.unit_overlaps.collect()
    except RunNotAvailable:
        overlaps = None
    if overlaps is not None:
        locs = overlaps.sort("piece_id", "loc").group_by("piece_id").agg(pl.col("loc"))
        for k in (1, 2):
            frame = frame.join(locs.rename({"piece_id": f"piece{k}", "loc": f"locs{k}"}), on=f"piece{k}", how="left")
    if ctx.options.excerpts == "surface":
        for k in (1, 2):
            frame = frame.with_columns(pl.Series(f"text{k}", ctx.run.excerpt(
                frame[f"doc{k}"].to_numpy(), frame[f"start{k}"].to_numpy(), frame[f"end{k}"].to_numpy())))
    if ctx.options.offsets == "utf16":
        frame = frame.with_columns(
            *[pl.Series(f"{p}{k}", ctx.offsets(frame[f"doc{k}"].to_numpy(), frame[f"{p}{k}"].to_numpy()))
              for k in (1, 2) for p in ("start", "end")])
    first = ["pair_id", "doc_id1", "start1", "end1", "doc_id2", "start2", "end2"]
    frame = frame.select(first + [c for c in frame.columns if c not in first])
    fmt = ctx.options.file_format
    if fmt not in ("parquet", "csv", "jsonl"):
        raise ExportError(f"export.file_format must be parquet, csv or jsonl, not {fmt!r}")
    path = ctx.out_dir / f"pairs_flat.{fmt}"
    tmp = path.with_name(path.name + ".tmp")
    if fmt == "parquet":
        frame.write_parquet(tmp, compression="zstd")
    elif fmt == "jsonl":
        frame.write_ndjson(tmp)
    else:
        lists = [c for c, dtype in frame.schema.items() if isinstance(dtype, pl.List)]
        frame.with_columns([pl.col(c).list.join("|") for c in lists]).write_csv(tmp)
    tmp.replace(path)
    return path, frame.height


# --------------------------------------------------------------------------- passim-pairwise

_ALIGN: dict[str, Any] = {}


def _align_init(run_dir: str) -> None:
    store = CorpusStore(run_dir)
    _ALIGN["store"] = store
    _ALIGN["texts"] = pl.read_parquet(Path(run_dir) / "corpus" / "texts.parquet").sort("doc_idx")["text"].to_list()


def _code_points(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)


def _side_string(span: np.ndarray, chunk: np.ndarray, width: np.ndarray, offset: np.ndarray) -> str:
    """Scatter one side's surface code points into its padded columns (``-`` elsewhere)."""
    out = np.full(int(width.sum()), ord("-"), dtype=np.uint32)
    if len(span):
        column = np.repeat(np.arange(len(chunk)), chunk)
        within = np.arange(len(span)) - np.repeat(np.cumsum(chunk) - chunk, chunk)
        out[offset[column] + within] = span
    return out.tobytes().decode("utf-32-le")


def aligned_surface(cols_adv1: np.ndarray, cols_adv2: np.ndarray, store: CorpusStore, texts: list[str],
                    doc1: int, s1: int, e1: int, doc2: int, s2: int, e2: int) -> tuple[str, str]:
    """passim-style ``s1``/``s2`` for letters ``[s1, e1)`` of doc1 and ``[s2, e2)`` of doc2 on the given columns."""
    a, b, _ = aligned_surface_columns(cols_adv1, cols_adv2, store, texts, doc1, s1, e1, doc2, s2, e2)
    return a, b


def aligned_surface_columns(cols_adv1: np.ndarray, cols_adv2: np.ndarray, store: CorpusStore, texts: list[str],
                            doc1: int, s1: int, e1: int, doc2: int, s2: int, e2: int
                            ) -> tuple[str, str, np.ndarray]:
    """Like :func:`aligned_surface`, plus the width of each alignment column in the strings."""
    chunks = []
    spans = []
    for adv, doc, a, b, text in ((cols_adv1, doc1, s1, e1, texts[doc1]), (cols_adv2, doc2, s2, e2, texts[doc2])):
        base = int(store.offsets[doc])
        starts = np.asarray(store.map_start[base + a : base + b], dtype=np.int64)
        bounds = np.append(starts, int(store.map_end[base + b - 1]))
        per_letter = np.diff(bounds)
        chunk = np.zeros(len(adv), dtype=np.int64)
        chunk[adv] = per_letter
        chunks.append(chunk)
        spans.append(_code_points(text[bounds[0] : bounds[-1]]))
    width = np.maximum(chunks[0], chunks[1])
    offset = np.cumsum(width) - width
    return (_side_string(spans[0], chunks[0], width, offset), _side_string(spans[1], chunks[1], width, offset),
            width)


def _align_chunk(frame: pl.DataFrame) -> pl.DataFrame:
    store, texts = _ALIGN["store"], _ALIGN["texts"]
    out1, out2 = [], []
    for btop, query_side, c0, c1, d1, s1, e1, d2, s2, e2 in frame.iter_rows():
        cols = parse_btop(btop, query_side).part(c0, c1)
        a, b = aligned_surface(cols.adv1, cols.adv2, store, texts, d1, s1, e1, d2, s2, e2)
        out1.append(a)
        out2.append(b)
    return pl.DataFrame({"s1": out1, "s2": out2}, schema={"s1": pl.String, "s2": pl.String})


def _joined_chunk(frame: pl.DataFrame) -> pl.DataFrame:
    """``s1``/``s2`` of joined hits: the members' alignments in order, the text between members unaligned (each
    side's gap text against ``-`` padding). A member overlapping the ones before it is trimmed to its new letters."""
    store, texts = _ALIGN["store"], _ALIGN["texts"]
    out1, out2 = [], []
    for members in frame["members"].to_list():
        parts1: list[str] = []
        parts2: list[str] = []
        reach1 = reach2 = None  # letters covered so far (exclusive end)
        for m in members:
            d1, d2 = m["doc1"], m["doc2"]
            s1, e1, s2, e2 = m["enc_start1"], m["enc_end1"], m["enc_start2"], m["enc_end2"]
            cols = parse_btop(m["btop"], m["query_side"])
            first = 0
            if reach1 is not None:
                skip1, skip2 = max(0, reach1 - s1), max(0, reach2 - s2)
                c1 = int(np.searchsorted(np.cumsum(cols.adv1), skip1, side="left")) + 1 if skip1 else 0
                c2 = int(np.searchsorted(np.cumsum(cols.adv2), skip2, side="left")) + 1 if skip2 else 0
                first = max(c1, c2)
                if first >= cols.length:
                    continue  # nothing new: the member lies inside the ones before it
                s1 += int(cols.adv1[:first].sum())
                s2 += int(cols.adv2[:first].sum())
                cols = cols.part(first, cols.length)
                if s1 >= e1 or s2 >= e2:
                    continue
                base1, base2 = int(store.offsets[d1]), int(store.offsets[d2])
                gap1 = texts[d1][int(store.map_end[base1 + reach1 - 1]):int(store.map_start[base1 + s1])]
                gap2 = texts[d2][int(store.map_end[base2 + reach2 - 1]):int(store.map_start[base2 + s2])]
                parts1.append(gap1 + "-" * len(gap2))
                parts2.append("-" * len(gap1) + gap2)
            a, b = aligned_surface(cols.adv1, cols.adv2, store, texts, d1, s1, e1, d2, s2, e2)
            parts1.append(a)
            parts2.append(b)
            reach1 = e1 if reach1 is None else max(reach1, e1)
            reach2 = e2 if reach2 is None else max(reach2, e2)
        out1.append("".join(parts1))
        out2.append("".join(parts2))
    return pl.DataFrame({"s1": out1, "s2": out2}, schema={"s1": pl.String, "s2": pl.String})


def _joined_alignments(run_dir: Path, hits: pl.DataFrame, workers: int) -> pl.DataFrame:
    members = (
        pl.read_parquet(run_dir / "tables" / "join_map.parquet")
        .join(hits.select("joined_id").with_row_index("_row"), on="joined_id")
        .join(pl.read_parquet(run_dir / "tables" / "hsps.parquet",
                              columns=["hsp_id", "btop", "query_side", "doc1", "enc_start1", "enc_end1", "doc2",
                                       "enc_start2", "enc_end2"]), on="hsp_id")
        .sort("_row", "enc_start1", "enc_end1", "hsp_id")
        .group_by("_row", maintain_order=True)
        .agg(pl.struct("btop", "query_side", "doc1", "enc_start1", "enc_end1", "doc2", "enc_start2",
                       "enc_end2").alias("members"))
    )
    n_chunks = max(1, workers * 8)
    size = max(1, -(-members.height // n_chunks))
    chunks = [members.slice(i, size) for i in range(0, members.height, size)]
    if workers > 1 and members.height > 2000:
        with process_pool(max_workers=workers, initializer=_align_init, initargs=(str(run_dir),)) as pool:
            parts = list(pool.map(_joined_chunk, chunks))
    else:
        _align_init(str(run_dir))
        parts = [_joined_chunk(c) for c in chunks]
    return pl.concat(parts) if parts else pl.DataFrame(schema={"s1": pl.String, "s2": pl.String})


def surface_alignments(run_dir: Path, hits: pl.DataFrame, workers: int) -> pl.DataFrame:
    """``s1``/``s2`` for every hit (or clipped segment, via its parent's ``btop`` and column range; or joined hit,
    from its members)."""
    if "joined_id" in hits.columns:
        return _joined_alignments(run_dir, hits, workers)
    if "clip_id" in hits.columns:
        parents = pl.read_parquet(run_dir / "tables" / "hsps.parquet", columns=["hsp_id", "btop", "query_side"])
        rows = hits.join(parents, on="hsp_id", how="left")
    else:
        if "btop" not in hits.columns:
            raise ExportError("passim-pairwise needs BLAST's btop; this run was made with --no-btop")
        rows = hits.with_columns(pl.lit(0).alias("col_start"), pl.col("length").alias("col_end"))
    rows = rows.select("btop", "query_side", "col_start", "col_end", "doc1", "enc_start1", "enc_end1", "doc2",
                       "enc_start2", "enc_end2")
    n_chunks = max(1, workers * 8)
    size = max(1, -(-rows.height // n_chunks))
    chunks = [rows.slice(i, size) for i in range(0, rows.height, size)]
    if workers > 1 and rows.height > 2000:
        with process_pool(max_workers=workers, initializer=_align_init, initargs=(str(run_dir),)) as pool:
            parts = list(pool.map(_align_chunk, chunks))
    else:
        _align_init(str(run_dir))
        parts = [_align_chunk(c) for c in chunks]
    return pl.concat(parts) if parts else pl.DataFrame(schema={"s1": pl.String, "s2": pl.String})


def export_passim_pairwise(ctx: Context) -> tuple[Path, int]:
    """passim 2.0's ``--pairwise`` records (``align.json``): uid/uid2, document fields with the ``2`` suffix,
    begin/end/begin2/end2, s1/s2 (aligned surface text, ``-`` padding) and matches (identical letters)."""
    if ctx.run.is_bundle:
        raise ExportError("passim-pairwise needs a run directory (the corpus store's maps)")
    hits, _ = kept_hits(ctx.run_dir, ctx.hit_filter, ctx.view)
    docs = ctx.run.documents.collect()
    fields = ["doc_id", *ctx.metadata_fields()]
    frame = hits.select("doc1", "doc2", pl.col("start1").alias("begin"), pl.col("end1").alias("end"),
                        pl.col("start2").alias("begin2"), pl.col("end2").alias("end2"),
                        pl.col("nident").alias("matches"))
    if ctx.options.excerpts == "surface":
        frame = frame.hstack(surface_alignments(ctx.run_dir, hits, ctx.workers))
    for k, suffix in ((1, ""), (2, "2")):
        meta = docs.select(pl.col("doc_idx").cast(pl.Int32).alias(f"doc{k}"),
                           *[pl.col(f).alias(("id" if f == "doc_id" else f) + suffix) for f in fields])
        frame = frame.join(meta, on=f"doc{k}", how="left")
    if ctx.options.offsets == "utf16":
        frame = frame.with_columns(
            pl.Series("begin", ctx.offsets(frame["doc1"].to_numpy(), frame["begin"].to_numpy())),
            pl.Series("end", ctx.offsets(frame["doc1"].to_numpy(), frame["end"].to_numpy())),
            pl.Series("begin2", ctx.offsets(frame["doc2"].to_numpy(), frame["begin2"].to_numpy())),
            pl.Series("end2", ctx.offsets(frame["doc2"].to_numpy(), frame["end2"].to_numpy())),
        )
    frame = frame.rename({"doc1": "uid", "doc2": "uid2"}).with_columns(pl.col("uid", "uid2").cast(pl.Int64))
    names = ["id" if f == "doc_id" else f for f in fields]
    order = ["uid", "uid2", *names, *[f"{n}2" for n in names], "begin", "end", "begin2", "end2",
             *(["s1", "s2"] if "s1" in frame.columns else []), "matches"]
    frame = frame.select(order)
    path = ctx.out_dir / "passim-pairwise.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    frame.write_ndjson(tmp)
    tmp.replace(path)
    return path, frame.height


# --------------------------------------------------------------------------- etl


def export_etl(ctx: Context) -> tuple[Path, int]:
    """The HPC-HD ETL input (``raw_textreuses.py``): one JSON line per hit, offsets 0-based and end-exclusive.

    The ETL expects raw hits: on the joined view, the members of the kept joined hits are written."""
    hits, _ = kept_hits(ctx.run_dir, ctx.hit_filter, ctx.view)
    if "joined_id" in hits.columns:
        log.info("etl: the joined view is written as its member hits (the ETL expects raw hits)")
        members = ctx.run.join_map.collect().join(hits.select("joined_id"), on="joined_id", how="semi")
        hits = ctx.run.hsps.collect().join(members.select("hsp_id"), on="hsp_id", how="semi")
    ids = ctx.run.documents.select(pl.col("doc_idx").cast(pl.Int32), "doc_id").collect()
    percent = "ppos" if "ppos" in hits.columns else "pident"
    frame = hits.select("doc1", "start1", "end1", "doc2", "start2", "end2", pl.col("length").alias("align_length"),
                        pl.col(percent).cast(pl.Float32).alias("positives_percent"))
    for k in (1, 2):
        frame = frame.join(ids.rename({"doc_idx": f"doc{k}", "doc_id": f"text{k}_id"}), on=f"doc{k}", how="left")
    if ctx.options.excerpts == "surface":
        for k in (1, 2):
            frame = frame.with_columns(pl.Series(f"text{k}_text", ctx.run.excerpt(
                frame[f"doc{k}"].to_numpy(), frame[f"start{k}"].to_numpy(), frame[f"end{k}"].to_numpy())))
    for k in (1, 2):
        frame = frame.with_columns(
            pl.Series(f"text{k}_text_start", ctx.offsets(frame[f"doc{k}"].to_numpy(), frame[f"start{k}"].to_numpy())),
            pl.Series(f"text{k}_text_end", ctx.offsets(frame[f"doc{k}"].to_numpy(), frame[f"end{k}"].to_numpy())),
        )
    order = ["align_length", "positives_percent"]
    for k in (1, 2):
        order += [f"text{k}_id", *([f"text{k}_text"] if ctx.options.excerpts == "surface" else []),
                  f"text{k}_text_end", f"text{k}_text_start"]
    frame = frame.select(order).with_columns(pl.col(r"^text\d_text_(start|end)$").cast(pl.Int32),
                                             pl.col("align_length").cast(pl.Int32))
    path = ctx.out_dir / "etl.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    frame.write_ndjson(tmp)
    tmp.replace(path)
    return path, frame.height


# --------------------------------------------------------------------------- passim-clusters


def merge_spans(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """passim's ``mergeSpans``: sorted spans merged while they overlap (``begin < curEnd``); returns the merged
    spans and, for each input span (in sorted order), the index of its merged span."""
    order = np.lexsort((end, start))
    s, e = start[order], end[order]
    new = np.ones(len(s), dtype=bool)
    if len(s) > 1:
        new[1:] = s[1:] >= np.maximum.accumulate(e)[:-1]
    group = np.cumsum(new) - 1
    m_start = s[new]
    m_end = np.zeros(int(group[-1]) + 1 if len(group) else 0, dtype=np.int64)
    np.maximum.at(m_end, group, e)
    back = np.empty(len(s), dtype=np.int64)
    back[order] = group
    return m_start, m_end, back


def export_passim_clusters(ctx: Context) -> tuple[Path, int]:
    """passim's cluster output (``out.json``): one record per passage of a cluster, i.e. the cluster's pieces in one
    document merged where they overlap (passim's ``mergeSpans``). ``size`` is the number of documents in the cluster,
    as in passim. passim's ``src`` is left out (pairs are undirected). ``pboiler`` is the share of the cluster's links
    whose two documents have the same value of ``export.boiler_field`` (passim's ``gid``), when it is set."""
    clusters = ctx.run.clusters.collect()
    pieces = ctx.run.pieces.select("piece_id", "doc", "start", "end").collect()
    docs = ctx.run.documents.collect()
    frame = clusters.join(pieces, on="piece_id").sort("cluster_id", "doc", "start", "end")
    # passim's mergeSpans per (cluster, document), vectorized: a span starts a new passage unless it begins before
    # the running maximum end of the spans before it in the same cluster and document
    out = (
        frame.with_columns(pl.col("end").cum_max().shift(1).over("cluster_id", "doc").alias("_reach"))
        .with_columns((pl.col("_reach").is_null() | (pl.col("start") >= pl.col("_reach"))).alias("_new"))
        .with_columns(pl.col("_new").cum_sum().alias("_passage"))
        .group_by("_passage", maintain_order=True)
        .agg(pl.col("cluster_id").first().cast(pl.Int64).alias("cluster"), pl.col("doc").first().cast(pl.Int64)
             .alias("uid"), pl.col("start").min().cast(pl.Int64).alias("begin"),
             pl.col("end").max().cast(pl.Int64).alias("end"))
        .drop("_passage")
    )
    size = out.group_by("cluster").agg(pl.col("uid").n_unique().alias("size"))
    out = out.join(size, on="cluster")
    series = ctx.options.boiler_field
    if series:
        if series not in docs.columns:
            raise ExportError(f"export.boiler_field: {series!r} is not a document field")
        name = "pairs_filtered" if ctx.hit_filter else "pairs"
        links = ctx.run.table(name).select("piece1", "doc1", "doc2").collect().join(
            clusters.rename({"piece_id": "piece1"}), on="piece1")
        value = docs.select(pl.col("doc_idx").cast(pl.Int32), pl.col(series).alias("_s"))
        links = links.join(value.rename({"doc_idx": "doc1", "_s": "_s1"}), on="doc1", how="left").join(
            value.rename({"doc_idx": "doc2", "_s": "_s2"}), on="doc2", how="left")
        boiler = links.group_by("cluster_id").agg(
            (pl.col("_s1") == pl.col("_s2")).fill_null(False).mean().cast(pl.Float32).alias("pboiler"))
        out = out.join(boiler.rename({"cluster_id": "cluster"}), on="cluster", how="left")
    fields = ["doc_id", *ctx.metadata_fields()]
    meta = docs.select(pl.col("doc_idx").cast(pl.Int64).alias("uid"),
                       *[pl.col(f).alias("id" if f == "doc_id" else f) for f in fields])
    out = out.join(meta, on="uid", how="left")
    try:
        overlaps = ctx.run.unit_overlaps.select("doc", "loc", "piece_id").collect()
    except RunNotAvailable:
        overlaps = None
    if overlaps is not None and overlaps.height:
        from .annotate import unit_overlaps

        units = ctx.run.units.select(pl.col("doc_idx").cast(pl.Int32).alias("doc"), "loc", "start",
                                     (pl.col("start") + pl.col("length")).alias("end")).collect().sort("doc", "start")
        spans = out.with_row_index("piece_id").select(pl.col("piece_id").cast(pl.Int64), pl.col("uid").cast(pl.Int32)
                                                      .alias("doc"), pl.col("begin").alias("start"), "end")
        found = unit_overlaps(spans, units)  # sorted interval join per document
        locs = found.join(units.select("doc", "loc", pl.col("start").alias("_s")), on=["doc", "loc"]).sort(
            "piece_id", "_s").group_by("piece_id", maintain_order=True).agg(pl.col("loc").alias("locs"))
        out = out.with_row_index("piece_id").with_columns(pl.col("piece_id").cast(pl.Int64)).join(
            locs, on="piece_id", how="left").drop("piece_id")
    if ctx.options.excerpts == "surface":
        out = out.with_columns(pl.Series("text", ctx.run.excerpt(out["uid"].to_numpy(), out["begin"].to_numpy(),
                                                                 out["end"].to_numpy())))
    if ctx.options.offsets == "utf16":
        out = out.with_columns(pl.Series("begin", ctx.offsets(out["uid"].to_numpy(), out["begin"].to_numpy())),
                               pl.Series("end", ctx.offsets(out["uid"].to_numpy(), out["end"].to_numpy())))
    first = ["cluster", "size", "uid", "id", "begin", "end"]
    out = out.sort("cluster", "uid", "begin").select(first + [c for c in out.columns if c not in first])
    path = ctx.out_dir / "passim-clusters.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    out.write_ndjson(tmp)
    tmp.replace(path)
    return path, out.height


# --------------------------------------------------------------------------- registry and command

CONVERTERS: dict[str, Converter] = {
    "bundle": Converter(export_bundle, ("hsps",)),
    "pairs_flat": Converter(export_pairs_flat, ("pairs", "pieces")),
    "passim-pairwise": Converter(export_passim_pairwise, ("hsps", "pieces")),
    "etl": Converter(export_etl, ("hsps",)),
    "passim-clusters": Converter(export_passim_clusters, ("clusters", "pieces")),
}


def _converter(name: str, options: ExportOptions) -> Converter:
    if name in options.converters:
        module, _, attr = options.converters[name].partition(":")
        try:
            func = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as error:
            raise ExportError(f"cannot load converter {name!r} ({options.converters[name]}): {error}") from error
        return Converter(func, tuple(getattr(func, "needs", ("hsps",))))
    if name not in CONVERTERS:
        available = ", ".join([*CONVERTERS, *options.converters])
        raise ExportError(f"unknown export format {name!r} (available: {available})")
    return CONVERTERS[name]


def export(run_dir: Path | str, formats: list[str], options: ExportOptions, workers: int | None = None
           ) -> dict[str, Any]:
    """Write each format to ``<out_dir>``; skipped when the tables and options are unchanged."""
    run_dir = Path(run_dir).expanduser()
    run = open_run(run_dir)
    manifest = read_manifest(run_dir)
    post_stage = manifest.get("stages", {}).get("post")
    if not post_stage:
        raise ExportError(f"no post-processed tables in {run_dir}; run `textblast post` first")
    if options.excerpts not in ("surface", "none") or options.offsets not in ("codepoints", "utf16"):
        raise ExportError("export.excerpts must be surface|none and export.offsets codepoints|utf16")
    post_settings = post_stage["settings"]
    view = options.view or post_settings["defrag"]["view"]
    if view == "clipped" and not post_settings["units"]["clip"]:
        raise ExportError("export.view: clipped needs units.clip: true (then run `post`)")
    if view == "joined" and not post_settings.get("join", {}).get("enabled"):
        raise ExportError("export.view: joined needs join.enabled: true (then run `post`)")
    if view not in ("hsps", "clipped", "joined"):
        raise ExportError(f"export.view must be hsps, clipped or joined, not {view!r}")
    out_dir = Path(options.out_dir).expanduser() if options.out_dir else run_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(run, run_dir, options, out_dir, post_settings.get("hit_filter"), view, workers or default_workers())
    results = {}
    done = manifest.get("stages", {}).get("export", {})
    for index, name in enumerate(formats, start=1):
        converter = _converter(name, options)
        missing = [t for t in converter.needs if not run._file(t).is_file()]
        if name == "passim-clusters" and "clusters" in missing:
            raise ExportError("passim-clusters needs clusters: set cluster.enabled: true and run `post`")
        if missing:
            raise ExportError(f"{name} needs tables {missing}; run `textblast post` (with the right settings)")
        fingerprint = hashlib.sha256(json.dumps(
            [post_stage["fingerprint"], name, asdict(options) | {"formats": None}, view, __version__],
            sort_keys=True, default=str).encode()).hexdigest()[:16]
        previous = done.get(name, {})
        if previous.get("fingerprint") == fingerprint and Path(previous.get("path", "")).exists():
            log.info(f"export {name}: {previous['path']} is up to date; nothing to do")
            results[name] = previous
            continue
        with Stage(index, len(formats), f"export {name}", log=log, show_memory=True) as stage:
            path, rows = converter.func(ctx)
            size = sum(f.stat().st_size for f in path.rglob("*")) if path.is_dir() else path.stat().st_size
            unit = "tables" if name == "bundle" else "rows"
            stage.summary = f"{format_count(rows)} {unit} · {format_bytes(size)} · {path}"
        results[name] = {"fingerprint": fingerprint, "path": str(path), "rows": rows, "bytes": size,
                         "seconds": round(stage.elapsed, 2), "excerpts": options.excerpts, "view": view}
        manifest = read_manifest(run_dir)
        manifest.setdefault("stages", {}).setdefault("export", {})[name] = results[name]
        write_manifest(run_dir, manifest)
    return results
