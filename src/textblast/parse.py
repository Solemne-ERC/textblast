"""BLAST tabular output → raw hits with exact surface offsets.

BLAST reports 1-based inclusive coordinates in the encoded letters; they are converted here, exactly once, to 0-based
end-exclusive encoded coordinates and mapped to the surface text through the corpus store. Sides are ordered so that
``doc1 < doc2``; ``query_side`` (1 or 2) records which side was BLAST's query, because ``btop`` is written from the
query's point of view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .mapping import CorpusStore

BLAST_COLUMNS = {
    "qseqid": pl.String, "sseqid": pl.String, "qstart": pl.Int64, "qend": pl.Int64, "sstart": pl.Int64,
    "send": pl.Int64, "length": pl.Int32, "nident": pl.Int32, "mismatch": pl.Int32, "gapopen": pl.Int32,
    "gaps": pl.Int32, "pident": pl.Float32, "ppos": pl.Float32, "score": pl.Int32, "bitscore": pl.Float32,
    "evalue": pl.Float64, "btop": pl.String,
}

HSP_SCHEMA = {
    "doc1": pl.Int32, "doc2": pl.Int32,
    "start1": pl.Int32, "end1": pl.Int32, "start2": pl.Int32, "end2": pl.Int32,
    "enc_start1": pl.Int32, "enc_end1": pl.Int32, "enc_start2": pl.Int32, "enc_end2": pl.Int32,
    "length": pl.Int32, "nident": pl.Int32, "mismatch": pl.Int32, "gapopen": pl.Int32, "gaps": pl.Int32,
    "pident": pl.Float32, "ppos": pl.Float32, "score": pl.Int32, "bitscore": pl.Float32, "evalue": pl.Float64,
    "btop": pl.String, "query_side": pl.Int8, "job": pl.Int32,
}


def read_blast_tsv(path: Path, btop: bool = True) -> pl.DataFrame:
    """BLAST ``-outfmt 6`` output with the columns of :data:`BLAST_COLUMNS`."""
    columns = dict(BLAST_COLUMNS)
    if not btop:
        columns.pop("btop")
    if path.stat().st_size == 0:
        return pl.DataFrame(schema=columns)
    return pl.read_csv(
        path, separator="\t", has_header=False, new_columns=list(columns), schema_overrides=columns,
        quote_char=None,
    )


def to_hsps(raw: pl.DataFrame, store: CorpusStore, job_id: int, btop: bool = True) -> pl.DataFrame:
    """Raw BLAST rows → hits with surface offsets, sides ordered ``doc1 < doc2`` (schema :data:`HSP_SCHEMA`)."""
    schema = dict(HSP_SCHEMA)
    if not btop:
        schema.pop("btop")
    if raw.is_empty():
        return pl.DataFrame(schema=schema)
    q_doc = raw["qseqid"].str.slice(1).cast(pl.Int64).to_numpy()
    s_doc = raw["sseqid"].str.slice(1).cast(pl.Int64).to_numpy()
    if np.any(q_doc == s_doc):
        raise ValueError(f"job {job_id}: a document was aligned with itself (planning error)")
    q0, q1 = raw["qstart"].to_numpy() - 1, raw["qend"].to_numpy()  # 1-based inclusive → 0-based exclusive
    s0, s1 = raw["sstart"].to_numpy() - 1, raw["send"].to_numpy()
    q_start, q_end = store.to_surface(q_doc, q0, q1)
    s_start, s_end = store.to_surface(s_doc, s0, s1)

    swap = q_doc > s_doc  # the query becomes side 2
    def side(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.where(swap, b, a), np.where(swap, a, b)

    doc1, doc2 = side(q_doc, s_doc)
    start1, start2 = side(q_start, s_start)
    end1, end2 = side(q_end, s_end)
    enc_start1, enc_start2 = side(q0, s0)
    enc_end1, enc_end2 = side(q1, s1)
    columns = {
        "doc1": doc1, "doc2": doc2, "start1": start1, "end1": end1, "start2": start2, "end2": end2,
        "enc_start1": enc_start1, "enc_end1": enc_end1, "enc_start2": enc_start2, "enc_end2": enc_end2,
    }
    scores = raw.select("length", "nident", "mismatch", "gapopen", "gaps", "pident", "ppos", "score", "bitscore",
                        "evalue", *(["btop"] if btop else []))
    frame = pl.DataFrame(columns).hstack(scores).with_columns(
        pl.Series("query_side", np.where(swap, 2, 1)),
        pl.lit(job_id).alias("job"),
    )
    return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
