"""Helpers shared by the normalization tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from textblast.normalize import SEP, DocContext, Normalizer, Spanned
from textblast.normalize.profiles import get_profile

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "corpus.jsonl"


def run(profile: str, text: str, **ctx_kwargs) -> tuple[Spanned, DocContext]:
    ctx = DocContext(0, "doc", len(text), **ctx_kwargs)
    return Normalizer(get_profile(profile).steps).run(text, ctx), ctx


def letters(profile: str, text: str) -> str:
    """Normalized text with separators collapsed to single spaces (easy to read in assertions)."""
    doc, _ = run(profile, text)
    return " ".join(doc.text.replace("", SEP).split())


def pairs(profile: str, text: str) -> list[tuple[str, str]]:
    """(letter, surface text it maps to) for every letter."""
    doc, _ = run(profile, text)
    return [
        (c, text[a:b]) for c, a, b in zip(doc.text, doc.start.tolist(), doc.end.tolist(), strict=True) if c.isalpha()
    ]


def halves_rule(pieces):
    """A test defragmentation plugin: raw pieces in the first and second half of the document form two groups."""
    import polars as pl

    middle = (pieces["start"].min() + pieces["end"].max()) // 2
    return pieces.select("raw_piece_id", (pl.col("start") >= middle).cast(pl.Int64).alias("group"))


def one_cluster(pieces: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """A cluster plugin putting every linked piece into one cluster."""
    linked = pl.concat([pairs.select(pl.col("piece1").alias("piece_id")),
                        pairs.select(pl.col("piece2").alias("piece_id"))]).unique()
    return linked.with_columns(pl.lit(1).alias("cluster"))
