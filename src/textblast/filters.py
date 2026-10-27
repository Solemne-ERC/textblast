"""Pair filter and hit filter.

Both are Polars SQL expressions with passim's convention: a field of the first document is written as is
(``date``), the same field of the second document with the suffix ``2`` (``date2``).

- The **pair filter** looks at document metadata only. It is opt-in and deletes: ``plan`` skips jobs whose document
  pairs are all excluded, and ``merge`` applies it exactly to ``hsps``. It is evaluated on the distinct combinations
  of the fields it references, never on all document pairs.
- The **hit filter** looks at pair, hit and piece fields plus document metadata. It never deletes from the core
  tables; it selects what goes into exports and ``pairs_filtered``. Named filters (``NAMED_FILTERS``) can be used in
  it as ``@name``, alone or combined, e.g. ``@no_formulae AND length >= 50``.

Pairs are always oriented ``doc1 < doc2`` (input order), as in the tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import polars as pl


class FilterError(ValueError):
    """An invalid filter expression or an unknown field."""


# Hit filters with a name, for use as @name in filters.hit_filter. None is on by default (recall is paramount).
NAMED_FILTERS: dict[str, str] = {
    # formulae: pieces linked to 30 or more documents and shorter than 60 letters (in an expert review of a Latin
    # corpus, 19 of 20 such pieces were formulae; 45% fewer pairs for 0–3.4 points of recall)
    "no_formulae": "NOT ((doc_degree1 >= 30 AND piece_letters1 < 60) OR (doc_degree2 >= 30 AND piece_letters2 < 60))",
    # reuse mediated by the Bible: both pieces mostly covered by hits with Bible documents (needs the reference set
    # `bible` in annotate.reference_sets)
    "no_bible_mediated": "NOT (ref_overlap_bible1 >= 0.8 AND ref_overlap_bible2 >= 0.8)",
}
_NAMED = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")


def expand_named(expression: str) -> str:
    """Replace each ``@name`` by its named filter, in parentheses."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in NAMED_FILTERS:
            raise FilterError(f"unknown named filter @{name} (available: "
                              f"{', '.join('@' + n for n in NAMED_FILTERS)})")
        return f"({NAMED_FILTERS[name]})"

    return _NAMED.sub(replace, expression)


@dataclass
class FilterOptions:
    """Filters; nothing is filtered unless set."""

    series_field: str | None = None  # opt-in: skip pairs within one value of this field (`field <> field2`)
    pair_filter: str | None = None  # SQL over document metadata, e.g. "abs(date - date2) <= 300" (deletes)
    hit_filter: str | None = None  # SQL over pair/hit fields and metadata, e.g. "length >= 50" (exports only)


def pair_expression(options: FilterOptions) -> str | None:
    """The combined pair filter (``series_field`` and ``pair_filter`` joined with AND), or None."""
    parts = []
    if options.series_field:
        name = options.series_field
        parts.append(f'("{name}" <> "{name}2")')
    if options.pair_filter:
        parts.append(f"({options.pair_filter})")
    return " AND ".join(parts) if parts else None


def referenced_fields(expression: str, columns: list[str]) -> tuple[list[str], list[str]]:
    """Document fields the expression uses for the first and for the second document."""
    try:
        names = pl.sql_expr(expression).meta.root_names()
    except Exception as error:  # polars raises several exception types for SQL
        raise FilterError(f"invalid filter expression {expression!r}: {error}") from error
    first, second = [], []
    for name in names:
        if name in columns:
            first.append(name)
        elif name.endswith("2") and name[:-1] in columns:
            second.append(name[:-1])
        else:
            raise FilterError(f"filter {expression!r}: unknown field {name!r} (document fields: {', '.join(columns)})")
    return sorted(set(first)), sorted(set(second))


@dataclass
class PairFilter:
    """A pair filter evaluated on groups of documents with equal values in the referenced fields."""

    expression: str
    group1: np.ndarray  # per doc_idx: group for the first-document fields
    group2: np.ndarray  # per doc_idx: group for the second-document fields
    allowed: np.ndarray  # bool [n_groups1, n_groups2]

    def keeps(self, doc1: np.ndarray, doc2: np.ndarray) -> np.ndarray:
        """Whether each pair (doc1 < doc2) passes."""
        return self.allowed[self.group1[doc1], self.group2[doc2]]

    def job_needed(self, query: list[int], db: list[int]) -> bool:
        """Conservative: True if any document of ``query`` with any of ``db`` passes, in either order."""
        q, d = np.asarray(query), np.asarray(db)
        g1q, g2q = np.unique(self.group1[q]), np.unique(self.group2[q])
        g1d, g2d = np.unique(self.group1[d]), np.unique(self.group2[d])
        return bool(self.allowed[np.ix_(g1q, g2d)].any() or self.allowed[np.ix_(g1d, g2q)].any())

    def excluded_pairs(self) -> tuple[int, int]:
        """(excluded, total) document pairs, counted per group (no pair is materialized)."""
        n = len(self.group1)
        n1, n2 = self.allowed.shape
        counts = np.zeros((n1, n2), dtype=np.int64)
        seen = np.zeros(n1, dtype=np.int64)  # documents so far, per first-document group
        for b in range(n):
            counts[:, self.group2[b]] += seen
            seen[self.group1[b]] += 1
        total = n * (n - 1) // 2
        return int(counts[~self.allowed].sum()), total


def build_pair_filter(expression: str, documents: pl.DataFrame) -> PairFilter:
    """Evaluate ``expression`` on the distinct value combinations of the fields it references."""
    columns = [c for c in documents.columns if c != "doc_idx"]
    first, second = referenced_fields(expression, columns)
    docs = documents.sort("doc_idx")

    def groups(names: list[str]) -> tuple[np.ndarray, pl.DataFrame]:
        if not names:
            return np.zeros(docs.height, dtype=np.int64), pl.DataFrame({"_g": [0]})
        keys = docs.select(names).unique(maintain_order=True).with_row_index("_g")
        ids = docs.select(names).join(keys, on=names, how="left", nulls_equal=True)["_g"].to_numpy()
        return ids.astype(np.int64), keys

    group1, keys1 = groups(first)
    group2, keys2 = groups(second)
    left = keys1.rename({"_g": "_g1"})
    right = keys2.rename({c: f"{c}2" for c in second} | {"_g": "_g2"})
    combos = left.join(right, how="cross")
    try:
        passed = combos.filter(pl.sql_expr(expression).fill_null(False))
    except Exception as error:
        raise FilterError(f"pair filter {expression!r} failed: {error}") from error
    allowed = np.zeros((keys1.height, keys2.height), dtype=bool)
    allowed[passed["_g1"].to_numpy(), passed["_g2"].to_numpy()] = True
    return PairFilter(expression, group1, group2, allowed)


def with_metadata(frame: pl.LazyFrame, documents: pl.DataFrame, fields: list[str] | None = None) -> pl.LazyFrame:
    """Add document fields to a frame with ``doc1``/``doc2``: ``field`` for doc1, ``field2`` for doc2."""
    names = [c for c in documents.columns if c != "doc_idx"] if fields is None else fields
    taken = set(frame.collect_schema().names())
    meta = documents.lazy().with_columns(pl.col("doc_idx").cast(pl.Int32))
    left = meta.select(pl.col("doc_idx").alias("doc1"), *[c for c in names if c not in taken])
    right = meta.select(pl.col("doc_idx").alias("doc2"), *[pl.col(c).alias(f"{c}2") for c in names
                                                           if f"{c}2" not in taken])
    return frame.join(left, on="doc1", how="left").join(right, on="doc2", how="left")


def apply_hit_filter(frame: pl.LazyFrame, expression: str | None, documents: pl.DataFrame) -> pl.LazyFrame:
    """Rows of ``frame`` (with ``doc1``/``doc2``) that pass the hit filter; metadata columns are not added."""
    if not expression:
        return frame
    sql = expand_named(expression)
    columns = frame.collect_schema().names()
    enriched = with_metadata(frame, documents)
    try:
        condition = pl.sql_expr(sql)
    except Exception as error:
        raise FilterError(f"hit filter {expression!r} failed: {error}") from error
    available = set(enriched.collect_schema().names())
    missing = [name for name in condition.meta.root_names() if name not in available]
    if missing:  # caught here: the frame is lazy, so Polars would only fail later, deep in the query
        hint = ""
        sets = sorted({m.removeprefix("ref_overlap_")[:-1] for m in missing if m.startswith("ref_overlap_")})
        if sets:
            hint = f"; add the reference set{'s' * (len(sets) > 1)} {', '.join(sets)} to annotate.reference_sets"
        raise FilterError(f"hit filter {expression!r} uses unknown columns: {', '.join(missing)}{hint}")
    return enriched.filter(condition.fill_null(False)).select(columns)
