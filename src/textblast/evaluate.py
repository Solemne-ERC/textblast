"""Recall against expected pairs: ``textblast evaluate``.

Generic: expected pairs name documents by ``doc_id`` and give spans in code points of the input text, so one file
serves every run of the same input. Format (Parquet; README, "Evaluating against expected pairs"):

- **expected pairs**, one row each: ``ref_id``, ``kind``, ``relation`` (``direct`` | ``cf``), ``doc_id1``, ``start1``,
  ``end1`` (the annotated passage), ``doc_id2``, ``start2``, ``end2`` (the target); optional ``similarity`` (0–1, how
  much of the text the two sides share), ``target_group`` (a ``doc_id`` prefix such as ``bible:``), ``exact1`` /
  ``exact2`` (whether that side's boundaries are the reused text's own, default true; overshoot is measured only on
  exact sides), and any other columns;
- **alternative targets** (optional): ``ref_id``, ``doc_id2``, ``start2``, ``end2``, ``match`` (a label such as
  ``near`` or ``parallel``).

Each expected pair gets the best of: **exact** (a hit overlaps the passage on one side and the target on the other),
**alternative** (the same against an alternative target), **elsewhere** (a hit overlaps the passage, with a document
of the target group), **missed**. Hits may have the two sides in either order.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .api import RunNotAvailable, open_run
from .logs import logger

log = logger
LEVELS = ("hits", "clipped", "joined", "pairs")
RESULTS = ("exact", "alternative", "elsewhere", "missed")
REQUIRED = {"ref_id": pl.String, "kind": pl.String, "relation": pl.String, "doc_id1": pl.String, "start1": pl.Int64,
            "end1": pl.Int64, "doc_id2": pl.String, "start2": pl.Int64, "end2": pl.Int64}
ALTERNATIVE_COLUMNS = {"ref_id": pl.String, "doc_id2": pl.String, "start2": pl.Int64, "end2": pl.Int64,
                       "match": pl.String}


class EvaluateError(ValueError):
    """Expected pairs or alternatives that cannot be used."""


@dataclass
class EvaluateOptions:
    level: str = "hits"  # hits | clipped | pairs
    length_bands: list[int] = field(default_factory=lambda: [50, 150, 500])  # passage length, characters
    similarity_bands: list[float] = field(default_factory=lambda: [0.5, 0.8])


# ---------------------------------------------------------------------------------------------------- inputs


def load_expected(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise EvaluateError(f"{path}: missing columns {', '.join(missing)} (see the README for the format)")
    frame = frame.with_columns([pl.col(c).cast(t) for c, t in REQUIRED.items()])
    defaults = {"similarity": pl.lit(None, pl.Float64), "target_group": pl.lit(None, pl.String),
                "exact1": pl.lit(True), "exact2": pl.lit(True)}
    frame = frame.with_columns([v.alias(k) for k, v in defaults.items() if k not in frame.columns])
    frame = frame.with_columns(pl.col("similarity").cast(pl.Float64), pl.col("exact1").cast(pl.Boolean),
                               pl.col("exact2").cast(pl.Boolean))
    if frame["ref_id"].n_unique() != frame.height:
        raise EvaluateError(f"{path}: ref_id must be unique")
    bad = frame.filter((pl.col("end1") <= pl.col("start1")) | (pl.col("end2") <= pl.col("start2"))
                       | (pl.col("start1") < 0) | (pl.col("start2") < 0))
    if bad.height:
        raise EvaluateError(f"{path}: {bad.height} rows with empty or negative spans (e.g. {bad['ref_id'][0]})")
    return frame


def load_alternatives(path: Path | None) -> pl.DataFrame:
    if path is None:
        return pl.DataFrame(schema=ALTERNATIVE_COLUMNS)
    frame = pl.read_parquet(path)
    missing = [c for c in ALTERNATIVE_COLUMNS if c not in frame.columns]
    if missing:
        raise EvaluateError(f"{path}: missing columns {', '.join(missing)}")
    return frame.select([pl.col(c).cast(t) for c, t in ALTERNATIVE_COLUMNS.items()])


def segments(run: Any, level: str) -> pl.DataFrame:
    """``seg_id, doc1, start1, end1, doc2, start2, end2, evalue`` at the chosen level (surface offsets)."""
    cols = ["doc1", "start1", "end1", "doc2", "start2", "end2"]
    if level == "hits":
        frame = run.hsps.select(pl.col("hsp_id").alias("seg_id"), *cols, "evalue").collect()
    elif level == "clipped":
        clipped = run.hsps_clipped.select(pl.col("clip_id").alias("seg_id"), "hsp_id", *cols)
        evalue = run.hsps.select("hsp_id", "evalue")
        frame = clipped.join(evalue, on="hsp_id", how="left").drop("hsp_id").collect()
    elif level == "joined":
        frame = run.joined.select(pl.col("joined_id").alias("seg_id"), *cols, "evalue").collect()
    elif level == "pairs":
        pieces = run.pieces.select("piece_id", "doc", "start", "end")
        frame = (
            run.pairs.select("pair_id", "piece1", "piece2", "evalue")
            .join(pieces.rename({"piece_id": "piece1", "doc": "doc1", "start": "start1", "end": "end1"}), on="piece1")
            .join(pieces.rename({"piece_id": "piece2", "doc": "doc2", "start": "start2", "end": "end2"}), on="piece2")
            .select(pl.col("pair_id").alias("seg_id"), *cols, "evalue")
            .collect()
        )
    else:
        raise EvaluateError(f"unknown level {level!r} (choose from {', '.join(LEVELS)})")
    return frame.with_columns([pl.col(c).cast(pl.Int64) for c in cols])


# ---------------------------------------------------------------------------------------------------- scoring


def interval_join(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """``left_row, right_row`` for every overlap of ``left`` and ``right`` (columns ``doc, start, end``), per document.

    Right intervals are sorted by start with a running maximum of their ends, so the candidates of a left interval are
    one contiguous range; candidates that do not overlap are dropped.
    """
    out_l, out_r = [], []
    right = right.with_row_index("_row").sort("doc", "start")
    by_doc = {key[0]: part for key, part in right.group_by(["doc"])}
    for (d,), part in left.with_row_index("_row").group_by(["doc"]):
        r = by_doc.get(d)
        if r is None:
            continue
        r_start, r_end, r_row = r["start"].to_numpy(), r["end"].to_numpy(), r["_row"].to_numpy()
        reach = np.maximum.accumulate(r_end)
        l_start, l_end, l_row = part["start"].to_numpy(), part["end"].to_numpy(), part["_row"].to_numpy()
        lo = np.searchsorted(reach, l_start, side="right")
        hi = np.searchsorted(r_start, l_end, side="left")
        count = np.maximum(hi - lo, 0)
        if not count.sum():
            continue
        li = np.repeat(np.arange(len(l_start)), count)
        ri = np.concatenate([np.arange(a, b) for a, b in zip(lo[count > 0], hi[count > 0], strict=True)])
        keep = (r_start[ri] < l_end[li]) & (r_end[ri] > l_start[li])
        out_l.append(l_row[li[keep]])
        out_r.append(r_row[ri[keep]])
    if not out_l:
        return pl.DataFrame(schema={"left_row": pl.UInt32, "right_row": pl.UInt32})
    return pl.DataFrame({"left_row": np.concatenate(out_l).astype(np.uint32),
                         "right_row": np.concatenate(out_r).astype(np.uint32)})


def _overlaps(s1: pl.Expr, e1: pl.Expr, s2: pl.Expr, e2: pl.Expr) -> pl.Expr:
    return (s1 < e2) & (e1 > s2)


def score(expected: pl.DataFrame, alternatives: pl.DataFrame, segs: pl.DataFrame, doc_ids: list[str]) -> pl.DataFrame:
    """One result row per expected pair (see the module docstring)."""
    index = {d: i for i, d in enumerate(doc_ids)}
    exp = expected.with_row_index("r").with_columns(
        pl.col("doc_id1").replace_strict(index, default=-1, return_dtype=pl.Int64).alias("a"),
        pl.col("doc_id2").replace_strict(index, default=-1, return_dtype=pl.Int64).alias("b"),
    )
    # both orientations of every segment: side A is matched against the passage, side B against the target
    names = ["seg_id", "evalue"]
    oriented = pl.concat([
        segs.select(*names, pl.col("doc1").alias("doc"), pl.col("start1").alias("start"), pl.col("end1").alias("end"),
                    pl.col("doc2").alias("bdoc"), pl.col("start2").alias("bs"), pl.col("end2").alias("be")),
        segs.select(*names, pl.col("doc2").alias("doc"), pl.col("start2").alias("start"), pl.col("end2").alias("end"),
                    pl.col("doc1").alias("bdoc"), pl.col("start1").alias("bs"), pl.col("end1").alias("be")),
    ])
    pairs = interval_join(exp.select(pl.col("a").alias("doc"), "start1", "end1").rename(
        {"start1": "start", "end1": "end"}), oriented.select("doc", "start", "end"))
    cand = (
        pairs.join(oriented.with_row_index("right_row"), on="right_row")
        .join(exp.select(pl.col("r").alias("left_row"), "r", "b", "start1", "end1", "start2", "end2",
                         "target_group"), on="left_row")
        .drop("left_row", "right_row")
        .with_columns(pl.col("bdoc").replace_strict(dict(enumerate(doc_ids)), return_dtype=pl.String).alias("bdoc_id"))
    )
    cand = cand.with_columns(
        ((pl.col("bdoc") == pl.col("b")) & _overlaps(pl.col("bs"), pl.col("be"), pl.col("start2"), pl.col("end2")))
        .alias("is_exact"),
        pl.when(pl.col("target_group").is_null()).then(False)
        .otherwise(pl.col("bdoc_id").str.starts_with(pl.col("target_group"))).alias("in_group"),
    )
    # alternatives: the same test against each alternative target of the pair
    alt = alternatives.with_columns(
        pl.col("doc_id2").replace_strict(index, default=-1, return_dtype=pl.Int64).alias("alt_doc"))
    alt = alt.join(exp.select("ref_id", "r"), on="ref_id").select(
        "r", "alt_doc", pl.col("start2").alias("alt_s"), pl.col("end2").alias("alt_e"), "match")
    alt_hits = (
        cand.select("r", "seg_id", "bdoc", "bs", "be").join(alt, on="r")
        .filter((pl.col("bdoc") == pl.col("alt_doc"))
                & _overlaps(pl.col("bs"), pl.col("be"), pl.col("alt_s"), pl.col("alt_e")))
        .group_by("r", "seg_id").agg(pl.col("match").first())
    )
    cand = cand.join(alt_hits, on=["r", "seg_id"], how="left").with_columns(
        pl.when(pl.col("is_exact")).then(pl.lit(0))
        .when(pl.col("match").is_not_null()).then(pl.lit(1))
        .when(pl.col("in_group")).then(pl.lit(2))
        .otherwise(pl.lit(3)).alias("cls"),
        (pl.min_horizontal("end", "end1") - pl.max_horizontal("start", "start1")).alias("ov1"),
    )
    best = cand.filter(pl.col("cls") < 3).filter(pl.col("cls") == pl.col("cls").min().over("r"))
    main = best.sort("r", "ov1", "evalue", descending=[False, True, False]).group_by("r").first()
    agg = best.group_by("r").agg(pl.len().alias("n_hits"), pl.col("evalue").min().alias("best_evalue"))
    coverage = _coverage(best)
    out = (
        exp.join(agg, on="r", how="left").join(coverage, on="r", how="left")
        .join(main.select("r", "match", pl.col("cls").alias("best"), pl.col("seg_id").alias("main_seg_id"),
                          pl.col("start").alias("ms1"), pl.col("end").alias("me1"),
                          pl.col("bs").alias("ms2"), pl.col("be").alias("me2")), on="r", how="left")
    )
    exact_main = pl.col("best") == 0

    def overshoot(side: str, ms: str, me: str, s: str, e: str) -> list[pl.Expr]:
        use = exact_main & pl.col(f"exact{side}")
        left = (pl.col(s) - pl.col(ms)).clip(0)
        right = (pl.col(me) - pl.col(e)).clip(0)
        return [pl.when(use).then(left).alias(f"overshoot_left{side}"),
                pl.when(use).then(right).alias(f"overshoot_right{side}"),
                pl.when(use).then((left + right) / (pl.col(me) - pl.col(ms))).alias(f"overshoot_share{side}")]

    out = out.with_columns(
        pl.when(pl.col("a") < 0).then(pl.lit("unknown_doc"))
        .otherwise(pl.col("best").replace_strict({0: "exact", 1: "alternative", 2: "elsewhere"}, default="missed",
                                                 return_dtype=pl.String)).alias("result"),
        (pl.col("end1") - pl.col("start1")).alias("length1"),
        pl.col("n_hits").fill_null(0),
        pl.col("coverage1").fill_null(0.0),
        *overshoot("1", "ms1", "me1", "start1", "end1"),
        *overshoot("2", "ms2", "me2", "start2", "end2"),
    )
    return out.select(
        "ref_id", "kind", "relation", "doc_id1", "start1", "end1", "doc_id2", "start2", "end2", "length1",
        "similarity", "result", "match", "n_hits", "best_evalue", "coverage1", "main_seg_id",
        "overshoot_left1", "overshoot_right1", "overshoot_share1",
        "overshoot_left2", "overshoot_right2", "overshoot_share2",
    ).sort("ref_id")


def _coverage(best: pl.DataFrame) -> pl.DataFrame:
    """Share of each passage covered by the A sides of its best-class hits (union of intervals)."""
    if best.is_empty():
        return pl.DataFrame(schema={"r": pl.UInt32, "coverage1": pl.Float64})
    clipped = best.select(
        "r", pl.max_horizontal("start", "start1").alias("s"), pl.min_horizontal("end", "end1").alias("e"),
        (pl.col("end1") - pl.col("start1")).alias("n"),
    ).sort("r", "s")
    clipped = clipped.with_columns(pl.col("e").cum_max().shift(1).over("r").alias("reach"))
    clipped = clipped.with_columns(
        (pl.col("e") - pl.max_horizontal("s", pl.col("reach").fill_null(pl.col("s")))).clip(0).alias("new"))
    return clipped.group_by("r").agg((pl.col("new").sum() / pl.col("n").first()).alias("coverage1"))


# ---------------------------------------------------------------------------------------------------- summary


def _band(column: str, edges: list[float]) -> pl.Expr:
    """Band labels ``< a``, ``a–b``, …, ``≥ z`` (and ``unknown``), as an Enum that sorts in band order."""
    labels = [f"< {edges[0]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(edges, edges[1:], strict=False)] \
        + [f"≥ {edges[-1]:g}"]
    expr: Any = pl.when(pl.col(column).is_null()).then(pl.lit("unknown"))
    for label, upper in zip(labels[:-1], edges, strict=True):
        expr = expr.when(pl.col(column) < upper).then(pl.lit(label))
    return expr.otherwise(pl.lit(labels[-1])).cast(pl.Enum([*labels, "unknown"]))


def recall_table(results: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Counts per result class and the recall shares for each group."""
    counts = [(pl.col("result") == r).sum().alias(r) for r in RESULTS]
    return (
        results.filter(pl.col("result") != "unknown_doc")
        .group_by(by).agg(pl.len().alias("n"), *counts)
        .with_columns(
            (pl.col("exact") / pl.col("n")).alias("recall"),
            ((pl.col("exact") + pl.col("alternative")) / pl.col("n")).alias("recall_alt"),
            ((pl.col("n") - pl.col("missed")) / pl.col("n")).alias("found_any"),
        )
        .sort(by)
    )


def overshoot_table(results: pl.DataFrame) -> pl.DataFrame:
    """Overshoot of the main exact hit, per kind and side (exact sides only)."""
    rows = []
    for side in ("1", "2"):
        part = results.filter(pl.col(f"overshoot_share{side}").is_not_null())
        if part.is_empty():
            continue
        chars = pl.col(f"overshoot_left{side}") + pl.col(f"overshoot_right{side}")
        share = pl.col(f"overshoot_share{side}")
        rows.append(part.group_by("kind", "relation").agg(
            pl.lit(int(side)).alias("side"), pl.len().alias("n"),
            chars.median().alias("chars_median"), chars.quantile(0.9).alias("chars_p90"),
            share.median().alias("share_median"), share.quantile(0.9).alias("share_p90"),
            (share > 0.2).mean().alias("over_20pct"), (chars == 0).mean().alias("none"),
        ))
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows).sort("kind", "relation", "side")


def summarize(results: pl.DataFrame, options: EvaluateOptions) -> dict[str, pl.DataFrame]:
    banded = results.with_columns(
        _band("length1", [float(x) for x in options.length_bands]).alias("length_band"),
        _band("similarity", options.similarity_bands).alias("similarity_band"),
    )
    return {
        "by_kind": recall_table(banded, ["kind", "relation"]),
        "by_length": recall_table(banded, ["kind", "relation", "length_band"]),
        "by_similarity": recall_table(banded, ["kind", "relation", "similarity_band"]),
        "overshoot": overshoot_table(results),
    }


# ---------------------------------------------------------------------------------------------------- command


def evaluate(run_dir: Path, expected_path: Path, alternatives_path: Path | None = None,
             options: EvaluateOptions | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    """Score a run against expected pairs; write ``results.parquet`` and ``summary.json`` to ``out_dir``."""
    options = options or EvaluateOptions()
    started = time.perf_counter()
    run = open_run(run_dir)
    expected = load_expected(expected_path)
    alternatives = load_alternatives(alternatives_path)
    try:
        segs = segments(run, options.level)
    except RunNotAvailable as error:
        raise EvaluateError(str(error)) from error
    doc_ids = run.documents.sort("doc_idx").collect()["doc_id"].to_list()
    results = score(expected, alternatives, segs, doc_ids)
    tables = summarize(results, options)
    unknown = results.filter(pl.col("result") == "unknown_doc").height
    if unknown:
        log.warning(f"{unknown:,} expected pairs name documents that are not in this run (left out)")
    out_dir = out_dir or Path(run_dir) / "evaluate" / f"{expected_path.stem}-{options.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.write_parquet(out_dir / "results.parquet")
    summary = {
        "run_dir": str(run_dir), "expected": str(expected_path),
        "alternatives": str(alternatives_path) if alternatives_path else None, "level": options.level,
        "segments": segs.height, "expected_pairs": expected.height, "unknown_doc": unknown,
        "seconds": round(time.perf_counter() - started, 2),
        "tables": {name: table.to_dicts() for name, table in tables.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log.info(f"evaluate: {expected.height:,} expected pairs against {segs.height:,} {options.level} "
             f"({summary['seconds']} s) → {out_dir}")
    for row in tables["by_kind"].iter_rows(named=True):
        log.info(f"  {row['kind']:<14} {row['relation']:<7} n {row['n']:>6,}  recall {row['recall']:6.1%}  "
                 f"+alternatives {row['recall_alt']:6.1%}  any {row['found_any']:6.1%}  "
                 f"(exact {row['exact']:,} · alternative {row['alternative']:,} · elsewhere {row['elsewhere']:,} · "
                 f"missed {row['missed']:,})")
    return summary
