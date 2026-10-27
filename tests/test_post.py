"""Post-processing: filters, defragmentation, btop and clipping, annotations, alignment strings, API (no engine)."""

from __future__ import annotations

import ast
import itertools
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from textblast.align import Columns, column_stats, parse_btop, segment_starts, surface_strings
from textblast.annotate import unit_at, unit_overlaps
from textblast.api import RunNotAvailable, open_run
from textblast.clip import split_hit
from textblast.defrag import DefragOptions, defragment, etl_mapping, node_similarity_groups
from textblast.filters import (
    NAMED_FILTERS,
    FilterError,
    FilterOptions,
    apply_hit_filter,
    build_pair_filter,
    expand_named,
    pair_expression,
    referenced_fields,
)

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference" / "code" / "textreuse-blast-master"

# --------------------------------------------------------------------------- filters

DOCS = pl.DataFrame({
    "doc_idx": [0, 1, 2, 3, 4],
    "doc_id": ["a", "b", "c", "d", "e"],
    "series": ["x", "x", "y", "y", None],
    "date": [1000, 1100, 1500, None, 1050],
})


def test_filter_fields_follow_passim_convention() -> None:
    assert referenced_fields("abs(date - date2) <= 300 AND series <> series2", DOCS.columns) == (
        ["date", "series"], ["date", "series"])
    with pytest.raises(FilterError, match="unknown field 'year'"):
        referenced_fields("year > 1000", DOCS.columns)
    with pytest.raises(FilterError, match="invalid filter"):
        referenced_fields("date >", DOCS.columns)
    assert pair_expression(FilterOptions()) is None
    assert pair_expression(FilterOptions(series_field="series", pair_filter="date < date2")) == (
        '("series" <> "series2") AND (date < date2)')


@pytest.mark.parametrize("expression", ['"series" <> "series2"', "abs(date - date2) <= 300", "date < date2",
                                        "series = 'x' OR series2 = 'x'"])
def test_pair_filter_matches_brute_force(expression: str) -> None:
    pf = build_pair_filter(expression, DOCS)
    pairs = list(itertools.combinations(range(5), 2))
    cross = pl.DataFrame([{**{k: DOCS[k][a] for k in ("series", "date")},
                           **{f"{k}2": DOCS[k][b] for k in ("series", "date")}} for a, b in pairs])
    expected = cross.select(pl.sql_expr(expression).fill_null(False))[:, 0].to_list()
    got = pf.keeps(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])).tolist()
    assert got == expected  # nulls fail the filter
    assert pf.excluded_pairs() == (expected.count(False), 10)
    # a job is needed iff some (query, db) pair passes in either order (conservative, never loses a pair)
    for q, d in ([0], [1]), ([0, 1], [2, 3]), ([2], [3, 4]):
        any_pass = any(expected[pairs.index((min(a, b), max(a, b)))] for a in q for b in d)
        assert pf.job_needed(q, d) >= any_pass


def test_hit_filter_sees_metadata_and_never_adds_columns() -> None:
    hits = pl.DataFrame({"hsp_id": [0, 1, 2], "doc1": [0, 0, 2], "doc2": [1, 2, 3], "length": [40, 80, 120]},
                        schema={"hsp_id": pl.Int64, "doc1": pl.Int32, "doc2": pl.Int32, "length": pl.Int32})
    kept = apply_hit_filter(hits.lazy(), "length >= 50 AND series <> series2", DOCS).collect()
    assert kept["hsp_id"].to_list() == [1] and kept.columns == hits.columns
    assert apply_hit_filter(hits.lazy(), None, DOCS).collect().equals(hits)


def test_named_filters() -> None:
    assert expand_named("@no_formulae AND length >= 50") == f"({NAMED_FILTERS['no_formulae']}) AND length >= 50"
    with pytest.raises(FilterError, match="@no_such"):
        expand_named("@no_such")
    rows = pl.DataFrame({
        "hsp_id": [0, 1, 2, 3], "doc1": [0, 0, 1, 2], "doc2": [1, 2, 2, 3],
        "doc_degree1": [40, 40, 5, 5], "piece_letters1": [30, 200, 30, 30],
        "doc_degree2": [3, 3, 50, 5], "piece_letters2": [30, 30, 59, 30],
        "ref_overlap_bible1": [0.0, 0.9, 0.9, 0.0], "ref_overlap_bible2": [0.0, 0.9, 0.5, 0.0],
    }, schema_overrides={"doc1": pl.Int32, "doc2": pl.Int32})
    # 0: a short, high-degree side (a formula); 2: the same on side 2; 1: high degree but long; 3: neither
    kept = apply_hit_filter(rows.lazy(), "@no_formulae", DOCS).collect().sort("hsp_id")
    assert kept["hsp_id"].to_list() == [1, 3]
    both = apply_hit_filter(rows.lazy(), "@no_formulae AND @no_bible_mediated", DOCS).collect()
    assert both["hsp_id"].to_list() == [3]  # 1 is mediated by the Bible on both sides
    # without the reference set, the error names the setting (not a failure deep in the lazy query)
    with pytest.raises(FilterError, match="ref_overlap_bible1.*add the reference set bible to annotate"):
        apply_hit_filter(rows.drop("ref_overlap_bible1", "ref_overlap_bible2").lazy(), "@no_bible_mediated", DOCS)
    with pytest.raises(FilterError, match="unknown columns: lenght"):
        apply_hit_filter(rows.lazy(), "lenght > 5", DOCS)


# --------------------------------------------------------------------------- the ETL rule


def etl_oracle(rows: list[tuple[int, int, int]]) -> dict[int, int]:
    """Literal transcription of the HPC-HD ETL's GetPieceIdMapping (reduce, then finish) over one document."""
    out: dict[int, int] = {}
    to_check: list[tuple[int, int, int]] = []
    for t_start, t_end, piece_id in rows:  # ORDER BY trs_start, piece_id
        idx = next((k for k, r in enumerate(to_check) if r[0] >= t_start - 180), -1)
        if idx < 0:
            to_check.clear()
        else:
            del to_check[0:idx]
        to_check.append((t_start, t_end, piece_id))
        t_start, t_end, piece_id = to_check[-1]
        for r in to_check:
            limit = min(max(min(t_end - t_start, r[1] - r[0]) // 4, 10), 180)
            if abs(r[0] - t_start) <= limit and abs(r[1] - t_end) <= limit:
                out[piece_id] = r[2]
                break
    return out


def run_etl(spans: list[tuple[int, int, int]]) -> list[int]:
    """(doc, start, end) → mapped index, with the vectorized rule (spans must be distinct)."""
    spans = sorted(spans)
    doc, start, end = (np.array(c) for c in zip(*spans, strict=True))
    return etl_mapping(doc, start, end).tolist()


@pytest.mark.parametrize("seed", range(8))
def test_etl_matches_the_literal_rule(seed: int) -> None:
    rng = random.Random(seed)
    spans = set()
    while len(spans) < 300:
        doc, s = rng.randint(0, 3), rng.randint(0, 3000)
        spans.add((doc, s, s + rng.choice([5, 20, 40, 80, 300, 900])))
    spans = sorted(spans)
    got = run_etl(spans)
    for d in range(4):
        idx = [i for i, sp in enumerate(spans) if sp[0] == d]
        expected = etl_oracle([(spans[i][1], spans[i][2], i) for i in idx])
        assert [got[i] for i in idx] == [expected[i] for i in idx]


def test_etl_limits_and_non_transitivity() -> None:
    # A = 0–100, B = 20–120, C = 40–140 (prompt example): B → A, C → B, so {A, B} and {C}
    assert run_etl([(0, 0, 100), (0, 20, 120), (0, 40, 140)]) == [0, 0, 1]
    # the floor of 10: short pieces 10 apart merge, 11 apart do not
    assert run_etl([(0, 0, 8), (0, 10, 18)]) == [0, 0]
    assert run_etl([(0, 0, 8), (0, 11, 19)]) == [0, 1]
    # a quarter of the shorter length: 400-long pieces 100 apart merge, 101 apart do not
    assert run_etl([(0, 0, 400), (0, 100, 500)]) == [0, 0]
    assert run_etl([(0, 0, 400), (0, 101, 501)]) == [0, 1]
    # the cap of 180, and candidates start at most 180 earlier
    assert run_etl([(0, 0, 2000), (0, 180, 2180)]) == [0, 0]
    assert run_etl([(0, 0, 2000), (0, 181, 2181)]) == [0, 1]
    # the leftmost qualifying candidate wins, not the closest
    assert run_etl([(0, 0, 100), (0, 20, 120), (0, 21, 121)]) == [0, 0, 0]
    # documents never mix
    assert run_etl([(0, 0, 100), (1, 0, 100)]) == [0, 1]


# --------------------------------------------------------------------------- node_similarity


def test_node_similarity_bug_and_fix() -> None:
    _, compat = node_similarity_groups([0, 2], [1000, 50], 0.9, compat=True)
    assert compat == [(1, 525)]  # the nested short piece shrinks the long one
    groups, fixed = node_similarity_groups([0, 2], [1000, 50], 0.9, compat=False)
    assert groups == [0, 1] and fixed == [(0, 1000), (2, 50)]


def _original_methods() -> type:
    """textreuse-blast's three methods, loaded from the reference source without importing its module."""
    tree = ast.parse((REFERENCE / "clusterizer.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelJobRunner")
    wanted = {"calculate_node_similarities_parallel", "similarity", "calculate_new_node", "stringify"}
    module = ast.parse("class Original:\n    pass\n")  # works on every Python ≥ 3.10
    module.body[0].body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(module), "clusterizer.py", "exec"),
         {"itemgetter": __import__("operator").itemgetter}, namespace)
    return namespace["Original"]


@pytest.mark.skipif(not (REFERENCE / "clusterizer.py").is_file(), reason="docs/reference not present")
@pytest.mark.parametrize("seed", range(5))
def test_node_similarity_compat_matches_the_original(seed: int) -> None:
    original = _original_methods()()
    original.node_similarity = 0.9
    rng = random.Random(seed)
    nodes = []
    for _ in range(200):
        s = rng.randint(0, 2000)
        nodes.append([s, s + rng.choice([10, 30, 60, 100, 500])])
    nodes.sort(key=lambda n: n[0])  # the original sorts by start too (stable)
    mapping, _ = original.calculate_node_similarities_parallel("d", [list(n) for n in nodes], 0)
    expected = [tuple(int(v) for v in mapping[f"d___{s}_{e}"].split("___")[1].split("_")) for s, e in nodes]
    group, centroids = node_similarity_groups([n[0] for n in nodes], [n[1] for n in nodes], 0.9, compat=True)
    by_span = {}
    for (s, e), g in zip(nodes, group, strict=True):
        by_span[(s, e)] = centroids[g]  # by span string, the last assignment wins
    assert [by_span[(s, e)] for s, e in nodes] == expected


# --------------------------------------------------------------------------- defragment: provenance and plugins


def synthetic_hits(seed: int = 1, n: int = 300) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        d1, d2 = sorted(rng.sample(range(6), 2))
        s1, s2, length = rng.randint(0, 5000), rng.randint(0, 5000), rng.choice([30, 60, 120, 400])
        rows.append({"hsp_id": i, "doc1": d1, "doc2": d2, "start1": s1, "end1": s1 + length, "start2": s2,
                     "end2": s2 + length, "enc_start1": s1, "enc_end1": s1 + length, "enc_start2": s2,
                     "enc_end2": s2 + length, "length": length, "pident": 90.0, "score": length * 8,
                     "bitscore": length * 3.0, "evalue": 1e-10})
    return pl.DataFrame(rows)


@pytest.mark.parametrize("rule", ["etl", "node_similarity", "node_similarity-compat", "tests.helpers:halves_rule"])
def test_provenance(rule: str) -> None:
    hits = synthetic_hits()
    pieces, piece_map, pairs = defragment(hits, "hsp_id", DefragOptions(rule=rule))
    assert piece_map.group_by("hsp_id").len()["len"].to_list() == [2] * hits.height  # two sides per hit
    assert piece_map["piece_id"].null_count() == 0
    raw = piece_map.select("raw_piece_id", "piece_id").unique()
    assert raw["raw_piece_id"].is_unique().all()  # every raw piece in exactly one piece
    assert pairs["num_orig_links"].sum() == hits.height
    assert (pairs["doc1"] < pairs["doc2"]).all()
    if rule in ("etl", "tests.helpers:halves_rule"):  # min/max spans contain their raw pieces
        sides = piece_map.join(pieces, on="piece_id").join(
            pl.concat([hits.select("hsp_id", pl.lit(1, pl.Int8).alias("side"), pl.col("start1").alias("s"),
                                   pl.col("end1").alias("e")),
                       hits.select("hsp_id", pl.lit(2, pl.Int8).alias("side"), pl.col("start2").alias("s"),
                                   pl.col("end2").alias("e"))]), on=["hsp_id", "side"])
        assert ((sides["start"] <= sides["s"]) & (sides["end"] >= sides["e"])).all()


def test_plugin_must_return_one_group_per_raw_piece() -> None:
    with pytest.raises(Exception, match="module:function|cannot load"):
        defragment(synthetic_hits(), "hsp_id", DefragOptions(rule="no_such_module_xyz:rule"))


# --------------------------------------------------------------------------- btop and clipping


def test_parse_btop_orients_sides() -> None:
    cols = parse_btop("3AG-C2T-", query_side=1)
    assert cols.adv1.tolist() == [1, 1, 1, 1, 0, 1, 1, 1] and cols.adv2.tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    assert cols.ident.tolist() == [1, 1, 1, 0, 0, 1, 1, 0]
    swapped = parse_btop("3AG-C2T-", query_side=2)
    assert swapped.adv1.tolist() == cols.adv2.tolist() and swapped.adv2.tolist() == cols.adv1.tolist()
    assert column_stats(cols) == {"length": 8, "nident": 5, "mismatch": 1, "gaps": 2, "gapopen": 2}
    assert column_stats(parse_btop("2-A-C2")) == {"length": 6, "nident": 4, "mismatch": 0, "gaps": 2, "gapopen": 1}


def test_clipping_tiles_the_alignment_and_respects_units() -> None:
    cols = parse_btop("4AG3-C3T-5")  # 18 columns: 17 letters on side 1, 17 on side 2
    n1, n2 = int(cols.adv1.sum()), int(cols.adv2.sum())
    zone1 = np.repeat([0, 1, 2], [6, 6, n1 - 12])  # unit boundaries after letters 6 and 12 of side 1
    zone2 = np.repeat([0, 1], [9, n2 - 9])
    starts = segment_starts(cols, zone1, zone2)
    bounds = [*starts, cols.length]
    assert bounds[0] == 0 and bounds[-1] == cols.length and bounds == sorted(set(bounds))  # a tiling
    segments, dropped = split_hit(cols, zone1, zone2, 100, 200)
    assert dropped == 0
    assert sum(s["nident"] for s in segments) == int(cols.ident.sum())
    assert segments[0]["enc_start1"] == 100 and segments[-1]["enc_end1"] == 100 + n1
    assert segments[-1]["enc_end2"] == 200 + n2
    for s in segments:  # no segment crosses a boundary on either side
        assert len(set(zone1[s["enc_start1"] - 100 : s["enc_end1"] - 100])) == 1
        assert len(set(zone2[s["enc_start2"] - 200 : s["enc_end2"] - 200])) == 1


def test_one_sided_segments_are_dropped() -> None:
    cols = Columns(np.array([1, 0, 0, 1], bool), np.array([1, 1, 1, 1], bool), np.array([1, 0, 0, 1], bool))
    zone2 = np.array([0, 1, 2, 2])  # side 2 crosses two boundaries inside a gap of side 1
    segments, dropped = split_hit(cols, np.array([0, 0]), zone2, 0, 0)
    assert dropped == 1 and len(segments) == 2


def test_surface_strings_give_back_both_spans() -> None:
    text1, text2 = "Pater, noster qui", "pater noster, qui"
    letters1 = [i for i, c in enumerate(text1) if c.isalpha()]
    letters2 = [i for i, c in enumerate(text2) if c.isalpha()]
    bounds1, bounds2 = np.array([*letters1, letters1[-1] + 1]), np.array([*letters2, letters2[-1] + 1])
    n = len(letters1)
    cols = Columns(np.ones(n, bool), np.ones(n, bool), np.ones(n, bool))
    s1, s2 = surface_strings(cols, text1, bounds1, text2, bounds2)
    assert len(s1) == len(s2)
    assert s1.replace("-", "") == text1 and s2.replace("-", "") == text2


# --------------------------------------------------------------------------- units


def test_unit_overlaps_and_unit_at_with_nested_units() -> None:
    units = pl.DataFrame({"doc": [0, 0, 0, 1], "loc": ["a", "a.1", "b", "z"], "start": [0, 10, 100, 0],
                          "end": [90, 20, 200, 50]}, schema_overrides={"doc": pl.Int32}).sort("doc", "start")
    pieces = pl.DataFrame({"piece_id": [0, 1, 2], "doc": [0, 0, 1], "start": [15, 80, 60], "end": [30, 120, 70]},
                          schema_overrides={"doc": pl.Int32})
    got = unit_overlaps(pieces, units)
    brute = [(p, u) for p in range(3) for u in range(4)
             if pieces["doc"][p] == units["doc"][u] and pieces["start"][p] < units["end"][u]
             and pieces["end"][p] > units["start"][u]]
    assert sorted(zip(got["piece_id"], got["loc"], strict=True)) == sorted(
        (p, units["loc"][u]) for p, u in brute)
    row = got.filter((pl.col("piece_id") == 1) & (pl.col("loc") == "b")).row(0, named=True)
    assert row["overlap_chars"] == 20 and abs(row["piece_share"] - 0.5) < 1e-6 and abs(row["unit_share"] - 0.2) < 1e-6
    # innermost unit; after a nested unit ends, the enclosing one; outside any unit, None
    assert unit_at(units, np.array([0, 0, 0, 0, 1]), np.array([15, 25, 95, 150, 60])) == ["a.1", "a", None, "b", None]


# --------------------------------------------------------------------------- API


def test_open_run_reads_tables_texts_and_utf16(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "corpus").mkdir(parents=True)
    (run / "tables").mkdir()
    (run / "manifest.json").write_text(json.dumps({"stages": {}}))
    pl.DataFrame({"doc_idx": [0, 1], "text": ["a𝔞b text", "plain"]}).write_parquet(run / "corpus" / "texts.parquet")
    pl.DataFrame({"pair_id": [0]}).write_parquet(run / "tables" / "pairs.parquet")
    r = open_run(run)
    assert r.pairs.collect().height == 1
    assert r.excerpt(0, 2, 3) == "b" and r.excerpt([0, 1], [0, 0], [3, 5]) == ["a𝔞b", "plain"]
    assert r.to_utf16(0, [0, 1, 2, 3]).tolist() == [0, 1, 3, 4]  # one astral character before offset 2
    with pytest.raises(RunNotAvailable, match="hsps_clipped"):
        r.hsps_clipped.collect()
    with pytest.raises(FileNotFoundError):
        open_run(tmp_path / "nothing")
