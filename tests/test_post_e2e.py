"""M3 end to end on a synthetic corpus with units, series and a repeated passage (marker ``engine``)."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import polars as pl
import pytest

from textblast import logs
from textblast.config import BlastOptions, EngineOptions, RunConfig
from textblast.defrag import DefragOptions
from textblast.export import ExportOptions
from textblast.filters import FilterOptions, build_pair_filter
from textblast.mapping import CorpusStore
from textblast.plan import load_plan
from textblast.post import UnitOptions
from textblast.prepare import IngestOptions
from textblast.run import run_all

pytestmark = pytest.mark.engine

PASSAGE = (
    "pater noster qui es in caelis sanctificetur nomen tuum adveniat regnum tuum fiat voluntas tua sicut in caelo "
    "et in terra panem nostrum cotidianum da nobis hodie et dimitte nobis debita nostra"
)
OTHER = (
    "in principio erat uerbum et uerbum erat apud deum et deus erat uerbum hoc erat in principio apud deum omnia per "
    "ipsum facta sunt et sine ipso factum est nihil"
)
LAYOUT = {  # doc → (series, middle parts)
    0: ("a", [PASSAGE]),
    1: ("a", [PASSAGE, OTHER]),
    2: ("b", [PASSAGE, PASSAGE]),  # the passage twice: one region of doc 0 aligned to two places
    3: ("b", [OTHER]),
    4: ("b", []),
}


def make_corpus(path: Path, seed: int = 11) -> None:
    rng = random.Random(seed)
    letters = "abcdefgilmnopqrstuxz"

    def filler(words: int) -> str:
        return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(words))

    rows = []
    for doc, (series, middle) in LAYOUT.items():
        parts = [filler(30)]
        for m in middle:
            parts += [m, filler(25)]
        text = " ".join(parts)
        units, pos, k = [], 0, 0
        while pos < len(text):  # units of about 120 characters, cut at spaces
            end = text.find(" ", pos + 120)
            end = len(text) if end < 0 else end + 1
            units.append({"loc": f"{doc}.{k}", "start": pos, "length": end - pos})
            pos, k = end, k + 1
        rows.append({"doc_id": f"doc{doc}", "text": text, "series": series, "date": 1000 + 10 * doc, "units": units})
    rows.append({"doc_id": "empty", "text": "...", "series": "a", "date": 1, "units": []})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def config(tmp_path: Path, engine, **sections) -> RunConfig:
    return RunConfig(
        input=str(tmp_path / "c.jsonl"), run_dir=str(tmp_path / "run"), workers=2,
        ingest=IngestOptions(locs_field="units"),
        blast=BlastOptions(workers=2, block_letters=900),
        engine=EngineOptions(version=engine.version, blast_bin=str(engine.bin_dir)),
        **sections,
    )


def test_post_and_exports(engine, tmp_path: Path) -> None:
    logs.setup_logging(stream=io.StringIO())
    make_corpus(tmp_path / "c.jsonl")
    cfg = config(tmp_path, engine, units=UnitOptions(clip=True), defrag=DefragOptions(rule="etl"),
                 export=ExportOptions(formats=["bundle", "pairs_flat", "passim-pairwise", "etl"]))
    run_all(cfg)
    run_dir = Path(cfg.run_dir)
    t = {n: pl.read_parquet(run_dir / "tables" / f"{n}.parquet")
         for n in ("hsps", "hsps_clipped", "pieces", "piece_map", "pairs", "unit_overlaps", "join_map")}
    hsps, pieces, piece_map, pairs = t["hsps"], t["pieces"], t["piece_map"], t["pairs"]
    # joining is off by default: the join map annotates every raw hit, but there is no joined view or cluster
    assert t["join_map"].height == hsps.height
    assert not (run_dir / "tables" / "hsps_joined.parquet").exists()
    assert not (run_dir / "tables" / "clusters.parquet").exists()
    found = set(zip(hsps["doc1"].to_list(), hsps["doc2"].to_list(), strict=True))
    assert {(0, 1), (0, 2), (1, 2), (1, 3)} <= found and not any(4 in p for p in found)

    # provenance
    assert piece_map.group_by("hsp_id").len()["len"].to_list() == [2] * hsps.height
    assert pairs["num_orig_links"].sum() == hsps.height
    # the repeated passage: one region of doc 0 is paired with two places in doc 2
    p02 = pairs.filter((pl.col("doc1") == 0) & (pl.col("doc2") == 2))
    assert p02["piece1"].n_unique() < p02.height

    # clipping: segments tile their hits (identities), never cross a unit, and carry their units
    clipped = t["hsps_clipped"]
    sums = clipped.group_by("hsp_id").agg(pl.col("nident").sum()).join(
        hsps.select("hsp_id", pl.col("nident").alias("parent")), on="hsp_id")
    assert (sums["nident"] == sums["parent"]).all() and sums.height == hsps.height
    units = CorpusStore(run_dir).units().with_columns((pl.col("start") + pl.col("length")).alias("end"))
    for k in (1, 2):
        joined = clipped.join(units, left_on=[f"doc{k}", f"loc{k}"], right_on=["doc_idx", "loc"], how="left")
        assert joined["start"].null_count() == 0
        assert ((joined[f"start{k}"] >= joined["start"]) & (joined[f"end{k}"] <= joined["end"])).all()
    assert (clipped["loc1"].is_not_null()).all()

    # unit overlaps and masked share are present
    assert t["unit_overlaps"].height >= pieces.height and "masked_share" in pieces.columns

    # exports
    texts = pl.read_parquet(run_dir / "corpus" / "texts.parquet").sort("doc_idx")["text"].to_list()
    flat = pl.read_parquet(run_dir / "exports" / "pairs_flat.parquet")
    assert flat.height == pairs.height and {"date1", "series2", "locs1", "text1"} <= set(flat.columns)
    passim = pl.read_ndjson(run_dir / "exports" / "passim-pairwise.jsonl")
    assert passim.height == hsps.height
    for r in passim.iter_rows(named=True):
        assert r["s1"].replace("-", "") == texts[r["uid"]][r["begin"]:r["end"]].replace("-", "")
        assert r["s2"].replace("-", "") == texts[r["uid2"]][r["begin2"]:r["end2"]].replace("-", "")
    assert passim["matches"].to_list() == hsps.sort("hsp_id")["nident"].to_list()
    etl = pl.read_ndjson(run_dir / "exports" / "etl.jsonl")
    assert etl.columns == ["align_length", "positives_percent", "text1_id", "text1_text", "text1_text_end",
                           "text1_text_start", "text2_id", "text2_text", "text2_text_end", "text2_text_start"]
    ids = {f"doc{i}": i for i in range(5)}
    for r in etl.iter_rows(named=True):
        assert r["text1_text"] == texts[ids[r["text1_id"]]][r["text1_text_start"]:r["text1_text_end"]]

    # the bundle opens like the run
    from textblast.api import open_run

    bundle, run = open_run(run_dir / "exports" / "bundle"), open_run(run_dir)
    for name in ("pairs", "pieces", "hsps_clipped", "documents"):
        assert getattr(bundle, name).collect().equals(getattr(run, name).collect())

    # a second run is a no-op (nothing rewritten)
    stamps = {p: p.stat().st_mtime_ns for p in (run_dir / "tables").glob("*.parquet")}
    run_all(cfg)
    assert stamps == {p: p.stat().st_mtime_ns for p in (run_dir / "tables").glob("*.parquet")}


def test_pair_filter_skips_jobs_and_equals_filtering_afterwards(engine, tmp_path: Path) -> None:
    logs.setup_logging(stream=io.StringIO())
    make_corpus(tmp_path / "c.jsonl")
    base = config(tmp_path, engine)
    run_all(base)
    run_dir = Path(base.run_dir)
    unfiltered = pl.read_parquet(run_dir / "tables" / "hsps.parquet")
    full_plan = load_plan(run_dir)
    results = {p.name: p.stat().st_mtime_ns for p in (run_dir / "blast" / "jobs").glob("*.parquet")}

    filtered_cfg = config(tmp_path, engine, filters=FilterOptions(series_field="series"))
    filtered_cfg.blast.group_blocks_by = "series"
    run_all(filtered_cfg)
    plan = load_plan(run_dir)
    assert plan["pair_filter_stats"]["jobs_skipped"] > 0 and plan["pair_filter"]  # same-series jobs skipped
    pf = build_pair_filter(plan["pair_filter"], CorpusStore(run_dir).documents())
    expected = unfiltered.filter(pl.Series(pf.keeps(unfiltered["doc1"].to_numpy(), unfiltered["doc2"].to_numpy())))
    got = pl.read_parquet(run_dir / "tables" / "hsps.parquet")
    cols = [c for c in got.columns if c not in ("hsp_id", "job")]
    assert got.select(cols).sort(cols).equals(expected.select(cols).sort(cols))
    assert not (got["doc1"].is_in([0, 1]) & got["doc2"].is_in([0, 1])).any()

    # back to no filter (same blocks as the first run): every job is found done, nothing reruns
    run_all(base)
    assert {j["hash"] for j in load_plan(run_dir)["jobs"]} == {j["hash"] for j in full_plan["jobs"]}
    now = {p.name: p.stat().st_mtime_ns for p in (run_dir / "blast" / "jobs").glob("*.parquet")}
    assert all(now[name] == stamp for name, stamp in results.items())
    assert pl.read_parquet(run_dir / "tables" / "hsps.parquet").equals(unfiltered)


def test_joining_annotations_clusters_and_exports(engine, tmp_path: Path) -> None:
    """A passage split by an insertion in one document is joined; annotations, clusters and the M5 exports."""
    from textblast.api import open_run
    from textblast.cluster import ClusterOptions
    from textblast.evaluate import EvaluateOptions, evaluate
    from textblast.export import export
    from textblast.join import JoinOptions
    from textblast.post import AnnotateOptions

    logs.setup_logging(stream=io.StringIO())
    rng = random.Random(5)
    letters = "abcdefgilmnopqrstuxz"

    def filler(words: int) -> str:
        return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(words))

    layout = {0: f"{filler(30)} {PASSAGE} {OTHER} {filler(30)}",
              1: f"{filler(30)} {PASSAGE} {filler(20)} {OTHER} {filler(30)}",  # an insertion between the two
              2: f"{filler(30)} {OTHER} {filler(30)}"}
    rows = [{"doc_id": f"doc{d}", "text": text, "series": "ref" if d == 2 else "a", "date": 1000 + d,
             "units": [{"loc": f"{d}.{k}", "start": s, "length": min(150, len(text) - s)}
                       for k, s in enumerate(range(0, len(text), 150))]}
            for d, text in layout.items()]
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = config(tmp_path, engine, join=JoinOptions(enabled=True),
                 defrag=DefragOptions(view="joined"),
                 annotate=AnnotateOptions(reference_sets={"ref": "series = 'ref'"}),
                 cluster=ClusterOptions(enabled=True, method="components"))
    run_all(cfg)
    run_dir = Path(cfg.run_dir)
    run = open_run(run_dir)
    hsps, joined, join_map = run.hsps.collect(), run.joined.collect(), run.join_map.collect()
    pair01 = hsps.filter((pl.col("doc1") == 0) & (pl.col("doc2") == 1))
    assert pair01.height >= 2  # the insertion splits the passage into (at least) two hits
    j01 = joined.filter((pl.col("doc1") == 0) & (pl.col("doc2") == 1))
    assert j01.height == 1 and j01["n_members"][0] == pair01.height and j01["max_gap"][0] > 50
    assert join_map.height == hsps.height and joined["n_members"].sum() == hsps.height

    # annotations: in doc 0, the joined passage (PASSAGE + OTHER, linked to doc 1) and OTHER alone (linked to the
    # reference document, doc 2) are two pieces under the etl rule, each linked to one document
    pieces = run.pieces.collect()
    assert {"doc_degree", "ref_overlap_ref"} <= set(pieces.columns)
    doc0 = pieces.filter(pl.col("doc") == 0).sort("start", "end")
    assert doc0.height == 2 and doc0["doc_degree"].to_list() == [1, 1]
    assert 0.3 < doc0["ref_overlap_ref"][0] < 0.7 and doc0["ref_overlap_ref"][1] > 0.95

    # clusters: the joined passages, and OTHER in all three documents
    clusters, summary = run.clusters.collect(), run.cluster_summary.collect()
    assert clusters.height == pieces.height and summary["n_documents"].to_list() == [3, 2]

    # exports on the joined view
    texts = pl.read_parquet(run_dir / "corpus" / "texts.parquet").sort("doc_idx")["text"].to_list()
    export(run_dir, ["passim-pairwise", "etl", "pairs_flat", "passim-clusters"], cfg.export)
    passim = pl.read_ndjson(run_dir / "exports" / "passim-pairwise.jsonl")
    assert passim.height == joined.height
    for r in passim.iter_rows(named=True):  # the joined text, gaps included, comes back from s1 and s2
        assert r["s1"].replace("-", "") == texts[r["uid"]][r["begin"]:r["end"]].replace("-", "")
        assert r["s2"].replace("-", "") == texts[r["uid2"]][r["begin2"]:r["end2"]].replace("-", "")
    assert pl.read_ndjson(run_dir / "exports" / "etl.jsonl").height == hsps.height  # members, not joined hits
    flat = pl.read_parquet(run_dir / "exports" / "pairs_flat.parquet")
    assert {"doc_degree1", "ref_overlap_ref2", "cluster1"} <= set(flat.columns)
    out = pl.read_ndjson(run_dir / "exports" / "passim-clusters.jsonl")
    assert {"cluster", "size", "uid", "id", "begin", "end", "text"} <= set(out.columns)
    assert sorted(out["size"].unique().to_list()) == [2, 3] and "pboiler" not in out.columns  # documents
    for r in out.iter_rows(named=True):
        assert r["text"] == texts[r["uid"]][r["begin"]:r["end"]]
        expected_locs = [f"{r['uid']}.{k}" for k in range(r["begin"] // 150, (r["end"] - 1) // 150 + 1)]
        assert r["locs"] == expected_locs  # the units the passage overlaps, in text order
    # pboiler: share of a cluster's links within one value of the boiler field (passim's gid)
    export(run_dir, ["passim-clusters"], ExportOptions(boiler_field="series"))
    boiler = pl.read_ndjson(run_dir / "exports" / "passim-clusters.jsonl").group_by("size").agg(
        pl.col("pboiler").first()).sort("size")
    assert boiler["pboiler"].to_list() == [1.0, 0.0]  # doc0–doc1 (both "a"); OTHER through the reference "ref"

    # evaluation at the joined level
    expected = pl.DataFrame({"ref_id": ["q"], "kind": ["quote"], "relation": ["direct"], "doc_id1": ["doc1"],
                             "start1": [layout[1].index(PASSAGE)], "end1": [layout[1].index(OTHER) + len(OTHER)],
                             "doc_id2": ["doc0"], "start2": [layout[0].index(PASSAGE)],
                             "end2": [layout[0].index(OTHER) + len(OTHER)]})
    expected.write_parquet(tmp_path / "expected.parquet")
    result = evaluate(run_dir, tmp_path / "expected.parquet", None, EvaluateOptions(level="joined"))
    assert result["tables"]["by_kind"][0]["recall"] == 1.0

    # the passim rule links OTHER to the joined passage inside doc 0 and doc 1 (same role, ≥ 0.9): one cluster
    from textblast.post import post

    post(run_dir, cfg.filters, cfg.units, DefragOptions(rule="passim", view="joined"), 2, cfg.join, cfg.annotate,
         cfg.cluster)
    assert run.pieces.filter(pl.col("doc") == 0).collect().height == 1
    assert run.cluster_summary.collect()["n_documents"].to_list() == [3]

    # Chinese Whispers: the rounds and whether it settled go to the manifest; stopping early is a warning
    stream = io.StringIO()
    logs.setup_logging(stream=stream)
    post(run_dir, cfg.filters, cfg.units, DefragOptions(view="joined"), 2, cfg.join, cfg.annotate,
         ClusterOptions(enabled=True))
    assert run.manifest["stages"]["post"]["cluster"]["settled"] is True
    post(run_dir, cfg.filters, cfg.units, DefragOptions(view="joined"), 2, cfg.join, cfg.annotate,
         ClusterOptions(enabled=True, iterations=1))
    assert run.manifest["stages"]["post"]["cluster"]["rounds"] == 1
    assert "labels still changing after 1 rounds" in stream.getvalue()
