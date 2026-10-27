"""``textblast evaluate``: scoring of expected pairs against synthetic hits (no engine, no corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from textblast.cli import main
from textblast.evaluate import (
    EvaluateError,
    EvaluateOptions,
    evaluate,
    load_expected,
    score,
    summarize,
)

DOCS = ["a", "bible:x", "bible:y", "c"]
A, X, Y, C = 0, 1, 2, 3


def hits(rows: list[tuple]) -> pl.DataFrame:
    """(doc1, start1, end1, doc2, start2, end2, evalue) → a segment table."""
    frame = pl.DataFrame(rows, schema=["doc1", "start1", "end1", "doc2", "start2", "end2", "evalue"], orient="row")
    return frame.with_row_index("seg_id").with_columns(
        pl.col("seg_id").cast(pl.Int64), *[pl.col(c).cast(pl.Int64) for c in frame.columns if c != "evalue"])


def expected(rows: list[dict]) -> pl.DataFrame:
    base = {"kind": "bible_quote", "relation": "direct", "target_group": "bible:", "similarity": 0.9,
            "exact1": True, "exact2": False}
    return pl.DataFrame([{**base, **r} for r in rows])


EXPECTED = expected([
    {"ref_id": "e1", "doc_id1": "a", "start1": 100, "end1": 200, "doc_id2": "bible:x", "start2": 0, "end2": 50},
    {"ref_id": "e2", "doc_id1": "c", "start1": 0, "end1": 100, "doc_id2": "a", "start2": 500, "end2": 600,
     "kind": "canon_note", "target_group": "a", "exact1": False},
    {"ref_id": "e3", "doc_id1": "a", "start1": 300, "end1": 400, "doc_id2": "bible:x", "start2": 100, "end2": 150},
    {"ref_id": "e4", "doc_id1": "a", "start1": 700, "end1": 800, "doc_id2": "bible:x", "start2": 200, "end2": 250},
    {"ref_id": "e5", "doc_id1": "a", "start1": 900, "end1": 950, "doc_id2": "bible:x", "start2": 300, "end2": 350,
     "similarity": 0.3},
    {"ref_id": "e6", "doc_id1": "zzz", "start1": 0, "end1": 10, "doc_id2": "bible:x", "start2": 0, "end2": 10},
    {"ref_id": "e7", "doc_id1": "a", "start1": 1000, "end1": 1100, "doc_id2": "bible:x", "start2": 400, "end2": 500,
     "relation": "cf"},
])
ALTERNATIVES = pl.DataFrame([
    {"ref_id": "e3", "doc_id2": "bible:y", "start2": 0, "end2": 40, "match": "parallel"},
    {"ref_id": "e3", "doc_id2": "bible:x", "start2": 150, "end2": 160, "match": "near"},
])
HITS = hits([
    (A, 90, 210, X, 10, 40, 1e-20),  # e1: exact, 10 characters past the passage at each end
    (A, 510, 590, C, 20, 80, 1e-8),  # e2: exact, passage on side 2 of the hit
    (A, 300, 400, Y, 5, 30, 1e-9),  # e3: the parallel alternative
    (A, 700, 800, Y, 500, 600, 1e-9),  # e4: elsewhere in the Bible
    (A, 900, 950, C, 0, 50, 1e-9),  # e5: overlaps the passage, but not with the target group: missed
    (A, 1000, 1030, X, 400, 430, 1e-5),  # e7: two exact hits covering half the passage
    (A, 1020, 1050, X, 420, 450, 1e-6),
])


def result(frame: pl.DataFrame, ref_id: str) -> dict:
    return frame.filter(pl.col("ref_id") == ref_id).row(0, named=True)


def test_score_classes_coverage_and_overshoot():
    out = score(load_expected_frame(EXPECTED), ALTERNATIVES, HITS, DOCS)
    assert out.height == EXPECTED.height
    assert dict(zip(out["ref_id"], out["result"], strict=True)) == {
        "e1": "exact", "e2": "exact", "e3": "alternative", "e4": "elsewhere", "e5": "missed", "e6": "unknown_doc",
        "e7": "exact"}
    e1 = result(out, "e1")
    assert (e1["overshoot_left1"], e1["overshoot_right1"]) == (10, 10)
    assert e1["overshoot_share1"] == pytest.approx(20 / 120)
    assert e1["overshoot_share2"] is None  # side 2 is not exact
    assert e1["coverage1"] == 1.0 and e1["n_hits"] == 1 and e1["best_evalue"] == 1e-20
    e2 = result(out, "e2")
    assert e2["coverage1"] == pytest.approx(0.6) and e2["overshoot_share1"] is None
    assert result(out, "e3")["match"] == "parallel"
    e7 = result(out, "e7")
    assert e7["n_hits"] == 2 and e7["coverage1"] == pytest.approx(0.5)
    assert e7["main_seg_id"] == 6  # equal overlap: the lower e-value wins
    assert result(out, "e5")["n_hits"] == 0


def load_expected_frame(frame: pl.DataFrame, tmp: Path | None = None) -> pl.DataFrame:
    """Run the frame through load_expected (defaults, types, checks)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "expected.parquet"
        frame.write_parquet(path)
        return load_expected(path)


def test_load_expected_defaults_and_checks(tmp_path):
    minimal = EXPECTED.select("ref_id", "kind", "relation", "doc_id1", "start1", "end1", "doc_id2", "start2", "end2")
    loaded = load_expected_frame(minimal)
    assert loaded["exact1"].all() and loaded["similarity"].is_null().all() and loaded["target_group"].is_null().all()
    out = score(loaded, ALTERNATIVES.clear(), HITS, DOCS)
    assert result(out, "e4")["result"] == "missed"  # no target group: no "elsewhere"
    assert result(out, "e3")["result"] == "missed"  # no alternatives given
    path = tmp_path / "bad.parquet"
    minimal.drop("kind").write_parquet(path)
    with pytest.raises(EvaluateError, match="missing columns kind"):
        load_expected(path)
    minimal.with_columns(pl.lit("same").alias("ref_id")).write_parquet(path)
    with pytest.raises(EvaluateError, match="unique"):
        load_expected(path)
    minimal.with_columns(pl.col("start1").alias("end1")).write_parquet(path)
    with pytest.raises(EvaluateError, match="empty or negative"):
        load_expected(path)


def test_summary_tables_and_bands():
    out = score(load_expected_frame(EXPECTED), ALTERNATIVES, HITS, DOCS)
    tables = summarize(out, EvaluateOptions(length_bands=[60, 100], similarity_bands=[0.5]))
    by_kind = {(r["kind"], r["relation"]): r for r in tables["by_kind"].iter_rows(named=True)}
    quotes = by_kind[("bible_quote", "direct")]  # e1 e3 e4 e5 (e6 has an unknown document)
    counts = tuple(quotes[c] for c in ("n", "exact", "alternative", "elsewhere", "missed"))
    assert counts == (4, 1, 1, 1, 1)
    assert quotes["recall"] == 0.25 and quotes["recall_alt"] == 0.5 and quotes["found_any"] == 0.75
    direct = (pl.col("kind") == "bible_quote") & (pl.col("relation") == "direct")
    quotes_by_length = tables["by_length"].filter(direct)
    assert quotes_by_length["length_band"].cast(pl.String).to_list() == ["< 60", "≥ 100"]  # band order, not text
    assert quotes_by_length["n"].to_list() == [1, 3]
    assert tables["by_similarity"].filter(pl.col("relation") == "direct")["similarity_band"].cast(
        pl.String).to_list()[:2] == ["< 0.5", "≥ 0.5"]
    over = tables["overshoot"].filter(pl.col("relation") == "direct").row(0, named=True)
    assert over["kind"] == "bible_quote" and over["side"] == 1 and over["n"] == 1 and over["chars_median"] == 20


def fake_run(path: Path) -> Path:
    """A run directory with just the tables evaluate reads."""
    (path / "corpus").mkdir(parents=True)
    (path / "tables").mkdir()
    (path / "manifest.json").write_text("{}")
    pl.DataFrame({"doc_idx": range(len(DOCS)), "doc_id": DOCS}).write_parquet(path / "corpus" / "documents.parquet")
    HITS.rename({"seg_id": "hsp_id"}).write_parquet(path / "tables" / "hsps.parquet")
    # pieces and pairs: e1's hit and one long piece pair around e4
    pieces = pl.DataFrame({"piece_id": [0, 1, 2, 3], "doc": [A, X, A, X], "start": [90, 10, 650, 190],
                           "end": [210, 40, 850, 260]})
    pieces.write_parquet(path / "tables" / "pieces.parquet")
    pl.DataFrame({"pair_id": [0, 1], "piece1": [0, 2], "piece2": [1, 3], "evalue": [1e-20, 1e-9]}).write_parquet(
        path / "tables" / "pairs.parquet")
    return path


def test_evaluate_command_hits_and_pairs(tmp_path):
    run_dir = fake_run(tmp_path / "run")
    exp_path, alt_path = tmp_path / "expected.parquet", tmp_path / "alternatives.parquet"
    EXPECTED.write_parquet(exp_path)
    ALTERNATIVES.write_parquet(alt_path)
    summary = evaluate(run_dir, exp_path, alt_path)
    assert summary["unknown_doc"] == 1 and summary["segments"] == HITS.height
    out = pl.read_parquet(run_dir / "evaluate" / "expected-hits" / "results.parquet")
    assert result(out, "e3")["result"] == "alternative"

    assert main(["evaluate", "--run-dir", str(run_dir), "--expected", str(exp_path), "--level", "pairs",
                 "--out", str(tmp_path / "pairs"), "-q"]) == 0
    pairs = pl.read_parquet(tmp_path / "pairs" / "results.parquet")
    assert result(pairs, "e1")["result"] == "exact"
    e4 = result(pairs, "e4")  # the piece pair 650–850 ↔ 190–260 reaches the target 200–250
    assert e4["result"] == "exact" and e4["overshoot_left1"] == 50 and e4["overshoot_right1"] == 50
    summary = json.loads((tmp_path / "pairs" / "summary.json").read_text())
    assert summary["level"] == "pairs" and summary["tables"]["by_kind"]

    assert main(["evaluate", "--run-dir", str(run_dir), "--expected", str(exp_path), "--level", "clipped",
                 "-q"]) == 1  # no clipped table in this run
