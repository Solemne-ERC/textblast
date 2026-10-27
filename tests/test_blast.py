"""End-to-end BLAST runs on a small synthetic corpus (marker ``engine``)."""

from __future__ import annotations

import dataclasses
import io
import json
import random
from pathlib import Path

import polars as pl
import pytest

from textblast import logs
from textblast import run as run_module
from textblast.config import BlastOptions, EngineOptions, RunConfig
from textblast.engine.locate import EngineError
from textblast.plan import load_plan
from textblast.run import RunError, blast, check_engine, job_states, merge, run_all

pytestmark = pytest.mark.engine

PASSAGE = (
    "pater noster qui es in caelis sanctificetur nomen tuum adveniat regnum tuum fiat voluntas tua sicut in caelo "
    "et in terra panem nostrum cotidianum da nobis hodie"
)
VARIANT = (  # near-identical: spelling variants
    "pater noster qui es in celis sanctificetur nomem tuum adueniat regnum tuum fiat uoluntas tua sicut in celo "
    "et in terra panem nostrum quotidianum da nobis odie"
)
SPLIT = (  # split by a declared gap in one document, whole in another
    "in principio erat uerbum et uerbum erat apud deum et deus erat uerbum hoc erat in principio apud deum",
    "omnia per ipsum facta sunt et sine ipso factum est nihil quod factum est in ipso uita erat et uita erat lux",
)
MASKED = (
    "beati pauperes spiritu quoniam ipsorum est regnum caelorum beati mites quoniam ipsi possidebunt terram beati "
    "qui lugent quoniam ipsi consolabuntur"
)
EXPECTED_PAIRS = {(0, 2), (0, 3), (2, 3), (4, 5)}


def make_corpus(path: Path, seed: int = 3) -> None:
    rng = random.Random(seed)
    letters = "abcdefgilmnopqrstuxz"

    def filler(words: int) -> str:
        return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(words))

    middles = {0: PASSAGE, 1: filler(25), 2: PASSAGE, 3: VARIANT, 4: f"{SPLIT[0]} [lacuna] {SPLIT[1]}",
               5: " ".join(SPLIT), 6: MASKED, 7: MASKED}
    rows = []
    for i, middle in middles.items():
        head = filler(rng.randint(20, 120))
        row = {"doc_id": f"doc{i}", "text": f"{head} {middle} {filler(40)}"}
        if middle is MASKED:  # masked in both documents: no seeds, so no hit
            row["exclusions"] = [{"start": len(head) + 1, "length": len(MASKED), "mode": "mask", "reason": "test"}]
        rows.append(row)
    rows.append({"doc_id": "empty", "text": "... ; ..."})  # no letters: never sent to BLAST
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def config(tmp_path: Path, engine, **blast_options) -> RunConfig:
    return RunConfig(
        input=str(tmp_path / "c.jsonl"), run_dir=str(tmp_path / "run"), workers=1,
        blast=BlastOptions(workers=2, block_letters=2500, **blast_options),
        engine=EngineOptions(version=engine.version, blast_bin=str(engine.bin_dir)),
    )


def test_run_finds_shared_passage_and_resumes(engine, tmp_path: Path) -> None:
    logs.setup_logging(stream=io.StringIO())
    make_corpus(tmp_path / "c.jsonl")
    cfg = config(tmp_path, engine)
    hsps = pl.read_parquet(run_all(cfg))
    run_dir = Path(cfg.run_dir)
    plan_body = load_plan(run_dir)
    assert {j["kind"] for j in plan_body["jobs"]} == {"blocks", "halves"}

    planned = {d for j in plan_body["jobs"] for d in j["query"]} | {d for b in plan_body["blocks"] for d in b}
    assert 8 not in planned  # the empty document

    # the identical, near-identical and split passages are found; nothing else (masked, filler, empty)
    found = set(zip(hsps["doc1"].to_list(), hsps["doc2"].to_list(), strict=True))
    assert found == EXPECTED_PAIRS
    assert hsps["hsp_id"].to_list() == list(range(hsps.height))
    assert (hsps["doc1"] < hsps["doc2"]).all()

    # surface offsets point at the passage in both documents (gapped extension may overhang it a little)
    texts = {r["doc_idx"]: r["text"] for r in pl.read_parquet(run_dir / "corpus" / "texts.parquet").to_dicts()}
    best = hsps.sort("score", descending=True).unique(["doc1", "doc2"], keep="first")
    for row in best.filter(pl.col("doc2") < 4).to_dicts():
        for side in (1, 2):
            span = texts[row[f"doc{side}"]][row[f"start{side}"]:row[f"end{side}"]]
            passage = VARIANT if row[f"doc{side}"] == 3 else PASSAGE
            assert passage[3:-3] in span and len(span) < 1.4 * len(passage), span

    # no hit crosses the barrier: both halves are found separately in document 4
    gap = texts[4].index("[lacuna]")
    split = hsps.filter((pl.col("doc1") == 4) & (pl.col("doc2") == 5))
    assert not split.filter((pl.col("start1") < gap) & (pl.col("end1") > gap)).height
    for half in SPLIT:
        at = texts[4].index(half)
        assert split.filter((pl.col("start1") <= at + 5) & (pl.col("end1") >= at + len(half) - 5)).height, half

    # resume: nothing is rerun, results are identical
    stamps = {p.name: p.stat().st_mtime_ns for p in (run_dir / "blast" / "jobs").glob("*.parquet")}
    run_all(cfg)
    assert stamps == {p.name: p.stat().st_mtime_ns for p in (run_dir / "blast" / "jobs").glob("*.parquet")}
    assert pl.read_parquet(run_dir / "tables" / "hsps.parquet").equals(hsps)

    # a changed setting changes every job's hash: all jobs run again, into new result files
    run_all(config(tmp_path, engine, evalue=1e-5))
    hashes = {j["hash"] for j in load_plan(run_dir)["jobs"]}
    assert not {f"job_{h}.parquet" for h in hashes} & set(stamps)
    assert all((run_dir / "blast" / "jobs" / f"job_{h}.parquet").is_file() for h in hashes)


def test_databases_follow_the_corpus_store(engine, tmp_path: Path) -> None:
    """A store rebuilt with another letter table must not search the old databases (M5: 0 hits otherwise)."""
    from textblast.prepare import NormalizeOptions

    logs.setup_logging(stream=io.StringIO())
    make_corpus(tmp_path / "c.jsonl")
    frozen = config(tmp_path, engine)
    frozen.normalize = NormalizeOptions(profile="latin-broad-frozen")
    before = pl.read_parquet(run_all(frozen))
    hsps = pl.read_parquet(run_all(config(tmp_path, engine)))  # latin-broad: the reference table
    assert set(zip(hsps["doc1"].to_list(), hsps["doc2"].to_list(), strict=True)) == EXPECTED_PAIRS
    assert before.height and hsps.height


def test_failed_jobs_are_reported_and_retried(engine, tmp_path: Path) -> None:
    logs.setup_logging(stream=io.StringIO())
    make_corpus(tmp_path / "c.jsonl")
    cfg = config(tmp_path, engine)
    run_all(cfg)
    run_dir = Path(cfg.run_dir)
    plan_path = run_dir / "blast" / "plan.json"
    good = plan_path.read_text(encoding="utf-8")
    broken = json.loads(good)
    broken["blast_args"] = [*broken["blast_args"], "-no_such_option"]
    broken["fingerprint"] = "broken"
    for job in broken["jobs"]:  # plan computes job hashes from the arguments; a hand-edited plan must change them
        job["hash"] = f"broken{job['job_id']:05d}"
    broken["has_barriers"] = False  # skip the pre-run barrier check, which would reject the arguments before any job
    plan_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RunError, match="jobs failed"):
        blast(run_dir, workers=2, only={1, 2})
    assert (run_dir / "logs" / "jobs" / "job_00001.err").is_file()
    with pytest.raises(RunError, match="not done"):
        merge(run_dir)
    assert merge(run_dir, partial=True).is_file()

    plan_path.write_text(good, encoding="utf-8")  # back to the working plan: its finished jobs are still done
    states = job_states(run_dir, json.loads(good))
    assert all(s["status"] == "done" for s in states.values())
    summary = blast(run_dir, workers=2)
    assert summary["done"] == summary["jobs"] and summary["failed"] == 0


def test_engine_checks_before_jobs(engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logs.setup_logging(stream=io.StringIO())
    plan_body = {
        "engine": {"version": engine.version, "fingerprint": engine.fingerprint, "bin_dir": str(engine.bin_dir)},
        "has_barriers": False, "barrier_length": 100, "blast_args": [],
    }
    unchecked = dataclasses.replace(engine, info={k: v for k, v in engine.info.items() if k != "last_check"})
    monkeypatch.setattr(run_module, "find_engine", lambda *a, **k: unchecked)
    with pytest.raises(EngineError, match="never checked"):  # run-job: no engine tests from many jobs at once
        check_engine(plan_body, run_missing_checks=False)
    check_engine(plan_body, force=True, run_missing_checks=False)  # --force-engine overrides
    other = dataclasses.replace(engine, fingerprint="000000000000")
    monkeypatch.setattr(run_module, "find_engine", lambda *a, **k: other)
    with pytest.raises(RunError, match="plan expects"):
        check_engine(plan_body)


def test_rejected_arguments_stop_the_run_before_any_job(engine, tmp_path: Path) -> None:
    logs.setup_logging(stream=io.StringIO())
    make_corpus(tmp_path / "c.jsonl")  # has a declared gap, so the barrier check runs
    cfg = config(tmp_path, engine)
    run_all(cfg)
    plan_path = Path(cfg.run_dir) / "blast" / "plan.json"
    broken = json.loads(plan_path.read_text(encoding="utf-8"))
    broken["blast_args"] = [*broken["blast_args"], "-no_such_option"]
    broken["fingerprint"] = "broken"
    for job in broken["jobs"]:  # plan computes job hashes from the arguments; a hand-edited plan must change them
        job["hash"] = f"broken{job['job_id']:05d}"
    plan_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RunError, match="rejects this run's arguments"):
        blast(Path(cfg.run_dir), workers=2)
