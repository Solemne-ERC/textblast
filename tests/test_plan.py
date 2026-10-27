"""Planning (no engine needed): every pair exactly once, never a self-pair; configuration; parsing."""

from __future__ import annotations

import io
import itertools
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from textblast import logs
from textblast.config import FIXED_ARGS, BlastOptions, blast_arguments, resolved_settings
from textblast.mapping import CorpusStore
from textblast.parse import BLAST_COLUMNS, to_hsps
from textblast.plan import default_block_letters, make_blocks, make_jobs, split_half
from textblast.prepare import PrepareConfig, prepare


def pairs_of(jobs, databases) -> list[tuple[int, int]]:
    members = {d.name: d.docs for d in databases}
    return [tuple(sorted((q, s))) for job in jobs for q in job.query for s in members[job.db]]


@pytest.mark.parametrize("seed", range(5))
def test_every_pair_exactly_once(seed: int) -> None:
    rng = random.Random(seed)
    docs = list(range(rng.randint(1, 40)))
    letters = {d: rng.choice([10, 100, 1000, 20000]) for d in docs}
    blocks = make_blocks(docs, letters, target=rng.choice([500, 5000, 50000]))
    databases, jobs = make_jobs(blocks, letters, None)
    found = pairs_of(jobs, databases)
    assert all(a != b for a, b in found)  # never a document against itself
    assert sorted(found) == sorted(itertools.combinations(docs, 2))  # every pair, exactly once
    assert all(job.max_target_seqs >= len(next(d for d in databases if d.name == job.db).docs) for job in jobs)
    assert {d.name for d in databases} == {j.db for j in jobs}  # every planned database is searched


def test_blocks_and_halves() -> None:
    letters = {0: 10, 1: 10, 2: 1000, 3: 10, 4: 10}
    assert make_blocks([0, 1, 2, 3, 4], letters, target=25) == [[0, 1], [2], [3, 4]]  # a large document alone
    a, b = split_half([0, 1, 3, 4], letters)
    assert a and b and a + b == [0, 1, 3, 4]
    databases, jobs = make_jobs([[0, 1, 3, 4]], letters, 500)
    assert {j.kind for j in jobs} == {"halves"} and all(j.max_target_seqs == 500 for j in jobs)


def test_default_block_letters_gives_ten_jobs_per_worker() -> None:
    target = default_block_letters(38_800_000, workers=10)
    k = -(-38_800_000 // target)
    assert k * (k - 1) // 2 >= 100 - k  # about 10 jobs per worker between blocks


def test_presets_and_overrides() -> None:
    default = resolved_settings(BlastOptions())  # decided after M4: comp_based_stats 0, evalue 1e-3
    assert BlastOptions().preset == "textblast"
    with pytest.raises(ValueError, match="now called 'textblast'"):
        resolved_settings(BlastOptions(preset="solemne-blast"))
    assert default == {"word_size": 6, "threshold": 400, "evalue": 1e-3, "window_size": 40, "comp_based_stats": 0}
    parity = resolved_settings(BlastOptions(preset="textreuse-blast"))  # unchanged, for parity
    assert parity == {"word_size": 6, "threshold": 400, "evalue": 1e-3, "window_size": 40, "comp_based_stats": 2}
    comhis = resolved_settings(BlastOptions(preset="comhis"))
    assert comhis["evalue"] == 1e-9 and comhis["max_target_seqs"] == 500_000_000
    tuned = resolved_settings(BlastOptions(window_size=0, xdrop_gap_final=50))
    assert tuned["window_size"] == 0 and tuned["xdrop_gap_final"] == 50
    with pytest.raises(ValueError, match="unknown BLAST preset"):
        resolved_settings(BlastOptions(preset="nope"))
    with pytest.raises(ValueError, match="comp_based_stats"):
        resolved_settings(BlastOptions(comp_based_stats=5))


def test_fixed_arguments_cannot_be_overridden() -> None:
    args = blast_arguments(resolved_settings(BlastOptions()), btop=True, dbsize=1234)
    assert args[: len(FIXED_ARGS)] == list(FIXED_ARGS)
    assert args.count("-matrix") == 1 and args.count("-gapopen") == 1 and args.count("-gapextend") == 1
    assert {"-lcase_masking", "-soft_masking", "-dbsize", "-comp_based_stats"} <= set(args)
    assert args[-1].endswith(" btop")
    assert not {"matrix", "gapopen", "gapextend"} & set(vars(BlastOptions()))  # no option exists for them


def test_parse_orders_sides_and_converts_coordinates(tmp_path: Path) -> None:
    logs.setup_logging(stream=io.StringIO())
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("".join(json.dumps({"doc_id": f"d{i}", "text": text}) + "\n"
                              for i, text in enumerate(["alpha beta", "gamma alpha beta", "delta"])))
    prepare(PrepareConfig(input=str(corpus), run_dir=str(tmp_path / "run"), workers=1))
    store = CorpusStore(tmp_path / "run")
    # query d1 (letters "gammaalpabeta", h deleted) against subject d0; BLAST is 1-based inclusive
    raw = pl.DataFrame(
        {"qseqid": ["d1"], "sseqid": ["d0"], "qstart": [6], "qend": [9], "sstart": [1], "send": [4],
         "length": [4], "nident": [4], "mismatch": [0], "gapopen": [0], "gaps": [0], "pident": [100.0],
         "ppos": [100.0], "score": [36], "bitscore": [20.0], "evalue": [1e-3], "btop": ["4"]},
        schema=BLAST_COLUMNS,
    )
    hits = to_hsps(raw, store, job_id=7)
    row = hits.row(0, named=True)
    assert (row["doc1"], row["doc2"], row["query_side"], row["job"]) == (0, 1, 2, 7)
    assert (row["enc_start1"], row["enc_end1"], row["enc_start2"], row["enc_end2"]) == (0, 4, 5, 9)
    # encoded "alpa" maps back to the surface "alpha" (the deleted h lies inside the span)
    assert store.text(0)[row["start1"]:row["end1"]] == "alpha"
    assert store.text(1)[row["start2"]:row["end2"]] == "alpha"
    assert np.all(hits["start1"].to_numpy() < hits["end1"].to_numpy())
    with pytest.raises(ValueError, match="aligned with itself"):
        to_hsps(raw.with_columns(pl.lit("d0").alias("qseqid")), store, job_id=1)


def test_short_flags_for_btop_and_raw_output() -> None:
    from textblast.cli import _run_config, build_parser

    parser = build_parser()
    plain = _run_config(parser.parse_args(["plan", "--input", "x.jsonl", "--run_dir", "r"])["plan"])
    assert plain.blast.btop and not plain.blast.keep_raw
    short = parser.parse_args(["plan", "--input", "x.jsonl", "--run_dir", "r", "--no-btop", "--keep-raw"])
    flagged = _run_config(short["plan"])
    assert not flagged.blast.btop and flagged.blast.keep_raw


def test_example_config_holds_every_default() -> None:
    """configs/example.yaml parses and its values are the defaults (it documents them)."""
    from textblast.cli import _run_config, build_parser
    from textblast.config import RunConfig

    example = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    parser = build_parser()
    loaded = _run_config(parser.parse_args(["plan", "--config", str(example)])["plan"])
    defaults = RunConfig(input=loaded.input, run_dir=loaded.run_dir)
    assert loaded == defaults
