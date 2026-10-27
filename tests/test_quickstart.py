"""The README's quickstart on examples/quickstart: the commands run, and they find what the README says."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import polars as pl
import pytest

from textblast import cli, logs, open_run
from textblast.engine import EngineError, find_engine

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "quickstart"

pytestmark = pytest.mark.engine


@pytest.fixture
def quickstart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The quickstart's files in a fresh folder, run from there as the README does (relative paths)."""
    if not EXAMPLE.is_dir():
        pytest.skip("no examples/ folder (e.g. tests from the sdist)")
    try:
        find_engine()
    except EngineError as error:
        pytest.skip(f"no engine: {error}")
    shutil.copytree(EXAMPLE, tmp_path / "examples" / "quickstart")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "setup_logging", lambda verbosity: logs.setup_logging(verbosity, stream=io.StringIO()))
    return tmp_path


def pairs_of_documents(run_dir: Path, table: str = "hsps") -> set[tuple[str, str]]:
    run = open_run(run_dir)
    ids = dict(run.documents.select("doc_idx", "doc_id").collect().iter_rows())
    hits = run.table(table).select("doc1", "doc2").collect()
    return {(ids[a], ids[b]) for a, b in hits.iter_rows()}


def test_quickstart(quickstart: Path) -> None:
    assert cli.main(["run", "--config", "examples/quickstart/run.yaml"]) == 0
    assert cli.main(["inspect", "--run-dir", "runs/quickstart"]) == 0
    run_dir = quickstart / "runs" / "quickstart"

    found = pairs_of_documents(run_dir)
    mill = {("letter-01", "letter-04"), ("letter-01", "letter-07"), ("letter-04", "letter-07")}
    closing = {(a, b) for a in ("letter-01", "letter-02", "letter-03", "letter-05", "letter-08")
               for b in ("letter-01", "letter-02", "letter-03", "letter-05", "letter-08") if a < b}
    assert mill | closing | {("letter-02", "letter-06")} <= found

    # the README's snippet
    pairs = pl.read_parquet(run_dir / "exports" / "pairs_flat.parquet")
    top = pairs.sort("total_length", descending=True).select("doc_id1", "doc_id2", "total_length", "text1").head(5)
    assert top.row(0)[:2] == ("letter-01", "letter-04") and "The miller shall grind" in top.row(0)[3]

    # the bridge order comes out as two hits; joining makes them one
    hits = open_run(run_dir).hsps.collect()
    bridge = hits.join(open_run(run_dir).documents.select("doc_idx", "doc_id").collect(),
                       left_on="doc2", right_on="doc_idx").filter(pl.col("doc_id") == "letter-06")
    assert bridge.filter(pl.col("length") > 50).height == 2
    assert cli.main(["post", "--config", "examples/quickstart/run.yaml", "--join.enabled", "true"]) == 0
    joined = open_run(run_dir).joined.collect()
    assert joined.filter(pl.col("n_members") == 2).height >= 1

    # a recipe: the length filter drops the short shared phrases
    assert cli.main(["post", "--config", "examples/quickstart/run.yaml", "--filters.hit_filter", "length >= 50"]) == 0
    assert open_run(run_dir).pairs_filtered.collect().height < open_run(run_dir).pairs.collect().height
