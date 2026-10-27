"""Install proof: an installed textblast, outside the repository, runs a tiny synthetic corpus end to end.

    python smoke_test.py [--expect-source bundled] [WORK_DIR]

Runs `engine info` and `engine check`, then prepare → plan → blast → merge → post → export on six synthetic
documents, two pairs of which share a passage, and checks that exactly those pairs are found. Uses only the installed
package (run it with the Python of the virtualenv that has textblast).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def words(rng: random.Random, count: int) -> str:
    letters = "abcdefghilmnopqrstuvxyz"
    return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(2, 9))) for _ in range(count))


def make_corpus(path: Path) -> set[tuple[str, str]]:
    """Six documents of random words; d0/d2 share one passage exactly, d1/d4 another with a few changed letters."""
    rng = random.Random(7)
    shared, varied = words(rng, 40), words(rng, 40)
    variant = "".join("e" if i % 37 == 5 and c.isalpha() else c for i, c in enumerate(varied))
    middles = {"d0": shared, "d1": varied, "d2": shared, "d3": words(rng, 40), "d4": variant, "d5": words(rng, 40)}
    rows = [{"doc_id": doc, "text": f"{words(rng, 60)} {middle} {words(rng, 60)}", "date": 1500 + i}
            for i, (doc, middle) in enumerate(middles.items())]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {("d0", "d2"), ("d1", "d4")}


def textblast(*args: str) -> str:
    command = [str(Path(sys.executable).parent / "textblast"), *args]
    print("$ textblast", " ".join(args), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout + result.stderr, flush=True)
    if result.returncode != 0:
        raise SystemExit(f"`textblast {' '.join(args)}` failed with exit code {result.returncode}")
    return result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("work_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--expect-source", default=None, help="the engine must be found this way, e.g. bundled")
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="textblast-smoke-"))
    work.mkdir(parents=True, exist_ok=True)

    info = textblast("engine", "info")
    if args.expect_source and f"(found via {args.expect_source})" not in info:
        raise SystemExit(f"the engine was not found via {args.expect_source}")
    textblast("engine", "check")

    expected = make_corpus(work / "corpus.jsonl")
    config = work / "run.yaml"
    config.write_text(f"input: {work / 'corpus.jsonl'}\nrun_dir: {work / 'run'}\nworkers: 2\nblast:\n  workers: 2\n")
    run_dir = str(work / "run")
    textblast("prepare", "--config", str(config))
    textblast("plan", "--config", str(config))
    textblast("blast", "--run-dir", run_dir, "--workers", "2")
    textblast("merge", "--run-dir", run_dir)
    textblast("post", "--config", str(config))
    textblast("export", "--config", str(config), "--format", "pairs_flat")
    textblast("inspect", "--run-dir", str(work / "run"))

    import polars as pl

    import textblast as package

    run = package.open_run(work / "run")
    ids = dict(run.documents.select("doc_idx", "doc_id").collect().iter_rows())
    hits = run.hsps.select("doc1", "doc2").unique().collect()
    found = {tuple(sorted((ids[a], ids[b]))) for a, b in hits.iter_rows()}
    flat = pl.read_parquet(work / "run" / "exports" / "pairs_flat.parquet")
    print(f"textblast {package.__version__}: document pairs found {sorted(found)}; pairs_flat {flat.height} rows")
    if found != expected:
        raise SystemExit(f"expected the pairs {sorted(expected)}, found {sorted(found)}")
    if flat.height == 0:
        raise SystemExit("pairs_flat is empty")
    print("install proof passed")


if __name__ == "__main__":
    main()
