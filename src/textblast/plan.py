"""``textblast plan``: blocks, databases and jobs.

Every pair of documents is searched exactly once and no document is ever searched against itself:

- block i against block j for i < j (block i is the query, block j the database);
- inside a block, recursive halving: half A (query) against half B (database), then inside each half.

Databases are small BLAST databases built on demand (one per block and one per "B" half); the plan only describes
them. Everything later commands need is written to ``<run_dir>/blast/plan.json``, so ``blast``, ``run-job`` and
``merge`` work from the run directory alone.

An opt-in pair filter (``filters``) removes jobs whose document pairs are all excluded. Each job carries a hash of its
own inputs only (corpus, engine, BLAST arguments, query documents, database documents), so a changed filter or a
renumbered job never invalidates finished work.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import __version__
from .config import RunConfig, blast_arguments, resolved_settings
from .engine.locate import Engine, find_engine
from .filters import build_pair_filter, pair_expression
from .logs import Stage, format_count, get_logger
from .mapping import CorpusStore
from .prepare import read_manifest, write_manifest
from .sysinfo import default_workers

log = get_logger("plan")

BLAST_DIR = "blast"
PLAN_FILE = "plan.json"


class PlanError(RuntimeError):
    """The run directory cannot be planned (e.g. no prepared corpus)."""


@dataclass
class Database:
    name: str
    docs: list[int]
    letters: int


@dataclass
class Job:
    job_id: int
    kind: str  # "blocks" | "halves"
    query: list[int]  # doc_idx of the query documents
    db: str  # database name
    query_letters: int
    db_letters: int
    max_target_seqs: int

    @property
    def cost(self) -> int:
        return self.query_letters * self.db_letters


def make_blocks(doc_idx: list[int], letters: dict[int, int], target: int) -> list[list[int]]:
    """Consecutive documents in blocks of about ``target`` letters; a larger document is its own block."""
    blocks: list[list[int]] = []
    current: list[int] = []
    size = 0
    for doc in doc_idx:
        if current and size + letters[doc] > target:
            blocks.append(current)
            current, size = [], 0
        current.append(doc)
        size += letters[doc]
    if current:
        blocks.append(current)
    return blocks


def split_half(docs: list[int], letters: dict[int, int]) -> tuple[list[int], list[int]]:
    """Two halves of similar letter counts (document order kept; each half non-empty)."""
    total = sum(letters[d] for d in docs)
    running = 0
    for k, doc in enumerate(docs[:-1], start=1):
        running += letters[doc]
        if running * 2 >= total:
            return docs[:k], docs[k:]
    return docs[:-1], docs[-1:]


def make_jobs(
    blocks: list[list[int]], letters: dict[int, int], max_target_seqs: int | None
) -> tuple[list[Database], list[Job]]:
    databases: list[Database] = []
    jobs: list[Job] = []

    def database(name: str, docs: list[int]) -> Database:
        db = Database(name, docs, sum(letters[d] for d in docs))
        databases.append(db)
        return db

    def job(kind: str, query: list[int], db: Database) -> None:
        targets = max(len(db.docs), max_target_seqs or 0)
        jobs.append(Job(len(jobs) + 1, kind, query, db.name, sum(letters[d] for d in query), db.letters, targets))

    block_dbs = [database(f"b{i:03d}", block) for i, block in enumerate(blocks)]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            job("blocks", blocks[i], block_dbs[j])

    def halve(docs: list[int], path: str) -> None:
        if len(docs) < 2:
            return
        a, b = split_half(docs, letters)
        job("halves", a, database(f"{path}h", b))
        halve(a, path + "a")
        halve(b, path + "b")

    for i, block in enumerate(blocks):
        halve(block, f"b{i:03d}")
    used = {j.db for j in jobs}  # block 0 is only ever a query
    return [d for d in databases if d.name in used], jobs


def default_block_letters(total_letters: int, workers: int) -> int:
    """K blocks with K(K+1)/2 ≥ 10 jobs per worker (plus the halving jobs inside blocks)."""
    k = max(1, math.ceil((math.sqrt(80 * workers + 1) - 1) / 2))
    return max(1, math.ceil(total_letters / k))


def plan(config: RunConfig) -> dict[str, Any]:
    """Write ``blast/plan.json`` and ``blast/jobs.tsv``; returns the plan."""
    run_dir = Path(config.run_dir).expanduser()
    manifest = read_manifest(run_dir)
    prepared = manifest.get("stages", {}).get("prepare")
    if not prepared:
        raise PlanError(f"no prepared corpus in {run_dir}; run `textblast prepare` first")
    store = CorpusStore(run_dir)
    documents = store.documents()
    workers = config.blast.workers or default_workers()
    settings = resolved_settings(config.blast)
    engine = find_engine(config.engine.blast_bin, config.engine.version)

    with Stage(1, 1, "plan", log=log) as stage:
        eligible = documents.filter(~pl.col("empty"))
        group_field = config.blast.group_blocks_by
        if group_field:
            if group_field not in eligible.columns:
                raise PlanError(f"blast.group_blocks_by: unknown document field {group_field!r}")
            eligible = eligible.sort(group_field, "doc_idx", nulls_last=True)
        doc_idx = eligible["doc_idx"].to_list()
        letters = dict(zip(doc_idx, eligible["n_encoded"].to_list(), strict=True))
        total = int(sum(letters.values()))
        if not doc_idx:
            raise PlanError("no document has letters to search")
        target = config.blast.block_letters or default_block_letters(total, workers)
        blocks = make_blocks(doc_idx, letters, target)
        databases, jobs = make_jobs(blocks, letters, settings.get("max_target_seqs"))
        args = blast_arguments(settings, config.blast.btop, dbsize=total)
        expression = pair_expression(config.filters)
        filter_stats: dict[str, Any] | None = None
        if expression:
            pair_filter = build_pair_filter(expression, documents)
            members = {d.name: d.docs for d in databases}
            kept = [j for j in jobs if pair_filter.job_needed(j.query, members[j.db])]
            excluded, all_pairs = pair_filter.excluded_pairs()
            filter_stats = {"jobs_skipped": len(jobs) - len(kept), "jobs_planned": len(jobs),
                            "pairs_excluded": excluded, "pairs_total": all_pairs}
            log.info(f"pair filter: {len(jobs) - len(kept)} of {len(jobs)} jobs skipped; {excluded:,} of "
                     f"{all_pairs:,} document pairs excluded ({expression})")
            for number, job in enumerate(kept, start=1):
                job.job_id = number
            used = {j.db for j in kept}
            jobs, databases = kept, [d for d in databases if d.name in used]
        if not jobs:
            raise PlanError("no job left to run (the pair filter excludes every document pair)")
        members = {d.name: d.docs for d in databases}
        job_records = []
        for j in jobs:
            identity = {"corpus": prepared["fingerprint"], "engine": engine.fingerprint, "args": args,
                        "query": j.query, "db": members[j.db], "max_target_seqs": j.max_target_seqs}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            job_records.append(asdict(j) | {"cost": j.cost, "hash": digest})
        body = {
            "textblast": __version__,
            "corpus_fingerprint": prepared["fingerprint"],
            "engine": _engine_record(engine),
            "settings": settings,
            "blast_args": args,
            "barrier_length": prepared["config"]["encode"]["barrier_length"],
            "has_barriers": bool(eligible["n_barriers"].sum() > 0),
            "dbsize": total,
            "threads_per_worker": config.blast.threads_per_worker,
            "btop": config.blast.btop,
            "keep_raw": config.blast.keep_raw,
            "block_letters": target,
            "blocks": blocks,
            "pair_filter": expression,
            "pair_filter_stats": filter_stats,
            "group_blocks_by": group_field,
            "databases": [asdict(d) for d in databases],
            "jobs": job_records,
        }
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
        body["fingerprint"] = fingerprint
        blast_dir = run_dir / BLAST_DIR
        blast_dir.mkdir(parents=True, exist_ok=True)
        previous = load_plan(run_dir, missing_ok=True)
        if previous and previous.get("fingerprint") == fingerprint:
            stage.summary = f"unchanged: {len(jobs)} jobs, {len(blocks)} blocks"
            return previous
        tmp = blast_dir / (PLAN_FILE + ".tmp")
        tmp.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
        tmp.replace(blast_dir / PLAN_FILE)
        _write_jobs_tsv(blast_dir / "jobs.tsv", jobs)
        costs = np.array([j.cost for j in jobs], dtype=np.float64)
        stage.summary = (
            f"{len(doc_idx)} documents · {format_count(total)} letters · {len(blocks)} blocks of "
            f"~{format_count(target)} · {len(jobs)} jobs ({sum(j.kind == 'blocks' for j in jobs)} between blocks) · "
            f"{len(databases)} databases · largest job {costs.max() / costs.sum():.1%} of the work"
        )

    manifest = read_manifest(run_dir)
    manifest.setdefault("stages", {})["plan"] = {
        "fingerprint": fingerprint, "jobs": len(jobs), "blocks": len(blocks), "databases": len(databases),
        "settings": settings, "engine": body["engine"], "pair_filter": expression, "pair_filter_stats": filter_stats,
    }
    write_manifest(run_dir, manifest)
    return body


def _engine_record(engine: Engine) -> dict[str, Any]:
    return {"version": engine.version, "fingerprint": engine.fingerprint, "bin_dir": str(engine.bin_dir.resolve()),
            "source": engine.source}


def _write_jobs_tsv(path: Path, jobs: list[Job]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("job_id\tkind\tdb\tquery_docs\tquery_letters\tdb_letters\tcost\tquery_doc_idx\n")
        for j in jobs:
            query = ",".join(map(str, j.query)) if len(j.query) <= 20 else f"{j.query[0]}…{j.query[-1]}"
            handle.write(f"{j.job_id}\t{j.kind}\t{j.db}\t{len(j.query)}\t{j.query_letters}\t{j.db_letters}\t{j.cost}\t"
                         f"{query}\n")


def load_plan(run_dir: Path, missing_ok: bool = False) -> dict[str, Any]:
    path = Path(run_dir) / BLAST_DIR / PLAN_FILE
    if not path.is_file():
        if missing_ok:
            return {}
        raise PlanError(f"no plan in {run_dir}; run `textblast plan` first")
    return json.loads(path.read_text(encoding="utf-8"))
