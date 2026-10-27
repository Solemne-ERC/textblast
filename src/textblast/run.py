"""``textblast blast``, ``run-job``, ``merge``, ``inspect`` and ``run``.

Each job writes ``blast/jobs/job_<hash>.parquet`` (parsed, mapped hits) and a small status file next to it, both
atomically. The hash covers only the job's own inputs (see :mod:`textblast.plan`), so a rerun skips finished work
even when a changed pair filter renumbers the jobs, and cluster jobs (``run-job``) can finish in any order without
sharing state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .config import RunConfig, prepare_part
from .engine.check import run_checks, verify_barrier
from .engine.locate import EngineError, find_engine
from .filters import build_pair_filter
from .logs import Stage, format_bytes, format_count, format_duration, get_logger, logger, peak_memory_bytes
from .mapping import CorpusStore
from .parallel import process_pool
from .parse import read_blast_tsv, to_hsps
from .plan import BLAST_DIR, load_plan, plan
from .prepare import append_event, prepare, read_manifest, write_manifest
from .sysinfo import default_workers

log = get_logger("run")

JOBS_DIR = "jobs"
DB_DIR = "db"
HSPS = Path("tables") / "hsps.parquet"


class RunError(RuntimeError):
    """A run cannot proceed (failed jobs, outdated plan, missing results)."""


@dataclass
class JobResult:
    job_id: int
    status: str  # done | failed
    hits: int
    seconds: float
    message: str = ""


# --------------------------------------------------------------------------- paths and state


def _blast_dir(run_dir: Path) -> Path:
    return Path(run_dir) / BLAST_DIR


def job_hash(plan_body: dict[str, Any], job: dict[str, Any]) -> str:
    """The job's own hash (its inputs only; computed by ``plan``)."""
    return str(job["hash"])


def _job_files(run_dir: Path, job: dict[str, Any]) -> tuple[Path, Path]:
    base = _blast_dir(run_dir) / JOBS_DIR / f"job_{job['hash']}"
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def job_states(run_dir: Path, plan_body: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Status of every planned job: done (current hash, results present), failed, or missing."""
    states: dict[int, dict[str, Any]] = {}
    for job in plan_body["jobs"]:
        if "hash" not in job:  # a plan written before jobs had hashes: nothing can be matched
            states[job["job_id"]] = {"status": "missing"}
            continue
        parquet, status_file = _job_files(run_dir, job)
        state: dict[str, Any] = {"status": "missing"}
        if status_file.is_file():
            recorded = json.loads(status_file.read_text(encoding="utf-8"))
            current = recorded.get("hash") == job_hash(plan_body, job)
            done = recorded.get("status") == "done" and parquet.is_file()
            if current and (done or recorded.get("status") == "failed"):
                state = recorded
        states[job["job_id"]] = state
    return states


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- databases


def write_fasta(path: Path, docs: list[int], store: CorpusStore) -> None:
    """Encoded letters as FASTA, one line per sequence, ids ``d<doc_idx>``; lowercase (masked) letters kept."""
    with path.open("wb") as handle:
        for doc in docs:
            handle.write(b">d%d\n" % doc)
            handle.write(store.encoded(doc))
            handle.write(b"\n")


def ensure_database(run_dir: Path, plan_body: dict[str, Any], name: str, store: CorpusStore) -> Path:
    """Build the named database unless it already exists for the same documents, corpus store and engine; returns
    its prefix. The store's fingerprint matters: a store rebuilt with another letter table encodes the same documents
    differently (as when ``latin-broad`` changed its table)."""
    spec = next(d for d in plan_body["databases"] if d["name"] == name)
    marker_value = hashlib.sha256(
        json.dumps([spec["docs"], plan_body["engine"]["fingerprint"], plan_body["corpus_fingerprint"]]).encode()
    ).hexdigest()[:16]
    final = _blast_dir(run_dir) / DB_DIR / name
    marker = final / "ok"
    if marker.is_file() and marker.read_text().strip() == marker_value:
        return final / "db"
    tmp = final.with_name(f"{name}.tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    fasta = tmp / "seqs.fa"
    write_fasta(fasta, spec["docs"], store)
    makeblastdb = Path(plan_body["engine"]["bin_dir"]) / "makeblastdb"
    result = subprocess.run(
        [str(makeblastdb), "-in", str(fasta), "-dbtype", "prot", "-parse_seqids", "-title", name,
         "-out", str(tmp / "db")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RunError(f"makeblastdb failed for {name}: {result.stderr.strip()[-500:]}")
    fasta.unlink()
    (tmp / "ok").write_text(marker_value + "\n")
    if marker.is_file() and marker.read_text().strip() == marker_value:  # another process finished it meanwhile
        shutil.rmtree(tmp, ignore_errors=True)
        return final / "db"
    if final.exists():  # stale: other documents or another engine
        shutil.rmtree(final, ignore_errors=True)
    try:
        tmp.rename(final)
    except OSError:  # another process finished the same database first
        shutil.rmtree(tmp, ignore_errors=True)
    return final / "db"


# --------------------------------------------------------------------------- one job (worker side)

_STORES: dict[str, CorpusStore] = {}
_PLANS: dict[str, dict[str, Any]] = {}


def _cached(run_dir: Path) -> tuple[CorpusStore, dict[str, Any]]:
    """Store and plan, loaded once per worker process (and again if the plan changed)."""
    plan_file = _blast_dir(run_dir) / "plan.json"
    key = f"{run_dir}:{plan_file.stat().st_mtime_ns if plan_file.is_file() else 0}"
    if key not in _STORES:
        _STORES[key] = CorpusStore(run_dir)
        _PLANS[key] = load_plan(run_dir)
    return _STORES[key], _PLANS[key]


def run_job(run_dir: Path | str, job_id: int) -> JobResult:
    """Run one job: database (if needed), blastp (retried once), parse, write results and status atomically."""
    run_dir = Path(run_dir)
    store, plan_body = _cached(run_dir)
    job = plan_body["jobs"][job_id - 1]
    started = time.monotonic()
    parquet, status_file = _job_files(run_dir, job)
    tmp_dir = _blast_dir(run_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    query = tmp_dir / f"job_{job_id:05d}.fa"
    raw = tmp_dir / f"job_{job_id:05d}.tsv"
    try:
        db = ensure_database(run_dir, plan_body, job["db"], store)
        write_fasta(query, job["query"], store)
        threads = int(plan_body.get("threads_per_worker", 1))
        cmd = [str(Path(plan_body["engine"]["bin_dir"]) / "blastp"), "-query", str(query), "-db", str(db),
               *plan_body["blast_args"], "-max_target_seqs", str(job["max_target_seqs"]),
               "-num_threads", str(threads), *(["-mt_mode", "1"] if threads > 1 else []), "-out", str(raw)]
        result = None
        blastp_started = time.monotonic()
        for _ in range(2):  # one retry
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
        blastp_seconds = time.monotonic() - blastp_started
        assert result is not None
        if result.returncode != 0:
            err = run_dir / "logs" / "jobs" / f"job_{job_id:05d}.err"
            err.parent.mkdir(parents=True, exist_ok=True)
            tail = "\n".join(result.stderr.strip().splitlines()[-40:])
            err.write_text("$ " + " ".join(cmd) + f"\n\nexit code {result.returncode}\n\n{tail}\n", encoding="utf-8")
            _write_json(status_file, {"job_id": job_id, "status": "failed", "hash": job_hash(plan_body, job),
                                      "message": f"blastp exit code {result.returncode}", "log": str(err)})
            return JobResult(job_id, "failed", 0, time.monotonic() - started, f"see {err}")
        hsps = to_hsps(read_blast_tsv(raw, plan_body["btop"]), store, job_id, plan_body["btop"])
        tmp_parquet = parquet.with_suffix(f".parquet.tmp{os.getpid()}")
        parquet.parent.mkdir(parents=True, exist_ok=True)
        hsps.write_parquet(tmp_parquet, compression="zstd")
        tmp_parquet.replace(parquet)
        if plan_body.get("keep_raw"):
            raw.replace(parquet.with_suffix(".tsv"))
        seconds = time.monotonic() - started
        _write_json(status_file, {"job_id": job_id, "status": "done", "hash": job_hash(plan_body, job),
                                  "hits": hsps.height, "seconds": round(seconds, 2),
                                  "blastp_seconds": round(blastp_seconds, 2)})
        return JobResult(job_id, "done", hsps.height, seconds)
    except Exception as error:  # report, never crash the pool
        _write_json(status_file, {"job_id": job_id, "status": "failed", "hash": job_hash(plan_body, job),
                                  "message": str(error)})
        return JobResult(job_id, "failed", 0, time.monotonic() - started, str(error))
    finally:
        for path in (query, raw):
            path.unlink(missing_ok=True)


def _build_database(run_dir: str, name: str) -> str:
    store, plan_body = _cached(Path(run_dir))
    ensure_database(Path(run_dir), plan_body, name, store)
    return name


# --------------------------------------------------------------------------- checks


def check_engine(plan_body: dict[str, Any], force: bool = False, run_missing_checks: bool = True) -> None:
    """The planned engine must still exist, be the same build, and have passed its engine tests; barriers must hold
    with the run's exact BLAST arguments.

    ``run_missing_checks``: run the engine tests if the engine was never checked (``blast``). Cluster jobs
    (``run-job``) refuse instead, so that many jobs never run them at once and write the engine's info file together.
    """
    record = plan_body["engine"]
    engine = find_engine(record["bin_dir"])
    if engine.version != record["version"] or engine.fingerprint != record["fingerprint"]:
        raise RunError(f"engine at {record['bin_dir']} is BLAST {engine.version} ({engine.fingerprint}), the plan "
                       f"expects {record['version']} ({record['fingerprint']}); run `textblast plan` again")
    last = engine.info.get("last_check")
    if not last and run_missing_checks:
        log.info(f"engine {engine.fingerprint} was never checked; running the engine tests first")
        last = run_checks(engine).to_dict()
    elif not last and not force:
        raise EngineError(f"engine {engine.fingerprint} was never checked; run `textblast engine check` once "
                          "before starting cluster jobs (or use --force-engine)")
    if last and not last.get("passed") and not force:
        raise EngineError("the engine failed its last `textblast engine check`; fix it or use --force-engine")
    if not last or not last.get("passed"):
        log.warning("the engine is unchecked or failed its last check; continuing because --force-engine was given")
    if not plan_body["has_barriers"]:
        return
    try:
        holds = verify_barrier(engine, plan_body["barrier_length"], plan_body["blast_args"], complete=True)
    except EngineError as error:
        raise RunError(f"blastp rejects this run's arguments: {str(error)[:300]}") from error
    if not holds:
        raise RunError(
            f"a barrier of {plan_body['barrier_length']} X would be crossed with this run's BLAST arguments; "
            "raise encode.barrier_length or lower the x-drop values"
        )


# --------------------------------------------------------------------------- blast (all jobs)


def blast(run_dir: Path | str, workers: int | None = None, force_engine: bool = False,
          only: set[int] | None = None) -> dict[str, Any]:
    """Run every job that is not done (or only ``only``), in parallel, longest first."""
    run_dir = Path(run_dir).expanduser()
    plan_body = load_plan(run_dir)
    workers = workers or default_workers()
    jobs = {j["job_id"]: j for j in plan_body["jobs"]}
    states = job_states(run_dir, plan_body)
    pending = [jid for jid in jobs if states[jid]["status"] != "done" and (only is None or jid in only)]
    done_before = sum(s["status"] == "done" for s in states.values())
    log.info(f"blast · {len(jobs)} jobs ({done_before} done) · {workers} workers × "
             f"{plan_body.get('threads_per_worker', 1)} thread · engine {plan_body['engine']['fingerprint']}")
    if not pending:
        log.info(f"all {len(jobs)} jobs are done; nothing to do")
        return _blast_summary(run_dir, plan_body)

    with Stage(1, 3, "checks", log=log):
        check_engine(plan_body, force_engine)

    needed = sorted({jobs[jid]["db"] for jid in pending})
    with Stage(2, 3, "databases", f"{len(needed)} needed", log=log) as stage:
        letters = {d["name"]: d["letters"] for d in plan_body["databases"]}
        with stage.progress(sum(letters[n] for n in needed), "letters") as progress, \
                process_pool(max_workers=workers) as pool:
            futures = [pool.submit(_build_database, str(run_dir), name) for name in needed]
            done = 0.0
            for future in futures:
                done += letters[future.result()]
                progress.set(done)
        stage.summary = f"{len(needed)} databases ready"

    order = sorted(pending, key=lambda jid: -jobs[jid]["cost"])
    total_cost = sum(jobs[jid]["cost"] for jid in order)
    started = time.monotonic()
    results: list[JobResult] = []
    with Stage(3, 3, "jobs", f"{len(order)} to run", log=log, show_memory=True) as stage:
        with stage.progress(total_cost, "letter pairs") as progress:
            queue = deque(order)
            done_cost = 0.0
            with process_pool(max_workers=workers) as pool:
                running = {}
                try:
                    while queue or running:
                        while queue and len(running) < workers * 2:
                            jid = queue.popleft()
                            running[pool.submit(run_job, str(run_dir), jid)] = jid
                        finished, _ = wait(running, return_when=FIRST_COMPLETED)
                        for future in finished:
                            jid = running.pop(future)
                            result = future.result()
                            results.append(result)
                            done_cost += jobs[jid]["cost"]
                            append_event(run_dir, event=f"job_{result.status}", job=jid, hits=result.hits,
                                         seconds=round(result.seconds, 2), message=result.message)
                            if result.status == "failed":
                                log.warning(f"job {jid} failed: {result.message}")
                            progress.set(done_cost, f"{len(results)}/{len(order)} jobs")
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
                    log.error("interrupted; finished jobs are kept — rerun `textblast blast` to continue")
                    raise
        failed = [r for r in results if r.status == "failed"]
        hits = sum(r.hits for r in results)
        stage.summary = (f"{len(results) - len(failed)} jobs done · {format_count(hits)} hits · "
                         f"{len(failed)} failed")
    summary = _blast_summary(run_dir, plan_body, wall=time.monotonic() - started)
    if summary["done"] == summary["jobs"]:  # leftovers of interrupted jobs
        shutil.rmtree(_blast_dir(run_dir) / "tmp", ignore_errors=True)
    if failed:
        ids = ", ".join(str(r.job_id) for r in failed[:20])
        raise RunError(f"{len(failed)} jobs failed ({ids}{'…' if len(failed) > 20 else ''}); details in "
                       f"{run_dir / 'logs' / 'jobs'}; rerun `textblast blast --run-dir {run_dir}` to retry them")
    return summary


def _blast_summary(run_dir: Path, plan_body: dict[str, Any], wall: float | None = None) -> dict[str, Any]:
    states = job_states(run_dir, plan_body)
    summary = {
        "plan_fingerprint": plan_body["fingerprint"],
        "jobs": len(states),
        "done": sum(s["status"] == "done" for s in states.values()),
        "failed": sum(s["status"] == "failed" for s in states.values()),
        "hits": sum(int(s.get("hits", 0)) for s in states.values()),
        "job_seconds": round(sum(float(s.get("seconds", 0)) for s in states.values()), 1),
    }
    if wall is not None:
        summary["wall_seconds"] = round(wall, 1)
        summary["peak_memory_bytes"] = peak_memory_bytes()
    manifest = read_manifest(run_dir)
    manifest.setdefault("stages", {})["blast"] = summary
    write_manifest(run_dir, manifest)
    return summary


# --------------------------------------------------------------------------- merge


def merge(run_dir: Path | str, partial: bool = False) -> Path:
    """All job results → ``tables/hsps.parquet``, sorted by doc1, doc2, start1; ``hsp_id`` is the row number."""
    run_dir = Path(run_dir).expanduser()
    plan_body = load_plan(run_dir)
    states = job_states(run_dir, plan_body)
    not_done = [jid for jid, s in states.items() if s["status"] != "done"]
    if not_done and not partial:
        raise RunError(f"{len(not_done)} jobs are not done (e.g. {not_done[:5]}); run `textblast blast` "
                       "first, or merge with --partial")
    done = sorted(jid for jid, s in states.items() if s["status"] == "done")
    expression = plan_body.get("pair_filter")
    fingerprint = _merge_fingerprint(states, expression)
    target = run_dir / HSPS
    previous = read_manifest(run_dir).get("stages", {}).get("merge", {})
    if previous.get("fingerprint") == fingerprint and target.is_file():
        log.info(f"{target} is up to date; nothing to do")
        return target
    with Stage(1, 1, "merge", f"{len(done)} jobs", log=log, show_memory=True) as stage:
        jobs = {j["job_id"]: j for j in plan_body["jobs"]}
        files = [str(_job_files(run_dir, jobs[jid])[0]) for jid in done]
        hsps = (
            pl.concat([pl.read_parquet(f) for f in files], how="vertical") if files else pl.DataFrame()
        )
        removed = 0
        if expression and hsps.height:  # exact: skipped jobs only spared work
            pair_filter = build_pair_filter(expression, CorpusStore(run_dir).documents())
            keep = pair_filter.keeps(hsps["doc1"].to_numpy(), hsps["doc2"].to_numpy())
            removed = int((~keep).sum())
            hsps = hsps.filter(pl.Series(keep))
        if hsps.width:
            hsps = hsps.sort("doc1", "doc2", "start1", "start2").with_row_index("hsp_id").with_columns(
                pl.col("hsp_id").cast(pl.Int64)
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        hsps.write_parquet(tmp, compression="zstd")
        tmp.replace(target)
        pairs = hsps.select("doc1", "doc2").unique().height if hsps.width else 0
        stage.summary = (f"{format_count(hsps.height)} hits · {format_count(pairs)} document pairs · "
                         f"{format_bytes(target.stat().st_size)}"
                         + (f" · pair filter removed {format_count(removed)} hits" if expression else ""))
    manifest = read_manifest(run_dir)
    manifest.setdefault("stages", {})["merge"] = {
        "fingerprint": fingerprint, "hits": hsps.height, "pairs": pairs, "bytes": target.stat().st_size,
        "partial": bool(not_done), "pair_filter": expression, "removed_by_pair_filter": removed,
    }
    write_manifest(run_dir, manifest)
    return target


def _merge_fingerprint(states: dict[int, dict[str, Any]], pair_filter: str | None = None) -> str:
    hashes = sorted(states[j]["hash"] for j in states if states[j]["status"] == "done")
    return hashlib.sha256(json.dumps([hashes, pair_filter]).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- run (everything) and inspect


def run_all(config: RunConfig) -> Path:
    """prepare → plan → blast → merge → post → export (``export.formats``), each skipped when up to date."""
    from .export import export
    from .post import post

    run_dir = Path(config.run_dir).expanduser()
    find_engine(config.engine.blast_bin, config.engine.version)  # no engine: say so before `prepare`, not after it
    prepare(prepare_part(config))
    plan(config)
    blast(run_dir, config.blast.workers, config.engine.force)
    target = merge(run_dir)
    post(run_dir, config.filters, config.units, config.defrag, config.workers, config.join, config.annotate,
         config.cluster)
    if config.export.formats:
        export(run_dir, config.export.formats, config.export, config.workers)
    return target


def inspect(run_dir: Path | str) -> list[str]:
    """A readable status of a run, also while it is running (manifest, plan, job status files)."""
    run_dir = Path(run_dir).expanduser()
    manifest = read_manifest(run_dir)
    stages = manifest.get("stages", {})
    lines = [f"run directory {run_dir}"]
    prepared = stages.get("prepare")
    if prepared:
        s = prepared["summary"]
        lines.append(f"prepare   {s['documents']} documents · {format_count(s['letters'])} letters · profile "
                     f"{prepared['profile']} · {prepared['wall_seconds']} s")
    else:
        lines.append("prepare   not done")
    plan_body = load_plan(run_dir, missing_ok=True)
    if not plan_body:
        lines.append("plan      not done")
        return lines
    if any("hash" not in j for j in plan_body["jobs"]):
        lines.append("plan      written by an older version (no job hashes): run `textblast plan` again")
    lines.append(f"plan      {len(plan_body['jobs'])} jobs · {len(plan_body['blocks'])} blocks · engine "
                 f"{plan_body['engine']['version']} ({plan_body['engine']['fingerprint']}) · "
                 f"evalue {plan_body['settings'].get('evalue')} · cbs {plan_body['settings'].get('comp_based_stats')}")
    states = job_states(run_dir, plan_body)
    costs = {j["job_id"]: j["cost"] for j in plan_body["jobs"]}
    done = [j for j, s in states.items() if s["status"] == "done"]
    failed = [j for j, s in states.items() if s["status"] == "failed"]
    share = sum(costs[j] for j in done) / max(1, sum(costs.values()))
    hits = sum(int(states[j].get("hits", 0)) for j in done)
    job_seconds = sum(float(states[j].get("seconds", 0)) for j in done)
    lines.append(f"blast     {len(done)}/{len(states)} jobs done ({share:.0%} of the work) · {len(failed)} failed · "
                 f"{format_count(hits)} hits · {format_duration(job_seconds)} of job time")
    if failed:
        lines.append(f"          failed jobs: {', '.join(map(str, failed[:20]))} (see logs/jobs/)")
    merged = stages.get("merge")
    if merged:
        note = " (partial)" if merged.get("partial") else ""
        current = _merge_fingerprint(states, plan_body.get("pair_filter"))
        if merged.get("fingerprint") != current or not (run_dir / HSPS).is_file():
            note += " (outdated: run `textblast merge`)"
        lines.append(f"merge     {format_count(merged['hits'])} hits · {format_count(merged['pairs'])} document pairs"
                     f" · {format_bytes(merged['bytes'])}{note}")
    else:
        lines.append("merge     not done")
    post_stage = stages.get("post")
    if post_stage:
        d = post_stage.get("defrag", {})
        settings = post_stage.get("settings", {})
        clip = post_stage.get("clip")
        lines.append(
            f"post      {format_count(d.get('raw_pieces', 0))} raw pieces → {format_count(d.get('pieces', 0))} pieces "
            f"· {format_count(d.get('pairs', 0))} pairs · rule {settings.get('defrag', {}).get('rule')} on "
            f"{settings.get('defrag', {}).get('view')}"
            + (f" · {format_count(clip['segments'])} clipped segments" if clip else "")
            + (f" · hit filter keeps {format_count(post_stage['hit_filter']['pairs_kept'])} pairs"
               if post_stage.get("hit_filter") else "")
        )
    else:
        lines.append("post      not done")
    for name, info in sorted(stages.get("export", {}).items()):
        unit = "tables" if name == "bundle" else "rows"
        lines.append(f"export    {name}: {format_count(info['rows'])} {unit} · {format_bytes(info['bytes'])} · "
                     f"{info['path']}")
    warnings = sorted((run_dir / "logs" / "warnings").glob("*.tsv"))
    if warnings:
        lines.append("warnings  " + ", ".join(w.name for w in warnings))
    return lines


def print_inspect(run_dir: Path | str) -> None:
    for line in inspect(run_dir):
        logger.info(line)
