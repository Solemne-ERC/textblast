"""``textblast prepare``: ingest → normalize → encode → corpus store (plan sections 2, 5, 6; M1).

Work is spread over worker processes in batches balanced by character count; workers write their arrays to files,
and the main process assembles the store. A rerun with unchanged input, configuration and code does nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import unicodedata
from collections import Counter
from concurrent.futures import as_completed
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from . import __version__
from .encode import EncodingTable, encode_document, suggest_table
from .engine.check import BARRIER_LENGTH
from .ingest import Corpus, Document, documents_frame, load_corpus
from .logs import Stage, add_log_file, format_bytes, format_count, get_logger, peak_memory_bytes, remove_handler
from .mapping import CORPUS_DIR
from .normalize import BARRIER, SEP, SILENT_SEP, DocContext, Normalizer
from .normalize.profiles import Profile, get_profile
from .normalize.spans import to_codes
from .parallel import process_pool
from .sysinfo import default_workers

log = get_logger("prepare")

# Deletions that are ordinary normalization rather than lost content (excluded from the high-deletion warning).
ROUTINE_REASONS = frozenset(
    {"whitespace", "punctuation", "numeral", "soft_hyphen", "format_character", "spelling_h", "compat_not_mapped",
     "line_break_hyphen", "supplement_bracket"}
)
_NON_LETTERS = np.array([ord(SEP), ord(SILENT_SEP), ord(BARRIER)], dtype=np.uint32)


class PrepareError(RuntimeError):
    """Invalid input or configuration for ``prepare``."""


# --------------------------------------------------------------------------- configuration


@dataclass
class IngestOptions:
    """Input fields."""

    locs_field: str = "locs"  # units, passim format [{loc, start, length}]
    exclusions_field: str = "exclusions"  # classified spans [{start, length, mode, reason}]
    exclusions_file: str | None = None  # JSONL with doc_id, start, end, mode, reason


@dataclass
class PatternRuleOptions:
    """Regex → action (delete | mask | barrier) → reason; optional metadata condition (``where``).

    Example ``where``: ``{source_type: transcription}``; a list value means "any of these".
    """

    pattern: str
    action: str
    reason: str
    where: dict[str, Any] | None = None
    ignore_case: bool = False


@dataclass
class NormalizeOptions:
    """Normalization profile and rules."""

    profile: str = "latin-broad"
    steps: list[Any] | None = None  # override the profile's steps: names, {name, params} or {class_path, init_args}
    pattern_rules: list[PatternRuleOptions] = field(default_factory=list)
    declared_gaps: bool = True  # declared gaps and export scaffolding become barriers
    high_deletion_share: float = 0.12  # warn above this share of non-routine deleted characters


@dataclass
class EncodeOptions:
    barrier_length: int = BARRIER_LENGTH  # X letters per barrier (see docs/reports/m0_engine.md)


@dataclass
class PrepareConfig:
    """Configuration of ``textblast prepare``."""

    input: str | None = None  # required by prepare (post and export need only run_dir)
    run_dir: str = ""
    workers: int | None = None  # default: performance cores
    ingest: IngestOptions = field(default_factory=IngestOptions)
    normalize: NormalizeOptions = field(default_factory=NormalizeOptions)
    encode: EncodeOptions = field(default_factory=EncodeOptions)


def step_specs(options: NormalizeOptions, profile: Profile) -> list[Any]:
    """The profile's steps (or the override), with the ``declared_gaps`` option passed to the spans step."""
    specs: list[Any] = []
    for spec in options.steps if options.steps is not None else profile.steps:
        if spec == "spans":
            spec = {"name": "spans", "params": {"gaps": options.declared_gaps}}
        specs.append(spec)
    return specs


def worker_spec(config: PrepareConfig, profile: Profile, with_table: bool = True) -> str:
    """Everything a worker needs, as a JSON string (picklable, and a cache key)."""
    table = profile.encoding_table() if with_table else None
    return json.dumps(
        {
            "steps": step_specs(config.normalize, profile),
            "pattern_rules": [asdict(r) for r in config.normalize.pattern_rules],
            "table": dict(table.mapping) if table else None,
            "table_name": profile.name,
            "allow_collisions": profile.allow_collisions,
            "barrier_length": config.encode.barrier_length,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------- worker side


@dataclass
class DocStats:
    doc_idx: int
    n_letters: int
    n_encoded: int
    n_masked: int
    n_barriers: int
    deleted: dict[str, int]
    masked: dict[str, int]


@dataclass
class BatchResult:
    batch_id: int
    n_chars: int
    docs: list[DocStats]
    unsupported: dict[str, list[int]]  # character -> [count, documents]
    applied: list[tuple[int, int, int, str, str, str]]  # doc_idx, start, end, mode, reason, source
    letters: dict[str, int]


_CACHE: dict[str, tuple[Normalizer, EncodingTable | None, int]] = {}


def _setup(spec_json: str) -> tuple[Normalizer, EncodingTable | None, int]:
    if spec_json not in _CACHE:
        spec = json.loads(spec_json)
        table = (
            EncodingTable(spec["table_name"], spec["table"], spec["allow_collisions"]) if spec["table"] else None
        )
        _CACHE[spec_json] = (Normalizer(spec["steps"], spec["pattern_rules"]), table, spec["barrier_length"])
    return _CACHE[spec_json]


def process_batch(batch_id: int, docs: list[Document], spec_json: str, out_path: str | None) -> BatchResult:
    """Normalize (and, if a table is given, encode) a batch of documents; arrays go to ``out_path`` (.npz)."""
    normalizer, table, barrier_length = _setup(spec_json)
    stats: list[DocStats] = []
    unsupported: dict[str, list[int]] = {}
    applied: list[tuple[int, int, int, str, str, str]] = []
    letters: Counter[str] = Counter()
    parts_symbols, parts_start, parts_end, lengths = [], [], [], []
    for doc in docs:
        ctx = DocContext(doc.doc_idx, doc.doc_id, len(doc.text), doc.metadata, list(doc.exclusions))
        try:
            normalized = normalizer.run(doc.text, ctx)
        except Exception as error:  # name the document, keep the cause
            raise PrepareError(f"normalizing document {doc.doc_id!r} failed: {error}") from error
        codes = to_codes(normalized.text)
        letter_codes, letter_counts = np.unique(codes[~np.isin(codes, _NON_LETTERS)], return_counts=True)
        letters.update({chr(c): int(n) for c, n in zip(letter_codes.tolist(), letter_counts.tolist(), strict=True)})
        n_encoded = n_masked = n_barriers = 0
        if table is not None:
            try:
                encoded = encode_document(normalized, table, barrier_length)
            except Exception as error:
                raise PrepareError(f"encoding document {doc.doc_id!r} failed: {error}") from error
            parts_symbols.append(encoded.symbols)
            parts_start.append(encoded.start)
            parts_end.append(encoded.end)
            lengths.append(len(encoded.symbols))
            n_encoded, n_masked, n_barriers = len(encoded.symbols), encoded.n_masked, encoded.n_barriers
        stats.append(
            DocStats(doc.doc_idx, int(letter_counts.sum()), n_encoded, n_masked, n_barriers,
                     dict(ctx.deleted), dict(ctx.masked_chars))
        )
        for ch, n in ctx.unsupported.items():
            entry = unsupported.setdefault(ch, [0, 0])
            entry[0] += n
            entry[1] += 1
        applied += [(doc.doc_idx, e.start, e.end, e.mode, e.reason, e.source) for e in ctx.applied]
    if out_path is not None:
        np.savez(
            out_path,
            doc_idx=np.array([d.doc_idx for d in docs], dtype=np.int64),
            lengths=np.array(lengths, dtype=np.int64),
            symbols=np.concatenate(parts_symbols) if parts_symbols else np.zeros(0, np.uint8),
            start=np.concatenate(parts_start) if parts_start else np.zeros(0, np.uint32),
            end=np.concatenate(parts_end) if parts_end else np.zeros(0, np.uint32),
        )
    return BatchResult(batch_id, sum(len(d.text) for d in docs), stats, unsupported, applied, dict(letters))


# --------------------------------------------------------------------------- main side


def make_batches(documents: list[Document], workers: int) -> list[list[Document]]:
    """Consecutive documents grouped to roughly equal character counts (a long document is its own batch)."""
    total = sum(len(d.text) for d in documents)
    target = max(50_000, total // max(1, workers * 8))
    batches: list[list[Document]] = []
    current: list[Document] = []
    size = 0
    for doc in documents:
        current.append(doc)
        size += len(doc.text) + 1
        if size >= target:
            batches.append(current)
            current, size = [], 0
    if current:
        batches.append(current)
    return batches


def run_batches(
    documents: list[Document], spec_json: str, workers: int, out_dir: Path | None, stage: Stage
) -> list[BatchResult]:
    """Process all batches in parallel (longest first), with a progress line measured in characters."""
    batches = make_batches(documents, workers)
    order = sorted(range(len(batches)), key=lambda k: -sum(len(d.text) for d in batches[k]))
    total = sum(len(d.text) for d in documents)
    results: list[BatchResult] = []
    with stage.progress(total, "chars") as progress:
        if workers == 1:
            for k in order:
                out = str(out_dir / f"batch_{k:05d}.npz") if out_dir else None
                results.append(process_batch(k, batches[k], spec_json, out))
                progress.advance(results[-1].n_chars)
        else:
            with process_pool(max_workers=workers) as pool:
                futures = [
                    pool.submit(process_batch, k, batches[k], spec_json,
                                str(out_dir / f"batch_{k:05d}.npz") if out_dir else None)
                    for k in order
                ]
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.advance(results[-1].n_chars)
    return sorted(results, key=lambda r: r.batch_id)


def _file_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _code_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for rel in ("ingest.py", "prepare.py", "encode.py", "mapping.py", "normalize/__init__.py",
                "normalize/spans.py", "normalize/steps.py", "normalize/profiles.py"):
        digest.update((root / rel).read_bytes())
    return digest.hexdigest()[:16]


def config_dict(config: PrepareConfig) -> dict[str, Any]:
    """The ``prepare`` sections only (a run configuration may carry more; they never affect the store)."""
    return {f.name: asdict(config)[f.name] for f in fields(PrepareConfig)}


def fingerprint(config: PrepareConfig) -> str:
    settings = config_dict(config)
    settings.pop("workers")  # the number of workers does not change the result
    settings.pop("run_dir")  # nor where the run lives: a copied or moved run stays valid
    exclusions = config.ingest.exclusions_file
    inputs = {
        "settings": settings,
        "input": _file_info(Path(config.input)) if config.input else None,
        "exclusions_file": _file_info(Path(exclusions)) if exclusions else None,
        "code": _code_hash(),
        "version": __version__,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(run_dir / "manifest.json")


def append_event(run_dir: Path, **values: Any) -> None:
    """One JSON line in ``logs/events.jsonl`` (stages, jobs; read by ``inspect``)."""
    path = run_dir / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **values}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, ensure_ascii=False, default=str) + "\n")


def _char_names(text: str) -> tuple[str, str]:
    codes = " ".join(f"U+{ord(c):04X}" for c in text)
    names = " + ".join(unicodedata.name(c, "?") for c in text)
    return codes, names


def _write_tsv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(v).replace("\t", " ").replace("\n", "\\n") for v in row) + "\n")


def _looks_like_units(corpus: Corpus, locs_field: str) -> list[str]:
    """Metadata fields that look like unit spans although they are not configured as such."""
    suspicious = []
    for doc in corpus.documents[:20]:
        for key, value in doc.metadata.items():
            if (
                key not in suspicious
                and isinstance(value, list)
                and value
                and isinstance(value[0], dict)
                and {"start", "length"} <= set(value[0])
            ):
                suspicious.append(key)
    return suspicious


def prepare(config: PrepareConfig) -> Path:
    """Build the corpus store in ``<run_dir>/corpus`` (see :mod:`textblast.mapping`)."""
    if not config.input:
        raise PrepareError("no input: set `input` in the configuration (or --input)")
    input_path = Path(config.input).expanduser()
    if not input_path.is_file():
        raise PrepareError(f"input not found: {input_path}")
    run_dir = Path(config.run_dir).expanduser()
    profile = get_profile(config.normalize.profile)
    table = profile.encoding_table()
    workers = config.workers or default_workers()
    spec_json = worker_spec(config, profile)
    fp = fingerprint(config)

    manifest = read_manifest(run_dir)
    previous = manifest.get("stages", {}).get("prepare", {})
    store_complete = all((run_dir / CORPUS_DIR / name).exists() for name in _STORE_FILES)
    if previous.get("fingerprint") == fp and store_complete:
        log.info(f"corpus store in {run_dir / CORPUS_DIR} is up to date; nothing to do")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    handler = add_log_file(run_dir / "logs" / "run.log")
    started = time.monotonic()
    timings: dict[str, float] = {}
    try:
        log.info(
            f"textblast {__version__} · prepare · profile {profile.name} ({len(table.mapping)} letters) · "
            f"{workers} workers · run dir {run_dir}"
        )
        tmp = run_dir / f"{CORPUS_DIR}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "batches").mkdir(parents=True)

        with Stage(1, 3, "ingest", str(input_path), log=log, show_memory=True) as stage:
            exclusions_file = Path(config.ingest.exclusions_file) if config.ingest.exclusions_file else None
            corpus = load_corpus(input_path, config.ingest.locs_field, config.ingest.exclusions_field, exclusions_file)
            if not corpus.n_docs:
                raise PrepareError(f"{input_path} contains no documents")
            pl.DataFrame(
                {"doc_idx": [d.doc_idx for d in corpus.documents], "text": [d.text for d in corpus.documents]},
                schema={"doc_idx": pl.Int32, "text": pl.String},
            ).write_parquet(tmp / "texts.parquet")
            corpus.units.write_parquet(tmp / "units.parquet")
            n_spans = sum(len(d.exclusions) for d in corpus.documents)
            stage.summary = (
                f"{corpus.n_docs} documents · {format_count(corpus.n_chars)} chars · "
                f"{format_count(len(corpus.units))} units · {n_spans} classified spans"
            )
        timings["ingest"] = stage.elapsed
        for key in _looks_like_units(corpus, config.ingest.locs_field):
            log.warning(f"field {key!r} looks like unit spans; set ingest.locs_field: {key} to use it as units")

        with Stage(2, 3, "normalize + encode", f"{len(make_batches(corpus.documents, workers))} batches",
                   log=log, show_memory=True) as stage:
            results = run_batches(corpus.documents, spec_json, workers, tmp / "batches", stage)
            n_letters = sum(d.n_letters for r in results for d in r.docs)
            stage.summary = f"{format_count(n_letters)} letters"
        timings["normalize_encode"] = stage.elapsed

        with Stage(3, 3, "store", log=log, show_memory=True) as stage:
            summary = _assemble(corpus, results, tmp, run_dir, config)
            stage.summary = (
                f"{format_count(summary['encoded_symbols'])} symbols · "
                f"{format_bytes(summary['store_bytes'])} in {run_dir / CORPUS_DIR}"
            )
        timings["store"] = stage.elapsed

        manifest = read_manifest(run_dir)
        manifest["textblast"] = __version__
        manifest.setdefault("stages", {})["prepare"] = {
            "fingerprint": fp,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input": _file_info(input_path),
            "exclusions_file": _file_info(exclusions_file),
            "config": config_dict(config),
            "profile": profile.name,
            "steps": json.loads(spec_json)["steps"],
            "table": dict(table.mapping),
            "workers": workers,
            "summary": summary,
            "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
            "wall_seconds": round(time.monotonic() - started, 2),
            "peak_memory_bytes": peak_memory_bytes(),
        }
        write_manifest(run_dir, manifest)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config_dict(config), sort_keys=False), encoding="utf-8")
        append_event(run_dir, event="prepare_done", fingerprint=fp, wall_seconds=round(time.monotonic() - started, 2))
        return run_dir
    finally:
        remove_handler(handler)


_STORE_FILES = (
    "documents.parquet", "texts.parquet", "units.parquet", "exclusions.parquet",
    "encoded.bin", "encoded_index.npy", "map_start.npy", "map_end.npy",
)


def _assemble(
    corpus: Corpus, results: list[BatchResult], tmp: Path, run_dir: Path, config: PrepareConfig
) -> dict[str, Any]:
    """Concatenate the batch arrays in document order, write tables and warnings, swap the store into place."""
    stats = {d.doc_idx: d for r in results for d in r.docs}
    n_docs = corpus.n_docs
    lengths = np.array([stats[i].n_encoded for i in range(n_docs)], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    total = int(offsets[-1])
    np.save(tmp / "encoded_index.npy", offsets)

    symbols = np.memmap(tmp / "encoded.bin", dtype=np.uint8, mode="w+", shape=(max(total, 1),))
    map_start = np.lib.format.open_memmap(tmp / "map_start.npy", mode="w+", dtype=np.uint32, shape=(total,))
    map_end = np.lib.format.open_memmap(tmp / "map_end.npy", mode="w+", dtype=np.uint32, shape=(total,))
    for batch_file in sorted((tmp / "batches").glob("batch_*.npz")):
        with np.load(batch_file) as batch:
            position = 0
            for doc_idx, length in zip(batch["doc_idx"].tolist(), batch["lengths"].tolist(), strict=True):
                a, b = int(offsets[doc_idx]), int(offsets[doc_idx]) + length
                symbols[a:b] = batch["symbols"][position : position + length]
                map_start[a:b] = batch["start"][position : position + length]
                map_end[a:b] = batch["end"][position : position + length]
                position += length
    symbols.flush()
    map_start.flush()
    map_end.flush()
    del symbols, map_start, map_end
    if total == 0:
        (tmp / "encoded.bin").write_bytes(b"")
    shutil.rmtree(tmp / "batches")

    # documents table: input metadata plus normalization figures
    deleted_total: Counter[str] = Counter()
    masked_total: Counter[str] = Counter()
    content_deleted = np.zeros(n_docs, dtype=np.int64)
    for i in range(n_docs):
        deleted_total.update(stats[i].deleted)
        masked_total.update(stats[i].masked)
        content_deleted[i] = sum(n for reason, n in stats[i].deleted.items() if reason not in ROUTINE_REASONS)
    n_chars = np.array([len(d.text) for d in corpus.documents], dtype=np.int64)
    figures = pl.DataFrame(
        {
            "n_letters": [stats[i].n_letters for i in range(n_docs)],
            "n_encoded": lengths.tolist(),
            "n_masked": [stats[i].n_masked for i in range(n_docs)],
            "n_barriers": [stats[i].n_barriers for i in range(n_docs)],
            "deleted_share": np.divide(content_deleted, np.maximum(n_chars, 1)).tolist(),
            "empty": [stats[i].n_letters == 0 for i in range(n_docs)],
        }
    )
    base = documents_frame(corpus)
    documents = base.select("doc_idx", "doc_id", "n_chars").hstack(figures).hstack(base.drop("doc_idx", "doc_id",
                                                                                               "n_chars"))
    documents.write_parquet(tmp / "documents.parquet")

    applied = [row for r in results for row in r.applied]
    pl.DataFrame(
        applied,
        schema={"doc_idx": pl.Int32, "start": pl.Int64, "end": pl.Int64, "mode": pl.String, "reason": pl.String,
                "source": pl.String},
        orient="row",
    ).sort("doc_idx", "start").write_parquet(tmp / "exclusions.parquet")

    # swap the finished store into place
    final = run_dir / CORPUS_DIR
    if final.exists():
        shutil.rmtree(final)
    tmp.replace(final)

    warnings = _write_warnings(corpus, results, stats, content_deleted, run_dir, config)
    letters: Counter[str] = Counter()
    for r in results:
        letters.update(r.letters)
    store_bytes = sum((final / name).stat().st_size for name in _STORE_FILES)
    return {
        "documents": n_docs,
        "characters": int(n_chars.sum()),
        "letters": int(sum(stats[i].n_letters for i in range(n_docs))),
        "encoded_symbols": total,
        "masked_letters": int(sum(stats[i].n_masked for i in range(n_docs))),
        "barriers": int(sum(stats[i].n_barriers for i in range(n_docs))),
        "units": len(corpus.units),
        "classified_spans": len(applied),
        "deleted_by_reason": dict(deleted_total.most_common()),
        "masked_by_reason": dict(masked_total.most_common()),
        "letter_counts": dict(letters.most_common()),
        "warnings": warnings,
        "store_bytes": store_bytes,
    }


def _write_warnings(
    corpus: Corpus,
    results: list[BatchResult],
    stats: dict[int, DocStats],
    content_deleted: np.ndarray,
    run_dir: Path,
    config: PrepareConfig,
) -> dict[str, int]:
    folder = run_dir / "logs" / "warnings"
    if folder.exists():
        shutil.rmtree(folder)
    docs = corpus.documents
    counts: dict[str, int] = {}

    unsupported: dict[str, list[int]] = {}
    for r in results:
        for ch, (n, n_docs) in r.unsupported.items():
            entry = unsupported.setdefault(ch, [0, 0])
            entry[0] += n
            entry[1] += n_docs
    if unsupported:
        rows = [[ch, *_char_names(ch), n, n_docs] for ch, (n, n_docs) in
                sorted(unsupported.items(), key=lambda item: -item[1][0])]
        _write_tsv(folder / "unsupported_chars.tsv", ["character", "code_points", "names", "count", "documents"], rows)
        occurrences = sum(n for n, _ in unsupported.values())
        log.warning(
            f"{occurrences:,} unsupported characters of {len(unsupported)} kinds were removed "
            f"(see {folder / 'unsupported_chars.tsv'})"
        )
        counts["unsupported_kinds"] = len(unsupported)

    empty = [d for d in docs if stats[d.doc_idx].n_letters == 0]
    if empty:
        _write_tsv(folder / "empty_documents.tsv", ["doc_idx", "doc_id", "n_chars"],
                   [[d.doc_idx, d.doc_id, len(d.text)] for d in empty])
        log.warning(
            f"{len(empty)} documents have no letters after normalization (see {folder / 'empty_documents.tsv'})"
        )
        counts["empty_documents"] = len(empty)

    limit = config.normalize.high_deletion_share
    high = [d for d in docs if len(d.text) and content_deleted[d.doc_idx] / len(d.text) > limit]
    if high:
        rows = []
        for d in sorted(high, key=lambda d: -content_deleted[d.doc_idx] / len(d.text)):
            top = Counter({k: v for k, v in stats[d.doc_idx].deleted.items() if k not in ROUTINE_REASONS})
            rows.append([d.doc_idx, d.doc_id, len(d.text), int(content_deleted[d.doc_idx]),
                         f"{content_deleted[d.doc_idx] / len(d.text):.3f}",
                         ", ".join(f"{k} {v}" for k, v in top.most_common(3))])
        _write_tsv(folder / "high_deletion.tsv",
                   ["doc_idx", "doc_id", "n_chars", "deleted_chars", "share", "main_reasons"], rows)
        log.warning(
            f"{len(high)} documents lose more than {limit:.0%} of their characters to non-routine deletions "
            f"(see {folder / 'high_deletion.tsv'})"
        )
        counts["high_deletion_documents"] = len(high)

    by_reason: dict[tuple[int, str, str, str], list[int]] = {}
    for r in results:
        for doc_idx, start, end, mode, reason, source in r.applied:
            entry = by_reason.setdefault((doc_idx, source, mode, reason), [0, 0])
            entry[0] += 1
            entry[1] += end - start
    if by_reason:
        rows = [[doc_idx, docs[doc_idx].doc_id, source, mode, reason, n, chars]
                for (doc_idx, source, mode, reason), (n, chars) in sorted(by_reason.items())]
        _write_tsv(folder / "exclusions_by_reason.tsv",
                   ["doc_idx", "doc_id", "source", "mode", "reason", "spans", "chars"], rows)
        log.info(f"classified spans applied: {sum(n for n, _ in by_reason.values()):,} "
                 f"(see {folder / 'exclusions_by_reason.tsv'})")
    return counts


# --------------------------------------------------------------------------- alphabet suggestion


def suggest_alphabet(
    input_path: Path,
    profile_name: str,
    sample_chars: int | None = None,
    locs_field: str = "locs",
    exclusions_field: str = "exclusions",
    workers: int | None = None,
) -> tuple[Counter[str], EncodingTable]:
    """Letter frequencies after normalization with ``profile_name``, and the table they suggest (notes Q10)."""
    profile = get_profile(profile_name)
    corpus = load_corpus(input_path, locs_field, exclusions_field)
    documents = corpus.documents
    if sample_chars:
        chosen, size = [], 0
        for doc in documents:
            chosen.append(doc)
            size += len(doc.text)
            if size >= sample_chars:
                break
        documents = chosen
    config = PrepareConfig(input=str(input_path), run_dir=".", normalize=NormalizeOptions(profile=profile_name))
    spec_json = worker_spec(config, profile, with_table=False)
    workers = workers or default_workers()
    with Stage(1, 1, "count letters", f"{len(documents)} documents, profile {profile_name}", log=log) as stage:
        results = run_batches(documents, spec_json, workers, None, stage)
        counts: Counter[str] = Counter()
        for r in results:
            counts.update(r.letters)
        stage.summary = f"{format_count(sum(counts.values()))} letters, {len(counts)} distinct"
    return counts, suggest_table(counts, name=f"{profile_name}-suggested")
