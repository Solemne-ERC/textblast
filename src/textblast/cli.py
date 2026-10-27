"""Command-line interface: ``textblast <command> ...`` (plan section 10)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jsonargparse import ArgumentParser, Namespace

from . import __version__
from .logs import NORMAL, QUIET, VERBOSE, logger, setup_logging


def _add_output_options(parser: ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug detail and tracebacks.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only warnings and errors.")


def _add_engine_commands(engine: ArgumentParser) -> None:
    """Subcommands of ``textblast engine`` (jsonargparse wants the parent attached before its children)."""
    from .engine.versions import DEFAULT_VERSION, RELEASES

    commands = engine.add_subcommands(dest="action", required=True)

    build = ArgumentParser(description="Download, patch and build BLAST+ into the engine cache.")
    build.add_argument("--version", default=DEFAULT_VERSION, choices=sorted(RELEASES), help="BLAST+ release.")
    build.add_argument("--jobs", type=int, default=None, help="Parallel make jobs (default: all cores).")
    build.add_argument("--cache", type=str, default=None, help="Cache directory (default: ~/.cache/textblast).")
    build.add_argument(
        "--openmp", default="auto", choices=["auto", "on", "off"], help="OpenMP (auto: off on macOS, on elsewhere)."
    )
    build.add_argument(
        "--projects", default="restricted", choices=["restricted", "full"],
        help="restricted: BLAST project list without unit tests and masker programs; full: everything.",
    )
    build.add_argument("--force", action="store_true", help="Rebuild even if a verified engine exists.")
    build.add_argument("--keep-build", action="store_true", help="Keep the build directory after success.")
    _add_output_options(build)
    commands.add_subcommand("build", build)

    for name, description in (
        ("check", "Run the engine tests on synthetic sequences and record the results."),
        ("info", "Show version, fingerprint, location and last check of an engine."),
    ):
        sub = ArgumentParser(description=description)
        sub.add_argument("--blast-bin", type=str, default=None, help="Directory with blastp, makeblastdb, blastdbcmd.")
        sub.add_argument("--version", default=None, choices=sorted(RELEASES), help="Engine version in the cache.")
        sub.add_argument("--cache", type=str, default=None, help="Cache directory (default: ~/.cache/textblast).")
        _add_output_options(sub)
        commands.add_subcommand(name, sub)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="textblast",
        description="Text reuse detection with a patched NCBI BLAST engine.",
        version=__version__,
    )
    commands = parser.add_subcommands(dest="command", required=True)
    engine = ArgumentParser(description="Build, check and inspect the patched BLAST engine.")
    commands.add_subcommand("engine", engine)
    _add_engine_commands(engine)

    commands.add_subcommand("prepare", _config_parser(
        "Ingest, normalize and encode a JSONL corpus into <run_dir>/corpus (the corpus store)."))
    commands.add_subcommand("plan", _config_parser("Plan blocks, databases and BLAST jobs for a prepared corpus."))
    commands.add_subcommand("run", _config_parser(
        "prepare → plan → blast → merge → post → export (export.formats), each skipped when up to date."))
    commands.add_subcommand("post", _config_parser(
        "Clip (optional), defragment, annotate and filter hits into tables/ (pieces, piece_map, pairs, …)."))
    export = _config_parser("Write exports from the core tables: bundle, pairs_flat, passim-pairwise, etl.")
    export.add_argument("--format", type=str, default=None,
                        help="Formats, comma-separated (default: export.formats).")
    export.add_argument("--excerpts", type=str, default=None, choices=["surface", "none"],
                        help="Text in exports (same as --export.excerpts).")
    export.add_argument("--offsets", type=str, default=None, choices=["codepoints", "utf16"],
                        help="Offset units (same as --export.offsets).")
    commands.add_subcommand("export", export)

    blast = _run_dir_parser("Run all BLAST jobs that are not done (in parallel, longest first).")
    blast.add_argument("--workers", type=int, default=None, help="Parallel jobs (default: performance cores).")
    blast.add_argument("--force-engine", action="store_true", help="Run even if the engine's last check failed.")
    blast.add_argument("--jobs", type=str, default=None, help="Only these jobs, e.g. 1-20,35.")
    commands.add_subcommand("blast", blast)
    run_job = _run_dir_parser("Run one job (for cluster arrays).")
    run_job.add_argument("--job", type=int, required=True, help="Job id (see blast/jobs.tsv).")
    run_job.add_argument("--force-engine", action="store_true", help="Run even if the engine's last check failed.")
    commands.add_subcommand("run-job", run_job)
    merge = _run_dir_parser("Merge all job results into tables/hsps.parquet.")
    merge.add_argument("--partial", action="store_true", help="Merge even if some jobs are not done.")
    commands.add_subcommand("merge", merge)
    commands.add_subcommand("inspect", _run_dir_parser("Show the status of a run (also while it runs)."))
    evaluate = _run_dir_parser("Recall of a run against expected pairs (README: evaluating against expected pairs).")
    evaluate.add_argument("--expected", type=str, required=True, help="Expected pairs (Parquet).")
    evaluate.add_argument("--alternatives", type=str, default=None, help="Alternative targets (Parquet).")
    evaluate.add_argument("--level", type=str, default="hits", choices=["hits", "clipped", "joined", "pairs"],
                          help="Raw hits, hits clipped at units, joined hits, or pairs of pieces.")
    evaluate.add_argument("--out", type=str, default=None,
                          help="Output directory (default: <run_dir>/evaluate/<expected>-<level>).")
    evaluate.add_argument("--length-bands", type=str, default="50,150,500", help="Passage length bands (characters).")
    evaluate.add_argument("--similarity-bands", type=str, default="0.5,0.8", help="Similarity bands.")
    commands.add_subcommand("evaluate", evaluate)

    alphabet = ArgumentParser(description="Letter statistics and encoding tables.")
    commands.add_subcommand("alphabet", alphabet)
    _add_alphabet_commands(alphabet)
    return parser


def _config_parser(description: str) -> ArgumentParser:
    """Commands configured by run.yaml (the full run configuration; prepare uses its own sections only)."""
    from jsonargparse import ActionConfigFile

    from .config import RunConfig

    parser = ArgumentParser(description=description)
    parser.add_argument("--config", action=ActionConfigFile, help="YAML configuration (run.yaml).")
    parser.add_class_arguments(RunConfig, nested_key=None, fail_untyped=False)
    parser.add_argument("--no-btop", action="store_true", help="Drop BLAST's alignment transcript "
                        "(same as --blast.btop false).")
    parser.add_argument("--keep-raw", action="store_true", help="Keep BLAST's tabular output next to the parsed jobs "
                        "(same as --blast.keep_raw true).")
    _add_output_options(parser)
    return parser


def _run_dir_parser(description: str) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory (run_dir of run.yaml).")
    _add_output_options(parser)
    return parser


def _job_selection(text: str | None) -> set[int] | None:
    if not text:
        return None
    selected: set[int] = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        selected.update(range(int(a), int(b or a) + 1))
    return selected


def _add_alphabet_commands(alphabet: ArgumentParser) -> None:
    commands = alphabet.add_subcommands(dest="action", required=True)
    suggest = ArgumentParser(description="Rank letters after normalization and propose a letter → symbol table.")
    suggest.add_argument("--input", type=str, required=True, help="JSONL corpus.")
    suggest.add_argument("--profile", type=str, default="latin-broad", help="Normalization profile.")
    suggest.add_argument("--sample-chars", type=int, default=None, help="Use only the first N characters' documents.")
    suggest.add_argument("--locs-field", type=str, default="locs", help="Units field of the input.")
    suggest.add_argument("--workers", type=int, default=None, help="Worker processes (default: performance cores).")
    _add_output_options(suggest)
    commands.add_subcommand("suggest", suggest)


def _verbosity(options: Namespace) -> int:
    if options.get("verbose"):
        return VERBOSE
    if options.get("quiet"):
        return QUIET
    return NORMAL


def _cache(options: Namespace) -> Path | None:
    return Path(options.cache).expanduser() if options.get("cache") else None


def _run_engine(action: str, options: Namespace) -> int:
    if action == "build":
        from .engine.build import build_engine

        build_engine(
            version=options.version,
            jobs=options.jobs,
            cache=_cache(options),
            openmp=options.openmp,
            projects=options.projects,
            force=options.force,
            keep_build=options.keep_build,
        )
        return 0

    from .engine.locate import find_engine

    engine = find_engine(options.blast_bin, options.version, _cache(options))
    if action == "check":
        from .engine.check import run_checks

        report = run_checks(engine, cache=_cache(options))
        return 0 if report.passed else 1
    if action == "info":
        from .engine.check import describe_engine

        for line in describe_engine(engine):
            logger.info(line)
        return 0
    raise AssertionError(action)  # pragma: no cover


def _run_config(options: Namespace) -> Any:
    from dataclasses import fields

    from .config import RunConfig

    values = _config_parser("").instantiate(options)
    config = RunConfig(**{f.name: getattr(values, f.name) for f in fields(RunConfig)})
    if not config.run_dir:
        raise ValueError("no run directory: set `run_dir` in the configuration (or --run-dir)")
    if options.get("no_btop"):
        config.blast.btop = False
    if options.get("keep_raw"):
        config.blast.keep_raw = True
    return config


def _run_pipeline(command: str, options: Namespace) -> int:
    from .config import prepare_part

    if command in CONFIG_COMMANDS:
        config = _run_config(options)
        if command == "prepare":
            from .prepare import prepare

            prepare(prepare_part(config))
        elif command == "plan":
            from .plan import plan

            plan(config)
        elif command == "post":
            from .post import post

            post(config.run_dir, config.filters, config.units, config.defrag, config.workers, config.join,
                 config.annotate, config.cluster)
        elif command == "export":
            from .export import export

            if options.get("excerpts"):
                config.export.excerpts = options.excerpts
            if options.get("offsets"):
                config.export.offsets = options.offsets
            formats = options.format.split(",") if options.get("format") else config.export.formats
            if not formats:
                raise ValueError("no export format: give --format (e.g. pairs_flat,bundle) or export.formats")
            export(config.run_dir, [f.strip() for f in formats], config.export, config.workers)
        else:
            from .run import run_all

            run_all(config)
        return 0

    from .run import blast, merge, print_inspect, run_job

    run_dir = Path(options.run_dir).expanduser()
    if command == "blast":
        blast(run_dir, options.workers, options.force_engine, _job_selection(options.jobs))
    elif command == "run-job":
        from .plan import load_plan
        from .run import check_engine

        check_engine(load_plan(run_dir), options.force_engine, run_missing_checks=False)
        result = run_job(run_dir, options.job)
        logger.info(f"job {result.job_id}: {result.status} · {result.hits} hits · {result.seconds:.1f} s"
                    f"{' · ' + result.message if result.message else ''}")
        return 0 if result.status == "done" else 1
    elif command == "merge":
        merge(run_dir, options.partial)
    elif command == "inspect":
        print_inspect(run_dir)
    elif command == "evaluate":
        from .evaluate import EvaluateOptions, evaluate

        options_ = EvaluateOptions(
            level=options.level,
            length_bands=[int(x) for x in options.length_bands.split(",")],
            similarity_bands=[float(x) for x in options.similarity_bands.split(",")],
        )
        evaluate(run_dir, Path(options.expected).expanduser(),
                 Path(options.alternatives).expanduser() if options.alternatives else None, options_,
                 Path(options.out).expanduser() if options.out else None)
    return 0


def _run_alphabet(action: str, options: Namespace) -> int:
    from .prepare import suggest_alphabet

    if action != "suggest":  # pragma: no cover
        raise AssertionError(action)
    counts, table = suggest_alphabet(
        Path(options.input).expanduser(), options.profile, options.sample_chars, options.locs_field,
        workers=options.workers,
    )
    total = sum(counts.values()) or 1
    logger.info("rank  letter  share     count        symbol")
    for rank, (letter, count) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), start=1):
        logger.info(f"{rank:>4}  {letter:<6}  {count / total:7.3%}  {count:>11,}  {table.mapping[letter]}")
    pairs = ", ".join(f'"{k}": "{v}"' for k, v in sorted(table.mapping.items()))
    logger.info(f"suggested table for profile {options.profile}: {{{pairs}}}")
    return 0


CONFIG_COMMANDS = ("prepare", "plan", "run", "post", "export")
PIPELINE_COMMANDS = (*CONFIG_COMMANDS, "blast", "run-job", "merge", "inspect", "evaluate")


def _normalize_argv(argv: list[str]) -> list[str]:
    """``--run-dir`` works for every command (configured commands call the field ``run_dir``)."""
    if not argv or argv[0] not in CONFIG_COMMANDS:
        return argv
    return [("--run_dir" + a[len("--run-dir"):]) if a == "--run-dir" or a.startswith("--run-dir=") else a
            for a in argv]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    options: Any = parser.parse_args(_normalize_argv(list(argv) if argv is not None else sys.argv[1:]))
    command = options.command
    sub = options[command]
    if command in PIPELINE_COMMANDS:
        action, leaf = "", sub
    else:
        action = sub.action
        leaf = sub[action]
    setup_logging(_verbosity(leaf))

    from .api import RunNotAvailable
    from .clip import ClipError
    from .defrag import DefragError
    from .encode import EncodingError
    from .engine.build import BuildError
    from .engine.locate import EngineError
    from .engine.patches import PatchError
    from .export import ExportError
    from .filters import FilterError
    from .ingest import IngestError
    from .normalize import NormalizationError
    from .plan import PlanError
    from .post import PostError
    from .prepare import PrepareError
    from .run import RunError

    try:
        if command == "engine":
            return _run_engine(action, leaf)
        if command in PIPELINE_COMMANDS:
            return _run_pipeline(command, leaf)
        if command == "alphabet":
            return _run_alphabet(action, leaf)
        raise AssertionError(command)  # pragma: no cover
    except (IngestError, NormalizationError, EncodingError, PrepareError, PlanError, RunError, PostError,
            ClipError, DefragError, ExportError, FilterError, RunNotAvailable) as error:
        logger.error(str(error), exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1
    except BuildError as error:
        tail = error.log_tail()
        if tail:
            logger.error(f"{error}; last {len(tail)} lines of {error.log_path}:\n" + "\n".join(tail))
        else:
            logger.error(str(error), exc_info=True)
        return 1
    except (EngineError, PatchError, ValueError) as error:
        logger.error(str(error), exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
