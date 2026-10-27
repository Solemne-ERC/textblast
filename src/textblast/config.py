"""Run configuration beyond ``prepare``: BLAST settings and engine choice.

One ``run.yaml`` serves all commands: ``prepare`` reads its own sections and ignores ``blast`` and ``engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .cluster import ClusterOptions
from .defrag import DefragOptions
from .engine.versions import DEFAULT_VERSION
from .export import ExportOptions
from .filters import FilterOptions
from .join import JoinOptions
from .post import AnnotateOptions, UnitOptions
from .prepare import PrepareConfig

# Never configurable: the patched matrix (passed as BLOSUM62) and the only gap costs its statistics were patched for.
FIXED_ARGS = ("-matrix", "BLOSUM62", "-gapopen", "3", "-gapextend", "11")
# Always passed: no low-complexity filter; lowercase = masked, seeds only (engine check 5).
MASKING_ARGS = ("-seg", "no", "-lcase_masking", "-soft_masking", "true")
OUTFMT_FIELDS = (
    "qseqid sseqid qstart qend sstart send length nident mismatch gapopen gaps pident ppos score bitscore evalue"
)

PRESETS: dict[str, dict[str, Any]] = {
    # the default: textreuse-blast's values with -comp_based_stats 0 (no composition-based re-alignment, which can
    # bridge masked barriers and makes results depend on the letter table)
    "textblast": {"word_size": 6, "threshold": 400, "evalue": 1e-3, "window_size": 40, "comp_based_stats": 0},
    # textreuse-blast's settings, kept for parity
    "textreuse-blast": {"word_size": 6, "threshold": 400, "evalue": 1e-3, "window_size": 40, "comp_based_stats": 2},
    # the COMHIS / ReceptionReader production runs (blast_ecco)
    "comhis": {"word_size": 6, "threshold": 400, "evalue": 1e-9, "window_size": 40, "comp_based_stats": 2,
               "max_target_seqs": 500_000_000},
}
RENAMED_PRESETS = {"solemne-blast": "textblast"}  # old name → new name (the package was renamed before 0.9.0)
TUNABLE = ("evalue", "word_size", "threshold", "window_size", "comp_based_stats", "xdrop_ungap", "xdrop_gap",
           "xdrop_gap_final", "searchsp", "max_target_seqs")


@dataclass
class BlastOptions:
    """BLAST settings; unset values come from the preset."""

    preset: str = "textblast"  # textblast | textreuse-blast | comhis
    evalue: float | None = None
    word_size: int | None = None
    threshold: float | None = None
    window_size: int | None = None  # 0 = one-hit seeding
    comp_based_stats: int | None = None  # 0 | 1 | 2 | 3
    xdrop_ungap: float | None = None
    xdrop_gap: float | None = None
    xdrop_gap_final: float | None = None
    searchsp: int | None = None  # fixed search space: e-values independent of document length
    max_target_seqs: int | None = None  # default: all sequences of the job's database
    workers: int | None = None  # default: performance cores
    threads_per_worker: int = 1
    block_letters: int | None = None  # default: at least 10 jobs per worker
    group_blocks_by: str | None = None  # order documents by this field before blocking (pair-filter skipping)
    btop: bool = True  # keep BLAST's alignment transcript
    keep_raw: bool = False  # keep BLAST's tabular output next to the parsed jobs


@dataclass
class EngineOptions:
    version: str = DEFAULT_VERSION
    blast_bin: str | None = None  # directory with blastp, makeblastdb, blastdbcmd
    force: bool = False  # run even if the engine's last check failed


@dataclass
class RunConfig(PrepareConfig):
    """Configuration of ``plan``, ``post``, ``export`` and ``run`` (a superset of ``prepare``'s)."""

    blast: BlastOptions = field(default_factory=BlastOptions)
    engine: EngineOptions = field(default_factory=EngineOptions)
    filters: FilterOptions = field(default_factory=FilterOptions)
    units: UnitOptions = field(default_factory=UnitOptions)
    join: JoinOptions = field(default_factory=JoinOptions)
    defrag: DefragOptions = field(default_factory=DefragOptions)
    annotate: AnnotateOptions = field(default_factory=AnnotateOptions)
    cluster: ClusterOptions = field(default_factory=ClusterOptions)
    export: ExportOptions = field(default_factory=ExportOptions)


def prepare_part(config: PrepareConfig) -> PrepareConfig:
    """Only the ``prepare`` sections, so BLAST settings never invalidate the corpus store."""
    return PrepareConfig(**{f.name: getattr(config, f.name) for f in fields(PrepareConfig)})


def resolved_settings(options: BlastOptions) -> dict[str, Any]:
    """Preset values overridden by explicitly set options."""
    if options.preset in RENAMED_PRESETS:
        raise ValueError(f"BLAST preset {options.preset!r} is now called {RENAMED_PRESETS[options.preset]!r} (same "
                         "settings); set blast.preset accordingly")
    if options.preset not in PRESETS:
        raise ValueError(f"unknown BLAST preset {options.preset!r} (available: {', '.join(PRESETS)})")
    settings = dict(PRESETS[options.preset])
    for name in TUNABLE:
        value = getattr(options, name)
        if value is not None:
            settings[name] = value
    if settings.get("comp_based_stats") not in (0, 1, 2, 3):
        raise ValueError("comp_based_stats must be 0, 1, 2 or 3")
    return settings


def blast_arguments(settings: dict[str, Any], btop: bool, dbsize: int) -> list[str]:
    """The arguments shared by every job (query, database, output and target limit are added per job)."""
    args = [*FIXED_ARGS, *MASKING_ARGS, "-dbsize", str(dbsize)]
    for name in ("word_size", "threshold", "evalue", "window_size", "comp_based_stats", "xdrop_ungap", "xdrop_gap",
                 "xdrop_gap_final", "searchsp"):
        value = settings.get(name)
        if value is not None:
            args += [f"-{name}", _number(value)]
    args += ["-outfmt", f"6 {OUTFMT_FIELDS}{' btop' if btop else ''}"]
    return args


def _number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e6:
        return str(int(value))
    return str(value)
