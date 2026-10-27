"""Build the patched BLAST engine from NCBI's official source (plan section 4).

Stages: download → verify → extract → patch → configure → make → install. configure/make output goes only to
``build.log``; the console shows stage lines and progress. The result is installed into
``<cache>/engine/<version>-<fingerprint>/`` with an ``engine.json`` describing exactly what was built.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..logs import Stage, format_bytes, format_count, format_duration, get_logger
from .locate import EngineError, blastp_version, default_cache_dir, engines_dir, verify_managed, write_json
from .patches import PATCHES, apply_patches
from .versions import BINARIES, DEFAULT_VERSION, PATCHSET, BlastRelease, get_release

log = get_logger("engine")

N_STAGES = 7

# Directories dropped from NCBI's BLAST project list for the restricted build: unit tests and stand-alone masker
# programs. The libraries they use stay, because BLAST itself links against them.
EXCLUDED_PROJECTS = (
    "algo/blast/unit_tests",
    "algo/blast/blastinput/unit_test",
    "app/winmasker",
    "app/dustmasker",
    "app/segmasker",
    "app/blastvdb",
)

_COMPILE_LINE = re.compile(r"\s-c\s.*\.(c|cc|cpp|cxx)\b")


class BuildError(EngineError):
    """A build step failed; ``log_path`` holds the full output."""

    def __init__(self, message: str, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path

    def log_tail(self, lines: int = 40) -> list[str]:
        if not self.log_path or not self.log_path.is_file():
            return []
        with self.log_path.open(encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=lines)]


@dataclass(frozen=True)
class BuildOptions:
    """Choices that change the binaries and are therefore part of the fingerprint."""

    openmp: str  # effective: "on" or "off"
    projects: str  # "restricted" or "full"

    def configure_args(self, project_list: Path | None) -> list[str]:
        args = ["--with-openmp" if self.openmp == "on" else "--without-openmp"]
        if self.projects == "restricted" and project_list is not None:
            args.append(f"--with-projects={project_list}")
        return args


def resolve_openmp(openmp: str) -> str:
    """``auto`` is off on macOS (Apple Clang has no OpenMP) and on elsewhere."""
    if openmp not in ("auto", "on", "off"):
        raise ValueError(f"--openmp must be auto, on or off, not {openmp!r}")
    if openmp == "auto":
        return "off" if sys.platform == "darwin" else "on"
    return openmp


def fingerprint(release: BlastRelease, options: BuildOptions) -> str:
    """Short hash of everything that determines the built binaries."""
    inputs = {
        "version": release.version,
        "tarball_sha256": release.sha256,
        "patchset": PATCHSET,
        "patches": list(release.patches),
        "openmp": options.openmp,
        "projects": options.projects,
        "system": platform.system(),
        "machine": platform.machine(),
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]


def _sha256_file(path: Path, chunk: int = 1 << 20) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def _compiler_version() -> str:
    try:
        out = subprocess.run([os.environ.get("CC", "cc"), "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _run_logged(cmd: list[str], cwd: Path, log_path: Path, stage: Stage, total: int | None, unit: str) -> None:
    """Run a command with all output appended to ``log_path``; progress counts compile commands."""
    with log_path.open("a", encoding="utf-8") as logfile, stage.progress(total, unit) as progress:
        logfile.write(f"\n$ {' '.join(cmd)}\n  (in {cwd})\n")
        logfile.flush()
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
        )
        assert process.stdout is not None
        count = 0
        for line in process.stdout:
            logfile.write(line)
            if total is None or _COMPILE_LINE.search(line):
                count += 1
                # An estimate, so never claim 100% before the command has finished.
                progress.set(min(count, int(total * 0.99)) if total else count)
        code = process.wait()
    if code != 0:
        raise BuildError(f"`{' '.join(cmd[:2])}` failed with exit code {code}", log_path)


def _download(release: BlastRelease, target: Path, stage: Stage) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(release.url, timeout=60) as response:
            size = int(response.headers.get("Content-Length") or 0) or None
            with partial.open("wb") as out, stage.progress(size, formatter=format_bytes) as progress:
                done = 0
                while block := response.read(1 << 20):
                    out.write(block)
                    done += len(block)
                    progress.set(done)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise BuildError(f"download of {release.url} failed: {error}") from None
    partial.replace(target)
    stage.summary = format_bytes(target.stat().st_size)


def _project_list(cpp_dir: Path, build_dir: Path) -> Path:
    """NCBI's BLAST project list without unit tests and stand-alone masker programs."""
    source = cpp_dir / "scripts/projects/blast/project.lst"
    if not source.is_file():
        raise BuildError(f"BLAST project list not found: {source}")
    kept = []
    for line in source.read_text().splitlines():
        entry = line.strip().split()[0].rstrip("$") if line.strip() else ""
        if entry and any(entry == excluded or entry.startswith(excluded + "/") for excluded in EXCLUDED_PROJECTS):
            continue
        kept.append(line)
    kept += [f"-{excluded}" for excluded in EXCLUDED_PROJECTS]
    target = build_dir / "textblast_projects.lst"
    target.write_text("\n".join(kept) + "\n")
    return target


def _count_sources(cpp_dir: Path, restricted: bool) -> int:
    """Rough number of compile steps, for the progress estimate of `make`."""
    count = 0
    for path in (cpp_dir / "src").rglob("*"):
        if path.suffix not in (".c", ".cpp", ".cc"):
            continue
        rel = path.relative_to(cpp_dir / "src").as_posix()
        if restricted and any(rel.startswith(excluded + "/") for excluded in EXCLUDED_PROJECTS):
            continue
        if "/test/" in rel or "/demo/" in rel or "/unit_test" in rel:
            continue
        count += 1
    return count


def build_engine(
    version: str = DEFAULT_VERSION,
    jobs: int | None = None,
    cache: Path | None = None,
    openmp: str = "auto",
    projects: str = "restricted",
    force: bool = False,
    keep_build: bool = False,
) -> Path:
    """Build (or reuse) the patched engine and return its directory."""
    if projects not in ("restricted", "full"):
        raise ValueError(f"--projects must be restricted or full, not {projects!r}")
    release = get_release(version)
    options = BuildOptions(openmp=resolve_openmp(openmp), projects=projects)
    fp = fingerprint(release, options)
    cache = cache or default_cache_dir()
    engine_dir = engines_dir(cache) / f"{release.version}-{fp}"
    jobs = jobs or os.cpu_count() or 1

    log.info(
        f"BLAST {release.version} ({release.role}) · fingerprint {fp} · patches: {', '.join(release.patches)} · "
        f"OpenMP {options.openmp} · {projects} build · {jobs} parallel jobs"
    )
    if engine_dir.exists() and not force:
        try:
            verify_managed(engine_dir, fp)
        except EngineError as error:
            log.warning(f"existing engine at {engine_dir} is incomplete ({error}); rebuilding")
        else:
            log.info(f"engine already built at {engine_dir}; nothing to do (use --force to rebuild)")
            return engine_dir

    started = time.monotonic()
    tarball = cache / "src" / release.tarball_name
    build_dir = cache / "build" / f"{release.version}-{fp}"
    log_path = build_dir / "build.log"

    with Stage(1, N_STAGES, "download", release.url if not tarball.is_file() else "cached", log=log) as stage:
        if not tarball.is_file():
            _download(release, tarball, stage)

    with Stage(2, N_STAGES, "verify", log=log) as stage:
        sha256, md5 = _sha256_file(tarball)
        if sha256 != release.sha256:
            bad = tarball.with_suffix(".bad")
            tarball.replace(bad)
            raise BuildError(f"checksum mismatch for {release.tarball_name}: sha256 {sha256}; moved to {bad}")
        if md5 != release.md5:
            raise BuildError(f"md5 mismatch for {release.tarball_name} (sha256 matched): {md5}")
        stage.summary = f"sha256 {sha256[:16]}… and NCBI md5 match"

    with Stage(3, N_STAGES, "extract", log=log) as stage:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        try:
            subprocess.run(["tar", "xzf", str(tarball), "-C", str(build_dir)], check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            raise BuildError(f"extraction failed: {error.stderr.decode(errors='replace').strip()}") from None
        cpp_dir = build_dir / release.source_dir_name / "c++"
        if not (cpp_dir / "configure").is_file():
            raise BuildError(f"unexpected source layout: {cpp_dir / 'configure'} not found")
        stage.summary = f"into {build_dir}"

    with Stage(4, N_STAGES, "patch", log=log) as stage:
        patch_results = apply_patches(cpp_dir, release.patches)
        for result in patch_results:
            log.debug(f"{result.name}: {result.status} ({result.file}, sha256 {result.sha256})")
        stage.summary = ", ".join(f"{r.name} {r.status}" for r in patch_results)

    project_list = _project_list(cpp_dir, build_dir) if projects == "restricted" else None
    configure_args = options.configure_args(project_list)
    log_path.write_text(f"textblast {__version__} build of BLAST {release.version} ({fp})\n")

    with Stage(5, N_STAGES, "configure", " ".join(configure_args), log=log) as stage:
        _run_logged(["./configure", *configure_args], cpp_dir, log_path, stage, None, "lines")
        build_root = cpp_dir / "ReleaseMT"
        if not (build_root / "build").is_dir():
            raise BuildError("configure did not create ReleaseMT/build", log_path)

    target = "all_p" if projects == "restricted" else "all_r"
    estimate = _count_sources(cpp_dir, projects == "restricted")
    with Stage(6, N_STAGES, "make", f"{target} with {jobs} jobs (~{format_count(estimate)} sources)", log=log) as stage:
        _run_logged(["make", f"-j{jobs}", target], build_root / "build", log_path, stage, estimate, "files")
        missing = [name for name in BINARIES if not (build_root / "bin" / name).is_file()]
        if missing:
            raise BuildError(f"make finished but these binaries are missing: {', '.join(missing)}", log_path)

    with Stage(7, N_STAGES, "install", log=log) as stage:
        if engine_dir.exists():
            shutil.rmtree(engine_dir)
        (engine_dir / "bin").mkdir(parents=True)
        binaries: dict[str, Any] = {}
        for name in BINARIES:
            dest = engine_dir / "bin" / name
            shutil.copy2(build_root / "bin" / name, dest)
            binaries[name] = {"size": dest.stat().st_size, "sha256": _sha256_file(dest)[0]}
        shutil.copy2(log_path, engine_dir / "build.log")
        _, version_line = blastp_version(engine_dir / "bin")
        info = {
            "schema": 1,
            "version": release.version,
            "role": release.role,
            "fingerprint": fp,
            "patchset": PATCHSET,
            "tarball": {"url": release.url, "sha256": sha256, "md5": md5},
            "patches": [
                {"name": r.name, "file": r.file, "status": r.status, "sha256": r.sha256,
                 "description": PATCHES[r.name].description}
                for r in patch_results
            ],
            "build": {
                "openmp": options.openmp,
                "projects": projects,
                "excluded_projects": list(EXCLUDED_PROJECTS) if projects == "restricted" else [],
                "configure_args": configure_args,
                "make_target": target,
                "jobs": jobs,
                "compiler": _compiler_version(),
                "system": f"{platform.system()} {platform.release()} {platform.machine()}",
                "built_by": f"textblast {__version__}",
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "build_seconds": round(time.monotonic() - started, 1),
            },
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "blastp_version": version_line,
            "binaries": binaries,
            "last_check": None,
        }
        write_json(engine_dir / "engine.json", info)
        verify_managed(engine_dir, fp)
        total_size = sum(item["size"] for item in binaries.values())
        stage.summary = f"{', '.join(BINARIES)} ({format_bytes(total_size)}) into {engine_dir}"

    if not keep_build:
        shutil.rmtree(build_dir, ignore_errors=True)
    log.info(f"engine built in {format_duration(time.monotonic() - started)}: {engine_dir}")
    return engine_dir
