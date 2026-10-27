"""Find and verify a patched BLAST engine.

Search order: an explicit ``--blast-bin`` directory, the ``TEXTBLAST_BIN`` environment variable, the engine bundled
in a platform wheel (``textblast/_engine``), then the cache (``~/.cache/textblast/engine``, or ``TEXTBLAST_CACHE``).

The bundled engine comes before the cache because it is the one this installed version was built and tested with;
an engine built into the cache is used when there is no bundle (an install from source) or when asked for
explicitly with ``--blast-bin``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ..logs import get_logger
from .versions import BINARIES, DEFAULT_VERSION

log = get_logger("engine")

BUNDLE_DIR = "_engine"


class EngineError(RuntimeError):
    """No usable engine was found, or an engine failed verification."""


def default_cache_dir() -> Path:
    return Path(os.environ.get("TEXTBLAST_CACHE", "~/.cache/textblast")).expanduser()


def engines_dir(cache: Path | None = None) -> Path:
    return (cache or default_cache_dir()) / "engine"


@dataclass
class Engine:
    """A located engine: its binaries and what is known about it."""

    bin_dir: Path
    version: str
    fingerprint: str
    source: str  # "argument", "environment", "bundled" or "cache"
    info: dict[str, Any] = field(default_factory=dict)
    info_path: Path | None = None  # engine.json (cache engines) or the external record (user-supplied binaries)
    check_path: Path | None = None  # where `engine check` records its results, if not in info_path

    def binary(self, name: str) -> Path:
        return self.bin_dir / name

    @property
    def managed(self) -> bool:
        """Built by ``textblast engine build`` (has its own engine.json)."""
        return self.info_path is not None and self.info_path.name == "engine.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write atomically (temporary file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


_VERSION_RE = re.compile(r"blastp:\s*(\d+\.\d+\.\d+)\+")


def blastp_version(bin_dir: Path) -> tuple[str, str]:
    """Return (version, first output line) of ``blastp -version``."""
    blastp = bin_dir / "blastp"
    try:
        out = subprocess.run([str(blastp), "-version"], capture_output=True, text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise EngineError(f"cannot run {blastp} -version: {error}") from None
    first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    match = _VERSION_RE.search(first)
    if not match:
        raise EngineError(f"unexpected output of {blastp} -version: {first!r}")
    return match.group(1), first


def missing_binaries(bin_dir: Path) -> list[str]:
    return [name for name in BINARIES if not os.access(bin_dir / name, os.X_OK)]


def verify_managed(engine_dir: Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    """Check an engine directory built by textblast; returns its engine.json or raises EngineError."""
    info_path = engine_dir / "engine.json"
    if not info_path.is_file():
        raise EngineError(f"no engine.json in {engine_dir}")
    info = read_json(info_path)
    if expected_fingerprint and info.get("fingerprint") != expected_fingerprint:
        raise EngineError(f"fingerprint mismatch in {info_path}")
    missing = missing_binaries(engine_dir / "bin")
    if missing:
        raise EngineError(f"missing binaries in {engine_dir / 'bin'}: {', '.join(missing)}")
    version, _ = blastp_version(engine_dir / "bin")
    if version != info.get("version"):
        raise EngineError(f"blastp reports {version}, engine.json says {info.get('version')}")
    return info


def _external_record(cache: Path | None, bin_dir: Path) -> Path:
    key = hashlib.sha256(str(bin_dir.resolve()).encode()).hexdigest()[:16]
    return (cache or default_cache_dir()) / "external" / f"{key}.json"


def bundled_engine_dir(package: str = "textblast") -> Path | None:
    """The engine directory inside an installed platform wheel, or None (an install from source has none)."""
    try:
        root = resources.files(package) / BUNDLE_DIR
    except ModuleNotFoundError:
        return None
    path = Path(str(root))
    return path if (path / "engine.json").is_file() else None


def _is_bundle(engine_dir: Path) -> bool:
    bundle = bundled_engine_dir()
    return bundle is not None and bundle.resolve() == engine_dir.resolve()


def _managed_engine(engine_dir: Path, source: str, cache: Path | None) -> Engine:
    """A verified engine directory with its engine.json. A bundled engine is part of the installed package, which may
    be read-only, so its check results live in the cache instead (keyed by its location)."""
    info = verify_managed(engine_dir)
    bin_dir = engine_dir / "bin"
    engine = Engine(bin_dir, info["version"], info["fingerprint"], source, info, engine_dir / "engine.json")
    if _is_bundle(engine_dir):
        engine.check_path = _external_record(cache, bin_dir)
        record = read_json(engine.check_path) if engine.check_path.is_file() else {}
        same_build = record.get("fingerprint") == info["fingerprint"]  # not a record left by an older wheel
        engine.info = {**info, "last_check": record.get("last_check") if same_build else None}
    return engine


def _bundled(version: str, cache: Path | None) -> Engine | None:
    engine_dir = bundled_engine_dir()
    if engine_dir is None:
        return None
    try:
        if read_json(engine_dir / "engine.json").get("version") != version:
            return None
        return _managed_engine(engine_dir, "bundled", cache)
    except (OSError, json.JSONDecodeError, EngineError) as error:
        log.warning(f"the bundled engine in {engine_dir} is unusable ({error}); looking in the cache instead")
        return None


def _from_bin_dir(bin_dir: Path, source: str, cache: Path | None) -> Engine:
    bin_dir = bin_dir.expanduser()
    if not bin_dir.is_dir():
        raise EngineError(f"BLAST binary directory not found: {bin_dir}")
    if (bin_dir.parent / "engine.json").is_file():
        return _managed_engine(bin_dir.parent, source, cache)
    missing = missing_binaries(bin_dir)
    if missing:
        raise EngineError(f"missing binaries in {bin_dir}: {', '.join(missing)}")
    version, version_line = blastp_version(bin_dir)
    record = _external_record(cache, bin_dir)
    info: dict[str, Any] = read_json(record) if record.is_file() else {}
    info.update({"version": version, "blastp_version": version_line, "bin_dir": str(bin_dir.resolve())})
    # Fingerprint of user-supplied binaries = hash of their behaviour in the engine tests (set by `engine check`).
    fingerprint = info.get("fingerprint", "unchecked")
    return Engine(bin_dir, version, fingerprint, source, info, record)


def list_managed(cache: Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """All engine directories in the cache with a readable engine.json, newest first."""
    root = engines_dir(cache)
    found = []
    if root.is_dir():
        for path in root.iterdir():
            info_path = path / "engine.json"
            if info_path.is_file():
                try:
                    found.append((path, read_json(info_path)))
                except (OSError, json.JSONDecodeError):
                    continue
    return sorted(found, key=lambda item: item[1].get("built_at", ""), reverse=True)


def find_engine(blast_bin: Path | str | None = None, version: str | None = None, cache: Path | None = None) -> Engine:
    """Locate a usable engine (see module docstring for the search order)."""
    if blast_bin:
        return _from_bin_dir(Path(blast_bin), "argument", cache)
    if os.environ.get("TEXTBLAST_BIN"):
        return _from_bin_dir(Path(os.environ["TEXTBLAST_BIN"]), "environment", cache)
    wanted = version or DEFAULT_VERSION
    bundled = _bundled(wanted, cache)
    if bundled is not None:
        return bundled
    for path, info in list_managed(cache):
        if info.get("version") != wanted:
            continue
        verified = verify_managed(path, info.get("fingerprint"))
        return Engine(path / "bin", wanted, verified["fingerprint"], "cache", verified, path / "engine.json")
    command = "textblast engine build" + ("" if wanted == DEFAULT_VERSION else f" --version {wanted}")
    bundle = "" if bundled_engine_dir() else "this installation has none bundled and "
    raise EngineError(
        f"no BLAST {wanted} engine found: {bundle}there is none in {engines_dir(cache)}; run `{command}` once "
        f"(it downloads and compiles BLAST, 10 to 30 minutes) or pass --blast-bin"
    )
