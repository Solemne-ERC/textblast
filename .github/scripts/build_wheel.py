"""Build a platform wheel that carries the default engine (blastp, makeblastdb, blastdbcmd) under textblast/_engine.

The package has no extension module, so one wheel per platform serves every Python >= 3.10 (tag py3-none-<platform>).
The sources are copied to a staging folder first, so the engine never lands in the working tree.

    python .github/scripts/build_wheel.py --plat-name macosx_11_0_arm64 [--engine-dir DIR] [--out dist]

On Linux, `auditwheel repair` runs afterwards: it copies the libraries that manylinux does not allow into the wheel
and sets the platform tag. On macOS the binaries may link only system libraries, which this script checks, and they
are signed again after stripping (arm64 refuses to run binaries whose signature no longer matches).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from textblast import __version__
from textblast.engine.build import BuildOptions, fingerprint, resolve_openmp
from textblast.engine.locate import BUNDLE_DIR, default_cache_dir, verify_managed
from textblast.engine.versions import BINARIES, DEFAULT_VERSION, get_release

REPO = Path(__file__).resolve().parents[2]
SOURCES = ("src", "pyproject.toml", "README.md", "LICENSE", "licenses")
SYSTEM_LIBRARIES = ("/usr/lib/", "/System/Library/")


def default_engine_dir(openmp: str = "auto") -> Path:
    release = get_release(DEFAULT_VERSION)
    fp = fingerprint(release, BuildOptions(openmp=resolve_openmp(openmp), projects="restricted"))
    return default_cache_dir() / "engine" / f"{release.version}-{fp}"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macos_target(plat_name: str) -> tuple[int, ...]:
    match = re.fullmatch(r"macosx_(\d+)_(\d+)_\w+", plat_name)
    if not match:
        raise SystemExit(f"not a macOS platform tag: {plat_name}")
    return int(match.group(1)), int(match.group(2))


def check_macos_binary(path: Path, plat_name: str) -> None:
    """Only system libraries, and built for a macOS no newer than the wheel's tag."""
    linked = run(["otool", "-L", str(path)], capture_output=True).stdout.splitlines()[1:]
    others = [line.split()[0] for line in linked if not line.strip().startswith(SYSTEM_LIBRARIES)]
    if others:
        raise SystemExit(f"{path.name} links libraries outside the system: {', '.join(others)}")
    load_commands = run(["otool", "-l", str(path)], capture_output=True).stdout
    match = re.search(r"cmd LC_BUILD_VERSION.*?minos (\d+)\.(\d+)", load_commands, re.S)
    if not match:
        raise SystemExit(f"{path.name}: no LC_BUILD_VERSION, cannot tell the oldest macOS it runs on")
    minos = (int(match.group(1)), int(match.group(2)))
    if minos > macos_target(plat_name):
        raise SystemExit(f"{path.name} needs macOS {minos[0]}.{minos[1]}, newer than the tag {plat_name}; "
                         f"build the engine with MACOSX_DEPLOYMENT_TARGET set")
    run(["codesign", "--verify", "--strict", str(path)])


def bundle_engine(engine_dir: Path, bundle: Path, plat_name: str) -> dict:
    info = verify_managed(engine_dir)
    (bundle / "bin").mkdir(parents=True)
    binaries = {}
    for name in BINARIES:
        target = bundle / "bin" / name
        shutil.copy2(engine_dir / "bin" / name, target)
        run(["strip", str(target)])
        if sys.platform == "darwin":
            run(["codesign", "--force", "--sign", "-", str(target)])
            check_macos_binary(target, plat_name)
        binaries[name] = {"size": target.stat().st_size, "sha256": sha256(target)}
    info.update({
        "binaries": binaries,
        "last_check": None,  # checked again where it is installed; the results go to the user's cache
        "bundle": {"platform": plat_name, "stripped": True, "bundled_by": f"textblast {__version__}"},
    })
    (bundle / "engine.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plat-name", required=True, help="e.g. macosx_11_0_arm64 or linux_x86_64")
    parser.add_argument("--engine-dir", type=Path, default=None, help="default: the default engine in the cache")
    parser.add_argument("--openmp", default="auto", help="which default engine (auto, on, off), as built")
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    args = parser.parse_args()
    engine_dir = args.engine_dir or default_engine_dir(args.openmp)

    with tempfile.TemporaryDirectory(prefix="textblast-wheel-") as tmp:
        staging = Path(tmp) / "textblast"
        staging.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.egg-info", BUNDLE_DIR)
        for name in SOURCES:
            source = REPO / name
            if source.is_dir():
                shutil.copytree(source, staging / name, ignore=ignore)
            else:
                shutil.copy2(source, staging / name)
        info = bundle_engine(engine_dir, staging / "src" / "textblast" / BUNDLE_DIR, args.plat_name)
        args.out.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(args.out),
             f"--config-settings=--build-option=--plat-name={args.plat_name}", str(staging)])

    wheel = next(args.out.glob(f"textblast-{__version__}-py3-none-{args.plat_name}.whl"))
    size = sum(b["size"] for b in info["binaries"].values())
    print(f"{wheel.name}: {wheel.stat().st_size / 1e6:.1f} MB (stripped binaries {size / 1e6:.1f} MB; "
          f"engine {info['version']} {info['fingerprint']})")


if __name__ == "__main__":
    main()
