"""Write the default engine's cache paths and key to $GITHUB_OUTPUT, without building anything.

    python engine_key.py [auto|on|off]   (the OpenMP choice, as given to `textblast engine build --openmp`)

The key is the engine fingerprint (BLAST version, source checksum, patches, build options, system and machine), so a
cache hit is exactly the engine that `textblast engine build` would produce.
"""

from __future__ import annotations

import os
import sys

from textblast.engine.build import BuildOptions, fingerprint, resolve_openmp
from textblast.engine.locate import default_cache_dir
from textblast.engine.versions import DEFAULT_VERSION, get_release


def main() -> None:
    openmp = sys.argv[1] if len(sys.argv) > 1 else "auto"  # as given to `textblast engine build --openmp`
    release = get_release(DEFAULT_VERSION)
    options = BuildOptions(openmp=resolve_openmp(openmp), projects="restricted")
    fp = fingerprint(release, options)
    cache = default_cache_dir()
    values = {
        "version": release.version,
        "fingerprint": fp,
        "engine_dir": str(cache / "engine" / f"{release.version}-{fp}"),
        "tarball": str(cache / "src" / release.tarball_name),
        "build_dir": str(cache / "build"),
        "key": f"engine-{release.version}-{fp}",
    }
    lines = [f"{name}={value}" for name, value in values.items()]
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
