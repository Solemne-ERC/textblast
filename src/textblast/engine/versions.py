"""Supported BLAST+ source releases.

Kept as a Python module rather than TOML: ``tomllib`` needs Python 3.11, and the package supports 3.10 without extra
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Version of the patch set. Bump it whenever a patcher's output changes: it is part of every engine fingerprint.
PATCHSET = "text-reuse-1"

# Binaries installed into an engine directory.
BINARIES = ("blastp", "makeblastdb", "blastdbcmd")

_URL = "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{v}/ncbi-blast-{v}+-src.tar.gz"


@dataclass(frozen=True)
class BlastRelease:
    """An NCBI BLAST+ source release and the patches it needs."""

    version: str
    sha256: str
    md5: str
    role: str
    patches: tuple[str, ...]

    @property
    def url(self) -> str:
        return _URL.format(v=self.version)

    @property
    def tarball_name(self) -> str:
        return f"ncbi-blast-{self.version}+-src.tar.gz"

    @property
    def source_dir_name(self) -> str:
        return f"ncbi-blast-{self.version}+-src"


TEXT_REUSE_PATCHES = ("blosum62_uniform", "blosum62_gap_3_11")

RELEASES: dict[str, BlastRelease] = {
    "2.17.0": BlastRelease(
        version="2.17.0",
        sha256="502057a88e9990e34e62758be21ea474cc0ad68d6a63a2e37b2372af1e5ea147",
        md5="d7e50e70e60d6bbd2caa12fdee4fe297",
        role="default",
        patches=TEXT_REUSE_PATCHES,
    ),
    "2.13.0": BlastRelease(
        version="2.13.0",
        sha256="89553714d133daf28c477f83d333794b3c62e4148408c072a1b4620e5ec4feb2",
        md5="0542f0e7b148d02db0854965e78e36c4",
        role="reference",
        patches=(*TEXT_REUSE_PATCHES, "bitmagic_allocator"),
    ),
}

DEFAULT_VERSION = "2.17.0"
REFERENCE_VERSION = "2.13.0"


def get_release(version: str) -> BlastRelease:
    try:
        return RELEASES[version]
    except KeyError:
        known = ", ".join(sorted(RELEASES))
        raise ValueError(f"unsupported BLAST version {version!r} (supported: {known})") from None
