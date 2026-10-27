"""Python API: ``open_run(path)`` → :class:`Run`, a thin reader over a run directory or a bundle (Q35/Q36).

Files are the interface: tables come back as Polars LazyFrames, nothing is recomputed, and nothing is held besides
lazily loaded texts. A bundle (``textblast export --format bundle``) has the core tables and optionally the texts,
but not the corpus store's maps, so :meth:`Run.to_surface` works only on a run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

BUNDLE_MARKER = "bundle.json"


class RunNotAvailable(LookupError):
    """A table or capability that this run directory or bundle does not have."""


class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.is_bundle = (path / BUNDLE_MARKER).is_file()
        self._texts: list[str] | None = None
        self._astral: dict[int, np.ndarray] = {}
        self._store: Any = None

    def __repr__(self) -> str:
        return f"Run({str(self.path)!r}{', bundle' if self.is_bundle else ''})"

    # ------------------------------------------------------------------ files
    def _file(self, name: str) -> Path:
        if self.is_bundle:
            return self.path / f"{name}.parquet"
        if name in ("documents", "texts", "units"):
            return self.path / "corpus" / f"{name}.parquet"
        return self.path / "tables" / f"{name}.parquet"

    def table(self, name: str) -> pl.LazyFrame:
        path = self._file(name)
        if not path.is_file():
            hint = {"hsps_clipped": " (set units.clip: true and run `post`)",
                    "hsps_joined": " (set join.enabled: true and run `post`)",
                    "join_map": " (run `post`; every run since M6 writes it)",
                    "clusters": " (set cluster.enabled: true and run `post`)",
                    "cluster_summary": " (set cluster.enabled: true and run `post`)",
                    "pairs_filtered": " (set filters.hit_filter and run `post`)",
                    "texts": " (the bundle was made without texts)"}.get(name, " (run `post`)")
            raise RunNotAvailable(f"{self} has no table {name!r}{hint}")
        return pl.scan_parquet(path)

    @property
    def manifest(self) -> dict[str, Any]:
        name = BUNDLE_MARKER if self.is_bundle else "manifest.json"
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    @property
    def documents(self) -> pl.LazyFrame:
        return self.table("documents")

    @property
    def hsps(self) -> pl.LazyFrame:
        return self.table("hsps")

    @property
    def hsps_clipped(self) -> pl.LazyFrame:
        return self.table("hsps_clipped")

    @property
    def joined(self) -> pl.LazyFrame:
        """Joined hits (``hsps_joined``): neighbouring hits of a pair joined into one (``join_map`` has members)."""
        return self.table("hsps_joined")

    @property
    def join_map(self) -> pl.LazyFrame:
        return self.table("join_map")

    @property
    def clusters(self) -> pl.LazyFrame:
        return self.table("clusters")

    @property
    def cluster_summary(self) -> pl.LazyFrame:
        return self.table("cluster_summary")

    @property
    def pieces(self) -> pl.LazyFrame:
        return self.table("pieces")

    @property
    def piece_map(self) -> pl.LazyFrame:
        return self.table("piece_map")

    @property
    def pairs(self) -> pl.LazyFrame:
        return self.table("pairs")

    @property
    def pairs_filtered(self) -> pl.LazyFrame:
        return self.table("pairs_filtered")

    @property
    def units(self) -> pl.LazyFrame:
        return self.table("units")

    @property
    def unit_overlaps(self) -> pl.LazyFrame:
        return self.table("unit_overlaps")

    # ------------------------------------------------------------------ text
    def texts(self) -> list[str]:
        if self._texts is None:
            self._texts = self.table("texts").sort("doc_idx").collect()["text"].to_list()
        return self._texts

    def excerpt(self, doc: Any, start: Any, end: Any) -> Any:
        """Surface text of ``[start, end)`` in ``doc``; scalars give a string, arrays a list of strings."""
        texts = self.texts()
        if np.ndim(doc) == 0 and np.ndim(start) == 0:
            return texts[int(doc)][int(start) : int(end)]
        d, s, e = (np.broadcast_to(np.asarray(x, dtype=np.int64), np.broadcast(doc, start, end).shape)
                   for x in (doc, start, end))
        return [texts[a][b:c] for a, b, c in zip(d.tolist(), s.tolist(), e.tolist(), strict=True)]

    def to_utf16(self, doc: Any, offsets: Any) -> np.ndarray:
        """Code-point offsets → UTF-16 code-unit offsets (for JavaScript), vectorized."""
        doc_idx = np.atleast_1d(np.asarray(doc, dtype=np.int64))
        offsets = np.atleast_1d(np.asarray(offsets, dtype=np.int64))
        doc_idx = np.broadcast_to(doc_idx, offsets.shape)
        result = offsets.copy()
        texts = self.texts()
        for d in np.unique(doc_idx).tolist():
            astral = self._astral.get(d)
            if astral is None:
                astral = self._astral[d] = np.array([i for i, c in enumerate(texts[d]) if ord(c) > 0xFFFF],
                                                    dtype=np.int64)
            if len(astral):
                rows = doc_idx == d
                result[rows] += np.searchsorted(astral, offsets[rows], side="left")
        return result

    def to_surface(self, doc: Any, enc_start: Any, enc_end: Any) -> tuple[np.ndarray, np.ndarray]:
        """Encoded ranges → surface spans (needs the corpus store: run directories only)."""
        if self.is_bundle:
            raise RunNotAvailable(f"{self}: to_surface needs the corpus store's maps, which bundles do not include")
        if self._store is None:
            from .mapping import CorpusStore

            self._store = CorpusStore(self.path)
        return self._store.to_surface(doc, enc_start, enc_end)


def open_run(path: str | Path) -> Run:
    """Open a run directory (with ``manifest.json``) or a bundle (with ``bundle.json``)."""
    path = Path(path).expanduser()
    if not (path / "manifest.json").is_file() and not (path / BUNDLE_MARKER).is_file():
        raise FileNotFoundError(f"{path} is neither a run directory nor a bundle")
    return Run(path)
