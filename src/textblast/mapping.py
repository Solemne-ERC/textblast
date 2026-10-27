"""The corpus store: encoded letters and their exact maps back to the surface text.

Layout of ``<run_dir>/corpus/``:

- ``documents.parquet``  doc_idx, doc_id, n_chars, n_letters, n_encoded, …, metadata
- ``texts.parquet``      doc_idx, text (exactly as given)
- ``units.parquet``      doc_idx, loc, start, length (code points)
- ``exclusions.parquet`` applied classified spans: doc_idx, start, end, mode, reason, source
- ``encoded.bin``        all documents' symbols, concatenated (ASCII; lowercase = masked; ``X`` runs = barriers)
- ``encoded_index.npy``  int64 offsets into ``encoded.bin`` (n_docs + 1 entries)
- ``map_start.npy`` / ``map_end.npy``  uint32 per symbol: its surface span, 0-based, end-exclusive code points
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CORPUS_DIR = "corpus"
FILES = (
    "documents.parquet",
    "texts.parquet",
    "units.parquet",
    "exclusions.parquet",
    "encoded.bin",
    "encoded_index.npy",
    "map_start.npy",
    "map_end.npy",
)


class CorpusStore:
    """Read access to a prepared corpus; arrays are memory-mapped, texts load on demand."""

    def __init__(self, run_dir: Path | str) -> None:
        self.dir = Path(run_dir) / CORPUS_DIR
        missing = [name for name in FILES if not (self.dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"incomplete corpus store in {self.dir}: missing {', '.join(missing)}")
        self.offsets = np.load(self.dir / "encoded_index.npy")
        total = int(self.offsets[-1])
        self.map_start = np.load(self.dir / "map_start.npy", mmap_mode="r") if total else np.zeros(0, np.uint32)
        self.map_end = np.load(self.dir / "map_end.npy", mmap_mode="r") if total else np.zeros(0, np.uint32)
        self.symbols = (
            np.memmap(self.dir / "encoded.bin", dtype=np.uint8, mode="r") if total else np.zeros(0, np.uint8)
        )
        self._texts: list[str] | None = None
        self._astral: dict[int, np.ndarray] = {}

    @property
    def n_docs(self) -> int:
        return len(self.offsets) - 1

    def documents(self) -> pl.DataFrame:
        return pl.read_parquet(self.dir / "documents.parquet")

    def units(self) -> pl.DataFrame:
        return pl.read_parquet(self.dir / "units.parquet")

    def exclusions(self) -> pl.DataFrame:
        return pl.read_parquet(self.dir / "exclusions.parquet")

    def encoded(self, doc_idx: int) -> bytes:
        """The document's symbols as BLAST will see them."""
        return bytes(self.symbols[self.offsets[doc_idx] : self.offsets[doc_idx + 1]])

    def encoded_length(self, doc_idx: int) -> int:
        return int(self.offsets[doc_idx + 1] - self.offsets[doc_idx])

    def text(self, doc_idx: int) -> str:
        if self._texts is None:
            self._texts = pl.read_parquet(self.dir / "texts.parquet")["text"].to_list()
        return self._texts[doc_idx]

    def to_surface(
        self, doc_idx: np.ndarray | int, enc_start: np.ndarray | int, enc_end: np.ndarray | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Surface spans of encoded ranges ``[enc_start, enc_end)`` (0-based, end-exclusive), vectorized.

        BLAST's 1-based inclusive ``qstart..qend`` is ``enc_start = qstart - 1``, ``enc_end = qend``.
        """
        doc_idx = np.asarray(doc_idx, dtype=np.int64)
        enc_start = np.asarray(enc_start, dtype=np.int64)
        enc_end = np.asarray(enc_end, dtype=np.int64)
        lengths = self.offsets[doc_idx + 1] - self.offsets[doc_idx]
        if np.any(enc_start < 0) or np.any(enc_end <= enc_start) or np.any(enc_end > lengths):
            raise ValueError("encoded ranges must satisfy 0 <= start < end <= encoded length")
        base = self.offsets[doc_idx]
        start = np.asarray(self.map_start[base + enc_start], dtype=np.int64)
        end = np.asarray(self.map_end[base + enc_end - 1], dtype=np.int64)
        return start, end

    def to_utf16(self, doc_idx: np.ndarray | int, offsets: np.ndarray | int) -> np.ndarray:
        """Code-point offsets → UTF-16 code-unit offsets (for JavaScript), vectorized per document."""
        doc_idx = np.atleast_1d(np.asarray(doc_idx, dtype=np.int64))
        offsets = np.atleast_1d(np.asarray(offsets, dtype=np.int64))
        doc_idx = np.broadcast_to(doc_idx, offsets.shape)
        result = offsets.copy()
        for doc in np.unique(doc_idx).tolist():
            astral = self._astral.get(doc)
            if astral is None:
                text = self.text(doc)
                astral = self._astral[doc] = np.array(
                    [i for i, c in enumerate(text) if ord(c) > 0xFFFF], dtype=np.int64
                )
            if len(astral):
                selected = doc_idx == doc
                result[selected] += np.searchsorted(astral, offsets[selected], side="left")
        return result
