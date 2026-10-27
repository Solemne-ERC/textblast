"""Reading and validating the input corpus (plan section 5).

Input: JSONL or JSONL.gz, one document per line with ``doc_id``, ``text``, optional metadata (passed through), optional
units (passim ``locs`` format ``[{loc, start, length}]``) and optional classified spans
(``exclusions: [{start, length, mode, reason}]``). Offsets are 0-based code points of ``text`` as given.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import polars as pl

from .normalize.steps import MODES, Exclusion


class IngestError(ValueError):
    """The input violates the expected format; the message names the file and line."""


@dataclass
class Document:
    doc_idx: int
    doc_id: str
    text: str
    metadata: dict[str, Any]
    exclusions: list[Exclusion] = field(default_factory=list)


@dataclass
class Corpus:
    documents: list[Document]
    units: pl.DataFrame  # doc_idx, loc, start, length
    n_chars: int

    @property
    def n_docs(self) -> int:
        return len(self.documents)


def _open(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, record) for every non-empty line."""
    with _open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise IngestError(f"{path}:{number}: invalid JSON ({error.msg})") from None
            if not isinstance(record, dict):
                raise IngestError(f"{path}:{number}: each line must be a JSON object")
            yield number, record


def _check_int(value: Any, what: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IngestError(f"{where}: {what} must be a non-negative integer, got {value!r}")
    return value


def _parse_units(value: Any, text_length: int, where: str) -> list[tuple[str, int, int]]:
    if not isinstance(value, list):
        raise IngestError(f"{where}: units must be a list of {{loc, start, length}}")
    units = []
    for k, unit in enumerate(value):
        if not isinstance(unit, dict) or not {"loc", "start", "length"} <= set(unit):
            raise IngestError(f"{where}: unit {k} must have loc, start and length")
        start = _check_int(unit["start"], f"unit {k} start", where)
        length = _check_int(unit["length"], f"unit {k} length", where)
        if start + length > text_length:
            raise IngestError(f"{where}: unit {k} ({start}+{length}) runs past the end of the text ({text_length})")
        units.append((str(unit["loc"]), start, length))
    return units


def _parse_exclusion(item: Any, text_length: int, where: str, source: str, end_key: str = "length") -> Exclusion:
    if not isinstance(item, dict) or not {"start", end_key, "mode", "reason"} <= set(item):
        raise IngestError(f"{where}: an exclusion needs start, {end_key}, mode and reason")
    start = _check_int(item["start"], "exclusion start", where)
    if end_key == "length":
        end = start + _check_int(item["length"], "exclusion length", where)
    else:
        end = _check_int(item["end"], "exclusion end", where)
    if end <= start:
        raise IngestError(f"{where}: exclusion [{start}, {end}) is empty")
    if end > text_length:
        raise IngestError(f"{where}: exclusion [{start}, {end}) runs past the end of the text ({text_length})")
    if item["mode"] not in MODES:
        raise IngestError(f"{where}: exclusion mode must be one of {', '.join(MODES)}, got {item['mode']!r}")
    if not isinstance(item["reason"], str) or not item["reason"]:
        raise IngestError(f"{where}: exclusion reason must be a non-empty string")
    return Exclusion(start, end, item["mode"], item["reason"], source)


def _check_overlaps(exclusions: list[Exclusion], doc_id: str) -> None:
    """Overlapping spans with different modes are contradictory input."""
    ordered = sorted(exclusions, key=lambda e: (e.start, e.end))
    active: list[Exclusion] = []
    for span in ordered:
        active = [a for a in active if a.end > span.start]
        for other in active:
            if other.mode != span.mode:
                raise IngestError(
                    f"document {doc_id!r}: exclusions [{other.start}, {other.end}) {other.mode} ({other.source}) and "
                    f"[{span.start}, {span.end}) {span.mode} ({span.source}) overlap with different modes"
                )
        active.append(span)


def load_corpus(
    path: Path,
    locs_field: str = "locs",
    exclusions_field: str = "exclusions",
    exclusions_file: Path | None = None,
) -> Corpus:
    """Read, validate and index the corpus (documents keep input order: ``doc_idx`` = position)."""
    documents: list[Document] = []
    seen: dict[str, int] = {}
    unit_rows: dict[str, list[Any]] = {"doc_idx": [], "loc": [], "start": [], "length": []}
    n_chars = 0
    for number, record in read_jsonl(path):
        where = f"{path}:{number}"
        doc_id = record.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise IngestError(f"{where}: doc_id must be a non-empty string")
        if doc_id in seen:
            raise IngestError(f"{where}: duplicate doc_id {doc_id!r} (first on document {seen[doc_id]})")
        text = record.get("text")
        if not isinstance(text, str):
            raise IngestError(f"{where}: text must be a string")
        doc_idx = len(documents)
        seen[doc_id] = doc_idx
        n_chars += len(text)
        metadata = {k: v for k, v in record.items() if k not in ("doc_id", "text", locs_field, exclusions_field)}
        if locs_field in record and record[locs_field] is not None:
            for loc, start, length in _parse_units(record[locs_field], len(text), where):
                unit_rows["doc_idx"].append(doc_idx)
                unit_rows["loc"].append(loc)
                unit_rows["start"].append(start)
                unit_rows["length"].append(length)
        exclusions = []
        if record.get(exclusions_field) is not None:
            if not isinstance(record[exclusions_field], list):
                raise IngestError(f"{where}: {exclusions_field} must be a list")
            exclusions = [_parse_exclusion(item, len(text), where, "field") for item in record[exclusions_field]]
        documents.append(Document(doc_idx, doc_id, text, metadata, exclusions))

    if exclusions_file is not None:
        for number, record in read_jsonl(exclusions_file):
            where = f"{exclusions_file}:{number}"
            doc_idx = seen.get(record.get("doc_id"))  # type: ignore[arg-type]
            if doc_idx is None:
                raise IngestError(f"{where}: unknown doc_id {record.get('doc_id')!r}")
            doc = documents[doc_idx]
            doc.exclusions.append(_parse_exclusion(record, len(doc.text), where, "file", end_key="end"))
    for doc in documents:
        if doc.exclusions:
            _check_overlaps(doc.exclusions, doc.doc_id)

    units = pl.DataFrame(
        unit_rows, schema={"doc_idx": pl.Int32, "loc": pl.String, "start": pl.Int64, "length": pl.Int64}
    )
    return Corpus(documents, units, n_chars)


def documents_frame(corpus: Corpus) -> pl.DataFrame:
    """doc_idx, doc_id, n_chars, then the metadata fields (types inferred by Polars)."""
    base = pl.DataFrame(
        {
            "doc_idx": [d.doc_idx for d in corpus.documents],
            "doc_id": [d.doc_id for d in corpus.documents],
            "n_chars": [len(d.text) for d in corpus.documents],
        },
        schema={"doc_idx": pl.Int32, "doc_id": pl.String, "n_chars": pl.Int64},
    )
    keys: list[str] = []
    for doc in corpus.documents:
        keys += [k for k in doc.metadata if k not in keys]
    if not keys:
        return base
    columns = {}
    for key in keys:
        values = [d.metadata.get(key) for d in corpus.documents]
        try:
            columns[key] = pl.Series(key, values, strict=False)
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            columns[key] = pl.Series(key, [None if v is None else json.dumps(v) for v in values])
    return base.hstack(list(columns.values()))
