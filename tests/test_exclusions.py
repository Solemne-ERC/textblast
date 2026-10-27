"""Classified spans: per-document field, separate file, pattern rules with metadata conditions, errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textblast.encode import encode_document
from textblast.ingest import IngestError, load_corpus
from textblast.normalize import BARRIER, DocContext, Exclusion, NormalizationError, Normalizer, PatternRule
from textblast.normalize.profiles import get_profile

TEXT = "alpha WATERMARK beta FORMULA gamma"


def normalize(text: str, exclusions=(), rules=(), metadata=None):
    ctx = DocContext(0, "d", len(text), metadata or {}, list(exclusions))
    doc = Normalizer(get_profile("latin").steps, rules).run(text, ctx)
    return doc, ctx


def span_of(word: str) -> tuple[int, int]:
    start = TEXT.index(word)
    return start, start + len(word)


def test_delete_mask_barrier_modes() -> None:
    w, f = span_of("WATERMARK"), span_of("FORMULA")
    doc, ctx = normalize(TEXT, [Exclusion(*w, "delete", "digitisation_watermark", "field"),
                                Exclusion(*f, "mask", "formula", "field")])
    encoded = encode_document(doc, get_profile("latin").encoding_table(), barrier_length=5)
    symbols = bytes(encoded.symbols).decode()
    table = get_profile("latin").encoding_table().mapping
    formula = "".join(table[c] for c in "formula").lower()
    assert symbols == "".join(table[c] for c in "alphabeta") + formula + "".join(table[c] for c in "gamma")
    # masked symbols map back to the formula's surface letters, one by one
    masked = [(TEXT[a:b]) for s, a, b in zip(symbols, encoded.start.tolist(), encoded.end.tolist(), strict=True)
              if s.islower()]
    assert "".join(masked) == "FORMULA"
    assert ctx.deleted["digitisation_watermark"] == len("WATERMARK")
    assert ctx.masked_chars["formula"] == len("FORMULA")
    assert {e.reason for e in ctx.applied} == {"digitisation_watermark", "formula"}

    b = span_of("beta")
    doc, _ = normalize(TEXT, [Exclusion(*b, "barrier", "break", "field")])
    barrier = [(a, z) for c, a, z in zip(doc.text, doc.start.tolist(), doc.end.tolist(), strict=True) if c == BARRIER]
    assert barrier == [b]
    encoded = encode_document(doc, get_profile("latin").encoding_table(), barrier_length=5)
    xs = [(a, z) for s, a, z in zip(bytes(encoded.symbols).decode(), encoded.start.tolist(), encoded.end.tolist(),
                                    strict=True) if s == "X"]
    assert xs == [b] * 5


def test_pattern_rules_respect_metadata_conditions() -> None:
    rule = PatternRule(r"\[fol\.? ?\d+[rv]?\]", "delete", "folio_marker", where={"source_type": "transcription"})
    text = "ante [fol. 12r] post"
    doc, ctx = normalize(text, rules=[rule], metadata={"source_type": "transcription"})
    assert " ".join(doc.text.split()) == "ante post" and ctx.deleted["folio_marker"] == len("[fol. 12r]")
    doc, ctx = normalize(text, rules=[rule], metadata={"source_type": "edition"})
    assert "folio_marker" not in ctx.deleted
    listed = PatternRule("x", "mask", "r", where={"series": ["a", "b"]})
    assert listed.applies_to({"series": "b"}) and not listed.applies_to({"series": "c"})


def test_precedence_exclusions_over_rules_over_gaps() -> None:
    text = "aa [...] bb"
    rule = PatternRule(r"\[\.\.\.\]", "mask", "rule")
    _, ctx = normalize(text, rules=[rule])
    assert ctx.masked_chars["rule"] == 5 and "declared_gap" not in ctx.deleted  # rule beats gap
    _, ctx = normalize(text, [Exclusion(3, 8, "delete", "explicit", "field")], rules=[rule])
    assert ctx.deleted["explicit"] == 5 and not ctx.masked_chars  # input beats rule


def test_spans_step_must_come_first() -> None:
    ctx = DocContext(0, "d", 3)
    with pytest.raises(NormalizationError, match="first step"):
        Normalizer(["graphemes", "spans"]).run("Æb", ctx)


def test_invalid_pattern_action() -> None:
    with pytest.raises(NormalizationError, match="action"):
        Normalizer(["spans"], [PatternRule("x", "erase", "r")])


# --------------------------------------------------------------------------- input validation


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_field_and_file_are_merged(tmp_path: Path) -> None:
    corpus = write_jsonl(tmp_path / "c.jsonl", [
        {"doc_id": "a", "text": TEXT, "exclusions": [{"start": 6, "length": 9, "mode": "delete", "reason": "w"}]},
    ])
    extra = write_jsonl(tmp_path / "x.jsonl", [{"doc_id": "a", "start": 21, "end": 28, "mode": "mask", "reason": "f"}])
    doc = load_corpus(corpus, exclusions_file=extra).documents[0]
    assert [(e.start, e.end, e.mode, e.source) for e in doc.exclusions] == [(6, 15, "delete", "field"),
                                                                            (21, 28, "mask", "file")]


@pytest.mark.parametrize(
    ("exclusions", "message"),
    [
        ([{"start": 30, "length": 10, "mode": "delete", "reason": "r"}], "past the end"),
        ([{"start": 0, "length": 3, "mode": "erase", "reason": "r"}], "mode must be"),
        ([{"start": 0, "length": 0, "mode": "delete", "reason": "r"}], "empty"),
        ([{"start": 0, "length": 3, "mode": "delete"}], "needs start"),
        ([{"start": 0, "length": 5, "mode": "delete", "reason": "a"},
          {"start": 3, "length": 5, "mode": "mask", "reason": "b"}], "different modes"),
    ],
)
def test_invalid_exclusions(tmp_path: Path, exclusions: list[dict], message: str) -> None:
    corpus = write_jsonl(tmp_path / "c.jsonl", [{"doc_id": "a", "text": TEXT, "exclusions": exclusions}])
    with pytest.raises(IngestError, match=message):
        load_corpus(corpus)


def test_same_mode_overlaps_are_fine(tmp_path: Path) -> None:
    corpus = write_jsonl(tmp_path / "c.jsonl", [{"doc_id": "a", "text": TEXT, "exclusions": [
        {"start": 0, "length": 5, "mode": "delete", "reason": "a"},
        {"start": 3, "length": 5, "mode": "delete", "reason": "b"}]}])
    assert len(load_corpus(corpus).documents[0].exclusions) == 2


def test_file_with_unknown_document(tmp_path: Path) -> None:
    corpus = write_jsonl(tmp_path / "c.jsonl", [{"doc_id": "a", "text": TEXT}])
    extra = write_jsonl(tmp_path / "x.jsonl", [{"doc_id": "zz", "start": 0, "end": 2, "mode": "mask", "reason": "f"}])
    with pytest.raises(IngestError, match="unknown doc_id"):
        load_corpus(corpus, exclusions_file=extra)
