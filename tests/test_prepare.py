"""End-to-end ``prepare`` on small corpora: store layout, maps, warnings, manifest, reruns, parallel = serial."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from textblast import cli, logs
from textblast.mapping import FILES, CorpusStore
from textblast.prepare import IngestOptions, NormalizeOptions, PatternRuleOptions, PrepareConfig, prepare

DOCS = [
    {"doc_id": "a", "text": "Hæc est uita æterna, ut cognoscant te [...] solum Deum uerum.", "series": "x",
     "date": 1000, "units": [{"loc": "a.1", "start": 0, "length": 20}],
     "exclusions": [{"start": 21, "length": 5, "mode": "mask", "reason": "formula"}]},
    {"doc_id": "b", "text": "Et uitam æternam 𓀀 dñs [fol. 3v] habeant.", "series": "y", "date": None, "units": []},
    {"doc_id": "c", "text": "123 — 456", "series": "x", "date": 1100},
]


def write(path: Path, docs: list[dict], gz: bool = False) -> Path:
    data = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def config(tmp_path: Path, corpus: Path, workers: int = 1, run: str = "run") -> PrepareConfig:
    return PrepareConfig(
        input=str(corpus),
        run_dir=str(tmp_path / run),
        workers=workers,
        ingest=IngestOptions(locs_field="units"),
        normalize=NormalizeOptions(
            pattern_rules=[PatternRuleOptions(r"\[fol\. ?\d+[rv]?\]", "delete", "folio_marker")]
        ),
    )


@pytest.fixture
def quiet_console() -> io.StringIO:
    stream = io.StringIO()
    logs.setup_logging(stream=stream)
    return stream


def test_store_layout_and_maps(tmp_path: Path, quiet_console: io.StringIO) -> None:
    run_dir = prepare(config(tmp_path, write(tmp_path / "c.jsonl", DOCS)))
    store = CorpusStore(run_dir)
    assert store.n_docs == 3 and all((store.dir / name).exists() for name in FILES)

    documents = store.documents()
    assert documents.columns[:3] == ["doc_idx", "doc_id", "n_chars"]
    assert {"series", "date", "n_letters", "n_encoded", "empty"} <= set(documents.columns)
    assert documents["empty"].to_list() == [False, False, True]
    assert store.units().to_dicts() == [{"doc_idx": 0, "loc": "a.1", "start": 0, "length": 20}]
    reasons = set(store.exclusions()["reason"].to_list())
    assert {"formula", "folio_marker", "declared_gap"} <= reasons

    encoded = store.encoded(0).decode()
    assert any(c.islower() for c in encoded)  # the masked formula ("cognoscant" is not masked: span 21..26)
    assert "X" * 100 in encoded  # the declared gap
    # encoded ranges map back to exact surface spans: "Hæc est" → e(æ) c e … (H deleted, æ → ae → e)
    start, end = store.to_surface(0, 0, 3)
    assert store.text(0)[int(start):int(end)] == "æc e"
    s, e = store.to_surface(np.array([0, 1]), np.array([0, 0]), np.array([1, 2]))
    assert (s.tolist(), e.tolist()) == ([1, 0], [2, 2])  # "æ" in document 0; "Et" in document 1
    with pytest.raises(ValueError):
        store.to_surface(0, 5, 5)


def test_to_utf16_counts_astral_characters(tmp_path: Path, quiet_console: io.StringIO) -> None:
    run_dir = prepare(config(tmp_path, write(tmp_path / "c.jsonl", DOCS)))
    store = CorpusStore(run_dir)
    text = store.text(1)
    astral = text.index("𓀀")
    offsets = np.array([0, astral, astral + 1, len(text)])
    assert store.to_utf16(1, offsets).tolist() == [0, astral, astral + 2, len(text) + 1]
    assert len(text.encode("utf-16-le")) // 2 == len(text) + 1


def test_warnings_and_manifest(tmp_path: Path, quiet_console: io.StringIO) -> None:
    run_dir = prepare(config(tmp_path, write(tmp_path / "c.jsonl", DOCS)))
    warnings = run_dir / "logs" / "warnings"
    unsupported = pl.read_csv(warnings / "unsupported_chars.tsv", separator="\t")
    assert {"𓀀", "ñ"} <= set(unsupported["character"].to_list())
    assert (warnings / "empty_documents.tsv").read_text().count("\n") == 2
    by_reason = pl.read_csv(warnings / "exclusions_by_reason.tsv", separator="\t")
    assert set(by_reason["reason"].to_list()) == {"formula", "folio_marker", "declared_gap"}
    manifest = json.loads((run_dir / "manifest.json").read_text())["stages"]["prepare"]
    assert manifest["profile"] == "latin-broad" and manifest["summary"]["documents"] == 3
    assert manifest["table"]["i"] == "G" and manifest["table"]["a"] == "A"  # the reference order (since M5)
    assert (run_dir / "config.yaml").is_file() and (run_dir / "logs" / "run.log").is_file()
    console = quiet_console.getvalue()
    assert "[1/3] ingest" in console and "[3/3] store done" in console
    assert "warning: 1 documents have no letters" in console


def test_rerun_is_a_noop_until_something_changes(tmp_path: Path, quiet_console: io.StringIO) -> None:
    corpus = write(tmp_path / "c.jsonl", DOCS)
    prepare(config(tmp_path, corpus))
    prepare(config(tmp_path, corpus, workers=2))  # the number of workers does not matter
    assert "up to date; nothing to do" in quiet_console.getvalue()
    changed = config(tmp_path, corpus)
    changed.normalize.profile = "latin"
    prepare(changed)
    assert quiet_console.getvalue().count("[1/3] ingest done") == 2


def test_store_from_older_code_or_table_is_rebuilt(tmp_path: Path, quiet_console: io.StringIO) -> None:
    """M5 changed latin-broad's table: a store made before (other code, or the frozen table) is never reused."""
    corpus = write(tmp_path / "c.jsonl", DOCS)
    frozen = config(tmp_path, corpus)
    frozen.normalize.profile = "latin-broad-frozen"
    run_dir = prepare(frozen)
    old_symbols = bytes(CorpusStore(run_dir).symbols)
    prepare(config(tmp_path, corpus))  # the new latin-broad: rebuilt with the reference table
    assert quiet_console.getvalue().count("[1/3] ingest done") == 2
    assert bytes(CorpusStore(run_dir).symbols) != old_symbols
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"]["prepare"]["fingerprint"] = "made-before-m5"  # as if written by older code
    manifest_path.write_text(json.dumps(manifest))
    prepare(config(tmp_path, corpus))
    assert quiet_console.getvalue().count("[1/3] ingest done") == 3


def test_parallel_equals_serial_and_gzip(tmp_path: Path, quiet_console: io.StringIO) -> None:
    many = [dict(d, doc_id=f"{d['doc_id']}{k}") for k in range(30) for d in DOCS]
    serial = CorpusStore(prepare(config(tmp_path, write(tmp_path / "c.jsonl", many), workers=1, run="s")))
    parallel = CorpusStore(prepare(config(tmp_path, write(tmp_path / "c.jsonl.gz", many, gz=True), workers=3,
                                          run="p")))
    assert np.array_equal(serial.offsets, parallel.offsets)
    assert np.array_equal(np.asarray(serial.symbols), np.asarray(parallel.symbols))
    assert np.array_equal(np.asarray(serial.map_start), np.asarray(parallel.map_start))
    assert np.array_equal(np.asarray(serial.map_end), np.asarray(parallel.map_end))


def test_cli_prepare_with_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = write(tmp_path / "c.jsonl", DOCS)
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        f"input: {corpus}\nrun_dir: {tmp_path / 'cli'}\nworkers: 1\ningest:\n  locs_field: units\n"
        "normalize:\n  profile: latin\n  pattern_rules:\n    - {pattern: 'fol', action: mask, reason: r}\n",
        encoding="utf-8",
    )
    stream = io.StringIO()
    monkeypatch.setattr(cli, "setup_logging", lambda verbosity: logs.setup_logging(verbosity, stream=stream))
    assert cli.main(["prepare", "--config", str(yaml_path)]) == 0
    manifest = json.loads((tmp_path / "cli" / "manifest.json").read_text())["stages"]["prepare"]
    assert manifest["profile"] == "latin" and manifest["config"]["normalize"]["pattern_rules"][0]["reason"] == "r"
    assert cli.main(["prepare", "--config", str(yaml_path), "--normalize.profile", "nope"]) == 1
    assert "unknown profile 'nope'" in stream.getvalue()


def test_ingest_errors_name_file_and_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = write(tmp_path / "c.jsonl", [{"doc_id": "a", "text": "x"}, {"doc_id": "a", "text": "y"}])
    stream = io.StringIO()
    monkeypatch.setattr(cli, "setup_logging", lambda verbosity: logs.setup_logging(verbosity, stream=stream))
    code = cli.main(["prepare", "--input", str(corpus), "--run_dir", str(tmp_path / "r"), "--workers", "1"])
    assert code == 1 and "c.jsonl:2: duplicate doc_id 'a'" in stream.getvalue()
