"""Checks on the user's corpus (marker ``corpus``; skipped when data/corpus.jsonl or docs/reference is absent)."""

from __future__ import annotations

import importlib.util
import io
import random
import re

import numpy as np
import pytest

from textblast import logs
from textblast.ingest import load_corpus
from textblast.mapping import CorpusStore
from textblast.normalize import DocContext, Normalizer
from textblast.normalize.profiles import get_profile
from textblast.normalize.steps import GAP_PATTERN, SCAFFOLD_PATTERN
from textblast.prepare import IngestOptions, PrepareConfig, prepare

from .conftest import REFERENCE
from .helpers import CORPUS, letters

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not CORPUS.is_file(), reason="data/corpus.jsonl not present"),
]

TEXT_ENCODER = REFERENCE / "code" / "textreuse-blast-master" / "text_encoder.py"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(CORPUS, locs_field="units")


@pytest.fixture(scope="module")
def reference_encoder():
    if not TEXT_ENCODER.is_file():
        pytest.skip("docs/reference/code/textreuse-blast-master not present")
    spec = importlib.util.spec_from_file_location("reference_text_encoder", TEXT_ENCODER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TextEncoder


@pytest.mark.parametrize(("profile", "language"), [("compat-eng", "eng"), ("compat-fin", "fin")])
def test_compat_profiles_match_textreuse_blast_byte_for_byte(corpus, reference_encoder, profile, language) -> None:
    from textblast.encode import encode_document

    reference = reference_encoder(language)
    table = get_profile(profile).encoding_table()
    normalizer = Normalizer(get_profile(profile).steps)
    for doc in corpus.documents:
        ctx = DocContext(doc.doc_idx, doc.doc_id, len(doc.text))
        encoded = encode_document(normalizer.run(doc.text, ctx), table)
        assert bytes(encoded.symbols).decode() == reference.encode_text(doc.text), doc.doc_id
        # and the offsets are exact: each symbol comes from one surface character that maps to it
        start = encoded.start.astype(np.int64)
        assert np.all(encoded.end.astype(np.int64) - start == 1)
        surface = [doc.text[i].lower() for i in start.tolist()]
        assert [table.mapping[c] for c in surface] == list(bytes(encoded.symbols).decode()), doc.doc_id


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> CorpusStore:
    logs.setup_logging(stream=io.StringIO())
    run_dir = tmp_path_factory.mktemp("corpus_run")
    prepare(PrepareConfig(input=str(CORPUS), run_dir=str(run_dir), ingest=IngestOptions(locs_field="units")))
    return CorpusStore(run_dir)


def test_spot_check_1000_random_positions(store: CorpusStore) -> None:
    """Acceptance criterion 4: decoding through the maps returns surface text that normalizes back to the letter."""
    table = get_profile("latin-broad").encoding_table()
    inverse = {symbol: letter for letter, symbol in table.mapping.items()}
    rng = random.Random(2026)
    total = int(store.offsets[-1])
    checked = 0
    for position in rng.sample(range(total), 1000):
        doc = int(np.searchsorted(store.offsets, position, side="right") - 1)
        a, b = int(store.map_start[position]), int(store.map_end[position])
        surface = store.text(doc)[a:b]
        symbol = chr(store.symbols[position])
        if symbol == "X":
            assert re.fullmatch(GAP_PATTERN, surface) or re.fullmatch(SCAFFOLD_PATTERN, surface)
            continue
        expected = letters("latin-broad", surface).replace(" ", "")
        letter = inverse[symbol.upper()]
        assert letter in expected or (letter == "t" and "c" in expected), (doc, a, b, letter, expected)
        checked += 1
    assert checked >= 990
