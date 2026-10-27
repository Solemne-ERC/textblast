"""Encoding tables, errors, masks, barriers, and the frequency-based table suggestion."""

from __future__ import annotations

import pytest

from textblast.encode import SINGLE_SYMBOLS, SYMBOLS, EncodingError, EncodingTable, encode_document, suggest_table
from textblast.engine.check import GROUP
from textblast.normalize import BARRIER, Spanned
from textblast.normalize.profiles import LATIN_BROAD_REFTABLE, PROFILES, get_profile


def test_collision_is_an_error() -> None:
    with pytest.raises(EncodingError, match="both map to A"):
        EncodingTable("t", {"a": "A", "n": "A"}).validate()


def test_x_for_a_letter_is_an_error() -> None:
    with pytest.raises(EncodingError, match="X never matches itself"):
        EncodingTable("t", {"k": "X"}).validate()


def test_too_many_letters_is_an_error() -> None:
    letters = "abcdefghijklmnopqrstuvwx"  # 24
    with pytest.raises(EncodingError, match="24 letters"):
        EncodingTable("t", dict(zip(letters, SYMBOLS + "A", strict=True))).validate()


def test_compat_tables_may_collide() -> None:
    get_profile("compat-eng").encoding_table()  # a/n → W and k → X are allowed there
    assert get_profile("compat-eng").allow_collisions and not get_profile("latin-broad").allow_collisions


def test_all_profiles_have_valid_tables() -> None:
    for profile in PROFILES.values():
        table = profile.encoding_table()
        assert table.mapping
    assert set(get_profile("latin-broad").table) == set("abcdefgilmnopqrstuxz")
    assert set(get_profile("latin").table) == set("abcdefghilmnopqrstuxz")
    assert set(LATIN_BROAD_REFTABLE) == set(get_profile("latin-broad").table)
    # latin-broad uses the established order; the earlier table stays, unchanged, as latin-broad-frozen
    assert dict(get_profile("latin-broad").table) == LATIN_BROAD_REFTABLE
    assert dict(get_profile("latin-broad-frozen").table) == {
        "i": "A", "e": "G", "u": "N", "t": "F", "s": "P", "a": "Y", "n": "H", "r": "C", "o": "W",
        "m": "D", "c": "S", "z": "T", "d": "I", "x": "V", "p": "L", "f": "M", "l": "K", "g": "R", "b": "E", "q": "Q",
    }


def test_unknown_letter_is_reported() -> None:
    doc = Spanned.from_surface("aw")
    with pytest.raises(EncodingError, match="no symbol for 'w'"):
        encode_document(doc, get_profile("latin-broad").encoding_table())


def test_mask_and_barrier_encoding() -> None:
    doc = Spanned.from_surface("ab" + BARRIER + " c")
    doc.mask[1] = True
    table = EncodingTable("t", {"a": "A", "b": "R", "c": "N"})
    encoded = encode_document(doc, table, barrier_length=3)
    assert bytes(encoded.symbols) == b"ArXXXN"
    assert encoded.start.tolist() == [0, 1, 2, 2, 2, 4]
    assert (encoded.n_letters, encoded.n_masked, encoded.n_barriers) == (3, 1, 1)


def test_suggest_table_frequent_letters_get_single_groups() -> None:
    counts = {letter: 1000 - 10 * k for k, letter in enumerate("ieutsanromcdplbqgfxzk")}
    counts["k"] = 1
    table = suggest_table(counts).mapping
    assert [table[c] for c in "ieutsanro"] == list(SINGLE_SYMBOLS)
    shared = {}
    for letter, symbol in table.items():
        if symbol not in SINGLE_SYMBOLS:
            shared.setdefault(GROUP[symbol], []).append(letter)
    assert all(len(group) == 2 for group in shared.values())
    assert sorted(shared.values(), key=sorted)[0] is not None
    assert ["k", "m"] in [sorted(g) for g in shared.values()]  # the rarest letter shares with the most frequent
    assert "J" not in table.values() and "Z" not in table.values()  # ambiguity codes left unused


def test_suggest_table_with_20_letters_leaves_the_most_frequent_shared_letter_alone() -> None:
    counts = {letter: 1000 - 10 * k for k, letter in enumerate("ieutsanromcdplbqgfxz")}
    table = suggest_table(counts).mapping
    groups: dict[int, list[str]] = {}
    for letter, symbol in table.items():
        if symbol not in SINGLE_SYMBOLS:
            groups.setdefault(GROUP[symbol], []).append(letter)
    assert sorted(sorted(g) for g in groups.values()) == sorted(
        [["m"], ["c", "z"], ["d", "x"], ["f", "p"], ["g", "l"], ["b", "q"]]
    )
    assert not {"B", "J", "Z"} & set(table.values())


def test_suggest_table_with_22_letters_uses_one_triple() -> None:
    counts = {letter: 1000 - 10 * k for k, letter in enumerate("ieutasnrocmdplbqgfhxzk")}
    table = suggest_table(counts).mapping
    assert len(table) == 22 and len(set(table.values())) == 22
