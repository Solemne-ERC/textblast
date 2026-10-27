"""Property tests on random Unicode text: spans are valid, ordered, never split a letter from its marks, and every
encoded letter normalizes back from its own surface span."""

from __future__ import annotations

import random
import unicodedata

import pytest

from textblast.encode import encode_document
from textblast.normalize import DocContext, Normalizer
from textblast.normalize.profiles import get_profile

from .helpers import letters

ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("áàâäéèêëíïóòôöúùûüçñōęæœſÆŒßĳﬁł")
    + ["́", "̀", "̈", "̃", "̄", "̧"]  # combining marks (supported and not)
    + list(" \t\n.,;:!?()[]«»—-·0123456789&⁊­")
    + list("αβγλόγος𓀀𝔞")
    + ["[...]", "[lacuna]", "…", "=== PAGE 2 ==="]
)


def random_text(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 60)))


def context_free(profile: str, text: str) -> str:
    return letters(profile, text).replace(" ", "")


@pytest.mark.parametrize("profile", ["latin", "latin-broad"])
def test_random_text_round_trips(profile: str) -> None:
    rng = random.Random(1234)
    table = get_profile(profile).encoding_table()
    inverse = {symbol: letter for letter, symbol in table.mapping.items()}
    normalizer = Normalizer(get_profile(profile).steps)
    for _ in range(400):
        text = random_text(rng)
        ctx = DocContext(0, "d", len(text))
        encoded = encode_document(normalizer.run(text, ctx), table, barrier_length=2)
        symbols = bytes(encoded.symbols).decode()
        starts, ends = encoded.start.tolist(), encoded.end.tolist()
        assert starts == sorted(starts) and ends == sorted(ends)
        for symbol, a, b in zip(symbols, starts, ends, strict=True):
            assert 0 <= a < b <= len(text), (text, symbol, a, b)
            if symbol == "X":
                continue  # barrier: the exact span of a declared gap (checked in the rule tests)
            # a letter's span never starts on, or stops before, a combining mark: letters keep their marks
            assert not unicodedata.combining(text[a]), (text, a)
            assert b == len(text) or not unicodedata.combining(text[b]), (text, b)
            letter = inverse[symbol.upper()]
            expected = context_free(profile, text[a:b])
            # c → t before i + vowel is the only context-dependent rule
            assert letter in expected or (letter == "t" and "c" in expected), (text, text[a:b], letter, expected)
