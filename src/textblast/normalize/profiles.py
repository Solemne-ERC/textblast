"""Normalization profiles: ordered steps plus the letter → symbol table.

Tables are fixed per profile, not computed per corpus, so results do not depend on the corpus under
``-comp_based_stats 2``. Suggest a table for another corpus with ``textblast alphabet suggest``. ``latin-broad``
uses the letter order of the established text-reuse normalization: under the default ``-comp_based_stats 0`` the
table barely matters (99.98% identical hits against a frequency-based table on a Latin corpus), so the established
order is preferred. The frequency-based table stays available as ``latin-broad-frozen``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..encode import EncodingTable
from .steps import COMPAT_TABLES


@dataclass(frozen=True)
class Profile:
    name: str
    steps: tuple[Any, ...]  # step names, or {name, params} specs
    table: Mapping[str, str] | None
    description: str
    allow_collisions: bool = False

    def encoding_table(self) -> EncodingTable:
        if self.table is None:
            raise ValueError(f"profile {self.name!r} has no frozen table yet (run `textblast alphabet suggest`)")
        table = EncodingTable(self.name, dict(self.table), self.allow_collisions)
        table.validate()
        return table


LATIN_STEPS = ("spans", "graphemes", "fold_jv", "fold_wy", "fold_kc")
LATIN_BROAD_STEPS = (*LATIN_STEPS, "delete_h", "vowel_pairs", "ci_vowel")

# Frozen from `textblast alphabet suggest --profile latin-broad` on a corpus of early modern Latin texts
# (2026-10-23, with k → c). Letters: a b c d e f g i l m n o p q r s t u x z (20).
# The 9 most frequent letters (i e u t s a n r o, 70% of all letters) have single-letter seed groups; m, the most
# frequent of the rest, has the D/B group to itself; the others pair frequent with rare (c/z, d/x, p/f, l/g, b/q),
# which is the optimum for this corpus (checked exhaustively). The ambiguity codes B, J, Z stay unused.
LATIN_BROAD_FROZEN_TABLE: dict[str, str] | None = {
    "i": "A", "e": "G", "u": "N", "t": "F", "s": "P", "a": "Y", "n": "H", "r": "C", "o": "W",
    "m": "D", "c": "S", "z": "T", "d": "I", "x": "V", "p": "L", "f": "M", "l": "K", "g": "R", "b": "E", "q": "Q",
}

# Same procedure for the non-broad profile (21 letters: h kept): pairs c/z, m/x, d/h, p/f, l/g, b/q; J and Z unused.
LATIN_TABLE: dict[str, str] | None = {
    "i": "A", "e": "G", "u": "N", "t": "F", "a": "P", "s": "Y", "n": "H", "r": "C", "o": "W",
    "c": "S", "z": "T", "m": "I", "x": "V", "d": "L", "h": "M", "p": "K", "f": "R", "l": "E", "g": "Q", "b": "D",
    "q": "B",
}

# The letter encoding of the established text-reuse normalization, restricted to the 20 letters of latin-broad
# (w, y and k are folded away): latin-broad's table. It has no h, so the non-broad profile keeps its frozen table.
_REFERENCE_ENCODING = dict(zip("abcdefgiklmnopqrstuwxyz", "ARNDCQEGHILKMFPSTWYVBJZ", strict=True))
LATIN_BROAD_REFTABLE = {k: v for k, v in _REFERENCE_ENCODING.items() if k not in "wyk"}
LATIN_BROAD_TABLE = LATIN_BROAD_REFTABLE

# textreuse-blast never turned declared gaps into barriers; explicit exclusions and pattern rules still apply.
_COMPAT_SPANS = {"name": "spans", "params": {"gaps": False}}
COMPAT_ENG_TABLE = COMPAT_TABLES["eng"]
COMPAT_FIN_TABLE = COMPAT_TABLES["fin"]

PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile(
            "latin-broad",
            LATIN_BROAD_STEPS,
            LATIN_BROAD_TABLE,
            "Aggressive Latin: shared rules + h deleted, ae/oe → e, c → t before i + vowel (20 letters).",
        ),
        Profile(
            "latin-broad-frozen",
            LATIN_BROAD_STEPS,
            LATIN_BROAD_FROZEN_TABLE,
            "latin-broad with the earlier frequency-based table (reproduces runs made with it).",
        ),
        Profile(
            "latin-broad-reftable",
            LATIN_BROAD_STEPS,
            LATIN_BROAD_REFTABLE,
            "The same as latin-broad (an earlier name, kept so that older settings still run).",
        ),
        Profile(
            "latin",
            LATIN_STEPS,
            LATIN_TABLE,
            "Latin, non-broad: shared rules (incl. w → u, y → i, k → c), like latin-broad (21 letters, h kept).",
        ),
        Profile("compat-eng", (_COMPAT_SPANS, "compat_eng"), COMPAT_ENG_TABLE,
                "textreuse-blast ENG encoding, byte for byte (parity tests only).", allow_collisions=True),
        Profile("compat-fin", (_COMPAT_SPANS, "compat_fin"), COMPAT_FIN_TABLE,
                "textreuse-blast FIN encoding, byte for byte (parity tests only).", allow_collisions=True),
    )
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r} (available: {', '.join(PROFILES)})") from None
