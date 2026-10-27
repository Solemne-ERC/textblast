"""Encoding normalized letters as BLAST protein symbols (plan section 6, notes Q10 and findings 8, 14).

Letters map one-to-one to the 23 symbols that score +9 against themselves in the patched matrix. ``X`` is never used
for a letter (``X/X`` scores −5); it only builds barriers. Masked letters become lowercase symbols (BLAST runs with
``-lcase_masking -soft_masking true``). Only the textreuse-blast compatibility tables may break these rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .engine.check import BARRIER_LENGTH, SEED_GROUPS
from .normalize.spans import BARRIER, SEP, SILENT_SEP, Spanned, to_codes

SYMBOLS = "ARNDCQEGHILKMFPSTWYVBJZ"

# The 9 single-letter seed groups, ordered by the real BLOSUM62 self-score (A 4 … W 11). Composition-based statistics
# re-scores with the real BLOSUM62, where rare residues score higher, so frequent letters go to low self-scores.
SINGLE_SYMBOLS = ("A", "G", "N", "F", "P", "Y", "H", "C", "W")
# Shared seed groups (letters in one group seed together); within a group the symbols are in order of preference:
# standard amino acids first, ambiguity codes (J, Z, B) last.
SHARED_GROUPS = (("S", "T"), ("I", "V", "J"), ("L", "M"), ("K", "R"), ("E", "Q", "Z"), ("D", "B"))

assert sorted("".join(SINGLE_SYMBOLS) + "".join("".join(g) for g in SHARED_GROUPS)) == sorted(SYMBOLS)
assert {frozenset(g) for g in SHARED_GROUPS} | {frozenset(s) for s in SINGLE_SYMBOLS} == {
    frozenset(g) for g in SEED_GROUPS
}


class EncodingError(ValueError):
    """An invalid table, or a letter the table cannot encode."""


@dataclass(frozen=True)
class EncodingTable:
    name: str
    mapping: Mapping[str, str]  # letter -> symbol
    allow_collisions: bool = False  # only for the textreuse-blast compatibility tables

    def validate(self) -> None:
        letters, symbols = list(self.mapping), list(self.mapping.values())
        bad = [s for s in symbols if len(s) != 1 or (s not in SYMBOLS and s != "X")]
        if bad:
            raise EncodingError(f"table {self.name}: invalid symbols {bad}")
        if self.allow_collisions:
            return
        if "X" in symbols:
            letter = letters[symbols.index("X")]
            raise EncodingError(
                f"table {self.name}: {letter!r} → X; X never matches itself and is reserved for barriers"
            )
        if len(letters) > len(SYMBOLS):
            raise EncodingError(
                f"table {self.name}: {len(letters)} letters but only {len(SYMBOLS)} usable symbols; "
                "fold letters together in the normalization profile"
            )
        seen: dict[str, str] = {}
        for letter, symbol in self.mapping.items():
            if symbol in seen:
                raise EncodingError(
                    f"table {self.name}: {seen[symbol]!r} and {letter!r} both map to {symbol}; collisions are not "
                    "allowed (merge letters explicitly in the normalization profile instead)"
                )
            seen[symbol] = letter


@dataclass
class EncodedDocument:
    symbols: np.ndarray  # uint8 ASCII: symbols, lowercase if masked, X runs for barriers
    start: np.ndarray  # uint32 surface start per symbol
    end: np.ndarray  # uint32 surface end (exclusive) per symbol
    n_letters: int
    n_masked: int
    n_barriers: int


def encode_document(doc: Spanned, table: EncodingTable, barrier_length: int = BARRIER_LENGTH) -> EncodedDocument:
    """Letters → symbols, separators dropped, each barrier → ``barrier_length`` × ``X`` (all mapped to its span)."""
    codes = to_codes(doc.text)
    if not len(codes):
        empty = np.zeros(0, dtype=np.uint32)
        return EncodedDocument(np.zeros(0, dtype=np.uint8), empty, empty, 0, 0, 0)
    unique, inverse = np.unique(codes, return_inverse=True)
    symbol_u = np.zeros(len(unique), dtype=np.uint8)
    length_u = np.zeros(len(unique), dtype=np.int64)
    is_letter_u = np.zeros(len(unique), dtype=bool)
    for i, code in enumerate(unique.tolist()):
        ch = chr(code)
        if ch in table.mapping:
            symbol_u[i], length_u[i], is_letter_u[i] = ord(table.mapping[ch]), 1, True
        elif ch == BARRIER:
            symbol_u[i], length_u[i] = ord("X"), barrier_length
        elif ch in (SEP, SILENT_SEP):
            continue
        else:
            raise EncodingError(
                f"table {table.name} has no symbol for {ch!r} (U+{code:04X}); add a normalization rule or a table entry"
            )
    symbols = symbol_u[inverse]
    is_letter = is_letter_u[inverse]
    masked = is_letter & doc.mask
    symbols = np.where(masked, symbols + 32, symbols).astype(np.uint8)  # lowercase = masked
    lengths = length_u[inverse]
    return EncodedDocument(
        symbols=np.repeat(symbols, lengths),
        start=np.repeat(doc.start, lengths).astype(np.uint32),
        end=np.repeat(doc.end, lengths).astype(np.uint32),
        n_letters=int(is_letter.sum()),
        n_masked=int(masked.sum()),
        n_barriers=int((codes == ord(BARRIER)).sum()),
    )


def suggest_table(counts: Mapping[str, int], name: str = "suggested") -> EncodingTable:
    """Frequent letters get single-letter seed groups; the rest share groups, frequent with rare (notes Q10).

    The cost of a shared group is the chance that two different letters of it meet in a seed (the product of their
    frequencies). With room to spare, the most frequent remaining letters keep a group to themselves; the others are
    paired largest with smallest, which minimizes the summed products (rearrangement inequality).
    """
    ranking = sorted(counts, key=lambda letter: (-counts[letter], letter))
    if len(ranking) > len(SYMBOLS):
        raise EncodingError(f"{len(ranking)} letters but only {len(SYMBOLS)} symbols")
    mapping = dict(zip(ranking, SINGLE_SYMBOLS, strict=False))
    rest = ranking[len(SINGLE_SYMBOLS) :]
    n_groups = len(SHARED_GROUPS)
    groups: list[list[str]] = [[] for _ in SHARED_GROUPS]
    if len(rest) <= 2 * n_groups:
        n_alone = n_groups if len(rest) <= n_groups else 2 * n_groups - len(rest)
        alone, paired = rest[:n_alone], rest[n_alone:]
        lists = [[letter] for letter in alone] + [
            [paired[i], paired[-1 - i]] for i in range(len(paired) // 2)
        ]
        # A letter alone takes a group whose second symbol is an ambiguity code first (D/B, then I/J/V, E/Q/Z), so B,
        # J and Z stay unused; pairs take the remaining groups.
        order_alone = [5, 1, 4, 0, 2, 3]
        free = list(range(n_groups))
        for letters in lists[:n_alone]:
            k = next(k for k in order_alone if k in free)
            groups[k] = letters
            free.remove(k)
        for letters, k in zip(lists[n_alone:], free, strict=False):
            groups[k] = letters
    else:  # more than 12 shared letters: one pair per group, the rarest extras into the least loaded triple group
        heads, tails = rest[:n_groups], rest[n_groups:]
        for group, letter in zip(groups, heads, strict=True):
            group.append(letter)
        load = [counts[g[0]] for g in groups]
        for letter in sorted(tails, key=lambda letter: (counts[letter], letter)):  # rarest first, to the busiest
            candidates = [k for k, g in enumerate(groups) if len(g) < len(SHARED_GROUPS[k])]
            k = max(candidates, key=lambda k: (1, load[k], -k) if len(groups[k]) < 2 else (0, -load[k], -k))
            groups[k].append(letter)
            load[k] += counts[letter]
    for group, symbols in zip(groups, SHARED_GROUPS, strict=True):
        mapping.update(zip(group, symbols, strict=False))
    table = EncodingTable(name, mapping)
    table.validate()
    return table
