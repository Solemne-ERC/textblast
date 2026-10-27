"""Alignments rebuilt from BLAST's ``btop``.

``btop`` is written from the query's side: a number is a run of identical letters, a two-letter token is one column
(query letter, subject letter), with ``-`` for a gap. Verified against BLAST's ``qseq``/``sseq`` on 2,727 sample hits
: the tokens rebuild both aligned strings exactly, and the numbers sum to ``nident``.

An alignment is handled as columns: whether each column consumes a letter of side 1 and of side 2, and whether it is
an identity. Columns are always oriented ``doc1``/``doc2`` (``query_side`` tells how to swap).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"(\d+)|(..)")


@dataclass
class Columns:
    """One alignment (or a part of it) as columns."""

    adv1: np.ndarray  # bool: the column consumes a letter of side 1
    adv2: np.ndarray  # bool: the column consumes a letter of side 2
    ident: np.ndarray  # bool: identical letters

    @property
    def length(self) -> int:
        return len(self.ident)

    def part(self, a: int, b: int) -> Columns:
        return Columns(self.adv1[a:b], self.adv2[a:b], self.ident[a:b])


def parse_btop(btop: str, query_side: int = 1) -> Columns:
    """Columns of one alignment; ``query_side`` = 2 swaps the sides so that side 1 is always ``doc1``."""
    tokens = _TOKEN.findall(btop)
    counts = np.array([int(number) if number else 1 for number, _ in tokens], dtype=np.int64)
    is_run = np.array([bool(number) for number, _ in tokens], dtype=bool)
    q = np.array([bool(number) or pair[0] != "-" for number, pair in tokens], dtype=bool)
    s = np.array([bool(number) or pair[1] != "-" for number, pair in tokens], dtype=bool)
    adv_q, adv_s, ident = np.repeat(q, counts), np.repeat(s, counts), np.repeat(is_run, counts)
    if query_side == 2:
        adv_q, adv_s = adv_s, adv_q
    return Columns(adv_q, adv_s, ident)


def column_stats(cols: Columns) -> dict[str, int]:
    """``length``, ``nident``, ``mismatch``, ``gaps`` and ``gapopen`` (runs of gap columns on one side)."""
    kind = np.where(cols.adv1 & ~cols.adv2, 1, 0) + np.where(cols.adv2 & ~cols.adv1, 2, 0)  # 1/2: gap on side 2/1
    previous = np.concatenate([[0], kind[:-1]]) if len(kind) else kind
    return {
        "length": cols.length,
        "nident": int(cols.ident.sum()),
        "mismatch": int((cols.adv1 & cols.adv2 & ~cols.ident).sum()),
        "gaps": int((kind != 0).sum()),
        "gapopen": int(((kind != 0) & (kind != previous)).sum()),
    }


def segment_starts(cols: Columns, zone1: np.ndarray, zone2: np.ndarray) -> list[int]:
    """Column indices where a new segment starts, i.e. where a side's next letter lies in another zone (unit).

    ``zone1``/``zone2`` give, for each letter of the side's aligned range in order, a zone number. The segments tile
    the alignment exactly; a segment may hold letters on one side only (gap columns across a boundary), which callers
    usually drop, since such a segment aligns nothing.
    """
    n = cols.length
    change = np.zeros(n, dtype=bool)
    for adv, zone in ((cols.adv1, zone1), (cols.adv2, zone2)):
        if len(zone) < 2:
            continue
        moved = np.concatenate([[False], zone[1:] != zone[:-1]])  # letter k starts a new zone
        columns = np.flatnonzero(adv)  # the column of each letter of this side
        change[columns[moved]] = True
    change[:1] = False
    return [0, *(np.flatnonzero(change)).tolist()]


def letter_bounds(map_start: np.ndarray, map_end: np.ndarray, enc_start: int, enc_end: int) -> np.ndarray:
    """Surface starts of letters ``enc_start .. enc_end - 1`` plus the surface end of the last one."""
    starts = np.asarray(map_start[enc_start:enc_end], dtype=np.int64)
    return np.append(starts, int(map_end[enc_end - 1]))


def surface_strings(cols: Columns, text1: str, bounds1: np.ndarray, text2: str, bounds2: np.ndarray) -> tuple[str, str]:
    """passim-style aligned surface strings ``s1`` and ``s2``.

    ``bounds1``/``bounds2`` come from :func:`letter_bounds`. Each letter's chunk is the text from its start to the
    next letter's start (the last one to its own end), so characters between letters stay with their side; in each
    column the shorter chunk is padded with ``-``. Removing the padding gives back each side's surface span exactly.
    """
    chunks1 = [text1[bounds1[i] : bounds1[i + 1]] for i in range(len(bounds1) - 1)]
    chunks2 = [text2[bounds2[i] : bounds2[i + 1]] for i in range(len(bounds2) - 1)]
    out1: list[str] = []
    out2: list[str] = []
    i1 = i2 = 0
    for a1, a2 in zip(cols.adv1.tolist(), cols.adv2.tolist(), strict=True):
        c1 = chunks1[i1] if a1 else ""
        c2 = chunks2[i2] if a2 else ""
        i1 += a1
        i2 += a2
        width = max(len(c1), len(c2))
        out1.append(c1.ljust(width, "-"))
        out2.append(c2.ljust(width, "-"))
    return "".join(out1), "".join(out2)
