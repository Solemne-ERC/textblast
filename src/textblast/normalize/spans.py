"""Text with an exact span map back to the surface text (plan section 6, notes Q12–Q16).

Every character of the current text knows the span ``[start, end)`` of the *original* input text it comes from (0-based,
end-exclusive Unicode code points). Steps always map back to the surface directly, so span maps compose automatically:
1:n expansions repeat a span, n:1 contractions merge spans, deletions drop them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# Internal markers. Both are private-use code points that never survive encoding as letters.
BARRIER = ""  # a barrier: expanded to a run of X by the encoder
SEP = " "  # a word separator (whitespace, punctuation, deleted text); dropped by the encoder
SILENT_SEP = ""  # a separator whose surface characters are already counted in the statistics


@dataclass
class Spanned:
    """Current text plus, per character, its surface span and whether it is masked."""

    text: str
    start: np.ndarray  # int64, one entry per character of ``text``
    end: np.ndarray  # int64, exclusive
    mask: np.ndarray  # bool

    @classmethod
    def from_surface(cls, text: str) -> Spanned:
        n = len(text)
        start = np.arange(n, dtype=np.int64)
        return cls(text, start, start + 1, np.zeros(n, dtype=bool))

    def __len__(self) -> int:
        return len(self.text)

    def is_identity(self, surface_length: int) -> bool:
        """True while no step has changed the text (the spans are the surface positions themselves)."""
        return len(self.text) == surface_length and bool(
            np.array_equal(self.start, np.arange(surface_length, dtype=np.int64))
        )

    def validate(self) -> None:
        n = len(self.text)
        if not (len(self.start) == len(self.end) == len(self.mask) == n):
            raise AssertionError("span arrays out of sync with text")

    # ------------------------------------------------------------------ vectorized edits

    def keep(self, keep: np.ndarray) -> Spanned:
        """Drop the characters where ``keep`` is False."""
        if bool(keep.all()):
            return self
        codes = to_codes(self.text)[keep]
        return Spanned(from_codes(codes), self.start[keep], self.end[keep], self.mask[keep])

    def translate_same_length(self, table: dict[int, int]) -> Spanned:
        """1:1 substitutions (``table`` maps code point to code point): spans are unchanged."""
        new = self.text.translate(table)
        if len(new) != len(self.text):  # pragma: no cover - guarded by callers
            raise ValueError("translate_same_length needs a 1:1 table")
        return Spanned(new, self.start, self.end, self.mask)

    def merge_pairs(self, first: np.ndarray, replacement: str) -> Spanned:
        """Replace each pair ``text[i:i+2]`` (``i`` in ``first``, non-overlapping) by ``replacement`` (one char)."""
        if len(first) == 0:
            return self
        codes = to_codes(self.text).copy()
        end = self.end.copy()
        mask = self.mask.copy()
        codes[first] = ord(replacement)
        end[first] = self.end[first + 1]
        mask[first] |= self.mask[first + 1]
        keep = np.ones(len(codes), dtype=bool)
        keep[first + 1] = False
        return Spanned(from_codes(codes[keep]), self.start[keep], end[keep], mask[keep])

    def splice(self, edits: Iterable[tuple[int, int, str]]) -> Spanned:
        """General edits ``(i, j, replacement)`` on current positions, sorted and non-overlapping.

        If the replacement has the same length as ``text[i:j]`` the spans are kept position by position; otherwise every
        replacement character gets the merged span ``[start[i], end[j-1])``; a pure insertion (``i == j``) gets a
        zero-width span at the insertion point.
        """
        pieces_text: list[str] = []
        pieces_start: list[np.ndarray] = []
        pieces_end: list[np.ndarray] = []
        pieces_mask: list[np.ndarray] = []
        last = 0
        for i, j, replacement in edits:
            if i < last or j < i:
                raise ValueError("edits must be sorted and non-overlapping")
            pieces_text.append(self.text[last:i])
            pieces_start.append(self.start[last:i])
            pieces_end.append(self.end[last:i])
            pieces_mask.append(self.mask[last:i])
            k = len(replacement)
            if k == j - i:
                pieces_start.append(self.start[i:j])
                pieces_end.append(self.end[i:j])
                pieces_mask.append(self.mask[i:j])
            elif k:
                if j > i:
                    a, b, m = self.start[i], self.end[j - 1], bool(self.mask[i:j].any())
                else:  # pure insertion: zero-width span at the insertion point
                    a = self.start[i] if i < len(self.text) else (self.end[i - 1] if i else 0)
                    b, m = a, False
                pieces_start.append(np.full(k, a, dtype=np.int64))
                pieces_end.append(np.full(k, b, dtype=np.int64))
                pieces_mask.append(np.full(k, m, dtype=bool))
            pieces_text.append(replacement)
            last = j
        pieces_text.append(self.text[last:])
        pieces_start.append(self.start[last:])
        pieces_end.append(self.end[last:])
        pieces_mask.append(self.mask[last:])
        return Spanned(
            "".join(pieces_text),
            np.concatenate(pieces_start) if pieces_start else self.start[:0],
            np.concatenate(pieces_end) if pieces_end else self.end[:0],
            np.concatenate(pieces_mask) if pieces_mask else self.mask[:0],
        )


def to_codes(text: str) -> np.ndarray:
    """Code points of ``text`` as uint32 (one entry per character)."""
    return np.frombuffer(text.encode("utf-32-le"), dtype="<u4")


def from_codes(codes: np.ndarray) -> str:
    return np.ascontiguousarray(codes, dtype="<u4").tobytes().decode("utf-32-le")
