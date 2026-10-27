"""Normalization steps (plan section 6).

A step turns a :class:`Spanned` text into another one; spans always point back to the surface text. Built-in steps are
registered by name; custom steps are loaded by import path (``class_path`` + ``init_args``) and must subclass
:class:`Step`.
"""

from __future__ import annotations

import importlib
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spans import BARRIER, SEP, SILENT_SEP, Spanned, from_codes, to_codes

MODES = ("delete", "mask", "barrier")


class NormalizationError(ValueError):
    """A step was misconfigured or misplaced."""


@dataclass(frozen=True)
class Exclusion:
    """A classified span of the surface text (from the ``exclusions`` field, a file, a pattern rule or a gap)."""

    start: int
    end: int
    mode: str  # delete | mask | barrier
    reason: str
    source: str  # field | file | pattern | gap


@dataclass(frozen=True)
class PatternRule:
    """Regex → action → reason, optionally only for documents whose metadata match ``where``."""

    pattern: str
    action: str
    reason: str
    where: Mapping[str, Any] | None = None
    ignore_case: bool = False

    def applies_to(self, metadata: Mapping[str, Any]) -> bool:
        if not self.where:
            return True
        for key, wanted in self.where.items():
            value = metadata.get(key)
            if isinstance(wanted, (list, tuple, set)):
                if value not in wanted:
                    return False
            elif value != wanted:
                return False
        return True


@dataclass
class DocContext:
    """What a step may need to know about the document, and where it records what it did."""

    doc_idx: int
    doc_id: str
    surface_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    exclusions: list[Exclusion] = field(default_factory=list)
    deleted: Counter[str] = field(default_factory=Counter)  # surface characters without an image, by reason
    unsupported: Counter[str] = field(default_factory=Counter)  # unsupported characters/graphemes and their counts
    applied: list[Exclusion] = field(default_factory=list)  # classified spans that were applied
    masked_chars: Counter[str] = field(default_factory=Counter)  # masked surface characters, by reason


class Step:
    """Base class for normalization steps."""

    name = "step"

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- classified spans

# Declared gaps and export scaffolding (from the user's normalization material).
GAP_PATTERN = re.compile(
    r"\[(?:\s*\.{2,}\s*|[^\]]*\b(?:lacuna|omitted|illegible)\b[^\]]*)\]|\.{3,}|…", re.IGNORECASE
)
SCAFFOLD_PATTERN = re.compile(r"\[(?:CANVAS|COLUMN|REGION|NO TEXT DETECTED)\b[^\]]*\]|===\s*(?:END )?PAGE\b.*?===")

_KEEP, _MASK, _DELETE, _BARRIER = 0, 1, 2, 3
_MODE_CODE = {"mask": _MASK, "delete": _DELETE, "barrier": _BARRIER}


class SpanRules(Step):
    """Applies classified spans to the surface text: declared gaps, pattern rules, then explicit exclusions.

    Precedence (later wins where spans overlap): gaps < pattern rules (earlier rules win over later ones) < exclusions
    from the input. ``delete`` removes text (it becomes a word separator), ``mask`` marks letters so BLAST starts no
    seed there, ``barrier`` replaces the span by one barrier (an ``X`` run after encoding). Must be the first step.
    """

    name = "spans"

    def __init__(self, pattern_rules: Iterable[PatternRule] = (), gaps: bool = True) -> None:
        self.rules = [r if isinstance(r, PatternRule) else PatternRule(**r) for r in pattern_rules]
        for rule in self.rules:
            if rule.action not in MODES:
                raise NormalizationError(f"pattern rule {rule.pattern!r}: action must be one of {MODES}")
        self.compiled = [re.compile(r.pattern, re.IGNORECASE if r.ignore_case else 0) for r in self.rules]
        self.gaps = gaps

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        if not doc.is_identity(ctx.surface_length):
            raise NormalizationError("the 'spans' step must be the first step (it works on surface positions)")
        text = doc.text
        spans: list[Exclusion] = []
        if self.gaps:
            for pattern, reason in ((GAP_PATTERN, "declared_gap"), (SCAFFOLD_PATTERN, "export_scaffolding")):
                spans += [Exclusion(m.start(), m.end(), "barrier", reason, "gap") for m in pattern.finditer(text)]
        for rule, compiled in reversed(list(zip(self.rules, self.compiled, strict=True))):
            if rule.applies_to(ctx.metadata):
                spans += [
                    Exclusion(m.start(), m.end(), rule.action, rule.reason, "pattern")
                    for m in compiled.finditer(text)
                    if m.end() > m.start()
                ]
        spans += sorted(ctx.exclusions, key=lambda e: (e.start, e.end))
        if not spans:
            return doc

        n = len(text)
        action = np.zeros(n, dtype=np.int8)
        reason_ids = np.full(n, -1, dtype=np.int32)
        reasons: list[str] = []
        for span in spans:  # in precedence order: later spans overwrite earlier ones
            if span.reason not in reasons:
                reasons.append(span.reason)
            action[span.start : span.end] = _MODE_CODE[span.mode]
            reason_ids[span.start : span.end] = reasons.index(span.reason)
        ctx.applied.extend(spans)

        for code, counter in ((_DELETE, ctx.deleted), (_BARRIER, ctx.deleted), (_MASK, ctx.masked_chars)):
            ids, counts = np.unique(reason_ids[action == code], return_counts=True)
            for rid, count in zip(ids.tolist(), counts.tolist(), strict=True):
                counter[reasons[rid]] += count

        mask = doc.mask | (action == _MASK)
        codes = to_codes(text).copy()
        codes[action == _DELETE] = ord(SILENT_SEP)
        result = Spanned(from_codes(codes), doc.start, doc.end, mask)

        barrier = action == _BARRIER
        if barrier.any():
            edges = np.flatnonzero(np.diff(np.concatenate(([0], barrier.view(np.int8), [0]))))
            result = result.splice((int(a), int(b), BARRIER) for a, b in zip(edges[::2], edges[1::2], strict=True))
        return result


# --------------------------------------------------------------------------- graphemes

# Combining marks that are plain accents (stripped). Others (tilde, macron, overline, ...) usually mark abbreviations
# in medieval texts: the grapheme is removed and counted as unresolved_abbreviation_mark (normalization.md).
SUPPORTED_MARKS = frozenset(
    chr(c) for c in (0x0300, 0x0301, 0x0302, 0x0306, 0x0307, 0x0308, 0x030A, 0x030B, 0x030C, 0x0327, 0x0328)
)
# Letters whose meaning is certain but which Unicode decomposition does not resolve. ``ł`` attached to a word is an
# abbreviation stroke on l (concił → concil); a standalone ``ł`` means *vel* (handled in context, see Graphemes).
LETTER_SPECIALS = {"ę": "ae", "Ę": "ae", "ı": "i", "ł": "l", "Ł": "l"}
# Signs with a fixed meaning in this material (round 5): ¬ is a line-break hyphen inside a word, ˙ is punctuation.
# < and > enclose editorial supplements: punctuation by default, removed silently inside a word (see Graphemes).
SIGN_SPECIALS = {"¬": ("silent", "line_break_hyphen"), "˙": ("sep", "punctuation"), "<": ("sep", "punctuation"),
                 ">": ("sep", "punctuation")}
CONTEXTUAL = frozenset("<>łŁ")

_LETTERS, _SEP, _DELETE_SEP, _DELETE_SILENT, _COMBINING, _MARKER = range(6)


def classify_char(ch: str, char_map: Mapping[str, str]) -> tuple[int, str, str | None]:
    """(kind, output, reason) for a single character, independent of context."""
    if ch in char_map:
        out = char_map[ch]
        return (_LETTERS, out, None) if out else (_DELETE_SILENT, "", "char_map")
    if ch == BARRIER:
        return _MARKER, BARRIER, None
    if ch == SILENT_SEP:
        return _MARKER, SEP, None
    if ch == "­":
        return _DELETE_SILENT, "", "soft_hyphen"
    if ch in "&⁊":
        return _LETTERS, " et ", None
    if ch in SIGN_SPECIALS:
        kind, reason = SIGN_SPECIALS[ch]
        return (_DELETE_SILENT, "", reason) if kind == "silent" else (_SEP, SEP, reason)
    category = unicodedata.category(ch)
    if category in ("Mn", "Me", "Mc"):
        return _COMBINING, "", None
    if category in ("Nd", "No", "Nl"):
        return _DELETE_SEP, SEP, "numeral"
    if category.startswith("Z") or category == "Cc":
        return _SEP, SEP, "whitespace"
    if category.startswith("P"):
        return _SEP, SEP, "punctuation"
    if category == "Cf":
        return _DELETE_SILENT, "", "format_character"
    if category.startswith("L"):
        folded = LETTER_SPECIALS.get(ch) or ch.casefold()
        folded = unicodedata.normalize("NFKD", folded)
        marks = [c for c in folded if unicodedata.combining(c)]
        letters = "".join(c for c in folded if not unicodedata.combining(c))
        letters = letters.replace("æ", "ae").replace("œ", "oe").replace("ſ", "s")
        if not letters or not all("a" <= c <= "z" for c in letters):
            return _DELETE_SEP, SEP, "unsupported_letter"  # e.g. Greek, ł, MUFI characters
        if any(m not in SUPPORTED_MARKS for m in marks):
            return _DELETE_SEP, SEP, "unresolved_abbreviation_mark"
        return _LETTERS, letters, None
    return _DELETE_SEP, SEP, "unsupported_sign"


class Graphemes(Step):
    """Case folding, Unicode decomposition and the shared character rules of ``normalization.md``.

    - letters: casefold, NFKD, plain accents stripped; ``æ → ae``, ``œ → oe``, ``ſ → s``, ``ę → ae``, ligatures split;
    - a letter carrying an unresolved mark (tilde, macron, ...) is removed with its marks
      (``unresolved_abbreviation_mark``);
    - ``&`` and ``⁊`` become the word ``et``; a standalone ``ł`` becomes ``uel`` (*vel*), an attached one ``l``;
    - whitespace, punctuation and numerals become word separators; soft hyphens and ``¬`` (line-break hyphen)
      disappear; ``<`` and ``>`` (editorial supplements) disappear inside a word (``do<mi>nus`` → ``dominus``) and
      separate words elsewhere; other signs and letters outside a–z are removed (``unsupported_sign`` /
      ``unsupported_letter``) and listed in the report.

    ``char_map`` adds or overrides single-character mappings (e.g. ``{"ł": "l"}``; an empty string deletes).
    """

    name = "graphemes"

    def __init__(self, char_map: Mapping[str, str] | None = None) -> None:
        self.char_map = dict(char_map or {})
        for key, value in self.char_map.items():
            if len(key) != 1 or not all("a" <= c <= "z" for c in value):
                raise NormalizationError(f"graphemes.char_map: {key!r} → {value!r} must map one character to a–z")
        self._cache: dict[int, tuple[int, str, str | None]] = {}
        self._contextual_codes = np.array(sorted(ord(c) for c in CONTEXTUAL if c not in self.char_map),
                                          dtype=np.uint32)

    def _classes(self, unique: np.ndarray) -> tuple[np.ndarray, list[str], list[str | None]]:
        kinds = np.empty(len(unique), dtype=np.int8)
        outputs: list[str] = []
        reasons: list[str | None] = []
        for i, code in enumerate(unique.tolist()):
            entry = self._cache.get(code)
            if entry is None:
                entry = self._cache[code] = classify_char(chr(code), self.char_map)
            kinds[i] = entry[0]
            outputs.append(entry[1])
            reasons.append(entry[2])
        return kinds, outputs, reasons

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        if not len(doc):
            return doc
        codes = to_codes(doc.text)
        unique, inverse = np.unique(codes, return_inverse=True)
        kinds_u, outputs_u, reasons_u = self._classes(unique)
        kinds = kinds_u[inverse]
        lengths = np.array([len(o) for o in outputs_u], dtype=np.int64)[inverse]
        end = doc.end
        override: dict[int, str] = {}  # position -> output for graphemes with combining marks
        extra_reasons: Counter[str] = Counter()

        combining = np.flatnonzero(kinds == _COMBINING)
        if len(combining):
            end = end.copy()
            text = doc.text
            i = 0
            while i < len(combining):
                first = int(combining[i])
                last = first
                while i + 1 < len(combining) and int(combining[i + 1]) == last + 1:
                    i += 1
                    last = int(combining[i])
                i += 1
                base = first - 1
                marks = text[first : last + 1]
                for p in range(first, last + 1):
                    override[p] = ""
                if base >= 0 and kinds[base] == _LETTERS and all(m in SUPPORTED_MARKS for m in marks):
                    end[base] = doc.end[last]  # the accent belongs to its letter
                elif base >= 0 and kinds[base] == _LETTERS:
                    override[base] = SEP
                    grapheme = text[base : last + 1]
                    ctx.unsupported[grapheme] += 1
                    extra_reasons["unresolved_abbreviation_mark"] += last + 1 - base
                else:
                    ctx.unsupported[marks] += 1
                    extra_reasons["unsupported_sign"] += last + 1 - first

        # context-dependent signs: supplements inside a word, standalone ł (rare: a Python loop is fine)
        contextual = np.flatnonzero(np.isin(codes, self._contextual_codes))
        if len(contextual):
            text = doc.text

            def letter_at(i: int) -> bool:
                return 0 <= i < len(text) and (text[i].isalpha() or bool(unicodedata.combining(text[i])))

            for p in contextual.tolist():
                inside = letter_at(p - 1) and letter_at(p + 1)
                if text[p] in "<>" and inside:
                    override[p] = ""
                    extra_reasons["supplement_bracket"] += 1
                elif text[p] in "łŁ" and not letter_at(p - 1) and not letter_at(p + 1):
                    override[p] = "uel"
        for p, out in override.items():
            lengths[p] = len(out)

        # statistics: surface characters without an image, by reason (overridden positions counted above)
        reason_index = {r: k for k, r in enumerate(sorted({r for r in reasons_u if r}))}
        if reason_index:
            ids = np.array([reason_index.get(r, -1) if r else -1 for r in reasons_u], dtype=np.int32)[inverse]
            if override:
                ids[np.fromiter(override, dtype=np.int64)] = -1
            valid = ids >= 0
            # weight = surface characters covered (1 unless an earlier step merged characters)
            counts = np.bincount(ids[valid], weights=(doc.end - doc.start)[valid], minlength=len(reason_index))
            for reason, k in reason_index.items():
                if counts[k]:
                    ctx.deleted[reason] += int(counts[k])
            # list unsupported characters individually
            for reason in ("unsupported_letter", "unsupported_sign", "unresolved_abbreviation_mark"):
                if reason in reason_index:
                    hits = np.flatnonzero(ids == reason_index[reason])
                    if len(hits):
                        chars, counts_c = np.unique(codes[hits], return_counts=True)
                        for c, n in zip(chars.tolist(), counts_c.tolist(), strict=True):
                            ctx.unsupported[chr(c)] += n
        ctx.deleted.update(extra_reasons)

        table = {int(u): o for u, o in zip(unique.tolist(), outputs_u, strict=True)}
        if override:
            # Route overridden positions through private markers so one translate() call builds the whole text.
            codes = codes.copy()
            markers = {out: 0xE010 + k for k, out in enumerate(sorted(set(override.values())))}
            for p, out in override.items():
                codes[p] = markers[out]
            table.update({code: out for out, code in markers.items()})
            source = from_codes(codes)
        else:
            source = doc.text
        new_text = source.translate(table)
        return Spanned(new_text, np.repeat(doc.start, lengths), np.repeat(end, lengths), np.repeat(doc.mask, lengths))


# --------------------------------------------------------------------------- letter-level rules


class Fold(Step):
    """One-to-one letter folds, e.g. ``j → i``, ``v → u`` (spans unchanged)."""

    def __init__(self, table: Mapping[str, str], name: str = "fold") -> None:
        for key, value in table.items():
            if len(key) != 1 or len(value) != 1:
                raise NormalizationError(f"{name}: {key!r} → {value!r} must map one letter to one letter")
        self.table = {ord(k): ord(v) for k, v in table.items()}
        self.name = name

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        return doc.translate_same_length(self.table) if self.table else doc


class DeleteLetters(Step):
    """Delete letters (``h`` in the broad profile), counted under ``reason``."""

    def __init__(self, letters: str, reason: str, name: str = "delete_letters") -> None:
        self.codes = np.array([ord(c) for c in letters], dtype=np.uint32)
        self.reason, self.name = reason, name

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        hit = np.isin(to_codes(doc.text), self.codes)
        if hit.any():
            ctx.deleted[self.reason] += int((doc.end[hit] - doc.start[hit]).sum())
            return doc.keep(~hit)
        return doc


class VowelPairs(Step):
    """``ae`` and ``oe`` become ``e`` inside a word, repeated until stable (broad profile)."""

    name = "vowel_pairs"
    _pattern = re.compile("ae|oe")

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        while True:
            starts = np.fromiter((m.start() for m in self._pattern.finditer(doc.text)), dtype=np.int64)
            if not len(starts):
                return doc
            doc = doc.merge_pairs(starts, "e")


class CiVowel(Step):
    """``c → t`` before ``i`` + vowel inside a word (``nuncius → nuntius``; broad profile)."""

    name = "ci_vowel"
    _pattern = re.compile("c(?=i[aeiou])")

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        positions = [m.start() for m in self._pattern.finditer(doc.text)]
        if not positions:
            return doc
        codes = to_codes(doc.text).copy()
        codes[positions] = ord("t")
        return Spanned(from_codes(codes), doc.start, doc.end, doc.mask)


# --------------------------------------------------------------------------- textreuse-blast compatibility

# textreuse-blast's tables (text_encoder.py, MIT licence, Aleksi Vesanto 2018): collisions (a/n, ä/f → W) and
# k/z → X included on purpose, for byte-for-byte parity tests only.
COMPAT_TABLES = {
    "eng": {
        "u": "E", "n": "W", "t": "R", "b": "V", "d": "C", "g": "Z", "f": "N", "s": "F", "m": "K", "w": "M", "i": "G",
        "p": "I", "o": "Y", "e": "H", "h": "S", "r": "T", "l": "A", "k": "X", "c": "B", "a": "W", "y": "Q", "j": "D",
        "v": "P",
    },
    "fin": {
        "ä": "W", "v": "V", "a": "B", "i": "C", "s": "H", "m": "N", "d": "Z", "h": "S", "e": "G", "u": "K", "f": "W",
        "w": "Y", "o": "I", "y": "T", "l": "F", "j": "R", "n": "E", "r": "P", "k": "M", "p": "Q", "t": "D", "z": "X",
    },
}
COMPAT_KEYS = {language: "".join(table) for language, table in COMPAT_TABLES.items()}


class Compat(Step):
    """Reproduces textreuse-blast's ``encode_text``: lowercase, keep only the mapped characters, with exact spans."""

    def __init__(self, language: str) -> None:
        if language not in COMPAT_KEYS:
            raise NormalizationError(f"compat: unknown language {language!r}")
        self.keys = np.array(sorted(ord(c) for c in COMPAT_KEYS[language]), dtype=np.uint32)
        self.name = f"compat_{language}"

    def apply(self, doc: Spanned, ctx: DocContext) -> Spanned:
        lowered = doc.text.lower()
        if len(lowered) != len(doc.text):  # rare: lowercase forms of different length (e.g. "İ")
            parts = [c.lower() for c in doc.text]
            lengths = np.array([len(p) for p in parts], dtype=np.int64)
            doc = Spanned("".join(parts), np.repeat(doc.start, lengths), np.repeat(doc.end, lengths),
                          np.repeat(doc.mask, lengths))
        else:
            doc = Spanned(lowered, doc.start, doc.end, doc.mask)
        codes = to_codes(doc.text)
        keep = np.isin(codes, self.keys) | (codes == ord(BARRIER))
        dropped = ~keep & (codes != ord(SILENT_SEP))
        if dropped.any():
            ctx.deleted["compat_not_mapped"] += int((doc.end[dropped] - doc.start[dropped]).sum())
        return doc.keep(keep)


# --------------------------------------------------------------------------- registry


BUILTIN_STEPS: dict[str, Callable[..., Step]] = {
    "spans": SpanRules,
    "graphemes": Graphemes,
    "fold_jv": lambda: Fold({"j": "i", "v": "u"}, name="fold_jv"),
    "fold_wy": lambda: Fold({"w": "u", "y": "i"}, name="fold_wy"),
    # k is a spelling variant of c in Latin (kalendae/calendae, karitas/caritas, Karolus/Carolus; round 5)
    "fold_kc": lambda: Fold({"k": "c"}, name="fold_kc"),
    "delete_h": lambda: DeleteLetters("h", reason="spelling_h", name="delete_h"),
    "vowel_pairs": VowelPairs,
    "ci_vowel": CiVowel,
    # Optional, off by default: letters frequently confused by HTR/OCR, folded together. The table is empty on
    # purpose; its contents are for experts (e.g. {"c": "e"} would merge c and e everywhere).
    "htr_confusions": lambda table=None: Fold(table or {}, name="htr_confusions"),
    "compat_eng": lambda: Compat("eng"),
    "compat_fin": lambda: Compat("fin"),
}


def build_step(spec: str | Mapping[str, Any], pattern_rules: Iterable[PatternRule] = ()) -> Step:
    """A step from ``"name"``, ``{"name": ..., "params": {...}}`` or ``{"class_path": ..., "init_args": {...}}``."""
    if isinstance(spec, str):
        spec = {"name": spec}
    if "class_path" in spec:
        module_name, _, class_name = str(spec["class_path"]).rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        if not (isinstance(cls, type) and issubclass(cls, Step)):
            raise NormalizationError(f"{spec['class_path']} is not a textblast.normalize.Step subclass")
        return cls(**dict(spec.get("init_args") or {}))
    name = str(spec.get("name"))
    if name not in BUILTIN_STEPS:
        raise NormalizationError(f"unknown step {name!r} (built-in: {', '.join(BUILTIN_STEPS)})")
    params = dict(spec.get("params") or {})
    if name == "spans":
        params.setdefault("pattern_rules", list(pattern_rules))
    return BUILTIN_STEPS[name](**params)
