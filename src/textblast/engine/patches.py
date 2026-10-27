"""Content-based patches that adapt NCBI BLAST+ to text reuse .

Each patcher finds its target by content, never by line number. It leaves an already-patched file unchanged and
fails with a clear message when its target is missing or ambiguous.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class PatchError(RuntimeError):
    """A patch target was missing, ambiguous or malformed."""


APPLIED, ALREADY = "applied", "already patched"


@dataclass(frozen=True)
class PatchResult:
    name: str
    file: str
    status: str
    sha256: str


@dataclass(frozen=True)
class Patch:
    name: str
    file: str  # relative to the source tree's c++/ directory
    description: str
    transform: Callable[[str], tuple[str, str]]  # text -> (new text, status)


# --------------------------------------------------------------------------- uniform BLOSUM62 matrix

MATRIX_SYMBOLS = "ARNDCQEGHILKMFPSTWYVBJZX*"
MATCH, MISMATCH = 9, -5

_MATRIX_BLOCK = re.compile(r"(static const TNCBIScore s_Blosum62PSM\[25 \* 25\] = \{)(.*?)(\};)", re.DOTALL)
_MATRIX_ORDER = re.compile(r'NCBISM_Blosum62 = \{\s*"([A-Z*]+)"')
_C_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def uniform_matrix() -> list[int]:
    """The text-reuse matrix: +9 on the diagonal except X/X, -5 elsewhere (row-major, 25 x 25)."""
    return [
        MATCH if i == j and a != "X" else MISMATCH
        for i, a in enumerate(MATRIX_SYMBOLS)
        for j, _ in enumerate(MATRIX_SYMBOLS)
    ]


def parse_matrix_values(text: str) -> list[int]:
    """The 625 values of ``s_Blosum62PSM`` in ``sm_blosum62.c`` (comments ignored)."""
    blocks = _MATRIX_BLOCK.findall(text)
    if len(blocks) != 1:
        raise PatchError(f"expected one s_Blosum62PSM table in sm_blosum62.c, found {len(blocks)}")
    values = [int(v) for v in re.findall(r"-?\d+", _C_COMMENT.sub(" ", blocks[0][1]))]
    if len(values) != 25 * 25:
        raise PatchError(f"s_Blosum62PSM should hold 625 values, found {len(values)}")
    return values


def _render_matrix(values: list[int]) -> str:
    def row(cells: list[int], last: bool) -> str:
        text = ", ".join(f"{v:2d}" for v in cells)
        return text if last else text + ","

    header_a = ",  ".join(MATRIX_SYMBOLS[:13])
    header_b = ",  ".join(MATRIX_SYMBOLS[13:])
    lines = [
        "",
        "    /* Uniform text-reuse matrix (textblast): +9 on the diagonal except X/X, -5 elsewhere. */",
        f"    /*       {header_a},",
        f"             {header_b}        */",
    ]
    for i, symbol in enumerate(MATRIX_SYMBOLS):
        cells = values[25 * i : 25 * (i + 1)]
        last_row = i == 24
        lines.append(f"    /*{symbol}*/   {row(cells[:13], False)}")
        lines.append(f"            {row(cells[13:], last_row)}")
    return "\n".join(lines) + "\n"


def _patch_matrix(text: str) -> tuple[str, str]:
    order = _MATRIX_ORDER.findall(text)
    if order != [MATRIX_SYMBOLS]:
        raise PatchError(f"unexpected NCBISM_Blosum62 symbol order {order!r} (expected {MATRIX_SYMBOLS!r})")
    if parse_matrix_values(text) == uniform_matrix():
        return text, ALREADY
    new_text = _MATRIX_BLOCK.sub(lambda m: m.group(1) + _render_matrix(uniform_matrix()) + m.group(3), text)
    return new_text, APPLIED


# --------------------------------------------------------------------------- gap costs 3/11 for BLOSUM62

GAP_ROW_NEW = "{3, 11, (double) INT2_MAX, 0.201, 0.012, 0.061, 3.3, -58, 0.740802, 140.417000, 141.882000},"

_BLOSUM62_TABLE = re.compile(
    r"(static array_of_8 blosum62_values\[BLOSUM62_VALUES_MAX\] = \{\n)(.*?)(\n\};)", re.DOTALL
)
_ROW_9_1 = re.compile(r"^([ \t]*)\{\s*9,\s*1,\s*\(double\)\s*INT2_MAX,\s*0\.206,[^\n]*\},[ \t]*$", re.MULTILINE)
_ROW_3_11 = re.compile(r"\{\s*3,\s*11,\s*\(double\)\s*INT2_MAX,")


def _patch_gap_row(text: str) -> tuple[str, str]:
    tables = _BLOSUM62_TABLE.findall(text)
    if len(tables) != 1:
        raise PatchError(f"expected one blosum62_values table in blast_stat.c, found {len(tables)}")
    body = tables[0][1]
    old_rows = _ROW_9_1.findall(body)
    has_new = bool(_ROW_3_11.search(body))
    if not old_rows and has_new:
        return text, ALREADY
    if has_new:
        raise PatchError("blosum62_values contains both a {9, 1, ...} row and a {3, 11, ...} row")
    if len(old_rows) != 1:
        raise PatchError(f"expected one {{9, 1, ..., 0.206, ...}} row in blosum62_values, found {len(old_rows)}")
    new_body = _ROW_9_1.sub(lambda m: m.group(1) + GAP_ROW_NEW, body)
    return _BLOSUM62_TABLE.sub(lambda m: m.group(1) + new_body + m.group(3), text), APPLIED


# --------------------------------------------------------------------------- BitMagic allocator calls (2.13.0)

_BITMAGIC_REPLACEMENTS = (
    (
        "buf_ = bvect_->allocate_tempblock();",
        "buf_ = (value_type*) bvect_->blockman_.get_allocator().alloc_bit_block();",
    ),
    (
        "bvect_->free_tempblock(buf_);",
        "bvect_->blockman_.get_allocator().free_bit_block((bm::word_t*)buf_);",
    ),
)


def _patch_bitmagic(text: str) -> tuple[str, str]:
    """Obsolete BitMagic allocator calls rejected by current Apple Clang (fixed upstream in later releases)."""
    counts = [text.count(old) for old, _ in _BITMAGIC_REPLACEMENTS]
    if counts == [0, 0]:
        if all(new in text for _, new in _BITMAGIC_REPLACEMENTS):
            return text, ALREADY
        raise PatchError("bm.h: neither the obsolete allocator calls nor their replacements were found")
    if counts != [1, 1]:
        raise PatchError(f"bm.h: expected each obsolete allocator call exactly once, found {counts}")
    for old, new in _BITMAGIC_REPLACEMENTS:
        text = text.replace(old, new)
    return text, APPLIED


# --------------------------------------------------------------------------- registry and application

PATCHES: dict[str, Patch] = {
    patch.name: patch
    for patch in (
        Patch(
            "blosum62_uniform",
            "src/util/tables/sm_blosum62.c",
            "BLOSUM62 matrix replaced by the uniform text-reuse matrix (+9/-5)",
            _patch_matrix,
        ),
        Patch(
            "blosum62_gap_3_11",
            "src/algo/blast/core/blast_stat.c",
            "BLOSUM62 gap row {9, 1} replaced by {3, 11}",
            _patch_gap_row,
        ),
        Patch(
            "bitmagic_allocator",
            "include/util/bitset/bm.h",
            "BitMagic allocator calls updated for current compilers",
            _patch_bitmagic,
        ),
    )
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("latin-1")).hexdigest()


def patch_text(name: str, text: str) -> tuple[str, str]:
    """Apply one patch to file contents; returns (new text, status)."""
    try:
        return PATCHES[name].transform(text)
    except KeyError:
        raise PatchError(f"unknown patch {name!r}") from None


def apply_patches(cpp_dir: Path, names: tuple[str, ...]) -> list[PatchResult]:
    """Apply the named patches inside a source tree's ``c++`` directory, in place."""
    results = []
    for name in names:
        patch = PATCHES[name]
        path = cpp_dir / patch.file
        if not path.is_file():
            raise PatchError(f"{name}: file not found: {path}")
        # latin-1 round-trips every byte, so untouched parts of the file stay byte-identical.
        text = path.read_text(encoding="latin-1")
        try:
            new_text, status = patch.transform(text)
        except PatchError as error:
            raise PatchError(f"{name} ({patch.file}): {error}") from None
        if new_text != text:
            path.write_text(new_text, encoding="latin-1")
        results.append(PatchResult(name, patch.file, status, sha256_text(new_text)))
    return results
