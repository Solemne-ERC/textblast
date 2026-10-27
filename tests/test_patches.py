"""Patchers: pristine → patched, patched → unchanged, altered target → clear error; real sources if cached."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from textblast.engine.locate import default_cache_dir
from textblast.engine.patches import (
    ALREADY,
    APPLIED,
    GAP_ROW_NEW,
    MATRIX_SYMBOLS,
    PATCHES,
    PatchError,
    apply_patches,
    parse_matrix_values,
    patch_text,
    uniform_matrix,
)
from textblast.engine.versions import RELEASES

from .conftest import REFERENCE

# --------------------------------------------------------------------------- synthetic files

_ROWS = ",\n".join("    " + ", ".join(str((i * 25 + j) % 7 - 3) for j in range(25)) for i in range(25))
SM_BLOSUM62 = f"""
static const TNCBIScore s_Blosum62PSM[25 * 25] = {{
    /*       A,  R, ...  */
{_ROWS}
}};
const SNCBIPackedScoreMatrix NCBISM_Blosum62 = {{
    "ARNDCQEGHILKMFPSTWYVBJZX*",
    s_Blosum62PSM,
    -4
}};
"""

BLAST_STAT = """
#define BLOSUM45_VALUES_MAX 2
static array_of_8 blosum45_values[BLOSUM45_VALUES_MAX] = {
    {(double) INT2_MAX, (double) INT2_MAX, (double) INT2_MAX, 0.2291, 0.0924, 0.2514, 0.9113, -5.7, 0.641, 1.0, 1.0},
    {9, 1, (double) INT2_MAX, 0.206, 0.010, 0.052, 4.0, -87, 0.731887, 210.333000, 214.842000},
};
#define BLOSUM62_VALUES_MAX 3
static array_of_8 blosum62_values[BLOSUM62_VALUES_MAX] = {
    {(double) INT2_MAX, (double) INT2_MAX, (double) INT2_MAX, 0.3176, 0.134, 0.4012, 0.7916, -3.2, 0.62, 4.96, 4.96},
    {11, 1, (double) INT2_MAX, 0.267, 0.041, 0.14, 1.9, -30, 0.669720, 42.602800, 43.636200},
    {9, 1, (double) INT2_MAX, 0.206, 0.010, 0.052, 4.0, -87, 0.731887, 210.333000, 214.842000},
}; /**< Supported values for BLOSUM62. */
"""

BM_H = """
    buf_ = bvect_->allocate_tempblock();
    // ...
    bvect_->free_tempblock(buf_);
"""


def test_matrix_pristine_patched_then_unchanged() -> None:
    patched, status = patch_text("blosum62_uniform", SM_BLOSUM62)
    assert status == APPLIED
    assert parse_matrix_values(patched) == uniform_matrix()
    again, status = patch_text("blosum62_uniform", patched)
    assert (again, status) == (patched, ALREADY)


def test_uniform_matrix_values() -> None:
    values = uniform_matrix()
    x, star = MATRIX_SYMBOLS.index("X"), MATRIX_SYMBOLS.index("*")
    assert values[x * 25 + x] == -5  # X never matches itself
    assert values[star * 25 + star] == 9
    assert all(values[i * 25 + i] == 9 for i in range(25) if i != x)
    assert sum(v == -5 for v in values) == 625 - 24


def test_matrix_missing_target() -> None:
    with pytest.raises(PatchError, match="s_Blosum62PSM"):
        patch_text("blosum62_uniform", SM_BLOSUM62.replace("s_Blosum62PSM[25 * 25]", "s_Other[25 * 25]"))


def test_matrix_wrong_symbol_order() -> None:
    with pytest.raises(PatchError, match="symbol order"):
        patch_text("blosum62_uniform", SM_BLOSUM62.replace("ARNDCQEGHILKMFPSTWYVBJZX*", "ARNDCQEGHILKMFPSTWYVBZJX*"))


def test_gap_row_only_in_blosum62_table() -> None:
    patched, status = patch_text("blosum62_gap_3_11", BLAST_STAT)
    assert status == APPLIED
    blosum45, blosum62 = patched.split("#define BLOSUM62_VALUES_MAX")
    assert "{9, 1," in blosum45  # the other matrix's row is untouched
    assert "{9, 1," not in blosum62 and GAP_ROW_NEW in blosum62
    assert "{11, 1," in blosum62
    assert patch_text("blosum62_gap_3_11", patched) == (patched, ALREADY)


def test_gap_row_missing_or_ambiguous() -> None:
    with pytest.raises(PatchError, match="found 0"):
        patch_text("blosum62_gap_3_11", BLAST_STAT.replace("0.206", "0.207"))
    doubled = BLAST_STAT.replace(
        "}; /**<", "    {9, 1, (double) INT2_MAX, 0.206, 0.010, 0.052, 4.0, -87, 0.7, 2.0, 2.0},\n}; /**<"
    )
    with pytest.raises(PatchError, match="found 2"):
        patch_text("blosum62_gap_3_11", doubled)
    with pytest.raises(PatchError, match="blosum62_values"):
        patch_text("blosum62_gap_3_11", BLAST_STAT.replace("blosum62_values", "blosum99_values"))


def test_bitmagic() -> None:
    patched, status = patch_text("bitmagic_allocator", BM_H)
    assert status == APPLIED and "allocate_tempblock" not in patched
    assert patch_text("bitmagic_allocator", patched) == (patched, ALREADY)
    with pytest.raises(PatchError):
        patch_text("bitmagic_allocator", "nothing relevant here")
    with pytest.raises(PatchError, match="exactly once"):
        patch_text("bitmagic_allocator", BM_H + BM_H)


def test_apply_patches_in_tree(tmp_path: Path) -> None:
    for name, text in (("blosum62_uniform", SM_BLOSUM62), ("blosum62_gap_3_11", BLAST_STAT)):
        path = tmp_path / PATCHES[name].file
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="latin-1")
    first = apply_patches(tmp_path, ("blosum62_uniform", "blosum62_gap_3_11"))
    second = apply_patches(tmp_path, ("blosum62_uniform", "blosum62_gap_3_11"))
    assert [r.status for r in first] == [APPLIED, APPLIED]
    assert [r.status for r in second] == [ALREADY, ALREADY]
    assert [r.sha256 for r in first] == [r.sha256 for r in second]
    with pytest.raises(PatchError, match="file not found"):
        apply_patches(tmp_path, ("bitmagic_allocator",))


# --------------------------------------------------------------------------- real NCBI sources (if cached)


def _source_file(version: str, rel: str) -> str:
    release = RELEASES[version]
    tarball = default_cache_dir() / "src" / release.tarball_name
    if not tarball.is_file():
        pytest.skip(f"{tarball} not downloaded (run `textblast engine build --version {version}`)")
    with tarfile.open(tarball) as archive:
        member = archive.extractfile(f"{release.source_dir_name}/c++/{rel}")
        assert member is not None
        return member.read().decode("latin-1")


@pytest.mark.parametrize("version", sorted(RELEASES))
def test_real_sources(version: str) -> None:
    for name in RELEASES[version].patches:
        original = _source_file(version, PATCHES[name].file)
        patched, status = patch_text(name, original)
        assert status == APPLIED, name
        assert patch_text(name, patched) == (patched, ALREADY), name


def test_bitmagic_already_fixed_upstream_in_2_17() -> None:
    original = _source_file("2.17.0", PATCHES["bitmagic_allocator"].file)
    assert patch_text("bitmagic_allocator", original) == (original, ALREADY)


def test_matrix_equals_comhis() -> None:
    comhis = REFERENCE / "code/text-reuse-blast-custom-main/modifications/sm_blosum62.c"
    if not comhis.is_file():
        pytest.skip("docs/reference not present")
    assert parse_matrix_values(comhis.read_text(encoding="latin-1")) == uniform_matrix()
    for version in RELEASES:
        patched, _ = patch_text("blosum62_uniform", _source_file(version, PATCHES["blosum62_uniform"].file))
        assert parse_matrix_values(patched) == uniform_matrix()


def test_blast_stat_equals_comhis_2_13() -> None:
    comhis = REFERENCE / (
        "code/text-reuse-blast-custom-main/ncbi-blast-2.13.0+-src_modified/c++/src/algo/blast/core/blast_stat.c"
    )
    if not comhis.is_file():
        pytest.skip("docs/reference not present")
    patched, _ = patch_text("blosum62_gap_3_11", _source_file("2.13.0", PATCHES["blosum62_gap_3_11"].file))
    assert patched == comhis.read_text(encoding="latin-1")
