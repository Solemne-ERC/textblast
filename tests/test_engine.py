"""Engine tests 1–5 against every built engine (marker ``engine``; skipped when none is built)."""

from __future__ import annotations

import pytest

from textblast.engine.check import (
    BARRIER_LENGTH,
    Runner,
    check_barrier,
    check_exact_seeds,
    check_gap_costs,
    check_mask,
    check_uniform_matrix,
    verify_barrier,
)

pytestmark = pytest.mark.engine

CHECKS = [check_uniform_matrix, check_gap_costs, check_exact_seeds, check_barrier, check_mask]


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_engine_check(engine, check, tmp_path) -> None:
    result = check(Runner(engine, tmp_path))
    assert result.passed, result.detail


def test_short_barrier_is_bridged_under_composition_statistics(engine) -> None:
    """Documents why the barrier is 100 letters: 20 hold under cbs 0 but not under cbs 2."""
    assert verify_barrier(engine, 20, ["-comp_based_stats", "0"])
    assert not verify_barrier(engine, 20, ["-comp_based_stats", "2"])
    assert verify_barrier(engine, BARRIER_LENGTH, ["-comp_based_stats", "2"])


def test_barrier_check_with_a_runs_complete_arguments(engine) -> None:
    """The pre-run check uses the run's own arguments (matrix, gaps, seeding, e-value, -dbsize) exactly."""
    from textblast.config import BlastOptions, blast_arguments, resolved_settings

    for preset in ("textblast", "textreuse-blast", "comhis"):
        settings = resolved_settings(BlastOptions(preset=preset))
        args = blast_arguments(settings, btop=True, dbsize=40_000_000)
        assert verify_barrier(engine, BARRIER_LENGTH, args, complete=True)
        # a short barrier is bridged only by the re-alignment of comp_based_stats 2 (M0, round 4)
        assert verify_barrier(engine, 20, args, complete=True) == (settings["comp_based_stats"] == 0)
