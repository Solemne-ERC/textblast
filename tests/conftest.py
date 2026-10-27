"""Shared fixtures: located engines (tests marked ``engine`` are skipped when none is built)."""

from __future__ import annotations

from pathlib import Path

import pytest

from textblast.engine.locate import EngineError, find_engine
from textblast.engine.versions import RELEASES

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "docs" / "reference"


@pytest.fixture(params=sorted(RELEASES), scope="session")
def engine(request: pytest.FixtureRequest):
    """Each built engine version in the cache (or the one given by TEXTBLAST_BIN)."""
    try:
        return find_engine(version=request.param)
    except EngineError as error:
        pytest.skip(f"no engine for BLAST {request.param}: {error}")
