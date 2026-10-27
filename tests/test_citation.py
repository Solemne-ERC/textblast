"""CITATION.cff (GitHub, citation managers) and .zenodo.json (Zenodo, which reads only it when present) agree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import textblast

REPO = Path(__file__).resolve().parents[1]
CFF, ZENODO = REPO / "CITATION.cff", REPO / ".zenodo.json"

pytestmark = pytest.mark.skipif(not ZENODO.is_file(), reason="no .zenodo.json (e.g. tests from the sdist)")


def test_citation_and_zenodo_agree() -> None:
    cff = yaml.safe_load(CFF.read_text(encoding="utf-8"))
    zenodo = json.loads(ZENODO.read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0" and cff["type"] == "software" and cff["message"]
    assert cff["version"] == zenodo["version"] == textblast.__version__
    assert cff["title"] == zenodo["title"]
    assert " ".join(cff["abstract"].split()) == zenodo["description"]
    assert cff["keywords"] == zenodo["keywords"]
    assert cff["license"].lower() == zenodo["license"]
    authors = [(f"{a['family-names']}, {a['given-names']}", a["orcid"].removeprefix("https://orcid.org/"),
                a["affiliation"]) for a in cff["authors"]]
    assert authors == [(c["name"], c["orcid"], c["affiliation"]) for c in zenodo["creators"]]
    assert zenodo["upload_type"] == "software"
    assert zenodo["grants"] == [{"id": "10.13039/501100000780::101087979"}]  # ERC, SOLEMNE (European Commission)
