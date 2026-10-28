"""Links in the README and the user guide point to files and headings that exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "guide"
LINK = re.compile(r"\]\(([^)\s]+)\)")

pytestmark = pytest.mark.skipif(not GUIDE.is_dir(), reason="no docs/ folder (e.g. tests from the sdist)")


def anchors(path: Path) -> set[str]:
    """GitHub's heading anchors: lower case, punctuation dropped, spaces as hyphens."""
    headings = re.findall(r"^#+\s+(.+)$", path.read_text(encoding="utf-8"), flags=re.M)
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in headings}


def test_relative_links_resolve() -> None:
    broken = []
    for page in [REPO / "README.md", REPO / "RELEASING.md", REPO / "licenses" / "README.md", *GUIDE.glob("*.md")]:
        for target in LINK.findall(page.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:", target):
                continue  # http(s) links are not checked here
            file, _, anchor = target.partition("#")
            path = (page.parent / file).resolve() if file else page
            if not path.exists():
                broken.append(f"{page.name}: {target} (no such file)")
            elif anchor and path.suffix == ".md" and anchor not in anchors(path):
                broken.append(f"{page.name}: {target} (no such heading)")
    assert broken == []
