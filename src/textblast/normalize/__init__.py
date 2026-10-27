"""Normalization: surface text → letters with exact spans (plan section 6)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .spans import BARRIER, SEP, SILENT_SEP, Spanned
from .steps import (
    BUILTIN_STEPS,
    DocContext,
    Exclusion,
    NormalizationError,
    PatternRule,
    Step,
    build_step,
)

__all__ = [
    "BARRIER",
    "BUILTIN_STEPS",
    "SEP",
    "SILENT_SEP",
    "DocContext",
    "Exclusion",
    "NormalizationError",
    "Normalizer",
    "PatternRule",
    "Spanned",
    "Step",
]


class Normalizer:
    """An ordered list of steps; ``run`` turns one document's surface text into letters with spans."""

    def __init__(
        self,
        steps: Sequence[str | Mapping[str, Any]],
        pattern_rules: Iterable[PatternRule | Mapping[str, Any]] = (),
    ) -> None:
        rules = [r if isinstance(r, PatternRule) else PatternRule(**r) for r in pattern_rules]
        self.steps = [build_step(spec, rules) for spec in steps]
        if rules and not any(step.name == "spans" for step in self.steps):
            raise NormalizationError("pattern rules need the 'spans' step")

    def run(self, text: str, ctx: DocContext) -> Spanned:
        doc = Spanned.from_surface(text)
        for step in self.steps:
            doc = step.apply(doc, ctx)
        return doc
