"""Engine tests on tiny synthetic sequences (plan section 4; notes findings 7–9, Q14).

Every test runs ``blastp -query a.fsa -subject b.fsa`` (no database) with the fixed text-reuse settings. Sequences
that must be unrelated are verified to share no seed word in BLAST's 15-group seed alphabet, so a test cannot pass or
fail by chance.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logs import get_logger
from .locate import Engine, EngineError, read_json, write_json

log = get_logger("engine")

SYMBOLS = "ARNDCQEGHILKMFPSTWYVBJZ"  # +9 on the diagonal of the patched matrix (X is excluded on purpose)
SEED_GROUPS = ("ST", "IJV", "LM", "KR", "EQZ", "A", "G", "BD", "P", "N", "F", "Y", "H", "C", "W")
GROUP = {symbol: index for index, group in enumerate(SEED_GROUPS) for symbol in group}
WORD = 6

BASE_ARGS = ["-matrix", "BLOSUM62", "-word_size", "6", "-threshold", "400"]
GAP_COSTS = ("3", "11")
OUTFMT = "6 qstart qend sstart send length nident mismatch gapopen gaps score bitscore evalue"

MASK_FLAGS: dict[str, list[str]] = {
    "-lcase_masking -soft_masking true": ["-lcase_masking", "-soft_masking", "true"],
    "-lcase_masking -soft_masking false": ["-lcase_masking", "-soft_masking", "false"],
    "-lcase_masking": ["-lcase_masking"],
    "(no masking flags)": [],
}
# Barrier length agreed in round 4: 20 X letters are bridged under -comp_based_stats 2 (X/X costs about -1 in the
# re-alignment matrix instead of -5); from about 80 they hold with BLAST's default x-drop settings.
BARRIER_LENGTH = 100
BARRIER_SWEEP = (20, 40, 60, 80, 100)


@dataclass(frozen=True)
class Hit:
    qstart: int
    qend: int
    sstart: int
    send: int
    length: int
    nident: int
    mismatch: int
    gapopen: int
    gaps: int
    score: int
    bitscore: float
    evalue: float

    def covers_query(self, start: int, end: int) -> bool:
        """True if the hit covers query positions start..end (1-based, inclusive)."""
        return self.qstart <= start and self.qend >= end

    def brief(self) -> str:
        return (
            f"q{self.qstart}-{self.qend} s{self.sstart}-{self.send} len {self.length} "
            f"mism {self.mismatch} gaps {self.gaps} score {self.score} e {self.evalue:.2g}"
        )


@dataclass
class TestResult:
    number: int
    name: str
    passed: bool | None  # None: report only
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckReport:
    engine: dict[str, Any]
    results: list[TestResult]
    checked_at: str

    @property
    def passed(self) -> bool:
        return all(r.passed is not False for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "passed": self.passed,
            "engine": self.engine,
            "results": [asdict(r) for r in self.results],
        }


# --------------------------------------------------------------------------- running blastp


class Runner:
    """Runs blastp on a query/subject pair in a scratch directory."""

    def __init__(self, engine: Engine, workdir: Path, run_args: Iterable[str] | None = None) -> None:
        """``run_args``: a run's complete BLAST arguments, used instead of the test defaults (``-outfmt`` is
        replaced by the one the checks parse)."""
        self.blastp = engine.binary("blastp")
        self.workdir = workdir
        self.calls = 0
        self.run_args = None if run_args is None else without_option(list(run_args), "-outfmt")

    def run_raw(
        self, query: str, subject: str, extra: Iterable[str] = (), gap: tuple[str, str] = GAP_COSTS
    ) -> subprocess.CompletedProcess[str]:
        self.calls += 1
        query_path = self.workdir / f"q{self.calls}.fsa"
        subject_path = self.workdir / f"s{self.calls}.fsa"
        query_path.write_text(f">q\n{query}\n")
        subject_path.write_text(f">s\n{subject}\n")
        base = self.run_args if self.run_args is not None else [*BASE_ARGS, "-gapopen", gap[0], "-gapextend", gap[1]]
        cmd = [str(self.blastp), "-query", str(query_path), "-subject", str(subject_path), *base,
               "-outfmt", OUTFMT, *extra]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    def run(self, query: str, subject: str, *extra: str, evalue: str = "1e-3") -> list[Hit]:
        evalue_args = [] if self.run_args is not None else ["-evalue", evalue]  # a run brings its own e-value
        result = self.run_raw(query, subject, [*evalue_args, *extra])
        if result.returncode != 0:
            raise EngineError(f"blastp failed ({result.returncode}): {result.stderr.strip()[:500]}")
        hits = []
        for line in result.stdout.splitlines():
            if line.strip():
                cells = line.split("\t")
                hits.append(Hit(*(int(c) for c in cells[:10]), float(cells[10]), float(cells[11])))
        return sorted(hits, key=lambda h: (h.qstart, h.sstart))


# --------------------------------------------------------------------------- synthetic sequences


def seed_words(sequence: str) -> set[tuple[int, ...]]:
    """Words of the 15-group seed alphabet (case-insensitive; windows containing X are never seeds)."""
    upper = sequence.upper()
    return {
        tuple(GROUP[c] for c in upper[i : i + WORD])
        for i in range(len(upper) - WORD + 1)
        if "X" not in upper[i : i + WORD]
    }


class Sequences:
    """Deterministic random sequences over the usable symbols."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def random(self, n: int) -> str:
        return "".join(self.rng.choice(SYMBOLS) for _ in range(n))

    def unrelated(self, n: int, *others: str) -> str:
        """A sequence sharing no seed word with ``others`` (nor with itself shifted)."""
        taken = set().union(*(seed_words(o) for o in others)) if others else set()
        for _ in range(1000):
            candidate = self.random(n)
            if not (seed_words(candidate) & taken):
                return candidate
        raise RuntimeError("could not generate an unrelated sequence")

    def other_group(self, symbol: str) -> str:
        """A symbol from a different seed group (so no seed can bridge the substitution)."""
        return self.rng.choice([c for c in SYMBOLS if GROUP[c] != GROUP[symbol]])


# --------------------------------------------------------------------------- the tests


def check_uniform_matrix(runner: Runner) -> TestResult:
    """1. With -comp_based_stats 0, an ungapped alignment scores exactly 9·matches − 5·mismatches."""
    seqs = Sequences(1)
    letters = list(SYMBOLS * 5) + [seqs.rng.choice(SYMBOLS) for _ in range(5)]
    seqs.rng.shuffle(letters)
    query = "".join(letters)  # every symbol occurs, so every diagonal entry is exercised
    subject = list(query)
    for pos in (30, 60, 90):
        subject[pos] = seqs.other_group(subject[pos])
    hits = runner.run(query, "".join(subject), "-comp_based_stats", "0")
    expected = 9 * (len(query) - 3) - 5 * 3
    ok = (
        len(hits) == 1
        and hits[0].gaps == 0
        and hits[0].length == len(query)
        and hits[0].nident == len(query) - 3
        and hits[0].score == expected == 9 * hits[0].nident - 5 * hits[0].mismatch
    )
    detail = (
        f"score {hits[0].score} = 9×{hits[0].nident} − 5×{hits[0].mismatch}" if ok
        else f"expected one ungapped hit scoring {expected}; got {[h.brief() for h in hits]}"
    )
    return TestResult(1, "uniform matrix", ok, detail, {"hits": [asdict(h) for h in hits]})


def check_gap_costs(runner: Runner) -> TestResult:
    """2. Gap costs 3/11 are accepted and scored as 3 + 11·length; the replaced {9, 1} row is rejected."""
    seqs = Sequences(2)
    query = seqs.random(200)
    subject = query[:100] + query[101:]  # one deleted letter
    hits = runner.run(query, subject, "-comp_based_stats", "0")
    gapped = [h for h in hits if h.gapopen == 1 and h.gaps == 1]
    gap_ok = len(gapped) == 1 and gapped[0].score == (
        9 * gapped[0].nident - 5 * gapped[0].mismatch - 3 * gapped[0].gapopen - 11 * gapped[0].gaps
    )
    rejected = runner.run_raw(query, query, gap=("9", "1"))
    rejected_ok = rejected.returncode != 0 and "gap" in rejected.stderr.lower()
    other = runner.run_raw(query, query, gap=("11", "1"))
    ok = gap_ok and rejected_ok
    parts = [
        f"1-letter gap scored {gapped[0].score} = 9×{gapped[0].nident} − (3 + 11×1)" if gap_ok
        else f"gapped scoring wrong: {[h.brief() for h in hits]}",
        "9/1 rejected" if rejected_ok else "9/1 still accepted",
        "11/1 (untouched row) accepted" if other.returncode == 0 else "11/1 rejected (unexpected)",
    ]
    return TestResult(2, "gap costs 3/11", ok, "; ".join(parts),
                      {"rejected_stderr": rejected.stderr.strip()[:300], "hits": [asdict(h) for h in hits]})


def check_exact_seeds(runner: Runner) -> TestResult:
    """3. Identical sequences give a hit; similar ones without a shared seed word give none."""
    seqs = Sequences(3)
    query = seqs.random(150)
    identical = runner.run(query, query, "-comp_based_stats", "0")
    identical_ok = any(h.covers_query(1, 150) for h in identical)
    for _ in range(100):
        subject = "".join(seqs.other_group(c) if i % 5 == 4 else c for i, c in enumerate(query))
        if not (seed_words(query) & seed_words(subject)):
            break
        query = seqs.random(150)
    else:  # pragma: no cover
        raise RuntimeError("could not build a seedless pair")
    unseeded = {cbs: runner.run(query, subject, "-comp_based_stats", cbs) for cbs in ("0", "2")}
    unseeded_ok = all(not hits for hits in unseeded.values())
    identity = sum(a == b for a, b in zip(query, subject, strict=True)) / len(query)
    detail = (
        f"identical pair found; {identity:.0%}-identical pair with no shared seed word gives no hit"
        if identical_ok and unseeded_ok
        else f"identical: {[h.brief() for h in identical]}; unseeded: "
        f"{ {k: [h.brief() for h in v] for k, v in unseeded.items()} }"
    )
    return TestResult(3, "exact seeds", identical_ok and unseeded_ok, detail)


def _barrier_holds(runner: Runner, seqs: Sequences, length: int, args: Iterable[str]) -> tuple[bool, list[Hit]]:
    """Two identical 100-letter blocks around an X run: True if BLAST reports them separately."""
    left = seqs.random(100)
    right = seqs.unrelated(100, left)
    flank_a = seqs.unrelated(30, left, right)
    flank_b = seqs.unrelated(30, left, right, flank_a)
    barrier = "X" * length
    query = left + barrier + right
    subject = flank_a + left + barrier + right + flank_b
    hits = runner.run(query, subject, *args)
    spanning = [h for h in hits if h.qstart <= 100 and h.qend >= 100 + length + 1]
    both_found = any(h.covers_query(5, 95) for h in hits) and any(
        h.covers_query(100 + length + 5, 100 + length + 95) for h in hits
    )
    return (not spanning and both_found), hits


def check_barrier(runner: Runner, length: int = BARRIER_LENGTH) -> TestResult:
    """4. An X run of the configured length separates two identical blocks under -comp_based_stats 0 and 2."""
    outcome: dict[str, tuple[bool, list[Hit]]] = {}
    for cbs in ("0", "2"):
        results = [_barrier_holds(runner, Sequences(seed), length, ["-comp_based_stats", cbs]) for seed in (4, 5, 6)]
        holds = all(ok for ok, _ in results)
        outcome[cbs] = (holds, next((hits for ok, hits in results if not ok), results[0][1]))
    sweep = {
        cbs: {n: _barrier_holds(runner, Sequences(40 + n), n, ["-comp_based_stats", cbs])[0] for n in BARRIER_SWEEP}
        for cbs in ("0", "2")
    }
    minimum = {
        cbs: next((n for n in BARRIER_SWEEP if all(results[m] for m in BARRIER_SWEEP if m >= n)), None)
        for cbs, results in sweep.items()
    }
    ok = all(holds for holds, _ in outcome.values())
    detail = f"{length} × X: " + "; ".join(
        f"cbs {cbs} {'holds' if holds else 'BRIDGED: ' + str([h.brief() for h in hits])}"
        for cbs, (holds, hits) in outcome.items()
    )
    detail += f" (shortest holding run in sweep {BARRIER_SWEEP}: cbs 0 → {minimum['0']}, cbs 2 → {minimum['2']})"
    return TestResult(4, "barrier (X run)", ok, detail, {"length": length, "sweep": sweep, "minimum": minimum})


def verify_barrier(engine: Engine, length: int, blast_args: Iterable[str], complete: bool = False) -> bool:
    """Does an X run of ``length`` hold with these BLAST arguments? (pre-run check, plan section 7).

    ``complete``: ``blast_args`` are a run's full argument list (matrix, gap costs, seeding, e-value, ``-dbsize``…),
    used exactly as given; otherwise they are added to the engine tests' defaults (word size 6, threshold 400).
    """
    args = list(blast_args)
    with tempfile.TemporaryDirectory(prefix="textblast-barrier-") as tmp:
        runner = Runner(engine, Path(tmp), run_args=args if complete else None)
        extra = [] if complete else args
        return all(_barrier_holds(runner, Sequences(seed), length, extra)[0] for seed in (4, 5, 6))


def without_option(args: list[str], name: str) -> list[str]:
    """``args`` without option ``name`` and its value."""
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == name:
            skip = True
        else:
            out.append(arg)
    return out


def check_mask(runner: Runner) -> TestResult:
    """5. Masked query letters start no seed but alignments extend through them; records the working flags."""
    seqs = Sequences(5)
    masked = seqs.random(60)
    u1 = seqs.unrelated(50, masked)
    u2 = seqs.unrelated(50, masked, u1)
    v1 = seqs.unrelated(50, masked, u1, u2)
    v2 = seqs.unrelated(50, masked, u1, u2, v1)
    seed_query, seed_subject = u1 + masked.lower() + u2, v1 + masked + v2

    a1 = seqs.random(80)
    a2 = seqs.unrelated(80, a1)
    middle = seqs.unrelated(30, a1, a2)
    f1 = seqs.unrelated(40, a1, a2, middle)
    f2 = seqs.unrelated(40, a1, a2, middle, f1)
    ext_query, ext_subject = a1 + middle.lower() + a2, f1 + a1 + middle + a2 + f2

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for label, flags in MASK_FLAGS.items():
        table[label] = {}
        for cbs in ("0", "2"):
            no_seed = not runner.run(seed_query, seed_subject, "-comp_based_stats", cbs, *flags)
            ext_hits = runner.run(ext_query, ext_subject, "-comp_based_stats", cbs, *flags)
            extends = any(h.covers_query(5, 185) for h in ext_hits)
            table[label][cbs] = {"no_seed": no_seed, "extends": extends, "works": no_seed and extends}
    working = [label for label, row in table.items() if all(cell["works"] for cell in row.values())]
    ok = "-lcase_masking -soft_masking true" in working
    detail = (
        f"'no seeds, extension allowed' with: {', '.join(working) or 'NO flag combination'} (cbs 0 and 2)"
    )
    return TestResult(5, "mask (soft masking)", ok, detail, {"table": table, "working": working})


def report_composition_stats(runner: Runner) -> TestResult:
    """6. Report only: two identical 100-letter blocks separated by 30 substituted letters (finding 7)."""
    seqs = Sequences(6)
    block_a = seqs.random(100)
    block_b = seqs.unrelated(100, block_a)
    middle = seqs.unrelated(30, block_a, block_b)
    substituted = "".join(seqs.other_group(c) for c in middle)
    q_flank_a = seqs.unrelated(100, block_a, block_b, middle, substituted)
    q_flank_b = seqs.unrelated(100, block_a, block_b, middle, substituted, q_flank_a)
    s_flank_a = seqs.unrelated(50, block_a, block_b, middle, substituted, q_flank_a, q_flank_b)
    s_flank_b = seqs.unrelated(50, block_a, block_b, middle, substituted, q_flank_a, q_flank_b, s_flank_a)
    query = q_flank_a + block_a + middle + block_b + q_flank_b
    subject = s_flank_a + block_a + substituted + block_b + s_flank_b
    rows = {}
    for cbs in ("0", "2"):
        hits = runner.run(query, subject, "-comp_based_stats", cbs)
        rows[cbs] = [asdict(h) for h in hits]
    detail = " | ".join(
        f"cbs {cbs}: {len(hits)} hit(s): " + "; ".join(Hit(**h).brief() for h in hits) for cbs, hits in rows.items()
    )
    return TestResult(6, "composition-based statistics (report)", None, detail, {"rows": rows})


CHECKS: tuple[Callable[[Runner], TestResult], ...] = (
    check_uniform_matrix,
    check_gap_costs,
    check_exact_seeds,
    check_barrier,
    check_mask,
    report_composition_stats,
)


# --------------------------------------------------------------------------- orchestration


def behaviour_fingerprint(report: CheckReport) -> str:
    """Hash of the engine's observable behaviour in the tests (fingerprint for user-supplied binaries)."""
    signature = [(r.number, r.passed, r.detail) for r in report.results]
    return "behaviour-" + hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:12]


def run_checks(engine: Engine, cache: Path | None = None, record: bool = True) -> CheckReport:
    """Run all engine tests, log a short table, and record the result with the engine."""
    log.info(f"engine check: BLAST {engine.version} · {engine.fingerprint} · {engine.bin_dir}")
    results = []
    with tempfile.TemporaryDirectory(prefix="textblast-check-") as tmp:
        runner = Runner(engine, Path(tmp))
        for check in CHECKS:
            try:
                results.append(check(runner))
            except EngineError as error:
                number = CHECKS.index(check) + 1
                results.append(TestResult(number, check.__name__, False, f"blastp error: {error}"))
    report = CheckReport(
        engine={"version": engine.version, "fingerprint": engine.fingerprint,
                "bin_dir": str(engine.bin_dir.resolve()), "source": engine.source},
        results=results,
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    for result in results:
        status = {True: "pass", False: "FAIL", None: "report"}[result.passed]
        log.info(f"  {result.number} {result.name:<40} {status:<6} {result.detail}")
    scored = [r for r in results if r.passed is not None]
    passed = sum(1 for r in scored if r.passed)
    (log.info if report.passed else log.error)(f"engine check: {passed}/{len(scored)} tests passed")
    if record:
        _record(engine, report)
    return report


def _record(engine: Engine, report: CheckReport) -> None:
    if engine.check_path is not None:  # bundled engine: the results go to the cache, not into the package
        record = {"bin_dir": str(engine.bin_dir.resolve()), "fingerprint": engine.fingerprint,
                  "last_check": report.to_dict()}
        write_json(engine.check_path, record)
        engine.info = {**engine.info, "last_check": record["last_check"]}
        return
    if engine.info_path is None:
        return
    info = read_json(engine.info_path) if engine.info_path.is_file() else dict(engine.info)
    info["last_check"] = report.to_dict()
    if not engine.managed:
        info["fingerprint"] = behaviour_fingerprint(report)
        info.setdefault("bin_dir", str(engine.bin_dir.resolve()))
        engine.fingerprint = info["fingerprint"]
    write_json(engine.info_path, info)
    engine.info = info


def describe_engine(engine: Engine) -> list[str]:
    """Lines for ``textblast engine info``."""
    info = engine.info
    lines = [
        f"version      {engine.version}  ({info.get('blastp_version', '')})",
        f"fingerprint  {engine.fingerprint}",
        f"binaries     {engine.bin_dir}  (found via {engine.source})",
    ]
    build = info.get("build")
    if build:
        lines.append(
            f"built        {build.get('built_at')} in {build.get('build_seconds')} s · {build.get('compiler')} · "
            f"OpenMP {build.get('openmp')} · {build.get('projects')} build"
        )
    for patch in info.get("patches", []):
        lines.append(f"patch        {patch['name']}: {patch['status']} ({patch['file']})")
    check = info.get("last_check")
    if check:
        scored = [r for r in check["results"] if r["passed"] is not None]
        passed = sum(1 for r in scored if r["passed"])
        lines.append(f"last check   {check['checked_at']}: {passed}/{len(scored)} passed")
        for r in check["results"]:
            status = {True: "pass", False: "FAIL", None: "report"}[r["passed"]]
            lines.append(f"  {r['number']} {r['name']:<40} {status}")
    else:
        lines.append("last check   never (run `textblast engine check`)")
    return lines
