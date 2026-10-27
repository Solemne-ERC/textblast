"""``textblast post``: clip (optional) → join (optional) → defragment → annotate → hit filter → cluster
(optional).

Inputs: ``tables/hsps.parquet`` (from ``merge``) and the corpus store. Outputs in ``tables/``: ``pieces``,
``piece_map``, ``pairs``, ``unit_overlaps``, ``join_map`` (which raw hits continue each other; always), optionally
``hsps_clipped``, ``hsps_joined``, ``pairs_filtered``, and ``clusters`` with ``cluster_summary``. The step is
skipped when its inputs and settings are unchanged.

Pieces are annotated with ``masked_share``, ``doc_degree`` (distinct partner documents) and ``ref_overlap_<name>``
for each reference set (share of the piece covered by hits with documents of the set).

The **hit filter** is evaluated on hit rows (the view being defragmented): every hit field, plus ``piece1``,
``piece2``, ``num_orig_links`` (of the hit's pair), the piece annotations of both sides (``masked_share1``,
``doc_degree2``, ``ref_overlap_bible1``, …), the pieces' lengths in letters (``piece_letters1``/``2``) and document
metadata (``field``/``field2``). Named filters (``@no_formulae``, …; ``filters.NAMED_FILTERS``) can be used in it.
``pairs_filtered`` holds the pairs that keep at least one hit, with statistics from the kept hits; exports of hits
use the same kept hits.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from . import __version__
from .annotate import (
    doc_degree,
    encoded_range,
    masked_share,
    reference_overlap,
    reference_set_members,
    unit_overlaps,
    units_frame,
)
from .clip import clip_hits
from .cluster import ClusterOptions, cluster
from .defrag import DefragOptions, defragment, make_pairs
from .filters import FilterOptions, apply_hit_filter
from .join import JoinOptions, crossed_neighbours, join_hits, join_map_for, join_stats
from .logs import Stage, format_bytes, format_count, get_logger, peak_memory_bytes
from .mapping import CorpusStore
from .prepare import append_event, read_manifest, write_manifest
from .sysinfo import default_workers

log = get_logger("post")

TABLES = Path("tables")
SCHEMAS = {  # table → schema version (the files are the interface for readers)
    "documents": 1, "hsps": 1, "hsps_clipped": 1, "hsps_joined": 1, "join_map": 2, "pieces": 2, "piece_map": 1,
    "pairs": 1, "pairs_filtered": 1, "unit_overlaps": 1, "units": 1, "texts": 1, "clusters": 1,
    "cluster_summary": 1,
}
VIEWS = {"hsps": ("hsps", "hsp_id"), "clipped": ("hsps_clipped", "clip_id"), "joined": ("hsps_joined", "joined_id")}
PIECE_ANNOTATIONS = ("masked_share", "doc_degree")  # plus ref_overlap_<name>


class PostError(RuntimeError):
    """``post`` cannot run (no merged hits, clipped view without clipping, …)."""


@dataclass
class UnitOptions:
    clip: bool = False  # also build tables/hsps_clipped.parquet (hits cut at unit boundaries)


@dataclass
class AnnotateOptions:
    # name → SQL over document fields, e.g. {"bible": "doc_id LIKE 'bible:%'"}: per piece ref_overlap_<name>
    reference_sets: dict[str, str] = field(default_factory=dict)


def _code_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("post.py", "defrag.py", "annotate.py", "clip.py", "align.py", "filters.py", "join.py", "cluster.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


def _write(run_dir: Path, name: str, frame: pl.DataFrame) -> int:
    path = run_dir / TABLES / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return path.stat().st_size


def _remove(run_dir: Path, name: str) -> None:
    path = run_dir / TABLES / f"{name}.parquet"
    if path.exists():
        path.unlink()
        log.info(f"removed {path.name} (no longer configured)")


def annotation_columns(pieces: pl.DataFrame) -> list[str]:
    """The piece annotations present (``masked_share``, ``doc_degree``, ``ref_overlap_*``)."""
    return [c for c in pieces.columns if c in PIECE_ANNOTATIONS or c.startswith("ref_overlap_")]


def hit_rows(hits: pl.DataFrame, id_col: str, piece_map: pl.DataFrame, pieces: pl.DataFrame,
             pairs: pl.DataFrame) -> pl.DataFrame:
    """Hit rows with the fields the hit filter can use (besides document metadata)."""
    side = {k: piece_map.filter(pl.col("side") == k).select(id_col, pl.col("piece_id").alias(f"piece{k}"))
            for k in (1, 2)}
    names = annotation_columns(pieces)
    rows = hits.join(side[1], on=id_col, how="left").join(side[2], on=id_col, how="left")
    for k in (1, 2):
        rows = rows.join(pieces.select(pl.col("piece_id").alias(f"piece{k}"), *[pl.col(n).alias(f"{n}{k}")
                                                                               for n in names],
                                       (pl.col("enc_end") - pl.col("enc_start")).cast(pl.Int32)
                                       .alias(f"piece_letters{k}")),
                         on=f"piece{k}", how="left")
    return rows.join(pairs.select("piece1", "piece2", "num_orig_links"), on=["piece1", "piece2"], how="left")


def kept_hits(run_dir: Path, hit_filter: str | None, view: str) -> tuple[pl.DataFrame, str]:
    """The hits of ``view`` that pass the hit filter (all of them without one), and their id column."""
    if view not in VIEWS:
        raise PostError(f"unknown view {view!r} (hsps, clipped or joined)")
    name, id_col = VIEWS[view]
    tables = run_dir / TABLES
    if not (tables / f"{name}.parquet").is_file():
        raise PostError(f"tables/{name}.parquet is missing; run `textblast post` (with units.clip for clipped, "
                        "join.enabled for joined)")
    hits = pl.read_parquet(tables / f"{name}.parquet")
    if not hit_filter:
        return hits, id_col
    rows = hit_rows(hits, id_col, pl.read_parquet(tables / "piece_map.parquet"),
                    pl.read_parquet(tables / "pieces.parquet"), pl.read_parquet(tables / "pairs.parquet"))
    documents = CorpusStore(run_dir).documents()
    kept = apply_hit_filter(rows.lazy(), hit_filter, documents).select(id_col).collect()
    return hits.join(kept, on=id_col, how="semi"), id_col


def post(run_dir: Path | str, filters: FilterOptions, units: UnitOptions, defrag: DefragOptions,
         workers: int | None = None, join: JoinOptions | None = None, annotate: AnnotateOptions | None = None,
         clustering: ClusterOptions | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser()
    workers = workers or default_workers()
    join = join or JoinOptions()
    annotate = annotate or AnnotateOptions()
    clustering = clustering or ClusterOptions()
    manifest = read_manifest(run_dir)
    merged = manifest.get("stages", {}).get("merge")
    hsps_path = run_dir / TABLES / "hsps.parquet"
    if not merged or not hsps_path.is_file():
        raise PostError(f"no merged hits in {run_dir}; run `textblast merge` (or `run`) first")
    if defrag.view not in VIEWS:
        raise PostError(f"defrag.view must be hsps, clipped or joined, not {defrag.view!r}")
    if defrag.view == "clipped" and not units.clip:
        raise PostError("defrag.view: clipped needs units.clip: true")
    if defrag.view == "joined" and not join.enabled:
        raise PostError("defrag.view: joined needs join.enabled: true")
    if clustering.filtered and not clustering.enabled:
        raise PostError("cluster.filtered needs cluster.enabled: true")
    if clustering.filtered and not filters.hit_filter:
        raise PostError("cluster.filtered needs filters.hit_filter")
    settings = {"units": asdict(units), "join": asdict(join), "defrag": asdict(defrag), "annotate": asdict(annotate),
                "hit_filter": filters.hit_filter, "cluster": asdict(clustering)}
    fingerprint = hashlib.sha256(json.dumps(
        [merged["fingerprint"], manifest["stages"]["prepare"]["fingerprint"], settings, _code_hash(), __version__],
        sort_keys=True).encode()).hexdigest()[:16]
    expected = (["pieces", "piece_map", "pairs", "unit_overlaps", "join_map"]
                + (["hsps_clipped"] if units.clip else []) + (["hsps_joined"] if join.enabled else [])
                + (["pairs_filtered"] if filters.hit_filter else [])
                + (["clusters", "cluster_summary"] if clustering.enabled else []))
    previous = manifest.get("stages", {}).get("post", {})
    if previous.get("fingerprint") == fingerprint and all(
        (run_dir / TABLES / f"{n}.parquet").is_file() for n in expected
    ):
        log.info("post-processed tables are up to date; nothing to do")
        return previous

    started = time.monotonic()
    store = CorpusStore(run_dir)
    documents = store.documents()
    hsps = pl.read_parquet(hsps_path)
    total = 6
    summary: dict[str, Any] = {"fingerprint": fingerprint, "settings": settings}
    log.info(f"post · {format_count(hsps.height)} hits · join {f'≤ {join.gap} letters' if join.enabled else 'off'} · "
             f"rule {defrag.rule} on {defrag.view} · clipping {'on' if units.clip else 'off'} · "
             f"hit filter {filters.hit_filter or 'none'} · "
             f"clusters {clustering.method if clustering.enabled else 'off'} · {workers} workers")
    sizes: dict[str, int] = {}

    with Stage(1, total, "clip", log=log, show_memory=True) as stage:
        if units.clip:
            clipped, clip_stats = clip_hits(run_dir, hsps, workers)
            size = _write(run_dir, "hsps_clipped", clipped)
            summary["clip"] = clip_stats | {"bytes": size}
            stage.summary = (f"{format_count(clip_stats['hits_cut'])} of {format_count(clip_stats['hits'])} hits cut "
                             f"→ {format_count(clip_stats['segments'])} segments · {clip_stats['dropped_one_sided']} "
                             f"one-sided segments dropped · {format_bytes(size)}")
        else:
            _remove(run_dir, "hsps_clipped")
            clipped = None
            stage.summary = "off (units.clip: false)"

    with Stage(2, total, "join", log=log, show_memory=True) as stage:
        if join.enabled:
            joined, join_map, stats = join_hits(hsps, join)
            sizes["hsps_joined"] = _write(run_dir, "hsps_joined", joined)
        else:
            join_map, _ = join_map_for(hsps, join)
            stats = join_stats(join_map)
            _remove(run_dir, "hsps_joined")
            joined = None
        stats["crossed_neighbours"] = crossed_neighbours(hsps, join.gap)
        sizes["join_map"] = _write(run_dir, "join_map", join_map)
        summary["join"] = stats
        groups = (f"{format_count(stats['hits_in_joints'])} hits continue a neighbour → "
                  f"{format_count(stats['joined_hits'])} groups")
        stage.summary = (f"{groups} (joined view built) " if join.enabled else
                         f"off: annotation only (join_map: {groups}) ") + (
                         f"· {format_count(stats['crossed_neighbours'])} crossed neighbours left apart")

    hits, id_col = {"hsps": (hsps, "hsp_id"), "clipped": (clipped, "clip_id"),
                    "joined": (joined, "joined_id")}[defrag.view]
    assert hits is not None
    with Stage(3, total, "defragment", f"{defrag.rule} on {defrag.view}", log=log, show_memory=True) as stage:
        pieces, piece_map, pairs = defragment(hits, id_col, defrag, workers)
        if defrag.rule != "etl":  # the rule chose its own spans: letters inside them
            enc_start, enc_end = encoded_range(store, pieces["doc"].to_numpy(), pieces["start"].to_numpy(),
                                               pieces["end"].to_numpy())
            pieces = pieces.with_columns(pl.Series("enc_start", enc_start), pl.Series("enc_end", enc_end))
        n_raw = piece_map["raw_piece_id"].n_unique()
        stage.summary = (f"{format_count(piece_map.height)} hit sides → {format_count(n_raw)} raw pieces → "
                         f"{format_count(pieces.height)} pieces · {format_count(pairs.height)} pairs")
        summary["defrag"] = {"hit_sides": piece_map.height, "raw_pieces": n_raw, "pieces": pieces.height,
                             "pairs": pairs.height}

    with Stage(4, total, "annotate", log=log, show_memory=True) as stage:
        share = masked_share(store, pieces["doc"].to_numpy(), pieces["enc_start"].to_numpy(),
                             pieces["enc_end"].to_numpy())
        pieces = pieces.with_columns(pl.Series("masked_share", share, dtype=pl.Float32),
                                     pl.Series("doc_degree", doc_degree(pieces, pairs), dtype=pl.Int32))
        refs = {}
        for name, expression in annotate.reference_sets.items():
            if not name.isidentifier():
                raise PostError(f"reference set name {name!r} must be a plain identifier (it names a column)")
            try:
                member = reference_set_members(documents, name, expression)
            except ValueError as error:
                raise PostError(str(error)) from error
            column = reference_overlap(pieces, hits, member)
            pieces = pieces.with_columns(pl.Series(f"ref_overlap_{name}", column, dtype=pl.Float32))
            refs[name] = {"documents": int(member.sum()), "pieces_touched": int((column > 0).sum())}
        overlaps = unit_overlaps(pieces, units_frame(store))
        multi = int((overlaps.group_by("piece_id").len()["len"] > 1).sum()) if overlaps.height else 0
        sizes |= {name: _write(run_dir, name, frame) for name, frame in
                  (("pieces", pieces), ("piece_map", piece_map), ("pairs", pairs), ("unit_overlaps", overlaps))}
        degree = pieces["doc_degree"]
        ref_text = "".join(f" · {n}: {format_count(r['pieces_touched'])} pieces touched" for n, r in refs.items())
        stage.summary = (f"{format_count(overlaps.height)} unit overlaps · {format_count(multi)} pieces span several "
                         f"units · {int((share > 0).sum())} pieces with masked letters · doc_degree median "
                         f"{degree.median() or 0:.0f}, max {degree.max() or 0}{ref_text}")
        summary["annotate"] = {"unit_overlaps": overlaps.height, "pieces_in_several_units": multi,
                               "pieces_with_masks": int((share > 0).sum()), "reference_sets": refs,
                               "doc_degree_median": float(degree.median() or 0), "doc_degree_max": int(
                                   degree.max() or 0)}

    with Stage(5, total, "hit filter", log=log, show_memory=True) as stage:
        filtered = None
        if filters.hit_filter:
            rows = hit_rows(hits, id_col, piece_map, pieces, pairs)
            kept = apply_hit_filter(rows.lazy(), filters.hit_filter, documents).select(hits.columns).collect()
            filtered = make_pairs(kept, id_col, piece_map).drop("pair_id").join(
                pairs.select("pair_id", "piece1", "piece2"), on=["piece1", "piece2"]
            ).select(pairs.columns)
            sizes["pairs_filtered"] = _write(run_dir, "pairs_filtered", filtered)
            stage.summary = (f"{format_count(kept.height)} of {format_count(hits.height)} hits · "
                             f"{format_count(filtered.height)} of {format_count(pairs.height)} pairs kept")
            summary["hit_filter"] = {"hits_kept": kept.height, "pairs_kept": filtered.height}
        else:
            _remove(run_dir, "pairs_filtered")
            stage.summary = "none (filters.hit_filter not set)"

    with Stage(6, total, "cluster", clustering.method if clustering.enabled else "", log=log,
               show_memory=True) as stage:
        if clustering.enabled:
            edges = filtered if clustering.filtered else pairs
            assert edges is not None
            clusters, cluster_summary, info = cluster(pieces, edges, clustering)
            if info and not info["settled"]:
                log.warning(f"chinese_whispers: labels still changing after {info['rounds']} rounds "
                            "(cluster.iterations); clusters are close to, but not exactly, the settled result")
            sizes["clusters"] = _write(run_dir, "clusters", clusters)
            sizes["cluster_summary"] = _write(run_dir, "cluster_summary", cluster_summary)
            largest = cluster_summary.row(0, named=True) if cluster_summary.height else {"n_pieces": 0,
                                                                                         "n_documents": 0}
            stage.summary = (f"{format_count(cluster_summary.height)} clusters of {format_count(clusters.height)} "
                             f"pieces · largest {format_count(largest['n_pieces'])} pieces in "
                             f"{largest['n_documents']} documents"
                             + (f" · {'settled' if info['settled'] else 'stopped'} after {info['rounds']} rounds"
                                if info else ""))
            summary["cluster"] = {"clusters": cluster_summary.height, "pieces": clusters.height,
                                  "largest_pieces": largest["n_pieces"],
                                  "largest_documents": largest["n_documents"]} | info
        else:
            _remove(run_dir, "clusters")
            _remove(run_dir, "cluster_summary")
            stage.summary = "off (cluster.enabled: false)"

    summary |= {"wall_seconds": round(time.monotonic() - started, 2), "peak_memory_bytes": peak_memory_bytes(),
                "bytes": sizes}
    manifest = read_manifest(run_dir)
    manifest.setdefault("stages", {})["post"] = summary
    manifest["schemas"] = SCHEMAS
    write_manifest(run_dir, manifest)
    append_event(run_dir, event="post_done", fingerprint=fingerprint, wall_seconds=summary["wall_seconds"])
    return summary
