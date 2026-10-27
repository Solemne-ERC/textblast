"""Clustering pieces through their pairs: an optional step of ``post``.

Methods:
- ``components``: connected components (label propagation with pointer jumping, numpy);
- ``chinese_whispers``: Chinese Whispers (Biemann 2006). Each piece takes the label with the largest summed edge
  weight among its neighbours. Pieces are updated in random batches (``batches`` per iteration, seeded), so a run is
  deterministic for a given seed; ties go to the smallest label. It stops when a round changes no label (on the
  corpus after 26–29 rounds) or after ``iterations`` rounds. Other seeds group borderline pieces differently (on the
  corpus, 5% of linked pairs), so cluster borders are not exact;
- a plugin by import path (``module:function``), called with ``(pieces, pairs)`` and returning a DataFrame with
  ``piece_id`` and ``cluster``.

Edge weights (``weight``) come from the pairs: ``total_length`` (aligned letters, the default, so long links count
more than short formulaic ones), ``num_orig_links``, ``score`` or ``none`` (all 1).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import polars as pl

METHODS = ("components", "chinese_whispers")
WEIGHTS = ("total_length", "num_orig_links", "score", "none")


@dataclass
class ClusterOptions:
    enabled: bool = False  # cluster the pieces (tables/clusters); off by default
    method: str = "chinese_whispers"  # chinese_whispers (default) | components | module:function
    weight: str = "total_length"  # chinese_whispers: total_length | num_orig_links | score | none
    iterations: int = 50  # chinese_whispers: the most rounds (it stops earlier when no label changes)
    batches: int = 10  # chinese_whispers: random batches per iteration
    seed: int = 0  # chinese_whispers
    filtered: bool = False  # cluster the pairs that pass the hit filter (pairs_filtered) instead of all pairs


class ClusterError(ValueError):
    """An unknown method or weight, or a plugin returning an invalid clustering."""


class Whispers(NamedTuple):
    label: np.ndarray  # label of each node
    rounds: int  # rounds run
    settled: bool  # the last round changed no label


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component label (the smallest node index) of each of ``n`` nodes, edges ``a``–``b``."""
    label = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return label
    a, b = a.astype(np.int64), b.astype(np.int64)
    while True:
        low = np.minimum(label[a], label[b])
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        while True:  # pointer jumping: each label points to its own label's label
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
        if np.array_equal(label, before):
            return label


def chinese_whispers(n: int, a: np.ndarray, b: np.ndarray, weight: np.ndarray, iterations: int = 20,
                     batches: int = 10, seed: int = 0) -> Whispers:
    """Chinese Whispers label of each of ``n`` nodes (edges ``a``–``b`` with ``weight``), deterministic for ``seed``."""
    label = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return Whispers(label, 0, True)
    src = np.concatenate([a, b]).astype(np.int64)
    dst = np.concatenate([b, a]).astype(np.int64)
    w = np.concatenate([weight, weight]).astype(np.float64)
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    rng = np.random.default_rng(seed)
    for rounds in range(1, iterations + 1):
        changed = 0
        batch_of = rng.integers(0, batches, size=n)
        for k in range(batches):
            edges = batch_of[src] == k
            if not edges.any():
                continue
            frame = pl.DataFrame({"node": src[edges], "label": label[dst[edges]], "w": w[edges]})
            best = (
                frame.group_by("node", "label").agg(pl.col("w").sum())
                .sort(["node", "w", "label"], descending=[False, True, False])
                .unique("node", keep="first")
            )
            nodes, new = best["node"].to_numpy(), best["label"].to_numpy()
            changed += int((label[nodes] != new).sum())
            label[nodes] = new
        if changed == 0:
            return Whispers(label, rounds, True)
    return Whispers(label, iterations, False)


def edge_weights(pairs: pl.DataFrame, weight: str) -> np.ndarray:
    if weight not in WEIGHTS:
        raise ClusterError(f"cluster.weight must be one of {', '.join(WEIGHTS)}, not {weight!r}")
    if weight == "none":
        return np.ones(pairs.height, dtype=np.float64)
    return pairs[weight].to_numpy().astype(np.float64)


def cluster(pieces: pl.DataFrame, pairs: pl.DataFrame,
            options: ClusterOptions) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """``(clusters, summary, info)``: ``piece_id → cluster_id`` for the pieces linked by ``pairs`` (clusters numbered
    by size, largest first); per cluster its number of pieces and of documents, the median piece length (characters)
    and the aligned letters of the pairs inside it (``total_length``), which rank borrowings above short phrases; and
    for Chinese Whispers the rounds run and whether it settled (``info``, empty for other methods)."""
    method = options.method
    info: dict[str, Any] = {}
    linked = pl.concat([pairs.select(pl.col("piece1").alias("piece_id")),
                        pairs.select(pl.col("piece2").alias("piece_id"))]).unique().sort("piece_id")
    ids = linked["piece_id"].to_numpy()
    a = np.searchsorted(ids, pairs["piece1"].to_numpy())
    b = np.searchsorted(ids, pairs["piece2"].to_numpy())
    if method == "components":
        label = components(len(ids), a, b)
    elif method == "chinese_whispers":
        label, rounds, settled = chinese_whispers(len(ids), a, b, edge_weights(pairs, options.weight),
                                                  options.iterations, options.batches, options.seed)
        info = {"rounds": rounds, "settled": settled}
    elif method and ":" in method:
        module, _, name = method.partition(":")
        try:
            func = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as error:
            raise ClusterError(f"cannot load cluster method {method!r}: {error}") from error
        result = func(pieces, pairs)
        if not isinstance(result, pl.DataFrame) or not {"piece_id", "cluster"} <= set(result.columns):
            raise ClusterError(f"cluster method {method!r} must return a DataFrame with piece_id and cluster")
        mapped = linked.join(result.select("piece_id", "cluster"), on="piece_id", how="left")
        if mapped["cluster"].null_count():
            raise ClusterError(f"cluster method {method!r} must assign every linked piece")
        label = mapped["cluster"].rank("dense").to_numpy().astype(np.int64)
    else:
        raise ClusterError(f"cluster.method must be one of {', '.join(METHODS)} or module:function, not {method!r}")
    frame = linked.with_columns(pl.Series("_label", label)).join(
        pieces.select("piece_id", "doc", (pl.col("end") - pl.col("start")).alias("_length")), on="piece_id", how="left")
    same = label[a] == label[b]  # pairs inside a cluster
    inside = pl.DataFrame({"_label": label[a][same], "w": pairs["total_length"].to_numpy()[same]}).group_by(
        "_label").agg(pl.col("w").sum().alias("total_length"))
    summary = (
        frame.group_by("_label").agg(pl.len().alias("n_pieces"), pl.col("doc").n_unique().alias("n_documents"),
                                     pl.col("_length").median().alias("piece_length_median"))
        .join(inside, on="_label", how="left")
        .sort(["n_documents", "n_pieces", "_label"], descending=[True, True, False])
        .with_row_index("cluster_id")
    )
    clusters = frame.join(summary.select("_label", "cluster_id"), on="_label").select(
        pl.col("piece_id").cast(pl.Int64), pl.col("cluster_id").cast(pl.Int64)).sort("piece_id")
    summary = summary.select(pl.col("cluster_id").cast(pl.Int64), pl.col("n_pieces").cast(pl.Int32),
                             pl.col("n_documents").cast(pl.Int32),
                             pl.col("piece_length_median").cast(pl.Float64),
                             pl.col("total_length").fill_null(0).cast(pl.Int64))
    return clusters, summary, info
