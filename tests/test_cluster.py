"""Clustering pieces: connected components, Chinese Whispers, plugins (no engine)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from textblast.cluster import ClusterError, ClusterOptions, chinese_whispers, cluster, components


def test_connected_components() -> None:
    assert components(7, np.array([0, 1, 3, 5]), np.array([1, 2, 4, 6])).tolist() == [0, 0, 0, 3, 3, 5, 5]
    assert components(6, np.array([5, 4, 3, 2, 1]), np.array([4, 3, 2, 1, 0])).tolist() == [0] * 6
    assert components(3, np.array([], dtype=np.int64), np.array([], dtype=np.int64)).tolist() == [0, 1, 2]


def two_communities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, w = [], [], []
    for base in (0, 5):
        for i in range(5):
            for j in range(i + 1, 5):
                a.append(base + i)
                b.append(base + j)
                w.append(10.0)
    a.append(4)
    b.append(5)
    w.append(1.0)  # one weak edge between the communities
    return np.array(a), np.array(b), np.array(w)


def test_chinese_whispers_splits_communities_and_is_deterministic() -> None:
    a, b, w = two_communities()
    labels, rounds, settled = chinese_whispers(10, a, b, w, seed=7)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1 and labels[0] != labels[5]
    assert settled and rounds < 50
    assert chinese_whispers(10, a, b, w, seed=7).label.tolist() == labels.tolist()
    assert not chinese_whispers(10, a, b, w, iterations=1, seed=7).settled  # stopped while labels still change
    assert len(set(components(10, a, b))) == 1  # components merge them


def test_cluster_tables_and_plugin() -> None:
    pieces = pl.DataFrame({"piece_id": [0, 1, 2, 3, 4], "doc": [0, 1, 1, 2, 3], "start": [0, 0, 500, 0, 0],
                           "end": [100, 110, 550, 40, 10]})
    pairs = pl.DataFrame({"piece1": [0, 2], "piece2": [1, 3], "doc1": [0, 1], "doc2": [1, 2],
                          "total_length": [100, 50], "num_orig_links": [1, 1], "score": [900, 450]})
    clusters, summary, info = cluster(pieces, pairs, ClusterOptions(method="components"))
    assert clusters["piece_id"].to_list() == [0, 1, 2, 3]  # piece 4 has no pair
    assert summary["n_pieces"].to_list() == [2, 2] and summary["n_documents"].to_list() == [2, 2]
    assert summary["piece_length_median"].to_list() == [105.0, 45.0]  # pieces 0 and 1; 2 and 3
    assert summary["total_length"].to_list() == [100, 50]  # the aligned letters of the pairs inside
    assert info == {}
    assert cluster(pieces, pairs, ClusterOptions(method="chinese_whispers"))[2] == {"rounds": 2, "settled": True}
    plugin, _, _ = cluster(pieces, pairs, ClusterOptions(method="tests.helpers:one_cluster"))
    assert plugin["cluster_id"].unique().to_list() == [0]
    with pytest.raises(ClusterError):
        cluster(pieces, pairs, ClusterOptions(method="spectral"))
    with pytest.raises(ClusterError):
        cluster(pieces, pairs, ClusterOptions(method="chinese_whispers", weight="length"))


def test_clustering_is_off_by_default_with_chinese_whispers_as_its_method() -> None:
    options = ClusterOptions()
    assert options.enabled is False and options.method == "chinese_whispers"
