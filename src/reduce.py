from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

try:  # Optional hdbscan dependency
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

import umap
from sklearn.cluster import KMeans


@st.cache_resource(show_spinner=False)
def umap_embed(
    V: np.ndarray,
    n_neighbors: int = 40,
    min_dist: float = 0.15,
    densmap: bool = False,
    y: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="cosine",
        densmap=bool(densmap),
        random_state=int(random_state),
        n_components=2,
    )
    return reducer.fit_transform(V, y=y)


@st.cache_resource(show_spinner=False)
def cluster_points(
    Y: np.ndarray,
    method: str = "hdbscan",
    k: int = 12,
    min_cluster_size: int = 25,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    if method == "none":
        return np.full(len(Y), -1, dtype=int)

    if method == "kmeans" or not _HAS_HDBSCAN:
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
        return km.fit_predict(Y)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=min_samples)
    return clusterer.fit_predict(Y)