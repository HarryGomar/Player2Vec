from __future__ import annotations

import os
import sys

import streamlit as st

# Allow importing from src/
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.constants import CATEGORICAL_DIMENSIONS, DEFAULT_HOVER_COLUMNS, NUMERIC_STATS
from src.data import as_matrix, load_players
from src.filters import apply_filters, render_filter_sidebar
from src.plotting import plot_umap
from src.reduce import cluster_points, umap_embed

st.title("🗺️ UMAP Explorer")

# Load data
path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)

# Filters
filter_selections = render_filter_sidebar(df)
filtered = apply_filters(df, filter_selections)

st.caption(f"Filtered rows: **{len(filtered):,}** / {len(df):,}")
if len(filtered) < 5:
    st.warning("Very few rows after filtering; UMAP may be unstable. Consider loosening filters.")

available_numeric = [c for c in NUMERIC_STATS if c in filtered.columns]
available_color = ["cluster", *[c for c in CATEGORICAL_DIMENSIONS if c in filtered.columns], "minutes", *available_numeric]
available_size = ["minutes", *available_numeric]

with st.expander("UMAP Parameters", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        n_neighbors = st.slider("n_neighbors", min_value=5, max_value=100, value=40, step=5)
    with c2:
        min_dist = st.slider("min_dist", min_value=0.0, max_value=0.99, value=0.15, step=0.01)
    with c3:
        densmap = st.checkbox("Use densMAP", value=False)
    with c4:
        supervise_col = st.selectbox(
            "Supervise by",
            options=["(none)"] + [c for c in ["position_mode", "position_mode_coarse", "team_name"] if c in filtered.columns],
            index=0,
        )

with st.expander("Clustering", expanded=True):
    cluster_method = st.selectbox("Method", options=["hdbscan", "kmeans", "none"], index=0 if len(filtered) >= 25 else 2)
    if cluster_method == "kmeans":
        cluster_k = st.slider("k (KMeans)", min_value=2, max_value=30, value=12)
        cluster_min_size = None
        cluster_min_samples = None
    elif cluster_method == "hdbscan":
        cluster_k = None
        max_cluster_size = max(5, len(filtered))
        default_cluster_size = max(5, min(25, len(filtered)))
        cluster_min_size = st.slider(
            "min_cluster_size",
            min_value=2,
            max_value=max_cluster_size,
            value=default_cluster_size,
        )
        cluster_min_samples = st.number_input("min_samples", min_value=1, value=min(10, max_cluster_size))
    else:
        cluster_k = cluster_min_size = cluster_min_samples = None

with st.expander("Plot Controls", expanded=True):
    color_default_index = 0 if "cluster" in available_color else 1
    color_by = st.selectbox("Colour", options=available_color, index=color_default_index)
    size_choice = st.selectbox("Marker size", options=["(none)"] + available_size, index=0)
    size_by = None if size_choice == "(none)" else size_choice

with st.expander("Labels", expanded=False):
    stats_to_label = st.multiselect("Label stats", options=available_numeric, help="We'll label players that sit in the top/bottom ranks for these stats.")
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.number_input("Top K", min_value=0, max_value=200, value=10, step=1)
    with c2:
        bottom_k = st.number_input("Bottom K", min_value=0, max_value=200, value=0, step=1)
    labels_conf = {
        "stats": stats_to_label,
        "top_k": int(top_k),
        "bottom_k": int(bottom_k),
    }

with st.expander("Highlights", expanded=False):
    highlight_names_default = st.session_state.get("umap_highlights", [])
    highlight_names = st.multiselect("Player(s) to highlight", sorted(filtered["player_name"].dropna().unique()), default=highlight_names_default)
    connect_highlights = st.checkbox("Connect highlighted seasons", value=True)

with st.expander("Hover data", expanded=False):
    hover_defaults = [c for c in DEFAULT_HOVER_COLUMNS if c in filtered.columns]
    hover_cols = st.multiselect("Columns to show", options=list(filtered.columns), default=hover_defaults)

compute = st.button("Compute / Refresh UMAP", type="primary", use_container_width=True)

if not compute:
    st.info("Set the filters and parameters, then click **Compute / Refresh UMAP**.")
    st.stop()

if len(filtered) < 2:
    st.error("Need at least two rows after filtering to compute UMAP.")
    st.stop()

with st.spinner("Running UMAP..."):
    V = as_matrix(filtered)
    supervision = None
    if supervise_col != "(none)" and supervise_col in filtered.columns:
        supervision = filtered[supervise_col]
    try:
        coords = umap_embed(V, n_neighbors=n_neighbors, min_dist=min_dist, densmap=densmap, y=supervision)
    except ImportError as e:
        st.error(
            "UMAP is not installed in this environment. To enable UMAP embedding, install `umap-learn`:\n\n"
            "pip: `pip install umap-learn`\n\n"
            "conda: `conda install -c conda-forge umap-learn`\n\n"
            "After installing, restart the app."
        )
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while running UMAP: {type(e).__name__}: {e}")
        st.stop()

with st.spinner("Clustering points..."):
    if cluster_method == "kmeans":
        labels = cluster_points(coords, method="kmeans", k=int(cluster_k or 8))
    elif cluster_method == "hdbscan":
        labels = cluster_points(
            coords,
            method="hdbscan",
            min_cluster_size=int(cluster_min_size or 15),
            min_samples=int(cluster_min_samples or 5),
        )
    else:
        labels = cluster_points(coords, method="none")

plot_df = filtered.copy().reset_index(drop=True)
plot_df["cluster"] = labels

fig = plot_umap(
    plot_df,
    coords,
    color=color_by,
    size=size_by,
    hover_cols=hover_cols,
    labels_conf=labels_conf,
    highlights=highlight_names,
    connect_highlights=connect_highlights,
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

download_df = plot_df[["ps_index", "player_name", "season_id", "team_name", "minutes", "cluster"]].copy()
download_df["umap_x"], download_df["umap_y"] = coords[:, 0], coords[:, 1]
st.download_button(
    "Download UMAP CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    mime="text/csv",
    file_name="umap_coordinates.csv",
)