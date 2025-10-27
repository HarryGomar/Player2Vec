from __future__ import annotations

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.constants import DEFAULT_HOVER_COLUMNS, NUMERIC_STATS
from src.data import as_matrix, load_players
from src.filters import apply_filters, render_filter_sidebar
from src.plotting import similarity_scatter
from src.sim import cosine_sim, get_query_vec_player

st.title("📈 Similarity Plot")

path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)
filter_selections = render_filter_sidebar(df)
candidates = apply_filters(df, filter_selections)

st.caption(f"Filtered rows: **{len(candidates):,}**")

st.markdown("Use the last query vector from **Player / Role Search** or build a quick one here.")
use_last = st.toggle("Use last query", value="last_query_vec" in st.session_state)

query_vec = None
if use_last:
    query_vec = st.session_state.get("last_query_vec")
else:
    names = sorted(df["player_name"].dropna().unique())
    quick_player = st.selectbox("Quick query player", names)
    if st.button("Build quick query", use_container_width=True):
        query_vec = get_query_vec_player(df, player_name=quick_player)

if query_vec is None:
    st.info("Build or reuse a query vector, then configure the plot.")
    st.stop()

st.session_state["last_query_vec"] = query_vec

V = as_matrix(candidates)
plot_df = candidates.copy()
plot_df["sim"] = cosine_sim(V, query_vec)

available_stats = [c for c in NUMERIC_STATS if c in plot_df.columns]

y_col = st.selectbox("Y-axis metric", options=available_stats, index=0 if available_stats else None)
if y_col is None:
    st.error("No numeric stats available in the dataset to plot.")
    st.stop()

color_options = ["team_name", "season_label", "season_id", "position_mode", "position_mode_coarse", "minutes", *available_stats]
color_by = st.selectbox("Colour by", options=[c for c in color_options if c in plot_df.columns], index=0)

size_options = ["(none)", "minutes", *available_stats]
size_choice = st.selectbox("Size by", options=[c for c in size_options if c == "(none)" or c in plot_df.columns], index=0)
size_by = None if size_choice == "(none)" else size_choice

with st.expander("Hover data", expanded=False):
    hover_defaults = [c for c in (DEFAULT_HOVER_COLUMNS + ["sim", y_col]) if c in plot_df.columns]
    hover_cols = st.multiselect("Columns", options=list(plot_df.columns), default=hover_defaults)

with st.expander("Highlight players", expanded=False):
    highlight_players = st.multiselect("Highlight", sorted(plot_df["player_name"].dropna().unique()), default=st.session_state.get("umap_highlights", []))

fig = similarity_scatter(
    plot_df,
    sim_col="sim",
    y_col=y_col,
    color=color_by,
    size=size_by,
    hover_cols=hover_cols,
    highlight_players=highlight_players,
)

sim_min = float(plot_df["sim"].min())
sim_max = float(plot_df["sim"].max())
if abs(sim_max - sim_min) < 1e-6:
    threshold = sim_min
    st.warning("Similarity scores are identical across points; threshold controls disabled.")
else:
    threshold = st.slider(
        "Similarity threshold",
        min_value=sim_min,
        max_value=sim_max,
        value=float(plot_df["sim"].quantile(0.8)),
        step=0.01,
    )
    fig.add_vline(x=threshold, line_width=2, line_dash="dash", line_color="#444", annotation_text=f"thr={threshold:.2f}")

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

if st.checkbox("Show table for points above threshold"):
    cols = ["player_name", "team_name", "season_label", "season_id", "sim", y_col]
    cols += [c for c in hover_cols if c not in cols]
    st.dataframe(plot_df[plot_df["sim"] >= threshold][cols].sort_values("sim", ascending=False), use_container_width=True)