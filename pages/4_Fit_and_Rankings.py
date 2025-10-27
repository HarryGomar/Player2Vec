from __future__ import annotations

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.constants import NUMERIC_STATS
from src.data import as_matrix, load_players
from src.filters import apply_filters, render_filter_sidebar
from src.scoring import fit_score
from src.sim import cosine_sim

st.title("🏅 Fit Score & Rankings")

path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)
filter_selections = render_filter_sidebar(df)
candidates = apply_filters(df, filter_selections)

st.caption(f"Filtered rows: **{len(candidates):,}**")

query_vec = st.session_state.get("last_query_vec")
if query_vec is None:
    st.warning("No query vector found. Build one on the 'Player / Role Search' page first.")
    st.stop()

V = as_matrix(candidates)
rank_df = candidates.copy()
rank_df["sim"] = cosine_sim(V, query_vec)

if rank_df.empty:
    st.warning("No candidates available after filtering.")
    st.stop()

st.subheader("Stat Preferences")
available_stats = [c for c in NUMERIC_STATS if c in rank_df.columns]
default_stats = [c for c in ("goals_per90", "xg_per90", "xa_proxy") if c in available_stats]
selected_stats = st.multiselect("Stats to include", options=available_stats, default=default_stats)

stat_prefs = []
for stat in selected_stats:
    c1, c2, c3 = st.columns([3, 1, 2])
    with c1:
        st.write(f"**{stat}**")
    with c2:
        direction = st.selectbox("Direction", options=["max", "min"], key=f"dir_{stat}")
    with c3:
        weight = st.slider("Weight", min_value=0.0, max_value=1.0, value=0.33, key=f"wt_{stat}")
    stat_prefs.append((stat, weight, direction))

w_sim = st.slider("Similarity weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
w_stats = 1.0 - w_sim
st.markdown(f"Stats weight automatically set to **{w_stats:.2f}**.")

rank_df["Fit"] = fit_score(rank_df, sim_col="sim", stat_prefs=stat_prefs, w_sim=w_sim, w_stats=w_stats)
rank_df = rank_df.sort_values(["Fit", "sim"], ascending=[False, False])

columns_to_show = [
    col
    for col in [
        "ps_index",
        "player_name",
    "season_label",
        "season_id",
        "team_name",
        "minutes",
        "position_mode",
        "position_mode_coarse",
        "sim",
        "Fit",
        *selected_stats,
    ]
    if col in rank_df.columns
]

top_n = st.slider(
    "Display top N",
    min_value=1,
    max_value=max(1, min(500, len(rank_df))),
    value=min(50, len(rank_df)),
)
st.dataframe(rank_df.head(top_n)[columns_to_show].reset_index(drop=True), use_container_width=True)

st.download_button(
    "Download rankings (CSV)",
    data=rank_df[columns_to_show].to_csv(index=False).encode("utf-8"),
    file_name="fit_rankings.csv",
)

push = st.number_input(
    "Store top N names for UMAP highlights",
    min_value=1,
    max_value=max(1, len(rank_df)),
    value=min(25, len(rank_df)),
)
if st.button("Store in session"):
    st.session_state["umap_highlights"] = rank_df.head(int(push))["player_name"].tolist()
    st.success(f"Stored {int(push)} player names for UMAP highlighting.")