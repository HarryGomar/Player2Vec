from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.constants import NUMERIC_STATS
from src.data import load_players
from src.filters import apply_filters, render_filter_sidebar
from src.sim import (
    as_matrix,
    build_similarity_table,
    cosine_sim,
    get_query_vec_player,
    get_query_vec_role,
)

st.title("🔎 Player / Role Search")

path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)

filter_selections = render_filter_sidebar(df)
candidates = apply_filters(df, filter_selections)

st.caption(f"Candidates after filters: **{len(candidates):,}**")

if candidates.empty:
    st.warning("No rows left after applying filters. Relax them to search.")
    st.stop()

mode = st.radio("Query mode", ["Player-based", "Role-based"], horizontal=True)

query_vector = None
query_description = ""

if mode == "Player-based":
    names = sorted(df["player_name"].dropna().unique())
    player = st.selectbox("Query player", names)
    seasons_opt = st.multiselect("Limit to seasons", sorted(df["season_id"].dropna().unique()))
    minutes_weighted = st.checkbox("Minutes-weighted average", value=True)
    if st.button("Build player query", type="primary", use_container_width=True):
        query_vector = get_query_vec_player(df, player_name=player, seasons=seasons_opt, minutes_weighted=minutes_weighted)
        query_description = f"Player-based: {player}"
else:
    c1, c2 = st.columns(2)
    with c1:
        pos_mode = st.selectbox("Position (fine)", options=[None] + sorted(df.get("position_mode", pd.Series(dtype=str)).dropna().unique().tolist()))
        team = st.selectbox("Team (optional)", options=[None] + sorted(df.get("team_name", pd.Series(dtype=str)).dropna().unique().tolist()))
    with c2:
        pos_coarse = st.selectbox("Position (coarse)", options=[None] + sorted(df.get("position_mode_coarse", pd.Series(dtype=str)).dropna().unique().tolist()))
        seasons_role = st.multiselect("Season(s) (optional)", sorted(df.get("season_id", pd.Series(dtype=str)).dropna().unique().tolist()))
    min_minutes_role = st.number_input("Min minutes in role pool", min_value=0, value=300, step=30)
    minutes_weighted = st.checkbox("Minutes-weighted centroid", value=True, key="role_minutes_weighted")
    if st.button("Build role centroid", type="primary", use_container_width=True):
        try:
            query_vector, pool = get_query_vec_role(
                df,
                pos_mode=pos_mode,
                pos_coarse=pos_coarse,
                team=team,
                seasons=seasons_role,
                min_minutes=int(min_minutes_role),
                minutes_weighted=minutes_weighted,
            )
            query_description = "Role-based centroid"
            st.success(f"Role pool size: {len(pool):,}")
        except ValueError as err:
            st.error(str(err))

if query_vector is None:
    st.info("Build a query vector above to score the filtered candidates.")
    st.stop()

st.success(f"Query ready → {query_description}")

similarity_df = build_similarity_table(candidates, query_vector, sim_col="sim")

add_stats = st.multiselect("Add stat columns", [c for c in NUMERIC_STATS if c in similarity_df.columns], default=[c for c in ("goals_per90", "xg_per90", "xa_proxy") if c in similarity_df.columns])
result_cols = [
    col
    for col in [
        "ps_index",
        "player_name",
        "season_id",
        "team_name",
        "minutes",
        "position_mode",
        "position_mode_coarse",
        "sim",
    ]
    if col in similarity_df.columns
]
result_cols += add_stats

top_n = st.slider("Show top N", min_value=10, max_value=min(500, len(similarity_df)), value=min(50, len(similarity_df)))

display_df = similarity_df[result_cols].sort_values("sim", ascending=False).head(top_n).reset_index(drop=True)
st.dataframe(display_df, use_container_width=True)

st.download_button(
    "Download similarity results (CSV)",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="similarity_results.csv",
)

st.session_state["last_query_vec"] = query_vector
st.session_state["last_candidates_df"] = candidates
st.session_state["last_sim_results"] = similarity_df