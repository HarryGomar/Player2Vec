from __future__ import annotations

import math
from typing import Dict, List, Tuple, TypedDict

import pandas as pd
import streamlit as st

from .constants import DEFAULT_MIN_MINUTES, NUMERIC_STATS

FILTER_PREFIX = "filters_shared"


class FilterSelections(TypedDict, total=False):
    teams: List[str]
    seasons: List[str]
    pos_mode: List[str]
    pos_coarse: List[str]
    min_minutes: float
    stat_ranges: Dict[str, Tuple[float, float]]


def _initialise_selection(state_key: str, options: List[str]) -> List[str]:
    # If the key is not present in session_state, initialise to the full options
    # copy (this is the default behaviour before the user interacts). However,
    # if the key is present (including an empty list because the user cleared
    # the selection), respect the user's choice and do NOT auto-reset to all
    # options. Also filter out any stale values that are no longer valid.
    current = st.session_state.get(state_key)
    if current is None:
        current = options.copy()
    else:
        # keep user intent: if they cleared (empty list), preserve it
        # but remove any values that are no longer valid options
        current = [item for item in current if item in options]

    st.session_state[state_key] = current
    return current


def _reset_filter_state():
    for key in list(st.session_state.keys()):
        if key.startswith(FILTER_PREFIX):
            del st.session_state[key]
    st.experimental_rerun()


def render_filter_sidebar(df: pd.DataFrame) -> FilterSelections:
    st.sidebar.header("Filters")

    teams = sorted(df.get("team_name", pd.Series(dtype=str)).dropna().unique().tolist())
    seasons = sorted(df.get("season_id", pd.Series(dtype=str)).dropna().unique().tolist())
    pos_mode = sorted(df.get("position_mode", pd.Series(dtype=str)).dropna().unique().tolist())
    pos_coarse = sorted(df.get("position_mode_coarse", pd.Series(dtype=str)).dropna().unique().tolist())

    teams_key = f"{FILTER_PREFIX}_teams"
    seasons_key = f"{FILTER_PREFIX}_seasons"
    pos_mode_key = f"{FILTER_PREFIX}_posmode"
    pos_coarse_key = f"{FILTER_PREFIX}_poscoarse"

    default_teams = _initialise_selection(teams_key, teams)
    default_seasons = _initialise_selection(seasons_key, seasons)
    default_pos_mode = _initialise_selection(pos_mode_key, pos_mode)
    default_pos_coarse = _initialise_selection(pos_coarse_key, pos_coarse)

    # Use the session_state key to control the widget's value. Do NOT pass
    # `default`/`value` when the same key is already present in
    # `st.session_state` because Streamlit will warn. We initialised the
    # keys above with `_initialise_selection` so relying on the `key`
    # argument is sufficient and preserves user actions (including empty
    # selections).
    sel_teams = st.sidebar.multiselect("Team", options=teams, key=teams_key)
    sel_seasons = st.sidebar.multiselect("Season(s)", options=seasons, key=seasons_key)
    sel_pos_mode = st.sidebar.multiselect("Position (fine)", options=pos_mode, key=pos_mode_key)
    sel_pos_coarse = st.sidebar.multiselect("Position (coarse)", options=pos_coarse, key=pos_coarse_key)

    st.sidebar.caption("Tip: leave selections untouched to include everyone. Use Reset to start over.")

    minutes_series = df.get("minutes", pd.Series([0.0], dtype="float32"))
    minutes_min = float(minutes_series.min() if not minutes_series.empty else 0.0)
    minutes_max = float(minutes_series.max() if not minutes_series.empty else 0.0)
    if math.isnan(minutes_min):
        minutes_min = 0.0
    if math.isnan(minutes_max):
        minutes_max = minutes_min + 1.0
    if minutes_max <= minutes_min:
        minutes_max = minutes_min + 1.0
    default_min = min(DEFAULT_MIN_MINUTES, minutes_max)

    min_key = f"{FILTER_PREFIX}_minmin"
    if min_key not in st.session_state:
        st.session_state[min_key] = float(default_min)
    # The slider value is managed via st.session_state[min_key], which was
    # initialised above if missing. Passing `value` together with `key`
    # triggers Streamlit warnings, so we only pass `key` and range limits.
    min_minutes = st.sidebar.slider(
        "Min minutes",
        min_value=float(minutes_min),
        max_value=float(minutes_max),
        step=10.0,
        key=min_key,
    )

    st.sidebar.markdown("---")
    controls_col1, controls_col2 = st.sidebar.columns([3, 1])
    with controls_col1:
        st.caption("Optional stat filters")
    with controls_col2:
        if st.button("Reset", key=f"{FILTER_PREFIX}_reset"):
            _reset_filter_state()

    stat_options = [c for c in NUMERIC_STATS if c in df.columns]
    stats_key = f"{FILTER_PREFIX}_statopts"
    if stats_key not in st.session_state:
        st.session_state[stats_key] = []
    selected_stats = st.sidebar.multiselect(
        "Stats to constrain",
        options=stat_options,
        key=stats_key,
    )

    stat_ranges: Dict[str, Tuple[float, float]] = {}
    for col in selected_stats:
        series = df[col].dropna()
        if series.empty:
            continue
        lo, hi = float(series.min()), float(series.max())
        if lo == hi:
            continue
        range_key = f"{FILTER_PREFIX}_range_{col}"
        prev = st.session_state.get(range_key, (lo, hi))
        clamped = (max(lo, min(prev[0], hi)), min(hi, max(prev[1], lo)))
        st.session_state[range_key] = clamped
        # range slider value is managed via session_state[range_key]
        stat_ranges[col] = st.sidebar.slider(
            f"{col} range",
            min_value=lo,
            max_value=hi,
            key=range_key,
        )

    return FilterSelections(
        teams=sel_teams,
        seasons=sel_seasons,
        pos_mode=sel_pos_mode,
        pos_coarse=sel_pos_coarse,
        min_minutes=float(min_minutes),
        stat_ranges=stat_ranges,
    )


def apply_filters(df: pd.DataFrame, selections: FilterSelections) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    teams = selections.get("teams", []) or []
    team_col = df.get("team_name")
    if teams and team_col is not None:
        available = team_col.dropna().unique()
        if len(teams) < len(available):
            mask &= team_col.isin(teams)

    seasons = selections.get("seasons", []) or []
    season_col = df.get("season_id")
    if seasons and season_col is not None:
        available = season_col.dropna().unique()
        if len(seasons) < len(available):
            mask &= season_col.isin(seasons)

    pos_mode = selections.get("pos_mode", []) or []
    pos_mode_col = df.get("position_mode")
    if pos_mode and pos_mode_col is not None:
        available = pos_mode_col.dropna().unique()
        if len(pos_mode) < len(available):
            mask &= pos_mode_col.isin(pos_mode)

    pos_coarse = selections.get("pos_coarse", []) or []
    pos_coarse_col = df.get("position_mode_coarse")
    if pos_coarse and pos_coarse_col is not None:
        available = pos_coarse_col.dropna().unique()
        if len(pos_coarse) < len(available):
            mask &= pos_coarse_col.isin(pos_coarse)

    min_minutes = selections.get("min_minutes")
    if min_minutes is not None and "minutes" in df:
        mask &= df["minutes"] >= float(min_minutes)

    stat_ranges = selections.get("stat_ranges", {})
    for col, (lo, hi) in stat_ranges.items():
        if col in df:
            mask &= df[col].between(lo, hi)

    return df.loc[mask].copy()
