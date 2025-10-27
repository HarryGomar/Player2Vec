from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from .constants import EMBED_COL, ID_COLS, NUMERIC_STATS
from .utils import to_season_label


@st.cache_data(show_spinner=False)
def load_players(path: str) -> pd.DataFrame:
    """Load and normalize the parquet into a tidy DataFrame."""
    df = pd.read_parquet(path)

    # Ensure numeric dtype for known stat columns
    numeric_cols = set(NUMERIC_STATS) | {"minutes", "events_count"}
    for col in numeric_cols.intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse embedding column
    if EMBED_COL not in df.columns:
        raise ValueError(f"Missing required column: {EMBED_COL}")

    def parse_vec(raw) -> np.ndarray:
        if isinstance(raw, np.ndarray):
            vec = raw.astype("float32")
        elif isinstance(raw, (list, tuple)):
            vec = np.asarray(raw, dtype="float32")
        else:
            serialized = str(raw)
            try:
                vec = np.asarray(json.loads(serialized), dtype="float32")
            except Exception:
                vec = np.asarray(eval(serialized), dtype="float32")  # legacy fallback
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    df[EMBED_COL] = df[EMBED_COL].apply(parse_vec)

    if "style_dim" in df.columns:
        df["style_dim"] = df["style_dim"].astype(int)
        mask = df[EMBED_COL].map(len) == df["style_dim"]
        df = df.loc[mask].copy()
    else:
        df["style_dim"] = df[EMBED_COL].map(len)

    # Drop rows missing key identifiers
    needed = ["ps_index", "player_name", "team_name", "season_id"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    if "season_id" in df.columns:
        df["season_code"] = df["season_id"]
        df["season_label"] = df["season_id"].apply(to_season_label)

    # Deduplicate ps_index if necessary
    if "ps_index" in df.columns and df["ps_index"].duplicated().any():
        df = df.drop_duplicates(subset=["ps_index"])

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_options(df: pd.DataFrame):
    teams = sorted(df.get("team_name", pd.Series(dtype=str)).dropna().unique().tolist())
    seasons = sorted(df.get("season_id", pd.Series(dtype=str)).dropna().unique().tolist())
    pos_mode = sorted(df.get("position_mode", pd.Series(dtype=str)).dropna().unique().tolist())
    pos_coarse = sorted(df.get("position_mode_coarse", pd.Series(dtype=str)).dropna().unique().tolist())
    return teams, seasons, pos_mode, pos_coarse


@st.cache_data(show_spinner=False)
def as_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack embedding column into a contiguous float32 matrix."""
    return np.stack(df[EMBED_COL].values).astype("float32")


@st.cache_data(show_spinner=False)
def subset(df: pd.DataFrame, idx: Iterable[int]) -> pd.DataFrame:
    return df.loc[idx].copy()