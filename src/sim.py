from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .constants import EMBED_COL


def as_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df[EMBED_COL].values).astype("float32")


def cosine_sim(V: np.ndarray, q: np.ndarray) -> np.ndarray:
    # Embeddings are assumed L2 normalised
    return V @ q.astype("float32")


def get_query_vec_player(
    df: pd.DataFrame,
    player_name: str,
    seasons: Iterable | None = None,
    minutes_weighted: bool = True,
) -> np.ndarray:
    rows = df[df["player_name"].eq(player_name)]
    season_field = "season_label" if "season_label" in df.columns else "season_id"
    if seasons is not None and len(list(seasons)) > 0:
        rows = rows[rows[season_field].isin(list(seasons))]
    if rows.empty:
        raise ValueError("No rows match the selected player/seasons")

    V = as_matrix(rows)
    if minutes_weighted and "minutes" in rows:
        weights = rows["minutes"].to_numpy(dtype="float32")
        weights = weights / (weights.sum() + 1e-9)
        vec = (V * weights[:, None]).sum(axis=0)
    else:
        vec = V.mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def get_query_vec_role(
    df: pd.DataFrame,
    pos_mode: str | None = None,
    pos_coarse: str | None = None,
    team: str | None = None,
    seasons: Iterable | None = None,
    min_minutes: int = 300,
    minutes_weighted: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    mask = pd.Series(True, index=df.index)
    mask &= df["minutes"] >= float(min_minutes)
    season_field = "season_label" if "season_label" in df.columns else "season_id"
    if seasons:
        mask &= df[season_field].isin(list(seasons))
    if team:
        mask &= df["team_name"].eq(team)
    if pos_mode:
        mask &= df["position_mode"].eq(pos_mode)
    if pos_coarse:
        mask &= df["position_mode_coarse"].eq(pos_coarse)

    pool = df.loc[mask]
    if pool.empty:
        raise ValueError("No rows match the selected role filters")

    V = as_matrix(pool)
    if minutes_weighted and "minutes" in pool:
        weights = pool["minutes"].to_numpy(dtype="float32")
        weights = weights / (weights.sum() + 1e-9)
        vec = (V * weights[:, None]).sum(axis=0)
    else:
        vec = V.mean(axis=0)
    norm = np.linalg.norm(vec)
    vec = vec / (norm + 1e-9)
    return vec.astype("float32"), pool


def build_similarity_table(df: pd.DataFrame, query_vec: np.ndarray, sim_col: str = "sim") -> pd.DataFrame:
    V = as_matrix(df)
    scores = cosine_sim(V, query_vec)
    result = df.copy()
    result[sim_col] = scores
    return result