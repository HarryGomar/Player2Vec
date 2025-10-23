from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def rank_percentile(x: pd.Series, ascending: bool = False) -> pd.Series:
    r = x.rank(method="average", ascending=ascending)
    return 1.0 - (r - 1) / (len(x) - 1 + 1e-9)


def aggregate_stats(df: pd.DataFrame, stat_prefs):
    """Combine selected stat z-scores into a single percentile score."""
    if not stat_prefs:
        return pd.Series(0.0, index=df.index)

    scores = []
    weights = []
    for col, weight, direction in stat_prefs:
        if col not in df:
            continue
        series = df[col].astype(float)
        if series.nunique(dropna=True) <= 1:
            continue
        z = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
        if direction == "min":
            z = -z
        clipped = np.clip(z, -4, 4)
        scores.append(norm.cdf(clipped))
        weights.append(float(weight))

    if not scores:
        return pd.Series(0.0, index=df.index)

    W = np.array(weights, dtype=float)
    W = W / (W.sum() + 1e-9)
    stacked = np.stack(scores, axis=1)
    combined = (stacked * W[None, :]).sum(axis=1)
    return pd.Series(combined, index=df.index)


def fit_score(df: pd.DataFrame, sim_col: str, stat_prefs, w_sim: float = 0.6, w_stats: float = 0.4) -> pd.Series:
    S_sim = rank_percentile(df[sim_col], ascending=False)
    S_stats = aggregate_stats(df, stat_prefs)
    w_sim = float(w_sim)
    w_stats = float(w_stats)
    if (w_sim + w_stats) == 0:
        return pd.Series(0.0, index=df.index)
    return w_sim * S_sim + w_stats * S_stats