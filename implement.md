Below is the complete multi‑file codebase. Use the folder `Player2Vec/`, place each file at the indicated path, install requirements, and run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## requirements.txt

```txt
streamlit>=1.37
pandas>=2.1
pyarrow>=15
numpy>=1.26
plotly>=5.22
scikit-learn>=1.4
umap-learn>=0.5.6
hdbscan>=0.8.40
scipy>=1.13
joblib>=1.4
```

---

## app.py (project root)

```python
from __future__ import annotations
import streamlit as st
import os

st.set_page_config(
    page_title="Player Style Explorer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ Player Style Explorer")

st.markdown(
    """
    **Welcome!** Use the pages on the left to:

    1. Explore UMAP style maps
    2. Build player/role similarity searches
    3. Visualize similarity vs. chosen metrics
    4. Rank candidates via a tunable **Fit Score**

    👉 Start on **UMAP Explorer** after selecting your data file.
    """
)

# --- Global data selection (sidebar) ---
with st.sidebar:
    st.header("Data Source")
    st.caption("Expected file: `players_clean.parquet`")

    default_path = os.path.join("data", "players_clean.parquet")
    data_path = st.text_input("Parquet path", value=default_path)
    uploaded = st.file_uploader("…or upload parquet", type=["parquet"])
    if uploaded is not None:
        # Store upload to a temporary file so all pages can read it
        os.makedirs(".uploads", exist_ok=True)
        tmp_path = os.path.join(".uploads", "players_clean_uploaded.parquet")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        data_path = tmp_path

    st.session_state.setdefault("data_path", data_path)
    st.session_state["data_path"] = data_path

st.info("Use the sidebar to set the data file. Then open a page from the left navigation.")
```

---

## src/constants.py

```python
NUMERIC_STATS = [
    "shots_per90","goals_per90","xg_total","xg_per90","passes_per90","carries_per90","dribbles_per90",
    "interceptions_per90","blocks_per90","clearances_per90","duels_per90","recoveries_per90","pass_len_avg",
    "pass_len_p25","pass_len_p50","pass_len_p75","pass_comp_rate","pass_expected_success_mean","xa_proxy",
    "obv_total_net_sum","obv_for_net_sum","obv_against_net_sum","obv_total_net_per90","obv_for_net_per90",
    "obv_against_net_per90","gk_actions","gk_actions_per90","gk_success_total","gk_success_rate",
    "gk_success_in_play","gk_success_out","gk_in_play_rate","gk_out_rate","gk_saved_to_post",
    "gk_saved_off_target","gk_punched_out",
]

ID_COLS = [
    "ps_index","player_id","player_name","season_id","team_id","team_season_id","team_name",
    "position_mode","position_mode_coarse","pos_group","minutes","events_count"
]

DEFAULT_MIN_MINUTES = 300
```

---

## src/data.py

```python
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Iterable
from .constants import NUMERIC_STATS, ID_COLS

@st.cache_data(show_spinner=False)
def load_players(path: str) -> pd.DataFrame:
    """Load and validate the parquet into a normalized DataFrame.
    Ensures `z_style_unit` exists as np.ndarray L2-normalized and `style_dim` matches length.
    """
    df = pd.read_parquet(path)

    # Ensure numeric dtypes
    numeric_cols = set(NUMERIC_STATS) | {"minutes","events_count"}
    for c in df.columns:
        if c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse embedding column
    def parse_vec(x):
        if isinstance(x, np.ndarray):
            v = x.astype("float32")
        elif isinstance(x, (list, tuple)):
            v = np.asarray(x, dtype="float32")
        else:
            # string-encoded list/JSON
            s = str(x)
            try:
                v = np.asarray(json.loads(s), dtype="float32")
            except Exception:
                v = np.asarray(eval(s), dtype="float32")  # fallback for legacy formats
        n = np.linalg.norm(v)
        return v / (n + 1e-9)

    if "z_style_unit" not in df.columns:
        raise ValueError("Missing required column: z_style_unit")
    df["z_style_unit"] = df["z_style_unit"].apply(parse_vec)

    # style_dim checks
    if "style_dim" in df.columns:
        ok = df["z_style_unit"].map(len) == df["style_dim"].astype(int)
        df = df.loc[ok].copy()
        df["style_dim"] = df["style_dim"].astype(int)
    else:
        df["style_dim"] = df["z_style_unit"].map(len)

    # Core non-nulls
    needed = ["ps_index","player_name","team_name","season_id"]
    df = df.dropna(subset=needed)

    # Deduplicate ps_index
    if df["ps_index"].duplicated().any():
        df = df.drop_duplicates(subset=["ps_index"])  # keep first

    return df

@st.cache_data(show_spinner=False)
def get_options(df: pd.DataFrame):
    teams = sorted([t for t in df["team_name"].dropna().unique()])
    seasons = sorted([s for s in df["season_id"].dropna().unique()])
    pos_mode = sorted([p for p in df["position_mode"].dropna().unique()]) if "position_mode" in df else []
    pos_coarse = sorted([p for p in df["position_mode_coarse"].dropna().unique()]) if "position_mode_coarse" in df else []
    return teams, seasons, pos_mode, pos_coarse

@st.cache_data(show_spinner=False)
def as_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack embedding column into a contiguous float32 matrix."""
    V = np.stack(df["z_style_unit"].values).astype("float32")
    return V

@st.cache_data(show_spinner=False)
def subset(df: pd.DataFrame, idx: Iterable[int]) -> pd.DataFrame:
    return df.loc[idx].copy()
```

---

## src/filters.py

```python
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict, Tuple
from .constants import NUMERIC_STATS, DEFAULT_MIN_MINUTES


def render_filter_sidebar(df: pd.DataFrame):
    st.sidebar.header("Filters")
    teams = sorted(df["team_name"].dropna().unique())
    seasons = sorted(df["season_id"].dropna().unique())
    pos_mode = sorted(df.get("position_mode", pd.Series(dtype=str)).dropna().unique())
    pos_coarse = sorted(df.get("position_mode_coarse", pd.Series(dtype=str)).dropna().unique())

    sel_teams = st.sidebar.multiselect("Team", teams)
    sel_seasons = st.sidebar.multiselect("Season(s)", seasons)
    sel_pos_mode = st.sidebar.multiselect("Position (fine)", pos_mode)
    sel_pos_coarse = st.sidebar.multiselect("Position (coarse)", pos_coarse)
    min_minutes = st.sidebar.number_input("Min minutes", value=DEFAULT_MIN_MINUTES, min_value=0, step=30)

    st.sidebar.markdown("---")
    st.sidebar.caption("Optional stat filters (range by column)")
    stats_filter_cols = {}
    with st.sidebar.expander("Add stat ranges", expanded=False):
        for c in NUMERIC_STATS:
            if c in df:
                lo, hi = float(df[c].min()), float(df[c].max())
                if lo == hi:
                    continue
                use = st.checkbox(f"Filter {c}", value=False, key=f"usef_{c}")
                if use:
                    r = st.slider(f"{c} range", min_value=lo, max_value=hi, value=(lo, hi))
                    stats_filter_cols[c] = r

    return sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols


def apply_filters(df: pd.DataFrame,
                  teams, seasons, pos_mode, pos_coarse,
                  min_minutes, stats_filter_cols: Dict[str, Tuple[float,float]]):
    m = pd.Series(True, index=df.index)
    if teams:
        m &= df["team_name"].isin(teams)
    if seasons:
        m &= df["season_id"].isin(seasons)
    if pos_mode:
        m &= df.get("position_mode").isin(pos_mode)
    if pos_coarse:
        m &= df.get("position_mode_coarse").isin(pos_coarse)
    if min_minutes is not None:
        m &= (df["minutes"] >= float(min_minutes))
    if stats_filter_cols:
        for col, (lo, hi) in stats_filter_cols.items():
            if col in df:
                m &= df[col].between(lo, hi)
    return df.loc[m].copy()
```

---

## src/reduce.py

```python
from __future__ import annotations
import numpy as np
import streamlit as st

try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

import umap
from sklearn.cluster import KMeans

@st.cache_resource(show_spinner=False)
def umap_embed(V: np.ndarray, n_neighbors=40, min_dist=0.15, densmap=False,
               y=None, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="cosine",
        densmap=bool(densmap),
        random_state=int(random_state),
        n_components=2,
    )
    Y = reducer.fit_transform(V, y=y)
    return Y

@st.cache_resource(show_spinner=False)
def cluster_points(Y: np.ndarray, method: str = "hdbscan", k: int = 12,
                   min_cluster_size: int = 25, min_samples: int | None = None):
    if method == "kmeans" or not _HAS_HDBSCAN:
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
        labels = km.fit_predict(Y)
        return labels
    else:
        cl = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                             min_samples=min_samples)
        labels = cl.fit_predict(Y)
        return labels
```

---

## src/sim.py

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple


def as_matrix(df: pd.DataFrame) -> np.ndarray:
    V = np.stack(df["z_style_unit"].values).astype("float32")
    return V


def cosine_sim(V: np.ndarray, q: np.ndarray) -> np.ndarray:
    # V and q assumed L2 normalized
    return V @ q.astype("float32")


def get_query_vec_player(df: pd.DataFrame, player_name: str,
                         seasons: Iterable | None = None,
                         minutes_weighted: bool = True) -> np.ndarray:
    q = df[df["player_name"] == player_name]
    if seasons is not None and len(list(seasons)) > 0:
        q = q[q["season_id"].isin(list(seasons))]
    if q.empty:
        raise ValueError("No rows match the selected player/seasons")
    V = as_matrix(q)
    if minutes_weighted and "minutes" in q:
        w = q["minutes"].to_numpy(dtype="float32")
        w = w / (w.sum() + 1e-9)
        v = (V * w[:, None]).sum(axis=0)
    else:
        v = V.mean(axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def get_query_vec_role(df: pd.DataFrame,
                       pos_mode: str | None = None,
                       pos_coarse: str | None = None,
                       team: str | None = None,
                       seasons: Iterable | None = None,
                       min_minutes: int = 300,
                       minutes_weighted: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    m = df["minutes"] >= float(min_minutes)
    if seasons:
        m &= df["season_id"].isin(list(seasons))
    if team:
        m &= df["team_name"].eq(team)
    if pos_mode:
        m &= df["position_mode"].eq(pos_mode)
    if pos_coarse:
        m &= df["position_mode_coarse"].eq(pos_coarse)

    S = df.loc[m]
    if S.empty:
        raise ValueError("No rows match the selected role filters")

    V = as_matrix(S)
    if minutes_weighted and "minutes" in S:
        w = S["minutes"].to_numpy(dtype="float32")
        w = w / (w.sum() + 1e-9)
        v = (V * w[:, None]).sum(axis=0)
    else:
        v = V.mean(axis=0)
    n = np.linalg.norm(v)
    v = v / (n + 1e-9)
    return v.astype("float32"), S
```

---

## src/scoring.py

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def rank_percentile(x: pd.Series, ascending: bool = False) -> pd.Series:
    r = x.rank(method="average", ascending=ascending)
    return 1.0 - (r - 1) / (len(x) - 1 + 1e-9)


def aggregate_stats(df: pd.DataFrame, stat_prefs):
    """stat_prefs: list of (col, weight, direction) where direction in {"max","min"}"""
    if not stat_prefs:
        return pd.Series(0.0, index=df.index)
    scores = []
    weights = []
    for col, w, d in stat_prefs:
        if col not in df:
            continue
        z = (df[col] - df[col].mean()) / (df[col].std(ddof=0) + 1e-9)
        if d == "min":
            z = -z
        scores.append(norm.cdf(np.clip(z, -4, 4)))
        weights.append(w)
    if not scores:
        return pd.Series(0.0, index=df.index)
    W = np.array(weights, dtype=float)
    W = W / (W.sum() + 1e-9)
    S = (np.stack(scores, axis=1) * W[None, :]).sum(axis=1)
    return pd.Series(S, index=df.index)


def fit_score(df: pd.DataFrame, sim_col: str, stat_prefs, w_sim: float = 0.6, w_stats: float = 0.4) -> pd.Series:
    S_sim = rank_percentile(df[sim_col], ascending=False)
    S_stats = aggregate_stats(df, stat_prefs)
    w_sim = float(w_sim)
    w_stats = float(w_stats)
    if (w_sim + w_stats) == 0:
        return pd.Series(0.0, index=df.index)
    return w_sim * S_sim + w_stats * S_stats
```

---

## src/plotting.py

```python
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _marker_sizes(series: pd.Series, min_size=5, max_size=20):
    s = series.fillna(series.median())
    if s.max() == s.min():
        return np.full(len(s), (min_size + max_size) / 2.0)
    v = (s - s.min()) / (s.max() - s.min())
    return (min_size + v * (max_size - min_size)).to_numpy()


def build_hover(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "%{text}"
    parts = [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(cols)]
    return "<br>".join(["%{text}"] + parts)


def plot_umap(df: pd.DataFrame, Y: np.ndarray, color: str, size: str | None,
              hover_cols, labels_conf, highlights):
    d = df.copy()
    d["umap_x"], d["umap_y"] = Y[:, 0], Y[:, 1]

    hover_cols = [c for c in hover_cols if c in d.columns]
    custom = d[hover_cols] if hover_cols else None

    marker_args = dict()
    if size and size in d:
        marker_args["size"] = _marker_sizes(d[size])

    fig = px.scatter(
        d, x="umap_x", y="umap_y", color=color if color in d else None,
        hover_name="player_name", hover_data=hover_cols,
        render_mode="webgl",
    )
    if size and size in d:
        fig.update_traces(marker=dict(size=marker_args["size"], line=dict(width=0.3)))
    else:
        fig.update_traces(marker=dict(line=dict(width=0.3)))

    # Labels overlay
    if labels_conf and labels_conf.get("stats"):
        k = int(labels_conf.get("k", 10))
        mode = labels_conf.get("mode", "top")
        for stat in labels_conf["stats"]:
            if stat not in d:
                continue
            order = d[stat].sort_values(ascending=(mode == "bottom")).index[:k]
            dd = d.loc[order]
            fig.add_trace(go.Scatter(
                x=dd["umap_x"], y=dd["umap_y"],
                mode="text",
                text=dd["player_name"] + " (" + dd["season_id"].astype(str) + ")",
                textposition="top center",
                textfont=dict(size=11),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Highlights + season lines per player
    if highlights:
        hi = d[d["player_name"].isin(highlights)].copy()
        # points
        fig.add_trace(go.Scattergl(
            x=hi["umap_x"], y=hi["umap_y"], mode="markers",
            marker=dict(size=18, symbol="circle-open", line=dict(width=2)),
            text=hi["player_name"], name="highlight",
        ))
        # polylines by player
        for pname, g in hi.groupby("player_name"):
            gg = g.sort_values("season_id")
            if len(gg) >= 2:
                fig.add_trace(go.Scatter(
                    x=gg["umap_x"], y=gg["umap_y"], mode="lines",
                    line=dict(width=2, dash="dot"), name=f"{pname} path",
                    showlegend=False,
                ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), legend_orientation="h",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
    )
    return fig


def similarity_scatter(df: pd.DataFrame, sim_col: str, y_col: str,
                       color: str | None, size: str | None, hover_cols):
    hover_cols = [c for c in hover_cols if c in df.columns]
    fig = px.scatter(
        df, x=sim_col, y=y_col, color=color if color in df else None,
        size=size if size in df else None,
        hover_data=hover_cols,
        render_mode="webgl",
    )
    fig.update_layout(xaxis_title="Cosine similarity", yaxis_title=y_col,
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig
```

---

## pages/1_UMAP_Explorer.py

```python
from __future__ import annotations
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

# Allow importing from src/
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from src.data import load_players, as_matrix
from src.filters import render_filter_sidebar, apply_filters
from src.reduce import umap_embed, cluster_points
from src.plotting import plot_umap
from src.constants import NUMERIC_STATS

st.title("🗺️ UMAP Explorer")

# Load data
path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)

# Filters
sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols = render_filter_sidebar(df)
S = apply_filters(df, sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols)

st.caption(f"Filtered rows: **{len(S):,}** / {len(df):,}")
if len(S) < 5:
    st.warning("Very few rows after filtering; UMAP may be unstable. Consider loosening filters.")

# UMAP controls
with st.expander("UMAP Parameters", expanded=True):
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        n_neighbors = st.slider("n_neighbors", 10, 100, 40, step=5)
    with c2:
        min_dist = st.slider("min_dist", 0.0, 0.8, 0.15, step=0.01)
    with c3:
        densmap = st.checkbox("densMAP", value=False)
    with c4:
        supervise_by = st.selectbox("Supervise by", ["None","position_mode","position_mode_coarse"], index=0)

compute = st.button("Compute / Refresh UMAP", type="primary", use_container_width=True)

if compute and len(S) >= 2:
    V = as_matrix(S)
    y = None
    if supervise_by != "None" and supervise_by in S:
        y = S[supervise_by]
    Y = umap_embed(V, n_neighbors=n_neighbors, min_dist=min_dist, densmap=densmap, y=y)

    # Clustering controls
    with st.expander("Clustering", expanded=True):
        method = st.selectbox("Method", ["hdbscan","kmeans"])
        if method == "kmeans":
            k = st.slider("k (KMeans)", 2, 25, 12)
            labels = cluster_points(Y, method="kmeans", k=k)
        else:
            min_cluster_size = st.slider("min_cluster_size (HDBSCAN)", 5, 100, 25)
            min_samples = st.number_input("min_samples (optional)", value=0, min_value=0)
            labels = cluster_points(Y, method="hdbscan",
                                    min_cluster_size=min_cluster_size,
                                    min_samples=None if min_samples == 0 else int(min_samples))

    S = S.copy()
    S["cluster"] = labels

    # Plot controls
    st.markdown("### Plot Controls")
    col_color, col_size = st.columns([2,1])
    with col_color:
        color_by = st.selectbox("Color by", options=["cluster","team_name","season_id","position_mode","position_mode_coarse","minutes"] + [c for c in NUMERIC_STATS if c in S], index=0)
    with col_size:
        size_by = st.selectbox("Size by", options=["(none)"] + [c for c in ["minutes"] + [c for c in NUMERIC_STATS if c in S]], index=0)
        size_by = None if size_by == "(none)" else size_by

    with st.expander("Labels", expanded=False):
        stats_to_label = st.multiselect("Stats to label (top/bottom k)", [c for c in NUMERIC_STATS if c in S])
        k = st.number_input("k", value=10, min_value=1)
        mode = st.selectbox("Mode", ["top","bottom"], index=0)
        labels_conf = {"stats": stats_to_label, "k": int(k), "mode": mode}

    with st.expander("Highlights", expanded=False):
        names = sorted(S["player_name"].unique())
        highlights = st.multiselect("Player(s) to highlight", names)

    with st.expander("Hover data", expanded=False):
        hover_cols = st.multiselect("Columns to show on hover", options=list(S.columns), default=["player_name","team_name","season_id","minutes","position_mode"])

    fig = plot_umap(S, Y, color=color_by, size=size_by, hover_cols=hover_cols, labels_conf=labels_conf, highlights=highlights)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Download data
    out = S[["ps_index","player_name","season_id","team_name","minutes","cluster"]].copy()
    out["umap_x"], out["umap_y"] = Y[:,0], Y[:,1]
    st.download_button("Download UMAP CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="umap_coordinates.csv")
else:
    st.info("Set parameters and click **Compute / Refresh UMAP**.")
```

---

## pages/2_Player_or_Role_Search.py

```python
from __future__ import annotations
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from src.data import load_players
from src.filters import render_filter_sidebar, apply_filters
from src.sim import get_query_vec_player, get_query_vec_role, as_matrix, cosine_sim
from src.constants import NUMERIC_STATS

st.title("🔎 Player / Role Search")

path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)
sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols = render_filter_sidebar(df)
S = apply_filters(df, sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols)

st.caption(f"Candidates after filters: **{len(S):,}**")

mode = st.radio("Query mode", ["Player-based","Role-based"], horizontal=True)

query_vec = None
query_desc = None

if mode == "Player-based":
    names = sorted(df["player_name"].unique())
    player = st.selectbox("Player", names)
    use_seasons = st.multiselect("Limit to season(s) for the query vector (optional)", sorted(df["season_id"].unique()))
    wmin = st.checkbox("Minutes-weighted", value=True)
    if st.button("Build query vector", type="primary"):
        query_vec = get_query_vec_player(df, player_name=player, seasons=use_seasons, minutes_weighted=wmin)
        query_desc = f"Player-based: {player} ({'weighted' if wmin else 'mean'})"
else:
    pos_mode = st.selectbox("Position (fine) (optional)", options=[None] + sorted(df.get("position_mode", pd.Series(dtype=str)).dropna().unique().tolist()))
    pos_coarse = st.selectbox("Position (coarse) (optional)", options=[None] + sorted(df.get("position_mode_coarse", pd.Series(dtype=str)).dropna().unique().tolist()))
    team = st.selectbox("Team (optional)", options=[None] + sorted(df["team_name"].unique().tolist()))
    seasons = st.multiselect("Season(s) (optional)", sorted(df["season_id"].unique()))
    minm = st.number_input("Min minutes for role pool", value=300, min_value=0, step=30)
    wmin = st.checkbox("Minutes-weighted", value=True, key="rb_wmin")
    if st.button("Build role centroid", type="primary"):
        query_vec, pool = get_query_vec_role(df, pos_mode=pos_mode, pos_coarse=pos_coarse, team=team,
                                             seasons=seasons, min_minutes=int(minm), minutes_weighted=wmin)
        query_desc = f"Role-based: pos={pos_mode or pos_coarse}, team={team}, seasons={seasons or 'all'}"

if query_vec is not None:
    st.success(f"Built query vector → {query_desc}")
    V = as_matrix(S)
    sims = cosine_sim(V, query_vec)
    R = S[["ps_index","player_name","season_id","team_name","minutes","position_mode","position_mode_coarse"]].copy()
    R["sim"] = sims

    # Choose columns to show
    add_cols = st.multiselect("Add stat columns", [c for c in NUMERIC_STATS if c in S], default=["goals_per90","xg_per90","xa_proxy"])
    cols = list(R.columns) + add_cols
    R = R.join(S[add_cols]) if add_cols else R

    # Sort and show
    topn = st.slider("Top N", 10, 200, 50)
    R = R.sort_values("sim", ascending=False).head(topn).reset_index(drop=True)
    st.dataframe(R, use_container_width=True)

    st.session_state["last_query_vec"] = query_vec
    st.session_state["last_candidates_df"] = S
    st.session_state["last_sim_results"] = R

    st.download_button("Download similarity results (CSV)", data=R.to_csv(index=False).encode("utf-8"), file_name="similarity_results.csv")
else:
    st.info("Configure and build a query vector to compute similarities.")
```

---

## pages/3_Similarity_Plot.py

```python
from __future__ import annotations
import os, sys
import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from src.data import load_players
from src.filters import render_filter_sidebar, apply_filters
from src.sim import as_matrix, cosine_sim
from src.plotting import similarity_scatter
from src.constants import NUMERIC_STATS

st.title("📈 Similarity Plot")

path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)
sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols = render_filter_sidebar(df)
S = apply_filters(df, sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols)

# Query vector source
st.markdown("Use the last built query vector (from Player/Role Search) or provide a quick ad-hoc selection below.")
use_last = st.toggle("Use last query vector", value=("last_query_vec" in st.session_state))

from src.sim import get_query_vec_player
query_vec = None
if use_last:
    query_vec = st.session_state.get("last_query_vec")
else:
    names = sorted(df["player_name"].unique())
    player = st.selectbox("Quick: choose a player as query", names)
    if st.button("Build quick query"):
        query_vec = get_query_vec_player(df, player_name=player)

if query_vec is not None:
    V = as_matrix(S)
    S = S.copy()
    S["sim"] = cosine_sim(V, query_vec)

    y_col = st.selectbox("Y-axis metric", [c for c in NUMERIC_STATS if c in S], index=0)
    color = st.selectbox("Color by", options=["team_name","season_id","position_mode","position_mode_coarse","minutes"] + [c for c in NUMERIC_STATS if c in S], index=0)
    size = st.selectbox("Size by", options=["(none)"] + ["minutes"] + [c for c in NUMERIC_STATS if c in S])
    size = None if size == "(none)" else size

    with st.expander("Hover data", expanded=False):
        hover = st.multiselect("Columns", options=list(S.columns), default=["player_name","team_name","season_id","minutes", y_col, "sim"])

    fig = similarity_scatter(S, sim_col="sim", y_col=y_col, color=color, size=size, hover_cols=hover)

    # Similarity threshold line
    thr = st.slider("Similarity threshold", 0.0, 1.0, 0.6, step=0.01)
    fig.add_vline(x=thr, line_width=2, line_dash="dash", annotation_text=f"thr={thr:.2f}")

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    if st.checkbox("Show only above threshold"):
        st.dataframe(S[S["sim"] >= thr].sort_values("sim", ascending=False), use_container_width=True)
else:
    st.info("Build or reuse a query vector, then configure the plot.")
```

---

## pages/4_Fit_and_Rankings.py

```python
from __future__ import annotations
import os, sys
import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from src.data import load_players
from src.filters import render_filter_sidebar, apply_filters
from src.sim import as_matrix, cosine_sim
from src.scoring import fit_score
from src.constants import NUMERIC_STATS

st.title("🏅 Fit Score & Rankings")

path = st.session_state.get("data_path")
if not path or not os.path.exists(path):
    st.error("Parquet file not found. Set it in the main page sidebar.")
    st.stop()

df = load_players(path)
sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols = render_filter_sidebar(df)
S = apply_filters(df, sel_teams, sel_seasons, sel_pos_mode, sel_pos_coarse, min_minutes, stats_filter_cols)

# Query vector: prefer last from session
query_vec = st.session_state.get("last_query_vec")
if query_vec is None:
    st.warning("No query vector found. Build one on the 'Player / Role Search' page.")
    st.stop()

# Compute similarity on candidate set
V = as_matrix(S)
S = S.copy()
S["sim"] = cosine_sim(V, query_vec)

# Stat preferences UI
st.subheader("Stat Preferences")
sel_stats = st.multiselect("Select stats to include in Fit", [c for c in NUMERIC_STATS if c in S], default=["goals_per90","xg_per90","xa_proxy"])
stat_prefs = []
for col in sel_stats:
    c1, c2, c3 = st.columns([3,1,2])
    with c1:
        st.write(f"**{col}**")
    with c2:
        direction = st.selectbox("direction", ["max","min"], key=f"dir_{col}")
    with c3:
        weight = st.slider("weight", 0.0, 1.0, 0.33, key=f"w_{col}")
    stat_prefs.append((col, weight, direction))

w_sim = st.slider("Weight: Similarity", 0.0, 1.0, 0.6)
w_stats = 1.0 - w_sim
st.write(f"Weight: Stats = **{w_stats:.2f}** (auto = 1 - Similarity)")

# Compute Fit
S["Fit"] = fit_score(S, sim_col="sim", stat_prefs=stat_prefs, w_sim=w_sim, w_stats=w_stats)
S = S.sort_values(["Fit","sim"], ascending=[False, False])

cols = ["ps_index","player_name","season_id","team_name","minutes","position_mode","position_mode_coarse","sim","Fit"] + sel_stats
st.dataframe(S[cols].reset_index(drop=True), use_container_width=True)

st.download_button("Download rankings (CSV)", data=S[cols].to_csv(index=False).encode("utf-8"), file_name="fit_rankings.csv")

# Option to push top-N to highlights (UMAP page)
push = st.number_input("Send top N to highlights (UMAP)", value=25, min_value=1)
if st.button("Store highlights in session"):
    top_names = S.head(int(push))["player_name"].tolist()
    st.session_state["umap_highlights"] = top_names
    st.success(f"Stored {len(top_names)} names; open UMAP Explorer and select 'Player(s) to highlight)' if needed.")
```

---

## README.md (optional)

````markdown
# Player Style Explorer

Interactive Streamlit app to explore player style embeddings, build UMAP maps, run player/role similarity searches, and compute tunable Fit rankings.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
````

Place your `players_clean.parquet` at `data/players_clean.parquet` or set the path/upload via the app sidebar. Ensure the file contains:

* `z_style_unit` (L2‑normalized vector per row) and `style_dim`
* metadata columns and numeric stats as described in your data dictionary

```
```
