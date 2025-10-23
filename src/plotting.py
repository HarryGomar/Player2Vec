from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative as px_qual


def _prepare_hover(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    cols = [c for c in columns if c in df.columns]
    return cols


def _collect_label_indices(df: pd.DataFrame, stats: List[str], top_k: int, bottom_k: int) -> List[int]:
    idx: set[int] = set()
    for stat in stats:
        if stat not in df.columns:
            continue
        if top_k > 0:
            top_idx = df[stat].nlargest(top_k, keep="all").index
            idx.update(top_idx)
        if bottom_k > 0:
            bottom_idx = df[stat].nsmallest(bottom_k, keep="all").index
            idx.update(bottom_idx)
    return list(idx)


def plot_umap(
    df: pd.DataFrame,
    Y: np.ndarray,
    color: Optional[str],
    size: Optional[str],
    hover_cols: Iterable[str],
    labels_conf: Optional[Dict],
    highlights: Optional[Iterable[str]],
    connect_highlights: bool = True,
) -> go.Figure:
    data = df.copy()
    data["umap_x"], data["umap_y"] = Y[:, 0], Y[:, 1]

    color_col = color if color in data.columns else None
    size_col = size if size in data.columns else None
    hover_cols = _prepare_hover(data, hover_cols)

    fig = px.scatter(
        data,
        x="umap_x",
        y="umap_y",
        color=color_col,
        size=size_col,
        size_max=18,
        hover_name="player_name" if "player_name" in data.columns else None,
        hover_data=hover_cols,
        render_mode="webgl",
    )

    fig.update_traces(marker=dict(line=dict(width=0.3, color="rgba(0,0,0,0.35)")))

    # Label overlay for top/bottom performers by stat
    if labels_conf:
        stats = labels_conf.get("stats", [])
        top_k = int(labels_conf.get("top_k", 0))
        bottom_k = int(labels_conf.get("bottom_k", 0))
        label_indices = _collect_label_indices(data, stats, top_k, bottom_k)
        if label_indices:
            label_df = data.loc[label_indices].copy()
            def _format_label(row: pd.Series) -> str:
                stat_bits = [
                    f"{stat}: {row[stat]:.2f}" for stat in stats if stat in row and pd.notnull(row[stat])
                ]
                suffix = f" ({', '.join(stat_bits)})" if stat_bits else ""
                season = f" [{row['season_id']}]" if "season_id" in row and pd.notnull(row["season_id"]) else ""
                return f"{row['player_name']}{season}{suffix}"

            label_df["label_text"] = label_df.apply(_format_label, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=label_df["umap_x"],
                    y=label_df["umap_y"],
                    mode="text",
                    text=label_df["label_text"],
                    textposition="top center",
                    textfont=dict(size=11, color="#111", family="Arial"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Highlight specific players and optionally connect seasons
    if highlights:
        highlight_df = data[data["player_name"].isin(list(highlights))]
        if not highlight_df.empty:
            palette = px_qual.Safe
            for idx, (player, group) in enumerate(highlight_df.groupby("player_name")):
                color_hex = palette[idx % len(palette)]
                group_sorted = group.sort_values(by=["season_id", "minutes"]) if "season_id" in group else group
                fig.add_trace(
                    go.Scattergl(
                        x=group_sorted["umap_x"],
                        y=group_sorted["umap_y"],
                        mode="markers",
                        marker=dict(
                            size=18,
                            color=color_hex,
                            symbol="circle-open",
                            line=dict(width=2, color="#222"),
                        ),
                        name=f"{player} highlight",
                        text=[f"{player} ({s})" for s in group_sorted.get("season_id", [])],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                    )
                )
                if connect_highlights and len(group_sorted) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=group_sorted["umap_x"],
                            y=group_sorted["umap_y"],
                            mode="lines",
                            line=dict(width=2, dash="dot", color=color_hex),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend_orientation="h",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        dragmode="pan",
    )
    return fig


def similarity_scatter(
    df: pd.DataFrame,
    sim_col: str,
    y_col: str,
    color: Optional[str],
    size: Optional[str],
    hover_cols: Iterable[str],
    highlight_players: Optional[Iterable[str]] = None,
) -> go.Figure:
    color_col = color if color in df.columns else None
    size_col = size if size in df.columns else None
    hover_cols = _prepare_hover(df, hover_cols)

    fig = px.scatter(
        df,
        x=sim_col,
        y=y_col,
        color=color_col,
        size=size_col,
        size_max=18,
        hover_name="player_name" if "player_name" in df.columns else None,
        hover_data=hover_cols,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(line=dict(width=0.3, color="rgba(0,0,0,0.35)")))

    if highlight_players:
        palette = px_qual.Vivid
        highlight_df = df[df["player_name"].isin(list(highlight_players))]
        for idx, (player, group) in enumerate(highlight_df.groupby("player_name")):
            fig.add_trace(
                go.Scatter(
                    x=group[sim_col],
                    y=group[y_col],
                    mode="markers+text",
                    marker=dict(size=16, color=palette[idx % len(palette)], symbol="diamond", line=dict(width=1.5)),
                    text=[player] * len(group),
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=f"{player}<extra></extra>",
                )
            )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Cosine similarity",
        yaxis_title=y_col,
        dragmode="pan",
    )
    return fig