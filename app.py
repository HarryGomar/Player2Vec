from __future__ import annotations
import os
import streamlit as st

st.set_page_config(
    page_title="Player Style Explorer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ Player Style Explorer")

st.markdown(
        """
        ## Welcome to Player Style Explorer

        Player Style Explorer helps you analyze player embeddings and identify similar
        players, roles, and candidate fits across your dataset. Use the left-hand pages
        to navigate distinct workflows — from interactive UMAP visualizations to
        similarity searching and ranking candidates by a custom fit score.

        This homepage explains what each page does, how to prepare your data, and
        step-by-step instructions so you can get meaningful results quickly.
        """
)

st.markdown("---")

st.header("What this app does")
st.markdown(
        """
        - Build 2D/3D style maps (UMAP) from player embeddings so you can visually
            explore clusters and play styles.
        - Search for similar players or roles using vector similarity in embedding
            space.
        - Compare similarity scores with custom metrics and visualize relationships
            between players.
        - Generate and tune a composite **Fit Score** to rank candidate players for a
            target role or profile.
        """
)

st.header("Quick start — 4 simple steps")
st.markdown(
        """
        1. In the sidebar choose your data source (a parquet file with cleaned player
             features). Default: `data/players_clean.parquet` or upload your own.
        2. Open **UMAP Explorer** to build and tune a UMAP projection. Inspect
             clusters and select players of interest.
        3. Use **Player or Role Search** to find nearest neighbours to a chosen
             player or role embedding. Adjust similarity metric and top-K results.
        4. Visit **Fit and Rankings** to create a composite fit score combining
             similarity and your custom metrics, then export ranked candidate lists.

        Tip: Start with a small sample of your data while experimenting with
        UMAP and similarity parameters, then scale up once settings look good.
        """
)

st.header("Pages walkthrough")
with st.expander("1 — UMAP Explorer (visualize)", expanded=True):
        st.markdown(
                """
                Create and tune UMAP projections from the selected embedding columns.
                - Choose embedding columns or a precomputed embedding column.
                - Adjust UMAP hyperparameters (n_neighbors, min_dist, n_components).
                - Color points by metadata (team, role, season) and inspect points.
                Use the interactive plot to pick players or regions for further analysis.
                """
        )

with st.expander("2 — Player or Role Search (find similar)"):
        st.markdown(
                """
                Find nearest neighbours for a player or role prototype.
                - Enter a player id or select multiple players to form a role vector.
                - Choose similarity metric (cosine, euclidean) and number of results.
                - Inspect returned players, their similarity scores, and key stats.
                """
        )

with st.expander("3 — Similarity Plot"):
        st.markdown(
                """
                Plot similarity scores against custom metrics.
                - Compare similarity with physical or performance metrics (goals, xG,
                    pass completion) to find trade-offs.
                - Use scatter, density, and marginal plots to examine distributions.
                """
        )

with st.expander("4 — Fit and Rankings"):
        st.markdown(
                """
                Define a composite Fit Score and rank candidates.
                - Choose weights for similarity and any other normalized metrics.
                - Preview ranked lists and export to CSV for downstream workflows.
                - Save favorite configurations for repeatable searches.
                """
        )

st.header("Data format & requirements")
st.markdown(
        """
        - Expected input: a cleaned parquet file (default path: `data/players_clean.parquet`).
        - Required columns: player id, embedding vector (or separate embedding columns),
            and any metadata you want to filter or color by (team, role, season).
        - If uploading, the uploader accepts `.parquet` files. Uploaded files are
            written to `.uploads/` and used for the session.

        If your embeddings are stored as multiple columns (e.g. emb_0..emb_127),
        the data loader will automatically detect them. If you have a single column
        storing a list/array per row, the loader will also accept that format.
        """
)

st.header("Tips, pitfalls & FAQs")
st.markdown(
        """
        - If UMAP looks noisy, try increasing `n_neighbors` or reducing `min_dist`.
        - Use cosine similarity for directional embedding comparisons (typical for
            neural embeddings). Use Euclidean when embeddings are already normalized
            for magnitude meaning.
        - For ranking, normalize all metrics to the same scale before combining.
        - Export results from the Fit and Rankings page for reporting or further
            filtering in Python / Excel.
        """
)

st.markdown("---")
st.info("Use the sidebar to set the data file. Then open a page from the left navigation.")

# --- Global data selection (sidebar) ---
with st.sidebar:
    st.header("Data Source")
    st.caption("Expected parquet: `players_clean.parquet`")

    default_path = os.path.join("data", "players_clean.parquet")
    data_path = st.text_input("Parquet path", value=default_path)
    uploaded = st.file_uploader("…or upload parquet", type=["parquet"])
    if uploaded is not None:
        os.makedirs(".uploads", exist_ok=True)
        tmp_path = os.path.join(".uploads", "players_clean_uploaded.parquet")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        data_path = tmp_path

    st.session_state.setdefault("data_path", data_path)
    st.session_state["data_path"] = data_path

st.info("Use the sidebar to set the data file. Then open a page from the left navigation.")