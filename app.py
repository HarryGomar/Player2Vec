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

st.header("Pages walkthrough")
st.header("Task-focused instructions")
st.markdown("Use the tabs below to open guided instructions for the two main workflows: building/exploring maps, and finding & ranking players.")

tab_maps, tab_search = st.tabs(["Maps (UMAP Explorer)", "Player finding (Search & Fit)"])

with tab_maps:
    st.subheader("Maps — what and why")
    st.markdown(
        """
        The Maps workflow (UMAP Explorer) reduces high-dimensional player embeddings
        into 2D or 3D for interactive exploration. This helps you find clusters of
        similar playing styles, outliers, and relationships between players and
        metadata (team, position, season).

        Typical uses:
        - Visualise embedding structure to discover natural groupings (styles/roles).
        - Inspect nearby players to a selected point (who occupies a region).
        - Select groups of players to create role prototypes for search.

        Key features available in the UMAP Explorer:
        - Select embedding columns or a single precomputed embedding vector column.
        - Tune UMAP hyperparameters (n_neighbors, min_dist, n_components) to
          control local/global structure.
        - Color and size points by metadata or numeric stats (team, position, goals,
          minutes) to reveal patterns.
        - Filter the dataset (season, team, minimum minutes) to focus on subsets.
        - Hover and click points to see player details and add them to a selection.
        - Export selected players for downstream similarity searches or ranking.

        Quick steps to use Maps:
        1. Choose embedding columns or a precomputed embedding column.
        2. Tune `n_neighbors` and `min_dist` to change granularity.
        3. Use Color/Size controls to map metadata or metrics to visuals.
        4. Use filters to restrict the plot to seasons, teams, or minimum minutes.
        5. Select interesting players or regions and export the selection for the
           Player finding workflow.

        Tips & caveats:
        - Start with a small sample while experimenting with parameters.
        - UMAP is stochastic: set a random seed for reproducible projections.
        - Normalise or scale features consistently before embedding.
        """
    )

with tab_search:
    st.subheader("Player finding — overview")
    st.markdown(
        """
        The Player finding workflow lets you search for players similar to a query
        and create ranked candidate lists using a composite Fit Score.

        Query types supported:
        - Single player: provide a player id to use that player's embedding as the
          query vector.
        - Prototype / Role vector: average embeddings from a selection of players
          (for example all players who play a specific position in a given team),
          or build a prototype by selecting multiple players from the Maps page.

        What the page does:
        1. Apply dataset filters (season, team, position, mins) to restrict candidates.
        2. Accept a query vector (single player or averaged prototype).
        3. Run a nearest-neighbour search (choose metric: cosine, euclidean) and
           return a table of the top-K most similar players with similarity scores.
        4. Visualise results with multiple indicators (scatter plots, radar charts,
           histograms) to compare similarity against performance stats.
        5. Build a Fit Score by combining similarity with other normalized metrics
           (e.g. goals per 90, xG, pass completion) using user-specified weights.
        6. Export ranked candidate lists to CSV for scouting or reporting.

        Step-by-step quick guide:
        1. Set filters to narrow the candidate pool (season, team, minimum minutes).
        2. Choose query type:
           - Enter a player id for a direct lookup, or
           - Select multiple players / position to compute an averaged prototype.
        3. Select similarity metric and top-K (e.g. top 50).
        4. Run the search — examine the returned table (similarity score + key stats).
        5. Use the Plot tab to visualise metrics for top candidates (use different
           indicators to inspect trade-offs between style and performance).
        6. Create a Fit Score: normalise chosen stats, set weights, preview the ranked list,
           and export the results.

        Practical tips:
        - When averaging embeddings to make a prototype, ensure the players are
          comparable (same position / similar minutes) to avoid noisy prototypes.
        - Use cosine similarity for directional comparisons (typical for neural
          embeddings). If embeddings encode magnitude meaning, consider Euclidean.
        - Normalize metrics to the same scale before combining into a Fit Score.
        - Save good weight configurations so you can reproduce searches later.
        """
    )

st.markdown("---")


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