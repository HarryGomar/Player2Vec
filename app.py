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
    **Welcome!** Use the pages on the left to explore the dataset:

    1. Build rich UMAP style maps
    2. Run player / role similarity searches
    3. Plot similarity vs custom metrics
    4. Rank candidates with a tunable **Fit Score**

    👉 Start on **UMAP Explorer** after selecting your data file.
    """
)

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