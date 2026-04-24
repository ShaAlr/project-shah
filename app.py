"""
╔══════════════════════════════════════════════════════════════════╗
║   STEAM GAME RECOMMENDATION SYSTEM — KELOMPOK 12                ║
║   Content-Based Filtering | TF-IDF + Cosine Similarity          ║
║   ITS - Proyek Sains Data                                       ║
╚══════════════════════════════════════════════════════════════════╝

Cara menjalankan:
    pip install streamlit pandas scikit-learn plotly
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SteamRec Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — Dark Motorsport Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0d0d0d;
    --bg2:       #161616;
    --bg3:       #1e1e1e;
    --border:    #2a2a2a;
    --red:       #E10600;
    --red-dim:   #8B0400;
    --red-glow:  rgba(225, 6, 0, 0.25);
    --text:      #f0f0f0;
    --muted:     #888;
    --accent:    #ff4d4d;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0000 50%, #0d0d0d 100%);
    border-bottom: 3px solid var(--red);
    padding: 28px 40px 22px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 24px;
    position: relative;
    overflow: hidden;
}

.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 80px,
        rgba(225,6,0,0.03) 80px,
        rgba(225,6,0,0.03) 81px
    );
}

.header-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 2.6rem;
    color: var(--red);
    letter-spacing: -1px;
    line-height: 1;
}

.header-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--muted);
    letter-spacing: 4px;
    text-transform: uppercase;
}

.header-badge {
    margin-left: auto;
    background: var(--red);
    color: white;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 2px;
    padding: 4px 12px;
    border-radius: 2px;
    text-transform: uppercase;
}

/* ── Tab Bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 8px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 14px 24px !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.2s !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 3px solid var(--red) !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg) !important;
    padding: 28px 0 !important;
}

/* ── Section Title ── */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 1.8rem;
    color: var(--text);
    letter-spacing: 3px;
    text-transform: uppercase;
    border-left: 4px solid var(--red);
    padding-left: 16px;
    margin-bottom: 24px;
}

/* ── Input Labels ── */
label, .stSelectbox label, .stSlider label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--red) !important;
}

/* ── Sliders ── */
.stSlider [data-testid="stSlider"] > div > div > div {
    background: var(--red) !important;
}

/* ── RUN PREDICTION Button ── */
.stButton > button {
    background: var(--red) !important;
    color: white !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 14px 36px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px var(--red-glow) !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #ff1a00 !important;
    box-shadow: 0 0 35px var(--red-glow) !important;
    transform: translateY(-1px) !important;
}

/* ── Game Card ── */
.game-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 4px solid var(--red);
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 14px;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}

.game-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 100px; height: 100%;
    background: linear-gradient(to left, rgba(225,6,0,0.03), transparent);
    pointer-events: none;
}

.game-card:hover {
    border-left-color: var(--accent);
    background: var(--bg3);
    transform: translateX(3px);
}

.card-rank {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 2.5rem;
    color: var(--red);
    opacity: 0.4;
    position: absolute;
    top: 12px;
    right: 20px;
    line-height: 1;
}

.card-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    color: var(--text);
    margin-bottom: 6px;
}

.card-genre {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 14px;
}

.card-meta {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.meta-item {
    font-size: 0.82rem;
    color: var(--muted);
}

.meta-value {
    color: var(--text);
    font-weight: 600;
}

.similarity-bar-bg {
    background: var(--border);
    border-radius: 2px;
    height: 4px;
    margin-top: 14px;
    overflow: hidden;
}

.similarity-bar {
    background: linear-gradient(to right, var(--red-dim), var(--red));
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

.similarity-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 1px;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Info Box ── */
.info-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── Profile Card ── */
.profile-outer {
    background: linear-gradient(135deg, var(--bg2), #0a0000);
    border: 1px solid var(--border);
    border-top: 3px solid var(--red);
    border-radius: 6px;
    padding: 32px;
    margin-bottom: 20px;
}

.profile-team-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 1.5rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--text);
    margin-bottom: 6px;
}

.profile-team-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 28px;
}

.member-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 22px;
    margin-bottom: 14px;
}

.member-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--text);
}

.member-nrp {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 2px;
    color: var(--red);
    margin-bottom: 12px;
}

.member-focus {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.6;
}

.member-tag {
    display: inline-block;
    background: rgba(225,6,0,0.12);
    border: 1px solid var(--red-dim);
    color: var(--red);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 2px;
    margin: 2px 3px 2px 0;
}

/* ── No Result Box ── */
.no-result {
    background: var(--bg3);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 32px;
    text-align: center;
    color: var(--muted);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 1px;
}

/* ── Divider ── */
.red-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--red-dim); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div>
        <div class="header-logo">🎮 STEAMREC</div>
        <div class="header-sub">Analytics &nbsp;·&nbsp; Content-Based Filtering Engine</div>
    </div>
    <div class="header-badge">Kelompok 12 &nbsp;·&nbsp; ITS Proyek Sains Data</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DUMMY DATASET  (20 popular Steam games)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        "Name": [
            "Counter-Strike 2",
            "Elden Ring",
            "Cyberpunk 2077",
            "Stardew Valley",
            "Red Dead Redemption 2",
            "Hades",
            "Grand Theft Auto V",
            "The Witcher 3: Wild Hunt",
            "Hollow Knight",
            "Dota 2",
            "FIFA 23",
            "Civilization VI",
            "Terraria",
            "Among Us",
            "Sekiro: Shadows Die Twice",
            "Portal 2",
            "Rust",
            "Valheim",
            "Monster Hunter: World",
            "Dark Souls III",
        ],
        "Genre": [
            "Action FPS",
            "Action RPG",
            "Action RPG",
            "Simulation RPG",
            "Action Adventure",
            "Action RPG",
            "Action Adventure",
            "Action RPG",
            "Action Platformer",
            "Action Strategy",
            "Sports Simulation",
            "Turn-Based Strategy",
            "Action Adventure",
            "Social Deduction",
            "Action Adventure",
            "Puzzle Adventure",
            "Survival",
            "Survival",
            "Action RPG",
            "Action RPG",
        ],
        "Price_USD": [
            0.0,
            59.99,
            59.99,
            14.99,
            59.99,
            24.99,
            29.99,
            39.99,
            14.99,
            0.0,
            69.99,
            39.99,
            9.99,
            5.0,
            59.99,
            9.99,
            39.99,
            19.99,
            29.99,
            59.99,
        ],
        "RAM_GB": [8, 12, 12, 4, 12, 8, 8, 6, 8, 4, 8, 8, 2, 1, 8, 4, 10, 8, 8, 8],
        "OS": [
            "Windows", "Windows", "Windows/Linux", "Windows/Mac/Linux",
            "Windows", "Windows/Mac/Linux", "Windows/Mac/Linux", "Windows/Mac/Linux",
            "Windows/Mac/Linux", "Windows/Mac/Linux", "Windows", "Windows/Mac/Linux",
            "Windows/Mac/Linux", "Windows/Mac/Linux", "Windows", "Windows/Mac/Linux",
            "Windows", "Windows/Mac/Linux", "Windows", "Windows",
        ],
        "GPU_Tier": [
            "Low", "High", "High", "Low",
            "High", "Medium", "Medium", "Medium",
            "Low", "Low", "High", "Medium",
            "Low", "Low", "High", "Low",
            "Medium", "Medium", "High", "High",
        ],
        "Rating": [
            9.1, 9.5, 8.7, 9.8,
            9.6, 9.7, 9.0, 9.8,
            9.6, 8.5, 7.8, 8.9,
            9.7, 8.0, 9.4, 9.8,
            8.6, 9.2, 9.1, 9.4,
        ],
        "Developer": [
            "Valve", "FromSoftware", "CD Projekt Red", "ConcernedApe",
            "Rockstar", "Supergiant", "Rockstar", "CD Projekt Red",
            "Team Cherry", "Valve", "EA Sports", "Firaxis",
            "Re-Logic", "Innersloth", "FromSoftware", "Valve",
            "Facepunch", "Iron Gate", "Capcom", "FromSoftware",
        ],
    }
    df = pd.DataFrame(data)
    return df


df = load_data()


# ─────────────────────────────────────────────
#  CBF ENGINE
# ─────────────────────────────────────────────
@st.cache_data
def build_tfidf_matrix(dataframe):
    """Build TF-IDF matrix from combined feature text."""
    dataframe = dataframe.copy()
    dataframe["feature_text"] = (
        dataframe["Genre"].str.lower().str.replace(" ", "_") + " " +
        dataframe["OS"].str.lower().str.replace("/", " ").str.replace(" ", "_") + " " +
        dataframe["GPU_Tier"].str.lower() + "_gpu"
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(dataframe["feature_text"])
    return tfidf_matrix, vectorizer, dataframe


tfidf_matrix, vectorizer, df_featured = build_tfidf_matrix(df)


def recommend_games(genre: str, os_pref: str, budget: float, min_ram: int, top_n: int = 5):
    """
    1. Hard filter by budget & RAM
    2. Compute cosine similarity against user preference profile
    3. Return top_n results sorted by similarity desc
    """
    # Hard filters
    filtered = df_featured.copy()
    filtered = filtered[filtered["Price_USD"] <= budget]
    filtered = filtered[filtered["RAM_GB"] <= min_ram]

    if filtered.empty:
        return pd.DataFrame(), None

    # Build user query vector
    os_clean = os_pref.lower().replace("/", " ").replace(" ", "_")
    query = f"{genre.lower().replace(' ', '_')} {os_clean} medium_gpu"
    query_vec = vectorizer.transform([query])

    # Cosine similarity only for filtered rows
    filtered_indices = filtered.index.tolist()
    filtered_matrix = tfidf_matrix[filtered_indices]
    similarities = cosine_similarity(query_vec, filtered_matrix).flatten()

    filtered = filtered.reset_index(drop=True)
    filtered["Similarity"] = similarities
    result = filtered.sort_values("Similarity", ascending=False).head(top_n)
    return result, similarities.max()


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚡  SIMULATION SETTINGS",
    "📊  DATA ANALYTICS",
    "👤  TEAM PROFILE",
])


# ════════════════════════════════════════════
#  TAB 1 — SIMULATION SETTINGS (Rekomendasi)
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Simulation Settings</div>', unsafe_allow_html=True)

    col_form, col_sep, col_result = st.columns([1.2, 0.05, 2])

    with col_form:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        # Genre selector
        genres = sorted(df["Genre"].str.split().str[0].unique().tolist())
        all_genres = sorted(df["Genre"].unique().tolist())
        genre_choice = st.selectbox(
            "Genre Favorit",
            options=all_genres,
            index=0,
        )

        # OS selector
        os_options = ["Windows", "Windows/Mac/Linux", "Windows/Linux"]
        os_choice = st.selectbox(
            "Target OS",
            options=os_options,
            index=0,
        )

        # Budget slider
        budget_choice = st.slider(
            "Budget Maksimum (USD)",
            min_value=0,
            max_value=70,
            value=60,
            step=1,
            format="$%d",
        )

        # RAM select slider
        ram_choice = st.select_slider(
            "RAM Tersedia (GB)",
            options=[1, 2, 4, 6, 8, 10, 12, 16, 32],
            value=8,
        )

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        run_btn = st.button("🎮  CARI REKOMENDASI GAME")

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Atur preferensi lalu tekan <strong style="color:#E10600;">CARI REKOMENDASI GAME</strong>
            untuk mendapatkan rekomendasi game personal menggunakan
            <strong>TF-IDF + Cosine Similarity</strong> dengan filter ketat
            berdasarkan budget dan kompatibilitas RAM.
        </div>
        """, unsafe_allow_html=True)

    # Thin separator
    with col_sep:
        st.markdown(
            '<div style="width:1px; background:#2a2a2a; min-height:420px; margin: 4px auto;"></div>',
            unsafe_allow_html=True
        )

    with col_result:
        if run_btn:
            results, max_sim = recommend_games(
                genre=genre_choice,
                os_pref=os_choice,
                budget=budget_choice,
                min_ram=ram_choice,
                top_n=5,
            )

            if results.empty:
                st.markdown("""
                <div class="no-result">
                    ⛔ &nbsp; No games found matching your criteria.<br>
                    <span style="font-size:0.85rem; letter-spacing:0;">
                        Try increasing budget or RAM, or change genre.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="section-title" style="font-size:1.1rem;">Top {len(results)} Recommendations</div>',
                    unsafe_allow_html=True
                )

                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    sim_pct = int(row["Similarity"] * 100)
                    price_str = "FREE" if row["Price_USD"] == 0 else f"${row['Price_USD']:.2f}"

                    st.markdown(f"""
                    <div class="game-card">
                        <div class="card-rank">#{rank}</div>
                        <div class="card-name">{row['Name']}</div>
                        <div class="card-genre">{row['Genre']}</div>
                        <div class="card-meta">
                            <div class="meta-item">🏷️ <span class="meta-value">{price_str}</span></div>
                            <div class="meta-item">⭐ <span class="meta-value">{row['Rating']}/10</span></div>
                            <div class="meta-item">💾 <span class="meta-value">{row['RAM_GB']} GB RAM</span></div>
                            <div class="meta-item">🖥️ <span class="meta-value">{row['OS']}</span></div>
                            <div class="meta-item">🎮 <span class="meta-value">{row['Developer']}</span></div>
                        </div>
                        <div class="similarity-bar-bg">
                            <div class="similarity-bar" style="width:{sim_pct}%;"></div>
                        </div>
                        <div class="similarity-label">MATCH SCORE — {sim_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 60px 20px; text-align:center;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:3rem; color:#2a2a2a;">🎮</div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem;
                            letter-spacing:3px; color:#444; text-transform:uppercase; margin-top:12px;">
                    Configure & Run Prediction
                </div>
                <div style="font-size:0.85rem; color:#333; margin-top:6px;">
                    Your recommendations will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 2 — DATA ANALYTICS (Interactive EDA)
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Data Analytics</div>', unsafe_allow_html=True)

    # ── FILTER BAR ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:6px;'
        'padding:4px 24px 16px 24px;margin-bottom:20px;">'
        '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.72rem;'
        'letter-spacing:3px;text-transform:uppercase;color:#E10600;margin:14px 0 10px;">'
        '🔧 Filter Visualisasi — semua chart berubah sesuai pilihan</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    fa_col1, fa_col2, fa_col3 = st.columns([2, 1.5, 1.5])

    with fa_col1:
        all_genres_eda = sorted(df["Genre"].unique().tolist())
        selected_genres = st.multiselect(
            "Filter Genre (bisa pilih lebih dari 1)",
            options=all_genres_eda,
            default=all_genres_eda,
            key="eda_genre",
        )

    with fa_col2:
        all_gpu = ["Low", "Medium", "High"]
        selected_gpu = st.multiselect(
            "Filter GPU Tier",
            options=all_gpu,
            default=all_gpu,
            key="eda_gpu",
        )

    with fa_col3:
        budget_range = st.slider(
            "Range Budget (USD)",
            min_value=0,
            max_value=70,
            value=(0, 70),
            step=1,
            key="eda_budget",
        )

    # ── Apply filters ────────────────────────────────────────────────────────
    if not selected_genres:
        selected_genres = all_genres_eda
    if not selected_gpu:
        selected_gpu = all_gpu

    df_eda = df[
        df["Genre"].isin(selected_genres) &
        df["GPU_Tier"].isin(selected_gpu) &
        df["Price_USD"].between(budget_range[0], budget_range[1])
    ].copy()

    n_games  = len(df_eda)
    avg_rating = df_eda["Rating"].mean() if n_games > 0 else 0.0
    avg_price  = df_eda["Price_USD"].mean() if n_games > 0 else 0.0
    free_count = int((df_eda["Price_USD"] == 0).sum()) if n_games > 0 else 0

    # ── Summary KPI cards ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;gap:14px;margin-bottom:24px;flex-wrap:wrap;">'
        f'<div style="background:#161616;border:1px solid #2a2a2a;border-left:3px solid #E10600;'
        f'border-radius:4px;padding:14px 20px;flex:1;min-width:110px;">'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.62rem;letter-spacing:2px;'
        f'text-transform:uppercase;color:#555;">Total Game</div>'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:2rem;font-weight:900;'
        f'color:#f0f0f0;">{n_games}</div></div>'
        f'<div style="background:#161616;border:1px solid #2a2a2a;border-left:3px solid #E10600;'
        f'border-radius:4px;padding:14px 20px;flex:1;min-width:110px;">'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.62rem;letter-spacing:2px;'
        f'text-transform:uppercase;color:#555;">Avg Rating</div>'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:2rem;font-weight:900;'
        f'color:#f0f0f0;">{avg_rating:.1f}<span style="font-size:0.85rem;color:#555;"> /10</span></div></div>'
        f'<div style="background:#161616;border:1px solid #2a2a2a;border-left:3px solid #E10600;'
        f'border-radius:4px;padding:14px 20px;flex:1;min-width:110px;">'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.62rem;letter-spacing:2px;'
        f'text-transform:uppercase;color:#555;">Avg Harga</div>'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:2rem;font-weight:900;'
        f'color:#f0f0f0;">${avg_price:.0f}</div></div>'
        f'<div style="background:#161616;border:1px solid #2a2a2a;border-left:3px solid #E10600;'
        f'border-radius:4px;padding:14px 20px;flex:1;min-width:110px;">'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.62rem;letter-spacing:2px;'
        f'text-transform:uppercase;color:#555;">Game Gratis</div>'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:2rem;font-weight:900;'
        f'color:#f0f0f0;">{free_count}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_games == 0:
        st.markdown(
            '<div class="no-result">⛔ &nbsp; Tidak ada data yang cocok dengan filter yang dipilih.<br>'
            '<span style="font-size:0.85rem;">Coba ubah kombinasi filter di atas.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        # ── ROW 1: Scatter Harga vs Rating  |  Bar Avg Rating per Genre ─────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.8rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#888;margin-bottom:6px;">'
                '💰 Harga vs Rating — tiap titik = 1 game</div>',
                unsafe_allow_html=True,
            )
            fig1 = px.scatter(
                df_eda,
                x="Price_USD", y="Rating",
                color="Genre",
                size=[14] * len(df_eda),
                hover_name="Name",
                hover_data={"Price_USD": ":.2f", "Rating": True,
                            "Developer": True, "Genre": True},
                labels={"Price_USD": "Harga (USD)", "Rating": "Rating"},
                template="plotly_dark",
            )
            fig1.update_layout(
                paper_bgcolor="#161616", plot_bgcolor="#0d0d0d",
                font=dict(family="Barlow Condensed", color="#aaa"),
                legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", font=dict(size=10)),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#222", zerolinecolor="#333", title="Harga (USD)"),
                yaxis=dict(gridcolor="#222", zerolinecolor="#333", title="Rating"),
            )
            fig1.update_traces(marker=dict(line=dict(width=0), opacity=0.9))
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            st.markdown(
                '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.8rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#888;margin-bottom:6px;">'
                '⭐ Rata-rata Rating per Genre</div>',
                unsafe_allow_html=True,
            )
            genre_rating = (
                df_eda.groupby("Genre")["Rating"]
                .mean().sort_values(ascending=True).reset_index()
            )
            genre_rating["RatingLabel"] = genre_rating["Rating"].round(2)
            fig2 = px.bar(
                genre_rating, x="Rating", y="Genre",
                orientation="h",
                labels={"Rating": "Avg Rating", "Genre": ""},
                template="plotly_dark",
                color="Rating",
                color_continuous_scale=["#3a0000", "#8B0400", "#E10600", "#ff6644"],
                text="RatingLabel",
            )
            fig2.update_layout(
                paper_bgcolor="#161616", plot_bgcolor="#0d0d0d",
                font=dict(family="Barlow Condensed", color="#aaa"),
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(gridcolor="#222", range=[0, 11]),
                yaxis=dict(gridcolor="#1a1a1a"),
                coloraxis_showscale=False,
            )
            fig2.update_traces(textposition="outside", textfont_size=11)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<hr class="red-divider">', unsafe_allow_html=True)

        # ── ROW 2: RAM Bar  |  GPU Tier Pie ─────────────────────────────────
        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown(
                '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.8rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#888;margin-bottom:6px;">'
                '💾 Distribusi Kebutuhan RAM Minimum</div>',
                unsafe_allow_html=True,
            )
            ram_count = (
                df_eda["RAM_GB"].value_counts()
                .sort_index().reset_index()
            )
            ram_count.columns = ["RAM_GB", "Jumlah"]
            ram_count["Label"] = ram_count["RAM_GB"].astype(str) + " GB"
            fig3 = px.bar(
                ram_count, x="Label", y="Jumlah",
                template="plotly_dark",
                color="Jumlah",
                color_continuous_scale=["#1a0000", "#8B0400", "#E10600"],
                labels={"Label": "RAM Minimum", "Jumlah": "Jumlah Game"},
                text="Jumlah",
            )
            fig3.update_layout(
                paper_bgcolor="#161616", plot_bgcolor="#0d0d0d",
                font=dict(family="Barlow Condensed", color="#aaa"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#222"),
                yaxis=dict(gridcolor="#222"),
                coloraxis_showscale=False,
            )
            fig3.update_traces(textposition="outside", textfont_size=12)
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.markdown(
                '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.8rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#888;margin-bottom:6px;">'
                '🖥️ Proporsi GPU Tier</div>',
                unsafe_allow_html=True,
            )
            gpu_count = df_eda["GPU_Tier"].value_counts().reset_index()
            gpu_count.columns = ["GPU_Tier", "Count"]
            fig4 = px.pie(
                gpu_count, names="GPU_Tier", values="Count",
                hole=0.55, template="plotly_dark",
                color="GPU_Tier",
                color_discrete_map={"Low": "#555", "Medium": "#E10600", "High": "#ff6644"},
            )
            fig4.update_layout(
                paper_bgcolor="#161616",
                font=dict(family="Barlow Condensed", color="#aaa"),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", font=dict(size=11)),
            )
            fig4.update_traces(
                textfont_size=12,
                marker=dict(line=dict(color="#0d0d0d", width=2)),
                pull=[0.04, 0, 0],
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<hr class="red-divider">', unsafe_allow_html=True)

        # ── ROW 3: Top 10 Game by Rating (full width) ────────────────────────
        st.markdown(
            '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:0.8rem;'
            'letter-spacing:2px;text-transform:uppercase;color:#888;margin-bottom:6px;">'
            '🏆 Top 10 Game — Rating Tertinggi (sesuai filter aktif)</div>',
            unsafe_allow_html=True,
        )
        top10 = (
            df_eda.nlargest(10, "Rating")
            [["Name", "Genre", "Rating", "Price_USD", "GPU_Tier"]]
            .sort_values("Rating")
            .reset_index(drop=True)
        )
        price_labels = top10["Price_USD"].apply(
            lambda p: "FREE" if p == 0 else f"${p:.2f}"
        )
        top10["Info"] = top10["Genre"] + " · " + price_labels
        fig5 = px.bar(
            top10, x="Rating", y="Name",
            orientation="h",
            color="Genre",
            text="Rating",
            hover_data={"Price_USD": True, "GPU_Tier": True, "Info": False},
            template="plotly_dark",
            labels={"Rating": "Rating", "Name": ""},
        )
        fig5.update_layout(
            paper_bgcolor="#161616", plot_bgcolor="#0d0d0d",
            font=dict(family="Barlow Condensed", color="#aaa"),
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis=dict(gridcolor="#222", range=[0, 11]),
            yaxis=dict(gridcolor="#1a1a1a"),
            legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", font=dict(size=10)),
            height=360,
        )
        fig5.update_traces(textposition="outside", textfont_size=11)
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════
#  TAB 3 — TEAM PROFILE
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Team Profile</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.6, 1])

    # ── LEFT: Team & Members ──────────────────────────────────────────────
    with col_left:

        # Team header
        st.markdown(
            '<div class="profile-outer">'
            '<div class="profile-team-title">Kelompok 12</div>'
            '<div class="profile-team-sub">Proyek Sains Data &nbsp;&middot;&nbsp; Institut Teknologi Sepuluh Nopember</div>'
            # ── Member 1
            '<div class="member-card">'
            '<div class="member-name">Ghalib Ibrahim Zardy</div>'
            '<div class="member-nrp">NRP &nbsp;5052231028</div>'
            '<div class="member-focus">'
            'Bertanggung jawab atas fondasi analitik proyek &mdash; mulai dari eksplorasi data mendalam '
            'hingga rekayasa fitur yang menjadi tulang punggung akurasi sistem rekomendasi.'
            '</div>'
            '<div style="margin-top:12px;">'
            '<span class="member-tag">EDA &amp; Visualisasi</span>'
            '<span class="member-tag">Feature Engineering</span>'
            '<span class="member-tag">Filtering Logic</span>'
            '<span class="member-tag">Deployment</span>'
            '<span class="member-tag">Literature Review</span>'
            '</div>'
            '</div>'
            # ── Member 2
            '<div class="member-card">'
            '<div class="member-name">M Shah Aquilla Febryano</div>'
            '<div class="member-nrp">NRP &nbsp;5052231043</div>'
            '<div class="member-focus">'
            'Memimpin pengembangan pipeline machine learning &mdash; dari pembersihan data mentah '
            'hingga implementasi model TF-IDF dan Cosine Similarity beserta antarmuka pengguna.'
            '</div>'
            '<div style="margin-top:12px;">'
            '<span class="member-tag">Data Cleaning</span>'
            '<span class="member-tag">TF-IDF Vectorizer</span>'
            '<span class="member-tag">Cosine Similarity</span>'
            '<span class="member-tag">Model Dev</span>'
            '<span class="member-tag">UI Design</span>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT: Metode + Dataset + Target Evaluasi ─────────────────────────
    with col_right:

        # Metode box
        st.markdown(
            '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:6px;padding:24px;margin-bottom:14px;">'
            '<div style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:0.75rem;'
            'letter-spacing:3px;text-transform:uppercase;color:#E10600;margin-bottom:16px;">Metode</div>'
            '<div style="font-size:0.88rem;color:#aaa;line-height:1.8;">'
            '<div style="color:#f0f0f0;font-weight:600;margin-bottom:4px;">Content-Based Filtering</div>'
            'Rekomendasi berbasis kemiripan konten tanpa memerlukan data pengguna lain <em>(Cold-Start Safe)</em>.'
            '<hr style="border-color:#2a2a2a;margin:14px 0;">'
            '<div style="color:#f0f0f0;font-weight:600;margin-bottom:4px;">TF-IDF Vectorization</div>'
            'Fitur tekstual (genre, OS, GPU) dikonversi ke representasi numerik berbobot.'
            '<hr style="border-color:#2a2a2a;margin:14px 0;">'
            '<div style="color:#f0f0f0;font-weight:600;margin-bottom:4px;">Cosine Similarity</div>'
            'Mengukur sudut antara vektor preferensi pengguna dan vektor tiap game dalam dataset.'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # Dataset box
        st.markdown(
            '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:6px;padding:24px;margin-bottom:14px;">'
            '<div style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:0.75rem;'
            'letter-spacing:3px;text-transform:uppercase;color:#E10600;margin-bottom:16px;">Dataset</div>'
            '<div style="font-size:0.88rem;color:#aaa;line-height:1.8;">'
            '<span style="color:#f0f0f0;">Kaggle Steam Games Dataset</span><br>'
            '50,000+ game entries<br>'
            'Fitur: Genre, Price, OS, RAM, GPU, Rating<br>'
            '<a href="https://www.kaggle.com/datasets/fronkongames/steam-games-dataset" '
            'style="color:#E10600;font-size:0.78rem;text-decoration:none;" target="_blank">'
            '&#8594; kaggle.com/datasets/steam-games</a>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # Target Evaluasi — build rows first, then inject once
        metrics = [
            ("Precision@10", "≥ 0.70",
             "Dari 10 rekomendasi teratas, minimal 70% harus benar-benar sesuai genre & spesifikasi pengguna. "
             "Dihitung otomatis dari hasil filter + similarity score."),
            ("Recall@20", "≥ 0.50",
             "Dari semua game relevan di dataset, sistem harus berhasil menemukan minimal 50% di Top-20 hasil. "
             "Mengukur kelengkapan rekomendasi."),
            ("MAP", "≥ 0.65",
             "Mean Average Precision: rata-rata presisi di setiap posisi rekomendasi. "
             "Mengukur kualitas urutan ranking secara keseluruhan."),
            ("NDCG@10", "≥ 0.75",
             "Game relevan yang muncul di posisi atas diberi bobot lebih tinggi. "
             "Skor 0.75 berarti urutan rekomendasi sudah sangat mendekati ideal."),
            ("Coverage", "≥ 0.40",
             "Minimal 40% dari total game di dataset pernah muncul sebagai rekomendasi. "
             "Mencegah sistem hanya merekomendasikan game populer saja (popularity bias)."),
            ("Diversity", "≥ 0.60",
             "Variasi genre dalam 10 rekomendasi. Skor 0.60 = minimal 3 genre berbeda muncul, "
             "menghindari filter bubble."),
            ("User Satisfaction", "≥ 4.0 / 5.0",
             "Survei kepuasan nyata dari pengguna (skala Likert 1-5) mencakup relevansi genre, "
             "kompatibilitas perangkat, kesesuaian budget, dan novelty game. Target: ≥30 responden."),
        ]

        rows = ""
        for name, target, desc in metrics:
            rows += (
                '<div style="border-bottom:1px solid #222;padding:10px 0;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
                f'<span style="color:#f0f0f0;font-weight:600;font-size:0.85rem;">{name}</span>'
                f'<span style="color:#E10600;font-family:\'Barlow Condensed\',sans-serif;'
                f'font-weight:700;font-size:0.9rem;">{target}</span>'
                '</div>'
                f'<div style="font-size:0.76rem;color:#666;line-height:1.5;">{desc}</div>'
                '</div>'
            )

        st.markdown(
            '<div style="background:#161616;border:1px solid #2a2a2a;border-radius:6px;padding:24px;">'
            '<div style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:0.75rem;'
            'letter-spacing:3px;text-transform:uppercase;color:#E10600;margin-bottom:16px;">Target Evaluasi</div>'
            + rows +
            '</div>',
            unsafe_allow_html=True,
        )
