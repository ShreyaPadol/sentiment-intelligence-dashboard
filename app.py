"""
Mumbai Petrol Pump Intelligence Platform
Customer Sentiment Analysis Dashboard
"""

import re
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter

from sentiment_engine import (
    predict_sentiment_batch,
    classify_issue,
    classify_issues_multi,
    normalize_dataframe,
    normalize_station_dataframe,
)
from mumbai_nlp_pipeline import (
    fit_lda,
    get_topic_words,
    assign_topic_label,
    brand_scorecard,
    build_priority_queue,
    compute_aspect_matrix,
    top_ngrams,
    extract_keyphrases,
    zone_nlp_summary,
    review_quality_score,
)

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai Petrol Pump Intelligence",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
BRAND   = "#1e40af"
BRAND_L = "#3b82f6"
NEG     = "#dc2626"
POS     = "#059669"
NEU     = "#d97706"
GREY    = "#64748b"

SENTIMENT_COLORS = {"Positive": POS, "Neutral": NEU, "Negative": NEG}

ISSUE_COLORS = {
    "Meter Tampering & Fraud":  "#7c3aed",
    "Staff Behaviour":          "#dc2626",
    "Fuel Short Filling":       "#ea580c",
    "Waiting Time & Queue":     "#d97706",
    "Payment Methods":          "#0891b2",
    "CNG Availability":         "#0d9488",
    "Fuel Quality":             "#65a30d",
    "Cleanliness & Hygiene":    "#0ea5e9",
    "Safety Concern":           "#b91c1c",
    "Facility Maintenance":     "#9333ea",
    "Billing Issue":            "#c026d3",
    "Staff Helpfulness":        "#059669",
    "Air, Tyre & Nitrogen":     "#2563eb",
    "Operating Hours":          "#475569",
    "Traffic & Accessibility":  "#84cc16",
    "Amenities & ATM":          "#f59e0b",
    "Pricing":                  "#6366f1",
    "Other":                    "#94a3b8",
}

STOPWORDS = {
    "the","a","an","is","it","in","of","and","to","for","with","on","at","by",
    "from","this","was","are","be","not","but","have","had","has","they","their",
    "there","that","very","so","my","we","i","its","here","get","got","also",
    "just","no","or","as","do","did","been","all","one","if","up","out","about",
    "than","more","when","will","can","good","place","nice","would","like",
    "really","come","petrol","pump","station","fuel","filling","mumbai","pump",
    # pronouns & filler
    "you","your","yours","he","she","her","him","his","them","us","our","me",
    "they","their","we","who","whom","what","which","where","how","why","when",
    "this","these","those","then","than","even","said","told","went","came",
    "going","saying","asked","told","made","take","taken","took","give","given",
    "gave","put","use","used","using","want","wanted","need","needed","let",
    "via","per","too","yet","off","own","new","old","big","small","first","last",
    "much","many","some","any","few","every","each","both","either","other",
    "same","well","still","back","over","around","re","ll","ve","don","doesn",
    # domain-generic noise
    "petrol","pump","station","fuel","filling","oil","gas","mumbai","india",
    "please","always","never","today","yesterday","now","time","ago","visit",
    "visited","come","went","came","day","week","month","year","times",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"], .main {
    background: #f0f4f8 !important;
    font-family: 'Inter', sans-serif !important;
    color: #0f172a !important;
}
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1500px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1.25rem; }
[data-testid="stSidebar"] * { color: #0f172a !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #0f172a !important; }
[data-testid="stSidebar"] .stRadio label { color: #374151 !important; font-size: 0.83rem !important; }
[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; margin: 1.1rem 0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #f8fafc !important; border-color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #0f172a !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] { background: #dbeafe !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #1e40af !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 20px 22px 16px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.65rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.9rem !important; font-weight: 800 !important;
    color: #0f172a !important; letter-spacing: -0.03em !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.73rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    border-radius: 12px 12px 0 0;
    padding: 0 8px;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.73rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 12px 20px !important; border-radius: 0 !important;
    color: #64748b !important; background: transparent !important;
    border-bottom: 3px solid transparent !important; margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #1e40af !important; border-bottom-color: #1e40af !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 1.25rem !important; }

/* ── Cards ── */
.card {
    background: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); padding: 22px 24px;
    margin-bottom: 1.25rem;
}
.chart-card {
    background: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); overflow: hidden;
    margin-bottom: 1.25rem; transition: box-shadow 0.15s;
}
.chart-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.09); }

/* ── Section headers ── */
.section-hdr {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.62rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.12em; color: #94a3b8;
    padding: 2rem 0 0.75rem 0;
    border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;
}
.section-hdr .bar { width: 3px; height: 13px; background: #1e40af; border-radius: 2px; }

/* ── Insight cards ── */
.ic {
    border-radius: 10px; padding: 13px 16px;
    margin-bottom: 9px; border-left: 3px solid;
    font-size: 0.845rem; line-height: 1.75; color: #1e293b;
}
.ic b { color: #0f172a; font-weight: 700; }

/* ── Badge ── */
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 99px;
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em;
}

/* ── Tables ── */
table { border-collapse: collapse; width: 100%; font-size: 0.83rem; }
thead tr { background: #f8fafc; border-bottom: 2px solid #e2e8f0; }
thead th {
    font-size: 0.64rem !important; font-weight: 800 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    color: #64748b !important; padding: 10px 14px !important;
}
tbody tr { border-bottom: 1px solid #f1f5f9; }
tbody td { padding: 9px 14px !important; color: #1e293b !important; }
tbody tr:hover { background: #f8fafc !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0 !important; }

/* ── Inputs ── */
[data-baseweb="select"] > div, [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #ffffff !important; border-color: #e2e8f0 !important; color: #0f172a !important;
}
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"],
.stRadio label, .stSelectbox label,
[data-testid="stSelectbox"] label p,
[data-testid="stRadio"] label p {
    color: #1e293b !important; font-weight: 500 !important; font-size: 0.83rem !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #1e293b !important; font-weight: 500 !important;
}
[data-baseweb="popover"] { background: #ffffff !important; }
[data-baseweb="popover"] li { background: #ffffff !important; color: #0f172a !important; font-size: 0.83rem !important; }
[data-baseweb="popover"] li:hover { background: #f8fafc !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: #1e40af !important; color: #ffffff !important;
    border-radius: 8px !important; border: none !important;
    font-size: 0.73rem !important; font-weight: 700 !important;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 10px 22px !important;
}
.stDownloadButton > button:hover { background: #1d4ed8 !important; }

/* ── Modebar ── */
.modebar { opacity: 0 !important; transition: opacity 0.2s; }
.modebar:hover { opacity: 1 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.855rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def cL(text=""):
    st.markdown(
        f'<div class="section-hdr"><div class="bar"></div>{text}</div>',
        unsafe_allow_html=True,
    )


def chart_card(fig):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def ic(color_pair, html):
    bg, border = color_pair
    st.markdown(
        f'<div class="ic" style="background:{bg};border-left-color:{border};">{html}</div>',
        unsafe_allow_html=True,
    )


INS = {
    "red":    ("#fef2f2", "#dc2626"),
    "amber":  ("#fffbeb", "#d97706"),
    "green":  ("#f0fdf4", "#059669"),
    "blue":   ("#eff6ff", "#1e40af"),
    "purple": ("#f5f3ff", "#7c3aed"),
    "teal":   ("#f0fdfa", "#0d9488"),
}

def plot_layout(**kw):
    base = dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=11, color="#1e293b"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        hoverlabel=dict(font_size=11, font_family="Inter, sans-serif", bgcolor="#ffffff"),
        xaxis=dict(tickfont=dict(color="#475569", size=10), title_font=dict(color="#475569", size=11),
                   linecolor="#e2e8f0", gridcolor="#f8fafc"),
        yaxis=dict(tickfont=dict(color="#475569", size=10), title_font=dict(color="#475569", size=11),
                   linecolor="#e2e8f0", gridcolor="#f8fafc"),
        legend=dict(font=dict(color="#1e293b", size=12, family="Inter, sans-serif"), title_font=dict(color="#1e293b", size=11)),
        margin=dict(t=36, b=28, l=16, r=16),
    )
    base.update(kw)
    return base




def top_keywords(text, n=12):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return Counter(w for w in words if w not in STOPWORDS).most_common(n)


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_default_pune():
    return pd.read_excel("Pune_Retail_outlet (1).xlsx")


@st.cache_data(show_spinner=False)
def load_default_mumbai():
    return pd.read_excel("mumbai_petrol_pumps.xlsx")


@st.cache_data(show_spinner=False)
def load_mumbai_nlp():
    import os
    nlp = "mumbai_petrol_reviews_nlp.csv"
    raw = "mumbai_petrol_reviews.csv"
    if os.path.exists(nlp) and (not os.path.exists(raw) or
       os.path.getmtime(nlp) >= os.path.getmtime(raw)):
        return pd.read_csv(nlp)
    if not os.path.exists(raw):
        return None
    df = pd.read_csv(raw)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].reset_index(drop=True)
    with st.spinner("Analysing customer reviews — this may take a moment on first load…"):
        results = predict_sentiment_batch(df["text"].fillna("").tolist(), batch_size=64)
    df["sentiment"]      = [r["sentiment"]   for r in results]
    df["confidence"]     = [r["confidence"]  for r in results]
    df["bert_star"]      = [r["star_rating"] for r in results]
    df["issue_category"] = df["text"].apply(classify_issue)
    df["issue_tags"]     = df["text"].apply(lambda x: ", ".join(classify_issues_multi(x, top_n=3)))
    df["quality_score"]  = df["text"].apply(review_quality_score)
    df.to_csv(nlp, index=False)
    return df


def is_station_level(df):
    col_lower = {c.lower().strip() for c in df.columns}
    return not bool({"review_text_final", "text", "review", "review_text", "comment", "body"} & col_lower)


@st.cache_data(show_spinner=False)
def run_bert(texts_tuple):
    return predict_sentiment_batch(list(texts_tuple), batch_size=32)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown("""
    <div style="padding:0 0 1.25rem 0;border-bottom:1px solid #e2e8f0;margin-bottom:1.25rem;">
        <div style="font-size:1.05rem;font-weight:800;color:#0f172a;letter-spacing:-0.02em;">⛽ Mumbai Fuel Intelligence</div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:3px;font-weight:400;">Customer Sentiment Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#64748b;margin-bottom:8px;">Select Dataset</div>', unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "Dataset",
        ["Mumbai Petrol Pumps", "Pune Retail Outlets", "Upload Your Own"],
        label_visibility="collapsed",
    )

    df_raw = None
    nlp_mode = False

    if mode == "Mumbai Petrol Pumps":
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#64748b;margin-bottom:8px;">Analysis Mode</div>', unsafe_allow_html=True)
        view = st.sidebar.radio(
            "View",
            ["Customer Review Analysis", "Station-Level Overview"],
            label_visibility="collapsed",
        )
        if view == "Customer Review Analysis":
            df_raw = load_mumbai_nlp()
            if df_raw is None or df_raw.empty:
                st.sidebar.error("No scraped data. Run:\n```python the data collection script```")
            else:
                st.sidebar.markdown(f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-top:8px;">
                    <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Customer Reviews</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#1e40af;margin-top:2px;">{len(df_raw):,}</div>
                    <div style="font-size:0.71rem;color:#64748b;margin-bottom:10px;">Verified Google Maps reviews</div>
                    <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Stations Covered</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#1e40af;margin-top:2px;">{df_raw.groupby(['lat','lng']).ngroups if 'lat' in df_raw.columns and 'lng' in df_raw.columns else df_raw['station_name'].nunique()}</div>
                    <div style="font-size:0.71rem;color:#64748b;">Petrol pump locations</div>
                </div>
                """, unsafe_allow_html=True)
                nlp_mode = True
        else:
            df_raw = load_default_mumbai()
            st.sidebar.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-top:8px;">
                <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Total Stations</div>
                <div style="font-size:1.2rem;font-weight:800;color:#1e40af;margin-top:2px;">{len(df_raw):,}</div>
                <div style="font-size:0.71rem;color:#64748b;">Mumbai petrol pump network</div>
            </div>
            """, unsafe_allow_html=True)

    elif mode == "Pune Retail Outlets":
        df_raw = load_default_pune()
        st.sidebar.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-top:8px;">
            <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Customer Reviews</div>
            <div style="font-size:1.2rem;font-weight:800;color:#1e40af;margin-top:2px;">{len(df_raw):,}</div>
            <div style="font-size:0.71rem;color:#64748b;">Pune retail outlet reviews</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        uploaded = st.sidebar.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx", "xls"])
        if uploaded:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.sidebar.success(f"{len(df_raw):,} rows loaded")
        else:
            st.sidebar.info("Upload a file to begin.")

    st.session_state["mumbai_nlp_mode"] = nlp_mode

    # ── Sidebar filters (NLP mode) ──
    if nlp_mode and df_raw is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#64748b;margin-bottom:8px;">Refine Results</div>', unsafe_allow_html=True)

        brands = sorted(df_raw["brand"].dropna().unique())
        sel_brands = st.sidebar.multiselect("Brand", brands, default=brands)
        zones = sorted(df_raw["zone"].dropna().unique())
        sel_zones = st.sidebar.multiselect("Zone", zones, default=zones)
        cats = sorted(df_raw["category"].dropna().unique())
        sel_cats = st.sidebar.multiselect("Category", cats, default=cats)
        sents = ["Positive", "Neutral", "Negative"]
        sel_sents = st.sidebar.multiselect("Sentiment", sents, default=sents)

        df_raw = df_raw[
            df_raw["brand"].isin(sel_brands) &
            df_raw["zone"].isin(sel_zones) &
            df_raw["category"].isin(sel_cats) &
            df_raw["sentiment"].isin(sel_sents)
        ]

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:0.6rem;color:#94a3b8;text-align:center;padding-top:0.5rem;line-height:1.6;">Customer data sourced from Google Maps<br>Sentiment analysis powered by AI</div>', unsafe_allow_html=True)

    return df_raw


# ── Main router ────────────────────────────────────────────────────────────────

def main():
    df_raw = sidebar()
    if df_raw is None:
        _landing()
        return

    if st.session_state.get("mumbai_nlp_mode"):
        if df_raw.empty:
            st.warning("No data matches the current filters. Adjust the sidebar.")
            return
        dashboard_mumbai_nlp(df_raw)
        return

    if is_station_level(df_raw):
        dashboard_station(df_raw)
        return

    dashboard_reviews(df_raw)


def _landing():
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="font-size:3rem;">⛽</div>
        <div style="font-size:1.8rem;font-weight:800;color:#0f172a;margin-top:1rem;">Mumbai Petrol Pump Intelligence</div>
        <div style="font-size:1rem;color:#64748b;margin-top:0.5rem;max-width:480px;margin-left:auto;margin-right:auto;">
            Verified customer reviews · Automated sentiment classification · 17 issue categories · Brand & geographic analysis
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MUMBAI NLP DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def dashboard_mumbai_nlp(df):
    total   = len(df)
    pos     = (df["sentiment"] == "Positive").sum()
    neg     = (df["sentiment"] == "Negative").sum()
    neu     = (df["sentiment"] == "Neutral").sum()
    stations = df.groupby(["lat","lng"]).ngroups if "lat" in df.columns and "lng" in df.columns else df["station_name"].nunique()
    avg_conf = df["confidence"].mean() * 100

    # ── Page header ──
    st.markdown(f"""
    <div style="background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;
                padding:24px 28px;margin-bottom:1.75rem;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div style="display:flex;align-items:center;gap:16px;">
                <div style="width:48px;height:48px;background:linear-gradient(135deg,#1e40af,#3b82f6);
                            border-radius:12px;display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">⛽</div>
                <div>
                    <div style="font-size:1.2rem;font-weight:800;color:#0f172a;letter-spacing:-0.025em;">
                        Mumbai Petrol Pump Intelligence</div>
                    <div style="font-size:0.8rem;color:#64748b;margin-top:2px;">
                        Customer reviews from Google Maps &nbsp;·&nbsp; AI Sentiment Analysis &nbsp;·&nbsp;
                        17 issue categories &nbsp;·&nbsp; 5 zones
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:800;color:#1e40af;">{total:,}</div>
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Reviews</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:800;color:#1e40af;">{stations}</div>
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Stations</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:800;color:#dc2626;">{neg/total*100:.0f}%</div>
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Negative</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem;font-weight:800;color:#059669;">{pos/total*100:.0f}%</div>
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Positive</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "Overview",
        "Issue Analysis",
        "Brand Performance",
        "Zone Intelligence",
        "Text Analysis",
        "Trend Forecasting",
        "Station Report",
        "Data Export",
    ])

    with tabs[0]: tab_overview(df)
    with tabs[1]: tab_issues(df)
    with tabs[2]: tab_brands(df)
    with tabs[3]: tab_zones(df)
    with tabs[4]: tab_topics(df)
    with tabs[5]: tab_forecasting(df)
    with tabs[6]: tab_station_dive(df)
    with tabs[7]: tab_export(df)


# ── Tab 0: Overview ────────────────────────────────────────────────────────────

def tab_overview(df):
    total = len(df)
    pos   = (df["sentiment"] == "Positive").sum()
    neg   = (df["sentiment"] == "Negative").sum()
    neu   = (df["sentiment"] == "Neutral").sum()
    avg_stars  = df["bert_star"].mean() if "bert_star" in df.columns else 0
    avg_conf   = df["confidence"].mean() * 100 if "confidence" in df.columns else 0
    top_issue  = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()
    top_zone   = df[df["sentiment"] == "Negative"]["zone"].value_counts().index[0] if "zone" in df.columns else "—"

    cL("Performance Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Reviews Analysed", f"{total:,}")
    c2.metric("Positive",        f"{pos:,}",  f"{pos/total*100:.1f}% of total")
    c3.metric("Neutral",         f"{neu:,}",  f"{neu/total*100:.1f}% of total")
    c4.metric("Negative",        f"{neg:,}",  f"{neg/total*100:.1f}% of total")
    c5.metric("Avg Predicted Rating",  f"{avg_stars:.2f} / 5")
    c6.metric("Model Confidence",f"{avg_conf:.1f}%")

    # ── Sentiment donut + star bar ──
    cL("Sentiment Distribution")
    col1, col2 = st.columns(2)
    with col1:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(counts, names="Sentiment", values="Count",
                     color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.58)
        fig.update_traces(
            textposition="outside", textinfo="percent+label",
            textfont=dict(size=11, color="#1e293b"),
            marker=dict(line=dict(color="#ffffff", width=2)),
        )
        fig.update_layout(**plot_layout(height=320, showlegend=False, margin=dict(t=20,b=20,l=20,r=20)))
        chart_card(fig)

    with col2:
        if "bert_star" in df.columns:
            star_sent = df.groupby(["bert_star", "sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(
                star_sent, x="bert_star", y="count",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"bert_star": "Predicted Star Rating (1–5)", "count": "Reviews", "sentiment": "Sentiment"},
                category_orders={"bert_star": [1, 2, 3, 4, 5]},
            )
            fig2.update_xaxes(
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1 ★", "2 ★", "3 ★", "4 ★", "5 ★"],
                range=[0.4, 5.6],
                tickfont=dict(color="#1e293b", size=11),
            )
            fig2.update_layout(**plot_layout(
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                            font=dict(color="#1e293b", size=12)),
            ))
            chart_card(fig2)

    # ── Issue breakdown ──
    cL("Customer Complaint Categories")
    ic_counts = df["issue_category"].value_counts().reset_index()
    ic_counts.columns = ["Issue", "Count"]
    ic_counts["Color"] = ic_counts["Issue"].map(ISSUE_COLORS).fillna("#94a3b8")
    ic_counts["Pct"] = (ic_counts["Count"] / total * 100).round(1)

    fig3 = px.bar(
        ic_counts.sort_values("Count"), x="Count", y="Issue", orientation="h",
        color="Issue", color_discrete_map=ISSUE_COLORS,
        text="Pct",
        labels={"Count": "Number of Reviews", "Issue": ""},
    )
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=10)
    fig3.update_layout(**plot_layout(height=480, showlegend=False,
                                     margin=dict(t=16, b=16, l=20, r=60)))
    chart_card(fig3)

    # ── Top keywords ──
    cL("Frequently Mentioned Terms by Sentiment")
    kc1, kc2, kc3 = st.columns(3)
    for col, sentiment, color in [
        (kc1, "Positive", POS), (kc2, "Neutral", NEU), (kc3, "Negative", NEG)
    ]:
        blob = " ".join(df[df["sentiment"] == sentiment]["text"].dropna())
        kws  = top_keywords(blob, n=12)
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Word", "Freq"]).sort_values("Freq")
            fig_k = px.bar(kw_df, x="Freq", y="Word", orientation="h",
                           color_discrete_sequence=[color],
                           labels={"Freq": "Frequency", "Word": ""})
            fig_k.update_layout(**plot_layout(height=360, showlegend=False,
                                               margin=dict(t=8,b=8,l=8,r=8)))
            with col:
                st.markdown(
                    f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.09em;color:{color};margin-bottom:6px;padding-left:4px;">'
                    f'{sentiment} Reviews</div>', unsafe_allow_html=True)
                chart_card(fig_k)

    # ── Auto insights ──
    cL("Operational Observations")
    neg_pct = neg / total * 100

    if neg_pct > 45:
        ic(INS["red"], f"<b>Critical: High Negativity Across the Network</b> — "
           f"{neg_pct:.1f}% of customer reviews are negative. This is significantly above "
           f"industry benchmarks and warrants immediate city-wide intervention.")
    elif neg_pct > 30:
        ic(INS["amber"], f"<b>Elevated Negativity ({neg_pct:.1f}%)</b> — "
           f"Nearly 1 in 3 reviews reflects a negative experience. Targeted operational "
           f"improvements at flagged stations can meaningfully lift network satisfaction.")
    else:
        ic(INS["green"], f"<b>Broadly Positive Network ({neg_pct:.1f}% negative)</b> — "
           f"Customer sentiment is healthy. Focus on sustaining best practices.")

    if not top_issue.empty:
        t, tc = top_issue.index[0], top_issue.iloc[0]
        ic(INS["red"], f"<b>Top Pain Point: {t}</b> — "
           f"{tc} negative reviews ({tc/neg*100:.1f}% of all negatives) cite this as the primary concern. "
           f"Immediate focus here will have the greatest impact on satisfaction scores.")

    fraud = df[df["issue_category"] == "Meter Tampering & Fraud"]
    if len(fraud) > 0:
        fraud_neg = (fraud["sentiment"] == "Negative").sum()
        ic(INS["purple"], f"<b>Fraud Allegations: {len(fraud)} Reviews Flagged</b> — "
           f"{fraud_neg} reviews explicitly mention meter tampering, scams or cheating. "
           f"This is a severe reputational and legal risk requiring immediate investigation.")

    ic(INS["blue"], f"<b>Worst-Affected Zone: {top_zone}</b> — "
       f"Concentrates the highest number of negative reviews among all five Mumbai zones. "
       f"Zone-level management review recommended.")

    staff_pos = ((df["issue_category"] == "Staff Helpfulness") & (df["sentiment"] == "Positive")).sum()
    if staff_pos > 0:
        ic(INS["teal"], f"<b>{staff_pos} Reviews Praise Staff</b> — "
           f"A meaningful positive signal. Stations with high staff ratings can serve as "
           f"training benchmarks for underperforming outlets.")


# ── Tab 1: Issue Analysis ──────────────────────────────────────────────────────

def tab_issues(df):
    total = len(df)
    neg   = (df["sentiment"] == "Negative").sum()

    cL("Issue Categories by Sentiment")
    issue_sent = df.groupby(["issue_category", "sentiment"]).size().reset_index(name="count")
    issue_order = df["issue_category"].value_counts().index.tolist()
    fig = px.bar(
        issue_sent, x="issue_category", y="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS, barmode="stack",
        category_orders={"issue_category": issue_order},
        labels={"issue_category": "", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig.update_xaxes(tickangle=35)
    fig.update_layout(**plot_layout(
        height=420, margin=dict(t=36, b=90, l=16, r=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1e293b", size=12)),
    ))
    chart_card(fig)

    col1, col2 = st.columns(2)

    with col1:
        cL("Complaint Rate by Issue Category")
        neg_rate = (
            df.groupby("issue_category")
            .apply(lambda x: (x["sentiment"] == "Negative").mean() * 100)
            .reset_index(name="Negative %")
            .sort_values("Negative %", ascending=True)
        )
        fig2 = px.bar(
            neg_rate, x="Negative %", y="issue_category", orientation="h",
            color="Negative %", color_continuous_scale=["#fef2f2","#fca5a5","#dc2626"],
            labels={"Negative %": "Negative Rate (%)", "issue_category": ""},
        )
        fig2.update_layout(**plot_layout(height=440, showlegend=False,
                                          coloraxis_showscale=False,
                                          margin=dict(t=16, b=16, l=16, r=16)))
        chart_card(fig2)

    with col2:
        cL("Distribution of Negative Feedback")
        neg_df = df[df["sentiment"] == "Negative"]["issue_category"].value_counts().reset_index()
        neg_df.columns = ["Issue", "Count"]
        fig3 = px.pie(
            neg_df, names="Issue", values="Count",
            color="Issue", color_discrete_map=ISSUE_COLORS, hole=0.45,
        )
        fig3.update_traces(
            textposition="outside", textinfo="percent+label",
            textfont=dict(size=9, color="#1e293b"),
            marker=dict(line=dict(color="#ffffff", width=1.5)),
        )
        fig3.update_layout(**plot_layout(height=440, showlegend=False,
                                          margin=dict(t=20,b=20,l=20,r=20)))
        chart_card(fig3)

    cL("Review Browser")
    sel_issue = st.selectbox("Select Issue Category", sorted(df["issue_category"].unique()), key="issue_sel")
    sel_sent  = st.radio("Sentiment", ["All", "Negative", "Positive", "Neutral"], horizontal=True, key="issue_sent_radio")
    sub = df[df["issue_category"] == sel_issue]
    if sel_sent != "All":
        sub = sub[sub["sentiment"] == sel_sent]

    st.markdown(f'<div style="font-size:0.78rem;color:#64748b;margin-bottom:8px;">{len(sub)} reviews matching filter</div>', unsafe_allow_html=True)

    display_cols = [c for c in ["station_name","brand","zone","author_name","rating","sentiment","confidence","text"]
                    if c in sub.columns]
    rename = {"station_name":"Station","brand":"Brand","zone":"Zone","author_name":"Reviewer",
               "rating":"Stars","sentiment":"Sentiment","confidence":"Confidence","text":"Review"}
    st.dataframe(
        sub[display_cols].rename(columns=rename).reset_index(drop=True),
        use_container_width=True, height=340,
    )


# ── Tab 2: Brand Performance ───────────────────────────────────────────────────

def tab_brands(df):
    sc = brand_scorecard(df, sentiment_col="sentiment", brand_col="brand",
                          rating_col="station_rating", review_count_col="station_reviews")

    cL("Brand-Level Sentiment Performance")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for _, row in sc.iterrows():
            fig.add_trace(go.Bar(
                name=row["Brand"], x=[row["Brand"]],
                y=[row["Positive %"]],
                marker_color=POS, showlegend=False,
                text=[f"{row['Positive %']:.1f}%"], textposition="outside",
            ))
        fig2 = px.bar(
            sc, x="Brand", y="Positive %", color="Brand",
            text=sc["Positive %"].apply(lambda x: f"{x:.1f}%"),
            labels={"Positive %": "Positive Sentiment (%)","Brand":""},
            color_discrete_sequence=["#1e40af","#0891b2","#059669","#7c3aed","#d97706","#dc2626"],
        )
        fig2.update_traces(textposition="outside", textfont_size=11)
        fig2.update_yaxes(range=[0, 80])
        fig2.update_layout(**plot_layout(height=340, showlegend=False))
        chart_card(fig2)

    with col2:
        fig3 = px.bar(
            sc.sort_values("Negative %", ascending=False),
            x="Brand", y="Negative %", color="Brand",
            text=sc.sort_values("Negative %", ascending=False)["Negative %"].apply(lambda x: f"{x:.1f}%"),
            labels={"Negative %": "Negative Sentiment (%)","Brand":""},
            color_discrete_sequence=["#dc2626","#ea580c","#d97706","#0891b2","#7c3aed","#059669"],
        )
        fig3.update_traces(textposition="outside", textfont_size=11)
        fig3.update_yaxes(range=[0, 45])
        fig3.update_layout(**plot_layout(height=340, showlegend=False))
        chart_card(fig3)

    cL("Google Rating vs Customer Sentiment Score")
    if "Avg Station Rating" in sc.columns:
        sc_plot = sc.dropna(subset=["Avg Station Rating"])
        fig4 = px.scatter(
            sc_plot, x="Avg Station Rating", y="Positive %",
            size="Reviews", color="Brand",
            hover_data=["Stations", "Negative %", "Reviews"],
            labels={"Avg Station Rating": "Google Avg Rating (1–5)",
                    "Positive %": "Positive Sentiment (%)"},
            size_max=55,
        )
        fig4.add_hline(y=50, line_dash="dash", line_color="#e2e8f0",
                       annotation_text="50% positive threshold")
        fig4.update_layout(**plot_layout(height=400))
        chart_card(fig4)
        st.caption("Bubble size = number of reviews. Brands positioned above the 50% threshold are performing well on cLP sentiment.")

    cL("Brand Performance Scorecard")
    display_sc = sc.copy()
    styled = (
        display_sc.style
            .background_gradient(subset=["Positive %"], cmap="Greens", vmin=30, vmax=70)
            .background_gradient(subset=["Negative %"], cmap="Reds", vmin=10, vmax=40)
            .format({"Positive %": "{:.1f}%", "Negative %": "{:.1f}%",
                     "Avg Station Rating": "{:.2f}", "Reviews": "{:,}",
                     "Stations": "{:.0f}"})
    )
    st.dataframe(styled, use_container_width=True)

    cL("Issue Breakdown by Oil Company")
    brand_issue = df.groupby(["brand", "issue_category"]).size().reset_index(name="count")
    fig5 = px.bar(
        brand_issue, x="brand", y="count", color="issue_category",
        color_discrete_map=ISSUE_COLORS, barmode="stack",
        labels={"brand": "Brand", "count": "Reviews", "issue_category": "Issue"},
    )
    fig5.update_layout(**plot_layout(
        height=400,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                    font=dict(size=10)),
        margin=dict(t=36, b=28, l=16, r=180),
    ))
    chart_card(fig5)

    cL("Brand-Level Observations")
    if not sc.empty:
        best  = sc.iloc[0]
        worst = sc.iloc[-1]
        ic(INS["green"], f"<b>Best Performing Brand: {best['Brand']}</b> — "
           f"{best['Positive %']:.1f}% positive customer sentiment across {int(best['Stations'])} stations "
           f"({int(best['Reviews'])} reviews). Avg Google rating: {best['Avg Station Rating']:.2f}★")
        ic(INS["red"], f"<b>Most Improvement Needed: {worst['Brand']}</b> — "
           f"{worst['Negative %']:.1f}% of reviews are negative. "
           f"Immediate focus on staff training and billing transparency recommended.")
        gap = best["Positive %"] - worst["Positive %"]
        ic(INS["amber"], f"<b>Performance Gap: {gap:.1f} percentage points</b> between the best and worst brand "
           f"suggests significant variance in operational standards across the city.")


# ── Tab 3: Zone Intelligence ───────────────────────────────────────────────────

def tab_zones(df):
    cL("Geographic Zone Performance")
    zone_stats = (
        df.groupby("zone")
        .agg(
            Reviews    =("sentiment", "count"),
            Positive   =("sentiment", lambda x: (x == "Positive").sum()),
            Negative   =("sentiment", lambda x: (x == "Negative").sum()),
            Neutral    =("sentiment", lambda x: (x == "Neutral").sum()),
            Avg_Rating =("station_rating", "mean"),
            Stations   =("station_address", "nunique"),
        )
        .assign(
            Pos_Pct=lambda x: x["Positive"] / x["Reviews"] * 100,
            Neg_Pct=lambda x: x["Negative"] / x["Reviews"] * 100,
        )
        .sort_values("Neg_Pct", ascending=False)
        .reset_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            zone_stats.sort_values("Neg_Pct"),
            x="Neg_Pct", y="zone", orientation="h",
            color="Neg_Pct",
            color_continuous_scale=["#fef2f2","#fca5a5","#dc2626"],
            text=zone_stats.sort_values("Neg_Pct")["Neg_Pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"Neg_Pct": "Negative Rate (%)", "zone": ""},
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(**plot_layout(height=320, showlegend=False,
                                         coloraxis_showscale=False,
                                         margin=dict(t=16,b=16,l=16,r=50)))
        chart_card(fig)

    with col2:
        zone_sent = df.groupby(["zone", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(
            zone_sent, x="zone", y="count",
            color="sentiment", color_discrete_map=SENTIMENT_COLORS, barmode="stack",
            labels={"zone": "", "count": "Reviews", "sentiment": "Sentiment"},
        )
        fig2.update_xaxes(tickangle=20)
        fig2.update_layout(**plot_layout(
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1e293b", size=12)),
        ))
        chart_card(fig2)

    cL("Issue Concentration by Zone")
    zone_issue = df.groupby(["zone", "issue_category"]).size().unstack(fill_value=0)
    fig3 = px.imshow(
        zone_issue, color_continuous_scale="Blues", aspect="auto",
        labels={"x": "Issue Category", "y": "Zone", "color": "Reviews"},
    )
    fig3.update_xaxes(tickangle=35)
    fig3.update_layout(**plot_layout(height=340, margin=dict(t=20, b=90, l=130, r=20)))
    chart_card(fig3)
    st.caption("Darker cells indicate a higher concentration of reviews mentioning a given issue in that geographic zone. Areas with dense colouring represent operational hotspots requiring targeted management intervention.")

    cL("Zone-Level Performance Summary")
    display_zone = zone_stats.copy()
    display_zone.columns = ["Zone", "Reviews", "Positive", "Negative", "Neutral",
                              "Avg Rating", "Stations", "Positive %", "Negative %"]
    styled = (
        display_zone.style
            .background_gradient(subset=["Negative %"], cmap="Reds")
            .background_gradient(subset=["Positive %"], cmap="Greens")
            .background_gradient(subset=["Avg Rating"],  cmap="RdYlGn")
            .format({"Avg Rating": "{:.2f}", "Positive %": "{:.1f}%", "Negative %": "{:.1f}%"})
    )
    st.dataframe(styled, use_container_width=True)

    cL("Zone-Level Observations")
    worst_zone = zone_stats.iloc[0]
    best_zone  = zone_stats.iloc[-1]
    ic(INS["red"], f"<b>Worst Zone: {worst_zone['zone']}</b> — "
       f"{worst_zone['Neg_Pct']:.1f}% negative rate across {int(worst_zone['Reviews'])} reviews. "
       f"Priority for operational audit and zone-manager escalation.")
    ic(INS["green"], f"<b>Best Zone: {best_zone['zone']}</b> — "
       f"Only {best_zone['Neg_Pct']:.1f}% negative rate. "
       f"Benchmark this zone's practices for city-wide rollout.")
    gap = worst_zone["Neg_Pct"] - best_zone["Neg_Pct"]
    ic(INS["amber"], f"<b>{gap:.1f}% gap</b> in negativity rate between worst and best zone. "
       f"Standardising practices across zones could significantly improve the overall network score.")


# ── Tab 4: Topics & Keywords ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_lda(texts_tuple, n_topics=8):
    lda, vec, doc_topics = fit_lda(list(texts_tuple), n_topics=n_topics)
    topic_words = get_topic_words(lda, vec, top_n=10)
    labels = {i: assign_topic_label(words) for i, words in topic_words}
    dominant = [labels[i] for i in doc_topics.argmax(axis=1)]
    return topic_words, labels, dominant


def tab_topics(df):
    cL("Topic Discovery")
    with st.spinner("Discovering themes in customer feedback…"):
        topic_words, labels, dominant = _cached_lda(
            tuple(df["text"].fillna("").tolist()), n_topics=8
        )
    df2 = df.copy()
    df2["topic"] = dominant

    # ── Topic frequency bar ──
    cL("Theme Frequency Distribution")
    tc = pd.Series(dominant).value_counts().reset_index()
    tc.columns = ["Topic", "Reviews"]
    fig = px.bar(
        tc.sort_values("Reviews"), x="Reviews", y="Topic", orientation="h",
        color="Reviews", color_continuous_scale="Blues",
        labels={"Reviews": "Number of Reviews", "Topic": ""},
        text="Reviews",
    )
    fig.update_traces(textposition="outside", textfont=dict(size=11, color="#1e293b"))
    fig.update_layout(**plot_layout(height=max(320, len(tc)*52),
                                     showlegend=False, coloraxis_showscale=False,
                                     margin=dict(t=16, b=16, l=16, r=60)))
    chart_card(fig)

    # ── Per-topic word table ──
    cL("Key Terms per Theme")
    # Build a clean table: one column per topic, top 8 words as rows
    topic_table = {}
    for i, words in topic_words:
        label = labels[i]
        # Filter out remaining noise
        clean_words = [w.title() for w in words
                       if len(w) > 3 and w not in {"other", "general", "service", "fill", "car", "air"}][:8]
        topic_table[label] = clean_words + [""] * (8 - len(clean_words))

    topic_df = pd.DataFrame(topic_table)
    topic_df.index = [f"Term {i+1}" for i in range(len(topic_df))]

    st.markdown('<div class="card" style="overflow-x:auto;">', unsafe_allow_html=True)
    st.dataframe(
        topic_df,
        use_container_width=True,
        hide_index=False,
        height=316,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Sentiment heatmap ──
    cL("Theme Distribution by Sentiment")
    pivot = df2.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
    # Reorder sentiment columns
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Positive", "Neutral", "Negative"]]
    fig2 = px.imshow(
        pivot, color_continuous_scale="Blues", aspect="auto",
        labels={"x": "Sentiment", "y": "Theme", "color": "Reviews"},
    )
    fig2.update_xaxes(tickfont=dict(color="#1e293b", size=12))
    fig2.update_yaxes(tickfont=dict(color="#1e293b", size=11))
    fig2.update_layout(**plot_layout(height=max(300, len(pivot)*52),
                                     margin=dict(t=20, b=20, l=200, r=20)))
    chart_card(fig2)

    cL("Frequently Occurring Phrases")
    col3, col4 = st.columns(2)
    neg_texts = df[df["sentiment"] == "Negative"]["text"].tolist()
    pos_texts = df[df["sentiment"] == "Positive"]["text"].tolist()

    with col3:
        st.markdown('<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:#dc2626;margin-bottom:6px;">Negative Bigrams</div>', unsafe_allow_html=True)
        bg_neg = top_ngrams(neg_texts, n=2, top_k=12)
        if bg_neg:
            bg_df = pd.DataFrame(bg_neg, columns=["Phrase","Count"]).sort_values("Count")
            fig3 = px.bar(bg_df, x="Count", y="Phrase", orientation="h",
                          color_discrete_sequence=[NEG],
                          labels={"Count":"Frequency","Phrase":""})
            fig3.update_layout(**plot_layout(height=360, showlegend=False,
                                              margin=dict(t=8,b=8,l=8,r=8)))
            chart_card(fig3)

    with col4:
        st.markdown('<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:#059669;margin-bottom:6px;">Positive Bigrams</div>', unsafe_allow_html=True)
        bg_pos = top_ngrams(pos_texts, n=2, top_k=12)
        if bg_pos:
            bg_df_p = pd.DataFrame(bg_pos, columns=["Phrase","Count"]).sort_values("Count")
            fig4 = px.bar(bg_df_p, x="Count", y="Phrase", orientation="h",
                          color_discrete_sequence=[POS],
                          labels={"Count":"Frequency","Phrase":""})
            fig4.update_layout(**plot_layout(height=360, showlegend=False,
                                              margin=dict(t=8,b=8,l=8,r=8)))
            chart_card(fig4)


# ── Tab 5: Priority Queue ──────────────────────────────────────────────────────

def tab_priority(df):
    cL("Station Intervention Priority")
    st.caption(
        "**Urgency Score** = (Negative %) × log(Google Review Count) × (5 − Avg Rating). "
        "Higher score = more evidence, lower rating, and higher negative rate → needs urgent attention."
    )

    pq = build_priority_queue(df, sentiment_col="sentiment",
                               rating_col="station_rating", review_count_col="station_reviews")

    cL("Stations Requiring Immediate Attention")
    top20 = pq.head(20).reset_index(drop=True)
    top20.index += 1  # 1-based rank

    fig = px.bar(
        top20[::-1].reset_index(), x="Urgency Score", y="Station", orientation="h",
        color="Urgency Score", color_continuous_scale=["#fef2f2","#fca5a5","#dc2626"],
        hover_data=["Zone","Brand","Avg Rating","Negative %"],
        labels={"Urgency Score":"Urgency Score","Station":""},
        text="Urgency Score",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=10)
    fig.update_layout(**plot_layout(height=540, showlegend=False,
                                     coloraxis_showscale=False,
                                     margin=dict(t=16,b=16,l=16,r=50)))
    chart_card(fig)

    col1, col2 = st.columns(2)
    with col1:
        cL("Urgency Distribution by Zone and Brand")
        heat = pq.groupby(["Zone","Brand"])["Urgency Score"].mean().unstack(fill_value=0)
        fig2 = px.imshow(heat, color_continuous_scale="OrRd", aspect="auto",
                         labels={"x":"Brand","y":"Zone","color":"Avg Urgency"})
        fig2.update_layout(**plot_layout(height=300, margin=dict(t=20,b=20,l=140,r=20)))
        chart_card(fig2)

    with col2:
        cL("Distribution of Negative Feedback Rates")
        fig3 = px.histogram(pq, x="Negative %", nbins=15, color_discrete_sequence=[NEG])
        fig3.add_vline(x=pq["Negative %"].mean(), line_dash="dash",
                       line_color=GREY, annotation_text=f"Avg {pq['Negative %'].mean():.1f}%")
        fig3.update_layout(**plot_layout(height=300, showlegend=False))
        chart_card(fig3)

    cL("Complete Priority Intervention List")
    styled_pq = (
        pq.style
            .background_gradient(subset=["Urgency Score"], cmap="Reds")
            .background_gradient(subset=["Negative %"],    cmap="Oranges")
            .background_gradient(subset=["Avg Rating"],    cmap="RdYlGn")
            .format({"Urgency Score":"{:.1f}","Negative %":"{:.1f}%",
                     "Avg Rating":"{:.2f}","Google Reviews":"{:,}"})
    )
    st.dataframe(styled_pq, use_container_width=True, height=460)

    csv = pq.to_csv(index=False).encode()
    st.download_button("⬇  Download Priority Queue CSV", data=csv,
                       file_name="mumbai_priority_queue.csv", mime="text/csv", key="dl_pq_tab")

    cL("Priority Observations")
    if not pq.empty:
        top = pq.iloc[0]
        ic(INS["red"], f"<b>Highest Priority: {top['Station']}</b> "
           f"({top['Zone']}, {top['Brand']}) — Urgency score <b>{top['Urgency Score']:.0f}</b>. "
           f"{top['Negative %']:.0f}% negative reviews · {int(top['Google Reviews']):,} Google reviews · "
           f"{top['Avg Rating']:.1f}★ avg rating.")
        critical = pq[pq["Negative %"] >= 40]
        if not critical.empty:
            ic(INS["amber"], f"<b>{len(critical)} stations have ≥ 40% negative reviews</b> — "
               f"These require urgent management intervention and full operational audit.")
        best = pq.iloc[-1]
        ic(INS["teal"], f"<b>Exemplary Station: {best['Station']}</b> — "
           f"Only {best['Negative %']:.0f}% negative reviews. "
           f"Use as a benchmark model for training and operational standards.")


# ── Tab 5b: Trend Forecasting (ARIMA / SARIMA) ────────────────────────────────

def _parse_relative_date(rel, ref_date=pd.Timestamp("2026-06-01")):
    """Convert a Google Maps relative date string to an approximate calendar month."""
    if not isinstance(rel, str):
        return None
    s = rel.lower().replace("edited ", "").strip()
    if "week" in s:
        return ref_date
    if re.match(r"a month", s):
        return ref_date - pd.DateOffset(months=1)
    m = re.match(r"(\d+)\s+month", s)
    if m:
        return ref_date - pd.DateOffset(months=int(m.group(1)))
    if re.match(r"a year", s):
        return ref_date - pd.DateOffset(months=12)
    m = re.match(r"(\d+)\s+year", s)
    if m:
        return ref_date - pd.DateOffset(months=int(m.group(1)) * 12)
    return None


@st.cache_data(show_spinner=False)
def _build_monthly_ts(idx_tuple, dates_tuple, sentiments_tuple, issues_tuple):
    df_ts = pd.DataFrame({
        "date_rel": list(dates_tuple),
        "sentiment": list(sentiments_tuple),
        "issue": list(issues_tuple),
    })
    df_ts["month"] = df_ts["date_rel"].apply(_parse_relative_date)
    df_ts = df_ts[df_ts["month"].notna()].copy()
    df_ts["month"] = pd.to_datetime(df_ts["month"]).dt.to_period("M").dt.to_timestamp()
    return df_ts


def _run_arima(series, steps=6, seasonal=False, m=12):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, m) if seasonal else (0, 0, 0, 0)
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=steps)
        fc_mean = forecast.predicted_mean
        fc_ci   = forecast.conf_int(alpha=0.20)
        return fit.fittedvalues, fc_mean, fc_ci, fit
    except Exception as e:
        return None, None, None, str(e)


def tab_forecasting(df):
    cL("Time-Series Overview")
    st.markdown("""
    <div class="ic" style="background:#eff6ff;border-left-color:#1e40af;font-size:0.84rem;line-height:1.75;margin-bottom:1rem;">
    <b>Methodology note</b> — Google Maps returns relative dates ("4 months ago", "2 years ago").
    These are reconstructed to calendar months relative to the data collection date (June 2026),
    carrying an estimated uncertainty of ±1 month. ARIMA and SARIMA models are then fitted to
    the resulting monthly time series. Forecasts should be interpreted as directional signals,
    not precise predictions.
    </div>
    """, unsafe_allow_html=True)

    df_ts = _build_monthly_ts(
        tuple(df.index),
        tuple(df["date_relative"].fillna("").tolist()),
        tuple(df["sentiment"].tolist()),
        tuple(df["issue_category"].tolist()),
    )

    if df_ts.empty or df_ts["month"].nunique() < 6:
        st.warning("Insufficient temporal data for forecasting (fewer than 6 distinct months detected).")
        return

    monthly_vol = df_ts.groupby("month").size().reset_index(name="Reviews").sort_values("month")

    cL("Monthly Review Volume")
    fig_vol = px.bar(monthly_vol, x="month", y="Reviews",
                     color_discrete_sequence=[BRAND_L],
                     labels={"month": "Month", "Reviews": "Number of Reviews"})
    fig_vol.update_layout(**plot_layout(height=300))
    chart_card(fig_vol)

    monthly_sent = df_ts.groupby(["month", "sentiment"]).size().reset_index(name="count").sort_values("month")

    cL("Monthly Sentiment Trend")
    fig_sent = px.line(monthly_sent, x="month", y="count", color="sentiment",
                       color_discrete_map=SENTIMENT_COLORS, markers=True,
                       labels={"month": "Month", "count": "Reviews", "sentiment": "Sentiment"})
    fig_sent.update_layout(**plot_layout(
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color="#1e293b", size=12)),
    ))
    chart_card(fig_sent)

    neg_rate_ts = (
        df_ts.groupby("month")
        .apply(lambda x: (x["sentiment"] == "Negative").mean() * 100)
        .reset_index(name="Negative Rate (%)")
        .sort_values("month")
    )

    cL("Monthly Negative Sentiment Rate")
    fig_neg = px.line(neg_rate_ts, x="month", y="Negative Rate (%)", markers=True,
                      color_discrete_sequence=[NEG], labels={"month": "Month"})
    fig_neg.add_hline(y=neg_rate_ts["Negative Rate (%)"].mean(), line_dash="dash",
                      line_color=GREY,
                      annotation_text=f"Avg {neg_rate_ts['Negative Rate (%)'].mean():.1f}%")
    fig_neg.update_layout(**plot_layout(height=300))
    chart_card(fig_neg)

    cL("Top Issue Categories Over Time")
    top5_issues = df_ts["issue"].value_counts().head(5).index.tolist()
    issue_ts = (
        df_ts[df_ts["issue"].isin(top5_issues)]
        .groupby(["month", "issue"]).size()
        .reset_index(name="count")
        .sort_values("month")
    )
    fig_issue = px.line(issue_ts, x="month", y="count", color="issue", markers=True,
                        color_discrete_map=ISSUE_COLORS,
                        labels={"month": "Month", "count": "Reviews", "issue": "Issue Category"})
    fig_issue.update_layout(**plot_layout(
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color="#1e293b", size=11)),
    ))
    chart_card(fig_issue)

    # ── ARIMA / SARIMA controls ──
    cL("ARIMA / SARIMA Forecast")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        forecast_target = st.selectbox(
            "Forecast metric",
            ["Total Review Volume", "Negative Review Count", "Negative Rate (%)"],
            key="fc_target",
        )
    with col_cfg2:
        model_type = st.radio("Model", ["ARIMA", "SARIMA"], horizontal=True, key="fc_model")
    with col_cfg3:
        fc_steps = st.slider("Forecast horizon (months)", 3, 12, 6, key="fc_steps")

    if forecast_target == "Total Review Volume":
        series = monthly_vol.set_index("month")["Reviews"]
        y_label = "Reviews per Month"
    elif forecast_target == "Negative Review Count":
        neg_counts = (
            df_ts[df_ts["sentiment"] == "Negative"]
            .groupby("month").size()
            .reindex(monthly_vol["month"], fill_value=0)
        )
        neg_counts.index = pd.DatetimeIndex(neg_counts.index)
        series = neg_counts
        y_label = "Negative Reviews per Month"
    else:
        series = neg_rate_ts.set_index("month")["Negative Rate (%)"]
        y_label = "Negative Rate (%)"

    series.index = pd.DatetimeIndex(series.index)
    series = series.sort_index()

    if len(series) < 6:
        st.warning("At least 6 monthly data points are required for forecasting.")
        return

    with st.spinner(f"Fitting {model_type} model…"):
        fitted, fc_mean, fc_ci, fit_obj = _run_arima(
            series, steps=fc_steps, seasonal=(model_type == "SARIMA"), m=12
        )

    if fc_mean is None:
        st.error(f"Model fitting failed: {fit_obj}")
        return

    last_date = series.index[-1]
    fc_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(fc_steps)]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines+markers",
        name="Actual", line=dict(color=BRAND, width=2), marker=dict(size=6),
    ))
    if fitted is not None:
        fig_fc.add_trace(go.Scatter(
            x=fitted.index, y=fitted.values, mode="lines",
            name="Fitted", line=dict(color=BRAND_L, width=1.5, dash="dot"),
        ))
    if fc_ci is not None:
        fig_fc.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0])[::-1],
            fill="toself", fillcolor="rgba(220,38,38,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Interval",
        ))
    fig_fc.add_trace(go.Scatter(
        x=fc_dates, y=fc_mean.values, mode="lines+markers",
        name="Forecast", line=dict(color=NEG, width=2.5, dash="dash"),
        marker=dict(symbol="diamond", size=7, color=NEG),
    ))
    fig_fc.add_vline(x=str(last_date), line_dash="dot", line_color=GREY,
                     annotation_text="Forecast starts", annotation_position="top left")
    fig_fc.update_layout(**plot_layout(
        height=420,
        title=dict(text=f"{model_type} Forecast — {forecast_target}", font=dict(size=13)),
        yaxis_title=y_label, xaxis_title="Month",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color="#1e293b", size=12)),
    ))
    chart_card(fig_fc)

    cL("Forecast Values")
    fc_df = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in fc_dates],
        "Forecast": fc_mean.values.round(1),
        "Lower 80% CI": fc_ci.iloc[:, 0].values.round(1),
        "Upper 80% CI": fc_ci.iloc[:, 1].values.round(1),
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    with st.expander("Model diagnostics (SARIMAX summary)"):
        if hasattr(fit_obj, "summary"):
            st.text(str(fit_obj.summary()))
        else:
            st.info("Diagnostics unavailable.")

    # ── Per-issue ARIMA ──
    cL("Issue-Level Forecasts (Top 4 Complaint Categories)")
    st.caption(
        "Individual ARIMA(1,1,1) models fitted per issue category using monthly negative review count. "
        "Rising slopes indicate worsening complaint volumes; declining slopes suggest improvement."
    )

    top4 = df_ts[df_ts["sentiment"] == "Negative"]["issue"].value_counts().head(4).index.tolist()
    cols_fc = st.columns(2)
    for idx, issue in enumerate(top4):
        issue_neg = (
            df_ts[(df_ts["issue"] == issue) & (df_ts["sentiment"] == "Negative")]
            .groupby("month").size()
            .reindex(monthly_vol["month"], fill_value=0)
        )
        issue_neg.index = pd.DatetimeIndex(issue_neg.index)
        issue_neg = issue_neg.sort_index()
        if len(issue_neg) < 6:
            continue
        _, fc_m, fc_c, _ = _run_arima(issue_neg, steps=4, seasonal=False)
        if fc_m is None:
            continue
        fc_dates_i = [issue_neg.index[-1] + pd.DateOffset(months=i + 1) for i in range(4)]
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=issue_neg.index, y=issue_neg.values, mode="lines+markers",
                                    name="Actual",
                                    line=dict(color=ISSUE_COLORS.get(issue, BRAND), width=2)))
        if fc_c is not None:
            fig_i.add_trace(go.Scatter(
                x=fc_dates_i + fc_dates_i[::-1],
                y=list(fc_c.iloc[:, 1]) + list(fc_c.iloc[:, 0])[::-1],
                fill="toself", fillcolor="rgba(220,38,38,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="80% CI",
            ))
        fig_i.add_trace(go.Scatter(x=fc_dates_i, y=fc_m.values, mode="lines+markers",
                                    name="Forecast",
                                    line=dict(color=NEG, width=2, dash="dash"),
                                    marker=dict(symbol="diamond", size=6)))
        fig_i.update_layout(**plot_layout(
            height=240,
            title=dict(text=issue, font=dict(size=11, color="#1e293b")),
            showlegend=False, margin=dict(t=36, b=28, l=16, r=16),
        ))
        with cols_fc[idx % 2]:
            chart_card(fig_i)

    cL("Forecasting Observations")
    trend_dir = "rising" if fc_mean.values[-1] > fc_mean.values[0] else "declining"
    color_key = "red" if (trend_dir == "rising" and "Negative" in forecast_target) else "green"
    ic(INS[color_key],
       f"<b>{model_type} Forecast — {forecast_target}</b>: The model projects a "
       f"<b>{trend_dir}</b> trend over the next {fc_steps} months, from "
       f"{fc_mean.values[0]:.1f} to {fc_mean.values[-1]:.1f}. "
       f"{'Warrants monitoring and pre-emptive operational action.' if trend_dir == 'rising' and 'Negative' in forecast_target else 'Suggests improving customer experience outcomes.'}")

    ic(INS["blue"],
       "<b>Interpretation guide</b> — ARIMA(1,1,1) models the review series using one lagged "
       "value, one differencing step, and one moving-average term. SARIMA(1,1,1)(1,1,1,12) "
       "additionally captures annual seasonality. The shaded band is the 80% confidence interval.")

    ic(INS["amber"],
       "<b>Limitation</b> — Relative date parsing introduces ±1-month uncertainty per review. "
       "Months with fewer than 5 reviews produce noisy estimates. Structural changes "
       "(new station openings, regulatory actions, fuel price shocks) are not captured by the model.")


# ── Tab 6: Station Deep-Dive ───────────────────────────────────────────────────

def tab_station_dive(df):
    cL("Individual Station Analysis")
    stations = sorted(df["station_name"].dropna().unique())
    sel = st.selectbox("Select a station", stations, key="station_sel")
    sdf = df[df["station_name"] == sel].copy()

    total   = len(sdf)
    pos     = (sdf["sentiment"] == "Positive").sum()
    neg     = (sdf["sentiment"] == "Negative").sum()
    neu     = (sdf["sentiment"] == "Neutral").sum()
    avg_r   = sdf["station_rating"].mean() if "station_rating" in sdf.columns else None
    brand   = sdf["brand"].iloc[0]  if "brand"  in sdf.columns else "—"
    zone    = sdf["zone"].iloc[0]   if "zone"   in sdf.columns else "—"
    cat     = sdf["category"].iloc[0] if "category" in sdf.columns else "—"

    # Station header card
    st.markdown(f"""
    <div class="card" style="border-left:4px solid #1e40af;">
        <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:center;">
            <div>
                <div style="font-size:1rem;font-weight:800;color:#0f172a;">{sel}</div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">{zone} &nbsp;·&nbsp; {brand} &nbsp;·&nbsp; {cat}</div>
            </div>
            <div style="display:flex;gap:24px;flex-wrap:wrap;">
                <div><div style="font-size:1.3rem;font-weight:800;color:#1e40af;">{total}</div><div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Reviews</div></div>
                <div><div style="font-size:1.3rem;font-weight:800;color:#059669;">{pos}</div><div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Positive</div></div>
                <div><div style="font-size:1.3rem;font-weight:800;color:#dc2626;">{neg}</div><div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Negative</div></div>
                {"" if avg_r is None else f'<div><div style="font-size:1.3rem;font-weight:800;color:#d97706;">{avg_r:.1f}★</div><div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Avg Rating</div></div>'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cL("Sentiment Mix")
        counts = sdf["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment","Count"]
        fig = px.pie(counts, names="Sentiment", values="Count",
                     color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.55)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11, color="#1e293b"),
                          marker=dict(line=dict(color="#ffffff", width=2)))
        fig.update_layout(**plot_layout(height=280, showlegend=False,
                                         margin=dict(t=20,b=20,l=20,r=20)))
        chart_card(fig)

    with col2:
        cL("Issue Breakdown")
        ic_s = sdf["issue_category"].value_counts().reset_index()
        ic_s.columns = ["Category","Count"]
        fig2 = px.bar(ic_s, x="Count", y="Category", orientation="h",
                      color="Category", color_discrete_map=ISSUE_COLORS,
                      labels={"Count":"Reviews","Category":""})
        fig2.update_layout(**plot_layout(height=280, showlegend=False,
                                          margin=dict(t=16,b=16,l=16,r=16)))
        chart_card(fig2)

    cL("Customer Reviews — Selected Station")
    display_cols = [c for c in ["author_name","rating","sentiment","confidence","issue_category","text"]
                    if c in sdf.columns]
    rename = {"author_name":"Reviewer","rating":"Stars","sentiment":"Sentiment",
               "confidence":"Confidence Score","issue_category":"Issue Category","text":"Review Text"}
    st.dataframe(sdf[display_cols].rename(columns=rename).reset_index(drop=True),
                 use_container_width=True, height=380)


# ── Tab 7: Data Export ─────────────────────────────────────────────────────────

def tab_export(df):
    cL("Download Analysis Report")

    total = len(df)
    pos   = (df["sentiment"] == "Positive").sum()
    neg   = (df["sentiment"] == "Negative").sum()
    neu   = (df["sentiment"] == "Neutral").sum()

    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:12px;">Dataset Summary</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
            <div style="background:#f8fafc;border-radius:10px;padding:14px 16px;">
                <div style="font-size:1.4rem;font-weight:800;color:#1e40af;">{total:,}</div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Total Reviews</div>
            </div>
            <div style="background:#f0fdf4;border-radius:10px;padding:14px 16px;">
                <div style="font-size:1.4rem;font-weight:800;color:#059669;">{pos:,}</div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Positive ({pos/total*100:.1f}%)</div>
            </div>
            <div style="background:#fffbeb;border-radius:10px;padding:14px 16px;">
                <div style="font-size:1.4rem;font-weight:800;color:#d97706;">{neu:,}</div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Neutral ({neu/total*100:.1f}%)</div>
            </div>
            <div style="background:#fef2f2;border-radius:10px;padding:14px 16px;">
                <div style="font-size:1.4rem;font-weight:800;color:#dc2626;">{neg:,}</div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Negative ({neg/total*100:.1f}%)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cL("Complete Review Dataset")
    export_cols = [c for c in [
        "station_name","station_address","brand","category","zone",
        "lat","lng","station_rating","station_reviews",
        "author_name","rating","text","date_relative",
        "sentiment","bert_star","confidence","issue_category","issue_tags","quality_score",
        "source","review_id"
    ] if c in df.columns]

    rename = {
        "station_name":"Station Name","station_address":"Station Address","brand":"Brand",
        "category":"Category","zone":"Zone","lat":"Latitude","lng":"Longitude",
        "station_rating":"Google Avg Rating","station_reviews":"Google Review Count",
        "author_name":"Reviewer","rating":"Review Stars","text":"Review Text",
        "date_relative":"Date (Relative)","sentiment":"Sentiment Classification",
        "bert_star":"Predicted Rating","confidence":"Model Confidence",
        "issue_category":"Primary Issue","issue_tags":"All Issues",
        "quality_score":"Quality Score","source":"Data Source","review_id":"Review ID",
    }

    export_df = df[export_cols].rename(columns=rename)
    st.dataframe(export_df, use_container_width=True, height=440)

    csv = export_df.to_csv(index=False).encode()
    st.download_button("⬇  Download Sentiment Analysis CSV", data=csv,
                       file_name="mumbai_sentiment_analysis.csv", mime="text/csv", key="dl_full_csv")


# ══════════════════════════════════════════════════════════════════════════════
# STATION-LEVEL DASHBOARD (aggregate ratings)
# ══════════════════════════════════════════════════════════════════════════════

def dashboard_station(df_raw):
    df = normalize_station_dataframe(df_raw.copy())
    df_v = df[df["sentiment"] != "Unknown"].copy()

    st.markdown("""
    <div style="background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;
                padding:24px 28px;margin-bottom:1.75rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:1.2rem;font-weight:800;color:#0f172a;">Mumbai Petrol Pump — Station Overview</div>
        <div style="font-size:0.8rem;color:#64748b;margin-top:3px;">Aggregate Google ratings · 168 stations · CNG coverage · Zone analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#475569;margin-bottom:8px;">Filters</div>', unsafe_allow_html=True)
    if "zone" in df_v.columns:
        sel_z = st.sidebar.multiselect("Zone", sorted(df_v["zone"].unique()), default=sorted(df_v["zone"].unique()))
        df_v = df_v[df_v["zone"].isin(sel_z)]
    if "fuel_type" in df_v.columns:
        sel_f = st.sidebar.multiselect("Fuel Type", sorted(df_v["fuel_type"].unique()), default=sorted(df_v["fuel_type"].unique()))
        df_v = df_v[df_v["fuel_type"].isin(sel_f)]
    sel_s = st.sidebar.multiselect("Sentiment", sorted(df_v["sentiment"].unique()), default=sorted(df_v["sentiment"].unique()))
    df_v = df_v[df_v["sentiment"].isin(sel_s)]

    if df_v.empty:
        st.warning("No stations match current filters.")
        return

    tabs = st.tabs(["📊  Overview", "🗺️  Map", "📍  Zone Analysis", "🏆  Scorecard", "💡  Insights"])
    with tabs[0]: _st_overview(df_v)
    with tabs[1]: _st_map(df_v)
    with tabs[2]: _st_zones(df_v)
    with tabs[3]: _st_scorecard(df_v)
    with tabs[4]: _st_insights(df_v)


def _st_overview(df):
    total = len(df)
    pos = (df["sentiment"]=="Positive").sum()
    neg = (df["sentiment"]=="Negative").sum()
    neu = (df["sentiment"]=="Neutral").sum()
    avg_r = df["_rating"].mean() if "_rating" in df.columns else None

    cL("Station Health Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Stations", f"{total:,}")
    c2.metric("Positive (≥3.6★)", f"{pos:,}", f"{pos/total*100:.1f}%")
    c3.metric("Neutral (2.6–3.5★)", f"{neu:,}", f"{neu/total*100:.1f}%")
    c4.metric("Negative (≤2.5★)", f"{neg:,}", f"{neg/total*100:.1f}%")

    col1,col2 = st.columns(2)
    with col1:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment","Stations"]
        fig = px.pie(counts, names="Sentiment", values="Stations",
                     color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.52)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="#ffffff",width=2)))
        fig.update_layout(**plot_layout(height=300, showlegend=False, margin=dict(t=20,b=20,l=20,r=20)))
        chart_card(fig)
    with col2:
        if "_rating" in df.columns:
            fig2 = px.histogram(df, x="_rating", nbins=20,
                                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                                barmode="overlay", opacity=0.75,
                                labels={"_rating":"Avg Station Rating","sentiment":"Sentiment"})
            fig2.update_layout(**plot_layout(height=300,
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(color="#1e293b",size=12))))
            chart_card(fig2)


def _st_map(df):
    if "_lat" not in df.columns or "_lng" not in df.columns:
        st.info("No coordinate data available.")
        return
    dm = df[df["_lat"].notna() & df["_lng"].notna()].copy()
    if "_review_count" in dm.columns:
        dm["_sz"] = np.log1p(dm["_review_count"]) * 2 + 5
    import json, os
    boundary = None
    if os.path.exists("mumbai_boundary.geojson"):
        with open("mumbai_boundary.geojson") as f:
            boundary = json.load(f)
    fig = px.scatter_mapbox(
        dm, lat="_lat", lon="_lng",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        size="_sz" if "_sz" in dm.columns else None, size_max=22,
        hover_name="_title" if "_title" in dm.columns else None,
        zoom=10.5, center={"lat": dm["_lat"].mean(),"lon": dm["_lng"].mean()},
        height=560,
    )
    layers = []
    if boundary:
        layers.append({"sourcetype":"geojson","source":boundary,"type":"line",
                       "color":"#64748b","line":{"width":1.5},"opacity":0.7})
    fig.update_layout(mapbox_style="open-street-map", mapbox_layers=layers,
                      **plot_layout(height=560, margin=dict(t=10,b=10,l=10,r=10),
                                    legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0)))
    st.plotly_chart(fig, use_container_width=True)


def _st_zones(df):
    if "zone" not in df.columns: st.info("Zone data unavailable."); return
    zs = (df.groupby("zone")
            .agg(Stations=("_rating","count"), Avg_Rating=("_rating","mean"),
                 Pos=("sentiment",lambda x:(x=="Positive").sum()),
                 Neg=("sentiment",lambda x:(x=="Negative").sum()))
            .assign(Pos_Pct=lambda x:x["Pos"]/x["Stations"]*100,
                    Neg_Pct=lambda x:x["Neg"]/x["Stations"]*100)
            .sort_values("Avg_Rating",ascending=False).reset_index())
    col1,col2 = st.columns(2)
    with col1:
        cL("Average Customer Rating by Zone")
        fig = px.bar(zs.sort_values("Avg_Rating"), x="Avg_Rating", y="zone", orientation="h",
                     color="Avg_Rating", color_continuous_scale="RdYlGn", range_color=[2.5,4.5],
                     labels={"Avg_Rating":"Avg Rating","zone":""})
        fig.update_layout(**plot_layout(height=300,showlegend=False,coloraxis_showscale=False))
        chart_card(fig)
    with col2:
        cL("Sentiment Breakdown by Zone")
        zs2 = df.groupby(["zone","sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(zs2, x="zone", y="count", color="sentiment",
                      color_discrete_map=SENTIMENT_COLORS, barmode="stack",
                      labels={"zone":"","count":"Stations","sentiment":"Sentiment"})
        fig2.update_xaxes(tickangle=15)
        fig2.update_layout(**plot_layout(height=300,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(color="#1e293b",size=12))))
        chart_card(fig2)


def _st_scorecard(df):
    cL("Station Rankings — Sorted by Customer Rating")
    cols = [c for c in ["_title","_rating","_review_count","sentiment","zone","fuel_type","_address"] if c in df.columns]
    rename = {"_title":"Station","_rating":"Avg Rating","_review_count":"Total Reviews",
               "sentiment":"Sentiment","zone":"Zone","fuel_type":"Fuel Type","_address":"Address"}
    sc = df[cols].rename(columns=rename).sort_values("Avg Rating",ascending=False)
    fmt = {}
    if "Avg Rating" in sc.columns: fmt["Avg Rating"] = "{:.2f}"
    if "Total Reviews" in sc.columns: fmt["Total Reviews"] = "{:,.0f}"
    styled = sc.style.format(fmt)
    if "Avg Rating" in sc.columns: styled = styled.background_gradient(subset=["Avg Rating"],cmap="RdYlGn")
    st.dataframe(styled, use_container_width=True, height=500)

    cL("Lowest-Rated Stations — Priority for Review")
    bottom = sc.tail(15).sort_values("Avg Rating")
    st.dataframe(bottom.style.background_gradient(subset=["Avg Rating"],cmap="Reds").format(fmt),
                 use_container_width=True)
    st.download_button("⬇  Download Scorecard CSV",
                       data=sc.to_csv(index=False).encode(),
                       file_name="mumbai_station_scorecard.csv", mime="text/csv", key="dl_scorecard")


def _st_insights(df):
    total = len(df)
    neg = (df["sentiment"]=="Negative").sum()
    neg_pct = neg/total*100
    avg_r = df["_rating"].mean() if "_rating" in df.columns else None

    cL("Operational Observations")
    if neg_pct > 30:
        ic(INS["red"], f"<b>High Negativity: {neg_pct:.1f}% of stations</b> have avg ratings ≤ 2.5★. City-wide operational review warranted.")
    elif neg_pct > 15:
        ic(INS["amber"], f"<b>Moderate Negativity: {neg_pct:.1f}%</b> of stations are rated poorly. Targeted improvement can lift performance.")
    else:
        ic(INS["green"], f"<b>Broadly Satisfactory:</b> Only {neg_pct:.1f}% of stations are rated negatively.")

    if avg_r:
        ic(INS["blue"], f"<b>City Average Rating: {avg_r:.2f} / 5.00</b> — "
           f"{'Above' if avg_r>=3.6 else 'Below' if avg_r<3.0 else 'At'} the positive threshold of 3.6★.")

    if "_review_count" in df.columns:
        low = (df["_review_count"]<50).sum()
        if low:
            ic(INS["amber"], f"<b>{int(low)} low-evidence stations</b> have fewer than 50 Google reviews. "
               f"Their ratings are statistically unreliable — monitor rather than act immediately.")

    if "_rating" in df.columns and "_title" in df.columns:
        df_e = df[df["_review_count"]>=50] if "_review_count" in df.columns else df
        if not df_e.empty:
            top5 = df_e.nlargest(5,"_rating")[["_title","_rating"]]
            ic(INS["green"], "<b>Top 5 Stations (min. 50 reviews):</b> " +
               "; ".join(f"<i>{r['_title']}</i> ({r['_rating']:.1f}★)" for _,r in top5.iterrows()))
            bot5 = df_e.nsmallest(5,"_rating")[["_title","_rating"]]
            ic(INS["red"], "<b>Bottom 5 — Immediate Action:</b> " +
               "; ".join(f"<i>{r['_title']}</i> ({r['_rating']:.1f}★)" for _,r in bot5.iterrows()))


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW-LEVEL DASHBOARD (Pune / uploaded)
# ══════════════════════════════════════════════════════════════════════════════

def dashboard_reviews(df_raw):
    df, _ = normalize_dataframe(df_raw.copy())
    if "_review_text" not in df.columns:
        st.error("Could not detect a review text column. Expected: text, review, review_text, comment, body.")
        return

    df = df[df["_review_text"].notna() & (df["_review_text"].str.strip()!="")].reset_index(drop=True)
    cache_key = str(len(df)) + str(df["_review_text"].iloc[0])
    if st.session_state.get("analyzed_hash") != cache_key:
        texts = tuple(df["_review_text"].fillna("").tolist())
        with st.spinner("Running AI-powered sentiment analysis…"):
            results = run_bert(texts)
        df["sentiment"]      = [r["sentiment"]    for r in results]
        df["confidence"]     = [r["confidence"]   for r in results]
        df["bert_star"]      = [r["star_rating"]  for r in results]
        df["prob_positive"]  = [r["prob_positive"] for r in results]
        df["prob_neutral"]   = [r["prob_neutral"]  for r in results]
        df["prob_negative"]  = [r["prob_negative"] for r in results]
        with st.spinner("Classifying issues…"):
            df["issue_category"] = df["_review_text"].apply(classify_issue)
            df["issue_tags"]     = df["_review_text"].apply(lambda x: ", ".join(classify_issues_multi(x, top_n=2)))
        st.session_state.analyzed_df   = df
        st.session_state.analyzed_hash = cache_key
    else:
        df = st.session_state.analyzed_df

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#475569;margin-bottom:8px;">Filters</div>', unsafe_allow_html=True)
    sel_s = st.sidebar.multiselect("Sentiment", sorted(df["sentiment"].unique()), default=sorted(df["sentiment"].unique()))
    sel_i = st.sidebar.multiselect("Issue", sorted(df["issue_category"].unique()), default=sorted(df["issue_category"].unique()))
    if "_title" in df.columns:
        outlets = sorted(df["_title"].dropna().unique())
        sel_o = st.sidebar.multiselect("Outlet", outlets, default=outlets)
        df = df[df["_title"].isin(sel_o)]
    if "_date" in df.columns and df["_date"].notna().sum() > 0:
        min_d, max_d = df["_date"].min().date(), df["_date"].max().date()
        dr = st.sidebar.date_input("Date range", value=(min_d, max_d))
        if len(dr) == 2:
            df = df[(df["_date"]>=pd.Timestamp(dr[0])) & (df["_date"]<=pd.Timestamp(dr[1]))]
    df = df[df["sentiment"].isin(sel_s) & df["issue_category"].isin(sel_i)]
    if df.empty:
        st.warning("No data matches filters.")
        return

    # Header
    total = len(df)
    pos = (df["sentiment"]=="Positive").sum()
    neg = (df["sentiment"]=="Negative").sum()
    st.markdown(f"""
    <div style="background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;
                padding:24px 28px;margin-bottom:1.75rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:1.2rem;font-weight:800;color:#0f172a;">Retail Outlet Sentiment Analysis</div>
        <div style="font-size:0.8rem;color:#64748b;margin-top:3px;">
            {total:,} reviews &nbsp;·&nbsp; {pos/total*100:.1f}% positive &nbsp;·&nbsp; {neg/total*100:.1f}% negative
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview","🔍 Issues","📈 Trends","💡 Insights","📋 Data"])
    with tab1: _rv_overview(df)
    with tab2: _rv_issues(df)
    with tab3: _rv_trends(df)
    with tab4: _rv_insights(df)
    with tab5: _rv_data(df, df_raw)


def _rv_overview(df):
    total = len(df); pos=(df["sentiment"]=="Positive").sum()
    neg=(df["sentiment"]=="Negative").sum(); neu=(df["sentiment"]=="Neutral").sum()
    cL("Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Reviews",f"{total:,}")
    c2.metric("Positive",f"{pos:,}",f"{pos/total*100:.1f}%")
    c3.metric("Neutral",f"{neu:,}",f"{neu/total*100:.1f}%")
    c4.metric("Negative",f"{neg:,}",f"{neg/total*100:.1f}%")

    col1,col2 = st.columns(2)
    with col1:
        cL("Sentiment Distribution")
        counts = df["sentiment"].value_counts().reset_index(); counts.columns=["Sentiment","Count"]
        fig = px.pie(counts,names="Sentiment",values="Count",color="Sentiment",
                     color_discrete_map=SENTIMENT_COLORS,hole=0.52)
        fig.update_traces(textposition="outside",textinfo="percent+label",
                          marker=dict(line=dict(color="#ffffff",width=2)))
        fig.update_layout(**plot_layout(height=300,showlegend=False,margin=dict(t=20,b=20,l=20,r=20)))
        chart_card(fig)
    with col2:
        if "_rating" in df.columns:
            cL("Ratings vs Sentiment")
            rs = df.groupby(["_rating","sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(rs,x="_rating",y="count",color="sentiment",
                          color_discrete_map=SENTIMENT_COLORS,barmode="stack",
                          labels={"_rating":"Star Rating","count":"Reviews","sentiment":"Sentiment"})
            fig2.update_layout(**plot_layout(height=300,
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(color="#1e293b",size=12))))
            chart_card(fig2)

    cL("Top Keywords")
    kc1,kc2,kc3 = st.columns(3)
    for col,sent,color in [(kc1,"Positive",POS),(kc2,"Neutral",NEU),(kc3,"Negative",NEG)]:
        blob = " ".join(df[df["sentiment"]==sent]["_review_text"].dropna())
        kws = top_keywords(blob,n=10)
        if kws:
            kw_df = pd.DataFrame(kws,columns=["Word","Freq"]).sort_values("Freq")
            fig_k = px.bar(kw_df,x="Freq",y="Word",orientation="h",
                           color_discrete_sequence=[color],labels={"Freq":"Freq","Word":""})
            fig_k.update_layout(**plot_layout(height=300,showlegend=False,margin=dict(t=8,b=8,l=8,r=8)))
            with col:
                st.markdown(f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:{color};margin-bottom:6px;">{sent}</div>',unsafe_allow_html=True)
                chart_card(fig_k)


def _rv_issues(df):
    cL("Issue × Sentiment")
    is2 = df.groupby(["issue_category","sentiment"]).size().reset_index(name="count")
    fig = px.bar(is2,x="issue_category",y="count",color="sentiment",
                 color_discrete_map=SENTIMENT_COLORS,barmode="stack",
                 labels={"issue_category":"","count":"Reviews","sentiment":"Sentiment"})
    fig.update_xaxes(tickangle=35)
    fig.update_layout(**plot_layout(height=380,margin=dict(t=36,b=90,l=16,r=16),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(color="#1e293b",size=12))))
    chart_card(fig)

    sel = st.selectbox("Browse issue category", sorted(df["issue_category"].unique()), key="pune_issue_sel")
    neg_s = df[(df["issue_category"]==sel)&(df["sentiment"]=="Negative")][["_review_text","_rating","confidence"]].head(10)
    if neg_s.empty: st.info("No negative reviews for this category.")
    else:
        st.dataframe(neg_s.rename(columns={"_review_text":"Review","_rating":"Stars","confidence":"Confidence"}),
                     use_container_width=True,height=280)


def _rv_trends(df):
    if "_date" not in df.columns or df["_date"].notna().sum()<5:
        st.info("No date column detected.")
        return
    df_t = df[df["_date"].notna()].copy()
    gran = st.radio("Granularity",["Monthly","Weekly","Quarterly"],horizontal=True, key="gran_radio")
    pcol = {"Monthly":"month","Weekly":"week","Quarterly":"quarter"}[gran]
    df_t["month"]=df_t["_date"].dt.to_period("M").dt.to_timestamp()
    df_t["week"]=df_t["_date"].dt.to_period("W").dt.to_timestamp()
    df_t["quarter"]=df_t["_date"].dt.to_period("Q").dt.to_timestamp()

    cL("Sentiment Volume Over Time")
    st2 = df_t.groupby([pcol,"sentiment"]).size().reset_index(name="count")
    fig = px.line(st2,x=pcol,y="count",color="sentiment",
                  color_discrete_map=SENTIMENT_COLORS,markers=True,
                  labels={pcol:"Period","count":"Reviews","sentiment":"Sentiment"})
    fig.update_layout(**plot_layout(height=320,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(color="#1e293b",size=12))))
    chart_card(fig)


def _rv_insights(df):
    total=len(df); neg=(df["sentiment"]=="Negative").sum(); neg_pct=neg/total*100
    cL("Operational Observations")
    if neg_pct>50: ic(INS["red"],f"<b>High Negativity ({neg_pct:.1f}%)</b> — Immediate attention required.")
    elif neg_pct>30: ic(INS["amber"],f"<b>Moderate Negativity ({neg_pct:.1f}%)</b> — Clear improvement opportunity.")
    else: ic(INS["green"],f"<b>Broadly Positive</b> — Only {neg_pct:.1f}% negative reviews.")

    ni = df[df["sentiment"]=="Negative"]["issue_category"].value_counts()
    if not ni.empty:
        ic(INS["amber"],f"<b>Top Pain Point: {ni.index[0]}</b> — {ni.iloc[0]} negative reviews ({ni.iloc[0]/neg*100:.1f}% of all negatives).")

    if "_rating" in df.columns:
        mm = ((df["_rating"]>=4)&(df["sentiment"]=="Negative")).sum()
        if mm>0:
            ic(INS["blue"],f"<b>Rating–Sentiment Mismatch: {mm} reviews</b> carry 4–5 stars yet the sentiment analysis identifies a negative tone — customers may be leaving courtesy ratings.")


def _rv_data(df, df_raw):
    cL("Analysed Reviews")
    dc = [c for c in ["_review_text","_rating","sentiment","bert_star","confidence","issue_category","issue_tags"] if c in df.columns]
    rn = {"_review_text":"Review","_rating":"Stars","sentiment":"Sentiment","bert_star":"Predicted Rating",
          "confidence":"Confidence","issue_category":"Issue","issue_tags":"All Issues"}
    st.dataframe(df[dc].rename(columns=rn),use_container_width=True,height=440)
    st.download_button("⬇  Download Results CSV",
                       data=df[dc].rename(columns=rn).to_csv(index=False).encode(),
                       file_name="sentiment_results.csv", mime="text/csv", key="dl_results")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
