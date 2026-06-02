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

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── theme ──────────────────────────────────────────────────────────────────────
_dark = False

T = dict(
    bg           = "#f8fafc",
    card         = "#ffffff",
    sidebar      = "#f1f5f9",
    border       = "#e2e8f0",
    text         = "#0f172a",
    text_sub     = "#475569",
    section_text = "#94a3b8",
    input_bg     = "#ffffff",
    # plotly
    plot_bg      = "#ffffff",
    paper_bg     = "#ffffff",
    font_color   = "#1e293b",
    grid_color   = "#f1f5f9",
    hover_bg     = "#ffffff",
    plotly_tpl   = "plotly_white",
    # insight card tints
    ins_red      = ("#fef2f2", "#dc2626"),
    ins_amber    = ("#fffbeb", "#b45309"),
    ins_green    = ("#f0fdf4", "#059669"),
    ins_blue     = ("#eff6ff", "#1d4ed8"),
    ins_purple   = ("#f5f3ff", "#6d28d9"),
    ins_text     = "#1e293b",
    # buttons
    btn_bg       = "#ffffff",
    btn_text     = "#1e293b"
)

SENTIMENT_COLORS = {
    "Positive": "#059669",
    "Neutral":  "#d97706",
    "Negative": "#dc2626",
}

STOPWORDS = {
    "the","a","an","is","it","in","of","and","to","for","with","on","at","by",
    "from","this","was","are","be","not","but","have","had","has","they","their",
    "there","that","very","so","my","we","i","its","here","get","got","also",
    "just","no","or","as","do","did","been","all","one","if","up","out","about",
    "than","more","when","will","can","good","place","nice","would","like",
    "really","come",
}

# ── css injection ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap');

    /* ── base reset ── */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="block-container"], .main {{
        background-color: {T['bg']} !important;
        color: {T['text']} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    .block-container {{
        padding: 1.75rem 2.5rem 3rem 2.5rem;
        max-width: 1440px;
        background-color: {T['bg']} !important;
    }}

    /* ── typography ── */
    h1, h2, h3, h4 {{
        color: {T['text']} !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }}
    h1 {{
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        margin-bottom: 0.1rem !important;
    }}
    p, li, span, label, div, caption {{
        color: {T['text']} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* ── page header area ── */
    .page-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 20px 24px;
        background: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .page-header-icon {{
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }}
    .page-header-title {{
        font-size: 1.2rem;
        font-weight: 800;
        color: {T['text']} !important;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }}
    .page-header-sub {{
        font-size: 0.8rem;
        color: {T['text_sub']} !important;
        font-weight: 400;
        margin-top: 2px;
        letter-spacing: 0;
    }}

    /* ── metric cards ── */
    [data-testid="stMetric"] {{
        background: {T['card']} !important;
        border-radius: 12px;
        padding: 20px 24px 18px 24px;
        border: 1px solid {T['border']} !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.09) !important;
    }}
    [data-testid="stMetricLabel"] > div {{
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.09em !important;
        color: {T['text_sub']} !important;
    }}
    [data-testid="stMetricValue"] > div {{
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: {T['text']} !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em !important;
        margin-top: 4px !important;
    }}
    [data-testid="stMetricDelta"] > div {{
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        color: {T['text_sub']} !important;
        margin-top: 4px !important;
    }}

    /* ── section titles ── */
    .section-title {{
        font-size: 0.62rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {T['section_text']} !important;
        margin: 2.25rem 0 1rem 0;
        padding-bottom: 10px;
        border-bottom: 1px solid {T['border']};
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* ── chart cards ── */
    .chart-card {{
        background: {T['card']};
        border-radius: 12px;
        border: 1px solid {T['border']};
        padding: 4px 4px 0 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 1.25rem;
        overflow: hidden;
        transition: box-shadow 0.15s ease;
    }}
    .chart-card:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }}

    /* ── insight cards ── */
    .insight-card {{
        border-radius: 10px;
        padding: 14px 18px 14px 16px;
        margin-bottom: 10px;
        border-left: 3px solid;
        font-size: 0.855rem;
        line-height: 1.7;
        color: {T['ins_text']} !important;
        transition: transform 0.1s ease;
    }}
    .insight-card:hover {{ transform: translateX(2px); }}
    .insight-card b {{
        color: {T['text']} !important;
        font-weight: 700;
    }}

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: transparent !important;
        border-bottom: 1px solid {T['border']} !important;
        padding: 0 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 10px 18px 9px 18px;
        border-radius: 0 !important;
        color: {T['section_text']} !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -1px;
        transition: color 0.15s;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {T['text_sub']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {T['text']} !important;
        border-bottom-color: #3b82f6 !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
        padding-top: 1.5rem !important;
    }}

    /* ── sidebar ── */
    [data-testid="stSidebar"] {{
        background: {T['sidebar']} !important;
        border-right: 1px solid {T['border']} !important;
    }}
    [data-testid="stSidebarContent"] {{
        padding: 1.75rem 1.25rem 1.5rem 1.25rem;
    }}
    [data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
    [data-testid="stSidebar"] h3 {{
        font-size: 0.65rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: {T['section_text']} !important;
        margin-bottom: 0.6rem !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {T['border']} !important;
        margin: 1.25rem 0 !important;
    }}

    /* ── inputs / selects / radio ── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stMultiSelect"] > div,
    .stRadio > div {{
        background: {T['input_bg']} !important;
        border-color: {T['border']} !important;
        color: {T['text']} !important;
    }}
    .stRadio label {{
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }}

    /* ── dataframe ── */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {T['border']} !important;
        background: {T['card']} !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    [data-testid="stDataFrame"] * {{ color: {T['text']} !important; }}

    /* ── alerts ── */
    [data-testid="stAlert"] {{
        border-radius: 10px !important;
        border-left-width: 3px !important;
        font-size: 0.855rem !important;
    }}

    /* ── download button ── */
    .stDownloadButton > button {{
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        background: {T['card']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: box-shadow 0.15s, background 0.15s !important;
    }}
    .stDownloadButton > button:hover {{
        background: {T['sidebar']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }}

    /* ── modebar ── */
    .modebar {{ opacity: 0 !important; transition: opacity 0.2s; }}
    .modebar:hover {{ opacity: 1 !important; }}

    /* ── hide sidebar collapse icon text ── */
    [data-testid="stSidebarCollapseButton"] svg {{ display: none; }}
    [data-testid="collapsedControl"] span,
    [data-testid="stSidebarCollapseButton"] span {{ font-size: 0 !important; }}
    button[kind="header"] span[class*="material"] {{ display: none !important; }}

    /* ── multiselect — fix first-chip left clipping ── */
    [data-baseweb="select"] > div:first-child {{
        padding-left: 10px !important;
        box-sizing: border-box !important;
    }}
    [data-baseweb="select"] [data-baseweb="input"] {{
        padding-left: 8px !important;
        min-height: 38px;
    }}
    [data-baseweb="tag"] {{ margin-left: 4px !important; }}

    /* ── date input ── */
    [data-testid="stDateInput"] input,
    [data-testid="stDateInput"] > div {{
        background-color: {T['input_bg']} !important;
        color: {T['text']} !important;
        border-color: {T['border']} !important;
    }}
    [data-testid="stDateInput"] * {{
        color: {T['text']} !important;
        background-color: {T['input_bg']} !important;
    }}

    /* ── caption ── */
    [data-testid="stCaptionContainer"] * {{
        color: {T['text_sub']} !important;
        font-size: 0.76rem !important;
    }}

    /* ── multiselect — container / input ── */
    [data-baseweb="select"] {{ background-color: {T['input_bg']} !important; }}
    [data-baseweb="select"] > div,
    [data-baseweb="select"] input {{
        background-color: {T['input_bg']} !important;
        color: {T['text']} !important;
    }}

    /* ── multiselect chips ── */
    [data-baseweb="tag"] {{
        background-color: #e8edf5 !important;
        border-radius: 5px !important;
        border: 1px solid {T['border']} !important;
    }}
    [data-baseweb="tag"] span {{
        color: {T['text']} !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }}
    [data-baseweb="tag"] [role="button"] {{ color: {T['text_sub']} !important; }}

    /* ── dropdown menus ── */
    [data-baseweb="popover"] {{ background-color: {T['card']} !important; }}
    [data-baseweb="popover"] li {{
        background-color: {T['card']} !important;
        color: {T['text']} !important;
        font-size: 0.83rem !important;
    }}
    [data-baseweb="popover"] li:hover {{ background-color: {T['sidebar']} !important; }}

    /* ── selectbox ── */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: {T['input_bg']} !important;
        border-color: {T['border']} !important;
        color: {T['text']} !important;
    }}

    /* ── table (st.table) ── */
    table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 0.83rem !important;
    }}
    thead tr {{
        background: {T['sidebar']} !important;
        border-bottom: 2px solid {T['border']};
    }}
    thead th {{
        font-size: 0.65rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {T['text_sub']} !important;
        padding: 10px 14px !important;
    }}
    tbody tr {{ border-bottom: 1px solid {T['border']}; }}
    tbody tr:last-child {{ border-bottom: none; }}
    tbody td {{
        padding: 9px 14px !important;
        color: {T['text']} !important;
    }}
    tbody tr:hover {{ background: #f8fafc !important; }}

    /* ── divider ── */
    hr {{
        border: none !important;
        border-top: 1px solid {T['border']} !important;
        margin: 1.5rem 0 !important;
    }}

    /* ── Plotly SVG text — legend labels, axis titles, tick labels ── */
    /* CSS `color` doesn't touch SVG; must use `fill` */
    .js-plotly-plot .plotly svg text,
    .js-plotly-plot .plotly .legendtext,
    .js-plotly-plot .plotly .legend text,
    .js-plotly-plot .plotly .gtitle text,
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle,
    .js-plotly-plot .plotly .g-xtitle text,
    .js-plotly-plot .plotly .g-ytitle text,
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .colorbar text,
    .js-plotly-plot .plotly .annotation text {{
        fill: {T['font_color']} !important;
    }}
    /* legend title specifically */
    .js-plotly-plot .plotly .legend .legendtitletext {{
        fill: {T['text_sub']} !important;
        font-weight: 600 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def chart_layout(**overrides):
    """Return a Plotly layout dict with theme colours applied."""
    axis_style = dict(
        tickfont=dict(color=T["font_color"], size=11),
        title_font=dict(color=T["font_color"], size=11),
        linecolor=T["border"],
        gridcolor=T["grid_color"],
    )
    base = dict(
        template      = T["plotly_tpl"],
        font          = dict(family="Inter, sans-serif", size=11, color=T["font_color"]),
        plot_bgcolor  = T["plot_bg"],
        paper_bgcolor = T["paper_bg"],
        hoverlabel    = dict(font_size=11, font_family="Inter, sans-serif", bgcolor=T["hover_bg"]),
        xaxis         = axis_style,
        yaxis         = axis_style,
        legend        = dict(font=dict(color=T["font_color"], size=11), title_font=dict(color=T["font_color"])),
    )
    base.update(overrides)
    base.pop("title_text", None)
    return base


def section(label):
    st.markdown(
        f'<div class="section-title">'
        f'<span style="display:inline-block;width:3px;height:12px;background:#3b82f6;border-radius:2px;flex-shrink:0;"></span>'
        f'{label}</div>',
        unsafe_allow_html=True,
    )


def chart(fig):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def insight_card(bg_border_tuple, html_text):
    bg, border = bg_border_tuple
    st.markdown(
        f'<div class="insight-card" style="background:{bg};border-left-color:{border};">'
        f'{html_text}</div>',
        unsafe_allow_html=True,
    )


def top_keywords(text, n=10):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return Counter(words).most_common(n)


@st.cache_data(show_spinner=False)
def load_default_pune():
    return pd.read_excel("Pune_Retail_outlet (1).xlsx")


@st.cache_data(show_spinner=False)
def load_default_mumbai():
    return pd.read_excel("mumbai_petrol_pumps.xlsx")


def is_station_level(df):
    """Return True when the dataframe is station-level (no review text column)."""
    col_lower = {c.lower().strip() for c in df.columns}
    review_indicators = {"review_text_final", "text", "review", "review_text", "comment", "body"}
    return not bool(review_indicators & col_lower)


def run_analysis(df):
    texts = df["_review_text"].fillna("").tolist()
    with st.spinner("Running sentiment analysis — this takes a moment on first load."):
        results = predict_sentiment_batch(texts, batch_size=32)

    df["sentiment"]     = [r["sentiment"]    for r in results]
    df["confidence"]    = [r["confidence"]   for r in results]
    df["bert_star"]     = [r["star_rating"]  for r in results]
    df["prob_positive"] = [r["prob_positive"] for r in results]
    df["prob_neutral"]  = [r["prob_neutral"]  for r in results]
    df["prob_negative"] = [r["prob_negative"] for r in results]

    with st.spinner("Classifying issue categories..."):
        df["issue_category"] = df["_review_text"].apply(classify_issue)
        df["issue_tags"]     = df["_review_text"].apply(
            lambda x: ", ".join(classify_issues_multi(x, top_n=2))
        )
    return df


# ── sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("### Data Source")
    mode = st.sidebar.radio(
        "Dataset",
        ["Pune Retail Outlet (default)", "Mumbai Petrol Pumps (163 stations)", "Upload CSV / XLSX"],
        label_visibility="collapsed",
    )

    df_raw = None
    if mode == "Pune Retail Outlet (default)":
        df_raw = load_default_pune()
        st.sidebar.success(f"{len(df_raw):,} reviews loaded")
    elif mode == "Mumbai Petrol Pumps (163 stations)":
        df_raw = load_default_mumbai()
        st.sidebar.success(f"{len(df_raw):,} stations loaded")
        st.sidebar.info("Station-level dataset — sentiment derived from aggregate ratings.")
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload file", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
        )
        if uploaded:
            df_raw = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
            st.sidebar.success(f"{len(df_raw):,} rows loaded")
        else:
            st.sidebar.info("Upload a CSV or XLSX file to begin.")
    return df_raw


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        '<div class="page-header">'
        '<div class="page-header-icon">📊</div>'
        '<div>'
        '<div class="page-header-title">Sentiment Intelligence Dashboard</div>'
        '<div class="page-header-sub">Sentiment analysis &nbsp;·&nbsp; Issue categorisation &nbsp;·&nbsp; Time-series trends &nbsp;·&nbsp; Operational insights</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    df_raw = sidebar()
    if df_raw is None:
        st.info("Select or upload a dataset from the sidebar to begin.")
        return

    if is_station_level(df_raw):
        main_station(df_raw)
        return

    df, _ = normalize_dataframe(df_raw.copy())

    if "_review_text" not in df.columns:
        st.error(
            "Could not detect a review text column. "
            "Expected one of: review_text_final, text, review, comment, body."
        )
        st.dataframe(df.head())
        return

    df = df[
        df["_review_text"].notna() & (df["_review_text"].str.strip() != "")
    ].reset_index(drop=True)

    cache_key = str(len(df)) + str(df["_review_text"].iloc[0])
    if st.session_state.get("analyzed_hash") != cache_key:
        df = run_analysis(df)
        st.session_state.analyzed_df   = df
        st.session_state.analyzed_hash = cache_key
    else:
        df = st.session_state.analyzed_df

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    sel_sentiments = st.sidebar.multiselect(
        "Sentiment",
        sorted(df["sentiment"].unique()),
        default=sorted(df["sentiment"].unique()),
    )
    sel_issues = st.sidebar.multiselect(
        "Issue Category",
        sorted(df["issue_category"].unique()),
        default=sorted(df["issue_category"].unique()),
    )

    if "_title" in df.columns:
        outlets = sorted(df["_title"].dropna().unique())
        sel_outlets = st.sidebar.multiselect("Outlet", outlets, default=outlets)
        df = df[df["_title"].isin(sel_outlets)]

    if "_date" in df.columns and df["_date"].notna().sum() > 0:
        min_d, max_d = df["_date"].min().date(), df["_date"].max().date()
        date_range = st.sidebar.date_input("Date range", value=(min_d, max_d))
        if len(date_range) == 2:
            df = df[
                (df["_date"] >= pd.Timestamp(date_range[0])) &
                (df["_date"] <= pd.Timestamp(date_range[1]))
            ]

    df = df[
        df["sentiment"].isin(sel_sentiments) &
        df["issue_category"].isin(sel_issues)
    ]

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Issue Analysis",
        "Time-Series",
        "Deep Insights",
        "Data Explorer",
    ])

    with tab1:  tab_overview(df)
    with tab2:  tab_issues(df)
    with tab3:  tab_timeseries(df)
    with tab4:  tab_insights(df)
    with tab5:  tab_data(df, df_raw)


# ── Tab 1: Overview ────────────────────────────────────────────────────────────

def tab_overview(df):
    total    = len(df)
    pos      = (df["sentiment"] == "Positive").sum()
    neg      = (df["sentiment"] == "Negative").sum()
    neu      = (df["sentiment"] == "Neutral").sum()
    dominant = max(["Positive", "Neutral", "Negative"], key=lambda s: (df["sentiment"] == s).sum())
    top_issue = df[
        (df["sentiment"] == "Negative") & (df["issue_category"] != "Other")
    ]["issue_category"].value_counts()

    st.markdown(
        f"Analysed **{total:,}** reviews. Sentiment is predominantly **{dominant}** — "
        f"{pos:,} positive, {neu:,} neutral, {neg:,} negative."
    )

    section("Sentiment Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Positive",      f"{pos:,}",  f"{pos/total*100:.1f}% of total")
    c3.metric("Neutral",       f"{neu:,}",  f"{neu/total*100:.1f}% of total")
    c4.metric("Negative",      f"{neg:,}",  f"{neg/total*100:.1f}% of total")

    if "_rating" in df.columns:
        avg_r    = df["_rating"].dropna().mean()
        pred_avg = df["bert_star"].mean()
        r1, r2, r3 = st.columns(3)
        r1.metric("Avg User Rating",   f"{avg_r:.2f} / 5.00")
        r2.metric("Predicted Rating",  f"{pred_avg:.2f} / 5.00", f"{pred_avg - avg_r:+.2f} vs user rating")
        if not top_issue.empty:
            r3.metric("Top Negative Issue", top_issue.index[0], f"{top_issue.iloc[0]:,} negative reviews")

    section("Sentiment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
            hole=0.52,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont=dict(size=11, color=T["font_color"]),
        )
        fig.update_layout(**chart_layout(height=340, showlegend=False, margin=dict(t=20, b=20, l=20, r=20)))
        chart(fig)

    with col2:
        if "_rating" in df.columns:
            rs = df.groupby(["_rating", "sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(
                rs, x="_rating", y="count",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"_rating": "Star Rating", "count": "Reviews", "sentiment": "Sentiment"},
            )
            fig2.update_layout(**chart_layout(
                height=340,
                margin=dict(t=48, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            ))
            chart(fig2)

    section("Top Keywords by Sentiment")
    kc1, kc2, kc3 = st.columns(3)
    for col, sentiment, bar_color in zip(
        [kc1, kc2, kc3],
        ["Positive", "Neutral", "Negative"],
        ["#059669", "#d97706", "#dc2626"],
    ):
        blob = " ".join(df[df["sentiment"] == sentiment]["_review_text"].dropna())
        kws  = top_keywords(blob, n=10)
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Word", "Count"]).sort_values("Count")
            fig_k = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color_discrete_sequence=[bar_color],
                labels={"Count": "Frequency", "Word": ""},
            )
            fig_k.update_layout(**chart_layout(
                height=310, showlegend=False, margin=dict(t=16, b=12, l=8, r=8),
            ))
            with col:
                st.markdown(f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{T["text_sub"]};margin-bottom:4px;">{sentiment}</div>', unsafe_allow_html=True)
                chart(fig_k)

    section("Model Confidence")
    fig3 = px.histogram(
        df, x="confidence", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=35, barmode="overlay", opacity=0.72,
        labels={"confidence": "Confidence Score", "sentiment": "Sentiment"},
    )
    fig3.update_layout(**chart_layout(
        height=260, margin=dict(t=48, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    ))
    chart(fig3)


# ── Tab 2: Issue Analysis ──────────────────────────────────────────────────────

def tab_issues(df):
    section("Review Volume by Issue Category")
    issue_sent = df.groupby(["issue_category", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(
        issue_sent, x="issue_category", y="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        barmode="stack",
        labels={"issue_category": "Issue Category", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig.update_xaxes(tickangle=35)
    fig.update_layout(**chart_layout(
        height=380, margin=dict(t=48, b=80, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    ))
    chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        section("Category Share")
        ic = df["issue_category"].value_counts().reset_index()
        ic.columns = ["Category", "Count"]
        fig2 = px.pie(
            ic, names="Category", values="Count",
            hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont=dict(size=10, color=T["font_color"]),
        )
        fig2.update_layout(**chart_layout(height=400, showlegend=False, margin=dict(t=20, b=20, l=20, r=20)))
        chart(fig2)

    with col2:
        section("Negative Rate per Category")
        neg_rate = (
            df.groupby("issue_category")
            .apply(lambda x: (x["sentiment"] == "Negative").mean() * 100)
            .reset_index(name="pct")
            .sort_values("pct", ascending=True)
        )
        fig3 = px.bar(
            neg_rate, x="pct", y="issue_category", orientation="h",
            color="pct", color_continuous_scale="Reds",
            labels={"pct": "Negative Reviews (%)", "issue_category": ""},
        )
        fig3.update_layout(**chart_layout(
            height=400, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=20, b=20, l=20, r=20),
        ))
        chart(fig3)

    if "_title" in df.columns and df["_title"].nunique() > 1:
        section("Outlet vs Issue Category")
        pivot = df.groupby(["_title", "issue_category"]).size().unstack(fill_value=0)
        cs = "Blues" if not _dark else "Purples"
        fig4 = px.imshow(
            pivot, aspect="auto", color_continuous_scale=cs,
            labels={"x": "Issue Category", "y": "Outlet", "color": "Reviews"},
        )
        fig4.update_layout(**chart_layout(height=420, margin=dict(t=20, b=20, l=180, r=20)))
        chart(fig4)

    section("Browse Negative Reviews")
    selected = st.selectbox(
        "Issue category", sorted(df["issue_category"].unique()), label_visibility="collapsed"
    )
    neg_samples = df[
        (df["issue_category"] == selected) & (df["sentiment"] == "Negative")
    ][["_review_text", "_rating", "confidence"]].head(10)

    if neg_samples.empty:
        st.info("No negative reviews for this category under the current filters.")
    else:
        st.dataframe(
            neg_samples.rename(columns={
                "_review_text": "Review Text",
                "_rating":      "User Rating",
                "confidence":   "Confidence",
            }),
            use_container_width=True, height=300,
        )


# ── Tab 3: Time-Series ─────────────────────────────────────────────────────────

def tab_timeseries(df):
    if "_date" not in df.columns or df["_date"].notna().sum() < 5:
        st.info("No date column detected. Time-series analysis is not available.")
        return

    df_t = df[df["_date"].notna()].copy()
    df_t["month"]   = df_t["_date"].dt.to_period("M").dt.to_timestamp()
    df_t["week"]    = df_t["_date"].dt.to_period("W").dt.to_timestamp()
    df_t["quarter"] = df_t["_date"].dt.to_period("Q").dt.to_timestamp()

    granularity = st.radio("Granularity", ["Monthly", "Weekly", "Quarterly"], horizontal=True)
    period_col  = {"Monthly": "month", "Weekly": "week", "Quarterly": "quarter"}[granularity]

    section("Sentiment Volume Over Time")
    sent_trend = df_t.groupby([period_col, "sentiment"]).size().reset_index(name="count")
    fig = px.line(
        sent_trend, x=period_col, y="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS, markers=True,
        labels={period_col: "Period", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig.update_layout(**chart_layout(
        height=340, margin=dict(t=48, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    ))
    chart(fig)

    section("Sentiment Share Over Time")
    pct_data = (
        sent_trend
        .pivot_table(index=period_col, columns="sentiment", values="count", fill_value=0)
        .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100)
        .reset_index()
        .melt(id_vars=period_col, var_name="Sentiment", value_name="Percent")
    )
    fig2 = px.area(
        pct_data, x=period_col, y="Percent",
        color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
        labels={period_col: "Period", "Percent": "Share (%)"},
    )
    fig2.update_layout(**chart_layout(
        height=300, margin=dict(t=48, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    ))
    chart(fig2)

    if "_rating" in df_t.columns:
        section("Average Rating and Review Volume")
        rt = df_t.groupby(period_col)["_rating"].agg(["mean", "count"]).reset_index()
        rt.columns = [period_col, "avg_rating", "review_count"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rt[period_col], y=rt["avg_rating"],
            mode="lines+markers", name="Avg Rating",
            line=dict(color="#3b82f6", width=2), marker=dict(size=6),
        ))
        fig3.add_trace(go.Bar(
            x=rt[period_col], y=rt["review_count"],
            name="Review Count", opacity=0.2,
            marker_color=T["text_sub"], yaxis="y2",
        ))
        fig3.update_layout(**chart_layout(
            height=320, margin=dict(t=48, b=20, l=20, r=48),
            yaxis=dict(title="Avg Rating", range=[0.8, 5.2], gridcolor=T["grid_color"]),
            yaxis2=dict(title="Review Count", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        ))
        chart(fig3)

    section("Issue Category Trends")
    top_issues = df_t["issue_category"].value_counts().nlargest(6).index.tolist()
    issue_trend = (
        df_t[df_t["issue_category"].isin(top_issues)]
        .groupby([period_col, "issue_category"])
        .size()
        .reset_index(name="count")
    )
    fig4 = px.line(
        issue_trend, x=period_col, y="count",
        color="issue_category", markers=True,
        labels={period_col: "Period", "count": "Reviews", "issue_category": "Issue"},
    )
    fig4.update_layout(**chart_layout(
        height=340, margin=dict(t=48, b=20, l=20, r=20),
        legend=dict(font=dict(size=10)),
    ))
    chart(fig4)

    neg_by_period = (
        df_t[df_t["sentiment"] == "Negative"]
        .groupby(period_col).size().reset_index(name="neg_count")
    )
    if len(neg_by_period) >= 4:
        mn, sd = neg_by_period["neg_count"].mean(), neg_by_period["neg_count"].std()
        neg_by_period["spike"] = neg_by_period["neg_count"] > mn + 1.5 * sd
        spikes = neg_by_period[neg_by_period["spike"]]
        if not spikes.empty:
            st.warning("Negative review spike detected in: " + ", ".join(spikes[period_col].astype(str).tolist()))
            section("Negative Review Spikes")
            fig5 = px.bar(
                neg_by_period, x=period_col, y="neg_count",
                color="spike",
                color_discrete_map={True: "#dc2626", False: T["text_sub"]},
                labels={period_col: "Period", "neg_count": "Negative Reviews"},
            )
            fig5.update_layout(**chart_layout(height=260, showlegend=False, margin=dict(t=20, b=20, l=20, r=20)))
            chart(fig5)


# ── Tab 4: Deep Insights ───────────────────────────────────────────────────────

def tab_insights(df):
    total     = len(df)
    neg_count = (df["sentiment"] == "Negative").sum()
    neg_pct   = neg_count / total * 100

    section("Automated Insights")

    if neg_pct > 50:
        insight_card(T["ins_red"],
            f"<b>High Negativity</b> — {neg_pct:.1f}% of reviews are negative. Immediate operational attention is required.")
    elif neg_pct > 30:
        insight_card(T["ins_amber"],
            f"<b>Moderate Negativity</b> — {neg_pct:.1f}% of reviews are negative. There is clear room for improvement.")
    else:
        insight_card(T["ins_green"],
            f"<b>Broadly Positive</b> — Only {neg_pct:.1f}% of reviews are negative.")

    neg_issues = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()
    if not neg_issues.empty:
        ti, tc = neg_issues.index[0], neg_issues.iloc[0]
        insight_card(T["ins_amber"],
            f"<b>Top Pain Point</b> — '{ti}' accounts for {tc} negative reviews ({tc/neg_count*100:.1f}% of all negatives).")

    staff_neg = ((df["issue_category"] == "Staff Behaviour") & (df["sentiment"] == "Negative")).sum()
    if staff_neg > 0:
        insight_card(T["ins_purple"],
            f"<b>Staff Behaviour</b> — {staff_neg} reviews raise concerns about staff conduct. Targeted training is recommended.")

    cng_neg = ((df["issue_category"] == "CNG Availability") & (df["sentiment"] == "Negative")).sum()
    if cng_neg > 0:
        insight_card(T["ins_red"],
            f"<b>CNG Availability</b> — {cng_neg} negative reviews mention CNG being unavailable.")

    billing_neg = ((df["issue_category"] == "Billing or Trust Issue") & (df["sentiment"] == "Negative")).sum()
    if billing_neg > 0:
        insight_card(T["ins_red"],
            f"<b>Billing and Trust</b> — {billing_neg} reviews flag incorrect billing or fraudulent behaviour.")

    if "_rating" in df.columns:
        mismatch = ((df["_rating"] >= 4) & (df["sentiment"] == "Negative")).sum()
        if mismatch > 0:
            insight_card(T["ins_blue"],
                f"<b>Rating–Sentiment Mismatch</b> — {mismatch} reviews carry a 4–5 star rating yet "
                "the model detects negative sentiment. Customers may be leaving courtesy ratings "
                "despite genuinely negative experiences.")

    if "_title" in df.columns and df["_title"].nunique() > 1:
        op = (
            df.groupby("_title")
            .apply(lambda x: (x["sentiment"] == "Positive").mean())
            .sort_values(ascending=False)
        )
        insight_card(T["ins_green"],
            f"<b>Outlet Performance</b> — Best: '{op.index[0]}' ({op.iloc[0]*100:.1f}% positive). "
            f"Lowest: '{op.index[-1]}' ({op.iloc[-1]*100:.1f}% positive).")

    st.markdown("<br>", unsafe_allow_html=True)

    if "_rating" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            section("Rating vs Predicted Sentiment")
            cross = pd.crosstab(df["_rating"].round(0).astype(int), df["sentiment"])
            cs = "Blues" if not _dark else "Purples"
            fig = px.imshow(
                cross, color_continuous_scale=cs, aspect="auto",
                labels={"x": "Predicted Sentiment", "y": "User Rating", "color": "Count"},
            )
            fig.update_layout(**chart_layout(height=320, margin=dict(t=20, b=20, l=60, r=20)))
            chart(fig)
            st.caption("High user ratings paired with negative predicted sentiment indicate courtesy ratings masking genuine dissatisfaction.")

        with col2:
            section("Sentiment Composition by Star Rating")
            rs = df.groupby(["_rating", "sentiment"]).size().reset_index(name="count")
            rs["pct"] = rs.groupby("_rating")["count"].transform(lambda x: x / x.sum() * 100)
            fig2 = px.bar(
                rs, x="_rating", y="pct",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"_rating": "Star Rating", "pct": "Share (%)", "sentiment": "Sentiment"},
            )
            fig2.update_layout(**chart_layout(
                height=320, margin=dict(t=48, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            ))
            chart(fig2)

    if "_title" in df.columns and df["_title"].nunique() > 1:
        section("Outlet Scorecard")
        sc = (
            df.groupby("_title")
            .agg(
                Total    =("sentiment", "count"),
                Positive =("sentiment", lambda x: (x == "Positive").sum()),
                Negative =("sentiment", lambda x: (x == "Negative").sum()),
                Avg_Conf =("confidence", "mean"),
            )
            .assign(
                Pos_Pct=lambda x: x["Positive"] / x["Total"] * 100,
                Neg_Pct=lambda x: x["Negative"] / x["Total"] * 100,
            )
            .sort_values("Neg_Pct", ascending=False)
            .reset_index()
        )
        sc.columns = ["Outlet", "Total", "Positive", "Negative", "Avg Confidence", "Positive %", "Negative %"]
        st.dataframe(
            sc.style
                .background_gradient(subset=["Negative %"], cmap="Reds")
                .background_gradient(subset=["Positive %"], cmap="Greens")
                .format({"Positive %": "{:.1f}", "Negative %": "{:.1f}", "Avg Confidence": "{:.2f}"}),
            use_container_width=True,
        )

    section("Recommended Actions")
    neg_rates = (
        df[df["sentiment"] == "Negative"]["issue_category"]
        .value_counts(normalize=True)
        .head(5)
    )
    priority_labels = ["Critical", "High", "Medium", "Medium", "Low"]
    rows = [
        {
            "Priority":           priority_labels[i],
            "Issue Area":         issue,
            "Negative Share":     f"{rate*100:.1f}%",
            "Recommended Action": _action_for(issue),
        }
        for i, (issue, rate) in enumerate(neg_rates.items())
    ]
    if rows:
        st.table(pd.DataFrame(rows))


def _action_for(issue):
    lookup = {
        "Staff Behaviour":             "Conduct targeted staff training and implement a service feedback mechanism at point-of-sale.",
        "Waiting Time":                "Review lane management and staffing levels; add attendants during peak hours.",
        "Payment Issue":               "Audit all POS terminals and ensure every digital payment channel is fully operational.",
        "Billing or Trust Issue":      "Install display meters, run regular oversight audits, and publicise the anti-fraud policy.",
        "CNG Availability":            "Monitor the CNG supply chain proactively and keep backup connections active.",
        "Cleanliness":                 "Increase cleaning frequency and assign dedicated housekeeping staff to the site.",
        "Fuel Quality":                "Schedule regular independent quality checks and display fuel certification prominently.",
        "Facility Maintenance":        "Implement a preventive maintenance schedule and track equipment health centrally.",
        "Customer Amenities":          "Upgrade water and restroom facilities; add seating where space permits.",
        "Safety Concern":              "Conduct a full safety audit and reinforce signage and fire-safety equipment.",
        "Traffic or Queue Management": "Redesign lane flow and consider installing a queue management display system.",
        "Accessibility":               "Improve directional signage and ensure disabled access pathways are unobstructed.",
    }
    return lookup.get(issue, "Investigate the root cause and implement a targeted operational improvement.")


# ── Tab 5: Data Explorer ───────────────────────────────────────────────────────

def tab_data(df, df_raw):
    section("Analysed Reviews")

    display_cols = ["_review_text", "_rating", "sentiment", "bert_star", "confidence", "issue_category", "issue_tags"]
    display_cols = [c for c in display_cols if c in df.columns]

    col_rename = {
        "_review_text":   "Review Text",
        "_rating":        "User Rating",
        "sentiment":      "Sentiment",
        "bert_star":      "Predicted Stars",
        "confidence":     "Confidence",
        "issue_category": "Issue Category",
        "issue_tags":     "Issue Tags",
    }

    st.dataframe(df[display_cols].rename(columns=col_rename), use_container_width=True, height=480)

    csv_bytes = df[display_cols].rename(columns=col_rename).to_csv(index=False).encode()
    st.download_button(
        "Download Results as CSV",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    section("Raw Data — First 50 Rows")
    st.dataframe(df_raw.head(50), use_container_width=True)


# ── Station-level dashboard (Mumbai petrol pumps) ─────────────────────────────

def main_station(df_raw):
    st.markdown(
        '<div class="page-header">'
        '<div class="page-header-icon">🗺️</div>'
        '<div>'
        '<div class="page-header-title">Mumbai Petrol Pump Intelligence</div>'
        '<div class="page-header-sub">'
        '163 stations &nbsp;·&nbsp; Station-level ratings &nbsp;·&nbsp; '
        'Geographic distribution &nbsp;·&nbsp; CNG coverage &nbsp;·&nbsp; Zone analysis'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    df = normalize_station_dataframe(df_raw.copy())
    df_valid = df[df["sentiment"] != "Unknown"].copy()

    # ── sidebar filters ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    sel_sentiments = st.sidebar.multiselect(
        "Sentiment (by avg rating)",
        sorted(df_valid["sentiment"].unique()),
        default=sorted(df_valid["sentiment"].unique()),
    )
    if "zone" in df_valid.columns:
        sel_zones = st.sidebar.multiselect(
            "Zone", sorted(df_valid["zone"].unique()),
            default=sorted(df_valid["zone"].unique()),
        )
        df_valid = df_valid[df_valid["zone"].isin(sel_zones)]
    if "fuel_type" in df_valid.columns:
        sel_ft = st.sidebar.multiselect(
            "Fuel Type", sorted(df_valid["fuel_type"].unique()),
            default=sorted(df_valid["fuel_type"].unique()),
        )
        df_valid = df_valid[df_valid["fuel_type"].isin(sel_ft)]

    df_valid = df_valid[df_valid["sentiment"].isin(sel_sentiments)]

    if df_valid.empty:
        st.warning("No stations match the current filters.")
        return

    tab_labels = ["Overview", "Geographic Map", "Zone Analysis", "Station Scorecard", "Insights & Actions", "Data Explorer"]
    tabs = st.tabs(tab_labels)
    with tabs[0]: tab_station_overview(df_valid)
    with tabs[1]: tab_geo_map(df_valid)
    with tabs[2]: tab_zone_analysis(df_valid)
    with tabs[3]: tab_station_scorecard(df_valid)
    with tabs[4]: tab_station_insights(df_valid)
    with tabs[5]: tab_station_data(df_valid, df_raw)


def tab_station_overview(df):
    total       = len(df)
    pos         = (df["sentiment"] == "Positive").sum()
    neu         = (df["sentiment"] == "Neutral").sum()
    neg         = (df["sentiment"] == "Negative").sum()
    avg_rating  = df["_rating"].mean() if "_rating" in df.columns else None
    cng_count   = (df.get("fuel_type", pd.Series()) == "CNG").sum()
    top_zone    = df["zone"].value_counts().index[0] if "zone" in df.columns else "—"

    st.markdown(
        f"Analysed **{total:,} Mumbai petrol pump stations**. "
        f"{pos} stations rated Positive (≥ 3.6★), {neu} Neutral (2.6–3.5★), {neg} Negative (≤ 2.5★). "
        f"CNG stations: **{int(cng_count)}** out of {total}."
    )

    section("Station Health Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stations",   f"{total:,}")
    c2.metric("Positive (≥ 3.6★)", f"{pos:,}",  f"{pos/total*100:.1f}% of total")
    c3.metric("Neutral (2.6–3.5★)", f"{neu:,}",  f"{neu/total*100:.1f}% of total")
    c4.metric("Negative (≤ 2.5★)", f"{neg:,}",  f"{neg/total*100:.1f}% of total")

    r1, r2, r3 = st.columns(3)
    if avg_rating is not None:
        r1.metric("Avg Station Rating",  f"{avg_rating:.2f} / 5.00")
    if "_review_count" in df.columns:
        total_reviews = int(df["_review_count"].sum())
        avg_reviews   = df["_review_count"].median()
        r2.metric("Total Reviews Covered", f"{total_reviews:,}")
        r3.metric("Median Reviews / Station", f"{avg_reviews:,.0f}")

    section("Rating Distribution")
    col1, col2 = st.columns(2)

    with col1:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Stations"]
        fig = px.pie(
            counts, names="Sentiment", values="Stations",
            color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.52,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11, color=T["font_color"]))
        fig.update_layout(**chart_layout(height=340, showlegend=False,
                                         margin=dict(t=20, b=20, l=20, r=20)))
        chart(fig)

    with col2:
        if "_rating" in df.columns:
            fig2 = px.histogram(
                df, x="_rating", nbins=20,
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="overlay", opacity=0.75,
                labels={"_rating": "Avg Station Rating (1–5)", "sentiment": "Sentiment"},
            )
            fig2.update_layout(**chart_layout(
                height=340, margin=dict(t=48, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, font=dict(size=11)),
            ))
            chart(fig2)

    section("Fuel Type Breakdown")
    if "fuel_type" in df.columns:
        ft_sent = df.groupby(["fuel_type", "sentiment"]).size().reset_index(name="count")
        fig3 = px.bar(
            ft_sent, x="fuel_type", y="count",
            color="sentiment", color_discrete_map=SENTIMENT_COLORS,
            barmode="group",
            labels={"fuel_type": "Fuel Type", "count": "Stations", "sentiment": "Sentiment"},
        )
        fig3.update_layout(**chart_layout(
            height=320, margin=dict(t=48, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=11)),
        ))
        chart(fig3)

    if "_review_count" in df.columns:
        section("Review Volume Distribution (log scale)")
        fig4 = px.histogram(
            df, x="_review_count", nbins=30, log_x=True,
            color="sentiment", color_discrete_map=SENTIMENT_COLORS,
            barmode="overlay", opacity=0.72,
            labels={"_review_count": "Total Google Reviews (log scale)", "sentiment": "Sentiment"},
        )
        fig4.update_layout(**chart_layout(
            height=260, margin=dict(t=48, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=11)),
        ))
        chart(fig4)


@st.cache_data(show_spinner=False)
def _load_mumbai_boundary():
    import json, os
    path = os.path.join(os.path.dirname(__file__), "mumbai_boundary.geojson")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tab_geo_map(df):
    if "_lat" not in df.columns or "_lng" not in df.columns:
        st.info("No latitude / longitude columns found in this dataset.")
        return

    df_map = df[df["_lat"].notna() & df["_lng"].notna()].copy()
    if df_map.empty:
        st.info("No stations with valid coordinates found.")
        return

    section("Mumbai Petrol Pump Map")
    st.caption(
        f"Geospatial distribution of {len(df_map):,} petrol pump stations across Greater Mumbai. "
        "Marker colour indicates sentiment classification derived from aggregate customer ratings. "
        "Marker size is proportional to total review volume — larger markers represent stations "
        "with a higher number of verified ratings, providing greater statistical confidence. "
        "The grey boundary delineates the Greater Mumbai Municipal Corporation (BMC) jurisdiction."
    )

    hover_cols = ["_title", "_rating", "_review_count", "zone", "fuel_type"]
    hover_cols = [c for c in hover_cols if c in df_map.columns]

    if "_review_count" in df_map.columns:
        df_map["_marker_size"] = np.log1p(df_map["_review_count"]) * 2 + 5
        size_col = "_marker_size"
    else:
        size_col = None

    fig = px.scatter_mapbox(
        df_map,
        lat="_lat", lon="_lng",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        size=size_col,
        size_max=22,
        hover_name="_title" if "_title" in df_map.columns else None,
        hover_data={c: True for c in hover_cols if c not in ["_title", "_marker_size"]},
        zoom=10.5,
        center={"lat": df_map["_lat"].mean(), "lon": df_map["_lng"].mean()},
        height=580,
    )

    # Overlay the Mumbai municipal boundary from the GeoJSON
    boundary_gj = _load_mumbai_boundary()
    mapbox_layers = []
    if boundary_gj:
        mapbox_layers.append({
            "sourcetype": "geojson",
            "source":     boundary_gj,
            "type":       "line",
            "color":      "#64748b",
            "line":       {"width": 1.5},
            "opacity":    0.7,
        })

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_layers=mapbox_layers,
        **chart_layout(
            height=580,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=11)),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    if "zone" in df_map.columns:
        section("Station Count by Zone")
        zone_counts = df_map["zone"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Stations"]
        fig2 = px.bar(
            zone_counts.sort_values("Stations"), x="Stations", y="Zone",
            orientation="h", color_discrete_sequence=["#3b82f6"],
            labels={"Stations": "Number of Petrol Pumps", "Zone": ""},
        )
        fig2.update_layout(**chart_layout(
            height=300, showlegend=False, margin=dict(t=10, b=20, l=20, r=20),
        ))
        chart(fig2)


def tab_zone_analysis(df):
    if "zone" not in df.columns:
        st.info("Zone data not available.")
        return

    section("Rating & Sentiment by Zone")
    zone_stats = (
        df.groupby("zone")
        .agg(
            Stations  =("_rating", "count"),
            Avg_Rating=("_rating", "mean"),
            Pos       =("sentiment", lambda x: (x == "Positive").sum()),
            Neg       =("sentiment", lambda x: (x == "Negative").sum()),
        )
        .assign(
            Pos_Pct=lambda x: x["Pos"] / x["Stations"] * 100,
            Neg_Pct=lambda x: x["Neg"] / x["Stations"] * 100,
        )
        .sort_values("Avg_Rating", ascending=False)
        .reset_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            zone_stats.sort_values("Avg_Rating"), x="Avg_Rating", y="zone",
            orientation="h", color="Avg_Rating",
            color_continuous_scale="RdYlGn",
            range_color=[2.5, 4.5],
            labels={"Avg_Rating": "Avg Station Rating", "zone": ""},
        )
        fig.update_layout(**chart_layout(
            height=340, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10, b=20, l=20, r=20),
        ))
        chart(fig)

    with col2:
        zone_sent = df.groupby(["zone", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(
            zone_sent, x="zone", y="count",
            color="sentiment", color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            labels={"zone": "Zone", "count": "Stations", "sentiment": "Sentiment"},
        )
        fig2.update_xaxes(tickangle=20)
        fig2.update_layout(**chart_layout(
            height=340, margin=dict(t=48, b=60, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=11)),
        ))
        chart(fig2)

    if "fuel_type" in df.columns:
        section("CNG Station Coverage by Zone")
        cng_zone = df.groupby(["zone", "fuel_type"]).size().reset_index(name="count")
        fig3 = px.bar(
            cng_zone, x="zone", y="count",
            color="fuel_type",
            color_discrete_sequence=["#3b82f6", "#10b981"],
            barmode="group",
            labels={"zone": "Zone", "count": "Stations", "fuel_type": "Type"},
        )
        fig3.update_xaxes(tickangle=20)
        fig3.update_layout(**chart_layout(
            height=320, margin=dict(t=48, b=60, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=11)),
        ))
        chart(fig3)

    section("Zone Summary Table")
    display_zone = zone_stats.copy()
    display_zone["Avg_Rating"]  = display_zone["Avg_Rating"].round(2)
    display_zone["Pos_Pct"]     = display_zone["Pos_Pct"].round(1)
    display_zone["Neg_Pct"]     = display_zone["Neg_Pct"].round(1)
    display_zone.columns = ["Zone", "Stations", "Avg Rating", "Positive", "Negative", "Positive %", "Negative %"]
    st.dataframe(
        display_zone.style
            .background_gradient(subset=["Avg Rating"], cmap="RdYlGn")
            .background_gradient(subset=["Negative %"], cmap="Reds")
            .format({"Avg Rating": "{:.2f}", "Positive %": "{:.1f}", "Negative %": "{:.1f}"}),
        use_container_width=True,
    )


def tab_station_scorecard(df):
    section("All Stations — Ranked by Rating")
    st.caption(
        "Sorted by average Google rating (highest first). "
        "Use Total Reviews to judge how much to trust each rating — "
        "stations with fewer than 50 reviews are flagged separately in the Insights tab."
    )

    cols = ["_title", "_rating", "_review_count", "sentiment", "zone", "fuel_type", "_address"]
    cols = [c for c in cols if c in df.columns]
    sc   = df[cols].copy().sort_values("_rating", ascending=False)

    rename = {
        "_title": "Station Name", "_rating": "Avg Rating",
        "_review_count": "Total Reviews",
        "sentiment": "Sentiment", "zone": "Zone",
        "fuel_type": "Fuel Type", "_address": "Address",
    }
    sc = sc.rename(columns=rename)

    style_cols = {}
    if "Avg Rating"    in sc.columns: style_cols["Avg Rating"]    = "{:.2f}"
    if "Total Reviews" in sc.columns: style_cols["Total Reviews"] = "{:,.0f}"

    styled = sc.style.format(style_cols)
    if "Avg Rating" in sc.columns:
        styled = styled.background_gradient(subset=["Avg Rating"], cmap="RdYlGn")

    st.dataframe(styled, use_container_width=True, height=520)

    section("Bottom 15 Stations — Priority Attention")
    bottom = sc.tail(15).sort_values("Avg Rating")
    st.dataframe(
        bottom.style
            .background_gradient(subset=["Avg Rating"], cmap="Reds")
            .format(style_cols),
        use_container_width=True,
    )

    csv_bytes = sc.to_csv(index=False).encode()
    st.download_button("Download Scorecard as CSV", data=csv_bytes,
                       file_name="mumbai_station_scorecard.csv", mime="text/csv")


def tab_station_insights(df):
    total       = len(df)
    neg_count   = (df["sentiment"] == "Negative").sum()
    neg_pct     = neg_count / total * 100 if total else 0
    avg_rating  = df["_rating"].mean() if "_rating" in df.columns else None

    section("Mumbai Petrol Pump — Automated Insights")

    if neg_pct > 30:
        insight_card(T["ins_red"],
            f"<b>High Negativity Across City</b> — {neg_pct:.1f}% of Mumbai stations "
            f"({neg_count} of {total}) have an average rating ≤ 2.5★, indicating widespread "
            "customer dissatisfaction that warrants a city-wide operational review.")
    elif neg_pct > 15:
        insight_card(T["ins_amber"],
            f"<b>Moderate Negativity</b> — {neg_pct:.1f}% of stations have low aggregate ratings. "
            "Targeted improvement in identified zones can meaningfully lift city-wide performance.")
    else:
        insight_card(T["ins_green"],
            f"<b>Broadly Satisfactory</b> — Only {neg_pct:.1f}% of Mumbai stations are rated negatively.")

    if avg_rating is not None:
        insight_card(T["ins_blue"],
            f"<b>City Average Rating: {avg_rating:.2f} / 5.00</b> — "
            f"{'Above' if avg_rating >= 3.6 else 'Below' if avg_rating < 3.0 else 'At'} "
            f"the Positive threshold (3.6). "
            "Pune average for reference: run the Pune dataset to compare directly.")

    if "fuel_type" in df.columns:
        cng  = (df["fuel_type"] == "CNG").sum()
        pct  = cng / total * 100
        cng_neg = ((df["fuel_type"] == "CNG") & (df["sentiment"] == "Negative")).sum()
        insight_card(T["ins_purple"],
            f"<b>CNG Coverage</b> — {int(cng)} of {total} stations ({pct:.1f}%) are CNG-dedicated. "
            f"{int(cng_neg)} CNG stations have low ratings, signalling potential supply or "
            "availability reliability issues.")

    if "zone" in df.columns:
        zone_neg = (
            df[df["sentiment"] == "Negative"]["zone"].value_counts()
        )
        if not zone_neg.empty:
            worst_zone = zone_neg.index[0]
            insight_card(T["ins_red"],
                f"<b>Worst-Performing Zone: {worst_zone}</b> — "
                f"{zone_neg.iloc[0]} negative-rated stations. "
                "Prioritise operational audits in this area.")

        zone_avg = df.groupby("zone")["_rating"].mean().sort_values(ascending=False)
        if len(zone_avg) >= 2:
            insight_card(T["ins_green"],
                f"<b>Best Zone: {zone_avg.index[0]}</b> ({zone_avg.iloc[0]:.2f}★ avg) vs "
                f"<b>Weakest Zone: {zone_avg.index[-1]}</b> ({zone_avg.iloc[-1]:.2f}★ avg). "
                f"Rating gap of {zone_avg.iloc[0]-zone_avg.iloc[-1]:.2f} stars between best and worst zone.")

    if "_review_count" in df.columns:
        low_evidence = (df["_review_count"] < 50).sum()
        if low_evidence > 0:
            insight_card(T["ins_amber"],
                f"<b>Low-Evidence Stations: {int(low_evidence)}</b> — "
                "These stations have fewer than 50 Google reviews. "
                "Their ratings are less statistically reliable and should be monitored rather than acted upon immediately.")

    if "_rating" in df.columns and "_title" in df.columns:
        # filter out low-evidence stations (< 50 reviews) before surfacing top/bottom
        df_evidence = df[df["_review_count"] >= 50] if "_review_count" in df.columns else df
        if not df_evidence.empty:
            top5    = df_evidence.nlargest(5, "_rating")[["_title", "_rating"]]
            bottom5 = df_evidence.nsmallest(5, "_rating")[["_title", "_rating"]]
            insight_card(T["ins_green"],
                "<b>Top 5 Stations by Rating (min. 50 reviews):</b> "
                + "; ".join(f"<i>{r['_title']}</i> ({r['_rating']:.1f}★)" for _, r in top5.iterrows()))
            insight_card(T["ins_red"],
                "<b>Bottom 5 Stations — Immediate Action Needed (min. 50 reviews):</b> "
                + "; ".join(f"<i>{r['_title']}</i> ({r['_rating']:.1f}★)" for _, r in bottom5.iterrows()))

    section("Recommended Actions")
    actions = []
    if "zone" in df.columns:
        for zone in df["zone"].unique():
            zn = df[df["zone"] == zone]
            neg_r = (zn["sentiment"] == "Negative").mean() * 100
            if neg_r > 30:
                actions.append({
                    "Priority": "Critical", "Zone": zone,
                    "Negative %": f"{neg_r:.1f}%",
                    "Action": "Conduct full operational audit; escalate to zone manager for immediate corrective plan.",
                })
            elif neg_r > 15:
                actions.append({
                    "Priority": "High", "Zone": zone,
                    "Negative %": f"{neg_r:.1f}%",
                    "Action": "Schedule structured review with outlet managers; monitor rating trends monthly.",
                })
    if "fuel_type" in df.columns:
        cng_neg_r = df[df["fuel_type"] == "CNG"]["sentiment"].eq("Negative").mean() * 100
        if cng_neg_r > 20:
            actions.append({
                "Priority": "High", "Zone": "City-wide (CNG)",
                "Negative %": f"{cng_neg_r:.1f}%",
                "Action": "Audit CNG supply chain reliability; investigate downtime frequency across low-rated CNG stations.",
            })
    if "_review_count" in df.columns:
        actions.append({
            "Priority": "Medium", "Zone": "All",
            "Negative %": "—",
            "Action": f"Encourage customer reviews at {int(low_evidence if '_review_count' in df.columns else 0)} low-evidence stations to improve data reliability.",
        })

    if actions:
        st.table(pd.DataFrame(actions))


def tab_station_data(df, df_raw):
    section("Processed Station Data")
    cols = ["_title", "_rating", "_review_count", "sentiment",
            "zone", "fuel_type", "_address", "_lat", "_lng"]
    cols = [c for c in cols if c in df.columns]
    rename = {
        "_title": "Station Name", "_rating": "Avg Rating",
        "_review_count": "Total Reviews",
        "sentiment": "Sentiment", "zone": "Zone",
        "fuel_type": "Fuel Type", "_address": "Address",
        "_lat": "Latitude", "_lng": "Longitude",
    }
    st.dataframe(df[cols].rename(columns=rename), use_container_width=True, height=480)

    csv_bytes = df[cols].rename(columns=rename).to_csv(index=False).encode()
    st.download_button("Download Processed Data as CSV", data=csv_bytes,
                       file_name="mumbai_stations_processed.csv", mime="text/csv")

    st.markdown("---")
    section("Raw Source Data — First 50 Rows")
    st.dataframe(df_raw.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
