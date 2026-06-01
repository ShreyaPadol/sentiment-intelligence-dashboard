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
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding: 2rem 2.5rem 2.5rem 2.5rem; }

    /* page title */
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: #1a1a2e !important; letter-spacing: -0.02em; }

    /* metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* section headers */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin: 2rem 0 0.75rem 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
    }

    /* insight cards */
    .insight-card {
        background: #ffffff;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border: 1px solid #e5e7eb;
        border-left-width: 4px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #374151;
    }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e5e7eb;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 20px;
        border-radius: 0;
        color: #6b7280;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a2e !important;
        border-bottom-color: #1a1a2e !important;
        background: transparent !important;
    }

    /* sidebar */
    div[data-testid="stSidebarContent"] { padding-top: 1.5rem; background: #f9fafb; }
    .st-emotion-cache-1gwvy71 h1 { font-size: 1rem !important; }

    /* dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid #e5e7eb; }

    /* buttons */
    .stDownloadButton button {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
        border-radius: 6px !important;
    }

    /* hide plotly mode bar until hover */
    .modebar { opacity: 0 !important; transition: opacity 0.2s; }
    .modebar:hover { opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Neutral":  "#d97706",
    "Negative": "#dc2626",
}

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=11, color="#374151"),
    title_font=dict(size=12, color="#111827", family="Inter, Arial, sans-serif"),
    margin=dict(t=48, b=20, l=20, r=20),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hoverlabel=dict(font_size=11, font_family="Inter, Arial, sans-serif"),
)

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "and", "to", "for", "with",
    "on", "at", "by", "from", "this", "was", "are", "be", "not", "but",
    "have", "had", "has", "they", "their", "there", "that", "very", "so",
    "my", "we", "i", "its", "here", "get", "got", "also", "just", "no",
    "or", "as", "do", "did", "been", "all", "one", "if", "up", "out",
    "about", "than", "more", "when", "will", "can", "good", "place",
    "nice", "would", "like", "really", "come",
}


def section(label):
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)


def top_keywords(text, n=10):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return Counter(words).most_common(n)


@st.cache_data(show_spinner=False)
def load_default_pune():
    return pd.read_excel("Pune_Retail_outlet (1).xlsx")


def run_analysis(df):
    texts = df["_review_text"].fillna("").tolist()
    with st.spinner("Running BERT sentiment analysis — this takes a moment on first load."):
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


def sidebar():
    st.sidebar.markdown("### Data Source")
    mode = st.sidebar.radio(
        "Dataset",
        ["Pune Retail Outlet (default)", "Upload CSV / XLSX"],
        label_visibility="collapsed",
    )

    df_raw = None
    if mode == "Pune Retail Outlet (default)":
        df_raw = load_default_pune()
        st.sidebar.success(f"{len(df_raw):,} reviews loaded")
    else:
        uploaded = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.sidebar.success(f"{len(df_raw):,} rows loaded")
        else:
            st.sidebar.info("Upload a CSV or XLSX file to begin.")
    return df_raw


def main():
    st.title("Sentiment Intelligence Dashboard")
    st.caption("BERT-powered review analysis  |  Issue categorisation  |  Time-series trends  |  Operational insights")

    df_raw = sidebar()
    if df_raw is None:
        st.info("Select or upload a dataset from the sidebar to begin.")
        return

    df, _ = normalize_dataframe(df_raw.copy())

    if "_review_text" not in df.columns:
        st.error(
            "Could not detect a review text column. "
            "Expected one of: review_text_final, text, review, comment, body."
        )
        st.dataframe(df.head())
        return

    df = df[df["_review_text"].notna() & (df["_review_text"].str.strip() != "")].reset_index(drop=True)

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
        "Sentiment", sorted(df["sentiment"].unique()), default=sorted(df["sentiment"].unique())
    )
    sel_issues = st.sidebar.multiselect(
        "Issue Category", sorted(df["issue_category"].unique()), default=sorted(df["issue_category"].unique())
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

    df = df[df["sentiment"].isin(sel_sentiments) & df["issue_category"].isin(sel_issues)]

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


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def tab_overview(df):
    total = len(df)
    pos   = (df["sentiment"] == "Positive").sum()
    neg   = (df["sentiment"] == "Negative").sum()
    neu   = (df["sentiment"] == "Neutral").sum()

    dominant  = max(["Positive", "Neutral", "Negative"], key=lambda s: (df["sentiment"] == s).sum())
    top_issue = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()

    st.markdown(
        f"Analysed **{total:,}** reviews. Sentiment is predominantly **{dominant}** "
        f"({pos:,} positive, {neu:,} neutral, {neg:,} negative).",
        help="Sentiment classified using nlptown/bert-base-multilingual-uncased-sentiment"
    )

    section("Sentiment Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews",  f"{total:,}")
    c2.metric("Positive",       f"{pos:,}",  f"{pos/total*100:.1f}%")
    c3.metric("Neutral",        f"{neu:,}",  f"{neu/total*100:.1f}%")
    c4.metric("Negative",       f"{neg:,}",  f"{neg/total*100:.1f}%")

    if "_rating" in df.columns:
        avg_r    = df["_rating"].dropna().mean()
        bert_avg = df["bert_star"].mean()
        r1, r2, r3 = st.columns(3)
        r1.metric("Avg User Rating",  f"{avg_r:.2f} / 5")
        r2.metric("Avg BERT Stars",   f"{bert_avg:.2f} / 5", f"{bert_avg - avg_r:+.2f} vs user rating")
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
            hole=0.5,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=320, title_text="Overall Sentiment Split")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "_rating" in df.columns:
            rs = df.groupby(["_rating", "sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(
                rs, x="_rating", y="count",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"_rating": "Star Rating", "count": "Reviews", "sentiment": "Sentiment"},
            )
            fig2.update_layout(**CHART_LAYOUT, height=320, title_text="BERT Sentiment by User Star Rating")
            st.plotly_chart(fig2, use_container_width=True)

    section("Top Keywords by Sentiment")
    kc1, kc2, kc3 = st.columns(3)
    for col, sentiment, color in zip(
        [kc1, kc2, kc3],
        ["Positive", "Neutral", "Negative"],
        ["#16a34a", "#d97706", "#dc2626"],
    ):
        blob = " ".join(df[df["sentiment"] == sentiment]["_review_text"].dropna())
        kws  = top_keywords(blob, n=10)
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Word", "Count"]).sort_values("Count")
            fig_k = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color_discrete_sequence=[color],
                labels={"Count": "Frequency", "Word": ""},
            )
            fig_k.update_layout(
                **CHART_LAYOUT,
                title_text=f"{sentiment} — Top Keywords",
                showlegend=False,
                height=300,
                margin=dict(t=44, b=12, l=8, r=8),
            )
            col.plotly_chart(fig_k, use_container_width=True)

    section("Model Confidence")
    fig3 = px.histogram(
        df, x="confidence", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=35, barmode="overlay", opacity=0.7,
        labels={"confidence": "Confidence Score", "sentiment": "Sentiment"},
    )
    fig3.update_layout(**CHART_LAYOUT, height=260, title_text="BERT Confidence Score Distribution")
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Issue Analysis
# ---------------------------------------------------------------------------

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
    fig.update_layout(**CHART_LAYOUT, height=380, title_text="Reviews by Issue Category and Sentiment")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Category Share")
        ic = df["issue_category"].value_counts().reset_index()
        ic.columns = ["Category", "Count"]
        fig2 = px.pie(
            ic, names="Category", values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label", textfont_size=10)
        fig2.update_layout(**CHART_LAYOUT, showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

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
        fig3.update_layout(**CHART_LAYOUT, height=380, showlegend=False,
                           title_text="Proportion of Negative Reviews per Category")
        st.plotly_chart(fig3, use_container_width=True)

    if "_title" in df.columns and df["_title"].nunique() > 1:
        section("Outlet vs Issue Category — Review Count")
        pivot = df.groupby(["_title", "issue_category"]).size().unstack(fill_value=0)
        fig4 = px.imshow(
            pivot, aspect="auto", color_continuous_scale="YlOrRd",
            labels={"x": "Issue Category", "y": "Outlet", "color": "Reviews"},
        )
        fig4.update_layout(**CHART_LAYOUT, height=420, margin=dict(t=48, b=20, l=170, r=20))
        st.plotly_chart(fig4, use_container_width=True)

    section("Browse Negative Reviews")
    selected = st.selectbox("Issue category", sorted(df["issue_category"].unique()), label_visibility="collapsed")
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
                "confidence":   "BERT Confidence",
            }),
            use_container_width=True,
            height=300,
        )


# ---------------------------------------------------------------------------
# Tab 3 — Time-Series
# ---------------------------------------------------------------------------

def tab_timeseries(df):
    if "_date" not in df.columns or df["_date"].notna().sum() < 5:
        st.info("No date column detected. Time-series analysis is not available for this dataset.")
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
    fig.update_layout(**CHART_LAYOUT, height=340, title_text=f"Review Volume by Sentiment ({granularity})")
    st.plotly_chart(fig, use_container_width=True)

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
    fig2.update_layout(**CHART_LAYOUT, height=300, title_text=f"Sentiment Share % ({granularity})")
    st.plotly_chart(fig2, use_container_width=True)

    if "_rating" in df_t.columns:
        section("Average Rating and Review Volume")
        rt = df_t.groupby(period_col)["_rating"].agg(["mean", "count"]).reset_index()
        rt.columns = [period_col, "avg_rating", "review_count"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rt[period_col], y=rt["avg_rating"],
            mode="lines+markers", name="Avg Rating",
            line=dict(color="#2563eb", width=2),
        ))
        fig3.add_trace(go.Bar(
            x=rt[period_col], y=rt["review_count"],
            name="Review Count", opacity=0.2,
            marker_color="#9ca3af", yaxis="y2",
        ))
        fig3.update_layout(
            **CHART_LAYOUT,
            height=320,
            title_text=f"Avg Rating and Review Volume ({granularity})",
            yaxis=dict(title="Avg Rating", range=[0.8, 5.2], gridcolor="#f3f4f6"),
            yaxis2=dict(title="Review Count", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

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
    fig4.update_layout(**CHART_LAYOUT, height=340, title_text=f"Top 6 Issue Categories ({granularity})")
    st.plotly_chart(fig4, use_container_width=True)

    neg_by_period = (
        df_t[df_t["sentiment"] == "Negative"]
        .groupby(period_col).size().reset_index(name="neg_count")
    )
    if len(neg_by_period) >= 4:
        mean_n = neg_by_period["neg_count"].mean()
        std_n  = neg_by_period["neg_count"].std()
        neg_by_period["spike"] = neg_by_period["neg_count"] > mean_n + 1.5 * std_n
        spikes = neg_by_period[neg_by_period["spike"]]
        if not spikes.empty:
            st.warning(
                "Negative review spike detected in: "
                + ", ".join(spikes[period_col].astype(str).tolist())
            )
            section("Negative Review Spikes")
            fig5 = px.bar(
                neg_by_period, x=period_col, y="neg_count",
                color="spike",
                color_discrete_map={True: "#dc2626", False: "#d1d5db"},
                labels={period_col: "Period", "neg_count": "Negative Reviews"},
            )
            fig5.update_layout(**CHART_LAYOUT, height=260, showlegend=False,
                               title_text="Negative Review Volume — Spikes Highlighted")
            st.plotly_chart(fig5, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Deep Insights
# ---------------------------------------------------------------------------

def tab_insights(df):
    total     = len(df)
    neg_count = (df["sentiment"] == "Negative").sum()
    neg_pct   = neg_count / total * 100

    section("Automated Insights")

    insights = []

    if neg_pct > 50:
        insights.append(("#fef2f2", "#dc2626",
            f"<b>High Negativity</b> — {neg_pct:.1f}% of reviews are negative. Immediate operational attention is required."))
    elif neg_pct > 30:
        insights.append(("#fffbeb", "#d97706",
            f"<b>Moderate Negativity</b> — {neg_pct:.1f}% of reviews are negative. There is clear room for improvement."))
    else:
        insights.append(("#f0fdf4", "#16a34a",
            f"<b>Broadly Positive</b> — Only {neg_pct:.1f}% of reviews are negative."))

    neg_issues = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()
    if not neg_issues.empty:
        ti, tc = neg_issues.index[0], neg_issues.iloc[0]
        insights.append(("#fffbeb", "#d97706",
            f"<b>Top Pain Point</b> — '{ti}' accounts for {tc} negative reviews "
            f"({tc/neg_count*100:.1f}% of all negatives)."))

    staff_neg = ((df["issue_category"] == "Staff Behaviour") & (df["sentiment"] == "Negative")).sum()
    if staff_neg > 0:
        insights.append(("#faf5ff", "#7c3aed",
            f"<b>Staff Behaviour</b> — {staff_neg} reviews raise concerns about staff conduct. Staff training is recommended."))

    cng_neg = ((df["issue_category"] == "CNG Availability") & (df["sentiment"] == "Negative")).sum()
    if cng_neg > 0:
        insights.append(("#fef2f2", "#dc2626",
            f"<b>CNG Availability</b> — {cng_neg} negative reviews specifically mention CNG unavailability."))

    billing_neg = ((df["issue_category"] == "Billing or Trust Issue") & (df["sentiment"] == "Negative")).sum()
    if billing_neg > 0:
        insights.append(("#fef2f2", "#dc2626",
            f"<b>Billing and Trust</b> — {billing_neg} reviews flag incorrect billing or fraudulent behaviour."))

    if "_rating" in df.columns:
        mismatch = ((df["_rating"] >= 4) & (df["sentiment"] == "Negative")).sum()
        if mismatch > 0:
            insights.append(("#eff6ff", "#2563eb",
                f"<b>Rating–Sentiment Mismatch</b> — {mismatch} reviews carry a 4–5 star rating yet "
                f"BERT detects negative sentiment. Customers may be leaving courtesy ratings "
                f"despite genuinely negative experiences."))

    if "_title" in df.columns and df["_title"].nunique() > 1:
        op = (
            df.groupby("_title")
            .apply(lambda x: (x["sentiment"] == "Positive").mean())
            .sort_values(ascending=False)
        )
        insights.append(("#f0fdf4", "#16a34a",
            f"<b>Outlet Performance</b> — Best performing: '{op.index[0]}' ({op.iloc[0]*100:.1f}% positive). "
            f"Lowest performing: '{op.index[-1]}' ({op.iloc[-1]*100:.1f}% positive)."))

    for bg, border, text in insights:
        st.markdown(
            f'<div class="insight-card" style="border-left-color:{border};background:{bg};">{text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if "_rating" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            section("Rating vs BERT Sentiment")
            cross = pd.crosstab(df["_rating"].round(0).astype(int), df["sentiment"])
            fig = px.imshow(
                cross, color_continuous_scale="RdYlGn", aspect="auto",
                labels={"x": "BERT Sentiment", "y": "User Rating", "color": "Count"},
            )
            fig.update_layout(**CHART_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Top-left cells (high user rating, negative BERT sentiment) indicate courtesy ratings "
                "that mask genuine dissatisfaction."
            )
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
            fig2.update_layout(**CHART_LAYOUT, height=320, title_text="Sentiment Mix per Rating Band")
            st.plotly_chart(fig2, use_container_width=True)

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
                .format({
                    "Positive %": "{:.1f}",
                    "Negative %": "{:.1f}",
                    "Avg Confidence": "{:.2f}",
                }),
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
        "Waiting Time":                "Review lane management and staffing levels; consider additional attendants during peak hours.",
        "Payment Issue":               "Audit all POS terminals and ensure every digital payment channel is fully operational.",
        "Billing or Trust Issue":      "Install display meters, run regular oversight audits, and publicise the anti-fraud policy.",
        "CNG Availability":            "Monitor the CNG supply chain proactively and keep backup connections active.",
        "Cleanliness":                 "Increase cleaning frequency and assign dedicated housekeeping staff to the site.",
        "Fuel Quality":                "Schedule regular independent quality checks and display fuel certification prominently.",
        "Facility Maintenance":        "Implement a preventive maintenance schedule and track equipment health centrally.",
        "Customer Amenities":          "Upgrade water and restroom facilities and add seating where space permits.",
        "Safety Concern":              "Conduct a full safety audit and reinforce signage and fire-safety equipment.",
        "Traffic or Queue Management": "Redesign lane flow and consider installing a queue management display system.",
        "Accessibility":               "Improve directional signage and ensure disabled access pathways are clear.",
    }
    return lookup.get(issue, "Investigate the root cause and implement a targeted operational improvement.")


# ---------------------------------------------------------------------------
# Tab 5 — Data Explorer
# ---------------------------------------------------------------------------

def tab_data(df, df_raw):
    section("Analysed Reviews")

    display_cols = [
        "_review_text", "_rating", "sentiment", "bert_star",
        "confidence", "issue_category", "issue_tags",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    col_rename = {
        "_review_text":   "Review Text",
        "_rating":        "User Rating",
        "sentiment":      "Sentiment",
        "bert_star":      "BERT Stars",
        "confidence":     "Confidence",
        "issue_category": "Issue Category",
        "issue_tags":     "Issue Tags",
    }

    st.dataframe(
        df[display_cols].rename(columns=col_rename),
        use_container_width=True,
        height=480,
    )

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


if __name__ == "__main__":
    main()
