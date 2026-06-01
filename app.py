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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* overall page padding */
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    /* metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-top: 4px solid #dee2e6;
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #6c757d; font-weight: 600; letter-spacing: 0.03em; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: #212529; }

    /* section dividers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #343a40;
        margin: 1.4rem 0 0.4rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e9ecef;
    }

    /* insight cards */
    .insight-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 13px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem;
        font-weight: 500;
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
    }

    /* sidebar */
    div[data-testid="stSidebarContent"] { padding-top: 1.2rem; }

    /* hide the plotly modebar by default */
    .modebar { opacity: 0.3; }
    .modebar:hover { opacity: 1; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral":  "#f39c12",
    "Negative": "#e74c3c",
}

CHART_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12),
    margin=dict(t=48, b=16, l=16, r=16),
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


# ── helpers ──────────────────────────────────────────────────────────────────

def section(label):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


def top_keywords(text, n=10):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return Counter(words).most_common(n)


@st.cache_data(show_spinner=False)
def load_default_pune():
    return pd.read_excel("Pune_Retail_outlet (1).xlsx")


def run_analysis(df):
    texts = df["_review_text"].fillna("").tolist()

    with st.spinner("Running BERT sentiment analysis — this takes a moment on first load…"):
        results = predict_sentiment_batch(texts, batch_size=32)

    df["sentiment"]    = [r["sentiment"]    for r in results]
    df["confidence"]   = [r["confidence"]   for r in results]
    df["bert_star"]    = [r["star_rating"]   for r in results]
    df["prob_positive"]= [r["prob_positive"] for r in results]
    df["prob_neutral"] = [r["prob_neutral"]  for r in results]
    df["prob_negative"]= [r["prob_negative"] for r in results]

    with st.spinner("Classifying issue categories…"):
        df["issue_category"] = df["_review_text"].apply(classify_issue)
        df["issue_tags"]     = df["_review_text"].apply(
            lambda x: ", ".join(classify_issues_multi(x, top_n=2))
        )
    return df


# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("📂 Data Source")
    mode = st.sidebar.radio(
        "Choose dataset",
        ["Pune Retail Outlet (default)", "Upload your own CSV / XLSX"],
    )

    df_raw = None
    if mode == "Pune Retail Outlet (default)":
        df_raw = load_default_pune()
        st.sidebar.success(f"Loaded {len(df_raw):,} reviews")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
        if uploaded:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.sidebar.success(f"Loaded {len(df_raw):,} rows")
        else:
            st.sidebar.info("Upload a file to get started")

    return df_raw


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Sentiment Intelligence Dashboard")
    st.caption("BERT-powered review analysis · Issue categorisation · Time-series trends · Actionable insights")

    df_raw = sidebar()
    if df_raw is None:
        st.info("👈 Select or upload a dataset from the sidebar to begin.")
        return

    df, _ = normalize_dataframe(df_raw.copy())

    if "_review_text" not in df.columns:
        st.error(
            "Couldn't find a review text column. "
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

    # ── filters ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")

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
        "📈  Overview",
        "🔖  Issue Analysis",
        "📅  Time-Series",
        "🔎  Deep Insights",
        "📋  Data Explorer",
    ])

    with tab1:  tab_overview(df)
    with tab2:  tab_issues(df)
    with tab3:  tab_timeseries(df)
    with tab4:  tab_insights(df)
    with tab5:  tab_data(df, df_raw)


# ── TAB 1 — Overview ─────────────────────────────────────────────────────────

def tab_overview(df):
    total = len(df)
    pos   = (df["sentiment"] == "Positive").sum()
    neg   = (df["sentiment"] == "Negative").sum()
    neu   = (df["sentiment"] == "Neutral").sum()

    dominant = max(["Positive", "Neutral", "Negative"], key=lambda s: (df["sentiment"] == s).sum())
    top_issue = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()
    top_issue_label = top_issue.index[0] if not top_issue.empty else "—"

    # summary sentence
    st.markdown(
        f"Analysed **{total:,}** reviews — sentiment is predominantly **{dominant}** "
        f"({pos:,} positive · {neu:,} neutral · {neg:,} negative)."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # row 1 — sentiment counts with coloured top borders
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Positive 😊", f"{pos:,}", f"{pos/total*100:.1f}% of total")
    c3.metric("Neutral 😐",  f"{neu:,}", f"{neu/total*100:.1f}% of total")
    c4.metric("Negative 😞", f"{neg:,}", f"{neg/total*100:.1f}% of total")

    # colour the metric card borders to match sentiment
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-top-color: #2ecc71; }
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-top-color: #f39c12; }
    div[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-top-color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

    # row 2 — rating / BERT averages
    if "_rating" in df.columns:
        avg_r    = df["_rating"].dropna().mean()
        bert_avg = df["bert_star"].mean()
        r1, r2, r3 = st.columns(3)
        r1.metric("Avg User Rating",    f"{avg_r:.2f} ⭐")
        r2.metric("Avg BERT Stars",     f"{bert_avg:.2f} ⭐", f"{bert_avg - avg_r:+.2f} vs user")
        r3.metric("Top Pain Point",     top_issue_label,
                  f"{top_issue.iloc[0]:,} negative reviews" if not top_issue.empty else "")

    st.markdown("")

    # charts row 1
    section("Sentiment Distribution")
    left, right = st.columns(2)

    with left:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(**CHART_DEFAULTS, showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "_rating" in df.columns:
            rs = df.groupby(["_rating", "sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(
                rs, x="_rating", y="count",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"_rating": "Star Rating", "count": "Reviews", "sentiment": "Sentiment"},
            )
            fig2.update_layout(**CHART_DEFAULTS, height=320, title_text="How Ratings Align with Sentiment")
            st.plotly_chart(fig2, use_container_width=True)

    # charts row 2 — top keywords per sentiment
    section("Most Common Words by Sentiment")
    kc1, kc2, kc3 = st.columns(3)
    for col, sentiment, color in zip(
        [kc1, kc2, kc3],
        ["Positive", "Neutral", "Negative"],
        ["#2ecc71", "#f39c12", "#e74c3c"],
    ):
        text_blob = " ".join(df[df["sentiment"] == sentiment]["_review_text"].dropna())
        kws = top_keywords(text_blob, n=10)
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Word", "Count"]).sort_values("Count")
            fig_k = px.bar(
                kw_df, x="Count", y="Word", orientation="h",
                color_discrete_sequence=[color],
                labels={"Count": "Frequency", "Word": ""},
            )
            fig_k.update_layout(
                **CHART_DEFAULTS,
                title_text=f"{sentiment} — Top Keywords",
                showlegend=False,
                height=300,
                margin=dict(t=44, b=10, l=8, r=8),
            )
            col.plotly_chart(fig_k, use_container_width=True)

    # model confidence
    section("Model Confidence")
    fig3 = px.histogram(
        df, x="confidence", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=35, barmode="overlay", opacity=0.72,
        labels={"confidence": "Confidence Score", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig3.update_layout(**CHART_DEFAULTS, height=280, title_text="Confidence Score Distribution by Sentiment")
    st.plotly_chart(fig3, use_container_width=True)


# ── TAB 2 — Issue Analysis ────────────────────────────────────────────────────

def tab_issues(df):
    section("Reviews by Issue Category")
    issue_sent = df.groupby(["issue_category", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(
        issue_sent, x="issue_category", y="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        barmode="stack",
        labels={"issue_category": "Issue Category", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig.update_xaxes(tickangle=35)
    fig.update_layout(**CHART_DEFAULTS, height=380, title_text="Review Volume by Issue & Sentiment")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Category Share")
        ic = df["issue_category"].value_counts().reset_index()
        ic.columns = ["Category", "Count"]
        fig2 = px.pie(
            ic, names="Category", values="Count",
            hole=0.38,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(**CHART_DEFAULTS, showlegend=False, height=360)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("Negativity Rate per Category")
        neg_rate = (
            df.groupby("issue_category")
            .apply(lambda x: (x["sentiment"] == "Negative").mean() * 100)
            .reset_index(name="pct")
            .sort_values("pct", ascending=True)
        )
        fig3 = px.bar(
            neg_rate, x="pct", y="issue_category", orientation="h",
            color="pct", color_continuous_scale="Reds",
            labels={"pct": "% Negative Reviews", "issue_category": ""},
        )
        fig3.update_layout(**CHART_DEFAULTS, height=360, showlegend=False,
                           title_text="% of Reviews that are Negative")
        st.plotly_chart(fig3, use_container_width=True)

    if "_title" in df.columns and df["_title"].nunique() > 1:
        section("Outlet × Issue Heatmap")
        pivot = df.groupby(["_title", "issue_category"]).size().unstack(fill_value=0)
        fig4 = px.imshow(
            pivot, aspect="auto", color_continuous_scale="YlOrRd",
            labels={"x": "Issue Category", "y": "Outlet", "color": "Reviews"},
        )
        fig4.update_layout(**CHART_DEFAULTS, height=420, margin=dict(t=48, b=16, l=160, r=16))
        st.plotly_chart(fig4, use_container_width=True)

    section("Browse Negative Reviews by Issue")
    selected = st.selectbox("Select an issue category", sorted(df["issue_category"].unique()))
    neg_samples = df[
        (df["issue_category"] == selected) & (df["sentiment"] == "Negative")
    ][["_review_text", "_rating", "confidence"]].head(10)

    if neg_samples.empty:
        st.info("No negative reviews for this category with the current filters applied.")
    else:
        st.dataframe(
            neg_samples.rename(columns={
                "_review_text": "Review Text",
                "_rating": "User Rating",
                "confidence": "BERT Confidence",
            }),
            use_container_width=True,
            height=320,
        )


# ── TAB 3 — Time-Series ───────────────────────────────────────────────────────

def tab_timeseries(df):
    if "_date" not in df.columns or df["_date"].notna().sum() < 5:
        st.info("No date column detected — time-series analysis is not available for this dataset.")
        return

    df_t = df[df["_date"].notna()].copy()
    df_t["month"]   = df_t["_date"].dt.to_period("M").dt.to_timestamp()
    df_t["week"]    = df_t["_date"].dt.to_period("W").dt.to_timestamp()
    df_t["quarter"] = df_t["_date"].dt.to_period("Q").dt.to_timestamp()

    granularity = st.radio("View by", ["Monthly", "Weekly", "Quarterly"], horizontal=True)
    period_col  = {"Monthly": "month", "Weekly": "week", "Quarterly": "quarter"}[granularity]

    section("Sentiment Volume Over Time")
    sent_trend = df_t.groupby([period_col, "sentiment"]).size().reset_index(name="count")
    fig = px.line(
        sent_trend, x=period_col, y="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        markers=True,
        labels={period_col: "Period", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig.update_layout(**CHART_DEFAULTS, height=340, title_text=f"Review Volume by Sentiment ({granularity})")
    st.plotly_chart(fig, use_container_width=True)

    section("Sentiment Share Over Time")
    pct_data = (
        sent_trend.pivot_table(index=period_col, columns="sentiment", values="count", fill_value=0)
        .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100)
        .reset_index()
        .melt(id_vars=period_col, var_name="Sentiment", value_name="Percent")
    )
    fig2 = px.area(
        pct_data, x=period_col, y="Percent",
        color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
        labels={period_col: "Period", "Percent": "Share (%)"},
    )
    fig2.update_layout(**CHART_DEFAULTS, height=300, title_text=f"Sentiment Share % ({granularity})")
    st.plotly_chart(fig2, use_container_width=True)

    if "_rating" in df_t.columns:
        section("Average Rating & Review Volume")
        rt = df_t.groupby(period_col)["_rating"].agg(["mean", "count"]).reset_index()
        rt.columns = [period_col, "avg_rating", "review_count"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rt[period_col], y=rt["avg_rating"],
            mode="lines+markers", name="Avg Rating",
            line=dict(color="#3498db", width=2.5),
        ))
        fig3.add_trace(go.Bar(
            x=rt[period_col], y=rt["review_count"],
            name="Review Count", opacity=0.25,
            marker_color="#95a5a6", yaxis="y2",
        ))
        fig3.update_layout(
            **CHART_DEFAULTS,
            height=320,
            title_text=f"Avg Rating & Volume ({granularity})",
            yaxis=dict(title="Avg Rating", range=[0.8, 5.2]),
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
    fig4.update_layout(**CHART_DEFAULTS, height=340, title_text=f"Top 6 Issue Categories ({granularity})")
    st.plotly_chart(fig4, use_container_width=True)

    # spike detection
    neg_by_period = (
        df_t[df_t["sentiment"] == "Negative"]
        .groupby(period_col)
        .size()
        .reset_index(name="neg_count")
    )
    if len(neg_by_period) >= 4:
        mean_n = neg_by_period["neg_count"].mean()
        std_n  = neg_by_period["neg_count"].std()
        neg_by_period["spike"] = neg_by_period["neg_count"] > mean_n + 1.5 * std_n
        spikes = neg_by_period[neg_by_period["spike"]]
        if not spikes.empty:
            st.warning(
                "⚠️ **Negative review spike detected** in: "
                + ", ".join(spikes[period_col].astype(str).tolist())
            )
            fig5 = px.bar(
                neg_by_period, x=period_col, y="neg_count",
                color="spike",
                color_discrete_map={True: "#e74c3c", False: "#adb5bd"},
                labels={period_col: "Period", "neg_count": "Negative Reviews", "spike": "Spike"},
            )
            fig5.update_layout(**CHART_DEFAULTS, height=280, title_text="Negative Review Spikes",
                               showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)


# ── TAB 4 — Deep Insights ─────────────────────────────────────────────────────

def tab_insights(df):
    total     = len(df)
    neg_count = (df["sentiment"] == "Negative").sum()
    neg_pct   = neg_count / total * 100

    section("Automated Insights")

    insights = []
    if neg_pct > 50:
        insights.append(("#ffeaea", "#e74c3c",
            f"🔴 <b>High negativity alert</b> — {neg_pct:.1f}% of reviews are negative. Immediate attention required."))
    elif neg_pct > 30:
        insights.append(("#fff4e5", "#f39c12",
            f"🟠 <b>Moderate negativity</b> — {neg_pct:.1f}% of reviews are negative."))
    else:
        insights.append(("#eafaf1", "#2ecc71",
            f"🟢 <b>Broadly positive</b> — only {neg_pct:.1f}% of reviews are negative."))

    neg_issues = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()
    if not neg_issues.empty:
        ti, tc = neg_issues.index[0], neg_issues.iloc[0]
        insights.append(("#fff4e5", "#f39c12",
            f"⚠️ <b>Top pain point</b> — '{ti}' has {tc} negative reviews "
            f"({tc/neg_count*100:.1f}% of all negatives)."))

    staff_neg = ((df["issue_category"] == "Staff Behaviour") & (df["sentiment"] == "Negative")).sum()
    if staff_neg > 0:
        insights.append(("#fdf2f8", "#8e44ad",
            f"👥 <b>Staff behaviour</b> — {staff_neg} reviews flag staff-related issues."))

    cng_neg = ((df["issue_category"] == "CNG Availability") & (df["sentiment"] == "Negative")).sum()
    if cng_neg > 0:
        insights.append(("#ffeaea", "#e74c3c",
            f"⛽ <b>CNG availability</b> — {cng_neg} negative reviews about CNG being unavailable."))

    billing_neg = ((df["issue_category"] == "Billing or Trust Issue") & (df["sentiment"] == "Negative")).sum()
    if billing_neg > 0:
        insights.append(("#ffeaea", "#e74c3c",
            f"💳 <b>Trust &amp; billing</b> — {billing_neg} reviews mention fraud or incorrect billing."))

    if "_rating" in df.columns:
        mismatch = ((df["_rating"] >= 4) & (df["sentiment"] == "Negative")).sum()
        if mismatch > 0:
            insights.append(("#eaf3fb", "#2980b9",
                f"🔍 <b>Rating–sentiment mismatch</b> — {mismatch} reviews have ≥4 stars "
                f"but BERT classifies them as negative. Customers may be leaving polite ratings "
                f"despite negative experiences."))

    if "_title" in df.columns and df["_title"].nunique() > 1:
        op = (
            df.groupby("_title")
            .apply(lambda x: (x["sentiment"] == "Positive").mean())
            .sort_values(ascending=False)
        )
        insights.append(("#eafaf1", "#27ae60",
            f"🏆 <b>Best outlet</b> — '{op.index[0]}' ({op.iloc[0]*100:.1f}% positive). "
            f"Lowest performing: '{op.index[-1]}' ({op.iloc[-1]*100:.1f}% positive)."))

    for bg, border, text in insights:
        st.markdown(
            f'<div class="insight-card" style="border-left:4px solid {border};background:{bg};">'
            f'{text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    if "_rating" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            section("Rating vs BERT Sentiment (Heatmap)")
            cross = pd.crosstab(df["_rating"].round(0).astype(int), df["sentiment"])
            fig = px.imshow(
                cross, color_continuous_scale="RdYlGn",
                labels={"x": "BERT Sentiment", "y": "User Rating", "color": "Count"},
                aspect="auto",
            )
            fig.update_layout(**CHART_DEFAULTS, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Cells in the top-left corner (high rating, negative sentiment) represent hidden dissatisfaction.")

        with col2:
            section("Sentiment by Star Rating")
            rs = df.groupby(["_rating", "sentiment"]).size().reset_index(name="count")
            rs["pct"] = rs.groupby("_rating")["count"].transform(lambda x: x / x.sum() * 100)
            fig2 = px.bar(
                rs, x="_rating", y="pct",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"_rating": "Star Rating", "pct": "Share (%)", "sentiment": "Sentiment"},
            )
            fig2.update_layout(**CHART_DEFAULTS, height=320, title_text="Sentiment Mix per Rating")
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
                .format({"Positive %": "{:.1f}", "Negative %": "{:.1f}", "Avg Confidence": "{:.2f}"}),
            use_container_width=True,
        )

    section("Recommended Actions")
    neg_rates = (
        df[df["sentiment"] == "Negative"]["issue_category"]
        .value_counts(normalize=True)
        .head(5)
    )
    priority_labels = ["🔴 Critical", "🟠 High", "🟡 Medium", "🟡 Medium", "🟢 Low"]
    rows = [
        {
            "Priority":       priority_labels[i],
            "Issue Area":     issue,
            "Negative Share": f"{rate*100:.1f}%",
            "Recommended Action": _action_for(issue),
        }
        for i, (issue, rate) in enumerate(neg_rates.items())
    ]
    if rows:
        st.table(pd.DataFrame(rows))


def _action_for(issue):
    lookup = {
        "Staff Behaviour":           "Conduct staff training; implement service feedback at point-of-sale.",
        "Waiting Time":              "Review lane management; add attendants during peak hours.",
        "Payment Issue":             "Audit POS terminals; ensure all digital payment channels are working.",
        "Billing or Trust Issue":    "Install display meters; run oversight audits; publicise anti-fraud policy.",
        "CNG Availability":          "Monitor CNG supply chain; keep backup connections active.",
        "Cleanliness":               "Increase cleaning frequency; assign dedicated housekeeping staff.",
        "Fuel Quality":              "Run regular quality checks; display fuel certifications visibly.",
        "Facility Maintenance":      "Set up a preventive maintenance schedule; track equipment health.",
        "Customer Amenities":        "Upgrade water and restroom facilities; add seating where space allows.",
        "Safety Concern":            "Conduct a safety audit; reinforce signage and fire-safety equipment.",
        "Traffic or Queue Management": "Redesign lane flow; consider a queue management display system.",
        "Accessibility":             "Improve signage and wayfinding; ensure disabled access is clear.",
    }
    return lookup.get(issue, "Investigate the root cause and implement a targeted operational improvement.")


# ── TAB 5 — Data Explorer ────────────────────────────────────────────────────

def tab_data(df, df_raw):
    section("Analysed Reviews")

    display_cols = ["_review_text", "_rating", "sentiment", "bert_star", "confidence", "issue_category", "issue_tags"]
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
        "⬇️  Download Results as CSV",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    section("Raw Data Preview (first 50 rows)")
    st.dataframe(df_raw.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
