import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sentiment_engine import (
    predict_sentiment_batch,
    classify_issue,
    classify_issues_multi,
    normalize_dataframe,
    ISSUE_CATEGORIES,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral": "#f39c12",
    "Negative": "#e74c3c",
}


@st.cache_data(show_spinner=False)
def load_default_pune():
    return pd.read_excel("Pune_Retail_outlet (1).xlsx")


def run_analysis(df):
    texts = df["_review_text"].fillna("").tolist()

    with st.spinner("Running BERT sentiment analysis… this may take a minute on first load"):
        results = predict_sentiment_batch(texts, batch_size=32)

    df["sentiment"] = [r["sentiment"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["bert_star"] = [r["star_rating"] for r in results]
    df["prob_positive"] = [r["prob_positive"] for r in results]
    df["prob_neutral"] = [r["prob_neutral"] for r in results]
    df["prob_negative"] = [r["prob_negative"] for r in results]

    with st.spinner("Classifying issue categories…"):
        df["issue_category"] = df["_review_text"].apply(classify_issue)
        df["issue_tags"] = df["_review_text"].apply(
            lambda x: ", ".join(classify_issues_multi(x, top_n=2))
        )

    return df


def sidebar():
    st.sidebar.title("📂 Data Source")
    mode = st.sidebar.radio(
        "Choose dataset",
        ["Pune Retail Outlet (default)", "Upload your own CSV / XLSX"],
    )

    df_raw = None
    if mode == "Pune Retail Outlet (default)":
        df_raw = load_default_pune()
        st.sidebar.success(f"Loaded {len(df_raw)} reviews")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            st.sidebar.success(f"Loaded {len(df_raw)} rows")
        else:
            st.sidebar.info("Upload a file to get started")

    return df_raw


def main():
    st.title("📊 Sentiment Intelligence Dashboard")
    st.caption("BERT-powered sentiment · Issue categorisation · Time-series insights")

    df_raw = sidebar()

    if df_raw is None:
        st.info("👈 Select or upload a dataset from the sidebar to begin.")
        return

    df, _ = normalize_dataframe(df_raw.copy())

    if "_review_text" not in df.columns:
        st.error(
            "Couldn't detect a review text column. "
            "Expected one of: review_text_final, text, review, comment, body"
        )
        st.dataframe(df.head())
        return

    df = df[df["_review_text"].notna() & (df["_review_text"].str.strip() != "")].reset_index(drop=True)

    # use session state to avoid re-running the model on every interaction
    cache_key = str(len(df)) + str(df["_review_text"].iloc[0])
    if st.session_state.get("analyzed_hash") != cache_key:
        df = run_analysis(df)
        st.session_state.analyzed_df = df
        st.session_state.analyzed_hash = cache_key
    else:
        df = st.session_state.analyzed_df

    # sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")

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
        min_d = df["_date"].min().date()
        max_d = df["_date"].max().date()
        date_range = st.sidebar.date_input("Date range", value=(min_d, max_d))
        if len(date_range) == 2:
            df = df[
                (df["_date"] >= pd.Timestamp(date_range[0]))
                & (df["_date"] <= pd.Timestamp(date_range[1]))
            ]

    df = df[df["sentiment"].isin(sel_sentiments) & df["issue_category"].isin(sel_issues)]

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🔖 Issue Analysis",
        "📅 Time-Series",
        "🔎 Deep Insights",
        "📋 Data Explorer",
    ])

    with tab1:
        show_overview(df)
    with tab2:
        show_issues(df)
    with tab3:
        show_timeseries(df)
    with tab4:
        show_insights(df)
    with tab5:
        show_data(df, df_raw)


def show_overview(df):
    total = len(df)
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    neu = (df["sentiment"] == "Neutral").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", total)
    c2.metric("Positive 😊", f"{pos} ({pos/total*100:.1f}%)")
    c3.metric("Neutral 😐", f"{neu} ({neu/total*100:.1f}%)")
    c4.metric("Negative 😞", f"{neg} ({neg/total*100:.1f}%)")

    if "_rating" in df.columns:
        avg_r = df["_rating"].dropna().mean()
        bert_avg = df["bert_star"].mean()
        c1b, c2b = st.columns(2)
        c1b.metric("Avg User Rating", f"{avg_r:.2f} ⭐")
        c2b.metric("Avg BERT-Predicted Stars", f"{bert_avg:.2f} ⭐")

    col1, col2 = st.columns(2)

    with col1:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts,
            names="Sentiment",
            values="Count",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment Distribution",
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "_rating" in df.columns:
            rating_sent = (
                df.groupby(["_rating", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            fig2 = px.bar(
                rating_sent,
                x="_rating",
                y="count",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                title="Rating vs BERT Sentiment",
                labels={"_rating": "Star Rating", "count": "Reviews"},
            )
            fig2.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        df,
        x="confidence",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=40,
        title="Model Confidence Distribution",
        labels={"confidence": "Confidence Score"},
        barmode="overlay",
        opacity=0.7,
    )
    fig3.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Word Clouds")
    wc1, wc2, wc3 = st.columns(3)
    for col, sentiment, cmap in zip(
        [wc1, wc2, wc3],
        ["Positive", "Neutral", "Negative"],
        ["Greens", "Oranges", "Reds"],
    ):
        texts = " ".join(df[df["sentiment"] == sentiment]["_review_text"].dropna().tolist())
        if texts.strip():
            wc = WordCloud(
                width=400, height=250,
                background_color="white",
                colormap=cmap,
                max_words=80,
            ).generate(texts)
            fig_wc, ax = plt.subplots(figsize=(4, 2.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(sentiment, fontsize=10)
            col.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)


def show_issues(df):
    st.subheader("Issue Category Breakdown")

    issue_sent = (
        df.groupby(["issue_category", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        issue_sent,
        x="issue_category",
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        barmode="stack",
        title="Reviews by Issue Category & Sentiment",
        labels={"issue_category": "Issue Category", "count": "Reviews"},
    )
    fig.update_xaxes(tickangle=30)
    fig.update_layout(margin=dict(t=40, b=80))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        issue_counts = df["issue_category"].value_counts().reset_index()
        issue_counts.columns = ["Category", "Count"]
        fig2 = px.pie(
            issue_counts,
            names="Category",
            values="Count",
            title="Issue Category Share",
            hole=0.35,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        neg_rate = (
            df.groupby("issue_category")
            .apply(lambda x: (x["sentiment"] == "Negative").mean() * 100)
            .reset_index(name="neg_rate")
            .sort_values("neg_rate", ascending=False)
        )
        fig3 = px.bar(
            neg_rate,
            x="issue_category",
            y="neg_rate",
            color="neg_rate",
            color_continuous_scale="Reds",
            title="Negative Review Rate per Category (%)",
            labels={"issue_category": "Category", "neg_rate": "% Negative"},
        )
        fig3.update_xaxes(tickangle=30)
        fig3.update_layout(margin=dict(t=40, b=80))
        st.plotly_chart(fig3, use_container_width=True)

    if "_title" in df.columns and df["_title"].nunique() > 1:
        st.subheader("Outlet × Issue Category Heatmap")
        pivot = (
            df.groupby(["_title", "issue_category"])
            .size()
            .unstack(fill_value=0)
        )
        fig4 = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Review Count by Outlet and Issue",
            labels={"x": "Issue Category", "y": "Outlet"},
        )
        fig4.update_layout(margin=dict(t=40, b=80, l=150))
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Negative Reviews by Issue")
    selected_issue = st.selectbox("Select issue category", sorted(df["issue_category"].unique()))
    neg_samples = df[
        (df["issue_category"] == selected_issue) & (df["sentiment"] == "Negative")
    ][["_review_text", "_rating", "sentiment", "confidence"]].head(10)

    if neg_samples.empty:
        st.info("No negative reviews for this category with current filters.")
    else:
        st.dataframe(
            neg_samples.rename(columns={
                "_review_text": "Review", "_rating": "Rating",
                "sentiment": "Sentiment", "confidence": "Confidence",
            }),
            use_container_width=True,
        )


def show_timeseries(df):
    if "_date" not in df.columns or df["_date"].notna().sum() < 5:
        st.info("No date column detected – time-series analysis not available.")
        return

    df_t = df[df["_date"].notna()].copy()
    df_t["month"] = df_t["_date"].dt.to_period("M").dt.to_timestamp()
    df_t["week"] = df_t["_date"].dt.to_period("W").dt.to_timestamp()
    df_t["quarter"] = df_t["_date"].dt.to_period("Q").dt.to_timestamp()

    st.subheader("Sentiment Trends Over Time")
    granularity = st.radio("Granularity", ["Monthly", "Weekly", "Quarterly"], horizontal=True)
    period_col = {"Monthly": "month", "Weekly": "week", "Quarterly": "quarter"}[granularity]

    sent_trend = (
        df_t.groupby([period_col, "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.line(
        sent_trend,
        x=period_col,
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        markers=True,
        title=f"Review Volume by Sentiment ({granularity})",
        labels={period_col: "Period", "count": "Reviews"},
    )
    st.plotly_chart(fig, use_container_width=True)

    sent_pct = (
        df_t.groupby([period_col, "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot_table(index=period_col, columns="sentiment", values="count", fill_value=0)
    )
    sent_pct = sent_pct.div(sent_pct.sum(axis=1), axis=0) * 100
    sent_pct = sent_pct.reset_index().melt(id_vars=period_col, var_name="Sentiment", value_name="Percent")
    fig2 = px.area(
        sent_pct,
        x=period_col,
        y="Percent",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title=f"Sentiment Share (%) – {granularity}",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "_rating" in df_t.columns:
        rating_trend = (
            df_t.groupby(period_col)["_rating"]
            .agg(["mean", "count"])
            .reset_index()
        )
        rating_trend.columns = [period_col, "avg_rating", "review_count"]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rating_trend[period_col], y=rating_trend["avg_rating"],
            mode="lines+markers", name="Avg Rating",
            line=dict(color="#3498db", width=2),
            yaxis="y1",
        ))
        fig3.add_trace(go.Bar(
            x=rating_trend[period_col], y=rating_trend["review_count"],
            name="Review Count", opacity=0.3, yaxis="y2",
            marker_color="#95a5a6",
        ))
        fig3.update_layout(
            title=f"Average Rating & Review Volume ({granularity})",
            yaxis=dict(title="Avg Rating", range=[1, 5]),
            yaxis2=dict(title="Review Count", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Issue Category Trends")
    top_issues = df_t["issue_category"].value_counts().nlargest(6).index.tolist()
    issue_trend = (
        df_t[df_t["issue_category"].isin(top_issues)]
        .groupby([period_col, "issue_category"])
        .size()
        .reset_index(name="count")
    )
    fig4 = px.line(
        issue_trend,
        x=period_col,
        y="count",
        color="issue_category",
        markers=True,
        title=f"Top Issue Categories Over Time ({granularity})",
        labels={period_col: "Period", "count": "Reviews"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    # flag periods with unusually high negative review counts
    neg_by_period = (
        df_t[df_t["sentiment"] == "Negative"]
        .groupby(period_col)
        .size()
        .reset_index(name="neg_count")
    )
    if len(neg_by_period) >= 4:
        mean_neg = neg_by_period["neg_count"].mean()
        std_neg = neg_by_period["neg_count"].std()
        neg_by_period["is_spike"] = neg_by_period["neg_count"] > mean_neg + 1.5 * std_neg
        spikes = neg_by_period[neg_by_period["is_spike"]]

        if not spikes.empty:
            st.warning(
                "⚠️ **Negative review spike detected** during: "
                + ", ".join(spikes[period_col].astype(str).tolist())
            )
            fig5 = px.bar(
                neg_by_period,
                x=period_col,
                y="neg_count",
                color="is_spike",
                color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
                title="Negative Review Volume (spikes highlighted)",
                labels={period_col: "Period", "neg_count": "Negative Reviews"},
            )
            st.plotly_chart(fig5, use_container_width=True)


def show_insights(df):
    st.subheader("Key Insights")

    total = len(df)
    neg_count = (df["sentiment"] == "Negative").sum()
    pos_count = (df["sentiment"] == "Positive").sum()
    neg_pct = neg_count / total * 100

    insights = []

    if neg_pct > 50:
        insights.append(f"🔴 **High negativity**: {neg_pct:.1f}% of reviews are negative – immediate attention needed.")
    elif neg_pct > 30:
        insights.append(f"🟠 **Moderate negativity**: {neg_pct:.1f}% of reviews are negative.")
    else:
        insights.append(f"🟢 **Overall positive**: Only {neg_pct:.1f}% of reviews are negative.")

    neg_issues = df[df["sentiment"] == "Negative"]["issue_category"].value_counts()
    if not neg_issues.empty:
        top_issue = neg_issues.index[0]
        top_count = neg_issues.iloc[0]
        insights.append(
            f"⚠️ **Top pain point**: '{top_issue}' – {top_count} negative reviews ({top_count/neg_count*100:.1f}% of negatives)."
        )

    staff_neg = ((df["issue_category"] == "Staff Behaviour") & (df["sentiment"] == "Negative")).sum()
    if staff_neg > 0:
        insights.append(f"👥 **Staff issues**: {staff_neg} negative reviews mention staff behaviour.")

    cng_neg = ((df["issue_category"] == "CNG Availability") & (df["sentiment"] == "Negative")).sum()
    if cng_neg > 0:
        insights.append(f"⛽ **CNG availability**: {cng_neg} negative reviews about CNG unavailability.")

    billing_neg = ((df["issue_category"] == "Billing or Trust Issue") & (df["sentiment"] == "Negative")).sum()
    if billing_neg > 0:
        insights.append(f"💳 **Trust issues**: {billing_neg} reviews flag billing fraud or scams.")

    if "_rating" in df.columns:
        mismatch = ((df["_rating"] >= 4) & (df["sentiment"] == "Negative")).sum()
        if mismatch > 0:
            insights.append(
                f"🔍 **Rating-sentiment mismatch**: {mismatch} reviews rated 4–5 stars but BERT detects negative sentiment."
            )

    if "_title" in df.columns and df["_title"].nunique() > 1:
        outlet_pos = (
            df.groupby("_title")
            .apply(lambda x: (x["sentiment"] == "Positive").mean())
            .sort_values(ascending=False)
        )
        best = outlet_pos.index[0]
        worst = outlet_pos.index[-1]
        insights.append(
            f"🏆 **Best outlet**: '{best}' ({outlet_pos.iloc[0]*100:.1f}% positive). "
            f"Needs work: '{worst}' ({outlet_pos.iloc[-1]*100:.1f}% positive)."
        )

    for insight in insights:
        st.markdown(f"- {insight}")

    if "_rating" in df.columns:
        st.subheader("BERT Sentiment vs User Star Ratings")
        cross = pd.crosstab(df["_rating"].round(0), df["sentiment"])
        fig = px.imshow(
            cross,
            color_continuous_scale="RdYlGn",
            title="User Rating vs BERT Sentiment",
            labels={"x": "BERT Sentiment", "y": "User Star Rating", "color": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "High star ratings with negative BERT sentiment = users giving courtesy ratings despite bad experiences."
        )

    if "_title" in df.columns and df["_title"].nunique() > 1:
        st.subheader("Outlet Scorecard")
        scorecard = (
            df.groupby("_title")
            .agg(
                total=("sentiment", "count"),
                positive=("sentiment", lambda x: (x == "Positive").sum()),
                negative=("sentiment", lambda x: (x == "Negative").sum()),
                avg_conf=("confidence", "mean"),
            )
            .assign(
                pos_rate=lambda x: x["positive"] / x["total"] * 100,
                neg_rate=lambda x: x["negative"] / x["total"] * 100,
            )
            .sort_values("neg_rate", ascending=False)
            .reset_index()
        )
        scorecard.columns = ["Outlet", "Total", "Positive", "Negative", "Avg Confidence", "Positive %", "Negative %"]
        st.dataframe(
            scorecard.style
                .background_gradient(subset=["Negative %"], cmap="Reds")
                .background_gradient(subset=["Positive %"], cmap="Greens")
                .format({"Positive %": "{:.1f}", "Negative %": "{:.1f}", "Avg Confidence": "{:.2f}"}),
            use_container_width=True,
        )

    st.subheader("Recommended Actions")
    neg_issue_rates = (
        df[df["sentiment"] == "Negative"]["issue_category"]
        .value_counts(normalize=True)
        .head(5)
    )
    priority_labels = ["🔴 Critical", "🟠 High", "🟡 Medium", "🟡 Medium", "🟢 Low"]
    actions = []
    for rank, (issue, rate) in enumerate(neg_issue_rates.items()):
        actions.append({
            "Priority": priority_labels[rank],
            "Issue Area": issue,
            "Negative Share": f"{rate*100:.1f}%",
            "Action": _action_for(issue),
        })
    if actions:
        st.table(pd.DataFrame(actions))


def _action_for(issue):
    actions = {
        "Staff Behaviour": "Conduct staff training; implement service feedback at point-of-sale.",
        "Waiting Time": "Review lane management; add attendants during peak hours.",
        "Payment Issue": "Audit POS terminals; ensure all digital payment channels work.",
        "Billing or Trust Issue": "Install display meters; run oversight audits; publicise anti-fraud policy.",
        "CNG Availability": "Monitor CNG supply chain; keep backup connections active.",
        "Cleanliness": "Increase cleaning frequency; assign dedicated housekeeping.",
        "Fuel Quality": "Run regular quality checks; display certifications visibly.",
        "Facility Maintenance": "Set up preventive maintenance schedule; track equipment health.",
        "Customer Amenities": "Upgrade water and restroom facilities; add seating where possible.",
        "Safety Concern": "Run safety audit; reinforce signage and fire-safety equipment.",
        "Traffic or Queue Management": "Redesign lane flow; use queue display system.",
        "Accessibility": "Improve signage; ensure disabled access and clear pathways.",
    }
    return actions.get(issue, "Investigate root cause and implement targeted operational improvement.")


def show_data(df, df_raw):
    st.subheader("Analysed Data")

    display_cols = ["_review_text", "_rating", "sentiment", "bert_star", "confidence", "issue_category", "issue_tags"]
    display_cols = [c for c in display_cols if c in df.columns]

    rename = {
        "_review_text": "Review",
        "_rating": "User Rating",
        "sentiment": "Sentiment",
        "bert_star": "BERT Stars",
        "confidence": "Confidence",
        "issue_category": "Issue Category",
        "issue_tags": "Issue Tags",
    }

    st.dataframe(
        df[display_cols].rename(columns=rename),
        use_container_width=True,
        height=500,
    )

    csv_bytes = df[display_cols].rename(columns=rename).to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Results CSV",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
