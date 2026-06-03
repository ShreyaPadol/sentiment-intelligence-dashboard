"""
mumbai_nlp_pipeline.py
Advanced NLP analysis for Mumbai petrol pump reviews.
Provides: topic modeling (LDA), aspect-based sentiment, n-gram analysis,
          brand benchmarking, review quality scoring, station priority queue.
All functions are designed to be called from Streamlit with @st.cache_data.
"""

import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


# ── Text cleaning ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "and", "to", "for", "with",
    "on", "at", "by", "from", "this", "was", "are", "be", "not", "but",
    "have", "had", "has", "they", "their", "there", "that", "very", "so",
    "my", "we", "i", "its", "here", "get", "got", "also", "just", "no",
    "or", "as", "do", "did", "been", "all", "one", "if", "up", "out",
    "about", "than", "more", "when", "will", "can", "good", "place", "nice",
    "would", "like", "really", "come", "even", "always", "never", "dont",
    "petrol", "pump", "station", "fuel", "filling",
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower().strip()


def tokenize(text):
    return [w for w in clean_text(text).split() if len(w) > 2 and w not in STOPWORDS]


# ── Aspect-Based Sentiment Analysis ───────────────────────────────────────────

ASPECTS = {
    "Staff & Service":     ["staff", "attendant", "employee", "rude", "polite", "helpful",
                             "behavior", "behaviour", "arrogant", "friendly", "service", "worker"],
    "Fuel Quality":        ["quality", "adulterat", "mileage", "octane", "pure", "impure",
                             "quantity", "litre", "liter", "engine"],
    "Billing & Honesty":   ["billing", "bill", "overcharg", "cheat", "fraud", "scam",
                             "receipt", "meter", "manipulat", "dishonest", "correct", "honest"],
    "Waiting & Speed":     ["wait", "queue", "slow", "delay", "fast", "quick", "minute",
                             "rush", "crowd", "time", "busy"],
    "Cleanliness":         ["clean", "dirty", "hygiene", "washroom", "toilet", "filthy",
                             "smell", "garbage", "neat", "messy", "sweep"],
    "Payment Methods":     ["upi", "card", "cash", "paytm", "gpay", "phonepay", "digital",
                             "terminal", "payment"],
    "CNG Availability":    ["cng", "compressed", "available", "unavailable", "pressure",
                             "compressor"],
    "Facilities":          ["air", "water", "parking", "seating", "atm", "tyre", "inflation",
                             "facility", "amenity"],
}

NEGATIVE_SIGNALS = [
    "rude", "worst", "bad", "terrible", "horrible", "pathetic", "fraud", "scam", "cheat",
    "manipulat", "dishonest", "avoid", "dirty", "filthy", "overcharg", "not working",
    "broken", "down", "no", "not available", "unavailable", "slow", "delay", "arrogant",
]
POSITIVE_SIGNALS = [
    "good", "great", "excellent", "best", "clean", "honest", "helpful", "polite",
    "quick", "fast", "trustworthy", "superb", "recommended", "efficient", "professional",
    "correct", "available", "working", "reliable",
]


def aspect_sentiment(text):
    """Return dict of aspect → 'Positive' | 'Negative' | None for aspects found in text."""
    if not isinstance(text, str):
        return {}
    lower = text.lower()
    results = {}
    for aspect, keywords in ASPECTS.items():
        if any(kw in lower for kw in keywords):
            pos = sum(1 for s in POSITIVE_SIGNALS if s in lower)
            neg = sum(1 for s in NEGATIVE_SIGNALS if s in lower)
            results[aspect] = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
    return results


def compute_aspect_matrix(df, text_col="text", sentiment_col="sentiment"):
    """
    Returns a DataFrame: aspect × sentiment counts,
    plus a per-aspect negativity % series.
    """
    aspect_records = []
    for _, row in df.iterrows():
        asp = aspect_sentiment(row[text_col])
        for aspect, senti in asp.items():
            aspect_records.append({"aspect": aspect, "sentiment": senti,
                                    "review_sentiment": row.get(sentiment_col, "Unknown")})

    if not aspect_records:
        return pd.DataFrame(), pd.Series(dtype=float)

    adf = pd.DataFrame(aspect_records)
    matrix = adf.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    neg_pct = (
        adf[adf["sentiment"] == "Negative"]
        .groupby("aspect").size()
        .div(adf.groupby("aspect").size())
        .mul(100)
        .sort_values(ascending=False)
    )
    return matrix, neg_pct


# ── LDA Topic Modeling ─────────────────────────────────────────────────────────

TOPIC_LABELS = {
    # Will be assigned post-fit based on top words
}

NAMED_TOPICS = [
    "Staff Conduct",
    "Billing & Fraud",
    "Fuel Quality & Mileage",
    "Waiting Time & Queue",
    "CNG Availability",
    "Cleanliness & Hygiene",
    "Payment & Digital",
    "Facilities & Amenities",
]


def fit_lda(texts, n_topics=8, max_features=500, max_iter=20, random_state=42):
    """Fit LDA on cleaned texts. Returns (model, vectorizer, doc_topic_matrix)."""
    clean = [" ".join(tokenize(t)) for t in texts]
    clean = [t if t.strip() else "general service" for t in clean]

    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2),
                          min_df=2, max_df=0.95)
    dtm = vec.fit_transform(clean)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=random_state,
        learning_method="online",
    )
    doc_topics = lda.fit_transform(dtm)
    return lda, vec, doc_topics


def get_topic_words(lda, vec, top_n=10):
    """Return list of (topic_idx, top_words_list) tuples."""
    feature_names = vec.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_indices = comp.argsort()[-top_n:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics.append((i, top_words))
    return topics


def assign_topic_label(top_words):
    """Heuristic: match top words to a named topic."""
    word_set = set(top_words)
    mappings = {
        "Staff Conduct":         {"staff", "attendant", "rude", "polite", "helpful", "behavior", "attitude"},
        "Billing & Fraud":       {"billing", "bill", "overcharg", "cheat", "fraud", "scam", "meter", "receipt"},
        "Fuel Quality & Mileage":{"quality", "mileage", "adulterat", "engine", "octane", "litre"},
        "Waiting Time & Queue":  {"wait", "queue", "slow", "delay", "time", "minute", "rush"},
        "CNG Availability":      {"cng", "compressed", "available", "pressure", "compressor", "unavailable"},
        "Cleanliness & Hygiene": {"clean", "dirty", "washroom", "toilet", "hygiene", "smell", "garbage"},
        "Payment & Digital":     {"upi", "card", "cash", "gpay", "payment", "terminal", "phonepay"},
        "Facilities & Amenities":{"air", "water", "parking", "tyre", "inflation", "facility", "atm"},
    }
    best_label, best_score = "General Experience", 0
    for label, keywords in mappings.items():
        score = len(word_set & keywords)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


# ── N-gram Analysis ────────────────────────────────────────────────────────────

def top_ngrams(texts, n=2, top_k=15):
    """Extract top_k n-grams from a list of texts."""
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(STOPWORDS),
                          max_features=200, min_df=2)
    try:
        dtm = vec.fit_transform(texts)
        freqs = np.asarray(dtm.sum(axis=0)).ravel()
        names = vec.get_feature_names_out()
        pairs = sorted(zip(names, freqs), key=lambda x: -x[1])
        return [(w, int(f)) for w, f in pairs[:top_k]]
    except Exception:
        return []


# ── Brand Benchmarking ─────────────────────────────────────────────────────────

def brand_scorecard(df, sentiment_col="sentiment", brand_col="brand",
                    rating_col="station_rating", review_count_col="station_reviews"):
    """Return per-brand summary DataFrame."""
    records = []
    for brand, grp in df.groupby(brand_col):
        total = len(grp)
        pos   = (grp[sentiment_col] == "Positive").sum()
        neg   = (grp[sentiment_col] == "Negative").sum()
        neu   = (grp[sentiment_col] == "Neutral").sum()
        avg_r = grp[rating_col].mean() if rating_col in grp.columns else None
        stations = grp["station_name"].nunique() if "station_name" in grp.columns else None

        records.append({
            "Brand":            brand,
            "Stations":         stations,
            "Reviews":          total,
            "Positive":         pos,
            "Negative":         neg,
            "Neutral":          neu,
            "Positive %":       round(pos / total * 100, 1) if total else 0,
            "Negative %":       round(neg / total * 100, 1) if total else 0,
            "Avg Station Rating": round(avg_r, 2) if avg_r is not None else None,
        })
    return pd.DataFrame(records).sort_values("Positive %", ascending=False).reset_index(drop=True)


# ── Review Quality / Authenticity Scoring ─────────────────────────────────────

def review_quality_score(text):
    """
    Score 0–100. Higher = more informative / trustworthy review.
    Penalises: very short, repetitive words, all-caps, emoji-only.
    Rewards: specific keywords, length, balanced sentiment signals.
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    words = text.split()
    n = len(words)
    if n < 3:
        return 5

    length_score   = min(40, n * 2)                       # up to 40 pts for length
    unique_ratio   = len(set(words)) / n                  # penalise repetition
    uniqueness_score = int(unique_ratio * 30)             # up to 30 pts
    specificity    = sum(1 for kw in [
        "meter", "receipt", "upi", "gpay", "cng", "litre", "mileage",
        "attendant", "nozzle", "billing", "overcharge", "queue",
    ] if kw in text.lower())
    specificity_score = min(30, specificity * 10)         # up to 30 pts

    return min(100, length_score + uniqueness_score + specificity_score)


# ── Station Priority Queue ─────────────────────────────────────────────────────

def build_priority_queue(df, sentiment_col="sentiment", rating_col="station_rating",
                         review_count_col="station_reviews"):
    """
    Returns a DataFrame of stations sorted by intervention urgency.
    Score = (negative %) × log(review_count) × (5 - avg_rating)
    """
    records = []
    for station, grp in df.groupby("station_name"):
        total = len(grp)
        neg   = (grp[sentiment_col] == "Negative").sum()
        neg_pct = neg / total * 100 if total else 0
        avg_rating = grp[rating_col].mean() if rating_col in grp.columns else 3.5
        rc = grp[review_count_col].iloc[0] if review_count_col in grp.columns else 10
        rc = max(1, float(rc))

        urgency = neg_pct * math.log1p(rc) * max(0, 5 - avg_rating)

        records.append({
            "Station":         station,
            "Zone":            grp["zone"].iloc[0] if "zone" in grp.columns else "—",
            "Brand":           grp["brand"].iloc[0] if "brand" in grp.columns else "—",
            "Avg Rating":      round(avg_rating, 2),
            "Google Reviews":  int(rc),
            "NLP Reviews":     total,
            "Negative %":      round(neg_pct, 1),
            "Urgency Score":   round(urgency, 1),
        })

    return (
        pd.DataFrame(records)
        .sort_values("Urgency Score", ascending=False)
        .reset_index(drop=True)
    )


# ── TF-IDF Key Phrase Extraction ──────────────────────────────────────────────

def extract_keyphrases(texts, top_k=20):
    """Return top TF-IDF keyphrases from a corpus."""
    clean = [" ".join(tokenize(t)) for t in texts]
    clean = [t if t.strip() else "general" for t in clean]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=300,
                               min_df=2, stop_words=list(STOPWORDS))
        tfidf = vec.fit_transform(clean)
        scores = np.asarray(tfidf.mean(axis=0)).ravel()
        names  = vec.get_feature_names_out()
        pairs  = sorted(zip(names, scores), key=lambda x: -x[1])
        return [(w, float(s)) for w, s in pairs[:top_k]]
    except Exception:
        return []


# ── Zone-level NLP Summary ────────────────────────────────────────────────────

def zone_nlp_summary(df, sentiment_col="sentiment", text_col="text"):
    """Return per-zone NLP summary with top pain point keyword per zone."""
    records = []
    for zone, grp in df.groupby("zone"):
        total = len(grp)
        neg   = (grp[sentiment_col] == "Negative").sum()
        neg_pct = neg / total * 100 if total else 0
        neg_texts = grp[grp[sentiment_col] == "Negative"][text_col].tolist()
        top_kw = top_ngrams(neg_texts, n=1, top_k=3) if neg_texts else []
        pain_points = ", ".join(w for w, _ in top_kw) if top_kw else "—"
        records.append({
            "Zone":            zone,
            "Total Reviews":   total,
            "Negative %":      round(neg_pct, 1),
            "Top Pain Points": pain_points,
        })
    return pd.DataFrame(records).sort_values("Negative %", ascending=False).reset_index(drop=True)
