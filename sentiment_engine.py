import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# issue categories for petrol pump / retail outlet reviews
# keywords are intentionally simple - weighted scoring works well here
ISSUE_CATEGORIES = {
    "Cleanliness": [
        "dirty", "clean", "hygiene", "hygienic", "garbage", "trash", "toilet",
        "washroom", "restroom", "filthy", "messy", "waste", "stink", "smell",
        "untidy", "neat", "dust", "sweep",
    ],
    "Staff Behaviour": [
        "staff", "attendant", "employee", "rude", "polite", "behaviour", "behavior",
        "service", "worker", "helper", "friendly", "arrogant", "helpful", "lazy",
        "aahe", "attitude", "personnel", "operator",
    ],
    "Payment Issue": [
        "payment", "pay", "upi", "cash", "card", "transaction", "digital",
        "paytm", "gpay", "phonepay", "amount", "money", "rupee", "inr",
        "change", "balance", "charged", "overcharge", "extra charge",
    ],
    "Waiting Time": [
        "wait", "waiting", "queue", "long", "slow", "delay", "late",
        "hour", "minute", "rush", "crowd", "busy", "fast", "quick",
    ],
    "Fuel Quality": [
        "fuel", "petrol", "diesel", "quality", "adulterat", "pure", "impure",
        "quantity", "litre", "liter", "mileage", "octane",
    ],
    "Billing or Trust Issue": [
        "bill", "billing", "scam", "fraud", "cheat", "trust", "honest",
        "dishonest", "receipt", "fake", "wrong", "incorrect", "overcharg",
        "manipulat", "deceiv", "ripoff", "theft",
    ],
    "Safety Concern": [
        "safe", "safety", "fire", "accident", "risk", "danger", "hazard",
        "security", "unsafe", "leakage", "spill", "explosion",
    ],
    "Facility Maintenance": [
        "broken", "repair", "maintain", "maintenance", "damage", "out of order",
        "not working", "machine", "pump", "nozzle", "equipment", "fix",
    ],
    "Customer Amenities": [
        "water", "drinking", "food", "cafe", "seating", "rest", "toilet",
        "parking", "atm", "air", "tyre", "inflation", "facility", "amenity",
        "convenience", "shop",
    ],
    "Traffic or Queue Management": [
        "traffic", "lane", "entry", "exit", "congestion", "block",
        "path", "route", "road",
    ],
    "Accessibility": [
        "accessible", "accessibility", "wheelchair", "disable", "disabled",
        "location", "reach", "distance", "signage", "visibility",
        "difficult to find", "far",
    ],
    "CNG Availability": [
        "cng", "compressed natural gas", "not available", "unavailable",
        "no cng", "cng station", "cng pump", "connection",
    ],
}

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()


def _stars_to_label(stars):
    if stars <= 2:
        return "Negative"
    if stars == 3:
        return "Neutral"
    return "Positive"


def predict_sentiment_batch(texts, batch_size=32):
    _load_model()
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch = [_clean(t) for t in texts[i:i + batch_size]]
        enc = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = _model(**enc).logits

        probs = torch.softmax(logits, dim=-1).numpy()
        predicted_stars = np.argmax(probs, axis=1) + 1

        for stars, row in zip(predicted_stars, probs):
            all_results.append({
                "sentiment": _stars_to_label(int(stars)),
                "confidence": float(row.max()),
                "star_rating": int(stars),
                "prob_negative": float(row[0] + row[1]),
                "prob_neutral": float(row[2]),
                "prob_positive": float(row[3] + row[4]),
            })

    return all_results


def classify_issue(text):
    if not isinstance(text, str) or not text.strip():
        return "Other"

    lower = text.lower()
    scores = {}
    for category, keywords in ISSUE_CATEGORIES.items():
        count = sum(1 for kw in keywords if kw in lower)
        if count > 0:
            scores[category] = count

    if not scores:
        return "Other"
    return max(scores, key=scores.get)


def classify_issues_multi(text, top_n=2):
    if not isinstance(text, str) or not text.strip():
        return ["Other"]

    lower = text.lower()
    scores = {}
    for category, keywords in ISSUE_CATEGORIES.items():
        count = sum(1 for kw in keywords if kw in lower)
        if count > 0:
            scores[category] = count

    if not scores:
        return ["Other"]

    sorted_cats = sorted(scores, key=scores.get, reverse=True)
    return sorted_cats[:top_n]


def _clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s,.!?'-]", " ", text)
    return text.strip()[:512]


def normalize_dataframe(df):
    import pandas as pd

    # map column names to lowercase for matching
    col_lower = {c.lower().strip(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in col_lower:
                return col_lower[c]
        return None

    text_col = find(["review_text_final", "text", "review", "review_text", "comment", "body"])
    rating_col = find(["stars", "rating", "star", "score", "place_rating"])
    date_col = find(["publishedatdate", "published_at_date", "date_iso", "date", "review_date", "publishat", "collected_at"])
    title_col = find(["title", "place_name", "outlet", "name", "location"])

    rename_map = {}
    if text_col:
        rename_map[text_col] = "_review_text"
    if rating_col:
        rename_map[rating_col] = "_rating"
    if date_col:
        rename_map[date_col] = "_date"
    if title_col:
        rename_map[title_col] = "_title"

    df = df.rename(columns=rename_map)

    if "_date" in df.columns:
        df["_date"] = pd.to_datetime(df["_date"], errors="coerce", utc=True)
        df["_date"] = df["_date"].dt.tz_localize(None)

    original_col_map = {v: k for k, v in rename_map.items()}
    return df, original_col_map
