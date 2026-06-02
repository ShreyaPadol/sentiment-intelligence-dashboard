# Methodology & Assumptions

## Data Sources

| Dataset | File | Format | Rows | Notes |
|---|---|---|---|---|
| Pune Petrol Pumps | `Pune_Retail_outlet (1).xlsx` | Review-level | 568 | Individual review text; enables BERT sentiment |
| Mumbai Petrol Pumps | `mumbai_petrol_pumps.xlsx` | Station-level | 168 | Aggregate ratings only; no review text |
| Mumbai Boundary | `mumbai_boundary.geojson` | GeoJSON | 1 polygon | MultiPolygon (5 rings); rendered as a grey outline layer on the Geographic Map tab |

---

## Assumptions — Pune / Review-Level Data

### A1 · BERT Sentiment Model
- Model: `nlptown/bert-base-multilingual-uncased-sentiment` (HuggingFace).
- Trained on Amazon/Yelp/TripAdvisor reviews in 6 languages; effective for short Indian-English reviews.
- Outputs a 1–5 star prediction; mapped to: 1–2 → Negative, 3 → Neutral, 4–5 → Positive.

### A2 · Negative Override Rules
- Hard-negative phrases (`fraud`, `scam`, `no cng`, `overcharg`, etc.) force a Positive prediction to Negative regardless of model output.
- Rationale: the multilingual BERT model occasionally misclassifies complaint language in code-switched text; hard overrides correct the highest-impact errors.

### A3 · Issue Classification
- Keyword-based scoring across 12 categories (see `sentiment_engine.py::ISSUE_CATEGORIES`).
- Each review is assigned the category with the most keyword matches; ties broken by first alphabetical match.
- Multi-label tagging uses top-2 categories.
- Reviews with zero keyword matches → "Other".

### A4 · Review Text Normalisation
- URLs stripped; non-word characters (except `,`, `.`, `!`, `?`, `'`, `-`) replaced with spaces.
- Text truncated to 512 characters before tokenisation.
- Blank or null review texts are excluded before analysis.

---

## Assumptions — Mumbai / Station-Level Data

### B1 · No Individual Review Text Available
- The Mumbai gas stations dataset (`gas_stations_inside_boundary_clean.xlsx`) contains **one row per station** with aggregate Google ratings (`totalScore`) and total review counts (`reviewsCount`).
- The BERT sentiment model **is not applied** to this dataset; it requires per-review text input.
- **Implication:** Mumbai sentiment labels are statistical proxies derived from aggregate ratings, not from text analysis.

### B2 · Sentiment Thresholds (Rating-Derived)
| Rating Band | Sentiment Label | Rationale |
|---|---|---|
| ≤ 2.5 ★ | Negative | Consistently poor customer experience |
| 2.6 – 3.5 ★ | Neutral | Mixed or average performance |
| > 3.5 ★ | Positive | Generally satisfactory |

These thresholds align with the BERT-to-stars mapping used for Pune: stars 1–2 = Negative (≤ 2.5), star 3 = Neutral (2.5–3.5), stars 4–5 = Positive (> 3.5).

### B3 · Stations Excluded from Rating Analysis
- 5 rows have `NaN` totalScore and are labelled "Unknown"; they appear in the raw data view but are excluded from all sentiment charts and the scorecard.
- 2 rows have null latitude / longitude and are excluded from the geographic map only.

### B4 · Geographic Zone Assignment
Zones are assigned by WGS-84 coordinate thresholds. These are approximations of Mumbai's administrative / colloquial boundaries:

| Zone | Latitude | Longitude | Covers |
|---|---|---|---|
| South Mumbai | lat < 18.970 | — | Colaba, Fort, Churchgate, Worli, Byculla |
| Central Mumbai | 18.970 ≤ lat < 19.050 | — | Dadar, Sion, Kurla, Chembur |
| Western Suburbs | lat ≥ 19.050, lat < 19.150 | lng < 72.880 | Bandra, Juhu, Andheri, Versova |
| Eastern Suburbs | lat ≥ 19.050, lat < 19.150 | lng ≥ 72.880 | Ghatkopar, Vikhroli, Mulund, Bhandup |
| North Suburbs | lat ≥ 19.150 | — | Goregaon, Malad, Kandivali, Borivali |

These boundaries do not account for ward-level administrative boundaries. For higher precision, a spatial join against Mumbai ward GeoJSON would be needed.

### B5 · Fuel Type Classification
- `categoryName = "Compressed natural gas station"` → `CNG`
- All other non-null categories (including `"Petrol Pump"`) → `Petrol / Diesel`
- Rows with null `categoryName` → `Petrol / Diesel` (default assumption; most gas stations in Mumbai offer petrol/diesel)

### B6 · CNG Availability Inference
- A station categorised as `"Compressed natural gas station"` is assumed to offer **only** CNG; it does not necessarily offer petrol or diesel.
- Stations categorised as `"Petrol Pump"` may or may not offer CNG as an additional service; the dataset does not capture this.
- This is a known data gap — a richer dataset (e.g. OMC portals, live API) would be needed to determine CNG availability at all petrol stations.

### B7 · Low-Evidence Stations
- Stations with fewer than 50 reviews are flagged as "low evidence" in the Insights tab.
- Their sentiment labels are less statistically reliable; actions should be deferred until review count improves.

---

## Validation Against Pune Baseline

| Metric | Pune (review-level) | Mumbai (station-level) | Comparable? |
|---|---|---|---|
| Sentiment model | BERT (text) | Rating threshold | No — different method |
| Issue categories | Keyword-NLP | Not available (no text) | No |
| Time-series | Yes (publishedAtDate) | No (no timestamps per station) | No |
| Geographic map | No (lat/lng available but not exposed in UI) | Yes — full map view | — |
| Outlet/station comparison | Yes (multi-outlet scorecard) | Yes (station scorecard) | Comparable |

**Key limitation:** Pune and Mumbai results are **not directly comparable** because Pune uses BERT on individual review text while Mumbai uses aggregate rating thresholds. A fair comparison would require either: (a) scraping individual reviews for all 168 Mumbai stations, or (b) using totalScore-derived sentiment for Pune as well.

---

## Known Data Gaps

1. **No per-review data for Mumbai petrol pumps** — limits depth of NLP analysis.
2. **No timestamp column** in the Mumbai stations file — time-series trends cannot be computed.
3. **CNG availability at multi-fuel stations** is unknown.
4. **Brand / OMC affiliation** (HPCL, BPCL, IOC, Shell) could be extracted from station titles via regex but was not implemented to avoid over-engineering.
5. **Opening hours** column is present but not parsed; 24-hour vs limited-hours stations could be a useful filter.

---

*Document created: 2026-06-02. Update alongside any changes to `sentiment_engine.py` or `app.py`.*
